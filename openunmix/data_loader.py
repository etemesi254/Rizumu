import os
import time
from typing import List, Optional

import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader

def load_info(path: str) -> dict:
    """Load audio metadata

    this is a backend_independent wrapper around torchaudio.info

    Args:
        path: Path of filename
    Returns:
        Dict: Metadata with
        `samplerate`, `samples` and `duration` in seconds

    """
    # get length of file in samples
    if torchaudio.get_audio_backend() == "sox":
        raise RuntimeError("Deprecated backend is not supported")

    info = {}
    si = torchaudio.info(str(path))
    info["samplerate"] = si.sample_rate
    info["samples"] = si.num_frames
    info["channels"] = si.num_channels
    info["duration"] = info["samples"] / info["samplerate"]
    return info


def load_audio(
    path: str,
    start: float = 0.0,
    dur: Optional[float] = None,
    info: Optional[dict] = None,
):

    # loads the full track duration
    if dur is None:
        # we ignore the case where start!=0 and dur=None
        # since we have to deal with fixed length audio
        sig, rate = torchaudio.load(path)
        return sig, rate
    else:
        if info is None:
            info = load_info(path)
        num_frames = int(dur * info["samplerate"])
        frame_offset = int(start * info["samplerate"])
        sig, rate = torchaudio.load(path, num_frames=num_frames, frame_offset=frame_offset)
        return sig, rate




class OpenUnmixMusicSeparatorDataset(Dataset):
    def __init__(self, root_dir, files_to_load: List[str]):
        """

        :param root_dir:  The root directory of the dataset, subdirectories should contain data files
        :param files_to_load: The order in which we load the files, e.g for the dnr, we can specify
        ["mix","speech","music","sfx"] which means that the _getitem torch output will have files
        ordered in mix, speech, music and sfx respectively.
        :param n_fft:  FFT size
        """
        self.root_dir = root_dir
        self.audio_files = self._get_audio_files(files_to_load)

    def _get_audio_files(self, files_to_load: List[str]):
        audio_files = []
        for folder_name in os.listdir(self.root_dir):
            folder_path = os.path.join(self.root_dir, folder_name)
            if os.path.isdir(folder_path):
                correct_path = [""]
                files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.wav')]
                correct_path = [""] * len(files_to_load)
                # sort based on files_to_load specification
                for file in files:
                    # split name and remove wav part
                    file_name = file.split("/").pop().split(".")[0]
                    # now see name index in files_to_load index
                    index = 0
                    for i in files_to_load:
                        if file_name == i:
                            correct_path[index] = file
                            break
                        index += 1
                audio_files.append(correct_path)
        return audio_files

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx) -> List[torch.Tensor]:
        loaded_files = []

        for file in self.audio_files[idx]:
            tensor, sr = load_audio(file)
            loaded_files.append(tensor)

        return loaded_files


if __name__ == '__main__':
    dataset = OpenUnmixMusicSeparatorDataset(root_dir="/Users/etemesi/PycharmProjects/Spite/data/dnr_v2/cv",
                                             files_to_load=["mix", "speech"])

    start = time.time()
    de = DataLoader(dataset=dataset, batch_size=3, shuffle=True, num_workers=os.cpu_count())

    for a in de:
        print(a[0].shape)
        break
    stop = time.time()
    print(stop - start)
