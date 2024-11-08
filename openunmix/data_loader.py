import os
import time
from typing import List

import numpy
import torch
import torchaudio
from scipy.fftpack import dct, idct
from torch.utils.data import Dataset, DataLoader


def load_audio(path: str):
    # we ignore the case where start!=0 and dur=None
    # since we have to deal with fixed length audio
    sig, rate = torchaudio.load(path)
    return sig, rate


def preprocess_dct(tensor: torch.Tensor, dct_scaler: float):
    tensor: numpy.ndarray = tensor.detach().numpy()
    dct_coeff: numpy.ndarray = dct(tensor, norm='ortho')
    inv = 1.0 / dct_scaler
    c = dct_coeff * dct_scaler
    d = c.astype("int").astype("float") * inv
    e = idct(d, norm="ortho")
    return e


class OpenUnmixMusicSeparatorDataset(Dataset):
    def __init__(self, root_dir, files_to_load: List[str],
                 preprocess_dct: bool,
                 dct_scaler: float):
        """

        :param root_dir:  The root directory of the dataset, subdirectories should contain data files
        :param files_to_load: The order in which we load the files, e.g for the dnr, we can specify
        ["mix","speech","music","sfx"] which means that the _getitem torch output will have files
        ordered in mix, speech, music and sfx respectively.
        :param n_fft:  FFT size
        """
        self.root_dir = root_dir
        self.preprocess_dct = preprocess_dct
        self.dct_scaler = dct_scaler
        self.audio_files = self._get_audio_files(files_to_load)

    def _get_audio_files(self, files_to_load: List[str]):
        audio_files = []
        for folder_name in os.listdir(self.root_dir):
            folder_path = os.path.join(self.root_dir, folder_name)
            if os.path.isdir(folder_path):
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
            if self.preprocess_dct:
                tensor = preprocess_dct(tensor, self.dct_scaler)
                loaded_files.append(torch.from_numpy(tensor))
            else:
                loaded_files.append(tensor)

        return loaded_files


if __name__ == '__main__':
    dataset = OpenUnmixMusicSeparatorDataset(root_dir="/Users/etemesi/PycharmProjects/Spite/data/dnr_v2",
                                             files_to_load=["mix", "speech"], preprocess_dct=True, dct_scaler=2000)

    start = time.time()
    de = DataLoader(dataset=dataset, batch_size=3, shuffle=False, num_workers=os.cpu_count())

    for a in de:
        print(a[0].sum())
        break
    stop = time.time()
    print(stop - start)
