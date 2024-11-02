import os
from typing import Tuple, List

import librosa
import torch
from torch.utils.data import Dataset


def load_and_pad(file: str, n_fft=2048):
    data, sr = librosa.load(file, sr=None,
                            mono=False)
    data_tc = torch.from_numpy(data)

    stft = data_tc.stft(n_fft=n_fft, window=None, return_complex=True)
    new_shape = stft.reshape((1, 1, stft.shape[0], stft.shape[1]))
    return new_shape, sr


class MusicSeparatorDataset(Dataset):
    def __init__(self, root_dir, files_to_load: List[str], n_fft=2048):
        """

        :param root_dir:  The root directory of the dataset, subdirectories should contain data files
        :param files_to_load: The order in which we load the files, e.g for the dnr, we can specify
        ["mix","speech","music","sfx"] which means that the _getitem torch output will have files
        ordered in mix, speech, music and sfx respectively.
        :param n_fft:  FFT size
        """
        self.root_dir = root_dir
        self.n_fft = n_fft
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
            tensor, sr = load_and_pad(file,self.n_fft)
            loaded_files.append(tensor)

        return loaded_files


if __name__ == '__main__':
    MusicSeparatorDataset(root_dir="/Users/etemesi/PycharmProjects/Spite/data/dnr_v2/cv",
                          files_to_load=["mix", "speech", "music", "sfx"])
