import os
import time
from pathlib import Path
from typing import List

import numpy as np
from numpy.lib.stride_tricks import as_strided

import torch
import torchaudio
from scipy.fftpack import dct, idct
from torch.utils.data import Dataset, DataLoader

def split_frame(
    x: np.ndarray,
    *,
    frame_length: int,
    hop_length: int,
    axis: int = -1,
    writeable: bool = False,
    subok: bool = False,
) -> np.ndarray:
    # This implementation is derived from numpy.lib.stride_tricks.sliding_window_view (1.20.0)
    # https://numpy.org/doc/stable/reference/generated/numpy.lib.stride_tricks.sliding_window_view.html

    x = np.array(x, copy=False, subok=subok)

    if x.shape[axis] < frame_length:
        raise Exception(
            f"Input is too short (n={x.shape[axis]:d}) for frame_length={frame_length:d}"
        )

    if hop_length < 1:
        raise Exception(f"Invalid hop_length: {hop_length:d}")

    # put our new within-frame axis at the end for now
    out_strides = x.strides + tuple([x.strides[axis]])

    # Reduce the shape on the framing axis
    x_shape_trimmed = list(x.shape)
    x_shape_trimmed[axis] -= frame_length - 1

    out_shape = tuple(x_shape_trimmed) + tuple([frame_length])
    xw = as_strided(
        x, strides=out_strides, shape=out_shape, subok=subok, writeable=writeable
    )

    if axis < 0:
        target_axis = axis - 1
    else:
        target_axis = axis + 1

    xw = np.moveaxis(xw, -1, target_axis)

    # Downsample along the target axis
    slices = [slice(None)] * xw.ndim
    slices[axis] = slice(0, None, hop_length)
    return xw[tuple(slices)]


def load_audio(path: str):
    # we ignore the case where start!=0 and dur=None
    # since we have to deal with fixed length audio
    sig, rate = torchaudio.load(path)
    return sig, rate


def exec_dct(t:np.ndarray,dct_scaler:float):
    dct_coeff: np.ndarray = dct(t, norm='ortho')
    inv = 1.0 / dct_scaler
    c = dct_coeff * dct_scaler
    d = c.astype("int").astype("float") * inv
    e = idct(d, norm="ortho")
    return e

def preprocess_dct(tensor: torch.Tensor, dct_scaler: float):
    t: np.ndarray = tensor.detach().numpy()
    initial_shape = t.shape
    frame_length = 1024
    hop_length = 1024

    frames = split_frame(t, frame_length=frame_length, hop_length=hop_length)
    dct_coefficients = np.array([exec_dct(frame,dct_scaler) for frame in frames.T])
    output = np.resize(dct_coefficients, initial_shape)
    c = 0
    return output
    #return exec_dct(t, dct_scaler)


class RizumuSeparatorDataset(Dataset):
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
        try:
            for folder_name in os.scandir(self.root_dir):

                folder_path = os.path.join(self.root_dir, folder_name.path)
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
        except Exception as e:
            pass
        return audio_files

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx) -> List[torch.Tensor]:
        loaded_files = []

        for file in self.audio_files[idx]:
            tensor, sr = load_audio(file)
            if self.preprocess_dct:
                tensor = preprocess_dct(tensor, self.dct_scaler)
                loaded_files.append(torch.from_numpy(tensor).clone().to(torch.float32))
            else:
                loaded_files.append(tensor)

        return loaded_files


if __name__ == '__main__':
    dataset = RizumuSeparatorDataset(root_dir="/Users/etemesi/PycharmProjects/Spite/data/dnr_v2",
                                             files_to_load=["mix", "speech"], preprocess_dct=False, dct_scaler=2000)

    start = time.time()
    de = DataLoader(dataset=dataset, batch_size=3, shuffle=False, num_workers=os.cpu_count())

    for a in de:
        print(a[0].shape)
        break
    stop = time.time()
    print(stop - start)
