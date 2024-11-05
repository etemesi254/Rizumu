from typing import List, Tuple

import numpy as np
import pytorch_lightning as pl
import torch
from torch import nn


def calculate_istft(tensor: torch.Tensor, n_fft: int = 2048) -> torch.Tensor:
    return torch.istft(tensor.squeeze().to("cpu"), window=torch.hann_window(window_length=n_fft),
                       n_fft=n_fft)


def calculate_loss(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    :param a:
    :param b:
    :return:
    """
    loss = torch.nn.functional.mse_loss(a, b)
    return loss


def calculate_sdr(target_tensor, output_tensor) -> float:
    """
     Calculate the signal to distortion ratio between target and output tensor
    :param target_tensor: The true expected output
    :param output_tensor: The predicted output
    :param n_fft: window length for istft
    :return:  The signal to distortion ratio between target and output tensor
    """

    target_tensor = target_tensor.detach().cpu().numpy()
    output_tensor = output_tensor.detach().cpu().numpy()

    target_power = np.sum(target_tensor ** 2)
    noise_power = np.sum((target_tensor - output_tensor) ** 2)

    if noise_power == 0:
        return float('inf')  # Handle the case where the noise power is zero to prevent division by zero

    sdr = 10 * np.log10(target_power / noise_power)
    return sdr


class BandPlModel(pl.LightningModule):
    def __init__(self,
                 model: nn.Module,
                 optimizer: torch.optim.Optimizer,
                 labels: List[str],
                 output_label_name: str,
                 mix_name: str,
                 n_fft: int = 2048):
        """
        BandSpitRnn model wrapper in pytorch lightning module
        :param model: The initialized bandspitrnn
        :param optimizer: The optimizer to use for optimizing the model
        :param labels: This a list of label names for the training set.
            E.g. for the DnR dataset, the labels would be ["mix", "speech", "music", "sfx"]
        :param output_label_name: The output label name for which we are training our model in
            E.g. if we want to make a speech model using the DnR dataset, the output_label_name would be "speech"
        :param mix_name: The name of the wav file containing the mixture sample.
            E.g. for the DnR dataset, the mix_name would be "mix"
        :param n_fft: N-fft window length for stft and istft calculation
        """
        super().__init__()
        self.model = model
        self.labels = labels
        self.output_label_name = output_label_name
        self.mix_name = mix_name

        self.optimizer = optimizer
        self.n_fft = n_fft

    def get_batch(self, batch: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:

        assert len(batch) == len(self.labels), "Length of batch and labels do not match"

        labels_and_batch = zip(self.labels, batch)
        # find mix label
        mix_input = None
        expected_output = None
        for (label, stft) in labels_and_batch:
            if label == self.mix_name:
                mix_input = stft
            elif label == self.output_label_name:
                expected_output = stft
        assert mix_input is not None, "Mix label cannot be None, did you name them correctly"
        assert expected_output is not None, "Output label cannot be None, did you name them correctly"
        return mix_input, expected_output

    def calculate_properties(self, output: torch.Tensor, expected_output: torch.Tensor) -> torch.Tensor:
        # perform istft
        output_istft = calculate_istft(output, self.n_fft)
        speech_istft = calculate_istft(expected_output, self.n_fft)

        loss = calculate_loss(output_istft, speech_istft)
        sdr = calculate_sdr(output_istft, speech_istft)

        self.log("train_loss", loss,prog_bar=True)
        self.log("train_sdr", sdr,prog_bar=True)
        return loss

    def training_step(self, batch: List[torch.Tensor], batch_idx):
        # from our batch  place labels with
        mix_input, expected_output = self.get_batch(batch)

        output = self.model(mix_input)

        return self.calculate_properties(output, expected_output)

    def configure_optimizers(self):
        return self.optimizer

    def validation_step(self, validation_batch, batch_idx):
        # from our batch  place labels with
        mix_input, expected_output = self.get_batch(validation_batch)

        output = self.model(mix_input)

        return self.calculate_properties(output, expected_output)
