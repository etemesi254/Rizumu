from typing import List, Tuple

import numpy as np
import pytorch_lightning as pl
import torch
from torch.optim import Adam

from rizumu.model import RizumuModel

loss_constant = 1000


def calculate_loss(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Calculate the  MSE loss between two tensors
    :param a: Input tensor
    :param b: Another tensor
    :return: The MSE loss
    """
    loss = torch.nn.functional.mse_loss(a, b)
    return loss


def calculate_sdr(target_tensor, output_tensor) -> float:
    """
     Calculate the signal to distortion ratio between target and output tensor
    :param target_tensor: The true expected output
    :param output_tensor: The predicted output
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


def calculate_snr(signal_tensor, noise_tensor):
    """
    Calculates the Signal-to-Noise Ratio (SNR) between a signal tensor and a noise tensor.

    Args:
        signal_tensor: The signal tensor.
        noise_tensor: The noise tensor.

    Returns:
        The SNR value in dB.
    """

    signal_power = torch.mean(signal_tensor ** 2)
    noise_power = torch.mean(noise_tensor ** 2)

    if noise_power == 0:
        return float('inf')

    snr = 10 * torch.log10(signal_power / noise_power)
    return snr.item()


class RizumuLightning(pl.LightningModule):
    def __init__(self,
                 labels: List[str],
                 output_label_name: str,
                 mix_name: str,
                 n_fft: int = 2048,
                 hidden_size: int = 512,
                 num_splits: int = 1,
                 input_channels=1,
                 output_channels=1,
                 lstm_layers: int = 1,
                 depth=4,

                 lr=1e-3):
        assert mix_name in labels, "Mix is not in labels please include it"
        assert output_label_name in labels, "Output label is not in labels please include it"

        super().__init__()

        self.save_hyperparameters()
        self.model = RizumuModel(n_fft=n_fft,
                                 num_splits=num_splits,
                                 hidden_size=hidden_size,
                                 depth=depth,
                                 input_channels=input_channels,
                                 output_channels=output_channels,
                                 lstm_layers=lstm_layers)
        self.optimizer = Adam(self.model.parameters(), lr=lr)
        self.labels = labels
        self.output_label_name = output_label_name
        self.mix_name = mix_name

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

    def calculate_properties(self, output_istft: torch.Tensor, speech_istft: torch.Tensor, prefix: str) -> torch.Tensor:
        # perform istft
        output_istft = output_istft.squeeze().to("cpu")
        speech_istft = speech_istft.squeeze().to("cpu")

        loss = calculate_loss(output_istft, speech_istft)
        sdr = calculate_sdr(output_istft, speech_istft)

        self.log(f"{prefix}_loss", loss, prog_bar=True)
        self.log(f"{prefix}_sdr", sdr, prog_bar=True)
        new_loss = loss + (100 - sdr) / 100
        return new_loss

    def training_step(self, batch: List[torch.Tensor], batch_idx):
        # from our batch  place labels with
        mix_input, expected_output = self.get_batch(batch)
        output = self.model(mix_input)

        return self.calculate_properties(output, expected_output, prefix="train")

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        output = self.model(tensor)
        return output

    def configure_optimizers(self):
        return self.optimizer

    def validation_step(self, validation_batch, batch_idx):
        # from our batch  place labels with
        mix_input, expected_output = self.get_batch(validation_batch)

        output = self.model(mix_input)

        return self.calculate_properties(output, expected_output, prefix="val")
