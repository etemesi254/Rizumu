from numpy.core.numeric import outer
import torch
from torch import nn
import pytorch_lightning as pl
import numpy as np

def calculate_loss(model_output: torch.Tensor, target: torch.Tensor, n_fft: int) -> torch.Tensor:
    a = torch.istft(model_output.squeeze().to("cpu"), window=torch.hann_window(window_length=n_fft),
                    n_fft=n_fft)
    b = torch.istft(target.squeeze().to("cpu"), window=torch.hann_window(window_length=n_fft), n_fft=n_fft)
    loss = torch.nn.functional.mse_loss(a, b)
    return loss

def calculate_sdr(target_tensor, output_tensor,n_fft:int):
    a = torch.istft(target_tensor.squeeze().to("cpu"), window=torch.hann_window(window_length=n_fft),
                    n_fft=n_fft)
    b = torch.istft(output_tensor.squeeze().to("cpu"), window=torch.hann_window(window_length=n_fft), n_fft=n_fft)

    target_tensor = a.detach().cpu().numpy()
    output_tensor = b.detach().cpu().numpy()

    target_power = np.sum(target_tensor**2)
    noise_power = np.sum((target_tensor - output_tensor)**2)

    if noise_power == 0:
        return float('inf')  # Handle the case where the noise power is zero to prevent division by zero

    sdr = 10 * np.log10(target_power / noise_power)
    return sdr

class BandPlModel(pl.LightningModule):
    def __init__(self, model: nn.Module, optimizer: torch.optim.Optimizer, n_fft: int = 2048):
        super().__init__()
        self.model = model

        self.optimizer = optimizer
        self.n_fft = n_fft

    def training_step(self, batch, batch_idx):
        mix, speech, vocals, sfx = batch
        output = self.model(mix)
        loss = calculate_loss(output, speech, self.n_fft)
        sdr  = calculate_sdr(output,speech,self.n_fft)

        self.log("train_loss", loss)
        self.log("train_sdr",sdr)
        return loss
    def configure_optimizers(self):
        return self.optimizer
    def validation_step(self, validation_batch, batch_idx):
        mix, speech, vocals, sfx = validation_batch
        output = self.model(mix)
        loss = calculate_loss(output, speech, self.n_fft)
        sdr = calculate_sdr(output,speech,self.n_fft)
        self.log("val_loss", loss)
        self.log("val_sdr",sdr)
