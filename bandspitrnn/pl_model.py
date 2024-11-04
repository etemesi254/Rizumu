import torch
from torch import nn
import pytorch_lightning as pl

def calculate_loss(model_output: torch.Tensor, target: torch.Tensor, n_fft: int) -> torch.Tensor:
    a = torch.istft(model_output.squeeze().to("cpu"), window=torch.hann_window(window_length=n_fft),
                    n_fft=n_fft)
    b = torch.istft(target.squeeze().to("cpu"), window=torch.hann_window(window_length=n_fft), n_fft=n_fft)
    loss = torch.nn.functional.mse_loss(a, b)
    return loss


class BandPlModel(pl.LightningModule):
    def __init__(self, model: nn.Module, optimizer: torch.optim.Optimizer, n_fft: int = 2048):
        super().__init__()
        self.model = model

        self.optimizer = optimizer
        self.n_fft = n_fft

    def training_step(self, batch, batch_idx):
        mix, speech, vocals, sfx = batch
        output = self.model(mix)
        loss = calculate_loss(output, mix, self.n_fft)
        self.log("train_loss", loss)
        return loss
    def configure_optimizers(self):
        return self.optimizer
    def validation_step(self, validation_batch, batch_idx):
        mix, speech, vocals, sfx = validation_batch
        output = self.model(mix)
        loss = calculate_loss(output, mix, self.n_fft)
        self.log("val_loss", loss)





