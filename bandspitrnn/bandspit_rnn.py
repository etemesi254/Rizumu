import typing as tp

import soundfile as sf
import torch
import torch.nn as nn
import torchinfo
from torch import Tensor

from .band_sequence import BandSequenceModelModule
from .band_spit import BandSplitModule
from .band_transformer import BandTransformerModelModule
from .mask_estimation import MaskEstimationModule



class BandSplitRNN(nn.Module):
    """
    BandSplitRNN as described in paper.
    """

    def __init__(
            self,
            sr: int,
            n_fft: int,
            bandsplits: tp.List[tp.Tuple[int, int]],
            complex_as_channel: bool,
            is_mono: bool,
            bottleneck_layer: str,
            t_timesteps: int,
            fc_dim: int,
            rnn_dim: int,
            rnn_type: str,
            bidirectional: bool,
            num_layers: int,
            mlp_dim: int,
            return_mask: bool = False
    ):
        super(BandSplitRNN, self).__init__()

        # encoder layer
        self.bandsplit = BandSplitModule(
            sr=sr,
            n_fft=n_fft,
            bandsplits=bandsplits,
            t_timesteps=t_timesteps,
            fc_dim=fc_dim,
            complex_as_channel=complex_as_channel,
            is_mono=is_mono,
        )

        # bottleneck layer
        if bottleneck_layer == 'rnn':
            self.bandsequence = BandSequenceModelModule(
                input_dim_size=fc_dim,
                hidden_dim_size=rnn_dim,
                rnn_type=rnn_type,
                bidirectional=bidirectional,
                num_layers=num_layers,
            )
        elif bottleneck_layer == 'att':
            self.bandsequence = BandTransformerModelModule(
                input_dim_size=fc_dim,
                hidden_dim_size=rnn_dim,
                num_layers=num_layers,
            )
        else:
            raise NotImplementedError

        # decoder layer
        self.maskest = MaskEstimationModule(
            sr=sr,
            n_fft=n_fft,
            bandsplits=bandsplits,
            t_timesteps=t_timesteps,
            fc_dim=fc_dim,
            mlp_dim=mlp_dim,
            complex_as_channel=complex_as_channel,
            is_mono=is_mono,
        )
        self.cac = complex_as_channel
        self.return_mask = return_mask

    def wiener(self, x_hat: torch.Tensor, x_complex: torch.Tensor) -> torch.Tensor:
        """
        Wiener filtering of the input signal
        """
        # TODO: add Wiener Filtering
        return x_hat

    def compute_mask(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes complex-valued T-F mask.
        """
        x = self.bandsplit(x)  # [batch_size, k_subbands, time, fc_dim]
        x = self.bandsequence(x)  # [batch_size, k_subbands, time, fc_dim]
        x = self.maskest(x)  # [batch_size, freq, time]

        return x

    def forward(self, x: torch.Tensor):
        """
        Input and output are T-F complex-valued features.
        Input shape: batch_size, n_channels, freq, time]
        Output shape: batch_size, n_channels, freq, time]
        """
        # use only magnitude if not using complex input
        x_complex = None
        if not self.cac:
            x_complex = x
            x = x.abs()
        # normalize
        # TODO: Try to normalize in bandsplit and denormalize in maskest
        mean = x.mean(dim=(1, 2, 3), keepdim=True)
        std = x.std(dim=(1, 2, 3), keepdim=True)
        x = (x - mean) / (std + 1e-5)

        # compute T-F mask
        mask = self.compute_mask(x)

        # multiply with original tensor
        x = mask if self.return_mask else mask * x

        # denormalize
        x = x * std + mean

        if not self.cac:
            x = self.wiener(x, x_complex)

        return x


def load_and_pad(file: str, n_fft=2048):
    import librosa

    data, sr = librosa.load(file, sr=None,
                            mono=False)

    data_tc = torch.from_numpy(data)

    # return d,n_channels,nfft,time
    window = torch.hamming_window(n_fft)
    stft = data_tc.stft(n_fft=n_fft, window=window, return_complex=True)
    new_shape = stft.reshape((1, 1, stft.shape[0], stft.shape[1]))
    return new_shape, sr


def convert_to_audio(tensor: Tensor, n_fft=2048, sample_rate=44100):
    tensor = tensor.squeeze()
    window = torch.hamming_window(n_fft)
    c = tensor.istft(n_fft, window=window)
    d = c.cpu().detach().numpy()

    sf.write(file="out.wav", samplerate=sample_rate, data=d, subtype='PCM_24')
    print(d.shape)


if __name__ == '__main__':
    n_fft = 2048
    (in_features, sr) = load_and_pad('/Users/etemesi/PycharmProjects/Spite/data/dnr_v2/cv/89918/mix.wav', n_fft)
    time = in_features.shape[-1]
    cfg = {
        "sr": sr,
        "n_fft": n_fft,
        "bandsplits": [
            (1000, 100),
            (2500, 200),
            (4000, 400),
            (8000, 600),
            (16000, 2000),
            (20000, 2000),
        ],
        "complex_as_channel": True,
        "is_mono": True,
        "bottleneck_layer": 'rnn',
        # "t_timesteps": time,
        "fc_dim": 128,
        "rnn_dim": 256,
        "rnn_type": "LSTM",
        "bidirectional": True,
        "num_layers": 1,
        "t_timesteps": time,
        "mlp_dim": 512,
        "return_mask": False,
    }
    model = BandSplitRNN(**cfg)
    _ = model.eval()

    with torch.no_grad():
        out_features: Tensor = model(in_features)

    torchinfo.summary(model)
    convert_to_audio(out_features)

    # print(model)
    print(f"Total number of parameters: {sum([p.numel() for p in model.parameters()])}")
    print(f"In shape: {in_features.shape}\nOut shape: {out_features.shape}")
    print(f"In dtype: {in_features.dtype}\nOut dtype: {out_features.dtype}")
