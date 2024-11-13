from typing import List, Any

import torch
from torch import nn, Tensor
from torch.nn import functional as F

from rizumu.filtering import wiener


class RSTFT(nn.Module):
    def __init__(self, n_fft: int = 4096,
                 win_length: int | None = None,
                 hop_length: int | None = None,
                 ):
        super(RSTFT, self).__init__()
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.window = torch.hann_window(self.n_fft)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.device != self.window.device:
            self.window = self.window.to(x.device)

        window_length = self.win_length if self.win_length is not None else (self.n_fft // 2) + 1
        previous_device = x.device
        if x.shape[-1] < window_length:
            raise Exception(f"Too small sample to apply stft, sample must be greater than {window_length}")
        stft = torch.stft(x, n_fft=self.n_fft,
                          win_length=self.win_length,
                          window=self.window,
                          hop_length=self.hop_length,
                          return_complex=True)

        return stft.to(previous_device)


class RISTFT(nn.Module):
    def __init__(self, n_fft: int = 2048, win_length: int | None = None, hop_length: int | None = None,
                 length: int | None = None):
        super(RISTFT, self).__init__()
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.length = length
        self.window = torch.hann_window(self.n_fft).to("cpu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        previous_device = x.device
        if x.device != self.window.device:
            x = x.to(self.window.device)

        stft = torch.istft(x, n_fft=self.n_fft,
                           win_length=self.win_length,
                           window=self.window,
                           hop_length=self.hop_length,
                           length=self.length
                           )
        return stft.to(previous_device)


class BLSTM(nn.Module):
    def __init__(self, dim, layers=1, skip=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lstm = nn.LSTM(bidirectional=True, num_layers=layers, hidden_size=dim, input_size=dim)
        self.linear = nn.Linear(2 * dim, dim)
        self.skip = skip

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x
        x = self.lstm(x)[0]
        x = self.linear(x)
        if self.skip:
            x = x + y
        return x


class TorchWiener(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return wiener(x, y)


class SingleEncoder(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super(SingleEncoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.l1 = nn.Linear(self.input_size, self.hidden_size, bias=False)
        self.bc1 = nn.BatchNorm1d(self.hidden_size)
        self.l2 = nn.Linear(hidden_size, output_size, bias=False)
        self.bc2 = nn.BatchNorm1d(output_size)
        self.tan1 = nn.Tanh()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xd = torch.min(x)
        if xd.isnan().any():
            raise Exception(f"xd is nan,\n{x}")
        # permute to have n-bins as final, and nb_frames as second last
        x = self.l1(x)
        x = x.permute(0, 2, 1)
        x = self.bc1(x)
        x = x.permute(0, 2, 1)
        x = self.l2(x)
        x = x.permute(0, 2, 1)
        x = self.bc2(x)
        x = x.permute(0, 2, 1)
        # limit between -1 and 1
        x = self.tan1(x)
        # if torch.isnan(self.l1.bias).any():
        #     raise Exception("nan found")
        return x


class SingleDecoder(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super(SingleDecoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.l1 = nn.Linear(input_size, hidden_size, bias=False)
        self.bc1 = nn.BatchNorm1d(hidden_size)
        self.l2 = nn.Linear(hidden_size, output_size, bias=False)
        self.bc2 = nn.BatchNorm1d(output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x = x.permute(0, 2, 1)
        # x = self.bc1(x)
        # x = x.permute(0, 2, 1)
        x = self.l1(x)
        x = x.permute(0, 2, 1)
        x = self.bc1(x)
        x = x.permute(0, 2, 1)
        x = self.l2(x)
        x = x.permute(0, 2, 1)
        x = self.bc2(x)
        x = x.permute(0, 2, 1)

        return x


class RizumuModel(nn.Module):

    def __init__(self, n_fft=4096,
                 hidden_size: int = 512,
                 real_layers: int = 3,
                 imag_layers: int = 2, weiner: bool = True):
        super(RizumuModel, self).__init__()

        # first layer is an stft layer to convert the waveform to stft
        self.stft = RSTFT(n_fft=n_fft)
        last_param = (n_fft // 2 + 1)
        self.last_param = last_param
        self.imag_layers = imag_layers
        self.real_layers = real_layers

        self.hidden_size = hidden_size
        # self.weiner = weiner
        # self.weiner_fn = TorchWiener()
        # self.weiner_fn.requires_grad_(False)

        hs_half = hidden_size // 2
        hs_quarter = hidden_size // 4

        # down u-net
        self.re1 = SingleEncoder(last_param, hidden_size, hs_half)
        self.re2 = SingleEncoder(hs_half, hidden_size, hs_quarter)

        self.ie1 = SingleEncoder(last_param, hidden_size, hs_half)
        self.ie2 = SingleEncoder(hs_half, hidden_size, hs_quarter)

        # bottleneck
        self.real_bottleneck = BLSTM(hs_quarter, layers=self.real_layers, skip=True)
        self.imag_bottleneck = BLSTM(hs_quarter, layers=self.imag_layers, skip=True)

        # up u-net
        self.real_decoder = nn.Sequential(SingleDecoder(hs_quarter, hidden_size, hs_half),
                                          SingleDecoder(hs_half, hidden_size, last_param))

        self.rd1 = SingleDecoder(hs_quarter, hidden_size, hs_half)
        self.rd2 = SingleDecoder(hs_half, hidden_size, last_param)

        self.id1 = SingleDecoder(hs_quarter, hidden_size, hs_half)
        self.id2 = SingleDecoder(hs_half, hidden_size, last_param)

        self.istft = RISTFT(n_fft=n_fft)

    def pxe(self, x: torch.Tensor, encoders: [nn.Module], bottleneck: nn.Module,
            decoders: [nn.Module]) -> torch.Tensor:
        y = x
        outputs: List[torch.Tensor] = []
        for encoder in encoders:
            x = encoder(x)
            outputs.append(x)
        x = bottleneck(x)
        # reverse outputs
        outputs.reverse()
        for arr, decoder in zip(outputs, decoders):
            x = decoder(arr * x)

        return x * y

    def normalize(self, x: torch.Tensor) -> tuple[Tensor | Any, Tensor, Tensor]:
        mean, std = torch.mean(x), torch.std(x),
        t = (x - mean) / std
        return t, mean, std

    def denormalize(self, x: Tensor, mean: torch.Tensor, std: torch.Tensor) -> Tensor:
        t = (x * std) + mean
        return t

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """

        :param x: Input tensor of shape (channels,channel_data), channel_data must be greater than
        (self.n_fft/2)+1 to allow STFT to take place
        :return: Output tensor of shape (channels,channel_data) size will match the input tensor
        """
        # I will refer to (n_fft/2+1) as nb_frames
        initial_size = x.shape[-1]
        if torch.isnan(x).any():
            raise Exception(f"x is nan")
        # compute stft of the batch
        x = self.stft(x)

        if torch.isnan(x).any():
            raise Exception("nan found")
        stft_mix = x
        # at this point we have the following dimensions
        # (n_channels,nb_frames,n_bins)
        # convert to real tensor, adds a dimension to x to separate real and imaginary
        x = torch.view_as_real(x)
        # change to (real_imaginary,n_channels,n_bins,nb_frames)
        x = x.permute(3, 0, 2, 1)

        # separate real and imaginary
        # new dimensions are (n_channels,n_bins,nb_frames)
        real, imag = torch.split(x, 1, dim=0)

        real = real.squeeze(dim=0)
        imag = imag.squeeze(dim=0)
        r_norm, r_mean, r_std = self.normalize(real)
        i_norm, i_mean, i_std = self.normalize(imag)

        # normalize
        real = r_norm
        imag = i_norm

        # decode and un-normalize
        # multiply by real to act as a mask, making it look
        # like a skip connection
        imag = self.pxe(imag, [self.ie1, self.ie2], self.imag_bottleneck, [self.id1, self.id2])
        real = self.pxe(real, [self.re1, self.re2], self.real_bottleneck, [self.rd1, self.rd2])

        real = self.denormalize(real.unsqueeze(-1), r_mean, r_std)
        imag = self.denormalize(imag.unsqueeze(-1), i_mean, i_std)
        # combine the real and imaginary layers back
        # add a new dimension we lost from the  squeeze and then join them on that layer
        # and squeeze them again
        x = torch.cat((real, imag), -1)
        x = torch.view_as_complex(x)

        if len(x.shape) == 2:
            # one dimensional layout, add the 2d that was lost
            # in the squeeze
            x = x.unsqueeze(0)

        x = x.permute(0, 2, 1)
        # if self.weiner:
        #     # spectogram is 3d, so add a 4d
        #
        #     p: torch.Tensor = x.clone().unsqueeze(-1)
        #     # permute to arrange to a weiner style
        #     p = p.permute(1, 2, 0, 3)
        #     stft_mix = stft_mix.permute(1, 2, 0)
        #
        #     p = self.weiner_fn(x, stft_mix)
        #     p = x.permute(3, 2, 0, 1)
        #     p = x.squeeze(dim=0)
        #     return p

        # compute istft
        self.istft.length = initial_size

        # # de normalize
        x = self.istft(x)
        if x.device == torch.device('mps'):
            torch.mps.empty_cache()

        return x


if __name__ == '__main__':
    with torch.autograd.set_detect_anomaly(True):
        model = RizumuModel(n_fft=4096)
        input = torch.randn(2, 199000)

        print(input.max())
        # torchinfo.summary(model, input_data=input)

        output: torch.Tensor = model(input)
        loss = F.mse_loss(output, input, reduction='mean')

        print(loss.item())
        #loss.backward()
