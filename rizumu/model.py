from typing import List

import torch
from torch import nn
from torch.nn import functional as F


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

        if x.shape[-1] < window_length:
            raise Exception(f"Too small sample to apply stft, sample must be greater than {window_length}")
        stft = torch.stft(x, n_fft=self.n_fft,
                          win_length=self.win_length,
                          window=self.window,
                          hop_length=self.hop_length,
                          return_complex=True)

        return stft


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
        if x.device != self.window.device:
            x = x.to(self.window.device)

        stft = torch.istft(x, n_fft=self.n_fft,
                           win_length=self.win_length,
                           window=self.window,
                           hop_length=self.hop_length,
                           length=self.length
                           )
        return stft


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


class SingleEncoder(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super(SingleEncoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.l1 = nn.Conv1d(self.input_size, self.hidden_size, 1)
        self.ls = nn.ReLU()
        self.c1 = nn.Conv1d(hidden_size, output_size, 1)
        self.bc1 = nn.BatchNorm1d(output_size)
        self.tan1 = nn.Tanh()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xd = torch.min(x)
        if xd.isnan().any():
            raise Exception(f"xd is nan")
        # permute to have n-bins as final, and nb_frames as second last
        rx_a = x.permute(0, 2, 1)
        rx_b = self.l1(rx_a)

        rx_c = self.c1(rx_b)

        rx_d = self.bc1(rx_c)
        # rx = self.tan1(rx)
        rx_e = rx_d.permute(0, 2, 1)
        if torch.isnan(self.l1.bias).any():
            raise Exception("nan found")
        return rx_e


class SingleDecoder(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super(SingleDecoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.tan1 = nn.Tanh()
        self.bc1 = nn.BatchNorm1d(input_size)
        self.c1 = nn.ConvTranspose1d(input_size, hidden_size, 1)
        self.ls = nn.ReLU()
        self.l1 = nn.ConvTranspose1d(hidden_size, output_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rx = x.permute(0, 2, 1)
        rx = self.tan1(rx)
        rx = self.bc1(rx)

        rx = self.c1(rx)
        rx = self.ls(rx)
        rx = self.l1(rx)
        rx = rx.permute(0, 2, 1)
        return rx


class RizumuModel(nn.Module):

    def __init__(self, n_fft=4096,
                 hidden_size: int = 512,
                 real_layers: int = 3,
                 imag_layers: int = 2):
        super(RizumuModel, self).__init__()

        # first layer is an stft layer to convert the waveform to stft
        self.stft = RSTFT(n_fft=n_fft)
        last_param = (n_fft // 2 + 1)
        self.last_param = last_param
        self.imag_layers = imag_layers
        self.real_layers = real_layers

        self.hidden_size = hidden_size

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
            x = decoder(arr*x)

        return x * y

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """

        :param x: Input tensor of shape (channels,channel_data), channel_data must be greater than
        (self.n_fft/2)+1 to allow STFT to take place
        :return: Output tensor of shape (channels,channel_data) size will match the input tensor
        """
        # I will refer to (n_fft/2+1) as nb_frames
        initial_size = x.shape[-1]
        # compute stft of the batch
        x = self.stft(x)
        # at this point we have the following dimensions
        # (n_channels,nb_frames,n_bins)
        # convert to real tensor, adds a dimension to x to separate real and imaginary
        x = torch.view_as_real(x)
        # change to (real_imaginary,n_channels,n_bins,nb_frames)
        x = x.permute(3, 0, 2, 1)

        # separate real and imaginary
        # new dimensions are (n_channels,n_bins,nb_frames)
        real = x[0:1].squeeze(dim=0)
        imag = x[1:2].squeeze(dim=0)

        inv_real_max = 1. / (real.max() + 1e-8)
        inv_imag_max = 1. / (imag.max() + 1e-8)

        real_max = 1.0 / inv_real_max
        imag_max = 1.0 / inv_imag_max

        # normalize
        real = real * inv_real_max
        imag = imag * inv_imag_max

        # # encode the real and imaginary parts
        # rx_1 = self.real_encoder(real)
        # # encode one layer
        #
        # ix_1 = self.imag_encoder(imag)
        # # bottleneck
        # ix_1 = self.imag_bottleneck(ix)
        # rx_1 = self.real_bottleneck(rx)

        # decode and unnormalize
        # multiply by real to act as a mask, making it look
        # like a skip connection
        ix_2 = self.pxe(imag, [self.ie1, self.ie2], self.imag_bottleneck, [self.id1, self.id2])
        rx_2 = self.pxe(real, [self.re1, self.re2], self.real_bottleneck, [self.rd1, self.rd2])

        # combine the real and imaginary layers back
        # add a new dimension we lost from the  squeeze and then join them on that layer
        # and squeeze them again
        x = torch.cat((rx_2.unsqueeze(-1) * real_max,
                       ix_2.unsqueeze(-1) * imag_max), -1)
        x = torch.view_as_complex(x)

        if len(x.shape) == 2:
            # one dimensional layout, add the 2d that was lost
            # in the squeeze
            x = x.unsqueeze(0)

        x = x.permute(0, 2, 1)
        # compute istft
        self.istft.length = initial_size
        # de normalize
        x = self.istft(x)
        return x


if __name__ == '__main__':
    import torchinfo

    model = RizumuModel(n_fft=4096)
    input = torch.randn(1, 199000)

    torchinfo.summary(model, input_data=input)

    output: torch.Tensor = model(input)
    loss = F.mse_loss(output, input, reduction='mean')

    print(loss.item())
    loss.backward()
