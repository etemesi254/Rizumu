from enum import Enum
from typing import List

import torch
from torch import nn, Tensor
from torch.nn import functional as F

norm_bias = 1e-8


class ModelType(Enum):
    Linear = 1
    Convolutional = 2


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
            x = torch.nn.functional.pad(x, (0, window_length - x.shape[-1]))
        stft = torch.stft(x,
                          n_fft=self.n_fft,
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
        x = self.lstm(x)[0]
        x = self.linear(x)

        return x


class SingleEncoder(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int,
                 activate: bool = True):
        super(SingleEncoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.activate = activate
        self.output_size = output_size

        self.l1 = nn.Linear(self.input_size, self.hidden_size, bias=True)
        self.bc1 = nn.BatchNorm1d(self.hidden_size)
        self.l2 = nn.Linear(hidden_size, output_size, bias=True)
        self.bc2 = nn.BatchNorm1d(output_size)
        self.tan1 = nn.Tanh()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.l1(x)
        x = x.permute(0, 2, 1)
        x = self.bc1(x)
        x = x.permute(0, 2, 1)
        x = self.l2(x)
        x = x.permute(0, 2, 1)
        x = self.bc2(x)
        x = x.permute(2, 0, 1)
        # # limit between -1 and 1
        if self.activate:
            x = self.tan1(x)
        return x


class SingleDecoder(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int, activate: bool = True):
        super(SingleDecoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.activate = activate

        self.l1 = nn.Linear(input_size, hidden_size, bias=True)
        self.bc1 = nn.BatchNorm1d(hidden_size)
        self.l2 = nn.Linear(hidden_size, output_size, bias=True)
        self.bc2 = nn.BatchNorm1d(output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(1, 0, 2)
        x = self.l1(x)
        x = x.permute(0, 2, 1)
        x = self.bc1(x)
        x = x.permute(0, 2, 1)
        x = self.l2(x)
        x = x.permute(0, 2, 1)
        x = self.bc2(x)
        # x = x.permute(0, 2, 1)
        # if self.activate:
        #     x = F.relu(x)
        return x


def denormalize(x: Tensor, mean: torch.Tensor, std: torch.Tensor) -> Tensor:
    t = (x * (std - norm_bias)) + mean
    return t


def normalize(x: torch.Tensor) -> tuple[Tensor, Tensor, Tensor]:
    mean, std = torch.mean(x), (torch.std(x) + norm_bias),
    t = (x - mean) / std
    return t, mean, std


def exec_unet(x: torch.Tensor, encoders: [nn.Module], bottleneck: nn.Module,
              decoders: [nn.Module], is_mask: bool) -> torch.Tensor:
    outputs: List[torch.Tensor] = []
    for encoder in encoders:
        x = encoder(x)
        outputs.append(x)
    x = bottleneck(x)
    # reverse outputs
    outputs.reverse()
    for arr, decoder in zip(outputs, decoders):
        if is_mask:
            x = decoder(x * arr)
        else:
            x = decoder(x - arr)
    #x = F.relu(x)
    return x


class RizumuBase(nn.Module):
    def __init__(self, size: int,
                 real_layers: int = 1,
                 imag_layers: int = 1,
                 n_fft: int = 2048,
                 hidden_size: int = 512,
                 activate: bool = True,
                 apply_istft: bool = True, is_mask: bool = False):
        super(RizumuBase, self).__init__()
        self.size = size
        hs_half = size // 2
        hs_quarter = size // 4
        self.real_layers = real_layers
        self.imag_layers = imag_layers
        self.is_mask = is_mask

        # down u-net
        self.re1 = SingleEncoder(self.size, hidden_size, hs_half, activate)
        self.ie1 = SingleEncoder(self.size, hidden_size, hs_half, activate)

        # bottleneck
        self.real_bottleneck = BLSTM(hs_half, layers=self.real_layers, skip=True)
        self.imag_bottleneck = BLSTM(hs_half, layers=self.imag_layers, skip=True)

        self.rd2 = SingleDecoder(hs_half, hidden_size, self.size, activate)
        self.id2 = SingleDecoder(hs_half, hidden_size, self.size, activate)

        self.apply_istft = apply_istft
        self.istft = RISTFT(n_fft=n_fft)

    def forward(self, initial_size: int,
                real: torch.Tensor, imag: torch.Tensor,
                r_mean: torch.Tensor, r_std: torch.Tensor,
                i_mean: torch.Tensor, i_std: torch.Tensor) -> torch.Tensor:

        imag = exec_unet(imag, [self.ie1, ], self.imag_bottleneck, [self.id2], self.is_mask)
        real = exec_unet(real, [self.re1, ], self.real_bottleneck, [self.rd2], self.is_mask)

        real = denormalize(real.unsqueeze(-1), r_mean, r_std)
        imag = denormalize(imag.unsqueeze(-1), i_mean, i_std)

        x = torch.cat((real, imag), -1)
        x = torch.view_as_complex(x)

        if len(x.shape) == 2:
            # one dimensional layout, add the 2d that was lost
            # in the squeeze
            x = x.unsqueeze(0)

        if self.apply_istft:
            self.istft.length = initial_size
            return self.istft(x)
        else:
            return x


class RizumuModel(nn.Module):

    def __init__(self, n_fft: int = 4096,
                 hidden_size: int = 512,
                 real_layers: int = 2,
                 imag_layers: int = 2,
                 ):
        super(RizumuModel, self).__init__()

        # first layer is an stft layer to convert the waveform to stft
        self.stft = RSTFT(n_fft=n_fft)
        last_param = (n_fft // 2) + 1
        self.last_param = last_param
        self.imag_layers = imag_layers
        self.real_layers = real_layers

        self.hidden_size = hidden_size
        self.m_model = RizumuBase(size=last_param,
                                  real_layers=real_layers,
                                  imag_layers=imag_layers,
                                  n_fft=n_fft,
                                  hidden_size=hidden_size,
                                  activate=True,
                                  apply_istft=False,
                                  is_mask=True)

        self.istft = RISTFT(n_fft=n_fft)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """

        :param x: Input tensor of shape (channels,channel_data), channel_data must be greater than
        (self.n_fft/2)+1 to allow STFT to take place
        :return: Output tensor of shape (channels,channel_data) size will match the input tensor
        """
        # I will refer to (n_fft/2+1) as nb_frames
        original = x
        initial_size = x.shape[-1]
        # compute stft of the batch
        x = self.stft(x)
        orig = x

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
        # normalize
        r_norm, r_mean, r_std = normalize(real)
        i_norm, i_mean, i_std = normalize(imag)

        # normalize
        real = r_norm
        imag = i_norm

        # decode and un-normalize
        m = self.m_model(initial_size, real, imag, r_mean, r_std, i_mean, i_std)
        # calculate mx part
        mx = (orig * m)
        self.istft.length = initial_size
        mx = self.istft(mx)

        return mx


class RizumuBaseV2(nn.Module):
    def __init__(self, size: int, hidden_size: int = 512, real_layers: int = 1, imag_layers: int = 1, activate=True):
        super(RizumuBaseV2, self).__init__()
        self.size = size
        self.hidden_size = hidden_size
        self.real_layers = real_layers
        self.imag_layers = imag_layers
        hs_half = size // 2
        self.is_mask = True

        # down u-net
        self.re1 = SingleEncoder(self.size, hidden_size, hs_half, activate)
        self.ie1 = SingleEncoder(self.size, hidden_size, hs_half, activate)

        # bottleneck
        self.real_bottleneck = BLSTM(hs_half, layers=self.real_layers, skip=True)
        self.imag_bottleneck = BLSTM(hs_half, layers=self.imag_layers, skip=True)

        self.rd2 = SingleDecoder(hs_half, hidden_size, self.size, activate)
        self.id2 = SingleDecoder(hs_half, hidden_size, self.size, activate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
        # normalize
        real, r_mean, r_std = normalize(real)
        imag, i_mean, i_std = normalize(imag)
        # generate mask
        mask_imag = exec_unet(imag, [self.ie1,], self.imag_bottleneck, [self.id2], self.is_mask)
        mask_real = exec_unet(real, [self.re1,], self.real_bottleneck, [self.rd2], self.is_mask)

        # the signal at this point is istft so we can multiply
        real = real.permute(0, 2, 1)
        imag = imag.permute(0, 2, 1)

        real = mask_real
        imag =  mask_imag

        real = denormalize(real.unsqueeze(-1), r_mean, r_std)
        imag = denormalize(imag.unsqueeze(-1), i_mean, i_std)

        # convert back to complex tensor and return
        x = torch.cat((real, imag), -1)
        x = torch.view_as_complex(x)

        return x


class RizumuModelV2(nn.Module):
    def __init__(self, n_fft: int = 2048, num_splits: int = 5, hidden_size: int = 512 ):
        super(RizumuModelV2, self).__init__()
        self.stft = RSTFT(n_fft=n_fft)
        self.istft = RISTFT(n_fft=n_fft)
        last_param = (n_fft // 2) + 1
        self.last_param = last_param
        self.num_splits = num_splits
        self.hidden_size = hidden_size
        split_sizes_diff = []
        single_split = (last_param + (self.num_splits - 1)) // self.num_splits

        for i in range(num_splits):
            start = single_split * i
            end = single_split * (i + 1)
            end = min(end, last_param)
            split_sizes_diff.append(end - start)
        self.models = nn.ModuleList([])
        for i in range(num_splits):
            self.models.append(RizumuBaseV2(size=split_sizes_diff[i], hidden_size=self.hidden_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # step 1. perfom stft on the signal
        initial_size = x.shape[-1]
        x = self.stft(x)

        # step 2, split based on configured categories
        # x shape is (channels,n_bins,n_timesteps)
        channels, n_bins, n_timesteps = x.shape
        # round up division.
        single_split = (n_bins + (self.num_splits - 1)) // self.num_splits

        # step 3, collect into categories
        input_divs = []
        for i in range(self.num_splits):
            input_divs.append(x[:, i * single_split:(i + 1) * single_split, :])

        results = []
        # run the models
        for pos, i in enumerate(input_divs):
            results.append(self.models[pos](i))

        # combine the models based on the split location
        results = torch.cat(results, dim=1)
        self.istft.length = initial_size
        x = self.istft(results)
        return x


if __name__ == '__main__':
    import torchinfo

    model = RizumuModelV2()
    input = torch.rand((1, 21203))
    # torchinfo.summary(model, input_data=input)
    # model(input)
    with torch.autograd.set_detect_anomaly(True):
        model = RizumuModelV2(n_fft=2048, num_splits=10)
        input = torch.randn((2, 59090))

        torchinfo.summary(model, input_data=input)

        output: torch.Tensor = model(input)
        loss = F.mse_loss(output, input, reduction='mean')

        print(loss.item())
        # loss.backward()
