import math
from typing import Union

import torch
from torch import nn

import typing as tp

if __name__=="__main__":
    from blstm import BLSTM
else:
    from .blstm import BLSTM

from torch.nn import functional as F


def center_trim(tensor: torch.Tensor, reference: Union[torch.Tensor, int]):
    """
    Center trim `tensor` with respect to `reference`, along the last dimension.
    `reference` can also be a number, representing the length to trim to.
    If the size difference != 0 mod 2, the extra sample is removed on the right side.
    """
    ref_size: int
    if isinstance(reference, torch.Tensor):
        ref_size = reference.size(-1)
    else:
        ref_size = reference
    delta = tensor.size(-1) - ref_size
    if delta < 0:
        raise ValueError("tensor must be larger than reference. " f"Delta is {delta}.")
    if delta:
        tensor = tensor[..., delta // 2:-(delta - delta // 2)]
    return tensor


class LayerScale(nn.Module):
    """Layer scale from [Touvron et al 2021] (https://arxiv.org/pdf/2103.17239.pdf).
    This rescales diagonaly residual outputs close to 0 initially, then learnt.
    """

    def __init__(self, channels: int, init: float = 0, channel_last=False):
        """
        channel_last = False corresponds to (B, C, T) tensors
        channel_last = True corresponds to (T, B, C) tensors
        """
        super().__init__()
        self.channel_last = channel_last
        self.scale = nn.Parameter(torch.zeros(channels, requires_grad=True))
        self.scale.data[:] = init

    def forward(self, x):
        if self.channel_last:
            return self.scale * x
        else:
            return self.scale[:, None] * x


class DConv(nn.Module):
    """
    New residual branches in each encoder layer.
    This alternates dilated convolutions, potentially with LSTMs and attention.
    Also before entering each residual branch, dimension is projected on a smaller subspace,
    e.g. of dim `channels // compress`.
    """
    def __init__(self, channels: int, compress: float = 4, depth: int = 2, init: float = 1e-4,
                 norm=True, attn=False, heads=4, ndecay=4, lstm=False, gelu=True,
                 kernel=3, dilate=True):
        """
        Args:
            channels: input/output channels for residual branch.
            compress: amount of channel compression inside the branch.
            depth: number of layers in the residual branch. Each layer has its own
                projection, and potentially LSTM and attention.
            init: initial scale for LayerNorm.
            norm: use GroupNorm.
            attn: use LocalAttention.
            heads: number of heads for the LocalAttention.
            ndecay: number of decay controls in the LocalAttention.
            lstm: use LSTM.
            gelu: Use GELU activation.
            kernel: kernel size for the (dilated) convolutions.
            dilate: if true, use dilation, increasing with the depth.
        """

        super().__init__()
        assert kernel % 2 == 1
        self.channels = channels
        self.compress = compress
        self.depth = abs(depth)
        dilate = depth > 0

        norm_fn: tp.Callable[[int], nn.Module]
        norm_fn = lambda d: nn.Identity()  # noqa
        if norm:
            norm_fn = lambda d: nn.GroupNorm(1, d)  # noqa

        hidden = int(channels / compress)

        act: tp.Type[nn.Module]
        if gelu:
            act = nn.GELU
        else:
            act = nn.ReLU

        self.layers = nn.ModuleList([])
        for d in range(self.depth):
            dilation = 2 ** d if dilate else 1
            padding = dilation * (kernel // 2)
            mods = [
                nn.Conv1d(channels, hidden, kernel, dilation=dilation, padding=padding),
                norm_fn(hidden), act(),
                nn.Conv1d(hidden, 2 * channels, 1),
                norm_fn(2 * channels), nn.GLU(1),
                LayerScale(channels, init),
            ]
            if attn:
                mods.insert(3, LocalState(hidden, heads=heads, ndecay=ndecay))
            if lstm:
                mods.insert(3, BLSTM(hidden, layers=2, max_steps=200, skip=True))
            layer = nn.Sequential(*mods)
            self.layers.append(layer)

    def forward(self, x):
        for layer in self.layers:
            x = x + layer(x)
        return x


class LocalState(nn.Module):
    """Local state allows to have attention based only on data (no positional embedding),
    but while setting a constraint on the time window (e.g. decaying penalty term).

    Also a failed experiments with trying to provide some frequency based attention.
    """
    def __init__(self, channels: int, heads: int = 4, nfreqs: int = 0, ndecay: int = 4):
        super().__init__()
        assert channels % heads == 0, (channels, heads)
        self.heads = heads
        self.nfreqs = nfreqs
        self.ndecay = ndecay
        self.content = nn.Conv1d(channels, channels, 1)
        self.query = nn.Conv1d(channels, channels, 1)
        self.key = nn.Conv1d(channels, channels, 1)
        if nfreqs:
            self.query_freqs = nn.Conv1d(channels, heads * nfreqs, 1)
        if ndecay:
            self.query_decay = nn.Conv1d(channels, heads * ndecay, 1)
            # Initialize decay close to zero (there is a sigmoid), for maximum initial window.
            self.query_decay.weight.data *= 0.01
            assert self.query_decay.bias is not None  # stupid type checker
            self.query_decay.bias.data[:] = -2
        self.proj = nn.Conv1d(channels + heads * nfreqs, channels, 1)

    def forward(self, x):
        B, C, T = x.shape
        heads = self.heads
        indexes = torch.arange(T, device=x.device, dtype=x.dtype)
        # left index are keys, right index are queries
        delta = indexes[:, None] - indexes[None, :]

        queries = self.query(x).view(B, heads, -1, T)
        keys = self.key(x).view(B, heads, -1, T)
        # t are keys, s are queries
        dots = torch.einsum("bhct,bhcs->bhts", keys, queries)
        dots /= keys.shape[2]**0.5
        if self.nfreqs:
            periods = torch.arange(1, self.nfreqs + 1, device=x.device, dtype=x.dtype)
            freq_kernel = torch.cos(2 * math.pi * delta / periods.view(-1, 1, 1))
            freq_q = self.query_freqs(x).view(B, heads, -1, T) / self.nfreqs ** 0.5
            dots += torch.einsum("fts,bhfs->bhts", freq_kernel, freq_q)
        if self.ndecay:
            decays = torch.arange(1, self.ndecay + 1, device=x.device, dtype=x.dtype)
            decay_q = self.query_decay(x).view(B, heads, -1, T)
            decay_q = torch.sigmoid(decay_q) / 2
            decay_kernel = - decays.view(-1, 1, 1) * delta.abs() / self.ndecay**0.5
            dots += torch.einsum("fts,bhfs->bhts", decay_kernel, decay_q)

        # Kill self reference.
        dots.masked_fill_(torch.eye(T, device=dots.device, dtype=torch.bool), -100)
        weights = torch.softmax(dots, dim=2)

        content = self.content(x).view(B, heads, -1, T)
        result = torch.einsum("bhts,bhct->bhcs", weights, content)
        if self.nfreqs:
            time_sig = torch.einsum("bhts,fts->bhfs", weights, freq_kernel)
            result = torch.cat([result, time_sig], 2)
        result = result.reshape(B, -1, T)
        return x + self.proj(result)


class Demucs(nn.Module):
    def __init__(self,
                 sources,
                 # Channels
                 audio_channels=2,
                 channels=64,
                 growth=2.,
                 # Main structure
                 depth=6,
                 rewrite=True,
                 lstm_layers=0,
                 # Convolutions
                 kernel_size=8,
                 stride=4,
                 context=1,
                 # Activations
                 gelu=True,
                 glu=True,
                 # Normalization
                 norm_starts=4,
                 norm_groups=4,
                 # DConv residual branch
                 dconv_mode=1,
                 dconv_depth=2,
                 dconv_comp=4,
                 dconv_attn=4,
                 dconv_lstm=4,
                 dconv_init=1e-4,
                 # Pre/post processing
                 normalize=True,
                 # Weight init
                 samplerate=44100,
                 segment=4 * 10):
        """
        Args:
            sources (list[str]): list of source names
            audio_channels (int): stereo or mono
            channels (int): first convolution channels
            depth (int): number of encoder/decoder layers
            growth (float): multiply (resp divide) number of channels by that
                for each layer of the encoder (resp decoder)
            depth (int): number of layers in the encoder and in the decoder.
            rewrite (bool): add 1x1 convolution to each layer.
            lstm_layers (int): number of lstm layers, 0 = no lstm. Deactivated
                by default, as this is now replaced by the smaller and faster small LSTMs
                in the DConv branches.
            kernel_size (int): kernel size for convolutions
            stride (int): stride for convolutions
            context (int): kernel size of the convolution in the
                decoder before the transposed convolution. If > 1,
                will provide some context from neighboring time steps.
            gelu: use GELU activation function.
            glu (bool): use glu instead of ReLU for the 1x1 rewrite conv.
            norm_starts: layer at which group norm starts being used.
                decoder layers are numbered in reverse order.
            norm_groups: number of groups for group norm.
            dconv_mode: if 1: dconv in encoder only, 2: decoder only, 3: both.
            dconv_depth: depth of residual DConv branch.
            dconv_comp: compression of DConv branch.
            dconv_attn: adds attention layers in DConv branch starting at this layer.
            dconv_lstm: adds a LSTM layer in DConv branch starting at this layer.
            dconv_init: initial scale for the DConv branch LayerScale.
            normalize (bool): normalizes the input audio on the fly, and scales back
                the output by the same amount.
            resample (bool): upsample x2 the input and downsample /2 the output.
            rescale (float): rescale initial weights of convolutions
                to get their standard deviation closer to `rescale`.
            samplerate (int): stored as meta information for easing
                future evaluations of the model.
            segment (float): duration of the chunks of audio to ideally evaluate the model on.
                This is used by `demucs.apply.apply_model`.
        """

        super().__init__()
        self.audio_channels = audio_channels
        self.sources = sources
        self.kernel_size = kernel_size
        self.context = context
        self.stride = stride
        self.depth = depth
        self.channels = channels
        self.normalize = normalize
        self.samplerate = samplerate
        self.segment = segment
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        self.skip_scales = nn.ModuleList()

        if glu:
            activation = nn.GLU(dim=1)
            ch_scale = 2
        else:
            activation = nn.ReLU()
            ch_scale = 1
        if gelu:
            act2 = nn.GELU
        else:
            act2 = nn.ReLU

        in_channels = audio_channels
        padding = 0
        for index in range(depth):
            norm_fn = lambda d: nn.Identity()  # noqa
            if index >= norm_starts:
                norm_fn = lambda d: nn.GroupNorm(norm_groups, d)  # noqa

            encode = []
            encode += [
                nn.Conv1d(in_channels, channels, kernel_size, stride),
                norm_fn(channels),
                act2(),
            ]
            attn = index >= dconv_attn
            lstm = index >= dconv_lstm
            if dconv_mode & 1:
                encode += [DConv(channels, depth=dconv_depth, init=dconv_init,
                                 compress=dconv_comp, attn=attn, lstm=lstm)]
            if rewrite:
                encode += [
                    nn.Conv1d(channels, ch_scale * channels, 1),
                    norm_fn(ch_scale * channels), activation]
            self.encoder.append(nn.Sequential(*encode))

            decode = []
            if index > 0:
                out_channels = in_channels
            else:
                out_channels = len(self.sources) * audio_channels
            if rewrite:
                decode += [
                    nn.Conv1d(channels, ch_scale * channels, 2 * context + 1, padding=context),
                    norm_fn(ch_scale * channels), activation]
            if dconv_mode & 2:
                decode += [DConv(channels, depth=dconv_depth, init=dconv_init,
                                 compress=dconv_comp, attn=attn, lstm=lstm)]
            decode += [nn.ConvTranspose1d(channels, out_channels,
                       kernel_size, stride, padding=padding)]
            if index > 0:
                decode += [norm_fn(out_channels), act2()]
            self.decoder.insert(0, nn.Sequential(*decode))
            in_channels = channels
            channels = int(growth * channels)

        channels = in_channels
        if lstm_layers:
            self.lstm = BLSTM(channels, lstm_layers)
        else:
            self.lstm = None



    def valid_length(self, length):
        """
        Return the nearest valid length to use with the model so that
        there is no time steps left over in a convolution, e.g. for all
        layers, size of the input - kernel_size % stride = 0.

        Note that input are automatically padded if necessary to ensure that the output
        has the same length as the input.
        """

        for _ in range(self.depth):
            length = math.ceil((length - self.kernel_size) / self.stride) + 1
            length = max(1, length)

        for idx in range(self.depth):
            length = (length - 1) * self.stride + self.kernel_size

        return int(length)

    def forward(self, mix):
        x = mix
        length = x.shape[-1]

        if self.normalize:
            mono = mix.mean(dim=1, keepdim=True)
            mean = mono.mean(dim=-1, keepdim=True)
            std = mono.std(dim=-1, keepdim=True)
            x = (x - mean) / (1e-5 + std)
        else:
            mean = 0
            std = 1

        delta = self.valid_length(length) - length
        x = F.pad(x, (delta // 2, delta - delta // 2))


        saved = []
        for encode in self.encoder:
            x = encode(x)
            saved.append(x)

        if self.lstm:
            x = self.lstm(x)

        for decode in self.decoder:
            skip = saved.pop(-1)
            skip = center_trim(skip, x)
            x = decode(x + skip)

        x = x * std + mean
        x = center_trim(x, length)
        x = x.view(x.size(0), len(self.sources), self.audio_channels, x.size(-1))
        return x



if __name__ == '__main__':

    import torchinfo
    model = Demucs(sources=["speech"])
    a, dim, c = 1, 2, 200
    input = torch.randn((a, dim, c))
    output = model(input)
    print(output.squeeze().shape)
    print(input.squeeze().shape)
    torchinfo.summary(model,input.shape)