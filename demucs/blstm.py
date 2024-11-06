import math

import torch
from torch import nn
from torch.nn import functional as F


def unfold(a, kernel_size, stride):
    """Given input of size [*OT, T], output Tensor of size [*OT, F, K]
    with K the kernel size, by extracting frames with the given stride.

    This will pad the input so that `F = ceil(T / K)`.

    see https://github.com/pytorch/pytorch/issues/60466
    """
    *shape, length = a.shape
    n_frames = math.ceil(length / stride)
    tgt_length = (n_frames - 1) * stride + kernel_size
    a = F.pad(a, (0, tgt_length - length))
    strides = list(a.stride())
    assert strides[-1] == 1, 'data should be contiguous'
    strides = strides[:-1] + [stride, 1]
    return a.as_strided([*shape, n_frames, kernel_size], strides)


class BLSTM(nn.Module):
    """
    BiLSTM with same hidden units as input dim.

    It consists of a bidirectional LSTM paired with a linear layer which combines
    the stacked layers from the bidirectional LSTM.

    If `max_steps` is not None, input will be splitting in overlapping
    chunks and the LSTM applied separately on each chunk.

    :param dim: LSTM dimension

    :param skip: If `False`, no skip connection (input is not added to output) otherwise
    input is added to output.

    Input dimensions will match the output dimensions
    """

    def __init__(self, dim, layers=1, max_steps=None, skip=False):

        super().__init__()
        assert max_steps is None or max_steps % 4 == 0
        ## with no max_steps
        ##─LSTM: 1-1                              [200, 10, 40]             96,000
        ##─Linear: 1-2                            [200, 10, 20]
        # with max_steps of 20
        ##─LSTM: 1-1                              [20, 200, 40]             96,000
        ##─Linear: 1-2                            [20, 200, 20]

        self.max_steps = max_steps
        self.lstm = nn.LSTM(bidirectional=True, num_layers=layers, hidden_size=dim, input_size=dim)
        self.linear = nn.Linear(2 * dim, dim)
        self.skip = skip

    def forward(self, x):
        B, C, T = x.shape
        y = x
        framed = False
        # to shut the analyser up
        stride,frames,width,n_frames = 1,1,1,1

        if self.max_steps is not None and T > self.max_steps:
            width = self.max_steps
            stride = width // 2
            frames = unfold(x, width, stride)
            n_frames = frames.shape[2]
            framed = True
            x = frames.permute(0, 2, 1, 3).reshape(-1, C, width)

        x = x.permute(2, 0, 1)

        x = self.lstm(x)[0]
        x = self.linear(x)
        x = x.permute(1, 2, 0)
        if framed:
            out = []
            frames = x.reshape(B, -1, C, width)
            limit = stride // 2
            for k in range(n_frames):
                if k == 0:
                    out.append(frames[:, k, :, :-limit])
                elif k == n_frames - 1:
                    out.append(frames[:, k, :, limit:])
                else:
                    out.append(frames[:, k, :, limit:-limit])
            out = torch.cat(out, -1)
            out = out[..., :T]
            x = out
        if self.skip:
            x = x + y
        return x


if __name__ == '__main__':
    import torchinfo

    a, dim, c = 10, 20, 200
    input = torch.randn((a, dim, c))

    model = BLSTM(dim=dim, layers=10)
    output = model(input)
    torchinfo.summary(model, input.shape)

    model=BLSTM(dim=dim,layers=10,max_steps=20)
    output = model(input)
    torchinfo.summary(model, input.shape)


