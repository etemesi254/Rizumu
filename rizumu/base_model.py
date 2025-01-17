import torch
import torch.nn as nn
import torch.nn.functional as F


def _decoder_block(in_channels, out_channels):
    """Create decoder block with convolution, normalization, and activation"""
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(inplace=True)
    )


def _encoder_block(in_channels, out_channels):
    """Create encoder block with convolution and activation"""
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2, stride=2)
    )


class SelfAttention(nn.Module):
    """Self-attention mechanism to enhance feature representation"""

    def __init__(self, channels):
        super().__init__()
        self.query = nn.Conv2d(channels, channels // 8, kernel_size=1)
        self.key = nn.Conv2d(channels, channels // 8, kernel_size=1)
        self.value = nn.Conv2d(channels, channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch, C, H, W = x.size()

        query = self.query(x).view(batch, -1, H * W).permute(0, 2, 1)
        key = self.key(x).view(batch, -1, H * W)
        energy = torch.bmm(query, key)
        attention = F.softmax(energy, dim=-1)

        value = self.value(x).view(batch, -1, H * W)
        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(batch, C, H, W)

        return self.gamma * out + x


class BLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, layers=1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv1 = nn.Conv2d(input_size, input_size, kernel_size=1)

        self.lstm = nn.LSTM(bidirectional=True,
                            num_layers=layers,
                            hidden_size=hidden_size,
                            input_size=input_size,
                            batch_first=True)

        self.linear = nn.Linear(2 * hidden_size, hidden_size)

        self.conv2 = nn.Conv2d(hidden_size, output_size, kernel_size=1)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        # reshape
        batch_size, channels, height, width = x.shape
        x = x.permute(0, 2, 3, 1).reshape(batch_size, height * width, channels)

        x = self.lstm(x)[0]
        x = self.linear(x)

        x = x.reshape(batch_size, height, width, -1).permute(0, 3, 1, 2).contiguous()

        x = self.conv2(x)
        x = self.relu(x)

        return x


def pad_to_multiple_of_n(tensor: torch.Tensor, n: int):
    """
    Pad a 4D PyTorch tensor so that the last two dimensions are divisible by n.

    Parameters:
    -----------
    tensor : torch.Tensor
        4-dimensional input tensor

    Returns:
    --------
    torch.Tensor
        Padded tensor with last two dimensions padded to be divisible by n
    """
    # Calculate padding needed for third dimension (height)
    height_pad = (n - (tensor.shape[-2] % n)) % n

    # Calculate padding needed for fourth dimension (width)
    width_pad = (n - (tensor.shape[-1] % n)) % n

    # Create padding configuration
    # PyTorch's pad function uses the reverse order of dimensions
    # and takes a 1D list of padding values
    padding = (0, width_pad, 0, height_pad)

    # Use torch.nn.functional.pad to add padding
    padded_tensor = torch.nn.functional.pad(tensor, padding)

    return padded_tensor


class SourceEncoder(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.block = _encoder_block(input_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block(x)
        return x


class SourceDecoder(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.conv = nn.ConvTranspose2d(input_size, output_size, kernel_size=2, stride=2)
        self.block = _decoder_block(input_size, output_size)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        up = self.conv(x)
        x = torch.cat([up, y], dim=1)

        x = self.block(x)
        return x


class SourceSeparationModel(nn.Module):
    def __init__(self, input_channels=1, output_channels=1, lstm_layers=1, hidden_size=512, depth=4):

        super(SourceSeparationModel, self).__init__()

        self.encoders = nn.ModuleList()
        self.depth = depth

        # build encoder
        #
        # n -> 1,64,128,256,512..2**n -> based on depth.
        # but first one must start at input_channels, so that is a special
        # case
        for i in range(depth):
            input_size = 2 ** (i + 5)
            output_size = 2 ** (i + 6)
            if i == 0:
                self.encoders.append(SourceEncoder(input_size=input_channels, output_size=output_size))
            else:
                self.encoders.append(SourceEncoder(input_size=input_size, output_size=output_size))

        # Bi-LSTM Bridge
        #
        # its input from the encoder can be calculated as where the encoder stops
        input_depth = 2 ** (depth + 5)
        self.bottleneck = nn.Sequential(SelfAttention(input_depth),
                                        BLSTM(input_size=input_depth,
                                              hidden_size=hidden_size,
                                              output_size=input_depth,
                                              layers=lstm_layers))

        # Decoders, logic is same as encoders but the data is reversed
        self.decoders = nn.ModuleList()

        for i in range(depth - 1):
            # the reduction networks create smaller outputs from
            # larger inputs
            input_size = 2 ** ((5 + depth) - i)
            output_size = 2 ** ((4 + depth) - i)
            decoder = SourceDecoder(input_size, output_size)
            self.decoders.append(decoder)

        # last layer does not do skip connections and hence why it is separate
        self.final = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.Conv2d(32, output_channels, kernel_size=1),
        )

    def forward(self, x):
        """
        Forward pass through the source separation network

        Args:
            x (torch.Tensor): Input spectrogram tensor
                               Shape: [batch_size, channels, height, width]

        Returns:
            torch.Tensor: Separated source spectrogram
        """
        orig_shape = x.shape

        # Pad the network since the downsample
        # network will reduce it and will lose the odd network
        # doing it to 2**depth because we encode depth layers deep and each layer
        # divides it by half.
        x = pad_to_multiple_of_n(x, 2 ** self.depth)

        encode_results = []

        for encoder in self.encoders:
            x = encoder(x)
            encode_results.append(x)

        # Bi-LSTM Bridge
        x = self.bottleneck(encode_results.pop(-1))

        # decoder
        for decoder in self.decoders:
            x = decoder(x, encode_results.pop(-1))

        x = self.final(x)

        # remove the padding by slicing
        x = x[:, :, :orig_shape[-2], :orig_shape[-1]]

        return x


if __name__ == "__main__":
    import torchinfo
    import torchaudio

    audio, sr = torchaudio.load("/Users/etemesi/Datasets/312/mix.wav")
    stft = torch.stft(audio, n_fft=2048, return_complex=True, window=torch.hann_window(2048))
    stft = stft.unsqueeze(0).to("mps")

    model = SourceSeparationModel(input_channels=1, output_channels=1, depth=4, hidden_size=1024, lstm_layers=1).to(
        "mps")
    s = torch.abs(stft).to("mps")
    c = torchinfo.summary(model, input_data=s, device="mps", depth=4)
