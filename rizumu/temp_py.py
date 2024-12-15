import torch
import torch.nn as nn


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


class BLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, layers=1, skip=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv1 = nn.Conv2d(input_size, input_size, kernel_size=1)

        self.lstm = nn.LSTM(bidirectional=True, num_layers=layers, hidden_size=hidden_size, input_size=input_size,
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

        x = x.reshape(batch_size, height, width, -1).permute(0, 3, 1, 2)

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
    def __init__(self, input_channels=1, output_channels=1, hidden_size=512, depth=4):
        """
        U-Net inspired architecture for music source separation with Bi-LSTM bridge

        Args:
            input_channels (int): Number of input audio channels (typically 1 for mono)
            output_channels (int): Number of separated source channels
            hidden_size (int): Hidden size for Bi-LSTM layers
        """
        super(SourceSeparationModel, self).__init__()

        # Encoder (Downsampling) path
        # self.enc1 = SourceEncoder(input_channels, 64)
        # self.enc2 = SourceEncoder(64, 128)
        # self.enc3 = SourceEncoder(128, 256)
        # self.enc4 = SourceEncoder(256, 512)
        self.encoders = nn.ModuleList()
        self.depth = depth
        for i in range(depth):
            n = 2 ** (i + 5)
            n_1 = 2 ** (i + 6)
            if i == 0:
                self.encoders.append(SourceEncoder(input_size=input_channels, output_size=n_1))
            else:
                self.encoders.append(SourceEncoder(input_size=n, output_size=n_1))

        # Bi-LSTM Bridge
        self.bottleneck = BLSTM(input_size=512,hidden_size=hidden_size, output_size=512)


        self.decoders = nn.ModuleList()
        for i in range(depth - 1):
            n_1 = 2 ** (i + 5 + depth)
            n = 2 ** (i + 4 + depth)
            dec4 = SourceDecoder(n_1, n)
            self.decoders.append(dec4)

        self.final = nn.Sequential(nn.Conv2d(64, output_channels, kernel_size=1), nn.ReLU())

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
        # doing it to 16 because we encode 4 layers deep and each layer
        # divides it by half.
        x = pad_to_multiple_of_n(x, 2**self.depth)

        encode_results = []
        for encoder in self.encoders:
            x = encoder(x)
            encode_results.append(x)

        # Bi-LSTM Bridge
        x = self.bottleneck(encode_results[-1])


        # decoder
        for decoder in self.decoders:
            x = decoder(x, encode_results.pop(-1))


        x = self.final(x)

        # remove the padding by slicing
        x = x[:, :, :orig_shape[-2], :orig_shape[-1]]
        # mask the original piece.
        return x


if __name__ == "__main__":
    import torchinfo
    import torchaudio

    audio, sr = torchaudio.load("/Users/etemesi/Datasets/dnr_v2/cv/258/mix.wav")
    stft = torch.stft(audio, n_fft=2048, return_complex=True, window=torch.hann_window(2048))
    stft = stft.unsqueeze(0).to("mps")

    model = SourceSeparationModel(input_channels=1, output_channels=1).to("mps")
    s = torch.abs(stft).to("mps")
    out = model(s)
    print(s.shape)
    c = torchinfo.summary(model, input_data=out, device="mps")
    print(out.shape)
