import torch
import torch.nn as nn
import torch.nn.functional as F


def round_to_nearest(x, n):
    return ((x + (n - 1)) // n) * n


def _decoder_block(in_channels, out_channels):
    """Create decoder block with convolution, normalization, and activation"""
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(inplace=True)
    )


def _encoder_block(in_channels, out_channels):
    """Create encoder block with convolution, normalization, and activation"""
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2, stride=2)
    )


class BLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, layers=1, skip=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lstm = nn.LSTM(bidirectional=True, num_layers=layers, hidden_size=hidden_size, input_size=input_size,
                            batch_first=True)
        self.linear = nn.Linear(2 * hidden_size, hidden_size)
        self.skip = skip

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.lstm(x)[0]
        x = F.relu(self.linear(x))
        # remove the squeeze

        return x


def pad_to_multiple_of_16(tensor: torch.Tensor):
    """
    Pad a 4D PyTorch tensor so that the last two dimensions are divisible by 16.

    Parameters:
    -----------
    tensor : torch.Tensor
        4-dimensional input tensor

    Returns:
    --------
    torch.Tensor
        Padded tensor with last two dimensions padded to be divisible by 16
    """
    # Calculate padding needed for third dimension (height)
    height_pad = (16 - (tensor.shape[-2] % 16)) % 16

    # Calculate padding needed for fourth dimension (width)
    width_pad = (16 - (tensor.shape[-1] % 16)) % 16

    # Create padding configuration
    # PyTorch's pad function uses the reverse order of dimensions
    # and takes a 1D list of padding values
    padding = (0, width_pad, 0, height_pad)

    # Use torch.nn.functional.pad to add padding
    padded_tensor = torch.nn.functional.pad(tensor, padding)

    return padded_tensor


class SourceSeparationModel(nn.Module):
    def __init__(self, input_channels=1, output_channels=1, hidden_size=512):
        """
        U-Net inspired architecture for music source separation with Bi-LSTM bridge

        Args:
            input_channels (int): Number of input audio channels (typically 1 for mono)
            output_channels (int): Number of separated source channels
            hidden_size (int): Hidden size for Bi-LSTM layers
        """
        super(SourceSeparationModel, self).__init__()

        # Encoder (Downsampling) path
        self.enc1 = _encoder_block(input_channels, 64)
        self.enc2 = _encoder_block(64, 128)
        self.enc3 = _encoder_block(128, 256)
        self.enc4 = _encoder_block(256, 512)

        # Bi-LSTM Bridge
        self.bridge_reshape = nn.Conv2d(512, 512, kernel_size=1)
        self.bilstm_bridge = BLSTM(input_size=512, hidden_size=hidden_size, layers=2)
        self.bridge_reconstruct = nn.Conv2d(hidden_size, 512, kernel_size=1)

        # Decoder (Upsampling) path
        self.upconv4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec4 = _decoder_block(512, 256)

        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3 = _decoder_block(256, 128)

        self.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = _decoder_block(128, 64)

        self.upconv1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec1 = _decoder_block(64, 32)

        # Final convolution to get output channels
        self.final_conv = nn.Conv2d(32, output_channels, kernel_size=1)

    def forward(self, x):
        """
        Forward pass through the source separation network

        Args:
            x (torch.Tensor): Input spectrogram tensor
                               Shape: [batch_size, channels, height, width]

        Returns:
            torch.Tensor: Separated source spectrogram
        """
        x_orig = x
        orig_shape = x.shape

        # Pad the network since the downsample
        # network will reduce it and will lose the odd network
        # doing it to 16 because we encode 4 layers deep and each layer
        # divides it by half.
        x = pad_to_multiple_of_16(x)


        # Encoder path
        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)

        # Bi-LSTM Bridge

        # Reshape and prepare for LSTM
        bridge_input = self.bridge_reshape(enc4)
        batch_size, channels, height, width = bridge_input.shape
        lstm_input = bridge_input.permute(0, 2, 3, 1).reshape(batch_size, height * width, channels)
        lstm_out = self.bilstm_bridge(lstm_input)
        lstm_out = lstm_out.reshape(batch_size, height, width, -1).permute(0, 3, 1, 2)
        bridge = self.bridge_reconstruct(lstm_out)

        # Decoder path with skip connections
        up4 = self.upconv4(bridge)
        up4 = torch.cat([up4, enc3], dim=1)
        dec4 = self.dec4(up4)

        up3 = self.upconv3(dec4)
        up3 = torch.cat([up3, enc2], dim=1)
        dec3 = self.dec3(up3)
        up2 = self.upconv2(dec3)
        up2 = torch.cat([up2, enc1], dim=1)
        dec2 = self.dec2(up2)

        up1 = self.upconv1(dec2)

        x_mask = self.final_conv(up1)

        # remove the padding by slicing
        x_mask = x_mask[:, :, :orig_shape[-2], :orig_shape[-1]]
        # mask the original piece.
        return x_orig * x_mask


# Rest of the previous implementation remains the same (spectral_loss, prepare_data, train_source_separation)
def spectral_loss(pred, target):
    """
    Custom loss function combining magnitude and phase preservation

    Args:
        pred (torch.Tensor): Predicted spectrogram
        target (torch.Tensor): Ground truth spectrogram

    Returns:
        torch.Tensor: Computed loss
    """
    # Magnitude loss (L1)
    mag_loss = F.l1_loss(torch.abs(pred), torch.abs(target))

    # Phase preservation loss
    phase_loss = F.mse_loss(torch.angle(pred), torch.angle(target))

    return mag_loss + 0.1 * phase_loss


def prepare_data(mix_spectrogram, target_spectrogram):
    """
    Prepare input data for the model

    Args:
        mix_spectrogram (torch.Tensor): Mixed music spectrogram
        target_spectrogram (torch.Tensor): Target source spectrogram

    Returns:
        tuple: Prepared input and target tensors
    """
    # Normalize spectrograms
    mix_spectrogram = (mix_spectrogram - mix_spectrogram.min()) / (mix_spectrogram.max() - mix_spectrogram.min())
    target_spectrogram = (target_spectrogram - target_spectrogram.min()) / (
            target_spectrogram.max() - target_spectrogram.min())

    return mix_spectrogram, target_spectrogram


# Example usage
def train_source_separation():
    # Initialize model
    model = SourceSeparationModel(input_channels=1, output_channels=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Simulated training loop (replace with actual data loading)
    for epoch in range(100):
        # Simulate input spectrograms (you'd use real data)
        mix_spec = torch.randn(1, 1, 256, 256)
        target_spec = torch.randn(1, 1, 256, 256)

        # Prepare data
        mix_spec, target_spec = prepare_data(mix_spec, target_spec)

        # Forward pass
        separated_spec = model(mix_spec)

        # Compute loss
        loss = spectral_loss(separated_spec, target_spec)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch + 1}, Loss: {loss.item()}")


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
