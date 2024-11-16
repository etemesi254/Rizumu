import torch

n_fft = 128
audio = torch.randn((1, n_fft))

c = torch.stft(audio, n_fft, return_complex=True, window=torch.hann_window(n_fft))

d = torch.view_as_real(c)
e, f = torch.split(d, 1, dim=-1)

print(e.max())
print(e.min())

print(f.max())
print(f.min())

g = d.permute(3, 0, 2, 1)
e, f = torch.split(g, 1, dim=0)
print("\n\n")
print(e.max())
print(e.min())

print(f.max())
print(f.min())
