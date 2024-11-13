import torch
import torchaudio

audio, sr = torchaudio.load("/Users/etemesi/PycharmProjects/Spite/data/dnr_v2/90449/mix.wav")

c = torch.stft(audio, 2048, return_complex=True, window=torch.hann_window(2048))

d = torch.view_as_real(c)
e,f = torch.split(d,1,dim=-1)

print(e.max())
print(e.min())

print(f.max())
print(f.min())


g = d.permute(3, 0, 2, 1)
e,f = torch.split(g,1,dim=0)
print("\n\n")
print(e.max())
print(e.min())

print(f.max())
print(f.min())