import soundfile as sf
import torch
import torchaudio
from torch import Tensor

from openunmix.model import Separator, OpenUnmix
from openunmix.pl_model import OpenUnmixLightning


def convert_to_audio(tensor: Tensor, sample_rate=44100):
    d = tensor.cpu().detach().numpy()

    sf.write(file="out.wav", samplerate=sample_rate, data=d, subtype='PCM_24')


def separate_audio(file: str):
    model = Separator(target_models={
        "speech": OpenUnmix(nb_channels=1,nb_bins=2049)})

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    pl_model = OpenUnmixLightning.load_from_checkpoint("/Users/etemesi/PycharmProjects/Rizumu/openunmix_logs/epoch=51-step=1664-v1.ckpt",model=model, optimizer=optimizer, labels=["mix", "speech"], output_label_name="speech",
                               mix_name="mix")

    # disable randomness, dropout, etc...
    pl_model.eval()

    file,sr = torchaudio.load("/Users/etemesi/PycharmProjects/Spite/data/dnr_v2/91421/mix.wav")

    file = file.reshape((1,file.shape[0],file.shape[1]))

    # predict with the model
    y_hat = model(file)
    output = y_hat.squeeze().detach().cpu().numpy()
    sf.write(file="out_cae.wav", samplerate=44100, data=output, subtype='PCM_24')

    print(output.shape)



if __name__ == "__main__":
    separate_audio(file="")
