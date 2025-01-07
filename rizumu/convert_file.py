import logging
import time
from os import PathLike
from typing import BinaryIO

import pytorch_lightning
import torch
import torchaudio

from rizumu.pl_model import RizumuLightning


def torch_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


@torch.no_grad()
def convert_file(input_file: BinaryIO | str | PathLike,
                 output_file: BinaryIO | str | PathLike,
                 model: pytorch_lightning.LightningModule,

                 seconds_split: int = 10,
                 device=torch_device(),
                 # checkpoint: str = "/Users/etemesi/PycharmProjects/Rizumu/chekpoints/rizumu_logs/epoch=57-step=64422.ckpt"
                 ):
    audio, sr = torchaudio.load(input_file)
    if audio.shape[0] == 2:
        # downsample
        logging.info("Down-sampling channels")
        audio = torch.mean(audio, 0, keepdim=True)
        logging.info("Successfully down-sampled to mono audio")
    audio.to(device)
    # to deal with long audios split the tensor into 30 seconds segments
    segments = audio.size(1)
    if audio.size(1) > (sr * seconds_split):
        segments = (sr * seconds_split)
    logging.info("Splitting audio into %d segments", audio.size(1) // segments)
    splits = torch.split(audio, segments, dim=1)
    model = model.to(device=device)
    # model = RizumuLightning.load_from_checkpoint(checkpoint).to(device)
    results = []
    root_start = time.time()
    for split in splits:
        start = time.time()
        output = model(split.to(device))
        end = time.time()
        logging.info(f"Took {end - start:.2f} seconds to execute a segment")
        results.append(output)
    new_output = torch.cat(results, dim=1)
    root_end = time.time()

    logging.info(f"Finished in {root_end - root_start:.2f} seconds\n")

    logging.info(f"Saving to {output_file}")
    torchaudio.save(output_file, new_output.detach().cpu(), sample_rate=sr, encoding="PCM_F", format="wav")
    logging.info(f"Saved to {output_file}")


if __name__ == "__main__":
    model = RizumuLightning.load_from_checkpoint(
        "/Users/etemesi/PycharmProjects/Rizumu/chekpoints/rizumu_logs/epoch=57-step=64422.ckpt")
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s - %(levelname)s - %(message)s")
    convert_file("/Users/etemesi/Datasets/312/mix.wav",
                 "./hello.wav",
                 model,
                 seconds_split=10,

                 )
