import os

import torch
from torch.utils.data import DataLoader

from rizumu.data_loader import RizumuSeparatorDataset
from rizumu.model import RizumuModel
from rizumu.pl_model import RizumuLightning, calculate_sdr


def load_and_iter(base_dir: str, checkpoint: str):
    model = RizumuLightning.load_from_checkpoint(checkpoint)

    model.eval()

    total_sdr = 0
    total_loss = 0

    dnr_dataset_train = RizumuSeparatorDataset(root_dir=base_dir,
                                               files_to_load=["mix", "speech"],
                                               preprocess_dct=True,
                                               dct_scaler=2000)

    dnr_train = DataLoader(dataset=dnr_dataset_train, num_workers=os.cpu_count(),
                           persistent_workers=True, batch_size=None)

    device = "mps"
    model = model.to(device, non_blocking=False)
    i=0
    with torch.no_grad():
        for batch in dnr_train:
            mix, speech = batch
            mix = mix.to(device, non_blocking=False)
            speech = speech.to(device, non_blocking=False)
            expected = model(mix)

            expected = expected.to(device, non_blocking=False)
            loss = torch.nn.functional.mse_loss(expected.squeeze(), speech.squeeze())
            if torch.isnan(loss):
                print("NaN loss")
                raise Exception()
            sdr = calculate_sdr(expected, speech)
            total_sdr += sdr
            total_loss += loss

            print("Total SDR:", total_sdr)
            print("Total Loss:", total_loss.item())
            print(f"{i}")
            i+=1
    print("AVG SDR", total_sdr/i)
    print("AVG Loss", total_loss.item()/i)

if __name__ == '__main__':
    load_and_iter('/Volumes/Untitled/DNR/dnr/dnr/dnr/tt',
                  checkpoint='/Users/etemesi/PycharmProjects/Rizumu/chekpoints/rizumu_logs/epoch=26-step=82566.ckpt')
