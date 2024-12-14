import logging
import os

import numpy as np
import torch
from torch.utils.data import random_split, DataLoader
from tqdm import tqdm


from data_loader import DemucsSeparator
from demucs import Demucs

def calculate_loss(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Calculate the  MSE loss between two tensors
    :param a: Input tensor
    :param b: Another tensor
    :return: The MSE loss
    """
    loss = torch.nn.functional.mse_loss(a, b)
    return loss


def calculate_sdr(target_tensor, output_tensor) -> float:


    target_tensor = target_tensor.detach().cpu().numpy()
    output_tensor = output_tensor.detach().cpu().numpy()

    target_power = np.sum(target_tensor ** 2)
    noise_power = np.sum((target_tensor - output_tensor) ** 2)

    if noise_power == 0:
        return float('inf')  # Handle the case where the noise power is zero to prevent division by zero

    sdr = 10 * np.log10(target_power / noise_power)
    return sdr




def demucs_train_oldschool():
    dataset = DemucsSeparator(root_dir="/Users/etemesi/Datasets/dnr_v2/cv",
                              files_to_load=["mix", "speech"],
                              preprocess_dct=False,
                              dct_scaler=10)

    # divide the dataset into train and test
    size = len(dataset)
    train_size = int(size * 0.8)
    test_size = size - train_size
    dnr_dataset_train, dnr_dataset_val = random_split(dataset=dataset, lengths=[train_size, test_size])

    dnr_train = DataLoader(dataset=dnr_dataset_train, num_workers=os.cpu_count(),
                           persistent_workers=True, batch_size=None)

    model = Demucs(sample_rate=44100,resample=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device, non_blocking=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    model.train()
    with torch.autograd.set_detect_anomaly(True):
        for epoch in range(50):
            sum_sdr = 0
            sum_loss = 0
            iteration = 0

            pbar = tqdm(total=len(dnr_train))

            for batch in dnr_train:
                pbar.update()
                pbar.set_description(f"Epoch {epoch + 1}/50")
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

                sum_sdr += sdr
                iteration += 1
                new_loss = loss * (100 - sdr)
                sum_loss += new_loss

                avg_loss = sum_loss / iteration

                pbar.set_postfix(
                    {"avg_sdr": sum_sdr / iteration, "sdr": sdr, "loss": new_loss.item(), "avg_loss": avg_loss.item()})

                new_loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            pbar.close()

if __name__ == '__main__':
    demucs_train_oldschool()