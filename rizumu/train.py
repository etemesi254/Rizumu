import logging
import os

import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import random_split, DataLoader
from tqdm import tqdm

from rizumu.data_loader import RizumuSeparatorDataset
from rizumu.model import RizumuModel
from rizumu.pl_model import RizumuLightning, calculate_sdr


def rizumu_train(cfg: DictConfig):
    model_config = cfg["dnr_dataset"]["rizumu"]
    dnr_dataset_train = RizumuSeparatorDataset(root_dir=model_config["training_set"],
                                     files_to_load=model_config["labels"],
                                     preprocess_dct=model_config["use_dct"],
                                     dct_scaler=model_config["quantizer"])
    # divide the dataset into train and test
    dnr_dataset_val = RizumuSeparatorDataset(root_dir=model_config["testing_set"],
                                               files_to_load=model_config["labels"],
                                               preprocess_dct=model_config["use_dct"],
                                               dct_scaler=model_config["quantizer"])
    dnr_train = DataLoader(dataset=dnr_dataset_train, num_workers=os.cpu_count(),
                           persistent_workers=True, batch_size=None)

    dnr_val = DataLoader(dataset=dnr_dataset_val, num_workers=os.cpu_count(),
                         persistent_workers=True, batch_size=None)
    labels = model_config["labels"]
    output_label_name = model_config["output_label"]
    mix_label_name = model_config["mix_name"]
    num_splits = model_config["num_splits"]
    hidden_size = model_config["hidden_size"]
    lstm_layers = model_config["lstm_layers"]
    checkpoint_callback = ModelCheckpoint(dirpath=model_config["log_dir"])
    pl_model = RizumuLightning(labels=labels,
                               output_label_name=output_label_name,
                               num_splits=num_splits,
                               hidden_size=hidden_size,
                               mix_name=mix_label_name, n_fft=2048,
                               depth=4,lstm_layers=lstm_layers)

    # mps accelerator generates,nan seems like a pytorch issue
    # see https://discuss.pytorch.org/t/device-mps-is-producing-nan-weights-in-nn-embedding/159067
    trainer = pl.Trainer(max_epochs=model_config["num_epochs"], log_every_n_steps=2,
                         callbacks=[checkpoint_callback])
    if model_config["checkpoint"]:
        # load the checkpoint path and resume training
        trainer.fit(pl_model, dnr_train, dnr_val, ckpt_path=model_config["checkpoint_path"])
    else:
        # otherwise start from scratch
        trainer.fit(pl_model, dnr_train, dnr_val)


def rizumu_train_oldschool(cfg: DictConfig):
    model_config = cfg["dnr_dataset"]["rizumu"]

    dnr_dataset_train = RizumuSeparatorDataset(root_dir=model_config["training_set"],
                                               files_to_load=model_config["labels"],
                                               preprocess_dct=model_config["use_dct"],
                                               dct_scaler=model_config["quantizer"])

    # divide the dataset into train and test
    dnr_dataset_val = RizumuSeparatorDataset(root_dir=model_config["testing_set"],
                                             files_to_load=model_config["labels"],
                                             preprocess_dct=model_config["use_dct"],
                                             dct_scaler=model_config["quantizer"])


    dnr_train = DataLoader(dataset=dnr_dataset_train, num_workers=os.cpu_count(),
                           persistent_workers=True, batch_size=None)

    model = RizumuModel(n_fft=2048)

    device = torch.device("cuda" if torch.cuda.is_available() else "mps")
    model = model.to(device, non_blocking=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    logging.basicConfig(level=logging.DEBUG)

    model.train()
    with torch.autograd.set_detect_anomaly(True):
        for epoch in range(model_config["num_epochs"]):
            sum_sdr = 0
            sum_loss = 0
            iteration = 0

            pbar = tqdm(total=len(dnr_train))

            for batch in dnr_train:
                pbar.update()
                pbar.set_description(f"Epoch {epoch + 1}/{model_config['num_epochs']}")
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
