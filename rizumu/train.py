import os

import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import random_split, DataLoader

from openunmix.model import Separator, OpenUnmix
from rizumu.data_loader import RizumuSeparatorDataset
from rizumu.model import RizumuModel
from rizumu.pl_model import RizumuLightning, calculate_sdr


def rizumu_train(cfg: DictConfig):
    model_config = cfg["dnr_dataset"]["rizumu"]

    dataset = RizumuSeparatorDataset(root_dir=model_config["dataset_dir"],
                                     files_to_load=model_config["labels"],
                                     preprocess_dct=model_config["use_dct"],
                                     dct_scaler=model_config["quantizer"])

    # divide the dataset into train and test
    size = len(dataset)
    train_size = int(size * 0.8)
    test_size = size - train_size
    dnr_dataset_train, dnr_dataset_val = random_split(dataset=dataset, lengths=[train_size, test_size])

    dnr_train = DataLoader(dataset=dnr_dataset_train, num_workers=os.cpu_count(),
                           persistent_workers=True,batch_size=None)

    dnr_val = DataLoader(dataset=dnr_dataset_val, num_workers=os.cpu_count(),
                         persistent_workers=True,batch_size=None)

    labels = model_config["labels"]
    output_label_name = model_config["output_label"]
    mix_label_name = model_config["mix_name"]

    checkpoint_callback = ModelCheckpoint(dirpath=model_config["log_dir"])

    pl_model = RizumuLightning(labels=labels, output_label_name=output_label_name,
                               mix_name=mix_label_name)

    trainer = pl.Trainer(limit_train_batches=32, max_epochs=model_config["num_epochs"], log_every_n_steps=2,
                         callbacks=[checkpoint_callback])

    if model_config["checkpoint"]:
        # load the checkpoint path and resume training
        trainer.fit(pl_model, dnr_train, dnr_val, ckpt_path=model_config["checkpoint_path"])
    else:
        # otherwise start from scratch
        trainer.fit(pl_model, dnr_train, dnr_val)


def rizumu_train_oldschool(cfg: DictConfig):
    model_config = cfg["dnr_dataset"]["rizumu"]

    dataset = RizumuSeparatorDataset(root_dir=model_config["dataset_dir"],
                                     files_to_load=model_config["labels"],
                                     preprocess_dct=model_config["use_dct"],
                                     dct_scaler=model_config["quantizer"])

    # divide the dataset into train and test
    size = len(dataset)
    train_size = int(size * 0.8)
    test_size = size - train_size
    dnr_dataset_train, dnr_dataset_val = random_split(dataset=dataset, lengths=[train_size, test_size])

    dnr_train = DataLoader(dataset=dnr_dataset_train, num_workers=os.cpu_count(),
                           persistent_workers=True, batch_size=None)


    if True:
        model = RizumuModel(n_fft=2048)
    else:
        model = Separator(target_models={"speech": OpenUnmix(nb_bins=2049,nb_channels=1,nb_layers=7)})

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.train()
    with torch.set_grad_enabled(True):
        for epoch in range(model_config["num_epochs"]):
            for batch in dnr_train:
                mix,speech  = batch

                expected = model(mix)
                loss = torch.nn.functional.mse_loss(expected, speech)
                print("Loss:", loss.item())
                sdr = calculate_sdr(speech,expected)
                print("SDR:", sdr)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()


