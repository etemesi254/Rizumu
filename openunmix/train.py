import os

import pytorch_lightning as pl
from omegaconf import DictConfig
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.optim import Adam
from torch.utils.data import random_split, DataLoader

from openunmix.data_loader import OpenUnmixMusicSeparatorDataset
from openunmix.model import Separator, OpenUnmix
from openunmix.pl_model import OpenUnmixLightning


def openunmix_train(cfg: DictConfig):
    model_config = cfg["dnr_dataset"]["openunmix"]

    dataset = OpenUnmixMusicSeparatorDataset(root_dir=model_config["dataset_dir"],
                                             files_to_load=model_config["labels"], )

    # divide the dataset into train and test
    size = len(dataset)
    train_size = int(size * 0.8)
    test_size = size - train_size
    dnr_dataset_train, dnr_dataset_val = random_split(dataset=dataset, lengths=[train_size, test_size])

    channels = model_config["dataset_config"]["nb_channels"]
    model = Separator(target_models={
        model_config["output_label"]: OpenUnmix(nb_channels=channels,
                                                nb_bins=model_config["dataset_config"]["nb_bins"])},
        nb_channels=channels)
    dnr_train = DataLoader(dataset=dnr_dataset_train, num_workers=os.cpu_count(),
                           persistent_workers=True)

    dnr_val = DataLoader(dataset=dnr_dataset_val, num_workers=os.cpu_count(),
                         persistent_workers=True)
    params = model.parameters()
    optimizer = Adam(params, lr=1e-3)
    labels = model_config["labels"]
    output_label_name = model_config["output_label"]
    mix_label_name = model_config["mix_name"]

    checkpoint_callback = ModelCheckpoint(dirpath=model_config["log_dir"])


    pl_model = OpenUnmixLightning(model=model, optimizer=optimizer, labels=labels, output_label_name=output_label_name,
                                  mix_name=mix_label_name)

    trainer = pl.Trainer(limit_train_batches=32, max_epochs=model_config["num_epochs"], log_every_n_steps=2,callbacks=[checkpoint_callback])

    if model_config["checkpoint"]:
        # load the checkpoint path and resume training
        trainer.fit(pl_model, dnr_train, dnr_val, ckpt_path=model_config["checkpoint_path"])
    else:
        # otherwise start from scratch
        trainer.fit(pl_model, dnr_train, dnr_val)

