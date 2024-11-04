import os

import pytorch_lightning as pl
from omegaconf import DictConfig
from torch.optim import Adam
from torch.utils.data import DataLoader, random_split

from bandspitrnn.bandspit_rnn import BandSplitRNN
from bandspitrnn.data_loader import MusicSeparatorDataset
from bandspitrnn.pl_model import BandPlModel


def bandspit_train(cfg: DictConfig):
    model_config = cfg["dnr_dataset"]["bandspit_rnn"]
    model = BandSplitRNN(**model_config["model_config"])

    # dataset loader
    dataset = MusicSeparatorDataset(root_dir=model_config["dataset_dir"],
                                    files_to_load=model_config["labels"])
    # divide the dataset into two options
    size = len(dataset)
    train_size = int(size * 0.8)
    test_size = size - train_size

    dnr_dataset_train, dnr_dataset_val = random_split(dataset=dataset, lengths=[train_size, test_size])

    # data loaders
    dnr_train = DataLoader(dataset=dnr_dataset_train,
                           num_workers=os.cpu_count(), persistent_workers=True
                           )
    dnr_val = DataLoader(dataset=dnr_dataset_val, num_workers=os.cpu_count(), persistent_workers=True
                         )

    # optimizer
    params = model.parameters()
    optimizer = Adam(params, lr=1e-3)

    n_fft = model_config["model_config"]["n_fft"]

    pl_model = BandPlModel(model=model, optimizer=optimizer, n_fft=n_fft)

    trainer = pl.Trainer(limit_train_batches=32, max_epochs=model_config["num_epochs"], log_every_n_steps=2)

    if model_config["checkpoint"]:
        # load the checkpoint path and resume training
        trainer.fit(pl_model, dnr_train, dnr_val, ckpt_path=model_config["checkpoint_path"])
    else:
        # otherwise start from scratch
        trainer.fit(pl_model, dnr_train, dnr_val)
