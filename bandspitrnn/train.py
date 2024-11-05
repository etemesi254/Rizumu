import os

import pytorch_lightning as pl
from omegaconf import DictConfig
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.optim import Adam
from torch.utils.data import DataLoader, random_split

from bandspitrnn.bandspit_rnn import BandSplitRNN
from bandspitrnn.data_loader import BandSpitMusicSeparatorDataset
from bandspitrnn.pl_model import BandPlModel


def bandspit_train(cfg: DictConfig):
    model_config = cfg["dnr_dataset"]["bandspit_rnn"]
    model = BandSplitRNN(**model_config["model_config"])

    # dataset loader
    dataset = BandSpitMusicSeparatorDataset(root_dir=model_config["dataset_dir"],
                                            files_to_load=model_config["labels"])

    # divide the dataset into train and test
    size = len(dataset)
    train_size = int(size * 0.8)
    test_size = size - train_size
    dnr_dataset_train, dnr_dataset_val = random_split(dataset=dataset, lengths=[train_size, test_size])

    # data loaders, divide into train and validation loader
    dnr_train = DataLoader(dataset=dnr_dataset_train,
                           num_workers=os.cpu_count(),
                           persistent_workers=True)
    dnr_val = DataLoader(dataset=dnr_dataset_val,
                         num_workers=os.cpu_count(),
                         persistent_workers=True)

    # optimizer
    params = model.parameters()
    optimizer = Adam(params, lr=1e-3)

    # take some configurations from the config file
    n_fft = model_config["model_config"]["n_fft"]
    output_label = model_config["output_label"]
    labels = model_config["labels"]
    mix_name = model_config["mix_name"]

    checkpoint_callback = ModelCheckpoint(dirpath=model_config["log_dir"])

    # make our model
    pl_model = BandPlModel(model=model, optimizer=optimizer, n_fft=n_fft,
                           labels=labels, output_label_name=output_label,
                           mix_name=mix_name)

    # setup trainer
    trainer = pl.Trainer(limit_train_batches=32, max_epochs=model_config["num_epochs"], log_every_n_steps=2,callbacks=[checkpoint_callback])

    if model_config["checkpoint"]:
        # load the checkpoint path and resume training
        trainer.fit(pl_model, dnr_train, dnr_val, ckpt_path=model_config["checkpoint_path"])
    else:
        # otherwise start from scratch
        trainer.fit(pl_model, dnr_train, dnr_val)
