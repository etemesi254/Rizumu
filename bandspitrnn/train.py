import os
import pprint

import torch
from omegaconf import DictConfig
from torch.optim import Adam
from torch.utils.data import DataLoader, random_split

from bandspitrnn.bandspit_rnn import BandSplitRNN
from bandspitrnn.data_loader import MusicSeparatorDataset


from bandspitrnn.pl_model import BandPlModel
import pytorch_lightning as pl

def complex_mse_loss(output, target):
    return (0.5 * (output - target) ** 2).mean(dtype=torch.complex64)


def calculate_loss(model_output: torch.Tensor, target: torch.Tensor, n_fft: int) -> torch.Tensor:
    a = torch.istft(model_output.squeeze(), window=torch.hann_window(window_length=n_fft),
                    n_fft=n_fft)
    b = torch.istft(target.squeeze(), window=torch.hann_window(window_length=n_fft), n_fft=n_fft)
    loss = torch.nn.functional.mse_loss(a, b)
    return loss


def bandspit_train(cfg: DictConfig):
    pprint.pprint(dict(cfg["dnr_dataset"]["bandspit_rnn"]))
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
                        num_workers=10,persistent_workers=True
                          )
    dnr_val = DataLoader(dataset=dnr_dataset_val,num_workers=10,persistent_workers=True
                         )

    # optimizer
    params = model.parameters()
    optimizer = Adam(params, lr=1e-3)

    n_fft = model_config["model_config"]["n_fft"]

    pl_model = BandPlModel(model=model,optimizer=optimizer,n_fft=n_fft)

    trainer = pl.Trainer(limit_train_batches=32,max_epochs=1,log_every_n_steps=32)
    trainer.fit(pl_model, dnr_train, dnr_val)
    model.train()
    num_epochs = model_config["num_epochs"]
    # for epoch in range(num_epochs):
    #     for batch in dnr_train:
    #         mix, speech, vocals, sfx = batch
    #         model_output = model.forward(mix)
    #         loss = calculate_loss(model_output, speech, n_fft)
    #         # print(loss)
    #         print("Loss: {}".format(loss.item()))
    #         optimizer.step()
    #         optimizer.zero_grad()
    #     break
