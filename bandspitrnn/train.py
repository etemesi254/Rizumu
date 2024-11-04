import os
import pprint

from omegaconf import DictConfig
from torch.utils.data import DataLoader

from bandspitrnn.bandspit_rnn import BandSplitRNN
from bandspitrnn.data_loader import MusicSeparatorDataset


def bandspit_train(cfg: DictConfig):
    pprint.pprint(dict(cfg["dnr_dataset"]["bandspit_rnn"]))
    model_config = cfg["dnr_dataset"]["bandspit_rnn"]
    model = BandSplitRNN(**model_config["model_config"])
    dataset = MusicSeparatorDataset(root_dir=model_config["dataset_dir"],
                                    files_to_load=model_config["labels"])

    de = DataLoader(dataset=dataset, batch_size=model_config["batch_size"], shuffle=model_config["shuffle"],
                    num_workers=os.cpu_count())
