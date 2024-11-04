import hydra
from omegaconf import DictConfig

import bandspitrnn


@hydra.main(version_base=None, config_path="./config", config_name="config")
def my_app(cfg: DictConfig) -> None:
    bandspitrnn.bandspit_train(cfg)
    pass


if __name__ == "__main__":
    my_app()
