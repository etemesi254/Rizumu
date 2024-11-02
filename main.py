from omegaconf import DictConfig, OmegaConf
import hydra
import pprint


@hydra.main(version_base=None, config_path="./config", config_name="config")
def my_app(cfg: DictConfig) -> None:
    pprint.pprint(dict(cfg["dnr_dataset"]["bandspit_rnn"]))

if __name__ == "__main__":
    my_app()