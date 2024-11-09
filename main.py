from argparse import ArgumentParser

from hydra import compose, initialize

from bandspitrnn.train import bandspit_train
from openunmix.train import openunmix_train


def my_app() -> None:
    initialize(version_base=None, config_path="./config", job_name="rizumu")

    parser = ArgumentParser()

    parser.add_argument("--train-bandspitrnn", action="store_true",
                        help="Train a bandspitrnn model with pre-configured outputs")
    parser.add_argument("--train-openunmix", action="store_true", help="Train an openunmix model with pre-configs")
    args = parser.parse_args()
    if args.train_bandspitrnn:
        cfg = compose(config_name="config")
        bandspit_train(cfg)
    elif args.train_openunmix:
        cfg = compose(config_name="config")
        openunmix_train(cfg)
    else:
        parser.print_help()

    pass


if __name__ == "__main__":
    my_app()
