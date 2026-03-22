from dataclasses import asdict
import os
import argparse
import yaml
from datetime import datetime

import torch

from sprim.configs.base import load_config, BaseConfig
from sprim.utils.io_utils import set_random_seed, init_trainer


def run(args):

    set_random_seed(42)

    # -----------------------
    # Config
    # -----------------------

    config: BaseConfig = load_config(args.config, args.log_dir, args.data_dir)

    # Create log_dir and copy config file
    os.makedirs(config.log_dir, exist_ok=True)
    yaml.dump(
        asdict(config), open(os.path.join(config.log_dir, "config_gaussians.yaml"), "w")
    )

    # TODO: provide proper scene bounds (with poses?)
    _, trainer = init_trainer(None, config)

    # training
    for _ in range(config.n_iterations + 1):
        if not trainer.training_step():
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default="./log/" + datetime.now().strftime("%m-%d-%H:%M:%S"),
    )
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    run(args)
