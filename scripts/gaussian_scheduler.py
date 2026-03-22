import os
from tqdm import tqdm
import argparse
import torch
import subprocess
import signal

from sprim.utils.exp_utils import (
    signal_handler,
)

DATA_ROOT = os.environ["DATA_ROOT"]
PRIMITIVES_ROOT = (
    os.environ["PRIMITIVES_ROOT"] if "PRIMITIVES_ROOT" in os.environ else "primitives"
)


def main(args):

    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)

    # Read GCA configs
    with open(args.scenes) as file:
        scene_full_names = [line.rstrip() for line in file]
        print(scene_full_names)

    # Prepare the GCA
    results = {}
    for scene_full_name in tqdm(scene_full_names):
        data_dir = os.path.join(DATA_ROOT, scene_full_name)
        log_dir = os.path.join(PRIMITIVES_ROOT, scene_full_name)

        print("log_dir", log_dir)

        cmd = f"python scripts/run.py --data_dir {data_dir} --log_dir {log_dir}"

        os.makedirs(log_dir, exist_ok=True)
        with open(os.path.join(log_dir, "train_logs.txt"), "w+") as f:
            result = subprocess.run(cmd, shell=True)
        results[scene_full_name] = result.returncode

    for k, v in results.items():
        print(k, v)


if __name__ == "__main__":

    signal.signal(signal.SIGINT, signal_handler)

    parser = argparse.ArgumentParser()
    parser.add_argument("--scenes", type=str, required=True)
    parser.add_argument("--gpu", type=str, default=None)

    args = parser.parse_args()

    main(args)
