import os
import yaml
import glob
import argparse
import subprocess
import signal
import shutil
import time
from tqdm import tqdm

from sprim.utils.exp_utils import (
    ExpPrimitiveEntry,
    EXP_PRIMITIVES_ROOT,
    prepare_gca_config,
    read_prim_and_gca,
    signal_handler,
    resolve_log_dir,
    check_files,
)


def main(args):

    gca_configs, prim_entries = read_prim_and_gca(args.gca_configs, args.prim_list)

    check_files(gca_configs)

    results = {}

    ENTRIES = [
        "scene_name",
        "exp_name",
        "raw",
        "start_time",
        "end_time",
        "duration",
        "it/sec",
        "size",
    ]

    os.makedirs(os.path.join(args.results_folder, args.results_prefix), exist_ok=True)
    # Prepare the GCA
    p_bar = tqdm(range(len(prim_entries) * len(gca_configs)))
    for entry in prim_entries:
        for config_path in gca_configs:

            exp_name = os.path.splitext(os.path.basename(config_path))[0]
            raw_name = os.path.splitext(os.path.basename(entry.raw))[0]

            data_root = os.path.join(
                EXP_PRIMITIVES_ROOT,
                entry.scene_name,
                # "latents_gca",
                "raw",
                entry.raw,
            )
            log_dir = resolve_log_dir(
                args.results_folder,
                args.results_prefix,
                exp_name,
                entry.scene_name,
                raw_name,
                test=args.test,
            )
            log_dir = os.path.abspath(log_dir)

            config = prepare_gca_config(
                config_path=config_path,
                data_root=data_root,
                log_dir=log_dir,
                test=args.test,
            )

            os.makedirs(log_dir, exist_ok=True)

            log_config_path = os.path.join(log_dir, "config.yaml")

            # Dump config
            yaml.dump(
                config,
                open(log_config_path, "w"),
            )

            # Copy raw for grower
            shutil.copyfile(data_root, os.path.join(log_dir, "raw.npz"))
            os.makedirs(
                os.path.join(args.results_folder, args.results_prefix, exp_name),
                exist_ok=True,
            )
            shutil.copyfile(
                config_path,
                os.path.join(
                    args.results_folder,
                    args.results_prefix,
                    exp_name,
                    "template.yaml",
                ),
            )

            cmd = f'cd deps/fast-gca && python scripts/run.py --config {log_config_path} --log-dir {log_dir} --override "device=cuda:0"'

            meta = {}

            # Then execute
            os.makedirs(log_dir, exist_ok=True)
            with open(os.path.join(log_dir, "train_logs.txt"), "w+") as f:
                start_time = time.time()
                result = subprocess.run(cmd, shell=True, stderr=f, stdout=f)
                end_time = time.time()

            meta["scene_name"] = entry.scene_name
            meta["exp_name"] = exp_name
            meta["raw"] = entry.raw

            meta["start_time"] = start_time
            meta["end_time"] = end_time
            meta["duration"] = end_time - start_time
            meta["it/sec"] = config["max_steps"] / meta["duration"]
            gca_paths = list(sorted(glob.glob(os.path.join(log_dir, "ckpts", "*"))))
            if len(gca_paths) > 0:
                gca_path = gca_paths[-1]
                file_stats = os.stat(gca_path)
                meta["size"] = file_stats.st_size
            else:
                meta["size"] = -1

            results[
                entry.schedule_name()
                + "_"
                + os.path.splitext(os.path.basename(config_path))[0],
            ] = result.returncode

            with open(
                os.path.join(args.results_folder, args.results_prefix, "report.txt"),
                "a+",
            ) as report_f:

                report_f.write(",".join([str(meta[k]) for k in ENTRIES]) + "\n")

            p_bar.update()

    for k, v in results.items():
        print(k, v)


if __name__ == "__main__":

    signal.signal(signal.SIGINT, signal_handler)

    parser = argparse.ArgumentParser()
    parser.add_argument("--gca_configs", type=str, required=True)
    parser.add_argument("--prim_list", type=str, required=True)
    parser.add_argument("--results_prefix", type=str, required=True)
    parser.add_argument("--results_folder", type=str, default="experiments/results/")
    parser.add_argument("--test", action="store_true")

    args = parser.parse_args()

    main(args)
