from __future__ import annotations

import os
import yaml
import csv
from dataclasses import dataclass
from typing import List

EXP_PRIMITIVES_ROOT = (
    os.environ["EXP_PRIMITIVES_ROOT"]
    if "EXP_PRIMITIVES_ROOT" in os.environ
    else "primitives"
)


def prepare_gca_config(
    config_path: str, data_root: str, log_dir: str, test: bool = False
):
    gca_config = yaml.load(
        open(os.path.expandvars(config_path)), Loader=yaml.FullLoader
    )
    gca_config["data_root"] = os.path.abspath(data_root)
    gca_config["log_dir"] = os.path.abspath(log_dir)

    if test:
        gca_config["max_steps"] = 10

    return gca_config


def resolve_log_dir(
    results_folder: str,
    results_prefix: str,
    exp_name: str,
    scene_name: str,
    raw_name: str,
    test: bool = False,
):
    log_dir = os.path.join(
        results_folder,
        results_prefix + ("_test" if test else ""),
        exp_name,
        scene_name,
        raw_name,
    )
    return os.path.abspath(log_dir)


@dataclass(kw_only=True)
class ExpPrimitiveEntry:
    # With path!
    scene_name: str
    gaussian_ckpt: str
    raw: str
    brush_painting: str
    camera_file: str

    def resolve(self) -> ExpPrimitiveEntry:
        return ExpPrimitiveEntry(
            scene_name=self.scene_name,
            gaussian_ckpt=os.path.join(
                EXP_PRIMITIVES_ROOT,
                self.scene_name,
                # "latents_gca",
                "gaussian_ckpt",
                self.gaussian_ckpt,
            ),
            brush_painting=os.path.join(
                EXP_PRIMITIVES_ROOT,
                self.scene_name,
                # "latents_gca",
                "brush_painting",
                self.brush_painting,
            ),
            raw=os.path.join(
                EXP_PRIMITIVES_ROOT,
                self.scene_name,
                # "latents_gca",
                "raw",
                self.brush_painting,
            ),
            camera_file=os.path.join(
                EXP_PRIMITIVES_ROOT,
                self.scene_name,
                # "latents_gca",
                "camera_file",
                self.camera_file,
            ),
        )

    def schedule_name(self) -> str:
        return "_".join([self.scene_name, self.raw])


def read_prim_and_gca(config_paths: str, prim_list: str):
    # Read GCA configs
    with open(config_paths) as file:
        gca_configs = [os.path.expandvars(line.rstrip()) for line in file]
        print(gca_configs)

    # Read Primitive List
    prim_entries = []
    with open(prim_list, newline="") as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=",", quotechar="|")

        for row in csv_reader:
            assert len(row) == 5
            prim_entries.append(
                ExpPrimitiveEntry(
                    scene_name=row[0],
                    gaussian_ckpt=row[1],
                    raw=row[2],
                    brush_painting=row[3],
                    camera_file=row[4],
                )
            )

    return gca_configs, prim_entries


def check_files(paths: List[str]) -> None:
    """Simply check that everything exists and paths are unique"""
    assert len(paths) == len(set(paths))
    for path in paths:
        if not os.path.exists(path):
            raise ValueError(f"Path does not exist: {path}")


def signal_handler(signal, frame):
    exit()
