import os
from typing import List
import glob
import argparse
import math
import signal
import polyscope as ps
import imageio
import time

import torch
import numpy as np

from sprim.utils.exp_utils import (
    ExpPrimitiveEntry,
    EXP_PRIMITIVES_ROOT,
    read_prim_and_gca,
    signal_handler,
    resolve_log_dir,
    check_files,
)
from sprim.inverse.grown_voxels import LatentMode
from sprim.gaussians.primitive_library import PrimitiveEntry
from sprim.utils.io_utils import load_camera_from_file
from sprim.utils.gui_utils import RenderMode
from sprim.utils.geometry import fov2focal
from nerfstudio.cameras.cameras import Cameras


RENDER_BACKGROUND: List[int] = [1.0, 1.0, 1.0]
TIMINGS_ENTRY = [
    "gca_time",
    "patch_time",
    "update_gaussian_time",
    "grown_voxels_gca",
    "grown_voxels_patch",
]


class RenderWrapper:

    def __init__(self, args) -> None:

        self.n_samples = args.n_samples
        self.features = args.features
        self.results_folder = args.results_folder
        self.results_prefix = args.results_prefix
        self.patch_size = 5

        self.gca_configs, self.prim_entries = read_prim_and_gca(
            args.gca_configs, args.prim_list
        )

        check_files(self.gca_configs)

        self.i_entry = 0
        self.i_config = -1
        self.i_sample = 0
        self.i_grow = 0
        self.primitive_entry = None

        # ---------------------
        # INIT POLYSCOPE
        # ---------------------

        ps.init()
        ps.set_ground_plane_mode("none")

        ps.set_user_callback(self.tick)
        ps.show()

    def next_entry_config_sample(self):
        # If we've reached the last entry
        if self.i_sample == self.n_samples - 1 or self.primitive_entry is None:

            if self.primitive_entry is not None:
                self.record_timings()

            if self.i_config == len(self.gca_configs) - 1:
                self.i_config = 0
                self.i_entry += 1
            else:
                self.i_config += 1

            # If we have gone through all entries, exit
            if self.i_entry == len(self.prim_entries):
                exit()

            self.i_sample = 0
            self.timings = []

            self.load_next_entry_config()

        # Otherwise, simply increment i_samples
        else:
            self.i_sample += 1

        self.sample_current()

        self.i_grow = 0

    def load_next_entry_config(self):

        config_path = self.gca_configs[self.i_config]
        entry = self.prim_entries[self.i_entry]

        # First grow, then render everyone one by one
        exp_name = os.path.splitext(os.path.basename(config_path))[0]
        raw_name = os.path.splitext(os.path.basename(entry.raw))[0]

        log_dir = resolve_log_dir(
            args.results_folder,
            args.results_prefix,
            exp_name,
            entry.scene_name,
            raw_name,
        )
        log_dir = os.path.abspath(log_dir)

        # Read GCA checkpoints
        gca_paths = list(sorted(glob.glob(os.path.join(log_dir, "ckpts", "*"))))
        if len(gca_paths) == 0:
            raise ValueError(f"No checkpoints at {log_dir}")
        gca_path = gca_paths[-1]

        resolved_entry: ExpPrimitiveEntry = entry.resolve()

        # Load corresponding entry
        self.primitive_entry = PrimitiveEntry.load_entry(
            path=resolved_entry.gaussian_ckpt,
            gca_path=gca_path,
            painting_path=resolved_entry.brush_painting,
            headless_mode=True,
        )
        self.primitive_entry.brush_painter._prepare()

        cam_params, window_size = load_camera_from_file(
            camera_path=resolved_entry.camera_file
        )
        assert cam_params is not None and window_size is not None
        self.cam_params = cam_params
        self.window_size = window_size

        # Set polyscope camera
        ps.set_window_size(window_size[0], window_size[1])
        ps.set_view_camera_parameters(self.cam_params)

        self.render_dir = os.path.join(log_dir, "renders")

        self.primitive_entry.grower.patch_parameters.patch_size = self.patch_size
        # Make sure the loaded grower has the right LatentMode
        self.primitive_entry.grower.latent_voxel_mode = LatentMode.NN_patch
        self.primitive_entry.grower.latent_voxels_shown = True

    def sample_current(self):
        self.primitive_entry.grower.restart()

        # NOTE: we empty cache because ME literaly shreds GPU memory :'(
        torch.cuda.empty_cache()

        # First, grow
        start_gca_time = time.time()
        while True:
            is_stepping = self.primitive_entry.grower.grow()
            if not is_stepping:
                break
        end_gca_time = time.time()

        # Count all voxels
        i_start_patch = len(self.primitive_entry.grower.grown_voxels)
        grown_voxels_gca = sum(
            [len(x.surface_voxels) for x in self.primitive_entry.grower.grown_voxels]
        )

        torch.cuda.empty_cache()

        # Then, patch
        start_patch_time = time.time()
        self.primitive_entry.grower.patch_current_state(True)
        end_patch_time = time.time()

        grown_voxels_patch = sum(
            [
                len(x.surface_voxels)
                for x in self.primitive_entry.grower.grown_voxels[i_start_patch:]
            ]
        )

        self.timings.append(
            {
                "gca_time": end_gca_time - start_gca_time,
                "patch_time": end_patch_time - start_patch_time,
                "grown_voxels_gca": grown_voxels_gca,
                "grown_voxels_patch": grown_voxels_patch,
            }
        )

    def record_timings(self):
        ENTRIES = [
            "scene_name",
            "exp_name",
            "raw",
            "n_samples",
            "ref_voxels",
        ] + TIMINGS_ENTRY

        exp_name = os.path.splitext(os.path.basename(self.gca_configs[self.i_config]))[
            0
        ]
        meta = {
            "scene_name": self.prim_entries[self.i_entry].scene_name,
            "exp_name": exp_name,
            "raw": self.prim_entries[self.i_entry].raw,
            "n_samples": self.n_samples,
            "ref_voxels": len(self.primitive_entry.grower.state.ref_coord[0]),
        }
        for k in TIMINGS_ENTRY:
            meta[k] = np.mean([sample_timings[k] for sample_timings in self.timings])

        with open(
            os.path.join(self.results_folder, self.results_prefix, "report_render.txt"),
            "a+",
        ) as report_f:

            report_f.write(",".join([str(meta[k]) for k in ENTRIES]) + "\n")

    def tick(self):
        print(
            f"ticking: {self.i_entry}; {self.i_config}; {self.i_sample}; {self.i_grow}"
        )
        if (
            self.primitive_entry is None
            or self.i_grow == len(self.primitive_entry.grower.grown_voxels) - 1
        ):
            self.next_entry_config_sample()
        else:
            # Otherwise, keep iterating
            self.primitive_entry.grower.i_trajectory = self.i_grow
            start_update_gaussian_time = time.time()
            self.primitive_entry.grower.update_render()
            placeholder_lock = len(
                self.primitive_entry.grower.gaussian_model.get_gaussian_set(grown=True)
            )
            end_update_gaussian_time = time.time()

            self.timings[-1]["update_gaussian_time"] = (
                end_update_gaussian_time - start_update_gaussian_time
            )

            self.render_save()

            self.i_grow += 1

    def render_save(self):

        sample_dir = os.path.join(
            self.render_dir, f"{self.i_sample:03d}_patch={self.patch_size}"
        )
        os.makedirs(sample_dir, exist_ok=True)

        patch_suffix = (
            "_patched"
            if (
                self.primitive_entry.grower.i_trajectory
                > self.primitive_entry.grower.growing_parameters.sampling_steps + 1
            )
            else ""
        )

        img = self.render()

        imageio.imwrite(
            os.path.join(sample_dir, f"rgb_{self.i_grow:02d}{patch_suffix}.png"),
            (255 * img.cpu().numpy()).astype(np.uint8),
        )

        if self.features:

            img = self.render(
                RenderMode.FEATURES,
            )

            imageio.imwrite(
                os.path.join(sample_dir, f"feat_{self.i_grow:02d}{patch_suffix}.png"),
                (255 * img.cpu().numpy()).astype(np.uint8),
            )

        ps.screenshot(
            os.path.join(sample_dir, f"zps_{self.i_grow:02d}{patch_suffix}.png")
        )

    def render(
        self,
        render_mode: RenderMode = RenderMode.RGB,
    ):
        self.primitive_entry.gaussian_model.eval()

        c2w = torch.linalg.inv(
            torch.tensor(self.cam_params.get_view_mat(), device="cuda")
        )

        WIDTH = self.window_size[0]
        HEIGHT = self.window_size[1]
        focal = fov2focal(
            self.cam_params.get_fov_vertical_deg() * math.pi / 180.0,
            HEIGHT,
        )

        camera = Cameras(
            camera_to_worlds=c2w[:3, :4].cpu(),
            fx=focal,
            fy=focal,
            cx=WIDTH / 2.0,
            cy=HEIGHT / 2.0,
            width=WIDTH,
            height=HEIGHT,
        ).to("cuda")

        background_color = torch.Tensor(RENDER_BACKGROUND).cuda()
        background_feat = torch.zeros(
            self.primitive_entry.config.feature_dim, device="cuda"
        )

        render_pkg = self.primitive_entry.gaussian_model.render(
            camera,
            self.primitive_entry.config,
            return_feat=render_mode == RenderMode.FEATURES,
            render_envmap=False,
            mask=None,
            background_color=background_color,
            background_feat=background_feat,
            render_grown=True,
        )

        alpha = render_pkg["alpha"]

        if render_mode == RenderMode.RGB:
            rgb = render_pkg["rgb"]
            rendered_image = torch.cat(
                [
                    rgb,
                    alpha,
                ],
                dim=-1,
            )
        else:
            feat = render_pkg["feat"]
            feat_shape = feat.shape

            rendered_feat = self.primitive_entry.gaussian_model.render_pca.render(feat)

            rendered_image = torch.cat(
                [
                    rendered_feat,
                    alpha,
                ],
                dim=-1,
            )

        return rendered_image


@torch.no_grad()
def main(args):

    RenderWrapper(args)


if __name__ == "__main__":

    signal.signal(signal.SIGINT, signal_handler)

    parser = argparse.ArgumentParser()
    parser.add_argument("--gca_configs", type=str, required=True)
    parser.add_argument("--prim_list", type=str, required=True)
    parser.add_argument("--results_prefix", type=str, required=True)
    parser.add_argument("--results_folder", type=str, default="experiments/results/")
    parser.add_argument("--n_samples", type=int, default=10)
    parser.add_argument("--features", action="store_true")
    parser.add_argument("--patch_size", type=int, nargs="+", default=[5])

    args = parser.parse_args()

    main(args)
