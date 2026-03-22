from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict
import os
import stat
import yaml

import torch
import numpy as np
import polyscope as ps
import polyscope.imgui as psim

from sklearn.decomposition import PCA
from torch_scatter import scatter, scatter_max

from sprim.gaussians.gaussian_model import GaussianModel
from sprim.configs.base import BaseConfig
from sprim.utils.viewer_utils import (
    CUBE_VERTICES_NP,
    CUBE_EDGES_NP,
)
from sprim.utils.gui_utils import state_button, KEY_HANDLER, save_popup
from sprim.utils.process_utils import (
    filter_bbox,
    world_to_voxel,
    flatten_coord,
    repeat_arbitrary,
    apply_transform,
)
from sprim.gaussians.suggestive_selection import SuggestiveSelection
from sprim.gaussians.pc_selector import PcSelector
from sprim.gaussians.primitive_library import PrimitiveLibrary
from sprim.utils.voxel_set import VoxelSet
from sprim.utils.extraction_helper import ExtractionHelper
from sprim.gaussians.global_state import GLOBAL_STATE
from sprim.utils.history_handler import HistoryHandler

INIT_RES = 128
TARGET_RESOLUTIONS = [64, 32, 16, 8]
DOWNSAMPLING_RATIOS = [2, 2, 2, 2]
DEFAULT_GCA_TEMPLATE = (
    "deps/fast-gca/configs/default/template_fine=1_coarse=3_geo=4_dino=4_s0_T=5.yaml"
)


# Just a small abstraction to easily perform filtering, etc...
@dataclass(kw_only=True)
class GaussianSet:

    mean: torch.Tensor
    opacity: torch.Tensor
    color_feat: torch.Tensor
    feat: torch.Tensor

    def filter(self, filter_mask: torch.Tensor) -> GaussianSet:
        return GaussianSet(
            mean=self.mean[filter_mask],
            opacity=self.opacity[filter_mask],
            color_feat=self.color_feat[filter_mask],
            feat=self.feat[filter_mask],
        )


# Stores results at one level of the hierarchy
@dataclass(kw_only=True)
class ResultLevel:
    res: int
    voxels: torch.Tensor
    voxel_latents: torch.Tensor
    voxel_alpha: torch.Tensor

    def downsample(self, down_ratio: int) -> ResultLevel:
        # Downsample
        new_voxels = self.voxels // down_ratio

        # Find unique and return inverse map to scatter and scatter (mean)
        new_voxels, invmap = torch.unique(new_voxels, dim=0, return_inverse=True)

        new_alphas, idx_max = scatter_max(self.voxel_alpha, invmap, dim=0)

        new_voxels_latents = self.voxel_latents[idx_max.squeeze()]

        return ResultLevel(
            res=self.res // down_ratio,
            voxels=new_voxels,
            voxel_latents=new_voxels_latents,
            voxel_alpha=new_alphas,
        )

    def filter(self, filter_mask) -> ResultLevel:
        return ResultLevel(
            res=self.res,
            voxels=self.voxels[filter_mask],
            voxel_latents=self.voxel_latents[filter_mask],
            voxel_alpha=self.voxel_alpha[filter_mask],
        )


@dataclass
class Results:
    bbox_min: torch.Tensor
    bbox_max: torch.Tensor

    # NEW (may prove to be useful very very soon)
    transform_from_original_scene: np.ndarray | None
    geo_dim: int | None
    feat_dim: int | None

    levels: List[ResultLevel]

    def __init__(
        self,
        bbox_min,
        bbox_max,
        init_level: ResultLevel,
        transform_from_original_scene: np.ndarray | None = None,
        geo_dim: int | None = None,
        feat_dim: int | None = None,
    ):
        self.bbox_min, self.bbox_max = bbox_min, bbox_max

        # NEW
        self.transform_from_original_scene = transform_from_original_scene
        self.geo_dim = geo_dim
        self.feat_dim = feat_dim

        self.levels = [init_level]
        self.downsample_all()

    def downsample_all(self):
        assert len(self.levels) > 0

        self.levels = self.levels[:1]

        for down_ratio in DOWNSAMPLING_RATIOS:
            new_level = self.levels[-1].downsample(down_ratio)
            self.levels.append(new_level)

    def to_dict(self) -> Dict:
        results = {
            "bbox_min": self.bbox_min.cpu().numpy(),
            "bbox_max": self.bbox_max.cpu().numpy(),
        }
        if self.transform_from_original_scene is not None:
            results["transform_from_original_scene"] = (
                self.transform_from_original_scene
            )
        if self.geo_dim is not None and self.feat_dim is not None:
            results["geo_dim"] = self.geo_dim
            results["feat_dim"] = self.feat_dim

        for i, level in enumerate(self.levels):
            results[f"{i}_res"] = level.res
            results[f"{i}_voxels"] = level.voxels.cpu().numpy()
            results[f"{i}_voxels_latents"] = level.voxel_latents.cpu().numpy()

        return results


class LatentExporterGaussians:

    def __init__(
        self,
        device,
        gaussian_model: GaussianModel,
        config: BaseConfig,
        suggestive_selector: SuggestiveSelection,
        get_mask,
        primitive_library: PrimitiveLibrary,
    ) -> None:
        self.device = device
        self.config = config
        self.primitive_library = primitive_library
        self.aggregation_mode: str = "max"
        self.target_dim: int = 8
        self.subdims: dict = {"geo_feat": 4, "feat": 4}
        self.renormalize: bool = True
        self.results: Results | None = None
        self.gaussian_model = gaussian_model
        self.suggestive_selector = suggestive_selector
        self.get_mask = get_mask
        self.config = config
        self.opacity_threshold = 0.1
        self.max_scale = 2.0 * 2.0 / 64.0
        self.filter_bbox: bool = True

        self.slice_plane = None
        self.limit_bbox = None
        self.extraction_helper = ExtractionHelper(
            bbox_min=np.array(self.config.aabb[:3]),
            bbox_max=np.array(self.config.aabb[3:]),
        )

        self.pc_selector = None
        self.results_history = HistoryHandler()

        self._reset_export_name()

    @torch.no_grad()
    def filter_callback(self, filtering_mask: torch.Tensor) -> None:
        # If the mask is empty, we delete everything...
        if torch.count_nonzero(filtering_mask) == 0:
            self.pc_selector.kill()
            self.pc_selector = None
            self.results = None

            return

        self.results.levels[0] = self.results.levels[0].filter(filtering_mask.clone())
        self.results_history.record_new(self.results.levels[0])
        self.results.downsample_all()
        self.preview()

    @torch.no_grad()
    def preview(self):

        bbox_min = torch.tensor(self.bbox_min).float().to(self.device)
        bbox_max = torch.tensor(self.bbox_max).float().to(self.device)

        # ----------------------------
        # Precompute PCA over codebook
        # ----------------------------

        for i, level in enumerate(self.results.levels):
            pos = (
                level.voxels.float() / float(level.res) * (bbox_max - bbox_min)
            ) + bbox_min

            if i == 0:
                if self.pc_selector is not None:
                    self.pc_selector.kill()
                    self.pc_selector = None

                self.pc_selector = PcSelector(
                    f"0_voxels",
                    pos,
                    transform=torch.tensor(self.limit_bbox.get_transform()).cuda(),
                    enabled=True,
                    filter_callback=self.filter_callback,
                )
                pc = self.pc_selector.ps_structure

                pca = PCA(n_components=3)
                pca = pca.fit(level.voxel_latents.cpu().numpy())
            else:
                # Apply forward transform for rendering
                transformed_pos = apply_transform(
                    pos, torch.tensor(self.limit_bbox.get_transform()).cuda()
                )
                pc = ps.register_point_cloud(
                    f"{i}_voxels",
                    transformed_pos.cpu().numpy(),
                    enabled=False,
                )

            feature_maps_pca = pca.transform(level.voxel_latents.cpu().numpy())
            pca_features = torch.sigmoid(torch.tensor(feature_maps_pca).cuda())
            voxel_set = VoxelSet(
                prefix=f"{i}_voxels",
                voxels=level.voxels,
                res=level.res,
                bbox_min=self.results.bbox_min,
                bbox_max=self.results.bbox_max,
                rgb=pca_features,
                enabled=False,
            )

    # We don't transform bbox_min and bbox_max anymore but do the inverse mapping instead
    # This also gives the nice property that everything that is reloaded will be aligned!
    @property
    def bbox_min(self):
        return np.array(self.config.aabb[:3])

    @property
    def bbox_max(self):
        return np.array(self.config.aabb[3:])

    @torch.no_grad()
    def precompute(self):

        # Predefine bounds and resolutions
        bbox_min = torch.tensor(self.bbox_min).to(self.device)
        bbox_max = torch.tensor(self.bbox_max).to(self.device)

        # Fetch gaussians
        mask = self.get_mask()
        if mask is None:
            filtered_gaussians = GaussianSet(
                mean=self.gaussian_model.means,
                opacity=self.gaussian_model.opacities,
                color_feat=torch.cat(
                    [
                        self.gaussian_model.features_dc,
                        self.gaussian_model.features_rest.reshape(
                            self.gaussian_model.features_rest.shape[0], -1
                        ),
                    ],
                    dim=1,
                ),
                feat=self.gaussian_model.features_feat,
            )
        else:
            filtered_gaussians = GaussianSet(
                mean=self.gaussian_model.means[mask],
                opacity=self.gaussian_model.opacities[mask],
                color_feat=torch.cat(
                    [
                        self.gaussian_model.features_dc[mask],
                        self.gaussian_model.features_rest[mask].reshape(
                            self.gaussian_model.features_rest[mask].shape[0], -1
                        ),
                    ],
                    dim=1,
                ),
                feat=self.gaussian_model.features_feat[mask],
            )

        # ---------------------------------------------
        # 0. Filter based on alpha threshold and bbox
        # ---------------------------------------------
        # TODO: filter size?
        filter_mask = (
            torch.sigmoid(filtered_gaussians.opacity).squeeze(1)
            > self.opacity_threshold
        )

        filtered_gaussians = filtered_gaussians.filter(filter_mask)

        # Apply inverse transform to the Gaussians
        inv_transform = torch.linalg.inv(
            torch.tensor(self.limit_bbox.get_transform()).cuda()
        )
        filtered_gaussians.mean = apply_transform(
            filtered_gaussians.mean, inv_transform
        )

        # Filter optionally everyone out of the bbox
        if self.filter_bbox:
            _, filtered_idx = filter_bbox(
                filtered_gaussians.mean, bbox_min, bbox_max, return_indices=True
            )

            filtered_gaussians = filtered_gaussians.filter(filtered_idx)

        # ---------------------------------------------
        # 1. Everything at occ_grid resolution!
        # ---------------------------------------------

        if self.aggregation_mode == "weighted_average":
            assert False
            geo_feat = (alphas * geo_feat).sum(1) / (alphas.sum(1) + 1e-8)
            feat = (alphas * feat).sum(1) / (alphas.sum(1) + 1e-8)
        else:
            # Map positions to the current target resolution
            gaussian_voxels = (
                (filtered_gaussians.mean - bbox_min.unsqueeze(0))
                / (bbox_max.unsqueeze(0) - bbox_min.unsqueeze(0))
                * INIT_RES
            ).int()

            # Find unique and return inverse map to scatter and scatter (mean)
            unique_voxels, invmap = torch.unique(
                gaussian_voxels, dim=0, return_inverse=True
            )

            unique_alpha, idx_max = scatter_max(
                filtered_gaussians.opacity, invmap, dim=0
            )

            unique_color_feat = filtered_gaussians.color_feat[idx_max.squeeze()]
            unique_feat = filtered_gaussians.feat[idx_max.squeeze()]

            # unique_rgb = SH2RGB(unique_color_feat[:, :3])

            # unique_voxels_world = ((bbox_max - bbox_min) * voxel_size).unsqueeze(
            #     0
            # ) * unique_voxels + bbox_min.unsqueeze(0)

            # ps_voxels = ps.register_point_cloud(
            #     "voxels", unique_voxels_world.cpu().numpy()
            # )
            # ps_voxels.add_color_quantity("rgb", unique_rgb.cpu().numpy(), enabled=True)

        all_feats = {"geo_feat": unique_color_feat, "feat": unique_feat}
        pca_feats = []
        for k, v in all_feats.items():
            if self.subdims[k] == 0:
                continue
            elif v.shape[-1] == self.subdims[k]:
                pca_feats.append(v)
            else:
                pca = PCA(n_components=self.subdims[k])
                feature_maps_pca = pca.fit_transform(v.cpu().numpy())
                pca_feats.append(torch.tensor(feature_maps_pca).to(self.device))

            if self.renormalize:
                pca_feats[-1] = (
                    pca_feats[-1] / torch.linalg.norm(pca_feats[-1], dim=1)[:, None]
                )

        pca_feats = torch.cat(pca_feats, dim=1)

        # -----------------------------------
        # Progressively downsample
        # -----------------------------------

        # For each position, find the corresponding voxels at INGP res
        ngp_voxels = torch.clone(unique_voxels)

        self.results = Results(
            bbox_min=bbox_min,
            bbox_max=bbox_max,
            geo_dim=self.subdims["geo_feat"],
            feat_dim=self.subdims["feat"],
            transform_from_original_scene=self.limit_bbox.get_transform(),
            init_level=ResultLevel(
                res=INIT_RES,
                voxels=ngp_voxels,
                voxel_latents=pca_feats,
                voxel_alpha=unique_alpha,
            ),
        )

        self.results_history.record_new(self.results.levels[0])

    def _reset_export_name(self):
        latent_gca_raw_folder = os.path.join(self.config.log_dir, "latents_gca", "raw")
        if os.path.exists(latent_gca_raw_folder):
            prev_list = list(sorted(os.listdir(latent_gca_raw_folder)))
            prev_list = [
                x.split(".")[0] for x in prev_list if x.split(".")[0].isdigit()
            ]
            prev_list = sorted([int(x) for x in prev_list])
            if len(prev_list) > 0:
                n_prev = prev_list[-1] + 1
                self.export_name = f"{n_prev:06d}"
            else:
                self.export_name = f"{0:06d}"
        else:
            self.export_name = f"{0:06d}"

    def get_outputs(self):
        assert os.path.exists(
            self.config.log_dir
        ), f"log_dir doesn't exist: {self.config.log_dir}"

        os.makedirs(
            os.path.join(self.config.log_dir, "latents_gca", "raw"), exist_ok=True
        )
        os.makedirs(
            os.path.join(self.config.log_dir, "latents_gca", "gca_config"),
            exist_ok=True,
        )
        os.makedirs(
            os.path.join(self.config.log_dir, "latents_gca", "gca_execute"),
            exist_ok=True,
        )
        os.makedirs(
            os.path.join(self.config.log_dir, "latents_gca", "gaussian_ckpt"),
            exist_ok=True,
        )

        output_latents = os.path.abspath(
            os.path.join(
                self.config.log_dir, "latents_gca", "raw", f"{self.export_name}.npz"
            )
        )
        output_config = os.path.abspath(
            os.path.join(
                self.config.log_dir,
                "latents_gca",
                "gca_config",
                f"{self.export_name}.yaml",
            )
        )
        output_execute = os.path.abspath(
            os.path.join(
                self.config.log_dir,
                "latents_gca",
                "gca_execute",
                f"{self.export_name}.sh",
            )
        )
        output_gaussian_ckpt = os.path.abspath(
            os.path.join(
                self.config.log_dir,
                "latents_gca",
                "gaussian_ckpt",
                f"{self.export_name}.pt",
            )
        )
        output_gca_log_dir = os.path.abspath(
            os.path.join(
                self.config.log_dir, "latents_gca", "gca_logs", f"{self.export_name}"
            )
        )

        return (
            output_latents,
            output_config,
            output_execute,
            output_gaussian_ckpt,
            output_gca_log_dir,
        )

    def _get_exported_gaussians(
        self,
        transform_back: bool = False,
        invert_selection: bool = False,
    ):
        """
        Create mask to only produce Gaussians within the selected voxels.
        This reuses the strategy used for copying gaussians
        """
        # TODO: unify everything?

        bbox_min = torch.tensor(self.bbox_min).to(self.device)
        bbox_max = torch.tensor(self.bbox_max).to(self.device)

        filtered_gaussians = self.gaussian_model.get_gaussian_set()
        mask = self.get_mask()
        if mask is not None:
            filtered_gaussians = filtered_gaussians.filter(mask)

        # Apply inverse transform to the Gaussians NOTE: this will make exported
        # Gaussians axis aligned because the bbox is standardized! In other
        # words, don't reopen the scene with the initial set of points!
        transform = torch.tensor(self.limit_bbox.get_transform()).cuda()
        inv_transform = torch.linalg.inv(transform)
        # Apply proper transform on GAUSSIANS here!
        filtered_gaussians = filtered_gaussians.transform(inv_transform)

        # Filter gaussians outside the bbox
        if not invert_selection:
            _, filter_idx = filter_bbox(
                filtered_gaussians.means, bbox_min, bbox_max, return_indices=True
            )

            filtered_gaussians = filtered_gaussians.filter(filter_idx)

        gaussian_voxels = world_to_voxel(
            filtered_gaussians.means, bbox_min, bbox_max, INIT_RES
        ).int()

        # Filter gaussians so that the prefix sum match concurrently
        # WARNING: there is no guarantee that, over time, the pytorch
        # implementation keeps the same behavior (actually it should be now)
        flat_gaussian_voxels = flatten_coord(None, gaussian_voxels).long()
        # TODO: remove this assert
        assert len(torch.unique(flat_gaussian_voxels)) == len(
            torch.unique(gaussian_voxels, dim=0)
        )
        sorted_idx = torch.argsort(flat_gaussian_voxels)

        gaussian_voxels = gaussian_voxels[sorted_idx]
        filtered_gaussians = filtered_gaussians.filter(sorted_idx)

        # Ref count is the number of gaussians for each voxel
        # Since we have presorted gaussians, ref_prefix_sum will directly
        # provide offset to read all gaussians within a given voxel (for `filtered_gaussians`)
        ref_voxels, ref_count = torch.unique(gaussian_voxels, dim=0, return_counts=True)
        ref_prefix_sum = torch.cat(
            [
                torch.tensor([0], dtype=torch.int, device=ref_count.device),
                torch.cumsum(ref_count, dim=0),
            ]
        )  # EXCLUSIVE prefixsum

        # ---------------------------------------
        # Index
        # ---------------------------------------

        # Now, we can map each generated voxel to its reference voxel
        # To do so, simply do a unique and it should index into the first array
        n_ref_voxels = ref_voxels.shape[0]
        _, invmap_to_ref, counts = torch.unique(
            torch.cat([ref_voxels, self.results.levels[0].voxels.int()], dim=0),
            dim=0,
            return_inverse=True,
            return_counts=True,
        )

        # This invmap gives us precisely the reference voxel for each new surface voxel
        invmap_to_ref = invmap_to_ref[n_ref_voxels:]

        # Safety!
        # assert (
        #     counts[invmap_to_ref].min().item() >= 2
        #     and invmap_to_ref.max() < n_ref_voxels
        # )

        # Filter out everyone that would fall outside...
        invmap_to_ref = invmap_to_ref[invmap_to_ref < n_ref_voxels]

        # Use invmap_to_ref to know the number of gaussians we'll have to copy for each voxel
        counts_in_new = ref_count[invmap_to_ref]

        # Then, we can derive the offsets with an EXCLUSIVE prefix sum
        prefix_sum_new = torch.cat(
            [
                torch.tensor([0], dtype=torch.int, device=counts_in_new.device),
                torch.cumsum(counts_in_new, dim=0),
            ]
        )  # EXCLUSIVE prefixsum

        # Initialize the final indices (proper indexing comes right after with offsets)
        all_new_gaussian_indices = torch.arange(
            prefix_sum_new[-1],
            device=prefix_sum_new.device,  # This is simply the total count
        )
        # offset in ref
        repeated_at_ref = repeat_arbitrary(ref_prefix_sum[invmap_to_ref], counts_in_new)

        # offset in new
        repeated_at_new, repeated_idx = repeat_arbitrary(
            prefix_sum_new[:-1],
            counts_in_new,
            return_indices=True,
        )

        # assert len(all_new_gaussian_indices) == len(repeated_at_ref) and len(
        #     repeated_at_ref
        # ) == len(repeated_at_new)

        # Then, we can derive the exact indices
        all_new_gaussian_indices = (
            all_new_gaussian_indices - repeated_at_new + repeated_at_ref
        )

        # ---------------------------------------
        # Create and remap
        # ---------------------------------------

        if invert_selection:
            # NOTE: we don't apply clip scaling for that
            inverted_mask = ~torch.isin(
                torch.arange(len(filtered_gaussians.means)).cuda(),
                all_new_gaussian_indices,
            )
            filtered_gaussians = filtered_gaussians.filter(inverted_mask)
        else:
            filtered_gaussians = filtered_gaussians.filter(all_new_gaussian_indices)

            # Apply scale filtering
            filtered_gaussians.clip_scale(self.max_scale)

        if transform_back:
            filtered_gaussians = filtered_gaussians.transform(transform)

        return filtered_gaussians, transform

    def create_layer(self, invert_selection: bool = False):
        assert self.results is not None

        # Kill the selection immediately!
        self.pc_selector.kill()
        self.pc_selector = None

        exported_gaussians, transform = self._get_exported_gaussians(
            transform_back=True, invert_selection=invert_selection
        )

        self.primitive_library.add_layer_from_gaussians(
            exported_gaussians, transform=transform
        )

    def export(self):

        assert self.results is not None

        # Kill the selection immediately!
        self.pc_selector.kill()
        self.pc_selector = None

        (
            output_latents,
            output_config,
            output_execute,
            output_gaussian_ckpt,
            output_gca_log_dir,
        ) = self.get_outputs()

        # Save voxel latents
        np.savez(output_latents, **self.results.to_dict())

        # Create a config file given the template: simply overwrite data_root
        # with the newly precomputed latents
        gca_config = yaml.load(
            open(os.path.expandvars(DEFAULT_GCA_TEMPLATE)), Loader=yaml.FullLoader
        )
        gca_config["data_root"] = output_latents
        gca_config["log_dir"] = output_gca_log_dir
        yaml.dump(
            gca_config,
            open(output_config, "w"),
        )

        # Create an executable file to start training
        executable_str = f'#!/bin/sh\ncd $GCA_ROOT\npython scripts/run.py --config {output_config} --override "device=cuda:0"'
        with open(output_execute, "w") as f:
            f.write(executable_str)
        st = os.stat(output_execute)
        os.chmod(output_execute, st.st_mode | stat.S_IEXEC)

        ref_gaussians = self.gaussian_model.get_gaussian_set()

        # Make sure to reset the mask to avoid any issues
        # self.suggestive_selector.reset_mask()
        exported_gaussians, _ = self._get_exported_gaussians()
        self.gaussian_model.set_gaussian_set(exported_gaussians)
        # self.gaussian_model.feature_quantizer = None

        torch.save(
            {
                "step": 0,
                "name": self.config.name,
                "gaussian_model": self.gaussian_model.state_dict(),
            },
            output_gaussian_ckpt,
        )

        # Reset everything
        self._reset_export_name()

        # Then reset rendering to the initial scene
        # This way, nothing will be broken...
        self.gaussian_model.set_gaussian_set(ref_gaussians)

        # Reset the library
        self.primitive_library.load_library()

        # Clear self.results to ensure we're not allowing another export on top of it
        self.results = None

    def preview_selection(self):
        if self.results is None:
            print("WARNING: cannot preview selection without an actual selection!")
            return

        filtered_gaussians, _ = self._get_exported_gaussians(transform_back=True)
        # First transform with the same transformation fo the current set of Gaussians
        if self.primitive_library.current_entry.transform is not None:
            filtered_gaussians = filtered_gaussians.transform(
                torch.inverse(self.primitive_library.current_entry.transform)
            )
        self.primitive_library.current_entry.gaussian_model.set_gaussian_set(
            filtered_gaussians, grown=True
        )
        self.primitive_library.gaussian_update_callback()

    # The limit bbox is initialized from the original scene bbox
    def init_limit_bbox(self):
        bbox_min = np.array(self.config.aabb[:3])
        bbox_max = np.array(self.config.aabb[3:])
        cube_vertices = (bbox_max - bbox_min) * CUBE_VERTICES_NP + bbox_min

        self.limit_bbox = ps.register_curve_network(
            "limit_bbox_latent_exporter", cube_vertices, CUBE_EDGES_NP, radius=0.01
        )
        self.limit_bbox.enable_transform_gizmo()
        self.limit_bbox.set_transform_mode_gizmo(
            ps.TransformMode.TRANSLATION | ps.TransformMode.ROTATION
        )
        self.limit_bbox_translation = True
        GLOBAL_STATE.use_depth = True

    # def init_checker_bbox(self):
    #     bbox_min = np.array(self.config.aabb[:3])
    #     bbox_max = np.array(self.config.aabb[3:])
    #     self.checker_bbox = create_checker_bbox(
    #         "checker_bbox_latent_exporter", 10, bbox_min, bbox_max
    #     )

    def _history_previous(self):
        level0 = self.results_history.previous()
        if level0 is not None:
            self.results.levels[0] = level0
            self.results.downsample_all()
            self.preview()

    def _history_next(self):
        level0 = self.results_history.next()
        if level0 is not None:
            self.results.levels[0] = level0
            self.results.downsample_all()
            self.preview()

    def gui(self):

        if psim.TreeNode("Latent Exporter##latent_exporter"):

            if self.limit_bbox is None:
                self.init_limit_bbox()
            else:
                # if psim.Button("Checker bbox"):
                #     self.init_checker_bbox()

                # if self.checker_bbox is not None:
                #     self.checker_bbox.set_transform(self.limit_bbox.get_transform())

                self.limit_bbox.set_enabled(True)

            if self.limit_bbox.is_enabled():
                if KEY_HANDLER("s"):
                    self.limit_bbox.set_transform_mode_gizmo(
                        ps.TransformMode.TRANSLATION | ps.TransformMode.ROTATION
                        if self.limit_bbox_translation
                        else ps.TransformMode.SCALE
                    )
                    self.limit_bbox_translation = not self.limit_bbox_translation

            if (
                not self.limit_bbox.is_enabled_transform_gizmo()
                and self.pc_selector is None
            ):
                if psim.Button("Limit bbox##latent_exporter"):
                    self.init_limit_bbox()

            has_grown_params = (
                self.primitive_library.current_entry.gaussian_model.grown_gauss_params
                is not None
            )
            clicked, _ = state_button(
                not has_grown_params,
                "Preview selection##latent_exporter",
                "Hide selection##latent_exporter",
            )
            if clicked or KEY_HANDLER("o"):
                if has_grown_params:
                    self.primitive_library.current_entry.gaussian_model.set_gaussian_set(
                        None, grown=True
                    )
                    self.primitive_library.gaussian_update_callback()
                else:
                    if self.pc_selector is not None:
                        self.pc_selector.set_enabled(False)
                    self.preview_selection()

            if psim.Button(
                "Precompute##latent_exporter"
                if self.pc_selector is None
                else "Re-compute##latent_exporter"
            ):
                self.precompute()
                self.preview()
                if self.limit_bbox is not None:
                    self.limit_bbox.enable_transform_gizmo(False)

            if self.results is not None:

                # --------------------------
                # HISTORY
                # --------------------------
                io = psim.GetIO()
                if io.KeyCtrl and KEY_HANDLER("z"):
                    self._history_previous()
                elif io.KeyCtrl and KEY_HANDLER("y"):
                    self._history_next()

                # --------------------------

                if psim.Button("Layer##latent_exporter"):
                    self.create_layer()

                psim.SameLine()

                if psim.Button("Inverted Layer##latent_exporter"):
                    self.create_layer(invert_selection=True)

                requested, self.export_name = save_popup(
                    "export##latent_exporter", self.export_name, "Export"
                )

                if requested:
                    self.export()

            if self.pc_selector is not None:
                self.pc_selector.gui()

                if psim.Button("Delete##latent_exporter"):
                    self.pc_selector.kill()
                    self.results = None
                    self.pc_selector = None

            if psim.TreeNode("Extraction strategy##latent_exporter"):

                _, self.target_dim = psim.InputInt(
                    "target_dim##latent_exporter", self.target_dim
                )

                clicked, self.subdims["geo_feat"] = psim.SliderInt(
                    "geo_feat_dim##latent_exporter",
                    self.subdims["geo_feat"],
                    v_min=0,
                    v_max=self.target_dim,
                )
                if clicked:
                    self.subdims["feat"] = self.target_dim - self.subdims["geo_feat"]

                clicked, self.subdims["feat"] = psim.SliderInt(
                    "feat_dim##latent_exporter",
                    self.subdims["feat"],
                    v_min=0,
                    v_max=self.target_dim,
                )
                if clicked:
                    self.subdims["geo_feat"] = self.target_dim - self.subdims["feat"]

                _, self.renormalize = psim.Checkbox(
                    "renormalize##latent_exporter", self.renormalize
                )

                _, self.opacity_threshold = psim.SliderFloat(
                    "opacity_threshold##latent_exporter",
                    self.opacity_threshold,
                    v_min=0.001,
                    v_max=0.5,
                )

                _, self.max_scale = psim.SliderFloat(
                    "max_scale##latent_exporter",
                    self.max_scale,
                    v_min=0.001,
                    v_max=1.0,
                    power=-2,
                )

                _, self.filter_bbox = psim.Checkbox("filter_bbox", self.filter_bbox)

                psim.TreePop()

            if self.limit_bbox is not None:
                self.extraction_helper.gui()
                self.extraction_helper.set_transform(self.limit_bbox.get_transform())

            psim.TreePop()

        else:
            # Hide selection bbox outside
            if self.limit_bbox is not None:
                self.limit_bbox.set_enabled(False)
                self.limit_bbox.enable_transform_gizmo(False)

            self.extraction_helper.set_enabled(False)

        psim.Separator()
