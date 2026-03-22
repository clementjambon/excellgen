from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Dict, Any, List, Callable
from enum import StrEnum
import os
import yaml

from torch.utils.tensorboard import SummaryWriter
import torch
import numpy as np
import polyscope.imgui as psim

import MinkowskiEngine as ME
from MinkowskiEngine import SparseTensor

from sprim.patch.exact_search import exact_search, PatchParameters
from sprim.configs.base import BaseConfig
from sprim.gaussians.gaussian_model import GaussianModel, GaussianSet
from sprim.utils.process_utils import (
    filter_bbox,
    repeat_arbitrary,
    voxel_to_world,
    world_to_voxel,
    flatten_coord,
)
from sprim.utils.voxel_set import VoxelSet

RESOLUTIONS = [128, 64, 32, 16, 8]
RESOLUTIONS_INV = {128: 0, 64: 1, 32: 2, 16: 3, 8: 4}


class LatentMode(StrEnum):
    RAW = "raw"
    NN_single = "nn_single"
    NN_patch = "nn_patch"


LATENT_MODE_MAP = {x: i for i, x in enumerate(LatentMode)}
LATENT_MODE_INVMAP = {i: x for i, x in enumerate(LatentMode)}


# Wrapper to hold outputs
@dataclass
class GrownVoxels:
    voxel_res: int
    voxel_res_idx: int

    surface_voxels: torch.Tensor
    latents: torch.Tensor
    latents_matched: torch.Tensor
    # Coordinates in the original NeRF
    coord_in_ref: torch.Tensor
    coord_in_ref_matched: torch.Tensor

    # For VoxelSet
    get_transform: Callable[[], torch.Tensor]
    pca_to_rgb: Callable[[int, torch.Tensor], torch.Tensor]
    bbox_min: torch.Tensor
    bbox_max: torch.Tensor

    voxel_set: VoxelSet = None

    def __init__(
        self,
        voxel_res: int,
        surface_voxels: torch.Tensor,
        latents: torch.Tensor,
        latents_matched: torch.Tensor,
        coord_in_ref: torch.Tensor,
        get_transform: Callable[[], torch.Tensor],
        pca_to_rgb: Callable[[int, torch.Tensor], torch.Tensor],
        bbox_min,
        bbox_max,
        voxel_res_idx: int | None = None,
    ):
        self.voxel_res = voxel_res
        self.voxel_res_idx = (
            RESOLUTIONS_INV[self.voxel_res] if voxel_res_idx is None else voxel_res_idx
        )

        self.surface_voxels = surface_voxels
        self.latents = latents
        self.latents_matched = latents_matched
        self.coord_in_ref = coord_in_ref

        # For Voxelset
        self.get_transform = get_transform
        self.pca_to_rgb = pca_to_rgb
        self.bbox_min = bbox_min
        self.bbox_max = bbox_max

        self.voxel_set = None

    def serialize(self):
        results = {
            "voxel_res": self.voxel_res,
            "voxel_res_idx": self.voxel_res_idx,
            "surface_voxels": self.surface_voxels.cpu().numpy(),
            "latents": self.latents.cpu().numpy(),
            "coord_in_ref": self.coord_in_ref.cpu().numpy(),
            # The gridmap can be reconstructed automatically
        }
        if self.latents_matched is not None:
            results["latents_matched"] = self.latents_matched.cpu().numpy()

        return results

    @staticmethod
    def deserialize(
        data: Dict[str, Any], get_transform, pca_to_rgb, bbox_min, bbox_max
    ) -> GrownVoxels:
        return GrownVoxels(
            voxel_res=data["voxel_res"],
            voxel_res_idx=data["voxel_res_idx"] if "voxel_res_idx" in data else None,
            surface_voxels=torch.tensor(data["surface_voxels"]).cuda().long(),
            latents=torch.tensor(data["latents"]).cuda().float(),
            latents_matched=(
                torch.tensor(data["latents"]).cuda().float()
                if "latents_matched"
                else None
            ),
            coord_in_ref=torch.tensor(data["coord_in_ref"]).cuda().long(),
            get_transform=get_transform,
            pca_to_rgb=pca_to_rgb,
            bbox_min=bbox_min,
            bbox_max=bbox_max,
        )

    # Similar to `create_blob(...)` without the offset
    def _all_subvoxels(self, ratio: int):
        axes_ranges = [torch.arange(ratio).cuda() for _ in range(3)]

        grids = torch.meshgrid(*axes_ranges, indexing="ij")
        grids = [grid.flatten() for grid in grids]
        blob_coord = torch.stack(grids, dim=-1)
        return blob_coord

    def to_binaries(self, binary_res: int):
        # TODO: make that less bruteforce by using the existing binaries within the
        # original "Estimator"
        assert binary_res >= self.voxel_res and binary_res % self.voxel_res == 0
        ratio = binary_res // self.voxel_res
        blob_coord = self._all_subvoxels(ratio)
        all_possible_subvoxels = ratio * torch.repeat_interleave(
            self.surface_voxels, blob_coord.shape[0], dim=0
        ) + blob_coord.repeat((self.surface_voxels.shape[0], 1))

        new_binaries = torch.zeros([binary_res] * 3, dtype=torch.bool).cuda()
        new_binaries[tuple(all_possible_subvoxels.T)] = True
        new_binaries = new_binaries.unsqueeze(0).contiguous()

        return new_binaries

    def get_coord_in_ref(
        self, ref_coord: torch.Tensor, ref_feat: torch.Tensor, latent_mode: LatentMode
    ):
        if ref_coord is None or ref_feat is None:
            return self.coord_in_ref

        if latent_mode == LatentMode.NN_single:
            _, _, new_ref_coords = exact_search(
                state_coord=self.surface_voxels,
                state_feat=self.latents,
                ref_coord=ref_coord,
                ref_feat=ref_feat,
                # Simple 1-NN
                patch_parameters=PatchParameters(patch_size=1, patch_iters=1),
            )
            coord_in_ref = new_ref_coords[-1]
        else:
            coord_in_ref = self.coord_in_ref
        return coord_in_ref

    def process_gaussians(
        self,
        gaussians: GaussianSet,
        bbox_min: torch.Tensor,
        bbox_max: torch.Tensor,
        ref_coord: torch.Tensor = None,
        ref_feat: torch.Tensor = None,
        latent_mode: LatentMode = LatentMode.RAW,
        filter_outside: bool = True,
    ) -> GaussianSet:

        # ---------------------------------------
        # Filter and count reference voxels
        # ---------------------------------------

        if filter_outside:
            # Filter the initial set of gaussians outside the bbox
            _, filter_idx = filter_bbox(
                gaussians.means, bbox_min, bbox_max, return_indices=True
            )
            filtered_gaussians = gaussians.filter(filter_idx)
        else:
            filtered_gaussians = gaussians

        gaussian_voxels = world_to_voxel(
            filtered_gaussians.means, bbox_min, bbox_max, self.voxel_res
        ).int()

        # Filter gaussians so that the prefix sum match concurrently
        # WARNING: there is no guarantee that, over time, the pytorch
        # implementation keeps the same behavior
        # NOTE: make it long just in case
        flat_gaussian_voxels = flatten_coord(None, gaussian_voxels).long()
        sorted_idx = torch.argsort(flat_gaussian_voxels)
        # flat_gaussian_voxels = flat_gaussian_voxels[sorted_idx]
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
            torch.cat(
                [
                    ref_voxels,
                    self.get_coord_in_ref(ref_coord, ref_feat, latent_mode).int(),
                ],
                dim=0,
            ),
            dim=0,
            return_inverse=True,
            return_counts=True,
        )

        # This invmap gives us precisely the reference voxel for each new surface voxel
        invmap_to_ref = invmap_to_ref[n_ref_voxels:]

        # Safety!
        # breakpoint()
        # assert (
        #     counts[invmap_to_ref].min().item() >= 2
        #     and invmap_to_ref.max() < n_ref_voxels
        # )

        # Filter out everyone that would fall outside...
        invmap_to_ref = invmap_to_ref[invmap_to_ref < n_ref_voxels]

        # if invmap_to_ref.max() >= n_ref_voxels:
        #     pass

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

        # And also the target offset locations of the new voxels
        all_new_gaussian_voxels = self.surface_voxels[repeated_idx]

        # ---------------------------------------
        # Create and remap
        # ---------------------------------------

        filtered_gaussians = filtered_gaussians.filter(all_new_gaussian_indices)
        filtered_voxel_pos = world_to_voxel(
            filtered_gaussians.means, bbox_min, bbox_max, self.voxel_res
        )
        rel_pos = filtered_voxel_pos - filtered_voxel_pos.int()

        new_pos = rel_pos + all_new_gaussian_voxels
        new_pos = voxel_to_world(new_pos, bbox_min, bbox_max, self.voxel_res)
        filtered_gaussians.means = new_pos

        return filtered_gaussians

    @torch.no_grad()
    def upsample(self, ratio: int = 2) -> GrownVoxels:
        assert self.voxel_res_idx > 0 and ratio == 2

        # Create new coords
        blob_coords = self._all_subvoxels(ratio=ratio)
        new_coords = ratio * torch.repeat_interleave(
            self.surface_voxels, blob_coords.shape[0], dim=0
        ) + blob_coords.repeat((self.surface_voxels.shape[0], 1))

        # Create new features
        new_features = torch.repeat_interleave(
            self.latents, blob_coords.shape[0], dim=0
        )

        new_coord_in_ref = ratio * torch.repeat_interleave(
            self.coord_in_ref, blob_coords.shape[0], dim=0
        ) + blob_coords.repeat((self.coord_in_ref.shape[0], 1))

        return GrownVoxels(
            voxel_res=RESOLUTIONS[self.voxel_res_idx - 1],
            voxel_res_idx=self.voxel_res_idx - 1,
            surface_voxels=new_coords,
            latents=new_features,
            coord_in_ref=new_coord_in_ref,
            latents_matched=None,
            get_transform=self.get_transform,
            pca_to_rgb=self.pca_to_rgb,
            bbox_min=self.bbox_min,
            bbox_max=self.bbox_max,
        )

    def get_voxel_set(
        self,
        ref_coord: torch.Tensor,
        ref_feat: torch.Tensor,
        latent_mode: LatentMode,
    ) -> VoxelSet:
        if latent_mode == LatentMode.RAW:
            latents = self.latents
        elif latent_mode == LatentMode.NN_single:
            _, new_feats, _ = exact_search(
                state_coord=self.surface_voxels,
                state_feat=self.latents,
                ref_coord=ref_coord,
                ref_feat=ref_feat,
                # Simple 1-NN
                patch_parameters=PatchParameters(patch_size=1, patch_iters=1),
            )
            latents = new_feats[-1]
        else:
            latents = (
                self.latents_matched
                if self.latents_matched is not None
                else self.latents
            )
        return VoxelSet(
            voxels=self.surface_voxels,
            get_transform=self.get_transform,
            res=self.voxel_res,
            rgb=self.pca_to_rgb(self.voxel_res_idx, latents),
            voxel_edge_width=0.0,
            prefix="trajectory",
            bbox_min=self.bbox_min,
            bbox_max=self.bbox_max,
            enabled=True,
        )
