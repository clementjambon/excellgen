from typing import Set, Dict

import torch

import polyscope as ps
import polyscope.imgui as psim

from torch_scatter import scatter, scatter_max

from sprim.gaussians.gaussian_model import GaussianModel, SH2RGB
from sprim.configs.base import BaseConfig
from sprim.inverse.grown_voxels import RESOLUTIONS
from sprim.utils.voxel_set import VoxelSet


class DebugVoxelSelection:

    mask: torch.Tensor | None
    gaussian_model: GaussianModel
    config: BaseConfig

    def __init__(self, gaussian_model: GaussianModel, config: BaseConfig):
        self.mask = None
        self.gaussian_model = gaussian_model
        self.config = config
        self.current_voxel = None
        self.res_idx = 1
        self.bbox_min = torch.Tensor(self.config.aabb[:3]).cuda()
        self.bbox_max = torch.Tensor(self.config.aabb[3:]).cuda()
        self.avg_color = None

    @torch.no_grad()
    def compute_surface_voxels(self):
        gaussian_voxels = (
            (self.gaussian_model.means - self.bbox_min)
            / (self.bbox_max - self.bbox_min)
            * RESOLUTIONS[self.res_idx]
        )
        gaussian_voxels = gaussian_voxels.int()
        self.gaussian_voxels, invmap = torch.unique(
            gaussian_voxels, dim=0, return_inverse=True
        )

        self.current_voxel = 0 if len(self.gaussian_voxels) > 0 else None

        voxel_colors_avg = SH2RGB(
            scatter(self.gaussian_model.features_dc, invmap, dim=0, reduce="mean")
        )

        unique_alpha, idx_max = scatter_max(
            self.gaussian_model.opacities, invmap, dim=0
        )

        voxel_colors_max = SH2RGB(self.gaussian_model.features_dc[idx_max.squeeze()])

        VoxelSet(
            self.gaussian_voxels,
            get_transform=lambda: None,
            res=RESOLUTIONS[self.res_idx],
            rgb=voxel_colors_avg,
            prefix="avg",
            voxel_edge_width=0.0,
        )

        VoxelSet(
            self.gaussian_voxels,
            get_transform=lambda: None,
            res=RESOLUTIONS[self.res_idx],
            rgb=voxel_colors_max,
            prefix="max",
            voxel_edge_width=0.0,
        )

    @torch.no_grad()
    def update_mask(self):
        if self.current_voxel is None:
            self.mask = None
            return

        # Update mask
        pos_in_voxels = (
            (self.gaussian_model.means - self.bbox_min)
            / (self.bbox_max - self.bbox_min)
            * RESOLUTIONS[self.res_idx]
        ).int()
        self.mask = torch.all(
            pos_in_voxels == self.gaussian_voxels[self.current_voxel][None, :], dim=1
        )

        voxel_pos = (
            self.gaussian_voxels[self.current_voxel].float() / RESOLUTIONS[self.res_idx]
        ) * (self.bbox_max - self.bbox_min) + self.bbox_min

        cam_pos = voxel_pos + 0.2 * torch.tensor([1.0, 0.0, 1.0]).cuda()

        # ps.look_at(cam_pos.cpu().numpy(), voxel_pos.cpu().numpy(), True)

        # vertices = (
        #     CUBE_VERTICES * (self.bbox_max - self.bbox_min) / RESOLUTIONS[self.res_idx]
        #     + voxel_pos
        # )

        # ps.register_curve_network(
        #     "debug_voxels",
        #     vertices.cpu().numpy(),
        #     CUBE_EDGES_NP,
        # )

        selected_colors = SH2RGB(self.gaussian_model.features_dc)[self.mask]
        self.avg_color = selected_colors.mean(0)

    def postprocess_features(
        self,
        features: torch.Tensor,
        indices: torch.Tensor,
        rendered_features: torch.Tensor,
        render_ratio: float = 1.0,
    ):

        return rendered_features

    def gui(self):
        if psim.TreeNode("Debug Selection"):

            _, self.res_idx = psim.SliderInt(
                "res_idx##debug_selection",
                self.res_idx,
                v_min=0,
                v_max=len(RESOLUTIONS) - 1,
                format=f"{RESOLUTIONS[self.res_idx]}",
            )

            _,

            if psim.Button("Gaussian Voxels##debug_selection"):
                self.compute_surface_voxels()
                self.update_mask()

            if self.current_voxel is not None and psim.Button("Next"):
                self.current_voxel = (self.current_voxel + 1) % len(
                    self.gaussian_voxels
                )
                self.update_mask()

            if self.avg_color is not None:
                psim.ColorEdit3(
                    "avg_color##debug_selection", self.avg_color.cpu().tolist()
                )

            psim.TreePop()
