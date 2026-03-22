from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Type, Union, Any
import math
import torch
import torch.nn as nn
import numpy as np

from gsplat._torch_impl import quat_to_rotmat
from gsplat.project_gaussians import project_gaussians
from gsplat.rasterize import rasterize_gaussians
from gsplat.sh import num_sh_bases, spherical_harmonics

from sprim.configs.base import BaseConfig
from sprim.utils.optim import build_quantizer
from sprim.utils.viewer_utils import RenderPCA
from sprim.utils.process_utils import rotmat_to_quat, transform_shs
from sprim.gaussians.envmap import EnvMap

from nerfstudio.cameras.cameras import Cameras
from nerfstudio.engine.optimizers import Optimizers


def random_quat_tensor(N):
    """
    Defines a random quaternion tensor of shape (N, 4)
    """
    u = torch.rand(N)
    v = torch.rand(N)
    w = torch.rand(N)
    return torch.stack(
        [
            torch.sqrt(1 - u) * torch.sin(2 * math.pi * v),
            torch.sqrt(1 - u) * torch.cos(2 * math.pi * v),
            torch.sqrt(u) * torch.sin(2 * math.pi * w),
            torch.sqrt(u) * torch.cos(2 * math.pi * w),
        ],
        dim=-1,
    )


def RGB2SH(rgb):
    """
    Converts from RGB values [0,1] to the 0th spherical harmonic coefficient
    """
    C0 = 0.28209479177387814
    return (rgb - 0.5) / C0


def SH2RGB(sh):
    """
    Converts from the 0th spherical harmonic coefficient to RGB values [0,1]
    """
    C0 = 0.28209479177387814
    return sh * C0 + 0.5


MAX_DEPTH = 10.0


@dataclass(kw_only=True, frozen=False)
class GaussianSet:
    means: torch.Tensor
    features_dc: torch.Tensor
    features_rest: torch.Tensor
    features_feat: torch.Tensor
    opacities: torch.Tensor
    scales: torch.Tensor
    quats: torch.Tensor

    @torch.no_grad()
    def clip_scale(self, max_scale: float):
        self.scales = torch.clip(self.scales, None, math.log(max_scale))

    @torch.no_grad()
    def filter(self, filter_mask) -> GaussianSet:
        return GaussianSet(
            means=self.means[filter_mask],
            features_dc=self.features_dc[filter_mask],
            features_rest=self.features_rest[filter_mask],
            features_feat=self.features_feat[filter_mask],
            opacities=self.opacities[filter_mask],
            scales=self.scales[filter_mask],
            quats=self.quats[filter_mask],
        )

    @torch.no_grad()
    def merge(self, other: "GaussianSet") -> GaussianSet:
        return GaussianSet(
            means=torch.cat([self.means, other.means], dim=0),
            features_dc=torch.cat([self.features_dc, other.features_dc], dim=0),
            features_rest=torch.cat([self.features_rest, other.features_rest], dim=0),
            features_feat=torch.cat([self.features_feat, other.features_feat], dim=0),
            opacities=torch.cat([self.opacities, other.opacities], dim=0),
            scales=torch.cat([self.scales, other.scales], dim=0),
            quats=torch.cat([self.quats, other.quats], dim=0),
        )

    @torch.no_grad()
    def transform(self, transform: torch.Tensor | None = None) -> GaussianSet:

        if transform is not None:

            # Positions
            transformed_pos = torch.cat(
                [
                    self.means,
                    torch.ones((self.means.shape[0], 1), device=self.means.device),
                ],
                dim=-1,
            )
            transformed_pos = torch.matmul(transform, transformed_pos.T).T
            transformed_pos = transformed_pos[:, :3] / transformed_pos[:, 3][:, None]

            # Rotations (remove scaling!)
            transform_rot = transform[:3, :3]
            transform_scale = torch.pow(torch.det(transform_rot), 1.0 / 3.0)
            transform_rot = transform_rot / transform_scale

            rotmat = quat_to_rotmat(self.quats)
            rotmat = torch.bmm(
                transform_rot.unsqueeze(0).repeat((len(transformed_pos), 1, 1)), rotmat
            )

            transformed_quats = rotmat_to_quat(rotmat)

            transformed_scales = self.scales + torch.log(transform_scale)

            transformed_features_rest = transform_shs(self.features_rest, transform_rot)

            return GaussianSet(
                means=transformed_pos,
                features_dc=self.features_dc,
                features_rest=transformed_features_rest,
                features_feat=self.features_feat,
                opacities=self.opacities,
                scales=transformed_scales,
                quats=transformed_quats,
            )

        else:
            return GaussianSet(
                means=self.means,
                features_dc=self.features_dc,
                features_rest=self.features_rest,
                features_feat=self.features_feat,
                opacities=self.opacities,
                scales=self.scales,
                quats=self.quats,
            )

    def __len__(self):
        return len(self.means)

    def serialize(self) -> Dict[str, Any]:
        return {
            "means": self.means.cpu().numpy(),
            "features_dc": self.features_dc.cpu().numpy(),
            "features_rest": self.features_rest.cpu().numpy(),
            "features_feat": self.features_feat.cpu().numpy(),
            "opacities": self.opacities.cpu().numpy(),
            "scales": self.scales.cpu().numpy(),
            "quats": self.quats.cpu().numpy(),
        }

    @staticmethod
    def deserialize(data: Dict[str, Any]) -> GaussianSet:
        return GaussianSet(
            means=torch.tensor(data["means"]).cuda(),
            features_dc=torch.tensor(data["features_dc"]).cuda(),
            features_rest=torch.tensor(data["features_rest"]).cuda(),
            features_feat=torch.tensor(data["features_feat"]).cuda(),
            opacities=torch.tensor(data["opacities"]).cuda(),
            scales=torch.tensor(data["scales"]).cuda(),
            quats=torch.tensor(data["quats"]).cuda(),
        )


# This is deliberately adapted from "splatfacto"
# https://github.com/nerfstudio-project/nerfstudio/blob/main/nerfstudio/models/splatfacto.py
# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


class GaussianModel(nn.Module):

    def __init__(
        self,
        device,
        num_train_data: int,
        sh_degree: int = 3,
        feature_dim: int = 8,
        num_random: int = 50000,
        random_scale: float = 10.0,
        seed_points: Tuple[torch.Tensor, torch.Tensor] | None = None,
        random_init: bool = False,
        feature_quantizer: dict = {"type": "none"},
        render_pca: RenderPCA | None = None,
    ):
        super().__init__()
        self.device = device
        self.seed_points = seed_points
        self.sh_degree = sh_degree
        self.features_dim = feature_dim
        self.random_scale = random_scale
        self.random_init = random_init
        self.num_random = num_random
        self.num_train_data = num_train_data

        self.warmup_length: int = 500
        """period of steps where refinement is turned off"""
        self.refine_every: int = 100
        """period of steps where gaussians are culled and densified"""
        self.resolution_schedule: int = 3000
        """training starts at 1/d resolution, every n steps this is doubled"""
        self.num_downscales: int = 2
        """at the beginning, resolution is 1/2^d, where d is this number"""
        self.cull_alpha_thresh: float = 0.1
        """threshold of opacity for culling gaussians. One can set it to a lower value (e.g. 0.005) for higher quality."""
        self.cull_scale_thresh: float = 0.5  # Original is 0.5
        """threshold of scale for culling huge gaussians"""
        self.continue_cull_post_densification: bool = True
        """If True, continue to cull gaussians post refinement"""
        self.reset_alpha_every: int = 30
        """Every this many refinement steps, reset the alpha"""
        self.densify_grad_thresh: float = 0.0002
        """threshold of positional gradient norm for densifying gaussians"""
        self.densify_size_thresh: float = 0.01  # Original is 0.01
        """below this size, gaussians are *duplicated*, otherwise split"""
        self.n_split_samples: int = 2
        """number of samples to split gaussians into"""
        self.sh_degree_interval: int = 1000
        """every n intervals turn on another sh degree"""
        self.cull_screen_size: float = 0.15  # Original is 0.15
        """if a gaussian is more than this percent of screen space, cull it"""
        self.split_screen_size: float = 0.05
        """if a gaussian is more than this percent of screen space, split it"""
        self.stop_screen_size_at: int = 4000
        """stop culling/splitting at this step WRT screen size of gaussians"""
        self.stop_split_at: int = 15000
        """stop splitting at this step"""
        self.max_gaussians: int = 2500000

        if self.seed_points is not None and not self.random_init:
            means = torch.nn.Parameter(self.seed_points[0])  # (Location, Color)
        else:
            means = torch.nn.Parameter(
                (torch.rand((self.num_random, 3)) - 0.5) * self.random_scale
            )
        self.xys_grad_norm = None
        self.max_2Dsize = None
        distances, _ = self.k_nearest_sklearn(means.data, 3)
        distances = torch.from_numpy(distances)
        # find the average of the three nearest neighbors for each point and use that as the scale
        avg_dist = distances.mean(dim=-1, keepdim=True)
        scales = torch.nn.Parameter(torch.clip(torch.log(avg_dist.repeat(1, 3)), -8))
        num_points = means.shape[0]
        quats = torch.nn.Parameter(random_quat_tensor(num_points))
        dim_sh = num_sh_bases(self.sh_degree)

        if (
            self.seed_points is not None
            and not self.random_init
            # We can have colors without points.
            and self.seed_points[1].shape[0] > 0
        ):
            shs = torch.zeros((self.seed_points[1].shape[0], dim_sh, 3)).float().cuda()
            if self.sh_degree > 0:
                shs[:, 0, :3] = RGB2SH(self.seed_points[1] / 255)
                shs[:, 1:, 3:] = 0.0
            else:
                print("use color only optimization with sigmoid activation")
                shs[:, 0, :3] = torch.logit(self.seed_points[1] / 255, eps=1e-10)
            features_dc = torch.nn.Parameter(shs[:, 0, :])
            features_rest = torch.nn.Parameter(shs[:, 1:, :])
            features_feat = torch.nn.Parameter(
                torch.randn(num_points, self.features_dim)
            )
        else:
            features_dc = torch.nn.Parameter(torch.rand(num_points, 3))
            features_rest = torch.nn.Parameter(torch.zeros((num_points, dim_sh - 1, 3)))
            features_feat = torch.nn.Parameter(
                torch.randn(num_points, self.features_dim)
            )

        opacities = torch.nn.Parameter(torch.logit(0.1 * torch.ones(num_points, 1)))
        self.gauss_params = torch.nn.ParameterDict(
            {
                "means": means,
                "scales": scales,
                "quats": quats,
                "features_dc": features_dc,
                "features_rest": features_rest,
                "features_feat": features_feat,
                "opacities": opacities,
            }
        )
        self.grown_gauss_params = None

        self.step = 0

        self.background_color = torch.tensor([0.0, 0.0, 0.0]).cuda()
        self.background_depth = torch.tensor([MAX_DEPTH]).cuda()
        self.background_feat = torch.zeros(self.features_dim).cuda()

        self.envmap: EnvMap | None = None

        # Feature quantizer
        self.feature_quantizer = (
            None
            if feature_quantizer["type"] == "none"
            else build_quantizer(feature_quantizer)
        )

        if render_pca is None:
            self.render_pca = RenderPCA.default(self.features_dim)
        else:
            self.render_pca = render_pca

    @property
    def num_points(self):
        return self.means.shape[0]

    @property
    def means(self):
        return self.gauss_params["means"]

    @property
    def scales(self):
        return self.gauss_params["scales"]

    @property
    def quats(self):
        return self.gauss_params["quats"]

    @property
    def features_dc(self):
        return self.gauss_params["features_dc"]

    @property
    def features_rest(self):
        return self.gauss_params["features_rest"]

    @property
    def features_feat(self):
        return self.gauss_params["features_feat"]

    @property
    def opacities(self):
        return self.gauss_params["opacities"]

    def get_gaussian_set(self, grown: bool = False) -> GaussianSet:
        if grown:
            return self.grown_gauss_params
        else:
            return GaussianSet(
                means=self.means,
                features_dc=self.features_dc,
                features_rest=self.features_rest,
                features_feat=self.features_feat,
                opacities=self.opacities,
                scales=self.scales,
                quats=self.quats,
            )

    def set_gaussian_set(self, gaussian_set: GaussianSet, grown: bool = False) -> None:
        if grown:
            self.grown_gauss_params = gaussian_set
        else:
            self.gauss_params = torch.nn.ParameterDict(
                {
                    "means": gaussian_set.means,
                    "scales": gaussian_set.scales,
                    "quats": gaussian_set.quats,
                    "features_dc": gaussian_set.features_dc,
                    "features_rest": gaussian_set.features_rest,
                    "features_feat": gaussian_set.features_feat,
                    "opacities": gaussian_set.opacities,
                }
            )

    def k_nearest_sklearn(self, x: torch.Tensor, k: int):
        """
            Find k-nearest neighbors using sklearn's NearestNeighbors.
        x: The data tensor of shape [num_samples, num_features]
        k: The number of neighbors to retrieve
        """
        # Convert tensor to numpy array
        x_np = x.cpu().numpy()

        # Build the nearest neighbors model
        from sklearn.neighbors import NearestNeighbors

        nn_model = NearestNeighbors(
            n_neighbors=k + 1, algorithm="auto", metric="euclidean"
        ).fit(x_np)

        # Find the k-nearest neighbors
        distances, indices = nn_model.kneighbors(x_np)

        # Exclude the point itself from the result and return
        return distances[:, 1:].astype(np.float32), indices[:, 1:].astype(np.float32)

    def get_gaussian_param_groups(self) -> Dict[str, List[torch.nn.Parameter]]:
        # Here we explicitly use the means, scales as parameters so that the user can override this function and
        # specify more if they want to add more optimizable params to gaussians.
        return {
            name: [self.gauss_params[name]]
            for name in [
                "means",
                "scales",
                "quats",
                "features_dc",
                "features_rest",
                "features_feat",
                "opacities",
            ]
        }

    def get_all_param_groups(self) -> Dict[str, List[torch.nn.Parameter]]:
        param_groups = self.get_gaussian_param_groups()
        # if self.envmap is not None:
        #     param_groups["envmap"] = [self.envmap]

        return param_groups

    def _get_downscale_factor(self):
        if self.training:
            return 2 ** max(
                (self.num_downscales - self.step // self.resolution_schedule),
                0,
            )
        else:
            return 1

    def _downscale_if_required(self, image):
        d = self._get_downscale_factor()
        if d > 1:
            newsize = [image.shape[0] // d, image.shape[1] // d]

            # torchvision can be slow to import, so we do it lazily.
            import torchvision.transforms.functional as TF

            return TF.resize(image.permute(2, 0, 1), newsize, antialias=None).permute(
                1, 2, 0
            )
        return image

    def render(
        self,
        camera: Cameras,
        config: BaseConfig,
        return_feat: bool = True,
        return_depth: bool = False,
        render_envmap: bool = True,
        mask: torch.Tensor | None = None,
        background_color: torch.Tensor | None = None,
        background_feat: torch.Tensor | None = None,
        max_scale: float = 10.0,
        render_grown: bool = False,
    ) -> Dict[str, Union[torch.Tensor, List]]:
        """Takes in a Ray Bundle and returns a dictionary of outputs.

        Args:
            ray_bundle: Input bundle of rays. This raybundle should have all the
            needed information to compute the outputs.

        Returns:
            Outputs of model. (ie. rendered colors)
        """
        if not isinstance(camera, Cameras):
            print("Called get_outputs with not a camera")
            return {}
        # assert camera.shape[0] == 1, "Only one camera at a time"

        # get the background color (override if provided)
        background = (
            self.background_color if self.background_color is None else background_color
        )

        # Downscale (training only)
        camera_downscale = self._get_downscale_factor()
        camera.rescale_output_resolution(1 / camera_downscale)

        # shift the camera to center of scene looking at center
        R = camera.camera_to_worlds[:3, :3]  # 3 x 3
        T = camera.camera_to_worlds[:3, 3:4]  # 3 x 1

        # flip the z and y axes to align with gsplat conventions
        R_edit = torch.diag(torch.tensor([1, -1, -1], device=R.device, dtype=R.dtype))
        R = R @ R_edit
        # analytic matrix inverse to get world2camera matrix
        R_inv = R.T
        T_inv = -R_inv @ T
        viewmat = torch.eye(4, device=R.device, dtype=R.dtype)
        viewmat[:3, :3] = R_inv
        viewmat[:3, 3:4] = T_inv
        # calculate the FOV of the camera given fx and fy, width and height
        cx = camera.cx.item()
        cy = camera.cy.item()
        W, H = int(camera.width.item()), int(camera.height.item())
        W_dino, H_dino = int(
            camera.width.item() * config.factor / config.factor_dino
        ), int(
            camera.height.item() * config.factor / config.factor_dino,
        )
        self.last_size = (H, W)

        if render_grown:
            opacities_crop = self.grown_gauss_params.opacities
            means_crop = self.grown_gauss_params.means
            features_dc_crop = self.grown_gauss_params.features_dc
            features_rest_crop = self.grown_gauss_params.features_rest
            scales_crop = self.grown_gauss_params.scales
            quats_crop = self.grown_gauss_params.quats
            features_feat_crop = self.grown_gauss_params.features_feat
        else:
            if mask is not None:
                opacities_crop = self.opacities[mask]
                means_crop = self.means[mask]
                features_dc_crop = self.features_dc[mask]
                features_rest_crop = self.features_rest[mask]
                scales_crop = self.scales[mask]
                quats_crop = self.quats[mask]
                features_feat_crop = self.features_feat[mask]
            else:
                opacities_crop = self.opacities
                means_crop = self.means
                features_dc_crop = self.features_dc
                features_rest_crop = self.features_rest
                scales_crop = self.scales
                quats_crop = self.quats
                features_feat_crop = self.features_feat

        colors_crop = torch.cat(
            (features_dc_crop[:, None, :], features_rest_crop), dim=1
        )

        def return_empty():
            rgb = background.repeat(H, W, 1)
            accumulation = background.new_zeros(*rgb.shape[:2], 1)

            if self.envmap is not None and render_envmap:
                rays = camera.generate_rays(0)
                theta = torch.acos(rays.directions[..., 2:3]) / (math.pi / 2.0) - 1.0
                phi = (
                    torch.fmod(
                        torch.atan2(
                            rays.directions[..., 1:2], rays.directions[..., 0:1]
                        )
                        / math.pi
                        + 2.0,
                        2.0,
                    )
                    - 1.0
                )

                projected_envmap = self.envmap.sample(phi=phi, theta=theta)
                rgb = projected_envmap
                accumulation[:] = 1.0

            results = {
                "rgb": rgb,
                "alpha": accumulation,
                "background": background,
            }

            if self.features_dim > 0 and return_feat:
                results["feat"] = self.background_feat.repeat(H, W, 1)

            if return_depth:
                results["depth"] = self.background_depth.repeat(H, W, 1)

            return results

        if len(means_crop) == 0:
            return return_empty()

        BLOCK_WIDTH = (
            16  # this controls the tile size of rasterization, 16 is a good default
        )

        self.xys, depths, self.radii, conics, comp, num_tiles_hit, cov3d = project_gaussians(  # type: ignore
            means_crop,
            torch.clip(torch.exp(scales_crop), 0, max_scale),
            1,
            quats_crop / quats_crop.norm(dim=-1, keepdim=True),
            viewmat.squeeze()[:3, :],
            camera.fx.item(),
            camera.fy.item(),
            cx,
            cy,
            H,
            W,
            BLOCK_WIDTH,
        )  # type: ignore

        # rescale the camera back to original dimensions before returning
        camera.rescale_output_resolution(camera_downscale)

        if (self.radii).sum() == 0:
            return return_empty()

        # Important to allow xys grads to populate properly
        if self.training:
            self.xys.retain_grad()

        if self.sh_degree > 0:
            viewdirs = (
                means_crop.detach() - camera.camera_to_worlds.detach()[:3, 3]
            )  # (N, 3)
            viewdirs = viewdirs / viewdirs.norm(dim=-1, keepdim=True)
            n = min(self.step // config.sh_degree_interval, config.sh_degree)
            rgbs = spherical_harmonics(n, viewdirs, colors_crop)
            rgbs = torch.clamp(rgbs + 0.5, min=0.0)  # type: ignore
        else:
            rgbs = torch.sigmoid(colors_crop[:, 0, :])

        assert (num_tiles_hit > 0).any()  # type: ignore

        # apply the compensation of screen space blurring to gaussians
        # opacities = None
        # if self.config.rasterize_mode == "antialiased":
        #     opacities = torch.sigmoid(opacities_crop) * comp[:, None]
        # elif self.config.rasterize_mode == "classic":
        opacities = torch.sigmoid(opacities_crop)
        # else:
        #     raise ValueError("Unknown rasterize_mode: %s", self.config.rasterize_mode)

        rgb, alpha = rasterize_gaussians(  # type: ignore
            self.xys,
            depths,
            self.radii,
            conics,
            num_tiles_hit,  # type: ignore
            torch.cat([rgbs, depths.unsqueeze(-1)], dim=-1) if return_depth else rgbs,
            opacities,
            H,
            W,
            BLOCK_WIDTH,
            background=(
                torch.cat([background, self.background_depth], dim=-1)
                if return_depth
                else background
            ),
            return_alpha=True,
        )  # type: ignore
        rgb, depth = rgb.split([3, 1], dim=-1) if return_depth else (rgb, None)
        alpha = alpha[..., None]

        if self.envmap is not None and render_envmap:
            rays = camera.generate_rays(0)
            theta = torch.acos(rays.directions[..., 2:3]) / (math.pi / 2.0) - 1.0
            phi = (
                torch.fmod(
                    torch.atan2(rays.directions[..., 1:2], rays.directions[..., 0:1])
                    / math.pi
                    + 2.0,
                    2.0,
                )
                - 1.0
            )

            projected_envmap = self.envmap.sample(phi=phi, theta=theta)

            # Transparency no more!
            rgb = alpha * rgb + (1 - alpha) * projected_envmap
            alpha = torch.ones_like(alpha)

        rgb = torch.clamp(rgb, max=1.0)  # type: ignore

        results = {"rgb": rgb, "depth": depth, "alpha": alpha, "background": background}  # type: ignore

        if self.features_dim > 0 and return_feat:
            background_feat = (
                self.background_feat if background_feat is None else background_feat
            )

            feat = rasterize_gaussians(  # type: ignore
                self.xys.detach(),
                depths.detach(),
                self.radii.detach(),
                conics.detach(),
                num_tiles_hit,  # type: ignore
                features_feat_crop,
                opacities.detach(),
                H,
                W,
                BLOCK_WIDTH,
                background=background_feat,
                return_alpha=False,
            )  # type: ignore

            results["feat"] = feat

        return results

    def after_train(self, step: int):
        assert step == self.step
        # to save some training time, we no longer need to update those stats post refinement
        if self.step >= self.stop_split_at:
            return
        with torch.no_grad():
            # keep track of a moving average of grad norms
            visible_mask = (self.radii > 0).flatten()
            assert self.xys.grad is not None
            grads = self.xys.grad.detach().norm(dim=-1)
            # print(f"grad norm min {grads.min().item()} max {grads.max().item()} mean {grads.mean().item()} size {grads.shape}")
            if self.xys_grad_norm is None:
                self.xys_grad_norm = grads
                self.vis_counts = torch.ones_like(self.xys_grad_norm)
            else:
                assert self.vis_counts is not None
                self.vis_counts[visible_mask] = self.vis_counts[visible_mask] + 1
                self.xys_grad_norm[visible_mask] = (
                    grads[visible_mask] + self.xys_grad_norm[visible_mask]
                )

            # update the max screen size, as a ratio of number of pixels
            if self.max_2Dsize is None:
                self.max_2Dsize = torch.zeros_like(self.radii, dtype=torch.float32)
            newradii = self.radii.detach()[visible_mask]
            self.max_2Dsize[visible_mask] = torch.maximum(
                self.max_2Dsize[visible_mask],
                newradii / float(max(self.last_size[0], self.last_size[1])),
            )

    def refinement_after(self, optimizers: Optimizers, step):
        assert step == self.step
        if self.step <= self.warmup_length:
            return
        with torch.no_grad():
            # Offset all the opacity reset logic by refine_every so that we don't
            # save checkpoints right when the opacity is reset (saves every 2k)
            # then cull
            # only split/cull if we've seen every image since opacity reset
            reset_interval = self.reset_alpha_every * self.refine_every
            do_densification = (
                self.step < self.stop_split_at
                and self.step % reset_interval > self.num_train_data + self.refine_every
            )
            # Added to make sure the number of Gaussians doesn't blow up
            too_many_points = self.means.shape[0] > self.max_gaussians

            if do_densification and not too_many_points:
                # then we densify
                assert (
                    self.xys_grad_norm is not None
                    and self.vis_counts is not None
                    and self.max_2Dsize is not None
                )
                avg_grad_norm = (
                    (self.xys_grad_norm / self.vis_counts)
                    * 0.5
                    * max(self.last_size[0], self.last_size[1])
                )
                high_grads = (avg_grad_norm > self.densify_grad_thresh).squeeze()
                splits = (
                    self.scales.exp().max(dim=-1).values > self.densify_size_thresh
                ).squeeze()
                if self.step < self.stop_screen_size_at:
                    splits |= (self.max_2Dsize > self.split_screen_size).squeeze()
                splits &= high_grads
                nsamps = self.n_split_samples
                split_params = self.split_gaussians(splits, nsamps)

                dups = (
                    self.scales.exp().max(dim=-1).values <= self.densify_size_thresh
                ).squeeze()
                dups &= high_grads
                dup_params = self.dup_gaussians(dups)
                for name, param in self.gauss_params.items():
                    self.gauss_params[name] = torch.nn.Parameter(
                        torch.cat(
                            [param.detach(), split_params[name], dup_params[name]],
                            dim=0,
                        )
                    )

                # append zeros to the max_2Dsize tensor
                self.max_2Dsize = torch.cat(
                    [
                        self.max_2Dsize,
                        torch.zeros_like(split_params["scales"][:, 0]),
                        torch.zeros_like(dup_params["scales"][:, 0]),
                    ],
                    dim=0,
                )

                split_idcs = torch.where(splits)[0]
                self.dup_in_all_optim(optimizers, split_idcs, nsamps)

                dup_idcs = torch.where(dups)[0]
                self.dup_in_all_optim(optimizers, dup_idcs, 1)

                # After a guassian is split into two new gaussians, the original one should also be pruned.
                splits_mask = torch.cat(
                    (
                        splits,
                        torch.zeros(
                            nsamps * splits.sum() + dups.sum(),
                            device=self.device,
                            dtype=torch.bool,
                        ),
                    )
                )

                deleted_mask = self.cull_gaussians(splits_mask)
            elif (
                self.step >= self.stop_split_at
                and self.continue_cull_post_densification
            ) or (do_densification and too_many_points):
                deleted_mask = self.cull_gaussians()
            else:
                # if we donot allow culling post refinement, no more gaussians will be pruned.
                deleted_mask = None

            if deleted_mask is not None:
                self.remove_from_all_optim(optimizers, deleted_mask)

            if (
                self.step < self.stop_split_at
                and self.step % reset_interval == self.refine_every
            ):
                # Reset value is set to be twice of the cull_alpha_thresh
                reset_value = self.cull_alpha_thresh * 2.0
                self.opacities.data = torch.clamp(
                    self.opacities.data,
                    max=torch.logit(
                        torch.tensor(reset_value, device=self.device)
                    ).item(),
                )
                # reset the exp of optimizer
                optim = optimizers.optimizers["opacities"]
                param = optim.param_groups[0]["params"][0]
                param_state = optim.state[param]
                param_state["exp_avg"] = torch.zeros_like(param_state["exp_avg"])
                param_state["exp_avg_sq"] = torch.zeros_like(param_state["exp_avg_sq"])

            self.xys_grad_norm = None
            self.vis_counts = None
            self.max_2Dsize = None

    def cull_gaussians(self, extra_cull_mask: Optional[torch.Tensor] = None):
        """
        This function deletes gaussians with under a certain opacity threshold
        extra_cull_mask: a mask indicates extra gaussians to cull besides existing culling criterion
        """
        n_bef = self.num_points
        # cull transparent ones
        culls = (torch.sigmoid(self.opacities) < self.cull_alpha_thresh).squeeze()
        below_alpha_count = torch.sum(culls).item()
        toobigs_count = 0
        if extra_cull_mask is not None:
            culls = culls | extra_cull_mask
        if self.step > self.refine_every * self.reset_alpha_every:
            # cull huge ones
            toobigs = (
                torch.exp(self.scales).max(dim=-1).values > self.cull_scale_thresh
            ).squeeze()
            if self.step < self.stop_screen_size_at:
                # cull big screen space
                assert self.max_2Dsize is not None
                toobigs = toobigs | (self.max_2Dsize > self.cull_screen_size).squeeze()
            culls = culls | toobigs
            toobigs_count = torch.sum(toobigs).item()
        for name, param in self.gauss_params.items():
            self.gauss_params[name] = torch.nn.Parameter(param[~culls])

        print(
            f"Culled {n_bef - self.num_points} gaussians "
            f"({below_alpha_count} below alpha thresh, {toobigs_count} too bigs, {self.num_points} remaining)"
        )

        return culls

    def split_gaussians(self, split_mask, samps):
        """
        This function splits gaussians that are too large
        """
        n_splits = split_mask.sum().item()
        print(
            f"Splitting {split_mask.sum().item()/self.num_points} gaussians: {n_splits}/{self.num_points}"
        )
        centered_samples = torch.randn(
            (samps * n_splits, 3), device=self.device
        )  # Nx3 of axis-aligned scales
        scaled_samples = (
            torch.exp(self.scales[split_mask].repeat(samps, 1)) * centered_samples
        )  # how these scales are rotated
        quats = self.quats[split_mask] / self.quats[split_mask].norm(
            dim=-1, keepdim=True
        )  # normalize them first
        rots = quat_to_rotmat(quats.repeat(samps, 1))  # how these scales are rotated
        rotated_samples = torch.bmm(rots, scaled_samples[..., None]).squeeze()
        new_means = rotated_samples + self.means[split_mask].repeat(samps, 1)
        # step 2, sample new colors
        new_features_dc = self.features_dc[split_mask].repeat(samps, 1)
        new_features_rest = self.features_rest[split_mask].repeat(samps, 1, 1)
        new_features_feat = self.features_feat[split_mask].repeat(samps, 1)
        # step 3, sample new opacities
        new_opacities = self.opacities[split_mask].repeat(samps, 1)
        # step 4, sample new scales
        size_fac = 1.6
        new_scales = torch.log(torch.exp(self.scales[split_mask]) / size_fac).repeat(
            samps, 1
        )
        self.scales[split_mask] = torch.log(
            torch.exp(self.scales[split_mask]) / size_fac
        )
        # step 5, sample new quats
        new_quats = self.quats[split_mask].repeat(samps, 1)
        out = {
            "means": new_means,
            "features_dc": new_features_dc,
            "features_rest": new_features_rest,
            "features_feat": new_features_feat,
            "opacities": new_opacities,
            "scales": new_scales,
            "quats": new_quats,
        }
        for name, param in self.gauss_params.items():
            if name not in out:
                out[name] = param[split_mask].repeat(samps, 1)
        return out

    def dup_gaussians(self, dup_mask):
        """
        This function duplicates gaussians that are too small
        """
        n_dups = dup_mask.sum().item()
        print(
            f"Duplicating {dup_mask.sum().item()/self.num_points} gaussians: {n_dups}/{self.num_points}"
        )
        new_dups = {}
        for name, param in self.gauss_params.items():
            new_dups[name] = param[dup_mask]
        return new_dups

    def remove_from_optim(self, optimizer, deleted_mask, new_params):
        """removes the deleted_mask from the optimizer provided"""
        assert len(new_params) == 1
        # assert isinstance(optimizer, torch.optim.Adam), "Only works with Adam"

        param = optimizer.param_groups[0]["params"][0]
        param_state = optimizer.state[param]
        del optimizer.state[param]

        # Modify the state directly without deleting and reassigning.
        if "exp_avg" in param_state:
            param_state["exp_avg"] = param_state["exp_avg"][~deleted_mask]
            param_state["exp_avg_sq"] = param_state["exp_avg_sq"][~deleted_mask]

        # Update the parameter in the optimizer's param group.
        del optimizer.param_groups[0]["params"][0]
        del optimizer.param_groups[0]["params"]
        optimizer.param_groups[0]["params"] = new_params
        optimizer.state[new_params[0]] = param_state

    def remove_from_all_optim(self, optimizers, deleted_mask):
        param_groups = self.get_gaussian_param_groups()
        for group, param in param_groups.items():
            self.remove_from_optim(optimizers.optimizers[group], deleted_mask, param)
        torch.cuda.empty_cache()

    def dup_in_optim(self, optimizer, dup_mask, new_params, n=2):
        """adds the parameters to the optimizer"""
        param = optimizer.param_groups[0]["params"][0]
        param_state = optimizer.state[param]
        if "exp_avg" in param_state:
            repeat_dims = (n,) + tuple(
                1 for _ in range(param_state["exp_avg"].dim() - 1)
            )
            param_state["exp_avg"] = torch.cat(
                [
                    param_state["exp_avg"],
                    torch.zeros_like(param_state["exp_avg"][dup_mask.squeeze()]).repeat(
                        *repeat_dims
                    ),
                ],
                dim=0,
            )
            param_state["exp_avg_sq"] = torch.cat(
                [
                    param_state["exp_avg_sq"],
                    torch.zeros_like(
                        param_state["exp_avg_sq"][dup_mask.squeeze()]
                    ).repeat(*repeat_dims),
                ],
                dim=0,
            )
        del optimizer.state[param]
        optimizer.state[new_params[0]] = param_state
        optimizer.param_groups[0]["params"] = new_params
        del param

    def dup_in_all_optim(self, optimizers, dup_mask, n):
        param_groups = self.get_gaussian_param_groups()
        for group, param in param_groups.items():
            self.dup_in_optim(optimizers.optimizers[group], dup_mask, param, n)
