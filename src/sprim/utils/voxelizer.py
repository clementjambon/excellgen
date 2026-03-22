import torch
import numpy as np

import trimesh
import point_cloud_utils as pcu

import polyscope.imgui as psim

from fast_gca.utils.window.voxel_set import VoxelSet

from sprim.utils.process_utils import filter_bbox


class Voxelizer:
    def __init__(self, vertices, faces, ps_mesh=None) -> None:
        self.vertices = vertices
        self.faces = faces
        self.ps_mesh = ps_mesh
        self.surface_voxels, self.coarse_voxels = None, None

    def _sample_surface(self, n: int = 100000):
        transformed_vertices = np.concatenate(
            [self.vertices, np.ones((self.vertices.shape[0], 1))], axis=-1
        )

        if self.ps_mesh is not None:

            transformed_vertices = np.matmul(
                self.ps_mesh.get_transform(), transformed_vertices.T
            ).T

        f_i, bc = pcu.sample_mesh_random(transformed_vertices[:, :3], self.faces, n)
        surf_pts = pcu.interpolate_barycentric_coords(
            self.faces, f_i, bc, transformed_vertices[:, :3]
        )
        return surf_pts

    def voxelize(self, res: int, bbox_min: torch.Tensor, bbox_max: torch.Tensor):
        surface_points = self._sample_surface()
        # Remap surface points to [0, 1] (from [-1, 1])
        surface_points = torch.tensor(surface_points, device="cuda")
        surface_points = (surface_points - bbox_min) / (bbox_max - bbox_min)
        surface_points = filter_bbox(
            surface_points,
            bbox_min=torch.zeros(3).to(surface_points),
            bbox_max=torch.ones(3).to(surface_points),
        )
        # Then quantize
        surface_voxels = (surface_points * res).int()
        surface_voxels = torch.unique(surface_voxels, dim=0)  # along dim=0

        return surface_voxels

    @staticmethod
    def downsample(voxels: torch.Tensor, factor: int = 2):

        coarse_voxels = voxels.int() // factor
        coarse_voxels = torch.unique(coarse_voxels, dim=0)

        return coarse_voxels

    def gui(self, bbox_min: torch.Tensor, bbox_max: torch.Tensor):
        if psim.Button("Voxelize"):
            # TODO: de-hardcord these values
            self.surface_voxels = self.voxelize(
                64, bbox_min=bbox_min, bbox_max=bbox_max
            )
            # self.coarse_voxels = self.downsample(self.surface_voxels)
            voxel_set = VoxelSet(
                original_s=self.surface_voxels,
                bbox_min=bbox_min,
                bbox_max=bbox_max,
                cond=None,
                res=64,
            )
            # voxel_set = VoxelSet(
            #     original_s=self.coarse_voxels, cond=None, res=32, prefix="coarse"
            # )

        # if self.surface_voxels is not None:
        #     if psim.Button("Export"):
        #         np.savez(
        #             "voxelized.npz", surface_voxels=self.surface_voxels.cpu().numpy()
        #         )
