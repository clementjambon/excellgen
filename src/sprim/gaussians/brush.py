from __future__ import annotations

from dataclasses import dataclass
import os

import torch
import numpy as np

import polyscope as ps
import polyscope.imgui as psim

from sprim.utils.voxelizer import Voxelizer
from sprim.utils.voxel_set import VoxelSet
from sprim.utils.gui_utils import state_button, KEY_HANDLER

N_POINTS_VOXELIZER = 50000


@dataclass
class Brush:

    points: torch.Tensor

    # Optional: in case the brush is a Mesh
    mesh_vertices: torch.Tensor | None
    mesh_faces: torch.Tensor | None

    ps_structure: ps.Structure | None
    ps_transform: np.ndarray | None

    exemplar_brush_mode: bool = True

    def __init__(
        self,
        points,
        primitive_entry,
        mesh_vertices: torch.Tensor | None = None,
        mesh_faces: torch.Tensor | None = None,
        ps_transform: torch.Tensor | None = None,
    ) -> None:
        self.points = points
        self.mesh_vertices = mesh_vertices
        self.mesh_faces = mesh_faces
        assert (self.mesh_vertices is None) == (self.mesh_faces is None)

        self.ps_structure = None
        self.ps_transform = ps_transform
        self.gizmo_translate: bool = True

        self.primitive_entry = primitive_entry
        if self.primitive_entry is not None:
            self.voxels = self.voxelize(self.primitive_entry.grower.coarse_res)

    @property
    def is_mesh(self):
        return self.mesh_vertices is not None

    @property
    def enabled(self):
        return self.ps_structure is not None and self.ps_structure.is_enabled()

    def export_brush(self, path):
        np.savez(path, points=self.points.cpu().numpy())
        print(f"Brush saved at: {os.path.abspath(path)}")

    def display(self):
        if self.is_mesh:
            self.ps_structure = ps.register_surface_mesh(
                "brush",
                vertices=self.mesh_vertices.cpu().numpy(),
                faces=self.mesh_faces.cpu().numpy(),
                enabled=True,
            )
        else:
            if self.exemplar_brush_mode:
                self.ps_structure = VoxelSet(
                    self.voxels,
                    get_transform=self.primitive_entry.get_transform,
                    res=self.primitive_entry.grower.coarse_res,
                    bbox_min=self.primitive_entry.bbox_min,
                    bbox_max=self.primitive_entry.bbox_max,
                    prefix="brush",
                    enabled=True,
                    rgb=torch.tensor([[64.0 / 255.0, 224.0 / 255.0, 208.0 / 255.0]])
                    .cuda()
                    .repeat((len(self.voxels), 1)),
                )
            else:
                self.ps_structure = ps.register_point_cloud(
                    "brush", points=self.points.cpu().numpy(), enabled=True
                )

        if self.ps_transform is not None:
            self.ps_structure.set_transform(self.ps_transform)

        self.ps_structure.enable_transform_gizmo()

    def enable_transform_gizmo(self, enabled: bool = True) -> None:
        if self.ps_structure is not None:
            self.ps_structure.enable_transform_gizmo(enabled)

    def is_enabled_transform_gizmo(self) -> bool:
        if self.ps_structure is not None:
            return self.ps_structure.is_enabled_transform_gizmo()
        return False

    def hide(self):
        if self.ps_structure is not None:
            self.ps_structure.set_enabled(False)
            self.ps_structure.enable_transform_gizmo(False)

    def show(self):
        if self.ps_structure is not None:
            self.ps_structure.set_enabled(True)
            self.ps_structure.enable_transform_gizmo(True)

    def set_transform(self, transform: np.ndarray | None = None):
        if self.ps_structure is not None:
            if transform is None:
                transform = np.eye(4)
            self.ps_structure.set_transform(transform)

    def get_transform(self):
        return self.ps_structure.get_transform()

    def remove(self):
        if self.ps_structure is not None:
            self.ps_structure.remove()
            self.ps_structure = None

    @staticmethod
    def from_mesh(mesh_path: str, primitive_entry) -> Brush:
        import trimesh

        # Load the mesh

        try:
            mesh = trimesh.load(mesh_path, force="mesh")
        except:
            print(f"Could not import mesh at location: {mesh_path}")
            return None

        # Sample the surface to create the brush
        voxelizer = Voxelizer(mesh.vertices, mesh.faces)
        surface_points = voxelizer._sample_surface(50000)

        # Apply centering tranform to match the size
        surface_points_max = surface_points.max(0)
        surface_points_min = surface_points.min(0)
        surface_points_middle = 0.5 * (surface_points_min + surface_points_max)

        surface_point_scale = (surface_points_max - surface_points_min).max()
        exemplar_scale = (
            (primitive_entry.bbox_max - primitive_entry.bbox_min).max().item()
        )

        # ps_transform = np.eye(4)
        # ps_transform[:3, 3] = surface_points_middle
        # ps_transform[:3, :3] *= exemplar_scale / surface_point_scale

        # Transform points actually...
        surface_points -= surface_points_middle[None, :]
        surface_points *= exemplar_scale / surface_point_scale
        vertices = (
            (mesh.vertices - surface_points_middle[None, :])
            * exemplar_scale
            / surface_point_scale
        )

        return Brush(
            torch.tensor(surface_points).float().cuda(),
            primitive_entry,
            torch.tensor(vertices).float().cuda(),
            torch.tensor(mesh.faces).float().cuda(),
            # ps_transform=ps_transform,
        )

    @staticmethod
    def import_brush(
        path, primitive_entry, transform: torch.Tensor | None = None
    ) -> Brush:
        points = np.load(path)["points"]
        points = torch.tensor(points).cuda()

        if transform is not None:
            transformed_pos = torch.cat(
                [
                    points,
                    torch.ones((points.shape[0], 1), device=points.device),
                ],
                dim=-1,
            )
            transformed_pos = torch.matmul(transform, transformed_pos.T).T
            points = transformed_pos[:, :3] / transformed_pos[:, 3][:, None]

        return Brush(points=points, primitive_entry=primitive_entry)

    def voxelize(self, res) -> torch.Tensor:
        bbox_min, bbox_max = (
            self.primitive_entry.bbox_min,
            self.primitive_entry.bbox_max,
        )

        surface_points = self.points

        # Apply the inverse transform (if necessary)
        if self.primitive_entry.transform is not None:
            transform = torch.linalg.inv(self.primitive_entry.transform)
            transformed_pos = torch.cat(
                [
                    surface_points,
                    torch.ones(
                        (surface_points.shape[0], 1), device=surface_points.device
                    ),
                ],
                dim=-1,
            )
            transformed_pos = torch.matmul(transform, transformed_pos.T).T
            surface_points = transformed_pos[:, :3] / transformed_pos[:, 3][:, None]

        surface_points = (surface_points - bbox_min) / (bbox_max - bbox_min)
        # surface_points = filter_bbox(
        #     surface_points,
        #     bbox_min=torch.zeros(3).to(surface_points),
        #     bbox_max=torch.ones(3).to(surface_points),
        # )
        # Then quantize
        surface_voxels = (surface_points * res).int()
        surface_voxels = torch.unique(surface_voxels, dim=0)  # along dim=0

        return surface_voxels

    def gui(self) -> None:

        if (
            self.ps_structure is not None
            and self.exemplar_brush_mode
            and not self.is_mesh
        ):
            self.ps_structure.update()

        if self.ps_structure is not None:
            enable_gizmo = self.is_enabled_transform_gizmo()
            clicked, enable_gizmo = state_button(
                enable_gizmo,
                enabled_str="Disable Gizmo##brush",
                disabled_str="Enable Gizmo##brush",
            )
            if clicked:
                self.enable_transform_gizmo(enable_gizmo)
                self.ps_structure.set_enabled(enable_gizmo)

            if KEY_HANDLER("k"):
                self.gizmo_translate = not self.gizmo_translate
                if self.gizmo_translate:
                    self.ps_structure.set_transform_mode_gizmo(
                        ps.TransformMode.TRANSLATION | ps.TransformMode.ROTATION
                    )
                else:
                    self.ps_structure.set_transform_mode_gizmo(ps.TransformMode.SCALE)

        clicked, self.exemplar_brush_mode = psim.Checkbox(
            "exemplar_mode", self.exemplar_brush_mode
        )

        if clicked:
            self.ps_structure.remove()
            self.display()
