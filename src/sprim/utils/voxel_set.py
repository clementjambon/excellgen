from enum import StrEnum
from typing import Callable
from dataclasses import dataclass

import torch
import numpy as np
import polyscope as ps
import polyscope.imgui as psim

from sprim.utils.process_utils import (
    apply_transform,
)
from sprim.utils.viewer_utils import (
    CUBE_VERTICES,
    CUBE_TRIANGLES,
    CUBE_EDGES,
    CUBE_FACE_NEIGHBOR_OFFSETS,
)
from sprim.utils.gui_utils import KEY_HANDLER
from sprim.gaussians.global_state import GLOBAL_STATE


class VoxelEditMode(StrEnum):
    ERASE = "erase"
    ADD = "add"


VOXEL_EDIT_MODE_MAP = {x: i for i, x in enumerate(VoxelEditMode)}
VOXEL_EDIT_MODE_INVMAP = {i: x for i, x in enumerate(VoxelEditMode)}

DEFAULT_HOVER_ADD_COLOR = torch.tensor([0.0, 1.0, 0.0], device="cuda")
DEFAULT_HOVER_ERASE_COLOR = torch.tensor([1.0, 0.0, 0.0], device="cuda")
DEFAULT_BASE_COLOR = torch.tensor([1.0, 1.0, 1.0], device="cuda")


@dataclass()
class VoxelSet:

    voxels: torch.Tensor
    get_transform_voxels: Callable[[], torch.Tensor | None]
    add_voxel_callback: Callable[[torch.Tensor], None] | None

    vertices: torch.Tensor
    faces: torch.Tensor
    edges: torch.Tensor
    rgb: torch.Tensor | None  # Defined per-voxel!

    mode: VoxelEditMode = VoxelEditMode.ADD

    def __init__(
        self,
        voxels: torch.Tensor,
        res: int,
        get_transform: Callable[[], torch.Tensor | None] = lambda: None,
        bbox_min: torch.Tensor = torch.Tensor([-1.0, -1.0, -1.0]).unsqueeze(0).cuda(),
        bbox_max: torch.Tensor = torch.Tensor([1.0, 1.0, 1.0]).unsqueeze(0).cuda(),
        rgb: torch.Tensor | None = None,
        voxel_edge_width: float = 0.0025,
        prefix: str = "",
        enabled: bool | None = None,
        add_erase_voxel_callback: Callable[[torch.Tensor], None] | None = None,
    ) -> None:

        # Make sure we have floats
        self.voxels = torch.clone(voxels.cuda()).float()
        self.get_transform_voxels = get_transform
        self.res = res
        self.ps_voxels = None
        self.ps_edges = None
        self.rgb = rgb
        self.voxel_edge_width = voxel_edge_width
        self.prefix = prefix + "_"
        self.bbox_min = bbox_min
        self.bbox_max = bbox_max
        self.time_stamp = 0
        self.last_hover_time_stamp = 0
        self.add_erase_voxel_callback = add_erase_voxel_callback

        self.reset_from_voxels(enabled)

    def is_enabled_transform_gizmo(self) -> bool:
        return self.ps_voxels.is_enabled_transform_gizmo()

    def enable_transform_gizmo(self, enabled: bool = True) -> None:
        self.ps_voxels.enable_transform_gizmo(enabled)

    def set_transform_mode_gizmo(self, mode):
        self.ps_voxels.set_transform_mode_gizmo(mode)

    def get_transform(self):
        return self.ps_voxels.get_transform()

    def set_transform(self, transform: np.ndarray):
        self.ps_voxels.set_transform(transform)
        self.update()

    def update(self):
        self.ps_edges.set_transform(self.ps_voxels.get_transform())

    def set_enabled(self, enabled: bool = True) -> None:
        if self.ps_voxels is not None:
            self.ps_voxels.set_enabled(enabled)
        if self.ps_edges is not None:
            self.ps_edges.set_enabled(enabled)

        GLOBAL_STATE.use_depth = not enabled

    def is_enabled(self) -> bool:
        return self.ps_voxels.is_enabled()

    def remove(self) -> None:
        if self.ps_voxels is not None:
            self.ps_voxels.remove()
        if self.ps_edges is not None:
            self.ps_edges.remove()

    def voxel_to_world(self, x: torch.Tensor) -> torch.Tensor:
        # Convert to bbox and apply transform
        world_pos = (self.bbox_max - self.bbox_min) * (
            1.0 / float(self.res)
        ) * x + self.bbox_min

        transform = self.get_transform_voxels()
        if transform is not None:

            world_pos = apply_transform(world_pos, transform)

        return world_pos

    def display(self, enabled: bool | None = None) -> None:
        # Don't forget to first kill the reference!
        # if self.ps_voxels is not None:
        #     self.ps_voxels.remove()
        #     self.ps_voxels = None

        vertices_in_world = self.voxel_to_world(self.vertices).cpu().numpy()

        self.ps_voxels = ps.register_surface_mesh(
            self.prefix + "s_voxels",
            vertices_in_world,
            self.faces.cpu().numpy(),
            enabled=enabled,
        )
        self.ps_voxels.set_edge_width(0.0)

        if self.rgb is not None:
            self.ps_voxels.add_color_quantity(
                self.prefix + "rgb",
                torch.repeat_interleave(self.rgb, 12, dim=0).cpu().numpy(),
                enabled=True,
                defined_on="faces",
            )
        else:
            self.ps_voxels.add_color_quantity(
                self.prefix + "selection",
                DEFAULT_BASE_COLOR.unsqueeze(0)
                .repeat((len(self.faces), 1))
                .cpu()
                .numpy(),
                defined_on="faces",
                enabled=True,
            )

            self.selection_buffer = self.ps_voxels.get_quantity_buffer(
                self.prefix + "selection", "colors"
            )

            self.ps_voxels.set_hover_callback(self.hover_callback)

        # if self.voxel_edge_width > 0.0:
        self.ps_edges = ps.register_curve_network(
            self.prefix + "s_edges",
            vertices_in_world,
            self.edges.cpu().numpy(),
            enabled=enabled,
            color=[0, 0, 0],
            radius=self.voxel_edge_width,
        )

    @torch.no_grad()
    def hover_callback(
        self, surface_pick_type: ps.SurfacePickType, surface_pick_id: int
    ):

        self.last_hover_time_stamp = self.time_stamp

        if surface_pick_type == ps.SurfacePickType.VERTEX:
            voxel_id = surface_pick_id // 8
        elif surface_pick_type == ps.SurfacePickType.FACE:
            voxel_id = surface_pick_id // 12
        else:
            raise ValueError("Cannot erase voxels from edge or halfedge")

        if psim.IsMouseClicked(0) and psim.GetIO().KeyAlt:
            if self.mode == VoxelEditMode.ERASE:
                self.erase_voxel(voxel_id)
            elif self.mode == VoxelEditMode.ADD:
                if surface_pick_type != ps.SurfacePickType.FACE:
                    print("Can only add voxel when picking face")
                    return
                relative_face = surface_pick_id % 12
                self.add_voxel(voxel_id, relative_face)

            # Make sure to return!
            return

        current_selection = DEFAULT_BASE_COLOR.unsqueeze(0).repeat(
            (12 * len(self.voxels), 1)
        )
        current_selection[12 * voxel_id + torch.arange(12)] = (
            DEFAULT_HOVER_ADD_COLOR
            if self.mode == VoxelEditMode.ADD
            else DEFAULT_HOVER_ERASE_COLOR
        )

        self.selection_buffer.update_data(current_selection.cpu().numpy())

    def reset_from_voxels(self, enabled: bool | None = None) -> None:
        # self.voxels = coord_bbox_filter(self.voxels, self.res)

        # TODO: switch that to torch (i.e., GPU)
        vertex_offsets = torch.repeat_interleave(self.voxels, 8, dim=0)
        self.vertices = CUBE_VERTICES.repeat((len(self.voxels), 1)) + vertex_offsets

        # 8 for 8 vertices
        triangles_offsets = torch.repeat_interleave(
            (8 * torch.arange(len(self.voxels), device="cuda"))[:, None].repeat((1, 3)),
            len(CUBE_TRIANGLES),
            dim=0,
        )
        self.faces = CUBE_TRIANGLES.repeat((len(self.voxels), 1)) + triangles_offsets

        edge_offsets = torch.repeat_interleave(
            (8 * torch.arange(len(self.voxels), device="cuda"))[:, None].repeat((1, 2)),
            len(CUBE_EDGES),
            dim=0,
        )
        self.edges = CUBE_EDGES.repeat((len(self.voxels), 1)) + edge_offsets

        self.display(enabled)

    def erase_voxel(self, voxel_id: int) -> None:
        if voxel_id >= len(self.voxels) or voxel_id < 0:
            print(f"Cannot erase voxel with id {voxel_id}!")
            return

        pos_mask = torch.ones(self.voxels.shape[0], dtype=bool, device="cuda")
        pos_mask[voxel_id : voxel_id + 1] = False

        self.voxels = self.voxels[pos_mask]

        if self.add_erase_voxel_callback is not None:
            self.add_erase_voxel_callback(self.voxels)

        self.reset_from_voxels()

    def add_voxel(self, voxel_id, relative_face):
        new_voxels = self.voxels[voxel_id] + CUBE_FACE_NEIGHBOR_OFFSETS[relative_face]
        new_voxels = new_voxels.unsqueeze(0)

        self.voxels = torch.cat([self.voxels, new_voxels])

        if self.add_erase_voxel_callback is not None:
            self.add_erase_voxel_callback(self.voxels)

        self.reset_from_voxels()
        # TODO: check bounds

    def gui(self):

        # Press "s" to switch between add/remove modes
        if KEY_HANDLER("s"):
            self.mode = VOXEL_EDIT_MODE_INVMAP[1 - VOXEL_EDIT_MODE_MAP[self.mode]]

        # If it wasn't hovered this frame, clear the selection
        if self.time_stamp != self.last_hover_time_stamp:
            current_selection = DEFAULT_BASE_COLOR.unsqueeze(0).repeat(
                (12 * len(self.voxels), 1)
            )
            self.selection_buffer.update_data(current_selection.cpu().numpy())

        self.time_stamp += 1

        window_flags = psim.ImGuiWindowFlags_MenuBar
        psim.PushStyleVar(psim.ImGuiStyleVar_ChildRounding, 5.0)
        psim.BeginChild(
            f"voxel_set_{self.prefix}", size=(0, 70), border=True, flags=window_flags
        )

        # =====================
        # Window starts here
        # =====================

        if psim.BeginMenuBar():
            psim.Text(f"VoxelSet: {self.prefix}")
            psim.EndMenuBar()

        clicked, mode_idx = psim.SliderInt(
            "Edit Mode",
            VOXEL_EDIT_MODE_MAP[self.mode],
            v_min=0,
            v_max=len(VoxelEditMode) - 1,
            format=f"{self.mode.value}",
        )

        if clicked:
            self.mode = VOXEL_EDIT_MODE_INVMAP[mode_idx]

        # =====================
        # Window ends here
        # =====================

        psim.EndChild()
        psim.PopStyleVar()
