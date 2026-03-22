from enum import StrEnum
from functools import partial

import numpy as np
import torch
import random

import polyscope as ps
import polyscope.imgui as psim

from sprim.utils.gui_utils import KEY_HANDLER
from sprim.utils.process_utils import filter_bbox, apply_transform
from sprim.gaussians.global_state import GLOBAL_STATE


class BrushMode(StrEnum):
    ADD = "add"
    REMOVE = "remove"


BRUSH_MODE_MAP = {x: i for i, x in enumerate(BrushMode)}
BRUSH_MODE_INVMAP = {i: x for i, x in enumerate(BrushMode)}

DEFAULT_HOVER_ADD_COLOR = torch.tensor([0.0, 1.0, 0.0], device="cuda")
DEFAULT_HOVER_REMOVE_COLOR = torch.tensor([1.0, 0.0, 0.0], device="cuda")
DEFAULT_MASKED_COLOR = torch.tensor([0.2, 0.2, 0.2], device="cuda")
DEFAULT_BASE_COLOR = torch.tensor([1.0, 1.0, 1.0], device="cuda")
MIN_SELECTION_RADIUS = 0.001
MAX_SELECTION_RADIUS = 1.0
DEFAULT_SECTION_RADIUS = 0.1
SELECTION_RADIUS_SENTIVITY = 0.01
MAX_POINTS = 50000
DEFAULT_BRUSH_MODE = BrushMode.REMOVE


class PcSelector:

    def __init__(
        self,
        name: str,
        pos: np.ndarray | torch.Tensor,
        transform: torch.Tensor | None = None,
        enabled: bool = True,
        filter_callback=None,
        selection_radius: float = DEFAULT_SECTION_RADIUS,
        base_color: torch.Tensor = DEFAULT_BASE_COLOR,
        masked_color: torch.Tensor = DEFAULT_MASKED_COLOR,
        hover_add_color: torch.Tensor = DEFAULT_HOVER_ADD_COLOR,
        hover_remove_color: torch.Tensor = DEFAULT_HOVER_REMOVE_COLOR,
        already_initialized: bool = True,
    ) -> None:

        self.name = name
        self.selection_radius = selection_radius
        self.brush_mode = DEFAULT_BRUSH_MODE
        self.transform = transform
        self.square_brush = False
        self.last_selected_point_id = -1
        self.base_color = base_color.cuda()
        self.masked_color = masked_color.cuda()
        self.hover_add_color = hover_add_color.cuda()
        self.hover_remove_color = hover_remove_color.cuda()
        # Note if there's a filter callback, we'll assume the structure will get destroyed afterwards
        self.filter_callback = filter_callback

        # Additional options
        self.max_subsampling_points = MAX_POINTS
        self.filter_bbox_min = [-1.0, -1.0, -1.0]
        self.filter_bbox_max = [1.0, 1.0, 1.0]

        self.pos = (
            torch.tensor(pos).float().cuda()
            if not isinstance(pos, torch.Tensor)
            else pos.float()
        ).clone()
        self.ps_structure = None
        self.reset_selection(None, already_initialized)
        GLOBAL_STATE.use_depth = False

    def kill(self):
        # This will automatically remove the callback!
        GLOBAL_STATE.use_depth = True
        self.ps_structure.remove()
        self.ps_structure = None

    def is_enabled(self):
        return self.ps_structure is not None and self.ps_structure.is_enabled()

    def set_enabled(self, enabled: bool = True):
        if self.ps_structure is not None:
            self.ps_structure.set_enabled(enabled)

    def invert_selection(self):
        self.selection_mask = ~self.selection_mask

    @torch.no_grad()
    def reset_selection(
        self, new_pos: torch.Tensor | None = None, already_initialized: bool = False
    ):

        if self.ps_structure is not None:
            self.ps_structure.remove()
            self.ps_structure = None

        if new_pos is not None:
            self.pos = new_pos

        self.selection_mask = (
            torch.ones(len(self.pos), dtype=bool, device="cuda")
            if self.brush_mode == BrushMode.REMOVE or already_initialized
            else torch.zeros(len(self.pos), dtype=bool, device="cuda")
        )

        transformed_pos = apply_transform(self.pos, self.transform)

        self.ps_structure = ps.register_point_cloud(
            self.name, transformed_pos.cpu().numpy(), enabled=True, radius=0.01
        )
        self.ps_structure.add_color_quantity(
            "selection",
            self.base_color.unsqueeze(0).repeat((len(self.pos), 1)).cpu().numpy(),
            enabled=True,
        )
        self.selection_buffer = self.ps_structure.get_quantity_buffer(
            "selection", "colors"
        )

        self.ps_structure.set_hover_callback(self.hover_callback)

    @torch.no_grad()
    def hover_callback(self, point_id: int):

        selected_pos = self.pos[min(point_id, len(self.pos) - 1)]

        if self.square_brush:
            within_radius = (torch.abs(selected_pos[None, :] - self.pos)).max(1)[
                0
            ] < self.selection_radius
        else:
            within_radius = ((selected_pos[None, :] - self.pos) ** 2).sum(
                1
            ) < self.selection_radius**2

        if (
            psim.IsMouseClicked(0)
            and psim.GetIO().KeyAlt
            and point_id != self.last_selected_point_id
        ):
            if self.brush_mode == BrushMode.REMOVE:
                self.selection_mask &= ~within_radius
            else:
                self.selection_mask |= within_radius

            self.last_selected_point_id = point_id

        current_selection = self.base_color.unsqueeze(0).repeat((len(self.pos), 1))
        current_selection[~self.selection_mask] = self.masked_color

        current_selection[within_radius] = (
            self.hover_add_color
            if self.brush_mode == BrushMode.ADD
            else self.hover_remove_color
        )
        self.selection_buffer.update_data_from_device(current_selection)

    def subsample(self):
        # NOTE: we need to apply a permutation because I sort Gaussians when
        # exporting them in the first place
        perm = torch.randperm(self.pos.size(0)).cuda()
        idx = perm[:MAX_POINTS]
        new_pos = self.pos[idx]
        self.reset_selection(new_pos)

    def gui(self):

        io = psim.GetIO()

        # Use wheel to increase/decrease brush radius
        if io.MouseWheel != 0 and io.KeyAlt:
            self.selection_radius += SELECTION_RADIUS_SENTIVITY * float(io.MouseWheel)
        self.selection_radius = max(
            MIN_SELECTION_RADIUS,
            min(self.selection_radius, MAX_SELECTION_RADIUS),
        )

        if KEY_HANDLER("s"):
            self.brush_mode = BRUSH_MODE_INVMAP[1 - BRUSH_MODE_MAP[self.brush_mode]]

        elif KEY_HANDLER("h"):
            self.set_enabled(not self.is_enabled())
        elif KEY_HANDLER("i"):
            self.invert_selection()

        window_flags = psim.ImGuiWindowFlags_MenuBar
        psim.PushStyleVar(psim.ImGuiStyleVar_ChildRounding, 5.0)
        psim.BeginChild(
            f"pc_selector_{self.name}", size=(0, 160), border=True, flags=window_flags
        )

        # =====================
        # Window starts here
        # =====================

        if psim.BeginMenuBar():
            psim.Text(f"PcSelector: {self.name}")
            psim.EndMenuBar()

        clicked, brush_mode_idx = psim.SliderInt(
            "Brush Mode",
            BRUSH_MODE_MAP[self.brush_mode],
            v_min=0,
            v_max=len(BrushMode) - 1,
            format=f"{self.brush_mode.value}",
        )
        if clicked:
            self.brush_mode = BRUSH_MODE_INVMAP[brush_mode_idx]

        if psim.Button(f"Reset Selection##pc_selector_{self.name}"):
            self.reset_selection()

        psim.SameLine()

        if psim.Button(f"Apply Selection##pc_selector_{self.name}") or KEY_HANDLER("a"):
            filtering_mask = self.selection_mask
            new_pos = self.pos[filtering_mask]
            if self.filter_callback is not None:
                self.filter_callback(filtering_mask)
            else:
                self.reset_selection(new_pos)

        if psim.Button("Subsample##brush_selection"):
            self.subsample()

        psim.SameLine()
        _, self.square_brush = psim.Checkbox("square", self.square_brush)

        if psim.TreeNode(f"Advanced options##pc_selector_{self.name}"):

            _, self.selection_radius = psim.SliderFloat(
                f"Selection Radius##pc_selector_{self.name}",
                self.selection_radius,
                v_min=MIN_SELECTION_RADIUS,
                v_max=MAX_SELECTION_RADIUS,
            )

            _, self.max_subsampling_points = psim.InputInt(
                f"subsampling pts", self.max_subsampling_points
            )

            _, self.filter_bbox_min = psim.InputFloat3("bbox_min", self.filter_bbox_min)
            _, self.filter_bbox_max = psim.InputFloat3("bbox_max", self.filter_bbox_max)

            if psim.Button(f"Filter##pc_selector_{self.name}"):
                new_pos, filtering_mask = filter_bbox(
                    self.pos,
                    torch.tensor(self.filter_bbox_min, device="cuda"),
                    torch.tensor(self.filter_bbox_max, device="cuda"),
                    return_indices=True,
                )

                if self.filter_callback is not None:
                    self.filter_callback(filtering_mask)
                else:
                    self.reset_selection(new_pos)

            psim.TreePop()

        # =====================
        # Window ends here
        # =====================

        psim.EndChild()
        psim.PopStyleVar()
