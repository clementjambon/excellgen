from dataclasses import dataclass
from enum import StrEnum

import torch
import numpy as np
import polyscope as ps
import polyscope.imgui as psim

from sprim.utils.viewer_utils import create_checker_bbox, create_checker_plane

DEFAULT_SUBDIVISION_IDX = 2
SUBDIVISIONS = [4, 8, 16, 32, 64]
WINDOW_SIZE = 140
AXIS_NAMES = ["x", "y", "z"]


class HelperMode(StrEnum):
    NONE = "none"
    PLANE = "plane"
    BOX = "box"


HELPER_MODE_MAP = {x: i for i, x in enumerate(HelperMode)}
HELPER_MODE_INVMAP = {i: x for i, x in enumerate(HelperMode)}


class ExtractionHelper:

    def __init__(self, bbox_min: np.ndarray, bbox_max: np.ndarray) -> None:
        self.helper_mode = HelperMode.NONE
        self.current_structure: ps.Structure = None
        self.bbox_min = bbox_min
        self.bbox_max = bbox_max
        self.current_subdivision_idx = DEFAULT_SUBDIVISION_IDX
        self.current_axis = 0

    def set_enabled(self, enabled: bool = True):
        if self.current_structure is not None:
            self.current_structure.set_enabled(enabled)

    def set_transform(self, transform) -> None:
        if self.current_structure is not None:
            self.current_structure.set_transform(transform)

    def switch_mode(self) -> None:
        if self.current_structure is not None:
            self.current_structure.remove()
            self.current_structure = None

        if self.helper_mode == HelperMode.BOX:
            self.current_structure = create_checker_bbox(
                "helper_box",
                SUBDIVISIONS[self.current_subdivision_idx] + 1,
                self.bbox_min,
                self.bbox_max,
            )
        elif self.helper_mode == HelperMode.PLANE:
            self.current_structure = create_checker_plane(
                "helper_box",
                SUBDIVISIONS[self.current_subdivision_idx] + 1,
                self.current_axis,
                self.bbox_min,
                self.bbox_max,
            )

    def gui(self) -> None:

        window_flags = psim.ImGuiWindowFlags_MenuBar
        psim.PushStyleVar(psim.ImGuiStyleVar_ChildRounding, 1.0)
        psim.PushStyleColor(
            psim.ImGuiCol_ChildBg, psim.ImColor.HSV(0.0 / 7.0, 0.5, 0.5)
        )
        psim.BeginChild(
            f"extraction_helper", size=(0, WINDOW_SIZE), border=True, flags=window_flags
        )

        # =====================
        # Window starts here
        # =====================

        if psim.BeginMenuBar():
            psim.Text(f"Extraction Helper")
            psim.EndMenuBar()

        clicked, helper_mode_idx = psim.SliderInt(
            "Helper Mode",
            HELPER_MODE_MAP[self.helper_mode],
            v_min=0,
            v_max=len(HelperMode) - 1,
            format=f"{self.helper_mode.value}",
        )

        if clicked:
            self.helper_mode = HELPER_MODE_INVMAP[helper_mode_idx]
            self.switch_mode()

        if self.helper_mode != HelperMode.NONE:
            clicked, self.current_subdivision_idx = psim.SliderInt(
                "subdivision##extraction_helper",
                self.current_subdivision_idx,
                v_min=0,
                v_max=len(SUBDIVISIONS) - 1,
                format=f"{SUBDIVISIONS[self.current_subdivision_idx]}",
            )
            if clicked:
                self.switch_mode()

        if self.helper_mode == HelperMode.PLANE:
            clicked, self.current_axis = psim.SliderInt(
                "axis##extraction_helper",
                self.current_axis,
                v_min=0,
                v_max=2,
                format=f"{AXIS_NAMES[self.current_axis]}",
            )
            if clicked:
                self.switch_mode()

        # =====================
        # Window ends here
        # =====================

        psim.EndChild()
        psim.PopStyleVar()
        psim.PopStyleColor()
