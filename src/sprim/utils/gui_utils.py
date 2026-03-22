import os
from collections import defaultdict
from enum import StrEnum
from typing import Tuple, Dict, Any
import math

import polyscope as ps
import polyscope.imgui as psim

import numpy as np
import torch

from nerfstudio.cameras.cameras import Cameras

from sprim.utils.geometry import focal2fov, fov2focal


class RenderMode(StrEnum):
    RGB = "rgb"
    FEATURES = "features"
    DEPTH = "depth"


RENDER_MODE_MAP = {x: i for i, x in enumerate(RenderMode)}
RENDER_MODE_INVMAP = {i: x for i, x in enumerate(RenderMode)}

# -----------------------------
# KEYMAP
# -----------------------------


KEYMAP = {
    "space": 32,
    "1": 49,
    "a": 65,
    "b": 66,
    "d": 68,
    "f": 70,
    "g": 71,
    "h": 72,
    "i": 73,
    "j": 74,
    "k": 75,
    "m": 77,
    "n": 78,
    "o": 79,
    "p": 80,
    "q": 81,
    "r": 82,
    "s": 83,
    "t": 84,
    "u": 85,
    "y": 89,
    "z": 90,
    "left_arrow": 263,
    "right_arrow": 262,
    "up_arrow": 265,
    "down_arrow": 264,
    "rshift": 344,
    "rctrl": 345,
    "page_up": 266,
    "page_down": 267,
    "enter": 257,
}


# Utility class to handle key presses
class KeyHandler:

    def __init__(self, interval: int = 10) -> None:
        self.history = defaultdict(lambda: 0)
        self.interval = interval
        self.lock_set = set()

    # Only return true if it has just been clicked on
    def __call__(self, key: str) -> bool:
        if len(self.lock_set) == 0:
            return bool(self.history[KEYMAP[key]] == self.interval)

    def step(self):
        for k in self.history:
            if self.history[k] > 0:
                self.history[k] -= 1
            elif psim.GetIO().KeysDown[k]:
                self.history[k] = self.interval

    def lock(self, name: str):
        self.lock_set.add(name)

    def unlock(self, name: str):
        if name in self.lock_set:
            self.lock_set.remove(name)
        else:
            print(f"KEY_HANDLER: tried to unlock '{name}' but it isn't there")

    def unlock_all(self):
        self.lock_set = set()


KEY_HANDLER = KeyHandler()


def state_button(
    value: bool,
    enabled_str: str,
    disabled_str: str,
    enabled_hue: float = 0.0,
    disable_hue: float = 0.4,
) -> Tuple[bool, bool]:
    clicked = False
    hue = enabled_hue if value else disable_hue
    psim.PushStyleColor(psim.ImGuiCol_Button, psim.ImColor.HSV(hue, 0.6, 0.6))
    psim.PushStyleColor(psim.ImGuiCol_ButtonHovered, psim.ImColor.HSV(hue, 0.7, 0.7))
    psim.PushStyleColor(psim.ImGuiCol_ButtonActive, psim.ImColor.HSV(hue, 0.8, 0.8))

    if psim.Button(enabled_str if value else disabled_str):
        clicked = True
        value = not value

    psim.PopStyleColor(3)

    return clicked, value


def colored_button(label: str, hue: float = 0.4) -> bool:
    psim.PushStyleColor(psim.ImGuiCol_Button, psim.ImColor.HSV(hue, 0.6, 0.6))
    psim.PushStyleColor(psim.ImGuiCol_ButtonHovered, psim.ImColor.HSV(hue, 0.7, 0.7))
    psim.PushStyleColor(psim.ImGuiCol_ButtonActive, psim.ImColor.HSV(hue, 0.8, 0.8))

    value = psim.Button(label)

    psim.PopStyleColor(3)

    return value


def save_popup(
    popup_name: str,
    path: str,
    save_label: str = "Save",
    confirm_label: str = "Confirm",
    show_warning: str = True,
):
    requested = False

    if psim.Button(f"{save_label}##{popup_name}"):
        psim.OpenPopup(f"save_popup##{popup_name}")
        KEY_HANDLER.lock(popup_name)

    if psim.BeginPopup(f"save_popup##{popup_name}"):

        _, path = psim.InputText(f"path##{popup_name}", path)

        if show_warning and os.path.exists(path):
            psim.Text("Warning: a file already exists at this location!")

        if (
            psim.Button(f"{confirm_label}##{popup_name}")
            or psim.GetIO().KeysDown[KEYMAP["enter"]]
        ):
            requested = True
            KEY_HANDLER.unlock(popup_name)
            psim.CloseCurrentPopup()

        psim.EndPopup()

    return requested, path


def camera_from_ps() -> Cameras:
    ps_view_camera_parameters = ps.get_view_camera_parameters()

    c2w = torch.linalg.inv(torch.tensor(ps_view_camera_parameters.get_view_mat()))

    window_size = ps.get_window_size()
    WIDTH = window_size[0]
    HEIGHT = window_size[1]
    focal = fov2focal(
        ps_view_camera_parameters.get_fov_vertical_deg() * math.pi / 180.0,
        HEIGHT,
    )

    return Cameras(
        camera_to_worlds=c2w[:3, :4].cpu(),
        fx=focal,
        fy=focal,
        cx=WIDTH / 2.0,
        cy=HEIGHT / 2.0,
        width=WIDTH,
        height=HEIGHT,
    )


def camera_to_ps(camera: Cameras) -> ps.CameraParameters:
    fov_vertical_deg = 180 * focal2fov(camera.fx, camera.height) / math.pi
    intrinsics = ps.CameraIntrinsics(
        fov_vertical_deg=fov_vertical_deg, aspect=camera.width / camera.height
    )
    c2w = (
        camera.camera_to_worlds
        if camera.camera_to_worlds.shape[0] == 4
        else torch.cat(
            [camera.camera_to_worlds, torch.tensor([[0.0, 0.0, 0.0, 1.0]])], dim=0
        )
    )
    extrinsics = ps.CameraExtrinsics(mat=torch.linalg.inv(c2w).cpu().numpy())
    return ps.CameraParameters(intrinsics=intrinsics, extrinsics=extrinsics)
