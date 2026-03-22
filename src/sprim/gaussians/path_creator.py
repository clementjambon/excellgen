from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Dict, Any
from copy import deepcopy
from enum import StrEnum

import deepdish as dd

import torch
import numpy as np
import polyscope as ps
import polyscope.imgui as psim

import splines
import splines.quaternion
from scipy import interpolate


from nerfstudio.cameras.cameras import Cameras
import viser.transforms as tf

from sprim.utils.gui_utils import camera_from_ps, camera_to_ps, save_popup
from sprim.utils.io_utils import resolve_screenshot_path


# This is freely adapted from (in a simplified form)
# https://github.com/nerfstudio-project/nerfstudio/blob/main/nerfstudio/viewer/render_panel.py
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

MIN_TRANSITION_TIME = 1.0
MAX_TRANSITION_TIME = 10.0
DEFAULT_CPATH = "camera_paths/trajectory.cpath"
TMP_CPATH = "camera_paths/_temp.cpath"


def serialize_cam(camera: Cameras) -> Dict[str, Any]:
    return {
        "cx": camera.cx,
        "cy": camera.cy,
        "fx": camera.fx,
        "fy": camera.fy,
        "width": camera.width,
        "height": camera.height,
        "c2w": camera.camera_to_worlds.cpu().numpy(),
    }


def deserialize_cam(data: Dict[str, Any]) -> Cameras:
    return Cameras(
        camera_to_worlds=torch.tensor(data["c2w"]).float(),
        fx=data["fx"],
        fy=data["fy"],
        cx=data["cx"],
        cy=data["cy"],
        width=data["width"],
        height=data["height"],
    )


@dataclass(kw_only=True)
class Keyframe:
    idx: int = -1
    camera: Cameras
    transition_time: float = 2.0
    ps_cam_view: ps.CameraView | None = None

    @property
    def wxyz(self):
        return tf.SO3.from_matrix(self.camera.camera_to_worlds[:3, :3]).wxyz

    @property
    def position(self):
        return self.camera.camera_to_worlds[:3, 3]

    def enable_gizmo(self, enabled: bool):
        if self.ps_cam_view is not None:
            self.ps_cam_view.set_transform_mode_gizmo(
                ps.TransformMode.TRANSLATION | ps.TransformMode.ROTATION
            )
            self.ps_cam_view.enable_transform_gizmo(enabled=enabled)

    def show(self):
        # if self.ps_cam_view is None:
        params = camera_to_ps(self.camera)
        self.ps_cam_view = ps.register_camera_view(
            f"keyframe_{self.idx}", camera_parameters=params
        )

    # else:
    #     self.ps_cam_view.set_enabled(True)

    # def hide(self):
    #     if self.ps_cam_view is not None:
    #         self.ps_cam_view.set_enabled(False)
    #         self.ps_cam_view.enable_transform_gizmo(False)

    def serialize(self) -> Dict[str, Any]:
        return {
            "transition_time": self.transition_time,
            "camera": serialize_cam(self.camera),
        }

    @staticmethod
    def deserialize(data: Dict[str, Any]) -> Keyframe:
        return Keyframe(
            transition_time=data["transition_time"],
            camera=deserialize_cam(data["camera"]),
        )


class QualityMode(StrEnum):
    LOSSLESS = "lossless"
    SLOW = "slow"
    MEDIUM = "medium"
    FAST = "fast"


QUALITY_MODE_MAP = {x: i for i, x in enumerate(QualityMode)}
QUALITY_MODE_INVMAP = {i: x for i, x in enumerate(QualityMode)}
QUALITY_FFMPEG_MAP = {
    QualityMode.LOSSLESS: "10",
    QualityMode.SLOW: "1",
    QualityMode.MEDIUM: "2",
    QualityMode.FAST: "3",
}


class PathCreator:

    def __init__(self) -> None:
        self.keyframes: List[Keyframe] = []
        self.framerate: int = 30
        self.tension: float = 0.5
        self.loop: bool = False
        self.vis_cameras: List[Cameras] | None = None
        self.i_camera: int = 0
        self.ps_curve: ps.CurveNetwork | None = None
        self.selected_keyframe: int = -1
        self.cpath = DEFAULT_CPATH

        # Recording
        self.is_recording: bool = False
        self.quality_mode: QualityMode = QualityMode.LOSSLESS

    def serialize(self) -> Dict[str, Any]:
        data = {
            "framerate": self.framerate,
            "tension": self.tension,
            "loop": self.loop,
            "quality_mode": QUALITY_MODE_MAP[self.quality_mode],
        }

        data["keyframes"] = []
        for keyframe in self.keyframes:
            data["keyframes"].append(keyframe.serialize())

        return data

    def load_serialized(self, data: Dict[str, Any]) -> None:
        self.framerate = data["framerate"]
        self.tension = data["tension"]
        self.loop = data["loop"]
        self.quality_mode = QUALITY_MODE_INVMAP[data["quality_mode"]]

        self.keyframes = []
        for keyframe in data["keyframes"]:
            self.add_keyframe(Keyframe.deserialize(keyframe))

    def save(self, cpath: str | None = None):
        data = self.serialize()
        cpath = self.cpath if cpath is None else cpath
        os.makedirs(os.path.abspath(os.path.dirname(cpath)), exist_ok=True)
        dd.io.save(cpath, data)

    def load(self):
        if not os.path.exists(self.cpath):
            print(f"Camera path does not exist at: {self.cpath}")
            return
        data = dd.io.load(self.cpath)
        self.load_serialized(data)

    def start_recording(self):
        # First, make sure splines are all up to date
        # TODO: this shouldn't really be necessary
        self.update_spline()
        self.save(TMP_CPATH)

        from torchaudio.io import StreamWriter

        # recording_path = self._resolve_screenshot_path(is_video=True)
        recording_path = resolve_screenshot_path(is_video=True)

        window_size = ps.get_window_size()

        self.recording_stream = StreamWriter(recording_path)
        cuda_conf = {
            "encoder": "h264_nvenc",  # Use CUDA HW decoder
            "encoder_format": "rgb0",
            "encoder_option": {
                "gpu": "0",
                "preset": QUALITY_FFMPEG_MAP[self.quality_mode],
            },
            "hw_accel": "cuda:0",
            "height": window_size[1],
            "width": window_size[0],
        }

        self.recording_stream.add_video_stream(frame_rate=self.framerate, **cuda_conf)
        self.recording_stream.open()
        self.i_camera = 0
        self.is_recording = True
        self.set_camera()

    def compute_duration(self) -> float:
        """Compute the total duration of the trajectory."""
        total = 0.0
        for i, keyframe in enumerate(self.keyframes):
            if i == 0 and not self.loop:
                continue
            total += keyframe.transition_time
        return total

    def compute_transition_times_cumsum(self) -> np.ndarray:
        """Compute the total duration of the trajectory."""
        total = 0.0
        out = [0.0]
        for i, keyframe in enumerate(self.keyframes):
            if i == 0:
                continue
            total += keyframe.transition_time
            out.append(total)

        # TODO: handle loops
        if self.loop:
            keyframe = next(iter(self.keyframes))
            total += keyframe.transition_time
            out.append(total)

        return np.array(out)

    def add_keyframe(self, keyframe: Keyframe):

        if self.selected_keyframe >= 0:
            for other in self.keyframes[self.selected_keyframe - 1 :]:
                other.idx += 1
                other.show()

            keyframe.idx = self.selected_keyframe
            self.keyframes = (
                self.keyframes[: self.selected_keyframe - 1]
                + [keyframe]
                + self.keyframes[self.selected_keyframe - 1 :]
            )
        else:

            keyframe.idx = len(self.keyframes)
            self.keyframes.append(keyframe)

        keyframe.show()

        if len(self.keyframes) >= 2:
            self.update_spline()

    def remove_keyframe(self, idx: int):
        del self.keyframes[idx]

        self.selected_keyframe = min(self.selected_keyframe, len(self.keyframes) - 1)

        if len(self.keyframes) >= 2:
            self.update_spline()

    def spline_t_from_t_sec(self, time: np.ndarray) -> np.ndarray:
        """From a time value in seconds, compute a t value for our geometric
        spline interpolation. An increment of 1 for the latter will move the
        camera forward by one keyframe.

        We use a PCHIP spline here to guarantee monotonicity.
        """
        transition_times_cumsum = self.compute_transition_times_cumsum()
        spline_indices = np.arange(transition_times_cumsum.shape[0])

        if self.loop:
            # In the case of a loop, we pad the spline to match the start/end
            # slopes.
            interpolator = interpolate.PchipInterpolator(
                x=np.concatenate(
                    [
                        [-(transition_times_cumsum[-1] - transition_times_cumsum[-2])],
                        transition_times_cumsum,
                        transition_times_cumsum[-1:] + transition_times_cumsum[1:2],
                    ],
                    axis=0,
                ),
                y=np.concatenate(
                    [[-1], spline_indices, [spline_indices[-1] + 1]], axis=0
                ),
            )
        else:
            interpolator = interpolate.PchipInterpolator(
                x=transition_times_cumsum, y=spline_indices
            )

        # Clip to account for floating point error.
        return np.clip(interpolator(time), 0, spline_indices[-1])

    # This updates all the splines that can then be rendered
    def update_spline(self):
        num_frames = int(self.compute_duration() * self.framerate)
        transition_times_cumsum = self.compute_transition_times_cumsum()
        keyframes = self.keyframes
        self._orientation_spline = splines.quaternion.KochanekBartels(
            [
                splines.quaternion.UnitQuaternion.from_unit_xyzw(
                    np.roll(keyframe.wxyz, shift=-1)
                )
                for keyframe in keyframes
            ],
            tcb=(self.tension, 0.0, 0.0),
            endconditions="closed" if self.loop else "natural",
        )
        self._position_spline = splines.KochanekBartels(
            [keyframe.position for keyframe in keyframes],
            tcb=(self.tension, 0.0, 0.0),
            endconditions="closed" if self.loop else "natural",
        )

        points_array = self._position_spline.evaluate(
            self.spline_t_from_t_sec(
                np.linspace(0, transition_times_cumsum[-1], num_frames)
            )
        )

        curve_edges = np.concatenate(
            [np.arange(num_frames - 1)[:, None], np.arange(1, num_frames)[:, None]],
            axis=-1,
        )

        # Render
        self.ps_curve = ps.register_curve_network(
            "camera_path", points_array, curve_edges
        )

        self.compute_cameras()

    def interpolate_pose_and_fov_rad(self, normalized_t: float) -> tf.SE3:
        # -> Optional[Union[Tuple[tf.SE3, float], Tuple[tf.SE3, float, float]]]:
        if len(self.keyframes) < 2:
            return None

        # self._fov_spline = splines.KochanekBartels(
        #     [
        #         (
        #             keyframe[0].override_fov_rad
        #             if keyframe[0].override_fov_enabled
        #             else self.default_fov
        #         )
        #         for keyframe in self._keyframes.values()
        #     ],
        #     tcb=(self.tension, 0.0, 0.0),
        #     endconditions="closed" if self.loop else "natural",
        # )

        # self._time_spline = splines.KochanekBartels(
        #     [
        #         (
        #             keyframe[0].override_time_val
        #             if keyframe[0].override_time_enabled
        #             else self.default_render_time
        #         )
        #         for keyframe in self._keyframes.values()
        #     ],
        #     tcb=(self.tension, 0.0, 0.0),
        #     endconditions="closed" if self.loop else "natural",
        # )

        assert self._orientation_spline is not None
        assert self._position_spline is not None
        # assert self._fov_spline is not None
        # if self.time_enabled:
        #     assert self._time_spline is not None
        max_t = self.compute_duration()
        t = max_t * normalized_t
        spline_t = float(self.spline_t_from_t_sec(np.array(t)))

        quat = self._orientation_spline.evaluate(spline_t)
        assert isinstance(quat, splines.quaternion.UnitQuaternion)
        # if self.time_enabled:
        #     return (
        #         tf.SE3.from_rotation_and_translation(
        #             tf.SO3(np.array([quat.scalar, *quat.vector])),
        #             self._position_spline.evaluate(spline_t),
        #         ),
        #         float(self._fov_spline.evaluate(spline_t)),
        #         float(self._time_spline.evaluate(spline_t)),
        #     )
        # else:
        #     return (
        #         tf.SE3.from_rotation_and_translation(
        #             tf.SO3(np.array([quat.scalar, *quat.vector])),
        #             self._position_spline.evaluate(spline_t),
        #         ),
        #         float(self._fov_spline.evaluate(spline_t)),
        #     )
        return tf.SE3.from_rotation_and_translation(
            tf.SO3(np.array([quat.scalar, *quat.vector])),
            self._position_spline.evaluate(spline_t),
        )

    def compute_cameras(self):
        num_frames = int(self.compute_duration() * self.framerate)
        self.vis_cameras = []
        for i in range(num_frames):
            pose = self.interpolate_pose_and_fov_rad(float(i) / num_frames)
            new_camera = deepcopy(self.keyframes[0].camera)
            new_camera.camera_to_worlds = torch.tensor(pose.as_matrix())
            self.vis_cameras.append(new_camera)

        self.i_camera = min(self.i_camera, len(self.vis_cameras) - 1)

    def _update_gizmos(self):
        for idx, keyframe in enumerate(self.keyframes):
            keyframe.enable_gizmo(idx == self.selected_keyframe)

    def _keyframe_selectable(self, idx: int):

        clicked, _ = psim.Selectable(
            f"{idx}##keyframe_{idx}",
            self.selected_keyframe == idx,
        )

        if clicked:
            if self.selected_keyframe == idx:
                self.selected_keyframe = -1
            else:
                self.selected_keyframe = idx

            # self._update_gizmos()

    def set_camera(self):
        if self.i_camera >= 0 and self.i_camera < len(self.vis_cameras):
            new_params = camera_to_ps(self.vis_cameras[self.i_camera])
            ps.set_view_camera_parameters(new_params)

    def render_callback(self, rendered_image: torch.Tensor):
        if self.is_recording:
            self.recording_stream.write_video_chunk(
                0,
                (rendered_image[..., :3].permute(2, 0, 1).unsqueeze(0) * 255).to(
                    torch.uint8
                ),
            )
            self.i_camera += 1
            self.set_camera()

            if self.i_camera >= len(self.vis_cameras):
                self.is_recording = False
                self.i_camera = 0
                self.recording_stream.close()

    def gui(self):

        if psim.TreeNode("Path Creator"):

            if psim.Button("Add Keyframe##path_creator"):
                camera = camera_from_ps()
                self.add_keyframe(Keyframe(camera=camera))

            psim.SameLine()
            if psim.Button("Render##path_creator"):
                self.start_recording()

            requested, self.cpath = save_popup("save##path_creator", self.cpath)
            if requested:
                self.save()

            psim.SameLine()

            requested, self.cpath = save_popup(
                "load##path_creator", self.cpath, save_label="Load", show_warning=False
            )
            if requested:
                self.load()

            clicked, self.tension = psim.SliderFloat(
                "tension##path_creator", self.tension, v_min=0.1, v_max=2.0
            )
            if clicked:
                self.update_spline()

            clicked, self.loop = psim.Checkbox("loop##path_creator", self.loop)
            if clicked:
                self.update_spline()

            if psim.TreeNode("Advanced options##path_creator"):

                clicked, quality_mode_idx = psim.SliderInt(
                    "Quality Mode##path_creator",
                    QUALITY_MODE_MAP[self.quality_mode],
                    v_min=0,
                    v_max=len(QualityMode) - 1,
                    format=f"{self.quality_mode.value}",
                )
                if clicked:
                    self.quality_mode = QUALITY_MODE_INVMAP[quality_mode_idx]

                clicked, self.framerate = psim.SliderInt(
                    "Framerate##path_creator", self.framerate, v_min=25, v_max=60
                )
                if clicked:
                    # This need to be done because we need to resample everyone
                    self.update_spline()

                psim.TreePop()

            if self.vis_cameras is not None and len(self.vis_cameras) > 0:
                clicked, self.i_camera = psim.SliderInt(
                    "i_camera##path_creator",
                    self.i_camera,
                    v_min=0,
                    v_max=len(self.vis_cameras) - 1,
                )
                if clicked:
                    self.set_camera()
                    if self.ps_curve is not None:
                        self.ps_curve.set_enabled(False)

            TEXT_BASE_HEIGHT = psim.GetTextLineHeightWithSpacing()
            MAX_DISPLAYED_KEYFRAMES = 10

            if psim.BeginTable(
                "Path Creator##table",
                3,
                psim.ImGuiTableFlags_ScrollY,
                (
                    0,
                    min(
                        TEXT_BASE_HEIGHT * (len(self.keyframes) + 1.5),
                        TEXT_BASE_HEIGHT * MAX_DISPLAYED_KEYFRAMES,
                    ),
                ),
            ):
                psim.TableSetupColumn(
                    "Name##path_creator",
                    psim.ImGuiTableColumnFlags_WidthFixed,
                    0.0,
                )
                psim.TableSetupColumn(
                    "Time##path_creator",
                    psim.ImGuiTableColumnFlags_WidthStretch
                    | psim.ImGuiTableColumnFlags_NoHide,
                    0.0,
                )
                psim.TableSetupColumn(
                    "Del##path_creator",
                    psim.ImGuiTableColumnFlags_WidthFixed
                    | psim.ImGuiTableColumnFlags_NoHide,
                    0.0,
                )

                psim.TableHeadersRow()

                for idx, keyframe in enumerate(self.keyframes):
                    psim.TableNextRow()

                    # Display Name
                    psim.TableNextColumn()
                    self._keyframe_selectable(idx)

                    # Time
                    psim.TableNextColumn()
                    # clicked, keyframe.transition_time = psim.SliderFloat(
                    #     f"time##keyframe_{idx}",
                    #     keyframe.transition_time,
                    #     v_min=MIN_TRANSITION_TIME,
                    #     v_max=MAX_TRANSITION_TIME,
                    # )
                    clicked, keyframe.transition_time = psim.InputFloat(
                        f"time##keyframe_{idx}",
                        keyframe.transition_time,
                    )

                    if clicked:
                        self.update_spline()

                    # Del
                    psim.TableNextColumn()
                    if psim.SmallButton(f"X##keyframe_{idx}"):
                        self.remove_keyframe(idx)
                        break

                psim.EndTable()
            psim.TreePop()
