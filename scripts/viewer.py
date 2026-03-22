import time
import os
from typing import List

import argparse
import torch
import numpy as np
import polyscope as ps
import polyscope.imgui as psim


from sprim.utils.gui_utils import (
    KEY_HANDLER,
    KEYMAP,
    RenderMode,
    RENDER_MODE_MAP,
    RENDER_MODE_INVMAP,
    state_button,
    save_popup,
    camera_from_ps,
)
from sprim.inverse.nerfstudio_loader import SubjectLoader
from sprim.gaussians.suggestive_selection import SuggestiveSelection
from sprim.gaussians.brush_creator import BrushCreator
from sprim.gaussians.latent_exporter import LatentExporterGaussians
from sprim.gaussians.primitive_library import PrimitiveLibrary, PrimitiveEntry
from sprim.gaussians.gaussian_model import GaussianModel, MAX_DEPTH
from sprim.utils.io_utils import (
    load_camera_from_file,
    load_camera_from_dict,
    get_camera_dict,
    resolve_screenshot_path,
)
from sprim.gaussians.path_creator import PathCreator
from sprim.gaussians.global_state import GLOBAL_STATE

from nerfstudio.cameras.cameras import Cameras

# Ratio used for adaptive fps
RATIO_INCREMENT = 0.1
RATIO_MIN, RATIO_MAX = 1.0, 6.0
# Tolerance in framerate
FPS_TOLERANCE = 2.0

sign = lambda x: 1.0 if (x >= 0.0) else -1.0

# import faulthandler

# faulthandler.enable()


class Viewer:
    def __init__(self, input_path: str | None, gca_path: str | None) -> None:

        self.device = "cuda"
        self.input_path = input_path

        ps.init()

        self.primitive_library = PrimitiveLibrary(
            self.load_from_entry, self.reset_viewer
        )

        self.reset_viewer()

        # TODO: refactor render mode with Enum
        self.render_mode: int = RenderMode.RGB
        self.render_background: List[int] = [1.0, 1.0, 1.0]
        self.screenshot_ratio: float = 1.0
        self.target_fps: float = 30
        self.render_envmap: bool = True
        self.fps = self.target_fps
        self.feat_type: int = 0
        self.feat_quantize: bool = True
        self.screenshot_requested: bool = False
        self.camera_path: str = "camera.npz"
        self.max_scale: float = 1.0
        self.cam_sensitivity: float = 0.02
        self.recording_fps: int = 25
        self.recording: bool = False
        self.recording_stream = None
        self.last_recorded_time: float = 0.0

        if input_path is not None:
            if os.path.splitext(input_path)[-1] == ".snp":
                self.primitive_library.load_session(
                    session_path=input_path, load_camera=self.load_camera
                )
            else:
                self.primitive_library.load_entry(self.input_path, gca_path)

        self.path_creator = PathCreator()

        ps.set_program_name("Specialized Generative Primitives")
        ps.set_SSAA_factor(4)

        self.ps_init()

        ps.set_user_callback(self.ps_callback)
        ps.show()

    def reset_viewer(self):
        self.training = False
        self.step = 0

        self.rendered_gaussian_model: GaussianModel = None
        self.primitive_entry: PrimitiveEntry = None
        self.suggestive_selector = None
        self.brush_creator = None

    def load_from_entry(
        self, rendered_gaussian_model: GaussianModel, primitive_entry: PrimitiveEntry
    ):
        # TODO: warning this is extremely DIY!
        self.rendered_gaussian_model = (
            rendered_gaussian_model
            if primitive_entry.trainer is None
            else primitive_entry.gaussian_model
        )

        if primitive_entry.trainer is not None:
            self.training = True

        self.primitive_entry = primitive_entry
        self.config = primitive_entry.config

        # WARNING: both the SuggestiveSelection and the LatentExporter take "rendered Gaussians"
        self.suggestive_selector = SuggestiveSelection(
            self.rendered_gaussian_model,
            self.config,
            self.primitive_entry.brush_painter,
        )

        self.latent_exporter = LatentExporterGaussians(
            self.device,
            self.rendered_gaussian_model,
            self.config,
            self.suggestive_selector,
            self.get_mask,
            self.primitive_library,
        )

        self.brush_creator = BrushCreator(
            self.rendered_gaussian_model,
            config=self.config,
            brush_painter=self.primitive_entry.brush_painter,
        )

    def update_render_sizes(self):
        self.window_size = ps.get_window_size()
        self.buffer_size = (
            int(self.window_size[0]),
            int(self.window_size[1]),
        )

    def ps_init(self):
        ps.set_ground_plane_mode("none")
        ps.set_up_dir("z_up")
        ps.set_front_dir("z_front")
        ps.set_max_fps(120)
        # Anti-aliasing
        ps.set_SSAA_factor(4)
        # Prevent polyscope from changing scales (including Gizmo!)
        ps.set_automatically_compute_scene_extents(False)

        # ps.set_window_size(1280, 720)
        self.update_render_sizes()
        self.init_render_buffer()

        self.last_time = time.time()

    def ps_callback(self):

        new_time = time.time()
        self.fps = 1.0 / (new_time - self.last_time)
        self.last_time = new_time

        if self.training:

            # TODO:
            # if self.trainer is None:
            #     self.primitive_entry.gaussian_model = self.init_trainer(
            #         self.primitive_entry.gaussian_model
            #     )
            #     self.rendered_gaussian_model = self.primitive_entry.gaussian_model

            self.training = self.primitive_entry.trainer.training_step()
            self.step = self.primitive_entry.trainer.step

        self.handle_control()
        self.gui()

        if self.rendered_gaussian_model is not None:

            self.draw()
            self.latent_exporter.gui()

            if self.brush_creator is not None:
                self.brush_creator.gui()

            if self.suggestive_selector is not None:
                self.suggestive_selector.gui()

            if self.primitive_entry.brush_painter is not None:
                self.primitive_entry.brush_painter.gui()

        if self.path_creator is not None:
            self.path_creator.gui()

        KEY_HANDLER.step()

    def init_render_buffer(self):
        # print(
        #     f"Initialized render_buffer with shape: {(self.buffer_size[1], self.buffer_size[0], 4)}"
        # )
        self.render_buffer_quantity = ps.add_raw_color_alpha_render_image_quantity(
            "render_buffer",
            MAX_DEPTH
            * np.ones((self.buffer_size[1], self.buffer_size[0]), dtype=float),
            np.zeros((self.buffer_size[1], self.buffer_size[0], 4), dtype=float),
            enabled=True,
            allow_fullscreen_compositing=True,
        )

        self.render_buffer = ps.get_quantity_buffer("render_buffer", "colors")
        self.render_buffer_depth = ps.get_quantity_buffer("render_buffer", "depths")

    def get_mask(self):
        mask = self.primitive_library.get_mask()
        if self.suggestive_selector is not None:
            # Force reset if size do not match
            # TODO: do proper state handling to avoid having to go through this...
            mask = self.suggestive_selector.get_mask(
                render_mode=self.render_mode, mask=mask
            )
        return mask

    @torch.no_grad()
    def draw(self):
        # Handle window resize
        if ps.get_window_size() != self.window_size:
            self.update_render_sizes()
            self.init_render_buffer()

        camera = camera_from_ps().to(self.device)

        self.rendered_gaussian_model.eval()

        if self.screenshot_requested:
            camera.rescale_output_resolution(self.screenshot_ratio)

        background_color = torch.Tensor(self.render_background).cuda()
        background_feat = torch.zeros(self.config.feature_dim, device="cuda")

        render_pkg = self.rendered_gaussian_model.render(
            camera,
            self.config,
            return_feat=self.render_mode == RenderMode.FEATURES,
            return_depth=self.render_mode == RenderMode.DEPTH or GLOBAL_STATE.use_depth,
            render_envmap=self.render_envmap,
            mask=self.get_mask(),
            background_color=background_color,
            background_feat=background_feat,
            max_scale=self.max_scale,
        )

        alpha = render_pkg["alpha"]
        if self.render_mode == RenderMode.RGB:
            rgb = render_pkg["rgb"]
            rendered_image = torch.cat(
                [
                    rgb,
                    alpha,
                ],
                dim=-1,
            )
        elif self.render_mode == RenderMode.DEPTH:
            rgb = render_pkg["depth"].repeat((1, 1, 3))
            rgb /= rgb.max()
            rendered_image = torch.cat(
                [
                    rgb,
                    alpha,
                ],
                dim=-1,
            )
        else:
            feat = render_pkg["feat"]
            feat_shape = feat.shape
            if (
                self.feat_type == 0
                and self.rendered_gaussian_model.feature_quantizer is not None
                and self.feat_quantize
            ):
                # Don't forget to freeze the codebook!
                feat, quantization_indices, _ = (
                    self.rendered_gaussian_model.feature_quantizer(
                        feat.view((-1, self.rendered_gaussian_model.features_dim)),
                        freeze_codebook=True,
                    )
                )
                feat = feat.view(feat_shape)
                quantization_indices = quantization_indices.view(
                    (feat_shape[0], feat_shape[1])
                )

            rendered_feat = self.rendered_gaussian_model.render_pca.render(feat)

            if (
                self.feat_type == 0
                and self.rendered_gaussian_model.feature_quantizer is not None
                and self.feat_quantize
                and self.suggestive_selector is not None
            ):
                rendered_feat = self.suggestive_selector.postprocess_feature(
                    feat, quantization_indices, rendered_feat
                )

            rendered_image = torch.cat(
                [
                    rendered_feat,
                    alpha,
                ],
                dim=-1,
            )

        if self.screenshot_requested:
            import imageio

            self.screenshot_requested = False
            screenshot_path = resolve_screenshot_path()
            export_img = (rendered_image.cpu().numpy() * 255).astype(np.uint8)
            imageio.imwrite(screenshot_path, export_img)

        else:
            self.render_buffer.update_data_from_device(rendered_image)
            if GLOBAL_STATE.use_depth:
                self.render_buffer_depth.update_data_from_device(
                    render_pkg["depth"].squeeze(-1)
                )
            else:
                self.render_buffer_depth.update_data_from_device(
                    MAX_DEPTH
                    * torch.ones(
                        (rendered_image.shape[0], rendered_image.shape[1]),
                        device="cuda",
                    )
                )
            # print(rendered_image.stride())

            if self.recording_stream is not None:
                current_time = time.time()
                if (current_time - self.last_recorded_time) >= 1.0 / float(
                    self.recording_fps
                ):
                    self.recording_stream.write_video_chunk(
                        0,
                        (
                            rendered_image[..., :3].permute(2, 0, 1).unsqueeze(0) * 255
                        ).to(torch.uint8),
                    )
                    self.last_recorded_time = current_time

            if self.path_creator is not None:
                self.path_creator.render_callback(rendered_image)

    def show_cameras(self):
        # First, we need to load cameras...
        cameras = SubjectLoader.load_cameras(self.config)
        c2w = cameras.camera_to_worlds
        self.c2w = torch.cat(
            [
                c2w,
                torch.tensor([[[0, 0, 0, 1]]], device=c2w.device).repeat(
                    c2w.shape[0], 1, 1
                ),
            ],
            dim=1,
        ).to(self.device)
        self.camera_centers = self.c2w[:, :3, 3]

        ps.register_point_cloud("cameras", self.camera_centers.cpu().numpy())

    def save_camera(self):
        camera_dict = get_camera_dict()
        np.savez(self.camera_path, **camera_dict)
        print(f"Saved camera at {self.camera_path}")

    def load_camera(self, data=None):
        if data is None:
            params, window_size = load_camera_from_file(self.camera_path)
        else:
            params, window_size = load_camera_from_dict(data)

        if params is None:
            print(f"No camera file at {self.camera_path}")
            return

        if window_size is not None:
            ps.set_window_size(window_size[0], window_size[1])
            self.update_render_sizes()
            self.init_render_buffer()

        ps.set_view_camera_parameters(params)

    def _move(self, dir: str):
        cam_params = ps.get_view_camera_parameters()
        look_dir = cam_params.get_look_dir()
        up_dir = cam_params.get_up_dir()
        right_dir = cam_params.get_right_dir()
        cam_pos = cam_params.get_position()
        if dir == "forward":
            cam_pos += self.cam_sensitivity * look_dir
        elif dir == "backward":
            cam_pos -= self.cam_sensitivity * look_dir
        elif dir == "right":
            cam_pos += self.cam_sensitivity * right_dir
        elif dir == "left":
            cam_pos -= self.cam_sensitivity * right_dir
        elif dir == "up":
            cam_pos += self.cam_sensitivity * up_dir
        elif dir == "down":
            cam_pos -= self.cam_sensitivity * up_dir
        new_params = ps.CameraParameters(
            intrinsics=cam_params.get_intrinsics(),
            extrinsics=ps.CameraExtrinsics(cam_pos, look_dir, up_dir),
        )
        ps.set_view_camera_parameters(new_params)

    def handle_control(self):

        io = psim.GetIO()
        if any([io.KeysDown[KEYMAP[key]] for key in ["b", "n", "m"]]):
            return

        if io.KeysDown[KEYMAP["up_arrow"]]:
            self._move("forward")
        elif io.KeysDown[KEYMAP["down_arrow"]]:
            self._move("backward")
        elif io.KeysDown[KEYMAP["right_arrow"]]:
            self._move("right")
        elif io.KeysDown[KEYMAP["left_arrow"]]:
            self._move("left")
        elif io.KeysDown[KEYMAP["rshift"]]:
            self._move("up")
        elif io.KeysDown[KEYMAP["rctrl"]]:
            self._move("down")

    @torch.no_grad()
    def gui(self):

        psim.Text(f"fps: {self.fps:.4f}; step: {self.step:07d}")

        if psim.BeginMenuBar():
            if psim.BeginMenu("Session"):
                self.primitive_library.save_gui(load_camera=self.load_camera)
                psim.EndMenu()
            if psim.BeginMenu("Library"):
                if psim.Button("Reload##reload_library"):
                    self.primitive_library.load_library()
                psim.EndMenu()
            if psim.BeginMenu("Tools"):
                self.primitive_library.tools_gui()
                psim.EndMenu()
            psim.EndMenuBar()

        if self.rendered_gaussian_model is not None:
            psim.Text(f"gaussians: {len(self.rendered_gaussian_model.means):08d}")

            _, self.training = state_button(self.training, "Stop", "Train")

            psim.SameLine()
            if psim.Button("Shot") or KEY_HANDLER("t"):
                self.screenshot_requested = True
            psim.SameLine()
            # NOTE: I add to change this for Ctrl+y
            if psim.Button("Shot(PS)") or KEY_HANDLER("u"):
                ps.screenshot(resolve_screenshot_path())
            psim.SameLine()
            clicked, self.recording = state_button(self.recording, "Stop", "Record")
            if clicked:
                if self.recording and self.recording_stream is None:
                    from torchaudio.io import StreamWriter

                    recording_path = resolve_screenshot_path(is_video=True)

                    self.recording_stream = StreamWriter(recording_path)
                    cuda_conf = {
                        "encoder": "h264_nvenc",  # Use CUDA HW decoder
                        "encoder_format": "rgb0",
                        "encoder_option": {"gpu": "0"},
                        "hw_accel": "cuda:0",
                        "height": self.window_size[1],
                        "width": self.window_size[0],
                    }

                    self.recording_stream.add_video_stream(
                        frame_rate=self.recording_fps, **cuda_conf
                    )
                    self.recording_stream.open()
                    self.last_recorded_time = time.time()
                elif not self.recording and self.recording_stream is not None:
                    self.recording_stream.close()
                    self.recording_stream = None

            psim.Separator()

        psim.Separator()

        if self.rendered_gaussian_model is not None:

            # Shortcuts
            # print([i for i, val in enumerate(psim.GetIO().KeysDown) if val])
            if KEY_HANDLER("f"):
                self.render_mode = RENDER_MODE_INVMAP[
                    1 - RENDER_MODE_MAP[self.render_mode]
                ]
                if (
                    self.feat_quantize
                    and self.suggestive_selector is not None
                    and self.rendered_gaussian_model.feature_quantizer is None
                ):
                    self.suggestive_selector.update_quantizer()

            if KEY_HANDLER("q"):
                self.feat_quantize = not self.feat_quantize
                if (
                    self.feat_quantize
                    and self.suggestive_selector is not None
                    and self.rendered_gaussian_model.feature_quantizer is None
                ):
                    self.suggestive_selector.update_quantizer()

            if psim.TreeNode("Rendering"):

                clicked, render_mode_idx = psim.SliderInt(
                    "Render Mode##viewer",
                    RENDER_MODE_MAP[self.render_mode],
                    v_min=0,
                    v_max=len(RenderMode) - 1,
                    format=f"{self.render_mode.value}",
                )
                if clicked:
                    self.render_mode = RENDER_MODE_INVMAP[render_mode_idx]

                _, GLOBAL_STATE.use_depth = psim.Checkbox(
                    "use_depth##viewer", GLOBAL_STATE.use_depth
                )

                _, GLOBAL_STATE.selection_color = psim.ColorEdit3(
                    "selection_color##viewer",
                    GLOBAL_STATE.selection_color,
                )

                _, self.cam_sensitivity = psim.SliderFloat(
                    "sensitivity##viewer", self.cam_sensitivity, v_min=0.005, v_max=0.2
                )

                _, self.recording_fps = psim.SliderInt(
                    "recording_fps##viewer", self.recording_fps, v_min=20, v_max=60
                )

                _, self.render_background = psim.ColorEdit3(
                    "background##viewer", self.render_background
                )

                _, self.max_scale = psim.SliderFloat(
                    "max_scale##viewer", self.max_scale, v_min=0.0, v_max=1.0, power=-2
                )

                _, self.screenshot_ratio = psim.SliderFloat(
                    "screenshot_ratio##viewer",
                    self.screenshot_ratio,
                    v_min=1.0,
                    v_max=4.0,
                )

                if self.render_mode == RenderMode.FEATURES:
                    _, self.feat_type = psim.SliderInt(
                        "feat_type", self.feat_type, v_min=0, v_max=1
                    )
                    if self.feat_type == 0:
                        _, self.feat_quantize = psim.Checkbox(
                            "feat_quantize", self.feat_quantize
                        )

                if self.rendered_gaussian_model.envmap is not None:
                    _, self.render_envmap = psim.Checkbox(
                        "render_envmap", self.render_envmap
                    )

                clicked, window_size = psim.InputInt2(
                    "resolution", (int(self.window_size[0]), int(self.window_size[1]))
                )
                if clicked:
                    ps.set_window_size(window_size[0], window_size[1])
                    self.update_render_sizes()
                    self.init_render_buffer()

                requested, self.camera_path = save_popup(
                    popup_name="save_cam", path=self.camera_path, save_label="Save cam"
                )
                if requested:
                    self.save_camera()
                psim.SameLine()
                requested, self.camera_path = save_popup(
                    popup_name="load_cam",
                    path=self.camera_path,
                    save_label="Load cam",
                    show_warning=False,
                )
                if requested:
                    self.load_camera()
                psim.SameLine()
                if psim.Button("Train cams"):
                    self.show_cameras()

                psim.TreePop()

            psim.Separator()

        if self.rendered_gaussian_model is None:
            psim.SetNextItemOpen(True, psim.ImGuiCond_Once)
        self.primitive_library.gui()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default=None)
    parser.add_argument("--gca", type=str, default=None)
    parser.add_argument("--prim", type=str, nargs="+", default=[])

    args = parser.parse_args()

    Viewer(args.input, args.gca)
