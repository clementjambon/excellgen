import os
from typing import List, Dict, Any

import torch
import deepdish as dd

import polyscope as ps
import polyscope.imgui as psim

from sprim.gaussians.brush import Brush
from sprim.configs.base import BaseConfig
from sprim.utils.process_utils import apply_transform
from sprim.utils.gui_utils import state_button, save_popup, KEY_HANDLER, KEYMAP
from sprim.utils.voxel_set import VoxelSet

BRUSH_PAINTING_FOLDER = "brush_painting"
EXEMPLAR_BRUSHES_FOLDER = "exemplar_brushes"
DEFAULT_PAINTING_PATH = "voxel_input.painting"
DEFAULT_PAINTING_BRUSH_PATH = "voxel_input.npz"


class BrushPainter:

    current_brush: Brush | None

    def __init__(
        self,
        config: BaseConfig,
        brush_paths: List[str],
        primitive_entry,
        painting_path: str = None,
        headless_mode: bool = False,
    ) -> None:
        self.primitive_entry = primitive_entry
        self.brush_folder = os.path.join(
            self.primitive_entry.config.log_dir, EXEMPLAR_BRUSHES_FOLDER
        )

        self.all_coarse_voxelset = None
        self.current_brush = None

        self.load_brushes(brush_paths)

        self.edit_stamp = 0
        self.painting_path = DEFAULT_PAINTING_PATH
        self.painting_brush_path = DEFAULT_PAINTING_BRUSH_PATH

        self.requested_patch = False

        if not headless_mode:
            ps.set_drop_callback(self.ps_drop_callback)

        self.log_dir = config.log_dir

        if painting_path is not None:
            self.painting_path = painting_path
            self.load(headless_mode)

    def load_brushes(self, brush_paths: List[str] | None = None) -> None:

        # ----------------------------------------
        # First, remove anything if necessary
        # ----------------------------------------

        if self.current_brush is not None:
            self.current_brush.remove()
            self.current_brush = None

        if self.all_coarse_voxelset is not None:
            self.all_coarse_voxelset.remove()
            self.all_coarse_voxelset = None

        # ----------------------------------------
        # Then, load_brushes
        # ----------------------------------------

        # Read paths already in default folder
        self.brush_paths = (
            (brush_paths if brush_paths is not None else [])
            + [
                os.path.join(self.brush_folder, name)
                for name in os.listdir(self.brush_folder)
            ]
            if os.path.exists(self.brush_folder)
            else []
        )
        self.brush_paths = list(reversed(sorted(self.brush_paths)))

        self.current_brush_idx = 0
        self.brush_names = []
        for path in self.brush_paths:
            self.brush_names.append(os.path.basename(path))

        self.reset_painting()

    def load_current_brush(self):
        if self.current_brush is not None:
            self.current_brush.remove()
            self.current_brush = None

        self.current_brush = Brush.import_brush(
            self.brush_paths[self.current_brush_idx],
            self.primitive_entry,
            self.primitive_entry.transform,
        )
        self.current_brush.display()

        if self.all_coarse_voxelset is not None:
            self.all_coarse_voxelset.set_enabled(True)

    def add_current_points(self):
        assert self.current_brush is not None

        print(self.current_brush.get_transform())

        transformed_pos = apply_transform(
            self.current_brush.points, self.current_brush.get_transform()
        )

        self.all_points.append(transformed_pos)

    def _clear_current_brush(self):
        self.current_brush.remove()
        self.current_brush = None

    def clear(self):
        if self.current_brush is not None:
            self._clear_current_brush()
        if self.all_coarse_voxelset is not None:
            self.all_coarse_voxelset.remove()
            self.all_coarse_voxelset = None
        self.reset_painting()

    def reset_painting(self):
        self.all_points = []

        self.surface_voxels = None
        self.surface_voxels_prepared_stamp = -1

        # This will automatically reset the structures anyway
        if self.current_brush is not None:
            self.current_brush.display()
            # self.current_brush.set_transform()

        if self.all_coarse_voxelset is not None:
            self.all_coarse_voxelset.remove()
            self.all_coarse_voxelset = None

    def _prepare(self):
        if self.all_coarse_voxelset is not None:
            self.primitive_entry.grower.prepare_state_custom(
                coarse_voxels=self.all_coarse_voxelset.voxels
            )
            self.all_coarse_voxelset.set_enabled(False)
        else:
            self.surface_voxels = self.voxelize(self.primitive_entry.grower.voxel_res)
            self.primitive_entry.grower.prepare_state_custom(
                surface_voxels=self.surface_voxels
            )
        self.surface_voxels_prepared_stamp = self.edit_stamp
        if self.current_brush is not None:
            self.current_brush.hide()

    def _grow(self):
        if self.surface_voxels_prepared_stamp != self.edit_stamp:
            self._prepare()
        if self.all_coarse_voxelset is not None:
            self.all_coarse_voxelset.set_enabled(False)
        self.primitive_entry.grower.restart()

    def ps_drop_callback(self, path: str) -> None:

        # Create new Brush
        self.current_brush = Brush.from_mesh(
            path,
            self.primitive_entry,
        )

        if self.current_brush is None:
            return

        self.current_brush.display()

        # Set to -1 for the GUI list
        self.current_brush_idx = -1

    def save(self):
        data = self.serialize()
        if self.log_dir is not None:
            os.makedirs(
                os.path.join(self.log_dir, BRUSH_PAINTING_FOLDER), exist_ok=True
            )
        save_path = (
            os.path.join(self.log_dir, BRUSH_PAINTING_FOLDER, self.painting_path)
            if self.log_dir is not None
            else self.painting_path
        )
        dd.io.save(save_path, data)

    def save_brush(self):
        if len(self.all_points) == 0:
            print("Cannot save empty brush!")
            return

        if self.log_dir is not None:
            os.makedirs(
                os.path.join(self.log_dir, EXEMPLAR_BRUSHES_FOLDER), exist_ok=True
            )

        save_path = (
            os.path.join(
                self.log_dir, EXEMPLAR_BRUSHES_FOLDER, self.painting_brush_path
            )
            if self.log_dir is not None
            else self.painting_brush_path
        )

        toy_brush = Brush(torch.cat(self.all_points), None)
        toy_brush.export_brush(save_path)

    def load(self, headless_mode: bool = False):
        self.clear()
        if headless_mode:
            load_path = self.painting_path
        else:
            load_path = (
                os.path.join(self.log_dir, BRUSH_PAINTING_FOLDER, self.painting_path)
                if self.log_dir is not None
                else self.painting_path
            )
        if not os.path.exists(load_path):
            print(f"Could not open painting at {load_path}")
            return
        self.load_serialized(dd.io.load(load_path), headless_mode=headless_mode)

    def gui(self):
        io = psim.GetIO()

        if len(self.brush_names) > 0:
            psim.SetNextItemOpen(True, psim.ImGuiCond_Once)

        if psim.TreeNode("Brush Painter"):

            clicked, new_brush_idx = psim.ListBox(
                "##brush_painter_primlist",
                self.current_brush_idx,
                self.brush_names,
            )

            if clicked:
                if new_brush_idx != self.current_brush_idx:
                    self.current_brush_idx = new_brush_idx
                    # self.clear()
                    self.load_current_brush()

                if self.current_brush is None:
                    self.load_current_brush()

            if self.current_brush is not None:
                if self.current_brush.enabled:
                    self.current_brush.gui()

                if KEY_HANDLER("a"):
                    self.edit_stamp += 1

                    if self.current_brush.enabled:
                        self.add_current_points()

                        coarse_voxels = self.voxelize(
                            self.primitive_entry.grower.coarse_res
                        )
                        self.all_coarse_voxelset = VoxelSet(
                            coarse_voxels,
                            get_transform=self.primitive_entry.get_transform,
                            res=self.primitive_entry.grower.coarse_res,
                            bbox_min=self.primitive_entry.bbox_min,
                            bbox_max=self.primitive_entry.bbox_max,
                            prefix="all_coarse_voxelset",
                            enabled=True,
                            add_erase_voxel_callback=self.add_erase_voxel_callback,
                        )
                    # Otherwise, simply show again
                    else:
                        self.current_brush.show()
                        if self.all_coarse_voxelset is not None:
                            self.all_coarse_voxelset.set_enabled(True)

                if (
                    self.all_coarse_voxelset is not None
                    and self.all_coarse_voxelset.is_enabled()
                ):
                    self.all_coarse_voxelset.gui()

            # TODO: handle selected brush
            if self.current_brush is None and psim.Button("Start##brush_painter"):

                self.load_current_brush()

            if self.current_brush is not None and psim.Button(
                "Clear all##brush_painter"
            ):
                self.clear()

            psim.SameLine()
            requested, self.painting_path = save_popup(
                "load_painting", self.painting_path, "Load", show_warning=False
            )
            if requested:
                self.load()

            if len(self.all_points) > 0:

                psim.SameLine()

                requested, self.painting_path = save_popup(
                    "save_painting", self.painting_path, "Save"
                )
                if requested:
                    self.save()

                psim.SameLine()
                requested, self.painting_brush_path = save_popup(
                    "save_painting_brush", self.painting_brush_path, "Save(brush)"
                )
                if requested:
                    self.save_brush()

                if psim.Button("Reset##brush_painter") or KEY_HANDLER("r"):

                    self.reset_painting()

                psim.SameLine()

                if psim.Button("Grow##brush_painter") or KEY_HANDLER("g"):
                    self._grow()
                    self.requested_patch = True

                if self.surface_voxels_prepared_stamp >= 0:
                    psim.SameLine()
                    if psim.Button("Patch##brush_painter") or KEY_HANDLER("p"):
                        self.primitive_entry.grower.patch_current_state()
                        self.requested_patch = False

                    if not self.primitive_entry.grower.grow() and self.requested_patch:
                        self.primitive_entry.grower.patch_current_state()
                        self.requested_patch = False

                if self.all_coarse_voxelset is not None:
                    clicked, enabled = state_button(
                        self.all_coarse_voxelset.is_enabled(),
                        "Hide painting##brush_painter",
                        "Show painting##brush_painter",
                    )
                    if clicked:
                        self.all_coarse_voxelset.set_enabled(enabled)

                if (
                    self.primitive_entry.grower is not None
                    and len(self.primitive_entry.grower.grown_voxels) > 0
                ):
                    self.primitive_entry.grower.gui()

            psim.TreePop()

        psim.Separator()

    # TODO: solve res ambiguity here
    def voxelize(self, res) -> torch.Tensor:
        bbox_min, bbox_max = (
            self.primitive_entry.bbox_min,
            self.primitive_entry.bbox_max,
        )

        surface_points = torch.cat(self.all_points, dim=0)

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

    def add_erase_voxel_callback(self, voxels: torch.Tensor) -> None:
        if self.all_coarse_voxelset is None:
            return

        # Convert to world position the voxels
        all_new_points = self.all_coarse_voxelset.voxel_to_world(voxels.float() + 0.5)

        # NOTE: this will erase all points!
        self.all_points = [all_new_points]

    def serialize(self) -> Dict[str, Any]:

        data = {}

        if self.all_points is not None:
            data["all_points"] = [x.cpu().numpy() for x in self.all_points]

        if self.all_coarse_voxelset is not None:
            data["coarse_voxels"] = self.all_coarse_voxelset.voxels.cpu().numpy()

        return data

    def load_serialized(self, data: Dict[str, Any], headless_mode: bool = False):

        if "all_points" in data and len(data["all_points"]) > 0:
            self.all_points = [torch.tensor(x).cuda() for x in data["all_points"]]

            coarse_voxels = self.voxelize(self.primitive_entry.grower.coarse_res)
            self.all_coarse_voxelset = VoxelSet(
                coarse_voxels,
                get_transform=self.primitive_entry.get_transform,
                res=self.primitive_entry.grower.coarse_res,
                bbox_min=self.primitive_entry.bbox_min,
                bbox_max=self.primitive_entry.bbox_max,
                prefix="all_coarse_voxelset",
                enabled=len(self.primitive_entry.grower.grown_voxels) == 0,
                add_erase_voxel_callback=self.add_erase_voxel_callback,
            )

        # if "coarse_voxels" in data and not headless_mode:
        #     self.all_coarse_voxelset = VoxelSet(
        #         torch.tensor(data["coarse_voxels"]).int().cuda(),
        #         get_transform=self.primitive_entry.get_transform,
        #         res=self.primitive_entry.grower.coarse_res,
        #         bbox_min=self.primitive_entry.bbox_min,
        #         bbox_max=self.primitive_entry.bbox_max,
        #         prefix="all_coarse_voxelset",
        #         enabled=True,
        #         add_erase_voxel_callback=self.add_erase_voxel_callback,
        #     )
