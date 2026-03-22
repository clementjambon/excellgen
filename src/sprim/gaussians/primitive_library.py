import os
import glob
from typing import List, Callable

from tqdm import tqdm

import torch
import numpy as np

import deepdish as dd

import polyscope as ps
import polyscope.imgui as psim

from sprim.gaussians.primitive_entry import PrimitiveEntry
from sprim.gaussians.tree_node import TreeNode
from sprim.utils.io_utils import get_camera_dict
from sprim.gaussians.gaussian_model import GaussianModel, GaussianSet
from sprim.utils.gui_utils import colored_button, save_popup, KEY_HANDLER, KEYMAP
from sprim.utils.process_utils import apply_transform
from sprim.gaussians.envmap import EnvMap
from sprim.gaussians.pc_selector import PcSelector
from sprim.gaussians.tonemapper import Tonemapper
from sprim.gaussians.global_state import GLOBAL_STATE

import viser.transforms as tf

LIBRARY_PATH = os.path.expandvars(os.environ.get("PRIMITIVES_ROOT", "primitives"))
DEFAULT_SESSION_PATH = "sessions/default.snp"
DEFAULT_ENVMAP_PATH = "data/envmaps/default.png"

MAX_LIBRARY_CATEGORY = 15
MAX_DISPLAYED_LAYERS = 10


class PrimitiveLibrary:

    def __init__(
        self,
        load_from_entry: Callable[[GaussianModel, PrimitiveEntry], None],
        reset_viewer: Callable[[], None],
    ) -> None:

        self.load_from_entry = load_from_entry
        self.reset_viewer = reset_viewer

        self.load_library()

        self.loaded_entries: List[PrimitiveEntry] = []
        self.loaded_offsets: List[int] = []
        self.loaded_mask: torch.Tensor | None = None
        self.loaded_current: int = -1

        # DEBUG
        self.ref_idx: int = 0
        self.filtering_res: int = 64

        # The rendered gaussian model is the full one with every points!
        self.rendered_gaussian_model = None
        self.moving = False
        self.moving_with_keys = False
        self.prev_scale = 1.0

        self.session_path = DEFAULT_SESSION_PATH

        self.envmap_path = DEFAULT_ENVMAP_PATH
        self.envmap: EnvMap | None = None

        self.large_pc_selector: PcSelector | None = None
        self.large_min_scale: float = 0.1

        self.show_bbox_transform: bool = True

        self.merge_target_idx = 0

        # import imageio

        # self.image_preview = (
        #     np.array(imageio.imread("test.png")).astype(np.float32) / 255.0
        # )
        # self.test_buffer_quantity = ps.add_color_image_quantity(
        #     "test_buffer",
        #     self.image_preview[..., :3],
        # )

    def load_library(self):
        # Crawl inside the library path to find models
        self.snapshots = []
        for root, _, files in os.walk(LIBRARY_PATH):
            if "config_gaussians.yaml" in files:
                self.snapshots.append(root)
            elif os.path.basename(root) == "gaussian_ckpt":
                for file in files:
                    self.snapshots.append(os.path.join(root, file))
        self.snapshots = sorted(self.snapshots)
        self.snapshots = ["." + x[len(LIBRARY_PATH) :] for x in self.snapshots]

        # We need to collect parents in order to build TreeNodes
        self.snapshots_with_parents = set()
        for snapshot in self.snapshots:
            current_path = snapshot
            while len(current_path) > 0 or current_path in self.snapshots_with_parents:
                self.snapshots_with_parents.add(current_path)
                current_path = os.path.dirname(current_path)
        self.snapshots_with_parents = list(sorted(self.snapshots_with_parents))

        # Add depth
        self.snapshots_with_parents_w_depth = []
        for snapshot in self.snapshots_with_parents:
            snapshot_w_depth = (len(snapshot.split("/")), snapshot)
            self.snapshots_with_parents_w_depth.append(snapshot_w_depth)
        self.snapshots_with_parents_w_depth = sorted(
            self.snapshots_with_parents_w_depth
        )

        # Filter 'latents_gca' and 'gaussian_ckpt'
        self.snapshots_with_parents_w_depth = [
            (depth, snapshot)
            for depth, snapshot in self.snapshots_with_parents_w_depth
            if not (
                snapshot.endswith("latents_gca") or snapshot.endswith("gaussian_ckpt")
            )
        ]

        # Create TreeNode with that
        tree_node_dict = {}
        self.tree_nodes = []
        self.loadable_tree_nodes = []

        self.n_scene_categories = 0

        for i, (depth, snapshot) in enumerate(self.snapshots_with_parents_w_depth):

            is_snapshot = snapshot in self.snapshots
            snapshot_path = os.path.join(LIBRARY_PATH, snapshot)
            gca_path = None
            ckpt_split = os.path.basename(snapshot).split(".")

            if depth <= 3:
                self.n_scene_categories += 1

            if is_snapshot and len(ckpt_split) > 1 and ckpt_split[1] == "pt":
                ref_dir = os.path.dirname(os.path.dirname(snapshot_path))
                gca_dir = os.path.join(ref_dir, "gca_logs", ckpt_split[0])
                gca_paths = list(sorted(glob.glob(os.path.join(gca_dir, "ckpts", "*"))))
                if len(gca_paths) > 0:
                    gca_path = gca_paths[-1]

            # Only create new node if

            new_node = TreeNode(
                os.path.basename(snapshot),
                snapshot,
                snapshot_path,
                load_callback=self.load_entry,
                is_snapshot=is_snapshot,
                gca_path=gca_path,
                depth=depth,
            )
            tree_node_dict[snapshot] = new_node
            self.tree_nodes.append(new_node)
            if is_snapshot:
                self.loadable_tree_nodes.append(new_node)

            parent_dir = os.path.dirname(snapshot)
            if len(parent_dir) > 0:
                # If we're dealing with a snapshot, skip depth
                if parent_dir.endswith("gaussian_ckpt"):
                    parent_dir = os.path.dirname(os.path.dirname(parent_dir))

                tree_node_dict[parent_dir].child_count += 1
                if tree_node_dict[parent_dir].child_idx < 0:
                    tree_node_dict[parent_dir].child_idx = i

        # for tree_node in self.tree_nodes:
        #     print(tree_node)

    @property
    def current_entry(self):
        return (
            self.loaded_entries[self.loaded_current]
            if len(self.loaded_entries) > 0
            else None
        )

    def get_mask(self):
        if self.current_entry is None or self.current_entry.trainer is not None:
            return None
        else:
            return self.loaded_mask

    def update_mask(self):
        self.loaded_mask = torch.zeros(
            self.loaded_offsets[-1], dtype=bool, device="cuda"
        )
        for i, entry in enumerate(self.loaded_entries):
            self.loaded_mask[self.loaded_offsets[i] : self.loaded_offsets[i + 1]] = (
                float(entry.active)
            )

    @torch.no_grad()
    def move_loaded(self):
        if self.moving_with_keys:
            return
        transform = torch.tensor(self.current_entry.bbox.get_transform()).cuda()
        self.current_entry.transform = transform
        self.current_entry.transform_history.record_new(transform)
        self.gaussian_update_callback()

    @torch.no_grad()
    def _updated_gaussian_set(self):
        gaussian_set = None
        current_offset = 0
        self.loaded_offsets = [current_offset]
        for idx, primitive_entry in enumerate(self.loaded_entries):
            primitive_set = (
                primitive_entry.gaussian_model.get_gaussian_set()
                if primitive_entry.gaussian_model.grown_gauss_params is None
                or not primitive_entry.show_grown
                else primitive_entry.gaussian_model.get_gaussian_set(True)
            )
            if (
                primitive_entry.tonemapper is not None
                and len(primitive_entry.tonemapper.active) > 0
            ):
                primitive_set = primitive_entry.tonemapper.apply(primitive_set)

            # Transform if necessary
            primitive_set = primitive_set.transform(primitive_entry.transform)

            size = len(primitive_set)
            current_offset += size
            self.loaded_offsets.append(current_offset)

            if gaussian_set is None:
                gaussian_set = primitive_set
            else:
                gaussian_set = gaussian_set.merge(primitive_set)

        return gaussian_set

    @torch.no_grad()
    def gaussian_update_callback(self):
        gaussian_set = self._updated_gaussian_set()

        self.update_mask()
        self._kill_filter_large()
        self.rendered_gaussian_model.set_gaussian_set(gaussian_set)

        if self.envmap is not None:
            self.rendered_gaussian_model.envmap = self.envmap

    @torch.no_grad()
    def update_viewer(self):
        if len(self.loaded_entries) == 0:
            self.reset_viewer()
            return

        # Update the viewer with all the new entries

        # Concatenate all the entries
        gaussian_set = self._updated_gaussian_set()

        # Keep the current entry
        config = self.current_entry.config

        self.update_mask()
        self._kill_filter_large()
        self.current_entry.update_bbox()

        # Make sure to disable any other bbox
        for i, entry in enumerate(self.loaded_entries):
            if i != self.current_entry and entry.bbox is not None:
                entry.bbox.set_enabled(False)

        # Set the current gaussian set
        gaussian_model = GaussianModel(
            device="cuda",
            # Warning this is hardcoded!
            num_train_data=200,
            sh_degree=config.sh_degree,
            feature_dim=config.feature_dim,
            feature_quantizer=config.feature_quantizer,
        ).to("cuda")

        gaussian_model.set_gaussian_set(gaussian_set)

        if self.envmap is not None:
            gaussian_model.envmap = self.envmap

        # TODO: handle grower
        self.rendered_gaussian_model = gaussian_model

        self.load_from_entry(self.rendered_gaussian_model, self.current_entry)

    def load_entry(self, path: str, gca_path: str = None):

        primitive_entry = PrimitiveEntry.load_entry(
            path=path,
            gca_path=gca_path,
            gaussian_update_callback=self.gaussian_update_callback,
        )

        self.loaded_entries.append(primitive_entry)
        self.loaded_current = len(self.loaded_entries) - 1
        self.update_viewer()

    def remove_entry(self, idx: int):
        if idx < len(self.loaded_entries) and idx >= 0:
            del self.loaded_entries[idx]
            if idx <= self.loaded_current:
                self.loaded_current -= 1
            self.update_viewer()
        else:
            print(f"WARNING: cannot remove entry with index {idx}")

    def prim_layer_name_selectable(self, label: str, idx: int):

        clicked, _ = psim.Selectable(
            label,
            self.loaded_current == idx,
            psim.ImGuiSelectableFlags_AllowDoubleClick,
        )

        popup_name = f"layer_rename##{idx}"
        if clicked and psim.IsMouseDoubleClicked(0):
            psim.OpenPopup(popup_name)
            KEY_HANDLER.lock(popup_name)

        elif clicked and idx != self.loaded_current:
            self.loaded_current = idx

            # This is a little bit brute-force but this will ensure we reset things properly...
            self.update_viewer()

        if psim.BeginPopup(popup_name):

            _, self.loaded_entries[idx].display_name = psim.InputText(
                f"name##layer_rename_{idx}", self.loaded_entries[idx].display_name
            )

            if psim.Button(f"Confirm##{popup_name}"):
                KEY_HANDLER.unlock(popup_name)
                psim.CloseCurrentPopup()

            psim.EndPopup()

    def save_gui(self, load_camera) -> None:

        requested, self.session_path = save_popup(
            popup_name="primitive_library_load",
            path=self.session_path,
            save_label="Load session",
            show_warning=False,
        )
        if requested:
            self.load_session(self.session_path, load_camera)

        if len(self.loaded_entries) > 0:
            psim.SameLine()

            requested, self.session_path = save_popup(
                popup_name="primitive_library_save",
                path=self.session_path,
                save_label="Save session",
            )

            if requested:
                self.save_session()

    def tools_gui(self) -> None:
        # DEBUG: for visualization purposes only
        if self.current_entry is not None:
            if colored_button(
                f"Load Hierarchy (DEBUG)##primitive_library",
                0.0,
            ):
                self.current_entry.load_hierarchy()

            requested, self.envmap_path = save_popup(
                "load_envmap", self.envmap_path, "Load Envmap", show_warning=False
            )
            if requested:
                self.envmap = EnvMap.from_image(self.envmap_path)
                self.gaussian_update_callback()

            if psim.Button("Filter Large##primitive_library"):
                self._init_filter_large()

            if psim.Button("Unlock keys##primitive_library"):
                KEY_HANDLER.unlock_all()

            _, self.show_bbox_transform = psim.Checkbox(
                "Bbox transform##primitive_library", self.show_bbox_transform
            )

            requested, merge_target_idx_str = save_popup(
                "merge_layer",
                str(self.merge_target_idx),
                "Merge Layer",
                show_warning=False,
            )
            try:
                self.merge_target_idx = int(merge_target_idx_str)
            except:
                self.merge_target_idx = 0
            if requested:
                self.merge_current_layer()

            if psim.Button("Reset transform##primitive_library"):
                if self.current_entry is not None:
                    self.current_entry.set_transform(None)
                    self.gaussian_update_callback()

            # Focus camera on the current asset
            if psim.Button("Focus##primitive_library"):
                self.focus_on_current()

    def focus_on_current(self) -> None:
        if self.current_entry is None:
            return
        transform = self.current_entry.get_transform()
        if transform is None:
            transform = torch.eye(4).cuda()

        world_transform_bbox_min = apply_transform(
            self.current_entry.transform_bbox_min, transform
        )
        world_transform_bbox_max = apply_transform(
            self.current_entry.transform_bbox_max, transform
        )
        new_bbox_min = torch.stack(
            [world_transform_bbox_min, world_transform_bbox_max], dim=0
        ).min(0)[0]
        new_bbox_max = torch.stack(
            [world_transform_bbox_min, world_transform_bbox_max], dim=0
        ).max(0)[0]
        mid_point = 0.5 * (new_bbox_min + new_bbox_max)

        ps.set_bounding_box(new_bbox_min.cpu().numpy(), new_bbox_max.cpu().numpy())
        ps.look_at(
            ps.get_view_camera_parameters().get_position(),
            mid_point.cpu().numpy(),
            fly_to=True,
        )

    # -----------------------
    # FILTER LARGE
    # -----------------------

    def _init_filter_large(self):
        self._kill_filter_large()

        # Take only the very large Gaussians
        current_set = self.current_entry.gaussian_model.get_gaussian_set()
        self.large_gaussians_idx = torch.argwhere(
            torch.exp(current_set.scales).max(-1)[0] > self.large_min_scale
        ).squeeze(-1)
        large_gaussians_pos = current_set.means[self.large_gaussians_idx]
        self.large_pc_selector = PcSelector(
            "large_pc_selector",
            large_gaussians_pos,
            transform=self.current_entry.transform,
            enabled=True,
            filter_callback=self._callback_filter_large,
        )

    def _callback_filter_large(self, selected_mask):
        current_set = self.current_entry.gaussian_model.get_gaussian_set()
        filtered_indices = self.large_gaussians_idx[selected_mask]
        all_mask = torch.ones(
            len(current_set.means),
            dtype=torch.bool,
            device="cuda",
        )
        all_mask[filtered_indices] = False

        self.current_entry.gaussian_model.set_gaussian_set(current_set.filter(all_mask))

        # Automatically kill it
        self._kill_filter_large()

        self.gaussian_update_callback()

    def _kill_filter_large(self):
        if self.large_pc_selector is not None:
            self.large_pc_selector.kill()
            self.large_pc_selector = None

    # -----------------------

    def merge_current_layer(self) -> None:
        if (
            self.merge_target_idx < 0
            or self.merge_target_idx >= len(self.loaded_entries)
            or self.merge_target_idx == self.loaded_current
        ):
            print(
                f"Cannot merge layer {self.loaded_current} with {self.merge_target_idx}!"
            )
            return

        current_set = (
            self.current_entry.gaussian_model.get_gaussian_set()
            if self.current_entry.gaussian_model.grown_gauss_params is None
            or not self.current_entry.show_grown
            else self.current_entry.gaussian_model.get_gaussian_set(True)
        )
        current_set = current_set.transform(self.current_entry.transform)

        target_entry = self.loaded_entries[self.merge_target_idx]
        target_set = (
            target_entry.gaussian_model.get_gaussian_set()
            if target_entry.gaussian_model.grown_gauss_params is None
            or not target_entry.show_grown
            else target_entry.gaussian_model.get_gaussian_set(True)
        )
        target_set = target_set.transform(target_entry.transform)

        merged_set = current_set.merge(target_set)

        tmp_current_idx = self.loaded_current

        self.add_layer_from_gaussians(merged_set)

        # Delete both (max first, min then)
        # TODO: do this with UUIDs!
        min_idx = min(tmp_current_idx, self.merge_target_idx)
        max_idx = max(tmp_current_idx, self.merge_target_idx)
        self.remove_entry(max_idx)
        self.remove_entry(min_idx)

    def duplicate_current_layer(self, erase_current: bool = False) -> None:
        primitive_set = (
            self.current_entry.gaussian_model.get_gaussian_set()
            if self.current_entry.gaussian_model.grown_gauss_params is None
            or not self.current_entry.show_grown
            else self.current_entry.gaussian_model.get_gaussian_set(True)
        )

        tmp_idx = self.loaded_current

        self.add_layer_from_gaussians(primitive_set)

        if erase_current:
            self.remove_entry(tmp_idx)

    @torch.no_grad()
    def add_layer_from_gaussians(
        self,
        gaussian_set: GaussianSet,
        tonemapper: Tonemapper | None = None,
        transform: torch.Tensor | None = None,
    ) -> None:
        config = self.current_entry.config

        gaussian_model = GaussianModel(
            device="cuda",
            # Warning this is hardcoded!
            num_train_data=200,
            sh_degree=config.sh_degree,
            feature_dim=config.feature_dim,
            feature_quantizer=config.feature_quantizer,
        ).to("cuda")

        gaussian_model.set_gaussian_set(gaussian_set)

        entry = PrimitiveEntry(
            path=None,
            gca_path=None,
            gaussian_model=gaussian_model,
            config=config,
            grower=None,
            trainer=None,
            brush_painter=None,
            bbox_min=self.current_entry.bbox_min,
            bbox_max=self.current_entry.bbox_max,
            transform=self.current_entry.transform if transform is None else transform,
        )
        # entry.brush_painter = BrushPainter(
        #     config=config, brush_paths=[], primitive_entry=entry
        # )

        if tonemapper is not None:
            entry.tonemapper = tonemapper

        self.loaded_entries.append(entry)
        self.loaded_current = len(self.loaded_entries) - 1
        self.update_viewer()

    def _history_previous(self):
        if self.current_entry is None:
            return
        transform = self.current_entry.transform_history.previous()
        self.current_entry.set_transform(transform)
        self.gaussian_update_callback()

    def _history_next(self):
        if self.current_entry is None:
            return
        transform = self.current_entry.transform_history.next()
        self.current_entry.set_transform(transform)
        self.gaussian_update_callback()

    def gui(self) -> None:

        if KEY_HANDLER("1"):
            self.focus_on_current()

        TEXT_SIZE = psim.CalcTextSize("A")[0]
        TEXT_BASE_HEIGHT = psim.GetTextLineHeightWithSpacing()

        if psim.TreeNode("Library##primitive_library"):

            outer_size = (
                0.0,
                TEXT_BASE_HEIGHT * MAX_LIBRARY_CATEGORY + 1,
            )
            if psim.BeginTable(
                "Primitive Searcher",
                2,
                psim.ImGuiTableFlags_ScrollY,
                outer_size,
            ):
                psim.TableSetupColumn("Name##prim_searcher")
                psim.TableSetupColumn(
                    "Load##prim_searcher",
                    psim.ImGuiTableColumnFlags_WidthFixed,
                    TEXT_SIZE * 5,
                )
                psim.TableHeadersRow()

                if len(self.tree_nodes) > 0:
                    self.tree_nodes[0].display_node(self.tree_nodes)

                psim.EndTable()

            psim.TreePop()

        psim.Separator()

        psim.SetNextItemOpen(len(self.loaded_entries) > 0, psim.ImGuiCond_Always)
        if psim.TreeNode("Layers##primitive_library"):

            if KEY_HANDLER("d"):
                self.duplicate_current_layer()

            if KEY_HANDLER("j"):
                self.duplicate_current_layer(erase_current=True)

            if psim.BeginTable(
                "Primitive Layers##table",
                4,
                psim.ImGuiTableFlags_ScrollY,
                (
                    0,
                    min(
                        TEXT_BASE_HEIGHT * (len(self.loaded_entries) + 1.5),
                        TEXT_BASE_HEIGHT * MAX_DISPLAYED_LAYERS,
                    ),
                ),
            ):
                psim.TableSetupColumn(
                    "Name",
                    psim.ImGuiTableColumnFlags_WidthStretch,
                    0.0,
                )
                psim.TableSetupColumn(
                    "Show",
                    psim.ImGuiTableColumnFlags_WidthFixed
                    | psim.ImGuiTableColumnFlags_NoHide,
                    0.0,
                )
                psim.TableSetupColumn(
                    "Grown",
                    psim.ImGuiTableColumnFlags_WidthFixed
                    | psim.ImGuiTableColumnFlags_NoHide,
                    0.0,
                )
                psim.TableSetupColumn(
                    "Del",
                    psim.ImGuiTableColumnFlags_WidthFixed
                    | psim.ImGuiTableColumnFlags_NoHide,
                    0.0,
                )

                psim.TableHeadersRow()

                for idx, primitive_entry in enumerate(self.loaded_entries):
                    # DEBUG
                    if "bush" in primitive_entry.display_name:
                        continue

                    psim.TableNextRow()

                    # Display Name
                    psim.TableNextColumn()
                    self.prim_layer_name_selectable(
                        f"{primitive_entry.display_name}##prim_layers_path_{idx}", idx
                    )

                    # Show
                    psim.TableNextColumn()
                    clicked, primitive_entry.active = psim.Checkbox(
                        f"##prim_layers_show_{idx}", primitive_entry.active
                    )
                    if clicked:
                        self.update_mask()

                    psim.TableNextColumn()
                    no_grown_params = (
                        primitive_entry.gaussian_model.grown_gauss_params is None
                    )

                    psim.BeginDisabled(no_grown_params)

                    clicked, primitive_entry.show_grown = psim.Checkbox(
                        f"##prim_layers_grown_{idx}", primitive_entry.show_grown
                    )
                    if clicked:
                        self.gaussian_update_callback()

                    psim.EndDisabled()

                    # Del
                    psim.TableNextColumn()
                    if psim.SmallButton(f"X##prim_layers_del_{idx}"):
                        self.remove_entry(idx)
                        break

                psim.EndTable()

            psim.TreePop()

        # --------------------------------
        # ENTRY TRANSFORM
        # --------------------------------

        io = psim.GetIO()

        # History
        if io.KeyCtrl and KEY_HANDLER("z"):
            self._history_previous()
        elif io.KeyCtrl and KEY_HANDLER("y"):
            self._history_next()

        if len(self.loaded_entries) > 0 and (
            io.KeysDown[KEYMAP["m"]]
            or io.KeysDown[KEYMAP["n"]]
            or io.KeysDown[KEYMAP["b"]]
        ):
            self.moving = True
            self.current_entry.bbox.enable_transform_gizmo(True)
            if self.show_bbox_transform:
                self.current_entry.bbox.set_enabled(True)
            if io.KeysDown[KEYMAP["m"]]:
                self.current_entry.bbox.set_transform_mode_gizmo(
                    ps.TransformMode.TRANSLATION | ps.TransformMode.ROTATION
                )
                self.handle_control("translation")
            elif io.KeysDown[KEYMAP["b"]]:
                self.current_entry.bbox.set_transform_mode_gizmo(
                    ps.TransformMode.ROTATION
                )
                self.handle_control("rotation")
            elif io.KeysDown[KEYMAP["n"]]:
                self.current_entry.bbox.set_transform_mode_gizmo(ps.TransformMode.SCALE)
                self.handle_control("scale")
        elif self.current_entry is not None and self.current_entry.bbox is not None:
            if self.moving:
                self.move_loaded()
                self.moving = False
                self.moving_with_keys = False
                self.current_entry.bbox.enable_transform_gizmo(False)
                self.current_entry.bbox.set_enabled(False)

        if self.current_entry is not None:
            if io.KeysDown[KEYMAP["page_up"]]:
                transform = self.current_entry.get_transform()
                if transform is None:
                    transform = torch.eye(4).cuda()
                transform[:3, :3] *= 1.01 ** (1.0 / 3.0)
                self.current_entry.set_transform(transform)
                self.gaussian_update_callback()
            elif io.KeysDown[KEYMAP["page_down"]]:
                transform = self.current_entry.get_transform()
                if transform is None:
                    transform = torch.eye(4).cuda()
                transform[:3, :3] /= 1.01 ** (1.0 / 3.0)
                self.current_entry.set_transform(transform)
                self.gaussian_update_callback()

        psim.Separator()

        # --------------------------------
        # LARGE PC SELECTOR
        # --------------------------------

        if self.large_pc_selector is not None:

            psim.SetNextItemOpen(True, psim.ImGuiCond_Always)
            if psim.TreeNode("Large Selector##primitive_library"):

                _, self.large_min_scale = psim.SliderFloat(
                    "large_min_scale##primitive_library",
                    self.large_min_scale,
                    v_min=0.005,
                    v_max=0.2,
                )

                if psim.Button("Reload##primitive_library"):
                    self._init_filter_large()
                psim.SameLine()
                if psim.Button("Delete##primitive_library"):
                    self._kill_filter_large()

                # NOTE: just in case it was deleted above
                if self.large_pc_selector is not None:
                    self.large_pc_selector.gui()

                psim.TreePop()

            psim.Separator()

        # --------------------------------
        # ENVIRONMENT MAP
        # --------------------------------

        if self.envmap is not None:

            if psim.TreeNode("Envmap##primitive_library"):
                self.envmap.gui()

                psim.TreePop()

            psim.Separator()

        if len(self.loaded_entries) > 0:

            # --------------------------------
            # TONEMAPPER
            # --------------------------------

            if psim.TreeNode("Tonemapper##primitive_library"):
                processed = self.current_entry.tonemapper.gui()

                if len(processed) > 0:
                    self.gaussian_update_callback()

                psim.TreePop()

            # --------------------------------
            # DEBUG ALIGNMENT
            # --------------------------------

            # if (
            #     self.current_entry.grower
            #     is not None
            #     # and self.current_entry.grower.transform_from_original_scene is not None
            #     # and not torch.allclose(
            #     #     self.current_entry.grower.transform_from_original_scene,
            #     #     torch.eye(4).cuda(),
            #     # )
            # ):
            #     # if psim.TreeNode("Debug align##primitive_library"):

            #     from sprim.utils.process_utils import isin_coord

            #     _, self.ref_idx = psim.SliderInt(
            #         "ref_idx##primitive_library",
            #         self.ref_idx,
            #         v_min=0,
            #         v_max=len(self.loaded_entries) - 1,
            #     )

            #     _, self.filtering_res = psim.SliderInt(
            #         "filtering_res##primitive_library",
            #         self.filtering_res,
            #         v_min=16,
            #         v_max=128,
            #     )

            #     if (
            #         psim.Button("Original Transform##primitive_library")
            #         and self.current_entry.grower.transform_from_original_scene
            #         is not None
            #     ):
            #         # Apply inverse transform
            #         self.loaded_entries[self.ref_idx].transform = torch.inverse(
            #             self.current_entry.grower.transform_from_original_scene
            #         )
            #         self.gaussian_update_callback()

            #     # NOTE: this is extremely DIY
            #     if psim.Button("Filter similar##primitive_library"):
            #         ref_entry = self.loaded_entries[self.ref_idx]
            #         ref_gaussian_set = ref_entry.gaussian_model.get_gaussian_set()
            #         prim_entry = self.current_entry

            #         def to_voxel(x: torch.Tensor) -> torch.Tensor:
            #             return (
            #                 (x - self.current_entry.bbox_min)
            #                 / (
            #                     self.current_entry.bbox_max
            #                     - self.current_entry.bbox_min
            #                 )
            #                 * self.filtering_res
            #             ).long()

            #         # Filter w.r.t. current bbox
            #         # self.add_layer_from_gaussians(
            #         #     ref_gaussian_set.transform(ref_entry.get_transform())
            #         # )
            #         # self.add_layer_from_gaussians(
            #         #     primitive_entry.gaussian_model.get_gaussian_set().transform(
            #         #         primitive_entry.get_transform()
            #         #     )
            #         # )
            #         ref_gaussians_voxels = to_voxel(
            #             ref_gaussian_set.transform(ref_entry.get_transform()).means
            #         )
            #         prim_gaussian_voxels = to_voxel(
            #             primitive_entry.gaussian_model.get_gaussian_set()
            #             .transform(primitive_entry.get_transform())
            #             .means
            #         )

            #         ref_unique_gaussians_voxels, invmap = torch.unique(
            #             ref_gaussians_voxels, dim=0, return_inverse=True
            #         )
            #         mask = isin_coord(
            #             ref_unique_gaussians_voxels,
            #             torch.unique(prim_gaussian_voxels, dim=0),
            #         )
            #         mask = mask[invmap]
            #         filtered_ref_gaussians = ref_gaussian_set.filter(~mask)
            #         ref_entry.gaussian_model.set_gaussian_set(filtered_ref_gaussians)
            #         # self.add_layer_from_gaussians(ref_gaussian_set.filter(mask))
            #         self.gaussian_update_callback()

            #     psim.TreePop()

            psim.Separator()

    def handle_control(self, transform_type: str):
        io = psim.GetIO()
        if io.KeysDown[KEYMAP["up_arrow"]]:
            self._transform("forward", transform_type)
        elif io.KeysDown[KEYMAP["down_arrow"]]:
            self._transform("backward", transform_type)
        elif io.KeysDown[KEYMAP["right_arrow"]]:
            self._transform("right", transform_type)
        elif io.KeysDown[KEYMAP["left_arrow"]]:
            self._transform("left", transform_type)
        elif io.KeysDown[KEYMAP["rshift"]]:
            self._transform("up", transform_type)
        elif io.KeysDown[KEYMAP["rctrl"]]:
            self._transform("down", transform_type)

    def _transform(self, dir: str, transform_type: str):
        transform = self.current_entry.get_transform()
        if transform is None:
            transform = torch.eye(4).cuda()

        if transform_type == "translation":
            x_dir = transform[:3, 0]
            y_dir = transform[:3, 1]
            z_dir = transform[:3, 2]
            if dir == "forward":
                transform[:3, 3] += GLOBAL_STATE.move_sensitivity * x_dir
            elif dir == "backward":
                transform[:3, 3] -= GLOBAL_STATE.move_sensitivity * x_dir
            elif dir == "right":
                transform[:3, 3] += GLOBAL_STATE.move_sensitivity * y_dir
            elif dir == "left":
                transform[:3, 3] -= GLOBAL_STATE.move_sensitivity * y_dir
            elif dir == "up":
                transform[:3, 3] += GLOBAL_STATE.move_sensitivity * z_dir
            elif dir == "down":
                transform[:3, 3] -= GLOBAL_STATE.move_sensitivity * z_dir
        elif transform_type == "scale":
            if dir == "forward":
                transform[:3, :3] *= (1.0 + GLOBAL_STATE.move_sensitivity) ** (
                    1.0 / 3.0
                )
            elif dir == "backward":
                transform[:3, :3] /= (1.0 + GLOBAL_STATE.move_sensitivity) ** (
                    1.0 / 3.0
                )
        elif transform_type == "rotation":
            if dir == "forward":
                rotation = torch.tensor(
                    tf.SO3.from_x_radians(GLOBAL_STATE.move_sensitivity)
                    .as_matrix()
                    .astype(np.float32)
                ).cuda()
                transform[:3, :3] = rotation @ transform[:3, :3]
            elif dir == "backward":
                rotation = torch.tensor(
                    tf.SO3.from_x_radians(GLOBAL_STATE.move_sensitivity)
                    .as_matrix()
                    .astype(np.float32)
                ).cuda()
                transform[:3, :3] = rotation.T @ transform[:3, :3]
            elif dir == "right":
                rotation = torch.tensor(
                    tf.SO3.from_y_radians(GLOBAL_STATE.move_sensitivity)
                    .as_matrix()
                    .astype(np.float32)
                ).cuda()
                transform[:3, :3] = rotation @ transform[:3, :3]
            elif dir == "left":
                rotation = torch.tensor(
                    tf.SO3.from_y_radians(GLOBAL_STATE.move_sensitivity)
                    .as_matrix()
                    .astype(np.float32)
                ).cuda()
                transform[:3, :3] = rotation.T @ transform[:3, :3]
            elif dir == "up":
                rotation = torch.tensor(
                    tf.SO3.from_z_radians(GLOBAL_STATE.move_sensitivity)
                    .as_matrix()
                    .astype(np.float32)
                ).cuda()
                transform[:3, :3] = rotation @ transform[:3, :3]
            elif dir == "down":
                rotation = torch.tensor(
                    tf.SO3.from_z_radians(GLOBAL_STATE.move_sensitivity)
                    .as_matrix()
                    .astype(np.float32)
                ).cuda()
                transform[:3, :3] = rotation.T @ transform[:3, :3]

        self.current_entry.set_transform(transform)
        self.moving_with_keys = True
        self.gaussian_update_callback()

    def save_session(self):

        data = {}
        data["loaded_current"] = self.loaded_current
        data["loaded_entries"] = []
        for entry in self.loaded_entries:
            data["loaded_entries"].append(entry.serialize())

        data["camera"] = get_camera_dict()

        if self.envmap is not None:
            data["envmap"] = self.envmap.serialize()

        os.makedirs(os.path.dirname(self.session_path), exist_ok=True)
        dd.io.save(self.session_path, data)

    # NOTE: this will write on top of the previously loaded session
    def load_session(self, session_path: str, load_camera):

        # Always offset from the already existing sessions
        offset = len(self.loaded_entries)

        data = dd.io.load(session_path)

        for i_entry, data_entry in tqdm(
            enumerate(data["loaded_entries"]), "Loading session"
        ):
            try:
                primitive_entry = PrimitiveEntry.deserialize(
                    data_entry, gaussian_update_callback=self.gaussian_update_callback
                )
            except Exception as error:
                if "data_entry" in data_entry:
                    print(
                        f"Could not load layer with name {data_entry['display_name']}: {error}"
                    )
                else:
                    print(f"Could not load layer number {i_entry}: {error}")
                continue

            self.loaded_entries.append(primitive_entry)

        # Make sure the loaded entry does not exceed the current layers
        # NOTE: this is in case some layers were not loaded
        self.loaded_current = min(
            data["loaded_current"] + offset, len(self.loaded_entries) - 1
        )

        # assert self.loaded_current < len(self.loaded_entries)

        self.update_viewer()

        if "camera" in data:
            load_camera(data["camera"])

        if "envmap" in data:
            self.envmap = EnvMap.deserialize(data["envmap"])
            self.gaussian_update_callback()
