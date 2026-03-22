import os
import glob
from typing import List
from dataclasses import dataclass

import numpy as np
import polyscope.imgui as psim

from sprim.utils.gui_utils import colored_button

TREE_NODE_FLAGS = psim.ImGuiTreeNodeFlags_None
MAX_PREVIEW_SIZE = 150


@dataclass
class TreeNode:

    name: str
    display_path: str
    path: str
    depth: int
    child_idx: int = -1
    child_count: int = 0

    def __init__(
        self,
        name: str,
        display_path: str,
        path: str,
        load_callback,
        is_snapshot: bool,
        gca_path: str,
        depth: int,
    ) -> None:
        self.name = name
        self.display_path = display_path
        self.path = path
        self.load_callback = load_callback
        self.is_snapshot = is_snapshot
        self.gca_path = gca_path
        self.depth = depth
        self.preview_path = None
        self.preview_quantity = None
        self.preview_size = None

        if is_snapshot and os.path.exists(os.path.join(self.path, "preview.png")):
            self.preview_path = os.path.join(self.path, "preview.png")

    def _load_button(self):
        if self.is_snapshot:
            if os.path.exists(os.path.join(self.path, "ckpts")):
                if colored_button(f"Load##{self.display_path}", 0.0):
                    # List snapshots inside
                    last = sorted(glob.glob(os.path.join(self.path, "ckpts", "*.pt")))[
                        -1
                    ]
                    self.load_callback(last)
            else:
                if colored_button(
                    f"Load##{self.display_path}",
                    0.4 if self.gca_path is not None else 0.0,
                ):
                    self.load_callback(self.path, self.gca_path)
        else:
            psim.TextDisabled("--")

    def _show_preview(self):
        if self.preview_path is None:
            return

        if psim.IsItemHovered():
            import imageio
            import polyscope as ps

            # Load the image if it wasn't loaded:
            if self.preview_quantity is None:
                self.image_preview = (
                    np.array(imageio.imread(self.preview_path)).astype(np.float32)
                    / 255.0
                )
                if self.image_preview.shape[2] == 3:
                    self.image_preview = np.concatenate(
                        [
                            self.image_preview,
                            np.ones(
                                (
                                    self.image_preview.shape[0],
                                    self.image_preview.shape[1],
                                    1,
                                ),
                                dtype=np.float32,
                            ),
                        ],
                        axis=-1,
                    )
                self.preview_quantity = ps.add_color_alpha_image_quantity(
                    f"{self.name}_preview_buffer",
                    self.image_preview,
                )
                h, w = self.image_preview.shape[:2]
                aspect_ratio = float(h) / float(w)
                clipped_h, clipped_w = min(h, MAX_PREVIEW_SIZE), min(
                    w, MAX_PREVIEW_SIZE
                )
                self.preview_size = (
                    int(min(clipped_h, clipped_w * aspect_ratio)),
                    int(min(clipped_w, clipped_h / aspect_ratio)),
                )

            psim.BeginTooltip()
            self.preview_quantity.imgui_image(
                self.preview_size[1], self.preview_size[0]
            )
            psim.EndTooltip()

    def display_node(self, all_nodes: List["TreeNode"]):
        # Skip root
        # TODO: make this less DIY
        if self.depth == 1:
            for child_n in range(0, self.child_count):
                all_nodes[self.child_idx + child_n].display_node(all_nodes)
            return

        psim.TableNextRow()
        psim.TableNextColumn()

        is_folder = self.child_count > 0
        if is_folder:
            tree_node_flags = TREE_NODE_FLAGS
            if not (
                self.is_snapshot and os.path.exists(os.path.join(self.path, "ckpts"))
            ):
                tree_node_flags |= psim.ImGuiTreeNodeFlags_DefaultOpen
            # if self.depth <= 1:
            #     tree_node_flags |= psim.ImGuiTreeNodeFlags_DefaultOpen
            open = psim.TreeNodeEx(self.name, tree_node_flags)
            self._show_preview()
            psim.TableNextColumn()
            self._load_button()
            if open:
                for child_n in range(0, self.child_count):
                    all_nodes[self.child_idx + child_n].display_node(all_nodes)
                psim.TreePop()
        else:
            tree_node_flags = (
                TREE_NODE_FLAGS
                | psim.ImGuiTreeNodeFlags_Leaf
                | psim.ImGuiTreeNodeFlags_Bullet
                | psim.ImGuiTreeNodeFlags_NoTreePushOnOpen
            )
            psim.TreeNodeEx(
                self.name,
                tree_node_flags,
            )
            self._show_preview()

            # self.selected = True
            psim.TableNextColumn()
            self._load_button()
