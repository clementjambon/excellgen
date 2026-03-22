from typing import Set, Dict
import os

import torch
import numpy as np

import polyscope as ps
import polyscope.imgui as psim

from sprim.gaussians.gaussian_model import GaussianModel
from sprim.gaussians.pc_selector import PcSelector
from sprim.configs.base import BaseConfig
from sprim.utils.process_utils import filter_bbox
from sprim.utils.viewer_utils import CUBE_EDGES_NP, CUBE_VERTICES_NP
from sprim.gaussians.brush_painter import BrushPainter, EXEMPLAR_BRUSHES_FOLDER
from sprim.gaussians.brush import Brush
from sprim.utils.gui_utils import save_popup
from sprim.gaussians.global_state import GLOBAL_STATE


class BrushCreator:

    selected_indices: Set[int]
    selected_colors = Dict[int, torch.Tensor]
    gaussian_model: GaussianModel
    pc_selector: PcSelector | None

    def __init__(
        self,
        gaussian_model: GaussianModel,
        config: BaseConfig,
        brush_painter: BrushPainter,
    ):
        self.selected_indices = set()
        self.selected_colors = dict()
        self.gaussian_model = gaussian_model
        self.brush_painter = brush_painter
        self.config = config

        self.pc_selector = None
        # TODO: sync with the reference
        self.limit_bbox = None

        # Resolve default brush name
        self.brush_folder = os.path.join(self.config.log_dir, EXEMPLAR_BRUSHES_FOLDER)
        self._resolve_brush_name()

    def _resolve_brush_name(self):
        n_prev = (
            len(os.listdir(self.brush_folder))
            if os.path.exists(self.brush_folder)
            else 0
        )
        self.brush_name = f"brush_{n_prev:04d}.npz"

    @torch.no_grad()
    def postprocess_features(
        self,
        features: torch.Tensor,
        indices: torch.Tensor,
        rendered_features: torch.Tensor,
        render_ratio: float = 1.0,
    ):

        io = psim.GetIO()
        if psim.IsMouseClicked(0) and io.KeyCtrl:
            mouse_pos = psim.GetMousePos()
            selected_index = indices[
                int(mouse_pos[1] / render_ratio), int(mouse_pos[0] / render_ratio)
            ].item()

            # Remove
            if selected_index in self.selected_indices:
                self.remove_selected_index(selected_index)
            # Add
            else:
                selected_color = (
                    rendered_features[
                        int(mouse_pos[1] / render_ratio),
                        int(mouse_pos[0] / render_ratio),
                    ]
                    .cpu()
                    .numpy()
                )
                self.add_selected_index(
                    selected_index=selected_index, selected_color=selected_color
                )

        return self._render_selection(rendered_features, indices)

    @torch.no_grad()
    def _render_selection(
        self, rendered_features: torch.Tensor, indices: torch.Tensor
    ) -> torch.Tensor:

        # Render selection with a specific color
        if len(self.selected_indices) > 0:
            mask = torch.zeros(indices.shape, dtype=torch.bool, device=indices.device)
            for selected_index in self.selected_indices:
                mask |= indices == selected_index

            img_idx = torch.argwhere(mask)
            rendered_features[tuple(img_idx.T)] = torch.tensor(
                [1.0, 0.0, 0.0], device=rendered_features.device
            )

        return rendered_features

    def add_selected_index(self, selected_index, selected_color):
        self.selected_indices.add(selected_index)
        self.selected_colors[selected_index] = selected_color

    def remove_selected_index(self, selected_index):
        self.selected_indices.remove(selected_index)
        del self.selected_colors[selected_index]

    def select(self, bbox: bool = False):
        if len(self.selected_indices) == 0:
            mask = torch.ones(
                self.gaussian_model.features_feat.shape[0],
                dtype=torch.bool,
                device=self.gaussian_model.features_feat.device,
            )
        else:
            mask = torch.zeros(
                self.gaussian_model.features_feat.shape[0],
                dtype=torch.bool,
                device=self.gaussian_model.features_feat.device,
            )
            _, all_indices, _ = self.gaussian_model.feature_quantizer(
                self.gaussian_model.features_feat.view(
                    -1, self.gaussian_model.features_dim
                )
            )
            for selected_index in self.selected_indices:
                mask |= all_indices == selected_index

        pos = self.gaussian_model.means[mask]
        if bbox:
            pos = filter_bbox(
                pos,
                torch.tensor(self.bbox_min, device="cuda"),
                torch.tensor(self.bbox_max, device="cuda"),
            )
        self.pc_selector = PcSelector("brush_selection", pos, enabled=True)
        self.pc_selector.subsample()

        # The limit bbox is initialized from the original scene bbox

    def init_limit_bbox(self):
        self.bbox_min = np.array(self.config.aabb[:3])
        self.bbox_max = np.array(self.config.aabb[3:])
        cube_vertices = (
            self.bbox_max - self.bbox_min
        ) * CUBE_VERTICES_NP + self.bbox_min

        self.limit_bbox = ps.register_curve_network(
            "limit_bbox_brush_selection", cube_vertices, CUBE_EDGES_NP, radius=0.01
        )

    @torch.no_grad()
    def gui(self):

        # psim.SetNextItemOpen(True, psim.ImGuiCond_Once)
        if psim.TreeNode("Brush Creator"):

            if self.limit_bbox is None:
                self.init_limit_bbox()

            if self.pc_selector is not None:

                requested, self.brush_name = save_popup("brush_save", self.brush_name)
                if requested:
                    brush = Brush(self.pc_selector.pos, None)
                    os.makedirs(self.brush_folder, exist_ok=True)
                    brush_path = os.path.join(self.brush_folder, self.brush_name)
                    brush.export_brush(brush_path)
                    self._resolve_brush_name()
                    self.brush_painter.load_brushes()

                    # Kill the current pc_selector
                    self.pc_selector.kill()
                    self.pc_selector = None
                    return

                active, opened = psim.CollapsingHeader(
                    "PcSelector##brush_selection",
                    True,
                    psim.ImGuiTreeNodeFlags_DefaultOpen,
                )
                if active:
                    self.pc_selector.gui()

                # If the collapsing header was killed, erase it
                if not opened:
                    self.pc_selector.kill()
                    del self.pc_selector
                    self.pc_selector = None

            else:

                if psim.Button("Select"):
                    self.select(True)

                if len(self.selected_indices) > 0:

                    psim.Separator()

                    if psim.BeginTabBar(
                        "FeatSelectionTabBar", psim.ImGuiTabBarFlags_Reorderable
                    ):

                        def render_all_colors():
                            for i, selected_index in enumerate(self.selected_indices):
                                if i > 0:
                                    psim.SameLine()

                                feat_color = self.selected_colors[selected_index]
                                psim.ColorButton(
                                    f"{selected_index}##feat_color",
                                    (
                                        feat_color[0],
                                        feat_color[1],
                                        feat_color[2],
                                        1.0,
                                    ),
                                )

                        for selected_index in self.selected_indices:
                            clicked, opened = psim.BeginTabItem(
                                f"{selected_index}##feat",
                                True,
                                psim.ImGuiTabItemFlags_None,
                            )
                            if clicked:
                                render_all_colors()

                                psim.EndTabItem()

                            # Delete color!
                            if not opened:
                                self.remove_selected_index(selected_index)
                                break

                        psim.EndTabBar()

                else:
                    psim.Text("Press 'Ctrl' to select a segment")

            psim.TreePop()

        else:
            if self.limit_bbox is not None:
                self.limit_bbox.remove()
                self.limit_bbox = None

        psim.Separator()
