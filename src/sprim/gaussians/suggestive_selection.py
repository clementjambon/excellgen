from enum import StrEnum
from typing import Set, Dict

import torch
import torch.nn as nn

import polyscope.imgui as psim

from sprim.configs.base import BaseConfig
from sprim.gaussians.gaussian_model import GaussianModel
from sprim.gaussians.brush_painter import BrushPainter
from sprim.utils.gui_utils import RenderMode, KEY_HANDLER
from sprim.gaussians.global_state import GLOBAL_STATE

DEFAULT_SCALE_THRESHOLD = 0.01
MIN_SCALE_THRESHOLD = 0.0
MAX_SCALE_THRESHOLD = 0.1

from vector_quantize_pytorch.vector_quantize_pytorch import EuclideanCodebook
from einops import rearrange


class QuantizerWrapper(nn.Module):

    def __init__(self, codebook: EuclideanCodebook) -> None:
        super().__init__()
        self._codebook = codebook

    def __call__(self, x: torch.Tensor, freeze_codebook: bool = False) -> torch.Any:
        codebook_forward_kwargs = dict(
            sample_codebook_temp=None,
            mask=None,
            freeze_codebook=True,
        )

        x = rearrange(x, "b d -> b 1 d")

        quantize, embed_ind, _ = self._codebook(x, **codebook_forward_kwargs)

        quantize = rearrange(quantize, "b 1 d -> b d")
        embed_ind = rearrange(embed_ind, "b 1 ... -> b ...")

        return quantize, embed_ind, None


class SuggestiveSelection:

    selected_indices: Set[int]
    selected_colors = Dict[int, torch.Tensor]
    mask: torch.Tensor | None
    gaussian_model: GaussianModel

    def __init__(
        self,
        gaussian_model: GaussianModel,
        config: BaseConfig,
        brush_painter: BrushPainter,
    ) -> None:

        self.gaussian_model = gaussian_model

        # Selection
        self.selected_indices = set()
        self.selected_colors = dict()
        self.mask = None
        self.filter_scale: bool = False
        self.scale_threshold: float = DEFAULT_SCALE_THRESHOLD

        # Quantization
        self.codebook_size: int = 10
        self.kmeans_iter: int = 10

    def reset_mask(self) -> None:
        self.mask = None

    def get_mask(self, render_mode: RenderMode, mask: torch.Tensor | None = None):

        # if self.processor_mode == ProcessorMode.DEBUG_VOXEL:
        #     return self.debug_voxel_selection.mask

        if render_mode == RenderMode.RGB:
            if mask is None:
                return self.mask
            elif self.mask is not None:
                # If a mask is provided, combine both
                if len(self.mask) != len(mask):
                    return mask
                return self.mask & mask
            else:
                return mask
        else:
            return mask

    def add_selected_index(self, selected_index, selected_color):
        self.selected_indices.add(selected_index)
        self.selected_colors[selected_index] = selected_color

        self.update_mask()

    def remove_selected_index(self, selected_index):
        self.selected_indices.remove(selected_index)
        del self.selected_colors[selected_index]

        self.update_mask()

    def reset_selection(self):
        self.selected_indices = set()
        self.selected_colors = dict()
        self.update_mask()

    def update_mask(self):
        scale_mask = None
        if self.filter_scale:
            scale_mask = (
                self.gaussian_model.scales.exp().max(dim=-1)[0] <= self.scale_threshold
            )

        # When nothing is selected, fall back to full scene
        if len(self.selected_indices) == 0:
            self.mask = scale_mask
            return

        # Update mask
        segment_mask = torch.zeros(
            self.gaussian_model.features_feat.shape[0],
            dtype=torch.bool,
            device=self.gaussian_model.features_feat.device,
        )
        _, all_indices, _ = self.gaussian_model.feature_quantizer(
            self.gaussian_model.features_feat.view(-1, self.gaussian_model.features_dim)
        )
        for selected_index in self.selected_indices:
            segment_mask |= all_indices == selected_index

        self.mask = (
            segment_mask & scale_mask if scale_mask is not None else segment_mask
        )

    @torch.no_grad()
    def postprocess_feature(
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
    def update_quantizer(self) -> None:
        new_codebook = EuclideanCodebook(
            dim=self.gaussian_model.features_dim,
            codebook_size=self.codebook_size,
            kmeans_init=True,
            kmeans_iters=self.kmeans_iter,
        ).cuda()

        new_codebook.forward(
            rearrange(self.gaussian_model.features_feat, "b d -> b 1 d")
        )

        new_quantizer = QuantizerWrapper(new_codebook)

        self.gaussian_model.feature_quantizer = new_quantizer

        # Reset selection if there was one
        self.reset_selection()

    def gui(self) -> None:

        if KEY_HANDLER("space"):
            self.update_quantizer()

        # psim.SetNextItemOpen(True, psim.ImGuiCond_Once)
        if psim.TreeNode("Suggestive Selection"):

            # Update the quantizer if there isn't one yet
            if self.gaussian_model.feature_quantizer is None:
                self.update_quantizer()

            # clicked, self.filter_scale = psim.Checkbox(
            #     "Filter scale##suggestive_selection", self.filter_scale
            # )
            # if clicked:
            #     self.update_mask()
            if self.filter_scale:
                clicked, self.scale_threshold = psim.SliderFloat(
                    "Scale threshold##suggestive_selection",
                    self.scale_threshold,
                    v_min=MIN_SCALE_THRESHOLD,
                    v_max=MAX_SCALE_THRESHOLD,
                )
                if clicked:
                    self.update_mask()

            if len(self.selected_indices) > 0 and psim.BeginTabBar(
                "FeatSelectionTabBar##suggestive_selection",
                psim.ImGuiTabBarFlags_Reorderable,
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

            if psim.TreeNode("Quantization"):

                _, self.codebook_size = psim.SliderInt(
                    "codebook_size##suggestive_selection",
                    self.codebook_size,
                    v_min=2,
                    v_max=20,
                )
                _, self.kmeans_iter = psim.SliderInt(
                    "kmeans_iter##suggestive_selection",
                    self.kmeans_iter,
                    v_min=5,
                    v_max=50,
                )

                if psim.Button("Update quantizer##suggestive_selection"):

                    self.update_quantizer()

                psim.TreePop()

            psim.TreePop()

        psim.Separator()

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
                GLOBAL_STATE.selection_color, device=rendered_features.device
            )

        return rendered_features
