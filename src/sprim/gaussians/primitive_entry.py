from __future__ import annotations

import os
import uuid
from dataclasses import dataclass, field
from typing import Dict, Any

import torch
import numpy as np

import polyscope as ps

from sprim.configs.base import BaseConfig
from sprim.utils.io_utils import load_models_only
from sprim.inverse.grown_voxels import RESOLUTIONS
from sprim.utils.voxel_set import VoxelSet
from sprim.gaussians.tonemapper import Tonemapper
from sprim.gaussians.gaussian_model import GaussianModel, GaussianSet
from sprim.gaussians.trainer import Trainer
from sprim.inverse.grower import Grower
from sprim.utils.viewer_utils import (
    CUBE_EDGES_NP,
    CUBE_VERTICES_NP,
)
from sprim.gaussians.brush_painter import BrushPainter
from sprim.utils.history_handler import HistoryHandler

from sklearn.decomposition import PCA


@dataclass(kw_only=True)
class LayerCounter:
    layer_count: int = 0

    def __init__(self) -> None:
        pass

    def get_count(self) -> int:
        self.layer_count += 1
        return self.layer_count - 1


LAYER_COUNTER = LayerCounter()


# Each loaded primitive is recorded as a primitive entry to provide the right
# context and states (this is much safer than having these dangling components
# everywhere)
@dataclass(kw_only=True, frozen=False)
class PrimitiveEntry:
    path: str
    gca_path: str
    id: str = field(default_factory=lambda: uuid.uuid1())

    # These three components really define an entry
    config: BaseConfig
    gaussian_model: GaussianModel
    grower: Grower | None
    trainer: Trainer | None
    brush_painter: BrushPainter | None

    # Each entry holds its own bbox and corresponding transform
    bbox_min: torch.Tensor
    bbox_max: torch.Tensor
    bbox: ps.CurveNetwork | None = None
    transform: torch.Tensor | None = None
    # The transform bbox is meant to be closer to the Gaussian point cloud
    transform_bbox_min: torch.Tensor | None = None
    transform_bbox_max: torch.Tensor | None = None
    transform_history: HistoryHandler | None = None

    # This is just a flag to know if it needs to be rendered or not
    active: bool = True
    show_grown: bool = True
    display_name: str | None = None

    def __post_init__(self):
        if self.display_name is None:
            self.display_name = f"layer_{LAYER_COUNTER.get_count()}"

        # Add a tonemapper
        self.tonemapper = Tonemapper()

        # Create bounding box based on the Gaussians directly
        gaussian_set = self.gaussian_model.get_gaussian_set()
        self.transform_bbox_min = gaussian_set.means.min(0)[0]
        self.transform_bbox_max = gaussian_set.means.max(0)[0]

        # Recentering is only allowed if it isn't a grower! because we need the
        # proper scale and positions to map to the initial set of Gaussians
        # NOTE: a way to deal with this would be to track both transforms
        if self.grower is None:

            # Update the transform w.r.t the center of the transform_bbox
            mid_transform_bbox = 0.5 * (
                self.transform_bbox_min + self.transform_bbox_max
            )
            reset_transform = torch.eye(4).cuda()
            inv_reset_transform = torch.eye(4).cuda()
            reset_transform[:3, 3] = mid_transform_bbox
            inv_reset_transform[:3, 3] -= mid_transform_bbox

            # Apply inverse transform
            self.gaussian_model.set_gaussian_set(
                gaussian_set=gaussian_set.transform(inv_reset_transform)
            )

            # Update transform by composing transformations
            if self.transform is None:
                self.transform = torch.eye(4).cuda()
            self.transform = self.transform @ reset_transform

            gaussian_set = self.gaussian_model.get_gaussian_set()
            self.transform_bbox_min = gaussian_set.means.min(0)[0]
            self.transform_bbox_max = gaussian_set.means.max(0)[0]

        # Create history handler
        self.transform_history = HistoryHandler()
        self.transform_history.record_new(self.transform)

    def update_bbox(self) -> None:
        if self.transform_bbox_min is not None and self.transform_bbox_max is not None:
            bbox_min = self.transform_bbox_min
            bbox_max = self.transform_bbox_max
        else:
            bbox_min = self.bbox_min
            bbox_max = self.bbox_max

        bbox_min_np = bbox_min.cpu().numpy()
        bbox_max_np = bbox_max.cpu().numpy()

        cube_vertices = (bbox_max_np - bbox_min_np) * CUBE_VERTICES_NP + bbox_min_np

        self.bbox = ps.register_curve_network(
            f"bbox_{self.id}", cube_vertices, CUBE_EDGES_NP, enabled=False, radius=0.01
        )

        # Apply the transform if a transform already existed on this primitive
        # WARNING: apply it on the structure directly! Otherwise, this will
        if self.transform is not None:
            self.bbox.set_transform(self.transform.cpu().numpy())

    def serialize(self) -> Dict[str, Any]:
        serialized_grower = self.grower.serialize() if self.grower is not None else None
        serialized_brush_painter = (
            self.brush_painter.serialize() if self.brush_painter is not None else None
        )

        result = {
            "path": self.path,
            "gca_path": self.gca_path,
            "active": self.active,
            "grower": serialized_grower,
            "brush_painter": serialized_brush_painter,
            "bbox_min": self.bbox_min.cpu().numpy(),
            "bbox_max": self.bbox_max.cpu().numpy(),
        }

        if self.transform is not None:
            result["transform"] = self.transform.cpu().numpy()

        if self.path is None:
            result["gaussian_set"] = self.gaussian_model.get_gaussian_set().serialize()

        if self.display_name is not None:
            result["display_name"] = self.display_name

        if self.tonemapper is not None:
            result["tonemapper"] = self.tonemapper.serialize()

        if self.transform_bbox_min is not None and self.transform_bbox_max is not None:
            result["transform_bbox_min"] = self.transform_bbox_min.cpu().numpy()
            result["transform_bbox_max"] = self.transform_bbox_max.cpu().numpy()

        return result

    def get_transform(self) -> torch.Tensor | None:
        return self.transform

    def set_transform(self, transform: torch.Tensor) -> None:
        if transform is None:
            transform = torch.eye(4).cuda()
        self.transform = transform
        if self.bbox is not None:
            self.bbox.set_transform(transform.cpu().numpy())

    @staticmethod
    def deserialize(data: Dict[str, Any], gaussian_update_callback) -> PrimitiveEntry:

        transform = (
            torch.tensor(data["transform"]).float().cuda()
            if "transform" in data
            else None
        )

        if data["path"] is None:
            gaussian_set = GaussianSet.deserialize(data["gaussian_set"])

            config = BaseConfig()

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
                bbox_min=torch.tensor(data["bbox_min"]).float().cuda(),
                bbox_max=torch.tensor(data["bbox_max"]).float().cuda(),
                transform=transform,
                active=data["active"],
            )
            # entry.brush_painter = BrushPainter(
            #     config=config, brush_paths=[], primitive_entry=entry
            # )
        else:
            entry = PrimitiveEntry.load_entry(
                data["path"],
                gca_path=data["gca_path"],
                gaussian_update_callback=gaussian_update_callback,
                transform=transform,
                active=data["active"],
            )

        if "grower" in data and data["grower"] is not None:
            entry.grower.load_serialized(data["grower"])

            if "brush_painter" in data and data["brush_painter"] is not None:
                entry.brush_painter.load_serialized(data["brush_painter"])

        if "display_name" in data:
            entry.display_name = data["display_name"]

        if "tonemapper" in data:
            entry.tonemapper = Tonemapper.deserialize(data["tonemapper"])

        if "transform_bbox_min" in data and "transform_bbox_max" in data:
            entry.transform_bbox_min = (
                torch.tensor(data["transform_bbox_min"]).float().cuda()
            )
            entry.transform_bbox_max = (
                torch.tensor(data["transform_bbox_max"]).float().cuda()
            )

        return entry

    @staticmethod
    def load_entry(
        path: str,
        gca_path: str = None,
        gaussian_update_callback=lambda: None,
        painting_path: str | None = None,
        headless_mode: bool = False,
        transform: torch.Tensor | None = None,
        active: bool = True,
    ) -> PrimitiveEntry:
        assert gaussian_update_callback is not None

        gaussian_model, config, grower, trainer = load_models_only(
            input_path=path,
            gca_path=gca_path,
            gaussian_update_callback=gaussian_update_callback,
        )

        bbox_min = torch.tensor(config.aabb[:3]).cuda()
        bbox_max = torch.tensor(config.aabb[3:]).cuda()
        if grower is not None:
            bbox_min = grower.bbox_min.squeeze(0)
            bbox_max = grower.bbox_max.squeeze(0)

        entry = PrimitiveEntry(
            path=path,
            gca_path=gca_path,
            gaussian_model=gaussian_model,
            config=config,
            grower=grower,
            trainer=trainer,
            brush_painter=None,
            bbox_min=bbox_min,
            bbox_max=bbox_max,
            transform=transform,
            active=active,
        )

        if entry.grower is not None:
            entry.brush_painter = BrushPainter(
                config=config,
                brush_paths=[],
                primitive_entry=entry,
                painting_path=painting_path,
                headless_mode=headless_mode,
            )

        if entry.grower is not None:
            entry.grower.get_transform = entry.get_transform

        return entry

    def load_hierarchy(self):
        # Check whether we can reload the corresponding hierarchy
        basename = os.path.basename(self.path).split(".")[0]
        hierarchy_path = os.path.join(
            os.path.dirname(os.path.dirname(self.path)), "raw", basename + ".npz"
        )
        if not os.path.exists(hierarchy_path):
            return

        # DIRECTLY borrowed from PREVIEW

        # NOTE: res_idx are set when exporting latents (i.e., res prefixes):
        # "lower is finer"
        # Find path
        data = np.load(hierarchy_path)

        for idx, res in enumerate(RESOLUTIONS):
            voxel_res = data[f"{idx}_res"].item()
            assert res == voxel_res
            surface_voxels = torch.tensor(data[f"{idx}_voxels"]).cuda()
            latents = torch.tensor(data[f"{idx}_voxels_latents"]).float().cuda()

            if idx == 0:
                pca = PCA(n_components=3)
                pca = pca.fit(latents.cpu().numpy())

            feature_maps_pca = pca.transform(latents.cpu().numpy())
            pca_features = torch.sigmoid(torch.tensor(feature_maps_pca).cuda())
            voxel_set = VoxelSet(
                prefix=f"{idx}_voxels",
                voxels=surface_voxels,
                res=res,
                bbox_min=self.bbox_min,
                bbox_max=self.bbox_max,
                rgb=pca_features,
                enabled=False,
            )
