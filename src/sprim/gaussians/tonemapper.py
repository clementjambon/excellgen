from __future__ import annotations

from typing import Set, Dict, Any, List
from dataclasses import dataclass
import math

import torch
import numpy as np

import polyscope.imgui as psim

from sprim.gaussians.gaussian_model import GaussianSet, RGB2SH, SH2RGB


# Adapted from
# https://github.com/limacv/RGB_HSV_HSL/blob/master/color_torch.py
def rgb2hsv_torch(rgb: torch.Tensor) -> torch.Tensor:
    cmax, cmax_idx = torch.max(rgb, dim=-1, keepdim=True)
    cmin = torch.min(rgb, dim=-1, keepdim=True)[0]
    delta = cmax - cmin
    hsv_h = torch.empty_like(rgb[:, 0:1])
    cmax_idx[delta == 0] = 3
    hsv_h[cmax_idx == 0] = (((rgb[:, 1:2] - rgb[:, 2:3]) / delta) % 6)[cmax_idx == 0]
    hsv_h[cmax_idx == 1] = (((rgb[:, 2:3] - rgb[:, 0:1]) / delta) + 2)[cmax_idx == 1]
    hsv_h[cmax_idx == 2] = (((rgb[:, 0:1] - rgb[:, 1:2]) / delta) + 4)[cmax_idx == 2]
    hsv_h[cmax_idx == 3] = 0.0
    hsv_h /= 6.0
    hsv_s = torch.where(cmax == 0, torch.tensor(0.0).type_as(rgb), delta / cmax)
    hsv_v = cmax
    return torch.cat([hsv_h, hsv_s, hsv_v], dim=1)


def hsv2rgb_torch(hsv: torch.Tensor) -> torch.Tensor:
    hsv_h, hsv_s, hsv_l = hsv[:, 0:1], hsv[:, 1:2], hsv[:, 2:3]
    _c = hsv_l * hsv_s
    _x = _c * (-torch.abs(hsv_h * 6.0 % 2.0 - 1) + 1.0)
    _m = hsv_l - _c
    _o = torch.zeros_like(_c)
    idx = (hsv_h * 6.0).type(torch.uint8)
    idx = (idx % 6).expand(-1, 3)
    rgb = torch.empty_like(hsv)
    rgb[idx == 0] = torch.cat([_c, _x, _o], dim=1)[idx == 0]
    rgb[idx == 1] = torch.cat([_x, _c, _o], dim=1)[idx == 1]
    rgb[idx == 2] = torch.cat([_o, _c, _x], dim=1)[idx == 2]
    rgb[idx == 3] = torch.cat([_o, _x, _c], dim=1)[idx == 3]
    rgb[idx == 4] = torch.cat([_x, _o, _c], dim=1)[idx == 4]
    rgb[idx == 5] = torch.cat([_c, _o, _x], dim=1)[idx == 5]
    rgb += _m
    return rgb


# This is freely adapted from
# https://raw.githubusercontent.com/yuanming-hu/exposure/master/user_study_ui/filters.py


@dataclass(kw_only=True)
class ExposureFilter:

    exposure: float = 0.0

    def reset(self):
        self.exposure = 0.0

    def gui(self) -> bool:
        clicked, self.exposure = psim.SliderFloat(
            "exposure##tonemapper", self.exposure, v_min=-1.0, v_max=1.0
        )
        return clicked

    def apply(self, gaussian_set: GaussianSet) -> GaussianSet:
        new_feature_dc = torch.clone(gaussian_set.features_dc)
        new_feature_dc += 0.5
        new_feature_dc *= math.pow(2, self.exposure)
        new_feature_dc -= 0.5

        return GaussianSet(
            means=gaussian_set.means,
            features_dc=new_feature_dc,
            features_rest=gaussian_set.features_rest,
            opacities=gaussian_set.opacities,
            scales=gaussian_set.scales,
            quats=gaussian_set.quats,
            features_feat=gaussian_set.features_feat,
        )

    def serialize(self) -> Dict[str, Any]:
        return {"exposure": self.exposure}

    @staticmethod
    def deserialize(data: Dict[str, Any]) -> ExposureFilter:
        return ExposureFilter(exposure=data["exposure"])


@dataclass(kw_only=True)
class WBFilter:

    LIMITS = 0.5

    temperature: float = 0.0
    tint: float = 0.0

    def reset(self):
        self.temperature = 0.0
        self.tint = 0.0

    def gui(self) -> bool:
        clicked, self.temperature = psim.SliderFloat(
            "temperature##tonemapper",
            self.temperature,
            v_min=-self.LIMITS,
            v_max=self.LIMITS,
        )
        clicked2, self.tint = psim.SliderFloat(
            "tint##tonemapper",
            self.tint,
            v_min=-self.LIMITS,
            v_max=self.LIMITS,
        )
        return clicked or clicked2

    def apply(self, gaussian_set: GaussianSet) -> GaussianSet:
        new_feature_dc = torch.clone(gaussian_set.features_dc)
        rgb = SH2RGB(new_feature_dc)

        color_scaling = torch.tensor(
            (
                1,
                math.exp(-self.tint),
                math.exp(-self.temperature),
            ),
        ).cuda()
        # There will be no division by zero here unless the WB range lower bound is 0
        # normalize by luminance
        color_scaling *= 1.0 / (
            1e-5
            + 0.27 * color_scaling[0]
            + 0.67 * color_scaling[1]
            + 0.06 * color_scaling[2]
        )
        rgb *= color_scaling[None, :]
        new_feature_dc = RGB2SH(rgb)

        return GaussianSet(
            means=gaussian_set.means,
            features_dc=new_feature_dc,
            features_rest=gaussian_set.features_rest,
            opacities=gaussian_set.opacities,
            scales=gaussian_set.scales,
            quats=gaussian_set.quats,
            features_feat=gaussian_set.features_feat,
        )

    def serialize(self) -> Dict[str, Any]:
        return {"temperature": self.temperature, "tint": self.tint}

    @staticmethod
    def deserialize(data: Dict[str, Any]) -> WBFilter:
        return WBFilter(temperature=data["temperature"], tint=data["tint"])


@dataclass(kw_only=True)
class SaturationFilter:

    saturation: float = 0.0

    def reset(self):
        self.saturation = 0.0

    def gui(self) -> bool:
        clicked, self.saturation = psim.SliderFloat(
            "saturation##tonemapper", self.saturation, v_min=-1.0, v_max=1.0
        )
        return clicked

    def apply(self, gaussian_set: GaussianSet) -> GaussianSet:
        new_feature_dc = torch.clone(gaussian_set.features_dc)
        rgb = SH2RGB(new_feature_dc)

        hsv = rgb2hsv_torch(rgb)
        s = hsv[:, 1:2]
        v = hsv[:, 2:3]
        enhanced_s = s + (1 - s) * (0.5 - abs(0.5 - v))
        hsv1 = torch.cat([hsv[:, 0:1], enhanced_s, hsv[:, 2:]], dim=1)
        hsv0 = torch.cat([hsv[:, 0:1], hsv[:, 1:2] * 0 + 0, hsv[:, 2:]], dim=1)
        bnw = hsv2rgb_torch(hsv0)
        full_color = hsv2rgb_torch(hsv1)

        param = torch.tensor([self.saturation]).float().cuda()

        param = param[:, None]

        bnw_param = torch.maximum(torch.zeros_like(param), -param)
        color_param = torch.maximum(torch.zeros_like(param), param)
        img_param = torch.maximum(torch.zeros_like(param), 1.0 - abs(param))

        rgb = bnw_param * bnw + rgb * img_param + full_color * color_param

        new_feature_dc = RGB2SH(rgb)

        return GaussianSet(
            means=gaussian_set.means,
            features_dc=new_feature_dc,
            features_rest=gaussian_set.features_rest,
            opacities=gaussian_set.opacities,
            scales=gaussian_set.scales,
            quats=gaussian_set.quats,
            features_feat=gaussian_set.features_feat,
        )

    def serialize(self) -> Dict[str, Any]:
        return {"saturation": self.saturation}

    @staticmethod
    def deserialize(data: Dict[str, Any]) -> SaturationFilter:
        return SaturationFilter(saturation=data["saturation"])


def inv_sigmoid(x: torch.Tensor):
    EPSILON = 1e-8
    return torch.log(x + EPSILON) - torch.log(1 - x + EPSILON)


@dataclass(kw_only=True)
class AlphaFilter:

    alpha_factor: float = 1.0

    def reset(self):
        self.alpha_factor = 1.0

    def gui(self) -> bool:
        clicked, self.alpha_factor = psim.SliderFloat(
            "alpha_factor##tonemapper", self.alpha_factor, v_min=0.0, v_max=1.0
        )
        return clicked

    def apply(self, gaussian_set: GaussianSet) -> GaussianSet:
        new_opacities = torch.clone(gaussian_set.opacities)
        alpha = torch.sigmoid(new_opacities)
        alpha *= self.alpha_factor
        alpha = torch.clip(alpha, 0.0, 1.0)
        new_opacities = inv_sigmoid(alpha)

        return GaussianSet(
            means=gaussian_set.means,
            features_dc=gaussian_set.features_dc,
            features_rest=gaussian_set.features_rest,
            opacities=new_opacities,
            scales=gaussian_set.scales,
            quats=gaussian_set.quats,
            features_feat=gaussian_set.features_feat,
        )

    def serialize(self) -> Dict[str, Any]:
        return {"alpha_factor": self.alpha_factor}

    @staticmethod
    def deserialize(data: Dict[str, Any]) -> AlphaFilter:
        return AlphaFilter(alpha_factor=data["alpha_factor"])


FILTERS = {
    "exposure": ExposureFilter,
    "saturation": SaturationFilter,
    "wb": WBFilter,
    "alpha": AlphaFilter,
}


class Tonemapper:

    def __init__(
        self, filters: Dict[str, Any] | None = None, active: List[str] | None = None
    ) -> None:
        if filters is None:
            self.filters = {k: filter_class() for k, filter_class in FILTERS.items()}
            self.active = set()
        else:
            self.filters = filters
            self.active = set(active)

    def apply(self, gaussian_set: GaussianSet) -> GaussianSet:
        current_set = gaussian_set
        for k in self.active:
            current_set = self.filters[k].apply(current_set)
        return current_set

    def gui(self) -> Set[str]:
        processed = set()
        for k, v in self.filters.items():
            modified = v.gui()
            if modified:
                processed.add(k)
                self.active.add(k)

        if psim.Button("Reset##tonemapper"):
            for k, v in self.filters.items():
                v.reset()
                processed = self.active

        return processed

    def serialize(self) -> Dict[str, Any]:
        data = {}

        for k, v in self.filters.items():
            data[k] = v.serialize()

        data["active"] = list(self.active)

        return data

    @staticmethod
    def deserialize(data: Dict[str, Any]) -> Tonemapper:
        filters = {}
        for k, v in FILTERS.items():
            if k in data:
                filters[k] = v.deserialize(data[k])
            else:
                filters[k] = v()

        return Tonemapper(filters=filters, active=data["active"])
