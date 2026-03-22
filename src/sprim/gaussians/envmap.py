from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any
import os
import imageio

import polyscope.imgui as psim

import torch


@dataclass(kw_only=True)
class EnvMap:

    image: torch.Tensor
    phi_offset: float = 0.0
    theta_offset: float = 0.0
    invert_theta: bool = False

    @staticmethod
    def from_image(image_path: str) -> EnvMap:
        if not os.path.exists(image_path):
            print(f"Envmap file not found at {image_path}")
            return None

        image = imageio.imread(image_path)
        image = torch.tensor(image, requires_grad=False).cuda().float() / 255.0
        image = image[..., :3].permute(2, 0, 1).unsqueeze(0)

        return EnvMap(image=image)

    def serialize(self) -> Dict[str, Any]:

        data = {}
        data["image"] = self.image.cpu().numpy()
        data["phi_offset"] = self.phi_offset
        data["theta_offset"] = self.theta_offset
        data["invert_theta"] = self.invert_theta

        return data

    @staticmethod
    def deserialize(data: Dict[str, Any]) -> EnvMap:
        image = torch.tensor(data["image"]).float().cuda()
        return EnvMap(
            image=image,
            phi_offset=data["phi_offset"],
            theta_offset=data["theta_offset"],
            invert_theta=data["invert_theta"],
        )

    def sample(self, phi: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        if self.invert_theta:
            theta *= -1.0
        grid = torch.cat(
            (
                phi - self.phi_offset,
                theta - self.theta_offset,
            ),
            dim=-1,
        ).unsqueeze(0)
        projected_envmap = (
            torch.nn.functional.grid_sample(
                self.image,
                grid,
                mode="bilinear",
                padding_mode="reflection",
                align_corners=False,
            )
            .squeeze(0)
            .permute(1, 2, 0)
        )

        return projected_envmap

    def gui(self):
        _, self.phi_offset = psim.SliderFloat(
            "phi_offset", self.phi_offset, v_min=-1.0, v_max=1.0
        )
        _, self.theta_offset = psim.SliderFloat(
            "theta_offset", self.theta_offset, v_min=-1.0, v_max=1.0
        )
        _, self.invert_theta = psim.Checkbox("invert_theta", self.invert_theta)
