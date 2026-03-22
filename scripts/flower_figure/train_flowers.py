import math
import os
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import tyro
from gsplat.project_gaussians import project_gaussians
from gsplat.rasterize import rasterize_gaussians
from PIL import Image
from torch import Tensor, optim


class SimpleTrainer:
    """Trains random gaussians to fit an image."""

    def __init__(
        self,
        gt_image: Tensor,
        gt_dino: Tensor | None,
        num_points: int = 2000,
    ):
        self.device = torch.device("cuda:0")
        self.gt_image = gt_image.to(device=self.device)
        self.gt_dino = gt_dino.to(device=self.device) if gt_dino is not None else None
        self.num_points = num_points

        fov_x = math.pi / 2.0
        self.H, self.W = gt_image.shape[0], gt_image.shape[1]
        self.focal = 0.5 * float(self.W) / math.tan(0.5 * fov_x)
        self.img_size = torch.tensor([self.W, self.H, 1], device=self.device)

        self._init_gaussians()

    def _init_gaussians(self):
        """Random gaussians"""
        bd = 20

        self.means = bd * (torch.rand(self.num_points, 3, device=self.device) - 0.5)
        self.scales = torch.rand(self.num_points, 3, device=self.device)
        d = 3
        self.rgbs = torch.rand(self.num_points, d, device=self.device)
        self.dino = (
            torch.rand(self.num_points, d, device=self.device)
            if self.gt_dino is not None
            else None
        )

        u = torch.rand(self.num_points, 1, device=self.device)
        v = torch.rand(self.num_points, 1, device=self.device)
        w = torch.rand(self.num_points, 1, device=self.device)

        self.quats = torch.cat(
            [
                torch.sqrt(1.0 - u) * torch.sin(2.0 * math.pi * v),
                torch.sqrt(1.0 - u) * torch.cos(2.0 * math.pi * v),
                torch.sqrt(u) * torch.sin(2.0 * math.pi * w),
                torch.sqrt(u) * torch.cos(2.0 * math.pi * w),
            ],
            -1,
        )
        self.opacities = torch.ones((self.num_points, 1), device=self.device)

        self.viewmat = torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 8.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            device=self.device,
        )
        self.background = torch.zeros(d, device=self.device)

        self.means.requires_grad = True
        self.scales.requires_grad = True
        self.quats.requires_grad = True
        self.rgbs.requires_grad = True
        self.opacities.requires_grad = True
        self.viewmat.requires_grad = False
        if self.dino is not None:
            self.dino.requires_grad = True

    @torch.no_grad()
    def render(
        self,
        factor: float = 4.0,
        B_SIZE: int = 14,
    ):
        focal = self.focal * factor
        H = int(self.H * factor)
        W = int(self.W * factor)
        (
            xys,
            depths,
            radii,
            conics,
            compensation,
            num_tiles_hit,
            cov3d,
        ) = project_gaussians(
            self.means,
            self.scales,
            1,
            self.quats / self.quats.norm(dim=-1, keepdim=True),
            self.viewmat,
            focal,
            focal,
            W / 2,
            H / 2,
            H,
            W,
            B_SIZE,
        )
        out_img, alpha = rasterize_gaussians(
            xys,
            depths,
            radii,
            conics,
            num_tiles_hit,
            torch.sigmoid(self.rgbs),
            torch.sigmoid(self.opacities),
            H,
            W,
            B_SIZE,
            self.background,
            return_alpha=True,
        )
        out_dino = None
        if self.gt_dino is not None:
            out_dino = rasterize_gaussians(
                xys.detach(),
                depths.detach(),
                radii.detach(),
                conics.detach(),
                num_tiles_hit.detach(),
                torch.sigmoid(self.dino),
                torch.sigmoid(self.opacities).detach(),
                H,
                W,
                B_SIZE,
                self.background,
            )

        return out_img, out_dino, alpha

    def train(
        self,
        iterations: int = 1000,
        lr: float = 0.01,
        save_imgs: bool = False,
        B_SIZE: int = 14,
    ):
        optimizer = optim.Adam(
            [self.rgbs, self.means, self.scales, self.opacities, self.quats]
            + ([self.dino] if self.dino is not None else []),
            lr,
        )
        mse_loss = torch.nn.MSELoss()
        frames = []
        times = [0] * 3  # project, rasterize, backward
        B_SIZE = 16
        for iter in range(iterations):
            start = time.time()
            (
                xys,
                depths,
                radii,
                conics,
                compensation,
                num_tiles_hit,
                cov3d,
            ) = project_gaussians(
                self.means,
                self.scales,
                1,
                self.quats / self.quats.norm(dim=-1, keepdim=True),
                self.viewmat,
                self.focal,
                self.focal,
                self.W / 2,
                self.H / 2,
                self.H,
                self.W,
                B_SIZE,
            )
            torch.cuda.synchronize()
            times[0] += time.time() - start
            start = time.time()
            out_img, alpha = rasterize_gaussians(
                xys,
                depths,
                radii,
                conics,
                num_tiles_hit,
                torch.sigmoid(self.rgbs),
                torch.sigmoid(self.opacities),
                self.H,
                self.W,
                B_SIZE,
                self.background,
                return_alpha=True,
            )
            out_dino = None
            if self.gt_dino is not None:
                out_dino = rasterize_gaussians(
                    xys.detach(),
                    depths.detach(),
                    radii.detach(),
                    conics.detach(),
                    num_tiles_hit,
                    torch.sigmoid(self.dino),
                    torch.sigmoid(self.opacities).detach(),
                    self.H,
                    self.W,
                    B_SIZE,
                    self.background,
                )
            torch.cuda.synchronize()
            times[1] += time.time() - start
            loss = mse_loss(
                torch.cat([out_img, alpha[..., None]], dim=-1), self.gt_image
            )
            if out_dino is not None:
                loss += 0.1 * mse_loss(out_dino, self.gt_dino)

            scale_exp = torch.exp(self.scales)
            scale_reg = (
                torch.maximum(
                    scale_exp.amax(dim=-1) / scale_exp.amin(dim=-1),
                    torch.tensor(1.0),
                )
                - 1.0
            )
            scale_reg = 0.1 * scale_reg.mean()
            loss += scale_reg

            optimizer.zero_grad()
            start = time.time()
            loss.backward()
            torch.cuda.synchronize()
            times[2] += time.time() - start
            optimizer.step()
            print(f"Iteration {iter + 1}/{iterations}, Loss: {loss.item()}")

            # if save_imgs and iter % 5 == 0:
            #     frames.append((torch.cat([out_img, alpha[..., None]], dim=-1).detach().cpu().numpy() * 255).astype(np.uint8))
        if save_imgs:
            # save them as a gif with PIL
            # frames = [Image.fromarray(frame) for frame in frames]
            # out_dir = os.path.join(os.getcwd(), "renders")
            # os.makedirs(out_dir, exist_ok=True)
            # frames[-1].save(
            #     f"{out_dir}/training.png",
            #     # save_all=True,
            #     # append_images=frames[1:],
            #         # optimize=False,
            #         # duration=5,
            #         # loop=0,
            # )
            out_img, out_dino, alpha = self.render()

            out_dir = os.path.join(os.getcwd(), "renders")
            frame = Image.fromarray(
                (
                    torch.cat([out_img, alpha.unsqueeze(-1)], dim=-1)
                    .detach()
                    .cpu()
                    .numpy()
                    * 255
                ).astype(np.uint8)
            )
            frame.save(f"training.png")
            if out_dino is not None:
                frame = Image.fromarray(
                    (
                        torch.cat([out_dino, alpha.unsqueeze(-1)], dim=-1)
                        .detach()
                        .cpu()
                        .numpy()
                        * 255
                    ).astype(np.uint8)
                )
                frame.save(f"training_dino.png")
        print(
            f"Total(s):\nProject: {times[0]:.3f}, Rasterize: {times[1]:.3f}, Backward: {times[2]:.3f}"
        )
        print(
            f"Per step(s):\nProject: {times[0]/iterations:.5f}, Rasterize: {times[1]/iterations:.5f}, Backward: {times[2]/iterations:.5f}"
        )


def image_path_to_tensor(image_path: Path, resize=None):
    import torchvision.transforms as transforms

    img = Image.open(image_path)
    transform = transforms.ToTensor()
    if resize:
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Resize(resize)]
        )
    img_tensor = transform(img).permute(1, 2, 0)[..., :4]
    return img_tensor


def main(
    height: int = 256,
    width: int = 256,
    num_points: int = 10000,
    save_imgs: bool = True,
    img_path: Optional[Path] = None,
    dino_path: Optional[Path] = None,
    iterations: int = 1000,
    lr: float = 0.01,
) -> None:
    gt_dino = None
    if img_path:
        gt_image = image_path_to_tensor(img_path)
        if dino_path:
            gt_dino = image_path_to_tensor(
                dino_path, resize=(gt_image.shape[0], gt_image.shape[1])
            )[..., :3]
    else:
        gt_image = torch.ones((height, width, 3)) * 1.0
        # make top left and bottom right red, blue
        gt_image[: height // 2, : width // 2, :] = torch.tensor([1.0, 0.0, 0.0])
        gt_image[height // 2 :, width // 2 :, :] = torch.tensor([0.0, 0.0, 1.0])

    trainer = SimpleTrainer(gt_image=gt_image, num_points=num_points, gt_dino=gt_dino)
    trainer.train(
        iterations=iterations,
        lr=lr,
        save_imgs=save_imgs,
    )


if __name__ == "__main__":
    tyro.cli(main)
