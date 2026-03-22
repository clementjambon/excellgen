import os
import time
import imageio
import numpy as np

import torch
import tqdm

from pytorch_msssim import SSIM

from sprim.inverse.nerfstudio_loader import SubjectLoader
from sprim.gaussians.gaussian_model import GaussianModel
from sprim.configs.base import BaseConfig

from nerfstudio.engine.optimizers import AdamOptimizerConfig, Optimizers
from nerfstudio.engine.schedulers import ExponentialDecaySchedulerConfig

optimizer_configs = {
    "means": {
        "optimizer": AdamOptimizerConfig(lr=1.6e-4, eps=1e-15),
        "scheduler": ExponentialDecaySchedulerConfig(
            lr_final=1.6e-6,
            max_steps=30000,
        ),
    },
    "features_dc": {
        "optimizer": AdamOptimizerConfig(lr=0.0025, eps=1e-15),
        "scheduler": None,
    },
    "features_rest": {
        "optimizer": AdamOptimizerConfig(lr=0.0025 / 20, eps=1e-15),
        "scheduler": None,
    },
    "features_feat": {
        "optimizer": AdamOptimizerConfig(lr=0.0025, eps=1e-15),
        "scheduler": None,
    },
    "opacities": {
        "optimizer": AdamOptimizerConfig(lr=0.05, eps=1e-15),
        "scheduler": None,
    },
    "scales": {
        "optimizer": AdamOptimizerConfig(lr=0.005, eps=1e-15),
        "scheduler": None,
    },
    "quats": {"optimizer": AdamOptimizerConfig(lr=0.001, eps=1e-15), "scheduler": None},
    "envmap": {
        "optimizer": AdamOptimizerConfig(lr=0.0025, eps=1e-15),
        "scheduler": None,
    },
}


class Trainer:

    def __init__(
        self,
        config: BaseConfig,
        gaussian_model: GaussianModel,
        optimizers: Optimizers,
        train_dataset: SubjectLoader,
        test_dataset: SubjectLoader,
        device: torch.device,
        step: int,
    ) -> None:

        self.config = config
        self.gaussian_model = gaussian_model
        self.optimizers = optimizers
        self.device = device
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset

        self.grad_scaler = torch.cuda.amp.GradScaler(2**10)

        # metrics
        from torchmetrics.image import PeakSignalNoiseRatio
        from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim = SSIM(data_range=1.0, size_average=True, channel=3)
        self.lpips = LearnedPerceptualImagePatchSimilarity(normalize=True)

        self.step = step

        self.tic = time.time()

    def training_step(self) -> bool:
        if self.step > self.config.n_iterations:
            return False

        self.gaussian_model.train()
        self.gaussian_model.step = self.step

        i = torch.randint(0, len(self.train_dataset), (1,)).item()
        data = self.train_dataset[i]

        # Specify background
        background = torch.zeros(3, dtype=torch.float32, device=self.device)
        if self.config.training_background_color == "white":
            background = torch.ones(3, dtype=torch.float32, device=self.device)
        if self.config.training_background_color == "random":
            background = torch.rand(3, dtype=torch.float32, device=self.device)

        render_pkg = self.gaussian_model.render(
            data["camera"],
            self.config,
            return_feat=self.step >= self.config.feat_optimization_start,
            background_color=background,
        )
        pred_img = render_pkg["rgb"]

        gt_img = data["image"]
        gt_img = self.gaussian_model._downscale_if_required(gt_img)

        # Composite the background color with
        if gt_img.shape[-1] == 4:
            gt_img = gt_img[:, :, :3] * gt_img[:, :, 3:4] + background.view(1, 1, 3) * (
                1 - gt_img[:, :, 3:4]
            )

        Ll1 = torch.abs(gt_img - pred_img).mean()
        simloss = 1 - self.ssim(
            gt_img.permute(2, 0, 1)[None, ...], pred_img.permute(2, 0, 1)[None, ...]
        )

        if self.config.use_scale_regularization and self.step % 10 == 0:
            scale_exp = torch.exp(self.gaussian_model.scales)
            scale_reg = (
                torch.maximum(
                    scale_exp.amax(dim=-1) / scale_exp.amin(dim=-1),
                    torch.tensor(self.config.max_gauss_ratio),
                )
                - self.config.max_gauss_ratio
            )
            scale_reg = 0.1 * scale_reg.mean()
        else:
            scale_reg = torch.tensor(0.0).to(self.device)

        # compute loss
        loss = (1 - self.config.ssim_lambda) * Ll1 + self.config.ssim_lambda * simloss
        loss += scale_reg

        # TODO: render at lower-res
        if self.step >= self.config.feat_optimization_start:
            pred_feat = render_pkg["feat"]
            gt_feat = data["feat"]
            gt_feat = self.gaussian_model._downscale_if_required(gt_feat)

            # Handle feature quantizer
            optimized_feat = pred_feat
            feat_shape = pred_feat.shape
            if self.gaussian_model.feature_quantizer is not None:
                quantized_feat, _, commit_loss = self.gaussian_model.feature_quantizer(
                    pred_feat.view((-1, self.gaussian_model.features_dim))
                )
                quantized_feat = quantized_feat.reshape(feat_shape)
                if self.config.loss_on_quantized_feature:
                    optimized_feat = quantized_feat
                loss += self.config.commit_loss_weight * commit_loss.mean()

            loss += (
                self.config.feature_weight * torch.abs(gt_feat - optimized_feat).mean()
            )

        self.optimizers.zero_grad_all()
        # do not unscale it because we are using Adam.
        # self.grad_scaler.scale(loss).backward()
        loss.backward()
        self.optimizers.optimizer_step_all()
        self.optimizers.scheduler_step_all(self.step)

        self.gaussian_model.after_train(self.step)

        if self.step > 0 and self.step % self.gaussian_model.refine_every == 0:
            self.gaussian_model.refinement_after(self.optimizers, self.step)

        if self.step % self.config.log_steps == 0:
            elapsed_time = time.time() - self.tic
            # loss = F.mse_loss(rgb, pixels)
            # psnr = -10.0 * torch.log(loss) / np.log(10.0)
            print(
                f"elapsed_time={elapsed_time:.2f}s | step={self.step} | "
                f"loss={loss:.5f}"
                # f"loss={loss:.5f} | psnr={psnr:.2f} | "
                # f"n_rendering_samples={n_rendering_samples:d} | num_rays={len(pixels):d} | "
                # f"max_depth={depth.max():.3f} | "
            )

        if (
            self.step > 0
            and self.step % self.config.ckpt_steps == 0
            or self.step == self.config.n_iterations
        ):
            ckpt_folder = os.path.join(self.config.log_dir, "ckpts")
            ckpt_path = os.path.join(
                ckpt_folder,
                f"{self.step:08d}.pt" if not self.config.ckpt_last_only else "last.pt",
            )
            os.makedirs(ckpt_folder, exist_ok=True)
            torch.save(
                {
                    "step": self.step,
                    "name": self.config.name,
                    "gaussian_model": self.gaussian_model.state_dict(),
                    "optimizers": {
                        k: v.state_dict()
                        for (k, v) in self.optimizers.optimizers.items()
                    },
                    "schedulers": {
                        k: v.state_dict()
                        for (k, v) in self.optimizers.schedulers.items()
                    },
                },
                ckpt_path,
            )
            print("Saved checkpoint at", ckpt_path)

        if self.step > 0 and self.step % self.config.test_steps == 0:
            # evaluation
            self.gaussian_model.eval()

            renders_folder = os.path.join(
                self.config.log_dir, "renders", str(self.step)
            )
            os.makedirs(renders_folder, exist_ok=True)

            psnrs = []
            lpips = []
            with torch.no_grad():
                for i in tqdm.tqdm(range(len(self.test_dataset))):
                    data = self.test_dataset[i]

                    render_pkg = self.gaussian_model.render(data["camera"], self.config)
                    pred_img = render_pkg["rgb"]

                    # mse = F.mse_loss(rgb, pixels)
                    # psnr = -10.0 * torch.log(mse) / np.log(10.0)
                    # psnrs.append(psnr.item())
                    # lpips.append(self.lpips_fn(rgb, pixels).item())

                    renders_path = os.path.join(renders_folder, f"rgb_{i:08d}.png")
                    imageio.imwrite(
                        renders_path,
                        (pred_img.cpu().numpy() * 255).astype(np.uint8),
                    )

                    # if self.config.test_pca:
                    #     feat_dict = {"geo_feat": geo_feat, "dino": feat}

                    #     for name, v in feat_dict.items():
                    #         pca = PCA(n_components=3)
                    #         feature_maps_pca = pca.fit_transform(
                    #             v.reshape(-1, v.shape[-1]).cpu().numpy()
                    #         )
                    #         pca_features_min = feature_maps_pca.min(axis=(0, 1))
                    #         pca_features_max = feature_maps_pca.max(axis=(0, 1))
                    #         pca_features = (feature_maps_pca - pca_features_min) / (
                    #             pca_features_max - pca_features_min
                    #         )

                    #         pca_features = pca_features.reshape(
                    #             rgb.shape[0], rgb.shape[1], 3
                    #         )

                    #         renders_path = os.path.join(
                    #             renders_folder, f"{name}_{i:08d}.png"
                    #         )
                    #         imageio.imwrite(
                    #             renders_path, (pca_features * 255).astype(np.uint8)
                    #         )

            # psnr_avg = sum(psnrs) / len(psnrs)
            # lpips_avg = sum(lpips) / len(lpips)
            # print(f"evaluation: psnr_avg={psnr_avg}, lpips_avg={lpips_avg}")

        self.step += 1
        return True
