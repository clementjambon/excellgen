from typing import Tuple, Any, Dict
from dataclasses import asdict
import os
import glob
import random
import numpy as np
from pathlib import Path
import yaml

import torch

from sprim.configs.base import load_config, BaseConfig
from sprim.gaussians.gaussian_model import GaussianModel
from sprim.gaussians.trainer import Trainer
from sprim.inverse.grower import Grower

SCREENSHOTS_FOLDER = "screenshots"
RECORDINGS_FOLDER = "recordings"


@torch.no_grad()
def load_model_and_data_inverse(
    ckpt_path,
    update_occ: bool = True,
    override_estimator: bool = False,
    alpha_thres: float = 0.01,
):
    # Load model and configs
    base_dir = os.path.dirname(os.path.dirname(ckpt_path))
    if os.path.basename(os.path.dirname(ckpt_path)) == "gaussian_ckpt":
        base_dir = os.path.dirname(base_dir)
    config_path = os.path.join(base_dir, "config_gaussians.yaml")
    config: BaseConfig = load_config(config_path)
    # Reset log_dir in case this was ,
    config.log_dir = base_dir
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(ckpt_path, map_location=device)

    assert config.name == "gaussian_model"

    gaussian_model = GaussianModel(
        device=device,
        # Warning this is hardcoded!
        num_train_data=200,
        random_init=config.random_init,
        # num_random=
        sh_degree=config.sh_degree,
        feature_dim=config.feature_dim,
        num_random=len(checkpoint["gaussian_model"]["gauss_params.means"]),
        feature_quantizer=config.feature_quantizer,
    ).to(device)

    k_to_delete = []
    for k in checkpoint["gaussian_model"].keys():
        if k.startswith("feature_quantizer"):
            k_to_delete.append(k)

    for k in k_to_delete:
        del checkpoint["gaussian_model"][k]

    gaussian_model.load_state_dict(state_dict=checkpoint["gaussian_model"])

    return gaussian_model, None, None, config, device


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def resolve_datasets(config: BaseConfig, device):
    if config.dataset_type == "nerfstudio":
        from sprim.inverse.nerfstudio_loader import SubjectLoader

        train_dataset = SubjectLoader(
            root_fp=config.data_dir,
            split="train",
            num_rays=config.init_batch_size,
            device=device,
            pca_dim=config.feature_dim,
            factor=config.factor,
            factor_dino=config.factor_dino,
            n_images=config.n_images,
            n_images_test=config.n_images_test,
            eval_mode=config.eval_mode,
            eval_interval=config.eval_interval,
            auto_scale_poses=config.auto_scale_poses,
        )

        test_dataset = SubjectLoader(
            root_fp=config.data_dir,
            split="test",
            num_rays=config.init_batch_size,
            device=device,
            pca_dim=config.feature_dim,
            factor=config.factor,
            factor_dino=config.factor_dino,
            n_images=config.n_images,
            n_images_test=config.n_images_test,
            eval_mode=config.eval_mode,
            eval_interval=config.eval_interval,
            auto_scale_poses=config.auto_scale_poses,
        )

    else:
        raise ValueError(f"{config.dataset_type} is not a valid dataset type!")

    return train_dataset, test_dataset


def init_trainer(
    gaussian_model: GaussianModel | None,
    config: BaseConfig,
    ckpt_path: str | None = None,
) -> Tuple[GaussianModel, Trainer]:

    assert config.name == "gaussian_model"

    from sprim.gaussians.trainer import Trainer, optimizer_configs
    from sprim.gaussians.gaussian_model import GaussianModel
    from nerfstudio.engine.optimizers import Optimizers

    device = "cuda"

    train_dataset, test_dataset = resolve_datasets(config=config, device=device)

    # TODO: provide proper scene bounds (with poses?)
    if gaussian_model is None:
        gaussian_model = GaussianModel(
            device=device,
            num_train_data=len(train_dataset),
            sh_degree=config.sh_degree,
            feature_dim=config.feature_dim,
            feature_quantizer=config.feature_quantizer,
            render_pca=train_dataset.render_pca,
            seed_points=train_dataset.seed_points,
            random_scale=config.random_scale,
        ).to(device)

    step = 0

    optimizers = Optimizers(
        config=optimizer_configs,
        param_groups=gaussian_model.get_all_param_groups(),
    )
    if ckpt_path is not None:
        checkpoint = torch.load(ckpt_path, map_location=device)
        if "optimizers" in checkpoint:
            optimizers.load_optimizers(checkpoint["optimizers"])
            optimizers.load_schedulers(checkpoint["schedulers"])
            step = checkpoint["step"]

    config.test_steps = 1000000000000000000000000
    trainer = Trainer(
        config,
        gaussian_model,
        optimizers=optimizers,
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        device="cuda",
        step=step,
    )

    return gaussian_model, trainer


def load_models_only(
    input_path,
    gca_path=None,
    gaussian_update_callback=lambda: None,
) -> Tuple[GaussianModel, BaseConfig, Grower, Trainer]:
    trainer = None

    if Path(input_path).suffix == ".yaml":
        config: BaseConfig = load_config(input_path)

        # Create log_dir and copy config file
        os.makedirs(config.log_dir, exist_ok=True)
        yaml.dump(
            asdict(config),
            open(os.path.join(config.log_dir, "config_gaussians.yaml"), "w"),
        )

        gaussian_model, trainer = init_trainer(None, config)

    elif Path(input_path).suffix == ".pt":
        (
            gaussian_model,
            _,
            _,
            config,
            _,
        ) = load_model_and_data_inverse(
            input_path,
        )

    else:
        raise ValueError("Wrong input file!")

    grower = (
        None
        if gca_path is None
        else Grower(
            gca_path,
            gaussian_model,
            gaussian_update_callback=gaussian_update_callback,
            get_transform=None,
        )
    )

    return gaussian_model, config, grower, trainer


def get_camera_dict() -> Dict[str, Any]:
    import polyscope as ps

    cam_parameters = ps.get_view_camera_parameters()
    fov_vertical_deg = cam_parameters.get_fov_vertical_deg()
    aspect = cam_parameters.get_aspect()
    E = cam_parameters.get_E()
    window_size = ps.get_window_size()
    window_size = np.array([int(window_size[0]), int(window_size[1])])
    return {
        "fov_vertical_deg": fov_vertical_deg,
        "aspect": aspect,
        "E": E,
        "window_size": window_size,
    }


def load_camera_from_dict(data: Dict[str, Any]):
    import polyscope as ps

    fov_vertical_deg = data["fov_vertical_deg"]
    aspect = data["aspect"]
    E = data["E"]

    window_size = None
    if "window_size" in data:
        window_size = data["window_size"]

    intrinsics = ps.CameraIntrinsics(fov_vertical_deg=fov_vertical_deg, aspect=aspect)
    extrinsics = ps.CameraExtrinsics(mat=E)
    params = ps.CameraParameters(intrinsics=intrinsics, extrinsics=extrinsics)

    return params, (int(window_size[0]), int(window_size[1]))


def load_camera_from_file(camera_path: str):
    if not os.path.exists(camera_path):
        print(f"No camera file at {camera_path}")
        return None, None
    data = np.load(camera_path)

    return load_camera_from_dict(data)


def resolve_screenshot_path(is_video: bool = False) -> str:
    screenshot_folder = RECORDINGS_FOLDER if is_video else SCREENSHOTS_FOLDER
    screenshot_extension = ".mp4" if is_video else ".png"
    screenshot_prefix = "recording" if is_video else "screenshot"
    # Resolve path
    os.makedirs(screenshot_folder, exist_ok=True)
    prev_names = sorted(
        glob.glob(f"{screenshot_folder}/{screenshot_prefix}_*{screenshot_extension}")
    )
    if len(prev_names) == 0:
        i_screenshot = 0
    else:
        i_screenshot = (
            int(
                os.path.splitext(os.path.basename(prev_names[-1]))[0][
                    len(f"{screenshot_prefix}_") :
                ]
            )
            + 1
        )

    return os.path.join(
        screenshot_folder,
        f"{screenshot_prefix}_{i_screenshot:06d}{screenshot_extension}",
    )
