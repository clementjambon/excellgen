from dataclasses import dataclass, field
from dataclass_wizard import YAMLWizard

import os
import yaml
from datetime import datetime

try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper


@dataclass
class BaseConfig(YAMLWizard):

    name: str = "gaussian_model"
    log_dir: str = ""
    gca_template: str = (
        "deps/fast-gca/configs/default/template_fine=1_coarse=3_geo=4_dino=4_s0_T=5.yaml"
    )

    # -----------------------
    # Training
    # -----------------------

    # TODO: count iterations in data samples somehow
    n_iterations: int = 25000
    init_batch_size: int = 1024
    target_sample_batch_size: int = 1 << 18

    optimizer: dict = field(
        default_factory=lambda: {
            "name": "Adam",
            "options": {"lr": 0.01, "eps": 1e-15, "weight_decay": 1e-6},
        }
    )

    feature_weight: float = 0.01
    dist_loss_weight: float = 0.0
    commit_loss_weight: float = 0.0

    feat_optimization_start: int = 0

    training_background_color: str = "black"

    # -----------------------
    # Data
    # -----------------------
    data_dir: str = ""
    dataset_type: str = "nerfstudio"
    test_skip: int = 8
    factor: int = 2  # Only for colmap
    factor_dino: int = 2
    n_images: int = -1
    n_images_test: int = 10
    eval_mode: str = "interval"
    eval_interval: int = 8
    aabb: list[int] = field(default_factory=lambda: [-1.0, -1.0, -1.0, 1.0, 1.0, 1.0])
    random_init: bool = False
    auto_scale_poses: bool = True
    random_scale: float = 10.0

    # -----------------------
    # Model
    # -----------------------

    grid_resolution: int = 128
    grid_nlvl: int = 4

    # render parameters
    render_step_size: float = 1e-3
    alpha_thre: float = 1e-2
    cone_angle: float = 0.004

    near_plane: float = 0.2
    far_plane: float = 1.0e10

    geo_feat_dim: int = 15

    # feature_dim
    feature_dim: int = 8
    loss_on_quantized_feature: bool = False
    feature_quantizer: dict = field(default_factory=lambda: {"type": "none"})

    # TODO: move somewhere else!
    sh_degree: int = 3
    sh_degree_interval: int = 1000
    use_scale_regularization: bool = True
    max_gauss_ratio: float = 10.0
    ssim_lambda: float = 0.2

    # -----------------------
    # IO
    # -----------------------

    log_steps: int = 100
    test_steps: int = 5000
    ckpt_steps: int = 10000
    ckpt_last_only: bool = True
    test_pca: bool = True


CONFIGS_MAP = {
    BaseConfig.name: BaseConfig,
}

DEFAULT_CONFIG = "configs/default.yaml"


def load_config(
    config_path: str, log_dir: str | None = None, data_dir: str | None = None
):

    if not os.path.exists(config_path):
        print("WARNING: could not find config path, opening default config!")
        config = yaml.load(open(DEFAULT_CONFIG, "r"), Loader)
        config: BaseConfig = CONFIGS_MAP[config["name"]].from_yaml_file(DEFAULT_CONFIG)
    else:
        config = yaml.load(open(config_path, "r"), Loader)
        config: BaseConfig = CONFIGS_MAP[config["name"]].from_yaml_file(config_path)

    config.gca_template = os.path.expandvars(config.gca_template)

    if log_dir is None:
        log_dir = "./log/" + datetime.now().strftime("%m-%d-%H:%M:%S")

    config.log_dir = os.path.expandvars(log_dir)
    if data_dir is not None:
        config.data_dir = data_dir
    # Overwrite with absolute path to be able to reload it from GCAs
    # config.data_dir = os.path.abspath(config.data_dir)

    return config
