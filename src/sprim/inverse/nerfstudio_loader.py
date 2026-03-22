"""
Copyright (c) 2022 Ruilong Li, UC Berkeley.
"""

from typing import List, Dict
import os
import json
from pathlib import Path
from typing import Tuple
from PIL import Image

import imageio
import numpy as np
import torch
from tqdm import tqdm

from sprim.utils.geometry import undistort_image
from sprim.inverse.feature_extractor import extract_features
from sprim.configs.base import BaseConfig

from nerfstudio.cameras import camera_utils
from nerfstudio.cameras.cameras import CAMERA_MODEL_TO_TYPE, Cameras, CameraType
from nerfstudio.data.utils.dataparsers_utils import (
    get_train_eval_split_all,
    get_train_eval_split_filename,
    get_train_eval_split_fraction,
    get_train_eval_split_interval,
)

MAX_AUTO_RESOLUTION = 1600


def _get_fname(
    filepath: Path,
    data_dir: Path,
    downsample_folder_prefix="images_",
    factor: int | None = None,
) -> Path:
    """Get the filename of the image file.
    downsample_folder_prefix can be used to point to auxiliary image data, e.g. masks

    filepath: the base file name of the transformations.
    data_dir: the directory of the data that contains the transform file
    downsample_folder_prefix: prefix of the newly generated downsampled images
    """

    if factor is None:
        if factor is None:
            test_img = Image.open(data_dir / filepath)
            h, w = test_img.size
            max_res = max(h, w)
            df = 0
            while True:
                if (max_res / 2 ** (df)) <= MAX_AUTO_RESOLUTION:
                    break
                if not (
                    data_dir / f"{downsample_folder_prefix}{2**(df+1)}" / filepath.name
                ).exists():
                    break
                df += 1

            factor = 2**df
            print(f"Auto image downscale factor of {factor}")

    if factor > 1:
        return data_dir / f"{downsample_folder_prefix}{factor}" / filepath.name, factor
    return data_dir / filepath, factor


def _load_3D_points(
    ply_file_path: Path, transform_matrix: torch.Tensor, scale_factor: float
):
    """Loads point clouds positions and colors from .ply

    Args:
        ply_file_path: Path to .ply file
        transform_matrix: Matrix to transform world coordinates
        scale_factor: How much to scale the camera origins by.

    Returns:
        A dictionary of points: points3D_xyz and colors: points3D_rgb
    """
    import open3d as o3d  # Importing open3d is slow, so we only do it if we need it.

    pcd = o3d.io.read_point_cloud(str(ply_file_path))

    # if no points found don't read in an initial point cloud
    if len(pcd.points) == 0:
        return None

    points3D = torch.from_numpy(np.asarray(pcd.points, dtype=np.float32))
    points3D = (
        torch.cat(
            (
                points3D,
                torch.ones_like(points3D[..., :1]),
            ),
            -1,
        )
        @ transform_matrix.T
    )
    points3D *= scale_factor
    points3D_rgb = torch.from_numpy((np.asarray(pcd.colors) * 255).astype(np.uint8))

    out = {
        "points3D_xyz": points3D,
        "points3D_rgb": points3D_rgb,
    }
    return out


def convert_filename(filepath):
    if len(os.path.splitext(filepath)[1]) == 0:
        return filepath + ".png"
    else:
        return filepath


def _load_nerfstudio(
    root_fp: str,
    split: str,
    factor: int = 1,
    factor_dino: int = 2,
    eval_mode: str = "all",
    eval_interval: int = 8,
    orientation_method: str = "up",
    center_method: str = "poses",
    auto_scale_poses: bool = True,
    cameras_only: bool = False,
    load_3d_points: bool = True,
):
    data_dir = Path(root_fp)

    transform_path = os.path.join(data_dir, "transforms.json")
    with open(transform_path, "r") as fp:
        meta = json.load(fp)

    image_filenames = []
    image_dino_filenames = []
    mask_filenames = []
    depth_filenames = []
    poses = []

    fx_fixed = "fl_x" in meta
    fy_fixed = "fl_y" in meta
    cx_fixed = "cx" in meta
    cy_fixed = "cy" in meta
    height_fixed = "h" in meta
    width_fixed = "w" in meta
    distort_fixed = False
    for distort_key in ["k1", "k2", "k3", "p1", "p2", "distortion_params"]:
        if distort_key in meta:
            distort_fixed = True
            break
    fisheye_crop_radius = meta.get("fisheye_crop_radius", None)
    fx = []
    fy = []
    cx = []
    cy = []
    height = []
    width = []
    distort = []

    # sort the frames by fname
    fnames = []
    fnames_dino = []
    for frame in meta["frames"]:
        filepath = Path(frame["file_path"])
        fname, factor = _get_fname(filepath, data_dir, factor=factor)
        fname_dino, _ = _get_fname(filepath, data_dir, factor=factor_dino)
        fnames.append(fname)
        fnames_dino.append(fname_dino)
    inds = np.argsort(fnames)
    frames = [meta["frames"][ind] for ind in inds]

    for frame in frames:
        filepath = Path(frame["file_path"])
        fname, factor = _get_fname(filepath, data_dir, factor=factor)
        fname_dino, _ = _get_fname(filepath, data_dir, factor=factor_dino)

        if not fx_fixed:
            assert "fl_x" in frame, "fx not specified in frame"
            fx.append(float(frame["fl_x"]))
        if not fy_fixed:
            assert "fl_y" in frame, "fy not specified in frame"
            fy.append(float(frame["fl_y"]))
        if not cx_fixed:
            assert "cx" in frame, "cx not specified in frame"
            cx.append(float(frame["cx"]))
        if not cy_fixed:
            assert "cy" in frame, "cy not specified in frame"
            cy.append(float(frame["cy"]))
        if not height_fixed:
            assert "h" in frame, "height not specified in frame"
            height.append(int(frame["h"]))
        if not width_fixed:
            assert "w" in frame, "width not specified in frame"
            width.append(int(frame["w"]))
        if not distort_fixed:
            distort.append(
                torch.tensor(frame["distortion_params"], dtype=torch.float32)
                if "distortion_params" in frame
                else camera_utils.get_distortion_params(
                    k1=float(frame["k1"]) if "k1" in frame else 0.0,
                    k2=float(frame["k2"]) if "k2" in frame else 0.0,
                    k3=float(frame["k3"]) if "k3" in frame else 0.0,
                    k4=float(frame["k4"]) if "k4" in frame else 0.0,
                    p1=float(frame["p1"]) if "p1" in frame else 0.0,
                    p2=float(frame["p2"]) if "p2" in frame else 0.0,
                )
            )

        image_filenames.append(convert_filename(str(fname)))
        image_dino_filenames.append(convert_filename(str(fname_dino)))
        poses.append(np.array(frame["transform_matrix"]))

    has_split_files_spec = any(
        f"{split}_filenames" in meta for split in ("train", "val", "test")
    )
    if f"{split}_filenames" in meta:
        assert False
        # Validate split first
        split_filenames = set(
            _get_fname(Path(x), data_dir, factor=factor)[0]
            for x in meta[f"{split}_filenames"]
        )
        unmatched_filenames = split_filenames.difference(image_filenames)
        if unmatched_filenames:
            raise RuntimeError(
                f"Some filenames for split {split} were not found: {unmatched_filenames}."
            )

        indices = [
            i for i, path in enumerate(image_filenames) if path in split_filenames
        ]
        print(f"[yellow] Dataset is overriding {split}_indices to {indices}")
        indices = np.array(indices, dtype=np.int32)
    elif has_split_files_spec:
        raise RuntimeError(
            f"The dataset's list of filenames for split {split} is missing."
        )
    else:
        # find train and eval indices based on the eval_mode specified
        if eval_mode == "fraction":
            i_train, i_eval = get_train_eval_split_fraction(
                image_filenames, self.config.train_split_fraction
            )
        elif eval_mode == "filename":
            i_train, i_eval = get_train_eval_split_filename(image_filenames)
        elif eval_mode == "interval":
            i_train, i_eval = get_train_eval_split_interval(
                image_filenames, eval_interval
            )
        elif eval_mode == "all":
            print(
                "[yellow] Be careful with '--eval-mode=all'. If using camera optimization, the cameras may diverge in the current implementation, giving unpredictable results."
            )
            i_train, i_eval = get_train_eval_split_all(image_filenames)
        else:
            raise ValueError(f"Unknown eval mode {eval_mode}")

        if split == "train":
            # perm = torch.randperm(len(i_train))
            # n_train = min(len(i_train), n_images) if n_images > 0 else len(i_train)
            # idx = perm[:n_train]

            # indices = i_train[idx]
            indices = i_train
        elif split in ["val", "test"]:
            # indices = i_eval[:n_images_test]
            indices = i_eval
        else:
            raise ValueError(f"Unknown dataparser split {split}")

    if "orientation_override" in meta:
        orientation_method = meta["orientation_override"]
        print(
            f"[yellow] Dataset is overriding orientation method to {orientation_method}"
        )
    else:
        orientation_method = orientation_method

    poses = torch.from_numpy(np.array(poses).astype(np.float32))
    poses, transform_matrix = camera_utils.auto_orient_and_center_poses(
        poses,
        method=orientation_method,
        center_method=center_method,
    )

    # Scale poses
    scale_factor = 1.0
    if auto_scale_poses:
        scale_factor /= float(torch.max(torch.abs(poses[:, :3, 3])))

    poses[:, :3, 3] *= scale_factor

    # Choose image_filenames and poses based on split, but after auto orient and scaling the poses.
    image_filenames = [image_filenames[i] for i in indices]
    image_dino_filenames = [image_dino_filenames[i] for i in indices]

    idx_tensor = torch.tensor(indices, dtype=torch.long)
    poses = poses[idx_tensor]

    if "camera_model" in meta:
        camera_type = CAMERA_MODEL_TO_TYPE[meta["camera_model"]]
    else:
        camera_type = CameraType.PERSPECTIVE

    fx = (
        float(meta["fl_x"])
        if fx_fixed
        else torch.tensor(fx, dtype=torch.float32)[idx_tensor]
    )
    fy = (
        float(meta["fl_y"])
        if fy_fixed
        else torch.tensor(fy, dtype=torch.float32)[idx_tensor]
    )
    cx = (
        float(meta["cx"])
        if cx_fixed
        else torch.tensor(cx, dtype=torch.float32)[idx_tensor]
    )
    cy = (
        float(meta["cy"])
        if cy_fixed
        else torch.tensor(cy, dtype=torch.float32)[idx_tensor]
    )
    height = (
        int(meta["h"])
        if height_fixed
        else torch.tensor(height, dtype=torch.int32)[idx_tensor]
    )
    width = (
        int(meta["w"])
        if width_fixed
        else torch.tensor(width, dtype=torch.int32)[idx_tensor]
    )
    if distort_fixed:
        distortion_params = (
            torch.tensor(meta["distortion_params"], dtype=torch.float32)
            if "distortion_params" in meta
            else camera_utils.get_distortion_params(
                k1=float(meta["k1"]) if "k1" in meta else 0.0,
                k2=float(meta["k2"]) if "k2" in meta else 0.0,
                k3=float(meta["k3"]) if "k3" in meta else 0.0,
                k4=float(meta["k4"]) if "k4" in meta else 0.0,
                p1=float(meta["p1"]) if "p1" in meta else 0.0,
                p2=float(meta["p2"]) if "p2" in meta else 0.0,
            )
        )
    else:
        distortion_params = torch.stack(distort, dim=0)[idx_tensor]

    # Only add fisheye crop radius parameter if the images are actually fisheye, to allow the same config to be used
    # for both fisheye and non-fisheye datasets.
    metadata = {}
    if (camera_type in [CameraType.FISHEYE, CameraType.FISHEYE624]) and (
        fisheye_crop_radius is not None
    ):
        metadata["fisheye_crop_radius"] = fisheye_crop_radius

    cameras = Cameras(
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
        distortion_params=distortion_params,
        height=height,
        width=width,
        camera_to_worlds=poses[:, :3, :4],
        camera_type=camera_type,
        metadata=metadata,
    )

    assert factor is not None
    cameras.rescale_output_resolution(scaling_factor=1.0 / factor)

    if cameras_only:
        return cameras

    # ------------------------
    # SPARSE PC
    # ------------------------

    if "applied_scale" in meta:
        applied_scale = float(meta["applied_scale"])
        scale_factor *= applied_scale

    sparse_points = None
    if load_3d_points:
        if "ply_file_path" in meta:
            ply_file_path = data_dir / meta["ply_file_path"]
        else:
            ply_file_path = None

        if ply_file_path is not None:
            sparse_points = _load_3D_points(
                ply_file_path, transform_matrix, scale_factor
            )
            if sparse_points is not None:
                print(
                    f"Loaded {len(sparse_points['points3D_xyz'])} points from sparse_pc"
                )

    # ------------------------
    # Load images
    # ------------------------

    images = [imageio.imread(str(x)) for i, x in enumerate(image_filenames)]
    images = np.stack(images, axis=0)

    images_dino = [imageio.imread(str(x)) for i, x in enumerate(image_dino_filenames)]
    images_dino = np.stack(images_dino, axis=0)

    return (
        cameras,
        images,
        images_dino,
        indices,
        image_filenames,
        image_dino_filenames,
        sparse_points,
    )


class SubjectLoader(torch.utils.data.Dataset):
    """Single subject data loader for training and evaluation."""

    SPLITS = ["train", "test"]

    OPENGL_CAMERA = False

    @staticmethod
    def load_cameras(config: BaseConfig) -> Cameras:

        return _load_nerfstudio(
            config.data_dir,
            split="train",
            factor=config.factor,
            eval_mode=config.eval_mode,
            eval_interval=config.eval_interval,
            cameras_only=True,
            auto_scale_poses=config.auto_scale_poses,
        )

    def undistort_idx(
        self,
        idx: int,
        images: np.ndarray,
        only_parameters: bool = False,
        cached_image=None,
        override_camera: bool = True,
    ) -> Dict[str, torch.Tensor]:
        assert cached_image is None or only_parameters

        camera = self.cameras[idx].reshape(())
        K = camera.get_intrinsics_matrices().numpy()
        image = images[idx]
        if camera.distortion_params is None:
            raise ValueError("This shouldn't be called if there is no distortion!")
        distortion_params = camera.distortion_params.numpy()
        image = images[idx]

        K, undistorted_image, _ = undistort_image(
            camera, distortion_params, {}, image, K, only_parameters
        )
        if only_parameters:
            undistorted_image = cached_image

        if override_camera:
            self.cameras.fx[idx] = float(K[0, 0])
            self.cameras.fy[idx] = float(K[1, 1])
            self.cameras.cx[idx] = float(K[0, 2])
            self.cameras.cy[idx] = float(K[1, 2])
            self.cameras.width[idx] = undistorted_image.shape[1]
            self.cameras.height[idx] = undistorted_image.shape[0]
        return undistorted_image

    def undistort_images(self, filepaths, filepaths_dino, factor, factor_dino):

        height_dino, width_dino = None, None
        # Start with DINO
        if factor != factor_dino:
            scale_factor = float(factor) / float(factor_dino)
            self.cameras.rescale_output_resolution(scale_factor)
            filepaths_undistort_dino = []
            for idx in tqdm(range(len(self.cameras))):
                if self.cameras[idx].distortion_params is not None:
                    cache_folder = os.path.join(
                        os.path.dirname(str(filepaths_dino[idx])),
                        "_undistort",
                    )
                    cache_file = os.path.join(
                        cache_folder, os.path.basename(filepaths_dino[idx])
                    )
                    filepaths_undistort_dino.append(cache_file)
                    os.makedirs(cache_folder, exist_ok=True)

                    already_exists = os.path.exists(cache_file)

                    undistorted_image = None
                    if already_exists:
                        undistorted_image = imageio.imread(cache_file)

                    undistorted_image = self.undistort_idx(
                        idx,
                        self.images_dino,
                        only_parameters=already_exists,
                        cached_image=undistorted_image,
                        override_camera=False,
                    )

                    if not already_exists:
                        imageio.imwrite(
                            cache_file,
                            undistorted_image,
                        )

                    assert undistorted_image.dtype == np.uint8
            width_dino = self.cameras.width[0, 0].item()
            height_dino = self.cameras.height[0, 0].item()
            self.cameras.rescale_output_resolution(1.0 / scale_factor)

        # Then normal with override
        undistorted_images = []
        filepaths_undistort = []  # overwrite filepaths with deformed image paths
        for idx in tqdm(range(len(self.cameras))):
            if self.cameras[idx].distortion_params is not None:
                cache_folder = os.path.join(
                    os.path.dirname(str(filepaths[idx])),
                    "_undistort",
                )
                cache_file = os.path.join(
                    cache_folder, os.path.basename(filepaths[idx])
                )
                filepaths_undistort.append(cache_file)
                os.makedirs(cache_folder, exist_ok=True)

                already_exists = os.path.exists(cache_file)

                undistorted_image = None
                if already_exists:
                    undistorted_image = imageio.imread(cache_file)

                undistorted_image = self.undistort_idx(
                    idx,
                    self.images,
                    only_parameters=already_exists,
                    cached_image=undistorted_image,
                    override_camera=True,
                )

                if not already_exists:
                    imageio.imwrite(
                        cache_file,
                        undistorted_image,
                    )

                assert undistorted_image.dtype == np.uint8

            undistorted_images.append(
                torch.from_numpy(undistorted_image).to(torch.uint8).to(self.device)
            )

        if factor == factor_dino:
            filepaths_undistort_dino = filepaths_undistort

        return (
            undistorted_images,
            filepaths_undistort,
            filepaths_undistort_dino,
            height_dino,
            width_dino,
        )

    # TODO: switch initialization of this to config
    def __init__(
        self,
        root_fp: str,
        split: str,
        color_bkgd_aug: str = "white",
        num_rays: int = None,
        near: float = None,
        far: float = None,
        batch_over_images: bool = True,
        factor: int = 1,
        factor_dino: int = 2,
        device: str = "cpu",
        dino_model: str = "dino_vitb8",
        pca_dim: int = 16,
        n_images: int = -1,
        n_images_test: int = 10,
        eval_mode: str = "interval",
        eval_interval: int = 8,
        auto_scale_poses: bool = True,
    ):
        super().__init__()
        assert split in self.SPLITS, "%s" % split
        assert color_bkgd_aug in ["white", "black", "random"]
        self.split = split
        self.num_rays = num_rays
        self.near = near
        self.far = far
        self.training = (num_rays is not None) and (split in ["train", "trainval"])
        self.color_bkgd_aug = color_bkgd_aug
        self.batch_over_images = batch_over_images
        self.device = device

        # Add environment variable for path resolution here
        root_fp = os.path.expandvars(root_fp)
        (
            self.cameras,
            self.images,
            self.images_dino,
            indices,
            filepaths,
            filepaths_dino,
            sparse_pc,
        ) = _load_nerfstudio(
            root_fp,
            split=split,
            factor=factor,
            factor_dino=factor_dino,
            eval_mode=eval_mode,
            eval_interval=eval_interval,
            auto_scale_poses=auto_scale_poses,
        )
        self.seed_points = (
            (sparse_pc["points3D_xyz"], sparse_pc["points3D_rgb"])
            if sparse_pc is not None
            else None
        )

        # If we want to rasterize, then we need to undistort the reference
        # images and adjust the corresponding camera parameters
        (
            undistorted_images,
            filepaths_undistort,
            filepaths_undistort_dino,
            self.HEIGHT_DINO,
            self.WIDTH_DINO,
        ) = self.undistort_images(filepaths, filepaths_dino, factor, factor_dino)
        self.images = torch.stack(undistorted_images, dim=0)
        filepaths = filepaths_undistort
        filepaths_dino = filepaths_undistort_dino

        self.cameras = self.cameras.to(device)

        # TODO: this is a little bit diy
        # WARNING: this will work properly only for ray-bundles! (size might differ after undistortion)
        self.WIDTH = self.cameras.width[0, 0].item()
        self.HEIGHT = self.cameras.height[0, 0].item()
        if self.HEIGHT_DINO is None or self.WIDTH_DINO is None:
            self.HEIGHT_DINO, self.WIDTH_DINO = self.HEIGHT, self.WIDTH

        self.g = torch.Generator(device=device)
        self.g.manual_seed(42)

        filepaths = [str(path) for path in filepaths]
        filepaths_dino = [str(path) for path in filepaths_dino]

        (
            self.features,
            self.feat_coord_remap,
            self.feat_pad,
            self.feat_resize,
            self.render_pca,
        ) = extract_features(
            model_type=dino_model,
            root_fp=root_fp,
            filepaths=filepaths_dino,
            split=split,
            suffix="_undistort",
            input_shape=(self.HEIGHT_DINO, self.WIDTH_DINO),
            pca_dim=pca_dim,
            vis_pca=True,
            factor=factor_dino,
        )

    def __len__(self):
        return len(self.cameras)

    @torch.no_grad()
    def __getitem__(self, index):
        return self.fetch_rasterization_data(index)

    def fetch_rasterization_data(self, index):
        resized_feat = self.feat_resize(self.features[index].unsqueeze(0))
        # TODO: This is weird: fix that!
        padded_feat = self.feat_pad(resized_feat.permute(0, 2, 1, 3))
        padded_feat = padded_feat.permute(0, 2, 1, 3).squeeze(0)
        if self.HEIGHT_DINO != self.HEIGHT or self.WIDTH_DINO != self.WIDTH:
            import torchvision.transforms.functional as TF

            padded_feat = TF.resize(
                padded_feat.permute(2, 0, 1), (self.HEIGHT, self.WIDTH), antialias=None
            ).permute(1, 2, 0)

        return {
            "camera": self.cameras[index],
            "image": (self.images[index].float() / 255.0),
            "feat": padded_feat,
        }

    def update_num_rays(self, num_rays):
        self.num_rays = num_rays
