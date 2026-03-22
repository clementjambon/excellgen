"""
Copyright (c) 2022 Ruilong Li, UC Berkeley.
"""

from typing import List, Tuple
import json
import os

import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from sklearn.decomposition import PCA

from sprim.utils.dino_extractor import extract_and_save_features
from sprim.utils.viewer_utils import RenderPCA

# Maximum number of images used to compute PCA (avoid OOM)
MAX_PCA_IMAGES = 200
PCA_TRANSFORM_BATCHES = 16


@torch.no_grad()
def _load_renderings(root_fp: str, split: str):
    """Load images from disk."""
    if not root_fp.startswith("/"):
        # allow relative path. e.g., "./data/nerf_synthetic/"
        root_fp = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..",
            "..",
            root_fp,
        )

    data_dir = root_fp
    transform_path = os.path.join(data_dir, "transforms_{}.json".format(split))
    skip = 1
    if not os.path.exists(transform_path):
        transform_path = os.path.join(data_dir, "transforms.json")
        skip = 8 if split == "test" else 1
    with open(transform_path, "r") as fp:
        meta = json.load(fp)
    images = []
    camtoworlds = []
    filepaths = []

    for i in range(0, len(meta["frames"]), skip):
        frame = meta["frames"][i]
        fname = os.path.join(data_dir, frame["file_path"] + ".png")
        rgba = imageio.imread(fname)
        camtoworlds.append(frame["transform_matrix"])
        images.append(rgba)
        filepaths.append(fname)

    images = np.stack(images, axis=0)
    camtoworlds = np.stack(camtoworlds, axis=0)

    h, w = images.shape[1:3]
    camera_angle_x = float(meta["camera_angle_x"])
    focal = 0.5 * w / np.tan(0.5 * camera_angle_x)

    return images, camtoworlds, focal, filepaths


@torch.no_grad()
def extract_features(
    model_type: str,
    root_fp: str,
    filepaths: List[str],
    split: str,
    suffix: str,
    factor: int,
    input_shape: Tuple[int, int],
    pca_dim: int,
    vis_pca: bool = True,
):
    assert model_type.startswith("dino")
    if model_type.startswith("dinov2"):
        stride = 7
    else:
        stride = 8

    input_files = [path.split(os.sep)[-1].split(".")[0] for path in filepaths]

    output_paths = [
        os.path.join(root_fp, model_type, split + suffix + f"_{factor}", f"{file}.npy")
        for file in input_files
    ]

    extraction_shape = (
        input_shape[0] // stride * stride,
        input_shape[1] // stride * stride,
    )

    # if not all(os.path.exists(path) for path in output_paths):
    pad, resize, coord_remap = extract_and_save_features(
        filepaths,
        output_paths,
        input_shape=input_shape,
        extraction_shape=extraction_shape,
        stride=stride,
        model_type=model_type,
    )

    if vis_pca:
        os.makedirs(
            os.path.join(root_fp, model_type, split + suffix + f"_{factor}" + "_pca"),
            exist_ok=True,
        )
        for i, path in tqdm(enumerate(output_paths)):
            renders_path = os.path.join(
                root_fp,
                model_type,
                split + suffix + f"_{factor}" + "_pca",
                input_files[i] + ".png",
            )
            if os.path.exists(renders_path):
                continue
            features = np.load(path)
            pca = PCA(n_components=3)
            feature_maps_pca = pca.fit_transform(
                features.reshape(-1, features.shape[-1])
            )
            pca_features_min = feature_maps_pca.min(axis=(0, 1))
            pca_features_max = feature_maps_pca.max(axis=(0, 1))
            pca_features = (feature_maps_pca - pca_features_min) / (
                pca_features_max - pca_features_min
            )

            pca_features = pca_features.reshape(features.shape[0], features.shape[1], 3)

            imageio.imwrite(renders_path, (pca_features * 255).astype(np.uint8))

    # Convert to Tensor
    dino_features = []
    for path in output_paths:
        dino_features.append(torch.tensor(np.load(path)).cuda())
    dino_features = torch.stack(dino_features)

    # --------------------------------
    # Perform PCA on all features
    # --------------------------------
    pca_path = os.path.join(
        root_fp, model_type, split + suffix + f"_{factor}" + f"_pca_{pca_dim}.npy"
    )
    if os.path.exists(pca_path):
        dino_features = torch.tensor(np.load(pca_path)).cuda()
    else:
        init_shape = dino_features.shape
        dino_dim = dino_features.shape[-1]

        # Cap the number of images used to compute the PCA to avoid runing out of memory
        perm = torch.randperm(dino_features.shape[0])
        n_pca_images = min(dino_features.shape[0], MAX_PCA_IMAGES)
        idx = perm[:n_pca_images]
        dino_for_pca = (
            dino_features[idx].reshape(-1, dino_features.shape[-1]).cpu().numpy()
        )

        print(
            f"PCA: {dino_dim} -> {pca_dim} (with {n_pca_images}/{dino_features.shape[0]} images)"
        )
        pca = PCA(n_components=pca_dim)
        pca = pca.fit(dino_for_pca)

        feature_maps_pca = []
        for i_batch in tqdm(range(0, dino_features.shape[0], PCA_TRANSFORM_BATCHES)):
            feature_maps_pca.append(
                torch.tensor(
                    pca.transform(
                        dino_features[i_batch : i_batch + PCA_TRANSFORM_BATCHES]
                        .reshape(-1, dino_features.shape[-1])
                        .cpu()
                        .numpy()
                    )
                )
            )
        feature_maps_pca = torch.cat(feature_maps_pca)
        dino_features = feature_maps_pca.reshape(
            init_shape[0], init_shape[1], init_shape[2], pca_dim
        ).cuda()
        np.save(
            pca_path,
            dino_features.cpu().numpy(),
        )

    # --------------------------------
    # Perform PCA to RGB
    # --------------------------------
    rgb_pca_path = os.path.join(
        root_fp, model_type, split + suffix + f"_{factor}" + f"_pca_{pca_dim}_rgb.npz"
    )
    if os.path.exists(rgb_pca_path):
        data = np.load(rgb_pca_path)
        render_pca = RenderPCA(
            mean=torch.tensor(data["mean"]).cuda(),
            components=torch.tensor(data["components"]).cuda(),
        )
    else:
        init_shape = dino_features.shape
        dino_dim = dino_features.shape[-1]

        # Cap the number of images used to compute the PCA to avoid running out of memory
        perm = torch.randperm(dino_features.shape[0])
        n_pca_images = min(dino_features.shape[0], MAX_PCA_IMAGES)
        idx = perm[:n_pca_images]
        dino_for_pca = (
            dino_features[idx].reshape(-1, dino_features.shape[-1]).cpu().numpy()
        )

        print(
            f"PCA: {dino_dim} -> {3} (with {n_pca_images}/{dino_features.shape[0]} images)"
        )
        pca = PCA(n_components=3)
        pca = pca.fit(dino_for_pca)

        np.savez(rgb_pca_path, mean=pca.mean_, components=pca.components_)

        render_pca = RenderPCA(
            mean=torch.tensor(pca.mean_).cuda(),
            components=torch.tensor(pca.components_).cuda(),
        )

    return dino_features, coord_remap, pad, resize, render_pca
