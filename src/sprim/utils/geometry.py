from typing import Tuple
import math

import numpy as np
import torch


def focal2fov(focal, pixels):
    return 2 * math.atan(pixels / (2 * focal))


def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))


# https://github.com/nerfstudio-project/nerfstudio/blob/main/nerfstudio/data/datamanagers/full_images_datamanager.py
import cv2
from nerfstudio.cameras.cameras import Cameras, CameraType
from typing import (
    Optional,
    Tuple,
)


def undistort_image(
    camera: Cameras,
    distortion_params: np.ndarray,
    data: dict,
    image: np.ndarray,
    K: np.ndarray,
    only_parameters: bool = False,
) -> Tuple[np.ndarray, np.ndarray, Optional[torch.Tensor]]:
    mask = None
    if camera.camera_type.item() == CameraType.PERSPECTIVE.value:
        distortion_params = np.array(
            [
                distortion_params[0],
                distortion_params[1],
                distortion_params[4],
                distortion_params[5],
                distortion_params[2],
                distortion_params[3],
                0,
                0,
            ]
        )
        if np.any(distortion_params):
            newK, roi = cv2.getOptimalNewCameraMatrix(
                K, distortion_params, (image.shape[1], image.shape[0]), 0
            )
            if not only_parameters:
                image = cv2.undistort(image, K, distortion_params, None, newK)  # type: ignore
            else:
                image = None
        else:
            newK = K
            roi = 0, 0, image.shape[1], image.shape[0]
        # crop the image and update the intrinsics accordingly
        if not only_parameters:
            x, y, w, h = roi
            image = image[y : y + h, x : x + w]
            if "depth_image" in data:
                data["depth_image"] = data["depth_image"][y : y + h, x : x + w]
            if "mask" in data:
                mask = data["mask"].numpy()
                mask = mask.astype(np.uint8) * 255
                if np.any(distortion_params):
                    mask = cv2.undistort(mask, K, distortion_params, None, newK)  # type: ignore
                mask = mask[y : y + h, x : x + w]
                mask = torch.from_numpy(mask).bool()
                if len(mask.shape) == 2:
                    mask = mask[:, :, None]
        K = newK

    else:
        raise NotImplementedError("Only perspective cameras are supported")

    return K, image, mask
