import torch

import MinkowskiEngine as ME
from MinkowskiEngine import SparseTensor, MinkowskiInterpolationFunction


# This function is borrowed from https://github.com/96lives/gca
def get_shifts(
    padding, data_dim, pad_type="hypercubic", include_batch=False, include_center=True
):
    """
    Arguments:
        padding: number of padding to add to shifts
        data_dim: dimension of data
    Returns
        shifts:
            Tensor of shape ((2 * padding + 1) ** data_dim) x  data_dim
            Each row of shifts represent nearby coordinates

    Ex)
        >>> get_shifts(1, 2)
        torch.tensor([
            [-1, -1], [-1, 0], [-1, 1],
            [0, -1], [0, 0], [0, 1],
            [1, -1], [1, 0], [1, -1]
        ])
    """

    shifts = []
    if pad_type == "hypercubic":
        for i in range(-padding, padding + 1):
            for j in range(-padding, padding + 1):
                if data_dim == 2:
                    if i == j and i == 0 and not include_center:
                        continue
                    shifts.append([i, j])
                    continue
                for k in range(-padding, padding + 1):
                    if i == j and j == k and i == 0 and not include_center:
                        continue
                    shifts.append([i, j, k])
    elif pad_type == "hypercross":
        for x in range(padding + 1):
            for y in range(padding + 1 - x):
                if data_dim == 2:
                    if x == y and x == 0 and not include_center:
                        continue
                    shifts.append([x, y])
                    shifts.append([-x, y])
                    shifts.append([x, -y])
                    shifts.append([-x, -y])
                    continue
                for z in range(padding + 1 - x - y):
                    if x == y and y == z and x == 0 and not include_center:
                        continue
                    shifts.append([x, y, z])
                    shifts.append([-x, y, z])
                    shifts.append([x, -y, z])
                    shifts.append([x, y, -z])
                    shifts.append([-x, -y, z])
                    shifts.append([-x, y, -z])
                    shifts.append([x, -y, -z])
                    shifts.append([-x, -y, -z])
    else:
        raise ValueError("pad_type {} not allowed".format(pad_type))

    shifts = torch.unique(torch.Tensor(shifts), dim=0)
    if include_batch:
        # shifts = ME.utils.batched_coordinates(shifts)
        shifts = torch.cat([torch.zeros(shifts.shape[0], 1), shifts], dim=1)
    return shifts


def _extract_patched_coords(coord: torch.Tensor, shifts: torch.Tensor):
    repeated_shifts = shifts.unsqueeze(0).repeat((coord.shape[0], 1, 1))
    repeated_coord = coord.unsqueeze(1).repeat((1, shifts.shape[0], 1))
    return repeated_coord + repeated_shifts


@staticmethod
def _extract_feats_from_coords(
    coord: torch.Tensor,
    feat_coord: torch.Tensor,
    feat: torch.Tensor,
    device: torch.device,
):
    init_shape = coord.shape
    if len(init_shape) == 3:
        flat_coord = coord.reshape(-1, 3)
    else:
        flat_coord = coord

    sparse_feat = SparseTensor(
        features=feat,
        coordinates=ME.utils.batched_coordinates([feat_coord]),
        device=device,
    )
    sparse_occ = SparseTensor(
        features=torch.ones((feat.shape[0], 1), device=device, dtype=torch.float),
        coordinates=ME.utils.batched_coordinates([feat_coord]),
        device=device,
    )

    # Use Minkowski engine to fetch the corresponding features
    features = MinkowskiInterpolationFunction.apply(
        sparse_feat.F,
        ME.utils.batched_coordinates([flat_coord], device=device).float(),
        sparse_feat.coordinate_map_key,
        sparse_feat.coordinate_manager,
    )[0]

    occupancy = MinkowskiInterpolationFunction.apply(
        sparse_occ.F,
        ME.utils.batched_coordinates([flat_coord], device=device).float(),
        sparse_occ.coordinate_map_key,
        sparse_occ.coordinate_manager,
    )[0]

    if len(init_shape) == 3:
        features = features.reshape((init_shape[0], init_shape[1], features.shape[-1]))
        occupancy = occupancy.reshape(
            (init_shape[0], init_shape[1], occupancy.shape[-1])
        )

    return features, occupancy


def extract_patches(
    device, coord: torch.Tensor, feat: torch.Tensor, patch_size: int = 5
):
    # First, get shifts
    padding = patch_size // 2
    shifts = (
        get_shifts(
            padding=padding,
            data_dim=3,
            include_batch=False,
        )
        .to(device)
        .int()
    )

    # Then, add them to reference and target
    patches = _extract_patched_coords(coord, shifts=shifts)
    patch_features, patch_occupancy = _extract_feats_from_coords(
        patches,
        coord,
        feat,
        device,
    )

    return patches, patch_occupancy, patch_features
