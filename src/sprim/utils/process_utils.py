import torch
import torch.nn.functional as F
import numpy as np
import einops
from e3nn import o3


def filter_bbox(
    coord: torch.Tensor,
    bbox_min: torch.Tensor,
    bbox_max: torch.Tensor,
    return_indices: bool = False,
):

    coord_idx = torch.argwhere(
        (torch.all(coord >= bbox_min, dim=1)) & (torch.all(coord < bbox_max, dim=1))
    ).squeeze(1)

    if return_indices:
        return coord[coord_idx], coord_idx
    return coord[coord_idx]


# https://github.com/cupy/cupy/issues/1918#issuecomment-676393859
def repeat_arbitrary(
    array: torch.Tensor, repeats: torch.Tensor, return_indices: bool = False
):
    all_stops = torch.cumsum(repeats, dim=0)
    parents = torch.zeros(all_stops[-1].item(), dtype=torch.long, device=array.device)
    stops, stop_counts = torch.unique(all_stops[:-1], return_counts=True)
    stops = stops.long()
    parents[stops] = stop_counts
    torch.cumsum(parents, dim=0, out=parents)
    if return_indices:
        return array[parents], parents
    return array[parents]


def world_to_voxel(
    x: torch.Tensor,
    bbox_min: torch.Tensor,
    bbox_max: torch.Tensor,
    voxel_res: int,
):
    return (x - bbox_min) / (bbox_max - bbox_min) * voxel_res


def voxel_to_world(
    x: torch.Tensor,
    bbox_min: torch.Tensor,
    bbox_max: torch.Tensor,
    voxel_res: int,
):
    return (bbox_max - bbox_min) * (1.0 / float(voxel_res)) * x + bbox_min


def coord_bbox_filter(coord: torch.Tensor, res: int, return_indices: bool = False):
    coord_idx = torch.argwhere(
        (torch.min(coord, dim=-1)[0] >= 0) & (torch.max(coord, dim=-1)[0] < res)
    ).squeeze(1)
    if return_indices:
        return coord[coord_idx], coord_idx
    return coord[coord_idx]


def flatten_coord(res: int | None, coords: torch.Tensor):
    if res is None:
        assert coords.shape[-1] == 3
        # Compute bounds manually and offsets
        min2, max2 = torch.min(coords[..., 2]), torch.max(coords[..., 2])
        min1, max1 = torch.min(coords[..., 1]), torch.max(coords[..., 1])
        min0 = torch.min(coords[..., 0])
        res2 = max2 - min2 + 1
        res1 = max1 - min1 + 1

        # NOTE: turn everyone to long for safety checks
        return (
            res1 * res2 * (coords[..., 0] - min0).long()
            + res2 * (coords[..., 1] - min1).long()
            + (coords[..., 2] - min2).long()
        )
    else:
        if coords.shape[-1] == 2:
            return res * coords[..., 0] + coords[..., 1]
        elif coords.shape[-1] == 3:
            return res * res * coords[..., 0] + res * coords[..., 1] + coords[..., 2]


def apply_transform(x: torch.Tensor, transform: torch.Tensor | np.ndarray | None):
    """
    Apply 4x4 transform matrix `transform` to `x`
    """
    x_shape = x.shape
    assert len(x_shape) == 1 or len(x_shape) == 2

    if transform is None:
        return x

    if isinstance(transform, torch.Tensor):
        actual_transform = transform
    else:
        actual_transform = torch.tensor(transform).to(x)

    transformed_pos = torch.cat(
        [
            x if len(x_shape) == 2 else x.unsqueeze(0),
            torch.ones(
                ((x_shape[0] if len(x_shape) == 2 else 1), 1),
                device=x.device,
            ),
        ],
        dim=-1,
    )

    transformed_pos = torch.matmul(actual_transform, transformed_pos.T).T
    transformed_pos = transformed_pos[:, :3] / transformed_pos[:, 3][:, None]

    return transformed_pos if x_shape == 2 else transformed_pos.squeeze(0)


# https://pytorch3d.readthedocs.io/en/latest/_modules/pytorch3d/transforms/rotation_conversions.html#matrix_to_quaternion
def rotmat_to_quat(matrix: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as rotation matrices to quaternions.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).

    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    """
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")

    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(
        matrix.reshape(batch_dim + (9,)), dim=-1
    )

    q_abs = _sqrt_positive_part(
        torch.stack(
            [
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ],
            dim=-1,
        )
    )

    # we produce the desired quaternion multiplied by each of r, i, j, k
    quat_by_rijk = torch.stack(
        [
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
        ],
        dim=-2,
    )

    # We floor here at 0.1 but the exact level is not important; if q_abs is small,
    # the candidate won't be picked.
    flr = torch.tensor(0.1).to(dtype=q_abs.dtype, device=q_abs.device)
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(flr))

    # if not for numerical problems, quat_candidates[i] should be same (up to a sign),
    # forall i; we pick the best-conditioned one (with the largest denominator)
    out = quat_candidates[
        F.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5, :
    ].reshape(batch_dim + (4,))
    return standardize_quaternion(out)


def standardize_quaternion(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert a unit quaternion to a standard form: one in which the real
    part is non negative.

    Args:
        quaternions: Quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Standardized quaternions as tensor of shape (..., 4).
    """
    return torch.where(quaternions[..., 0:1] < 0, -quaternions, quaternions)


def _sqrt_positive_part(x: torch.Tensor) -> torch.Tensor:
    """
    Returns torch.sqrt(torch.max(0, x))
    but with a zero subgradient where x is 0.
    """
    ret = torch.zeros_like(x)
    positive_mask = x > 0
    ret[positive_mask] = torch.sqrt(x[positive_mask])
    return ret


def isin_coord(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    _, invmap, counts = torch.unique(
        torch.cat([x, ref], dim=0), dim=0, return_inverse=True, return_counts=True
    )

    return counts[invmap[: len(x)]] >= 2


# Adapted from https://github.com/graphdeco-inria/gaussian-splatting/issues/176#issuecomment-2127778441
def transform_shs(shs_feat, rotation_matrix):

    ## rotate shs
    P = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])  # switch axes: yzx -> xyz
    permuted_rotation_matrix = P @ rotation_matrix.cpu().numpy()
    rot_angles = o3._rotation.matrix_to_angles(
        torch.from_numpy(permuted_rotation_matrix)
    )

    # Construction coefficient
    D_0 = o3.wigner_D(0, rot_angles[0], rot_angles[1], rot_angles[2]).float().cuda()
    D_1 = o3.wigner_D(1, rot_angles[0], rot_angles[1], rot_angles[2]).float().cuda()
    D_2 = o3.wigner_D(2, rot_angles[0], rot_angles[1], rot_angles[2]).float().cuda()
    D_3 = o3.wigner_D(3, rot_angles[0], rot_angles[1], rot_angles[2]).float().cuda()

    # rotation of the shs features
    two_degree_shs = shs_feat[:, 0:3]
    two_degree_shs = einops.rearrange(two_degree_shs, "n shs_num rgb -> n rgb shs_num")
    two_degree_shs = torch.einsum(
        "... i j, ... j -> ... i",
        D_1,
        two_degree_shs,
    )
    two_degree_shs = einops.rearrange(two_degree_shs, "n rgb shs_num -> n shs_num rgb")
    shs_feat[:, 0:3] = two_degree_shs

    three_degree_shs = shs_feat[:, 3:8]
    three_degree_shs = einops.rearrange(
        three_degree_shs, "n shs_num rgb -> n rgb shs_num"
    )
    three_degree_shs = torch.einsum(
        "... i j, ... j -> ... i",
        D_2,
        three_degree_shs,
    )
    three_degree_shs = einops.rearrange(
        three_degree_shs, "n rgb shs_num -> n shs_num rgb"
    )
    shs_feat[:, 3:8] = three_degree_shs

    four_degree_shs = shs_feat[:, 8:15]
    four_degree_shs = einops.rearrange(
        four_degree_shs, "n shs_num rgb -> n rgb shs_num"
    )
    four_degree_shs = torch.einsum(
        "... i j, ... j -> ... i",
        D_3,
        four_degree_shs,
    )
    four_degree_shs = einops.rearrange(
        four_degree_shs, "n rgb shs_num -> n shs_num rgb"
    )
    shs_feat[:, 8:15] = four_degree_shs

    return shs_feat
