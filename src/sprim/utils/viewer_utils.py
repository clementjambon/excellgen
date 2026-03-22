from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


# -----------------------------
# LISTS
# -----------------------------

CUBE_VERTICES_LIST = [
    [0.0, 0.0, 1.0],
    [1.0, 0.0, 1.0],
    [1.0, 1.0, 1.0],
    [0.0, 1.0, 1.0],
    [0.0, 0.0, 0.0],
    [1.0, 0.0, 0.0],
    [1.0, 1.0, 0.0],
    [0.0, 1.0, 0.0],
]

CUBE_EDGES_LIST = [
    [0, 1],  # 0
    [1, 2],  # 1
    [2, 3],  # 2
    [3, 0],  # 3
    [4, 5],  # 4
    [5, 6],  # 5
    [6, 7],  # 6
    [7, 4],  # 7
    [0, 4],  # 8
    [1, 5],  # 9
    [2, 6],  # 10
    [3, 7],  # 11
]

CUBE_TRIANGLES_LIST = [
    [0, 1, 2],
    [2, 3, 0],
    [6, 5, 4],
    [4, 7, 6],
    [5, 2, 1],
    [2, 5, 6],
    [0, 3, 4],
    [4, 3, 7],
    [2, 6, 3],
    [3, 6, 7],
    [0, 4, 1],
    [1, 4, 5],
]

# -----------------------------
# NUMPY
# -----------------------------

CUBE_VERTICES_NP = np.array(CUBE_VERTICES_LIST)

CUBE_EDGES_NP = np.array(CUBE_EDGES_LIST)

# -----------------------------
# TORCH
# -----------------------------

# N = 8
CUBE_VERTICES = torch.tensor(
    CUBE_VERTICES_LIST,
    device="cuda",
)

# N = 2 * 6 = 12
CUBE_TRIANGLES = torch.tensor(
    CUBE_TRIANGLES_LIST,
    dtype=torch.int,
    device="cuda",
)

CUBE_FACE_NEIGHBOR_OFFSETS = torch.tensor(
    [
        [0, 0, 1],
        [0, 0, 1],
        [0, 0, -1],
        [0, 0, -1],
        [1, 0, 0],
        [1, 0, 0],
        [-1, 0, 0],
        [-1, 0, 0],
        [0, 1, 0],
        [0, 1, 0],
        [0, -1, 0],
        [0, -1, 0],
    ],
    device="cuda",
)

CUBE_EDGES = torch.tensor(
    CUBE_EDGES_LIST,
    dtype=torch.int,
    device="cuda",
)

CUBE_CHECKER_DISPLACEMENTS = [
    (0, [0.0, 1.0, 0.0]),
    (4, [0.0, 1.0, 0.0]),
    (4, [0.0, 0.0, 1.0]),
    (5, [0.0, 0.0, 1.0]),
    (6, [0.0, 0.0, 1.0]),
    (7, [0.0, 0.0, 1.0]),
]

PLANE_CHECKER_DISPLACEMENTS_X = [(8, [0.0, 1.0, 0.0]), (7, [0.0, 0.0, 1.0])]
PLANE_CHECKER_DISPLACEMENTS_Y = [(4, [0.0, 0.0, 1.0]), (8, [1.0, 0.0, 0.0])]
PLANE_CHECKER_DISPLACEMENTS_Z = [(4, [0.0, 1.0, 0.0]), (7, [1.0, 0.0, 0.0])]

PLANE_CHECKER_DISPLACEMENTS = [
    PLANE_CHECKER_DISPLACEMENTS_X,
    PLANE_CHECKER_DISPLACEMENTS_Y,
    PLANE_CHECKER_DISPLACEMENTS_Z,
]


def create_checker_bbox(
    name: str, reps: int, bbox_min: np.ndarray, bbox_max: np.ndarray
):
    assert reps >= 2

    import polyscope as ps

    tot_vertices = []
    for edge, dir in CUBE_CHECKER_DISPLACEMENTS:
        disp = np.array(dir).astype(np.float32) / (reps - 1)
        for i in range(reps):
            tot_vertices += [
                i * disp + np.array(CUBE_VERTICES_LIST[CUBE_EDGES_LIST[edge][0]]),
                i * disp + np.array(CUBE_VERTICES_LIST[CUBE_EDGES_LIST[edge][1]]),
            ]

    tot_edges = np.arange(reps * 6 * 2).reshape(-1, 2)
    tot_vertices = np.array(tot_vertices)

    tot_vertices = (bbox_max - bbox_min) * tot_vertices + bbox_min

    return ps.register_curve_network(name, tot_vertices, tot_edges, radius=0.002)


def create_checker_plane(
    name: str, reps: int, axis: int, bbox_min: np.ndarray, bbox_max: np.ndarray
):
    assert reps >= 2

    import polyscope as ps

    axis_vec = [0.0] * 3
    axis_vec[axis] = 1.0
    axis_vec = np.array(axis_vec)

    tot_vertices = []
    for edge, dir in PLANE_CHECKER_DISPLACEMENTS[axis]:
        disp = np.array(dir).astype(np.float32) / (reps - 1)
        for i in range(reps):
            tot_vertices += [
                i * disp
                + np.array(CUBE_VERTICES_LIST[CUBE_EDGES_LIST[edge][0]])
                + 0.5 * axis_vec,
                i * disp
                + np.array(CUBE_VERTICES_LIST[CUBE_EDGES_LIST[edge][1]])
                + 0.5 * axis_vec,
            ]

    tot_edges = np.arange(reps * 2 * 2).reshape(-1, 2)
    tot_vertices = np.array(tot_vertices)

    tot_vertices = (bbox_max - bbox_min) * tot_vertices + bbox_min

    return ps.register_curve_network(name, tot_vertices, tot_edges, radius=0.002)


class RenderPCA(nn.Module):

    def __init__(self, mean: torch.Tensor, components: torch.Tensor):
        super().__init__()
        self.mean = nn.Parameter(mean, requires_grad=False)
        self.components = nn.Parameter(components, requires_grad=False)

    def transform(self, X) -> torch.Tensor:
        return (X - self.mean) @ self.components.T

    def render(self, X) -> torch.Tensor:
        # We apply sigmoid for rendering to bound everyone in [0, 1]
        return torch.sigmoid(self.transform(X))

    @staticmethod
    def default(feature_dim: int) -> RenderPCA:
        mean = torch.zeros(feature_dim).cuda()
        components = torch.zeros(3, feature_dim).cuda()
        components[0, 0] = components[1, 1] = components[2, 2] = 1.0
        return RenderPCA(mean=mean, components=components)
