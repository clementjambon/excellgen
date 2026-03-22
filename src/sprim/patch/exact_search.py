# This is adapted from http://weiyuli.xyz/Sin3DGen/
# Check the discussion in the manuscript for more details on the differences

from dataclasses import dataclass
import functools
from tqdm import tqdm
import torch
import torch.nn.functional as F

import polyscope.imgui as psim

from torch_scatter import scatter

from sprim.utils.process_utils import coord_bbox_filter, isin_coord
from sprim.patch.extract_patches import extract_patches


# Just lay everything flat (which is already the case in our case)
def sparse_distance(
    X: torch.Tensor, Y: torch.Tensor, w: float, mode: int = "cosine_similarity"
):
    assert w >= 0.0 and w <= 1.0

    X_occ, Y_occ = X[..., 0].reshape(*X.shape[:-2], -1), Y[..., 0].reshape(
        *Y.shape[:-2], -1
    )
    X_lat, Y_lat = X[..., 1:], Y[..., 1:]
    D = X_lat.shape[-1]

    # Renormalize both dimension split separately
    if mode == "cosine_similarity":
        X_lat[..., : D // 2], Y_lat[..., : D // 2] = F.normalize(
            X_lat[..., : D // 2], dim=-1
        ), F.normalize(Y_lat[..., : D // 2], dim=-1)
        X_lat[..., D // 2 :], Y_lat[..., D // 2 :] = F.normalize(
            X_lat[..., D // 2 :], dim=-1
        ), F.normalize(Y_lat[..., D // 2 :], dim=-1)
    else:
        assert False

    X_lat, Y_lat = X_lat.reshape(*X.shape[:-2], -1), Y_lat.reshape(*Y.shape[:-2], -1)

    intersection_counts = torch.mm(X_occ, Y_occ.T)

    if w == 0.0:
        dist = 1.0 - efficient_cdist_prod(X_occ, Y_occ) / X_occ.shape[-1]
        dist_occ = dist
        dist_lat = torch.zeros_like(dist)
    elif w == 1.0:
        dist = 1.0 - efficient_cdist_prod(X_lat, Y_lat) / 2.0 / intersection_counts
        dist_lat = dist
        dist_occ = torch.zeros_like(dist)
    else:
        dist_occ = 1.0 - efficient_cdist_prod(X_occ, Y_occ) / X_occ.shape[-1]
        dist_lat = (
            1.0 - efficient_cdist_prod(X_lat, Y_lat) / (2.0) / intersection_counts
        )
        dist = dist_occ * (1 - w) + dist_lat * w

    return dist, dist_occ, dist_lat


@torch.no_grad()
def efficient_cdist(X, Y):
    dist = (
        (X * X).sum(1)[:, None]
        + (Y * Y).sum(1)[None, :]
        - 2.0 * torch.mm(X, torch.transpose(Y, 0, 1))
    )

    return dist


@torch.no_grad()
def efficient_cdist_prod(X, Y):
    dist = torch.mm(X, torch.transpose(Y, 0, 1))

    return dist  # DO NOT use torch.sqrt


@torch.no_grad()
def get_col_mins_efficient(dist_fn, X, Y, b=1024):
    n_batches = len(Y) // b
    mins = torch.zeros(Y.shape[0], dtype=X.dtype, device=X.device)
    for i in range(n_batches):
        mins[i * b : (i + 1) * b] = dist_fn(X, Y[i * b : (i + 1) * b]).min(0)[0]
    if len(Y) % b != 0:
        mins[n_batches * b :] = dist_fn(X, Y[n_batches * b :]).min(0)[0]

    return mins


@torch.no_grad()
def get_NNs_Dists(dist_fn, X, Y, alpha=None, b=1024):
    if alpha is not None:
        normalizing_row = get_col_mins_efficient(dist_fn, X, Y, b=b)
        normalizing_row = alpha + normalizing_row[None, :]
    else:
        normalizing_row = 1

    NNs = torch.zeros(X.shape[0], dtype=torch.long, device=X.device)
    Dists = torch.zeros(X.shape[0], dtype=torch.float, device=X.device)
    Dists_occ = torch.zeros(X.shape[0], dtype=torch.float, device=X.device)
    Dists_lat = torch.zeros(X.shape[0], dtype=torch.float, device=X.device)

    n_batches = len(X) // b
    for i in range(n_batches):
        dists, dists_occ, dists_lat = dist_fn(X[i * b : (i + 1) * b], Y)

        dists /= normalizing_row
        dists_occ /= normalizing_row
        dists_lat /= normalizing_row

        dists_min, dists_min_idx = dists.min(1)
        NNs[i * b : (i + 1) * b] = dists_min_idx
        Dists[i * b : (i + 1) * b] = dists_min
        Dists_occ[i * b : (i + 1) * b] = dists_occ[
            torch.arange(len(dists)), dists_min_idx
        ]
        Dists_lat[i * b : (i + 1) * b] = dists_lat[
            torch.arange(len(dists)), dists_min_idx
        ]
    if len(X) % b != 0:
        dists, dists_occ, dists_lat = dist_fn(X[n_batches * b :], Y)

        dists /= normalizing_row
        dists_occ /= normalizing_row
        dists_lat /= normalizing_row

        dists_min, dists_min_idx = dists.min(1)
        NNs[n_batches * b :] = dists_min_idx
        Dists[n_batches * b :] = dists_min

        Dists_occ[n_batches * b :] = dists_occ[torch.arange(len(dists)), dists_min_idx]
        Dists_lat[n_batches * b :] = dists_lat[torch.arange(len(dists)), dists_min_idx]

    return NNs, Dists, Dists_occ, Dists_lat


@dataclass(kw_only=True)
class PatchParameters:
    patch_size: int = 5
    patch_iters: int = 7
    limit_geometry_dist: float = 2.0
    distance_w: float = 0.5
    # TODO!
    approximate: bool = False

    def gui(self) -> None:
        if psim.TreeNode("Patch parameters"):

            patch_half = self.patch_size // 2
            _, patch_half = psim.SliderInt(
                "patch_size",
                patch_half,
                v_min=0,
                v_max=4,
                format=f"{2 * patch_half + 1}",
            )
            self.patch_size = 2 * patch_half + 1

            _, self.patch_iters = psim.SliderInt(
                "patch_iters", self.patch_iters, v_min=1, v_max=20
            )

            _, self.limit_geometry_dist = psim.SliderFloat(
                "limit_geometry_dist",
                self.limit_geometry_dist,
                v_min=0.0,
                v_max=10.0,
            )

            _, self.distance_w = psim.SliderFloat(
                "distance_w", self.distance_w, v_min=0.0, v_max=1.0
            )

            psim.TreePop()


@torch.no_grad()
def exact_search(
    state_coord: torch.Tensor,
    state_feat: torch.Tensor,
    ref_coord: torch.Tensor,
    ref_feat: torch.Tensor,
    patch_parameters: PatchParameters,
    single_step: bool = False,
    alpha=None,
    chunk_size=4096,
    return_last_only: bool = False,
):
    num_iters = patch_parameters.patch_iters if not single_step else 1

    dist_fn = functools.partial(sparse_distance, w=patch_parameters.distance_w)

    ref_patch_coord, ref_patch_occ, ref_patch_feat = extract_patches(
        ref_coord.device,
        coord=ref_coord,
        feat=ref_feat,
        patch_size=patch_parameters.patch_size,
    )
    # SANITY
    assert ref_patch_occ.shape[-1] == 1

    init_coord = torch.clone(state_coord)
    current_coord = torch.clone(state_coord)
    current_feat = torch.clone(state_feat)

    new_coords = []
    new_feats = []
    new_ref_coords = []
    Dists = []
    pbar = tqdm(total=num_iters, desc="Starting")
    for itr in range(1, num_iters + 1):
        state_patch_coord, state_patch_occ, state_patch_feat = extract_patches(
            state_coord.device,
            coord=current_coord,
            feat=current_feat,
            patch_size=patch_parameters.patch_size,
        )

        nns, dists, dists_occ, dists_lat = get_NNs_Dists(
            dist_fn,
            torch.cat([state_patch_occ, state_patch_feat], dim=-1),
            torch.cat([ref_patch_occ, ref_patch_feat], dim=-1),
            alpha=alpha,
            b=chunk_size,
        )
        pbar.update(1)
        pbar.set_description(
            f"[iter {itr}] full: {dists.mean().item():.6f}; occ: {dists_occ.mean().item():.6f}; lat: {dists_lat.mean().item():.6f}"
        )
        Dists += [dists.mean().item()]

        # Combine patches
        # For the first iterations, average
        if itr < num_iters - 1:

            # We must return before voting!
            if not return_last_only:
                new_coords.append(current_coord)
                new_feats.append(ref_feat[nns])
                new_ref_coords.append(ref_coord[nns])

            unique_coord, unique_coord_invmap = torch.unique(
                state_patch_coord.reshape(-1, 3), dim=0, return_inverse=True
            )
            # breakpoint()
            occ_votes = scatter(ref_patch_occ[nns].view(-1), unique_coord_invmap)
            tot_count = scatter(
                torch.ones_like(ref_patch_occ[nns].view(-1)), unique_coord_invmap
            )
            feat_mean = scatter(
                ref_patch_feat[nns].reshape(-1, current_feat.shape[-1]),
                unique_coord_invmap,
                dim=0,
                reduce="mean",
            )

            # Keep only those who have been voted more than half
            # TODO: make that a probabilistic number
            selected = occ_votes > tot_count // 2
            current_coord = unique_coord[selected]
            current_feat = feat_mean[selected]

            # current_coord, valid_indices = coord_bbox_filter(
            #     selected_coord, res, return_indices=True
            # )
            # current_feat = selected_feat[valid_indices]

            if patch_parameters.limit_geometry_dist > 0.0:
                coord_closest_dist = efficient_cdist(
                    current_coord.float(), init_coord.float()
                ).min(1)[0]
                closest_mask = coord_closest_dist < patch_parameters.limit_geometry_dist
                current_coord = current_coord[closest_mask]
                current_feat = current_feat[closest_mask]

            # breakpoint()
            if len(current_coord) == 0:
                break
        # For the last iteration, the new current feature is given by the nn in ref, no blending!
        else:
            current_feat = ref_feat[nns]
            current_coord = current_coord

            new_coords.append(current_coord)
            new_feats.append(current_feat)
            new_ref_coords.append(ref_coord[nns])

    return new_coords, new_feats, new_ref_coords
