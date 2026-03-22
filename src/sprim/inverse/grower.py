from dataclasses import dataclass, fields
from typing import Dict, Any, List
from enum import StrEnum
import os
import yaml
import time

from torch.utils.tensorboard import SummaryWriter
import torch
import numpy as np
import polyscope.imgui as psim

from sprim.patch.exact_search import exact_search, PatchParameters
from sprim.gaussians.gaussian_model import GaussianModel, GaussianSet
from sprim.inverse.grown_voxels import (
    RESOLUTIONS,
    RESOLUTIONS_INV,
    GrownVoxels,
    LatentMode,
    LATENT_MODE_MAP,
    LATENT_MODE_INVMAP,
)
from sprim.utils.voxel_set import VoxelSet
from sprim.utils.gui_utils import KEY_HANDLER

from fast_gca.models import MODEL
from fast_gca.datasets import DATASET
from fast_gca.models.components.phase_profile import PhaseProfile
from fast_gca.utils.growing_parameters import GrowingParameters
from fast_gca.datasets.fine_stage_dataset import FineStageStateS0
from fast_gca.utils.phase import UnitializedPhase

from sklearn.decomposition import PCA


RESET_GPU_CACHE_THRESHOLD = 0.2


def load_gca_model_and_data(
    gca_ckpt: str, no_writer: bool = False, load_datasets: bool = False
):
    # Load gca_model and gca_configs
    base_dir = os.path.dirname(os.path.dirname(gca_ckpt))
    gca_config_path = os.path.join(base_dir, "config.yaml")
    gca_config = yaml.load(open(gca_config_path), Loader=yaml.FullLoader)
    # Make sure that device is "cuda:0"
    gca_config["device"] = "cuda:0"
    torch.cuda.set_device("cuda:0")

    # Build components
    writer = None if no_writer else SummaryWriter(gca_config["log_dir"])
    gca_model = MODEL[gca_config["model"]](gca_config, writer)
    gca_model.to(gca_config["device"])
    checkpoint = torch.load(gca_ckpt, map_location=gca_config["device"])
    gca_model.load_state_dict(checkpoint["model_state_dict"])

    if load_datasets:
        train_dataset = DATASET[gca_config["dataset"]](
            gca_config, mode="train", batch_size=gca_config["batch_size"]
        )
        test_datasets = [
            DATASET[x[0]](
                gca_config,
                mode=x[1],
                data_root=None,
                fine_surface_voxels=train_dataset.preloaded_data[0].fine_surface_voxels,
                fine_voxel_latents=train_dataset.preloaded_data[0].fine_voxel_latents,
            )
            for x in gca_config["test_datasets"]
        ]
    else:
        train_dataset = None
        test_datasets = None

    gca_model.eval()

    return (
        gca_model,
        gca_config,
        {
            "test": test_datasets[0] if test_datasets is not None else None,
            "train": train_dataset,
        },
    )


def prepare_state_s0(
    coarse_voxels,
    offset_voxels,
    ref_coord,
    ref_feat,
    context_res,
    coarse_res,
    fine_res,
    offset,
    input_occ: bool = True,
    z_dim: int = 8,
):
    # UPDATE: Downsample offset voxels to coarse resolution and then, subsample them
    # NOTE: this is slightly different from what is effectively done below for conditioning because for
    # conditioning, we match to a stride that corresponds to an initial offset of 0!
    fine_coarse_ratio = fine_res // coarse_res
    coarse_offset_voxels = (offset_voxels // fine_coarse_ratio).int()
    coarse_offset_voxels = torch.unique(coarse_offset_voxels, dim=0)

    # And contrary to "explicit", these will be both our own init and conditioning state
    upsampled_coarse_offset_voxels = fine_coarse_ratio * coarse_offset_voxels

    state_coord = upsampled_coarse_offset_voxels

    state_feat = torch.randn((state_coord.shape[0], z_dim), device=state_coord.device)

    if input_occ:
        state_feat = torch.cat(
            [
                torch.ones((state_coord.shape[0], 1), device=state_feat.device),
                state_feat,
            ],
            dim=1,
        )

    # For each of them, create a fine_coarse_ratio list of elements to sample from
    cond_feat = torch.ones((state_coord.shape[0], 1), device=state_coord.device)

    return FineStageStateS0(
        state_coord=state_coord,
        state_feat=state_feat,
        ref_coord=ref_coord,
        ref_coord_coarse=coarse_offset_voxels,
        ref_feat=ref_feat,
        appearance_feat=torch.randn(
            (state_coord.shape[0], ref_feat.shape[-1]), device=state_coord.device
        ),
        cond_coord=state_coord,
        cond_feat=cond_feat,
        phase=UnitializedPhase(),
        cond_dropout_p=0.0,
        # TODO: this is overkill!
        offset=offset,
        fine_coarse_ratio=fine_coarse_ratio,
        coarse_res=coarse_res,
        fine_res=fine_res,
        context_res=context_res,
    )


def dataclass_to_cuda(data):
    result = {}

    for field in fields(data):
        if type(getattr(data, field.name)) == torch.Tensor:
            result[field.name] = getattr(data, field.name).cuda()
        else:
            result[field.name] = getattr(data, field.name)

    return type(data)(**result)


class GenerationMode(StrEnum):
    GCA = "gca"
    PATCHMATCH = "patch-match"


GENERATION_MODE_MAP = {x: i for i, x in enumerate(GenerationMode)}
GENERATION_MODE_INVMAP = {i: x for i, x in enumerate(GenerationMode)}

MODE_SEEKING_STEPS = 1


class Grower:

    def __init__(
        self,
        gca_input: str,
        gaussian_model: GaussianModel | None,
        gaussian_update_callback=None,
        get_transform=None,
    ) -> None:
        self.generation_mode = GenerationMode.GCA

        self.gca_input = gca_input

        self.gaussian_model = gaussian_model
        self.ref_gaussians = gaussian_model.get_gaussian_set()
        self.gaussian_update_callback = gaussian_update_callback
        # TODO: do that more properly...
        self.get_transform = get_transform

        self.model, self.config, self.datasets = load_gca_model_and_data(
            gca_input, no_writer=True, load_datasets=False
        )
        self.voxel_res = RESOLUTIONS[self.config["res_idx"]]
        self.coarse_res = RESOLUTIONS[self.config["res_idx_coarse"]]

        self.phase_profile: PhaseProfile = self.model.phase_profile
        self.growing_parameters = GrowingParameters(
            sampling_steps=7,
            # sampling_steps=self.phase_profile.get_test_max_phase(
            #     self.config["max_steps"]
            # ),
            # mode_seeking_steps=self.phase_profile.get_test_mode_seeking_steps(
            #     gca_config["max_steps"]
            # ),
            mode_seeking_steps=MODE_SEEKING_STEPS,
        )

        self.patch_parameters = PatchParameters()

        # TODO: this is debug only. This should use config...
        self.input_occ = True
        self.cond_category = 0

        self.FineStageState = FineStageStateS0
        self.prepare_fine_state = prepare_state_s0

        self.ref_pcas = None

        self.coarse_voxel_input = None
        self.visualization_input = None

        self.transform_from_original_scene = None
        self.load_hierarchy()
        self.target_res_idx = 1
        self.step_per_res = 1
        self.current_step_at_res = 0
        self.patch_match_input = None

        # Visualize latent voxels
        # NOTE: we make grower "own" these to avoid dangling pointers
        self.latent_voxels_shown = False
        self.latent_voxel_set: VoxelSet | None = None
        self.latent_voxel_mode: LatentMode = LatentMode.NN_patch

        self.restart(False)

    def load_hierarchy(self):
        self.voxel_hierarchy = []
        self.ref_pcas = []

        # NOTE: res_idx are set when exporting latents (i.e., res prefixes):
        # "lower is finer"
        gca_log_path = os.path.dirname(os.path.dirname(self.gca_input))
        if os.path.exists(os.path.join(gca_log_path, "raw.npz")):
            raw_path = os.path.join(gca_log_path, "raw.npz")
        else:
            gca_name = os.path.basename(gca_log_path)
            gca_latents_path = os.path.dirname(os.path.dirname(gca_log_path))
            raw_path = os.path.join(gca_latents_path, "raw", gca_name + ".npz")

        data = np.load(raw_path)

        self.bbox_min = torch.tensor(data["bbox_min"]).float().cuda().unsqueeze(0)
        self.bbox_max = torch.tensor(data["bbox_max"]).float().cuda().unsqueeze(0)

        self.transform_from_original_scene = (
            torch.tensor(data["transform_from_original_scene"]).float().cuda()
            if "transform_from_original_scene" in data
            else None
        )

        for idx, res in enumerate(RESOLUTIONS):
            voxel_res = data[f"{idx}_res"].item()
            assert res == voxel_res
            surface_voxels = torch.tensor(data[f"{idx}_voxels"]).cuda()
            latents = torch.tensor(data[f"{idx}_voxels_latents"]).float().cuda()

            pca = PCA(n_components=3)
            pca = pca.fit(latents.cpu().numpy())

            self.ref_pcas.append(pca)

            self.voxel_hierarchy.append(
                GrownVoxels(
                    voxel_res=res,
                    voxel_res_idx=idx,
                    surface_voxels=surface_voxels,
                    latents=latents,
                    latents_matched=None,
                    coord_in_ref=surface_voxels,
                    get_transform=self.get_transform,
                    pca_to_rgb=self.pca_to_rgb,
                    bbox_min=self.bbox_min,
                    bbox_max=self.bbox_max,
                )
            )

    @torch.no_grad()
    def prepare_state_custom(self, surface_voxels=None, coarse_voxels=None):
        assert surface_voxels is not None or coarse_voxels is not None

        ref_coord = self.voxel_hierarchy[RESOLUTIONS_INV[self.voxel_res]].surface_voxels
        ref_feat = self.voxel_hierarchy[RESOLUTIONS_INV[self.voxel_res]].latents

        fine_coarse_ratio = self.voxel_res // self.coarse_res

        if surface_voxels is None:
            coarse_offset_voxels = coarse_voxels.int()
            surface_voxels = (fine_coarse_ratio * coarse_voxels).int()
        else:
            coarse_offset_voxels = (surface_voxels // fine_coarse_ratio).int()

        self.visualization_input = self.prepare_fine_state(
            coarse_voxels=coarse_offset_voxels,
            offset_voxels=surface_voxels,
            ref_coord=ref_coord,
            ref_feat=ref_feat,
            context_res=self.voxel_res,
            coarse_res=self.coarse_res,
            fine_res=self.voxel_res,
            offset=torch.zeros((1, 3)).cuda(),
        )

        self.visualization_input = dataclass_to_cuda(self.visualization_input)

        self.coarse_voxel_input = coarse_offset_voxels

        # TODO: decouple this to allow start at coarser resolution
        n, d = coarse_offset_voxels.shape[0], self.voxel_hierarchy[0].latents.shape[-1]
        latents_sh = torch.rand((n, d // 2))
        latents_sh /= torch.linalg.norm(latents_sh, dim=-1, keepdim=True)
        latents_feat = torch.rand((n, d // 2))
        latents_feat /= torch.linalg.norm(latents_feat, dim=-1, keepdim=True)
        latents = torch.cat(
            [latents_sh, latents_feat],
            dim=-1,
        )
        self.patch_match_input = GrownVoxels(
            voxel_res=self.coarse_res,
            surface_voxels=coarse_offset_voxels,
            latents=latents,
            latents_matched=None,
            coord_in_ref=None,
            get_transform=self.get_transform,
            pca_to_rgb=self.pca_to_rgb,
            bbox_min=self.bbox_min,
            bbox_max=self.bbox_max,
        )

        new_coords, new_feats, new_ref_coords = exact_search(
            self.patch_match_input.surface_voxels,
            self.patch_match_input.latents,
            self.voxel_hierarchy[self.patch_match_input.voxel_res_idx].surface_voxels,
            self.voxel_hierarchy[self.patch_match_input.voxel_res_idx].latents,
            patch_parameters=self.patch_parameters,
            single_step=True,
        )

        self.patch_match_input.coord_in_ref = new_ref_coords[-1]

    def restart(self, is_stepping: bool = True):
        self.state = None
        self.is_stepping = is_stepping
        # self.trajectory = []
        self.grown_voxels: List[GrownVoxels] = []
        self.i_trajectory = 0

        self.timings = {"grow_gca": [], "grow_patch": [], "patch": []}

        if (
            self.generation_mode == GenerationMode.PATCHMATCH
            and self.patch_match_input is not None
        ):
            self.state = self.patch_match_input

            # Change latent
            n, d = (
                self.state.surface_voxels.shape[0],
                self.voxel_hierarchy[0].latents.shape[-1],
            )
            latents_sh = torch.rand((n, d // 2))
            latents_sh /= torch.linalg.norm(latents_sh, dim=-1, keepdim=True)
            latents_feat = torch.rand((n, d // 2))
            latents_feat /= torch.linalg.norm(latents_feat, dim=-1, keepdim=True)
            latents = torch.cat(
                [latents_sh, latents_feat],
                dim=-1,
            )
            self.state.latents = latents

            self.current_step_at_res = 0
            self.grown_voxels = [self.state]

    def update_latent_voxels(self):
        if self.latent_voxels_shown and self.i_trajectory < len(self.grown_voxels):
            # If there is a voxel_set, remove it
            if self.latent_voxel_set is not None:
                self.latent_voxel_set.remove()
                self.latent_voxel_set = None
            self.latent_voxel_set = self.grown_voxels[self.i_trajectory].get_voxel_set(
                ref_coord=self.voxel_hierarchy[self.target_res_idx].surface_voxels,
                ref_feat=self.voxel_hierarchy[self.target_res_idx].latents,
                latent_mode=self.latent_voxel_mode,
            )
        else:
            # Let's remove otherwise, it will be conservative this way
            if self.latent_voxel_set is not None:
                self.latent_voxel_set.remove()
                self.latent_voxel_set = None

    def update_render(self):
        self.update_latent_voxels()

        if self.gaussian_model is not None:
            self.gaussian_model.set_gaussian_set(
                self.grown_voxels[self.i_trajectory].process_gaussians(
                    self.ref_gaussians,
                    self.bbox_min,
                    self.bbox_max,
                    ref_coord=self.voxel_hierarchy[self.target_res_idx].surface_voxels,
                    ref_feat=self.voxel_hierarchy[self.target_res_idx].latents,
                    latent_mode=self.latent_voxel_mode,
                ),
                grown=True,
            )
            if self.gaussian_update_callback is not None:
                self.gaussian_update_callback()

    def gui(self):

        window_flags = psim.ImGuiWindowFlags_MenuBar
        psim.PushStyleVar(psim.ImGuiStyleVar_ChildRounding, 1.0)
        psim.PushStyleColor(
            psim.ImGuiCol_ChildBg, psim.ImColor.HSV(4.0 / 7.0, 0.5, 0.5)
        )
        psim.BeginChild(f"grower", size=(0, 200), border=True, flags=window_flags)

        # =====================
        # Window starts here
        # =====================

        if psim.BeginMenuBar():
            psim.Text(f"Grower")
            psim.EndMenuBar()

            display_timings  = ""
            for k, v in self.timings.items():
                if len(v) > 0:
                    display_timings += f"{k}: {sum(v):0.4f};"

            if len(display_timings) > 0:
                psim.Text(display_timings)

            clicked, generation_mode_idx = psim.SliderInt(
                "Generation Mode",
                GENERATION_MODE_MAP[self.generation_mode],
                v_min=0,
                v_max=len(GenerationMode) - 1,
                format=f"{self.generation_mode.value}",
            )

            if clicked:
                self.generation_mode = GENERATION_MODE_INVMAP[generation_mode_idx]

            # if psim.Button("Grow"):
            #     self.grow()

            # TODO: patch
            if psim.Button("Restart"):
                # Prepare again just in case!
                # self.prepare_state(self.current_idx)
                self.restart()

            psim.SameLine()

            if psim.Button("Patch") and self.generation_mode == GenerationMode.GCA:
                self.patch_current_state()

            if len(self.grown_voxels) > 0:

                # Switch on and off VoxelSets
                if KEY_HANDLER("h"):
                    self.latent_voxels_shown = not self.latent_voxels_shown
                    self.update_latent_voxels()

                clicked, latent_mode_idx = psim.SliderInt(
                    "Latent Mode##grower",
                    LATENT_MODE_MAP[self.latent_voxel_mode],
                    v_min=0,
                    v_max=len(LatentMode) - 1,
                    format=f"{self.latent_voxel_mode.value}",
                )
                if clicked:
                    self.latent_voxel_mode = LATENT_MODE_INVMAP[latent_mode_idx]
                    # TODO: avoid updating full render! (only voxels should suffice)
                    self.update_render()

                test_cond = (
                    self.i_trajectory > self.growing_parameters.sampling_steps + 1
                )
                if test_cond:
                    psim.PushStyleColor(
                        psim.ImGuiCol_FrameBg, psim.ImColor.HSV(0 / 7.0, 0.5, 0.5)
                    )
                    psim.PushStyleColor(
                        psim.ImGuiCol_FrameBgHovered,
                        psim.ImColor.HSV(0 / 7.0, 0.6, 0.5),
                    )
                    psim.PushStyleColor(
                        psim.ImGuiCol_FrameBgActive, psim.ImColor.HSV(0 / 7.0, 0.7, 0.5)
                    )
                    psim.PushStyleColor(
                        psim.ImGuiCol_SliderGrab, psim.ImColor.HSV(0 / 7.0, 0.9, 0.9)
                    )

                clicked, self.i_trajectory = psim.SliderInt(
                    "step",
                    self.i_trajectory,
                    0,
                    len(self.grown_voxels) - 1,
                    format="%d" + " (patched)" if test_cond else "",
                )
                if test_cond:
                    psim.PopStyleColor(4)
                if clicked:
                    self.update_render()

            self.patch_parameters.gui()

            self.growing_parameters.gui()

        # =====================
        # Window ends here
        # =====================

        psim.EndChild()
        psim.PopStyleVar()
        psim.PopStyleColor()

    @torch.no_grad()
    def grow(self):
        if self.generation_mode == GenerationMode.GCA:
            if self.is_stepping:
                time_before = time.time()
            else:
                time_before = None
            # ========================
            self.grow_gca()
            # ========================
            if time_before is not None:
                self.timings["grow_gca"].append(time.time() - time_before)
        else:
            if self.is_stepping:
                time_before = time.time()
            else:
                time_before = None
            # ========================
            self.grow_patchmatch()
            # ========================
            if time_before is not None:
                self.timings["grow_patch"].append(time.time() - time_before)

        return self.is_stepping

    @torch.no_grad()
    def grow_patchmatch(self):
        if self.is_stepping:

            self.state: GrownVoxels = self.state

            new_coords, new_feats, new_ref_coords = exact_search(
                self.state.surface_voxels,
                self.state.latents,
                self.voxel_hierarchy[self.state.voxel_res_idx].surface_voxels,
                self.voxel_hierarchy[self.state.voxel_res_idx].latents,
                patch_parameters=self.patch_parameters,
            )

            for new_coord, new_feat, new_ref_coord in zip(
                new_coords, new_feats, new_ref_coords
            ):

                self.grown_voxels.append(
                    GrownVoxels(
                        voxel_res=self.state.voxel_res,
                        voxel_res_idx=self.state.voxel_res_idx,
                        surface_voxels=new_coord,
                        latents=new_feat,
                        latents_matched=new_feat,
                        coord_in_ref=new_ref_coord,
                        get_transform=self.get_transform,
                        pca_to_rgb=self.pca_to_rgb,
                        bbox_min=self.bbox_min,
                        bbox_max=self.bbox_max,
                    )
                )

                # pca_rgb_features = self.pca_to_rgb(
                #     RESOLUTIONS_INV[self.state.voxel_res], new_feat
                # )

                # self.trajectory.append(
                #     VoxelSet(
                #         voxels=new_coord,
                #         get_transform=self.get_transform,
                #         res=self.state.voxel_res,
                #         rgb=pca_rgb_features,
                #         voxel_edge_width=0.0,
                #         prefix="trajectory",
                #         bbox_min=self.bbox_min,
                #         bbox_max=self.bbox_max,
                #         enabled=False,
                #     )
                #     if not headless_mode
                #     else None
                # )

            self.state = GrownVoxels(
                voxel_res=self.state.voxel_res,
                voxel_res_idx=self.state.voxel_res_idx,
                surface_voxels=new_coords[-1],
                latents=new_feats[-1],
                latents_matched=new_feats[-1],
                coord_in_ref=new_ref_coords[-1],
                get_transform=self.get_transform,
                pca_to_rgb=self.pca_to_rgb,
                bbox_min=self.bbox_min,
                bbox_max=self.bbox_max,
            )

            self.i_trajectory = len(self.grown_voxels) - 1

            self.current_step_at_res += 1

            self.update_render()

            # Stop
            if (
                self.state.voxel_res_idx == self.target_res_idx
                and self.current_step_at_res == self.step_per_res
            ):
                self.is_stepping = False
                return

            # TODO: add jitter at every upsampling step
            if self.current_step_at_res == self.step_per_res:
                self.state = self.state.upsample()
                self.current_step_at_res = 0

    @torch.no_grad()
    def grow_gca(self):

        if self.is_stepping:

            mem_info = torch.cuda.mem_get_info()
            if mem_info[0] / mem_info[1] < RESET_GPU_CACHE_THRESHOLD:
                print(
                    f"Hit {mem_info[0] / mem_info[1]} < {RESET_GPU_CACHE_THRESHOLD} memory ratio -> emptying cache!"
                )
                torch.cuda.empty_cache()

            (
                self.state,
                self.is_stepping,
            ) = self.model.visualize_step(
                self.visualization_input,
                self.state,
                growing_parameters=self.growing_parameters,
                infused_inference=None,
            )

            # Stop here to avoid doubling step!
            if not self.is_stepping:
                return

            # Grown voxels attribute are always derived after a first match to get the corresponding coordinates
            new_coords, new_feats, new_ref_coords = exact_search(
                self.state.state_coord[0],
                self.state.appearance_feat[0],
                self.state.ref_coord[0],
                self.state.ref_feat[0],
                patch_parameters=self.patch_parameters,
                single_step=True,
            )

            # Just sanity
            assert len(new_coords[-1]) == len(self.state.state_coord[0])

            self.grown_voxels.append(
                GrownVoxels(
                    voxel_res=self.state.fine_res[0],
                    surface_voxels=new_coords[-1],
                    latents=self.state.appearance_feat[0],
                    latents_matched=new_feats[-1],
                    coord_in_ref=new_ref_coords[-1],
                    get_transform=self.get_transform,
                    pca_to_rgb=self.pca_to_rgb,
                    bbox_min=self.bbox_min,
                    bbox_max=self.bbox_max,
                )
            )

            # pca_rgb_features = self.pca_to_rgb(
            #     RESOLUTIONS_INV[self.state.fine_res[0]], new_feats[-1]
            # )

            # # TODO: keep everything on the GPU
            # # It is clearly doable (cf. discussion with Nick)
            # self.trajectory.append(
            #     VoxelSet(
            #         voxels=self.state.state_coord[0],
            #         get_transform=self.get_transform,
            #         res=self.state.fine_res[0],
            #         rgb=pca_rgb_features,
            #         voxel_edge_width=0.0,
            #         prefix="trajectory",
            #         bbox_min=self.bbox_min,
            #         bbox_max=self.bbox_max,
            #         enabled=False,
            #     )
            #     if not headless_mode
            #     else None
            # )

            # prev_i_trajectory = self.i_trajectory
            self.i_trajectory = len(self.grown_voxels) - 1

            self.update_render()

    def patch_current_state(self, headless_mode: bool = False):
        if self.generation_mode == GenerationMode.GCA:
            time_before = time.time()
            self.patch_current_state_gca(headless_mode)
            self.timings["patch"].append(time.time() - time_before)
        else:
            print("Cannot patch when generating with PatchMatch!")

    @torch.no_grad()
    def patch_current_state_gca(self, headless_mode: bool = False):

        # Make sure to cut everything that happened before to avoid re-appending
        # "+1" because we also have the initial step
        # self.trajectory = self.trajectory[
        #     : self.growing_parameters.sampling_steps + MODE_SEEKING_STEPS + 1
        # ]
        self.grown_voxels = self.grown_voxels[
            : self.growing_parameters.sampling_steps + MODE_SEEKING_STEPS + 1
        ]

        # Grown voxels attribute are always derived after a first match to get the corresponding coordinates
        new_coords, new_feats, new_ref_coords = exact_search(
            self.state.state_coord[0],
            self.state.appearance_feat[0],
            self.state.ref_coord[0],
            self.state.ref_feat[0],
            patch_parameters=self.patch_parameters,
        )
        for new_coord, new_feat, new_ref_coord in zip(
            new_coords, new_feats, new_ref_coords
        ):
            self.grown_voxels.append(
                GrownVoxels(
                    voxel_res=self.state.fine_res[0],
                    surface_voxels=new_coord,
                    latents=new_feat,
                    latents_matched=new_feat,
                    coord_in_ref=new_ref_coord,
                    get_transform=self.get_transform,
                    pca_to_rgb=self.pca_to_rgb,
                    bbox_min=self.bbox_min,
                    bbox_max=self.bbox_max,
                )
            )

            # pca_rgb_features = self.pca_to_rgb(
            #     RESOLUTIONS_INV[self.state.fine_res[0]], new_feat
            # )

            # self.trajectory.append(
            #     VoxelSet(
            #         voxels=new_coord,
            #         get_transform=self.get_transform,
            #         res=self.state.fine_res[0],
            #         rgb=pca_rgb_features,
            #         voxel_edge_width=0.0,
            #         prefix="trajectory",
            #         bbox_min=self.bbox_min,
            #         bbox_max=self.bbox_max,
            #         enabled=False,
            #     )
            #     if not headless_mode
            #     else None
            # )

        self.i_trajectory = len(self.grown_voxels) - 1

        self.update_render()

    @torch.no_grad()
    def pca_to_rgb(self, res_idx, latent: torch.Tensor) -> torch.Tensor:

        if self.ref_pcas is None:

            # This is extremely DIY but this shouldn't happen at all
            if len(latent.cpu().numpy()) > 3:

                # PCA on matched features
                pca = PCA(n_components=3)
                feature_maps_pca = pca.fit_transform(latent.cpu().numpy())
                pca_features_min = feature_maps_pca.min(axis=(0, 1))
                pca_features_max = feature_maps_pca.max(axis=(0, 1))
                pca_features = (feature_maps_pca - pca_features_min) / (
                    pca_features_max - pca_features_min
                )
                pca_features = torch.tensor(pca_features).cuda()
            else:
                pca_features = torch.zeros((len(latent.cpu().numpy()), 3)).cuda()

            return pca_features

        else:
            # TODO: handle PCA on GPU (this isn't a bottleneck though)
            return torch.sigmoid(
                torch.tensor(self.ref_pcas[res_idx].transform(latent.cpu().numpy()))
            ).cuda()

    def serialize(self) -> Dict[str, Any]:

        # Grown voxels
        data = {"grown_voxels": []}
        for grown_voxel in self.grown_voxels:
            data["grown_voxels"].append(grown_voxel.serialize())

        data["i_trajectory"] = self.i_trajectory

        if self.visualization_input is not None:
            data["visualization_input"] = self.visualization_input.serialize()

        if self.coarse_voxel_input is not None:
            data["coarse_voxel_input"] = self.coarse_voxel_input.cpu().numpy()

        return data

    def load_serialized(self, data: Dict[str, Any]):

        for entry in data["grown_voxels"]:
            if entry is None:
                continue
            grown_voxel = GrownVoxels.deserialize(
                entry,
                get_transform=self.get_transform,
                pca_to_rgb=self.pca_to_rgb,
                bbox_min=self.bbox_min,
                bbox_max=self.bbox_max,
            )
            self.grown_voxels.append(grown_voxel)

            # TODO: handle features properly for deserialization

            # PCA on matched features
            # pca = PCA(n_components=3)
            # feature_maps_pca = pca.fit_transform(grown_voxel.latents.cpu().numpy())
            # pca_features_min = feature_maps_pca.min(axis=(0, 1))
            # pca_features_max = feature_maps_pca.max(axis=(0, 1))
            # pca_features = (feature_maps_pca - pca_features_min) / (
            #     pca_features_max - pca_features_min
            # )

            # voxel_set = VoxelSet(
            #     voxels=grown_voxel.surface_voxels,
            #     get_transform=self.get_transform,
            #     res=grown_voxel.voxel_res,
            #     rgb=torch.tensor(pca_features).cuda(),
            #     voxel_edge_width=0.0,
            #     prefix="trajectory",
            #     bbox_min=self.bbox_min,
            #     bbox_max=self.bbox_max,
            #     enabled=False,
            # )
            # self.trajectory.append(voxel_set)

        self.i_trajectory = min(data["i_trajectory"], len(self.grown_voxels) - 1)

        if "visualization_input" in data:
            self.visualization_input = self.FineStageState.deserialize(
                data["visualization_input"]
            )
            self.visualization_input = dataclass_to_cuda(self.visualization_input)

        if "coarse_voxel_input" in data:
            self.coarse_voxel_input = torch.tensor(data["coarse_voxel_input"]).cuda()

        # Render the desired step
        if len(self.grown_voxels) > 0:
            print("processed gaussians")
            self.gaussian_model.set_gaussian_set(
                self.grown_voxels[self.i_trajectory].process_gaussians(
                    self.ref_gaussians,
                    self.bbox_min,
                    self.bbox_max,
                ),
                grown=True,
            )
        # if self.gaussian_update_callback is not None:
        #     self.gaussian_update_callback()
