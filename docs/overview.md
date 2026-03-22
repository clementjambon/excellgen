# Codebase overview

For transparency, we list all the dependencies our pipeline relies on and the code we adapted.

## Modules

* [Polyscope](https://polyscope.run/py/). Our viewer is directly built on top of `polyscope-py`. However, it was [forked](https://github.com/clementjambon/polyscope-py/) and we made several changes and added features necessary to our framework:
    * We introduced a couple of callbacks to support hovering (used for point cloud and voxel selection/editing) and file drag-and-dropping.
    * The initial `imgui` bindings were extended to support missing features.
    * We modified the original gizmo to support multiple/configurable configurations.
* [Minkowski Engine](https://github.com/NVIDIA/MinkowskiEngine) wasn't modified except compilation fixes to support newer versions of CUDA.
* [fast-gca](https://github.com/clementjambon/excellgen-gca) started from the original [gca](https://github.com/96lives/gca) repository. It is the result of many experiments with GCA. This explains why the code may differ from the original implementation. Our manuscript details the fundamental changes to the method. Note however that, for the ultimate purpose of this project, we tried to stay as close as possible to the hyperparameters of [cGCA](https://arxiv.org/abs/2204.01264).

## Snippets

* Our `GaussianModel` (in `gaussians/gaussian_model.py`) is directly adapted from the [Splatfacto](https://docs.nerf.studio/nerfology/methods/splat.html) of [nerfstudio](https://docs.nerf.studio/).
* We rely on [gsplat](https://github.com/nerfstudio-project/gsplat) v0.1.11 for the 3D Gaussian rasterization routines.
* Our dataset loader `SubjectLoader` (in `inverse/nerfstudio_loader.py`) targets datasets processed by/for [nerfstudio](https://docs.nerf.studio/). As a consequence, it is directly adapted from [nerfstudio_dataparser.py](https://github.com/nerfstudio-project/nerfstudio/blob/main/nerfstudio/data/dataparsers/nerfstudio_dataparser.py).
* Our camera path spline system (in `gaussians/path_creator.py`) is directly based on the [code](https://github.com/nerfstudio-project/nerfstudio/blob/main/nerfstudio/viewer/render_panel.py) of [nerfstudio](https://docs.nerf.studio/) and its web viewer [viser](https://github.com/nerfstudio-project/viser).
* We adapted the code of [EmerNeRF](https://emernerf.github.io/) to extract DINO features. Note that it is itself originally adapted from the code of [Deep ViT Features as Dense Visual Descriptors](https://dino-vit-features.github.io/).
* The code of our patch consistency step (in `patch/exact_search.py`) was built on top of the implementation of [Sin3DGen](https://github.com/wyysf-98/Sin3DGen).
* Our basic tonemapper (in `gaussians/tonemapper`) adapts some routines from the codebase of [Exposure: A White-Box Photo Post-Processing Framework](https://github.com/yuanming-hu/exposure)