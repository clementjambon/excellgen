# ExCellGen: Fast, Controllable, Photorealistic 3D Scene Generation from a Single Real-World Exemplar

![teaser](assets/teaser.jpg)

Official code release for the paper:  
**ExCellGen: Fast, Controllable, Photorealistic 3D Scene Generation from a Single Real-World Exemplar**  
CVM 2026 (Journal Track)
[Clément Jambon](https://clementjambon.github.io/), [Changwoon Choi](https://www.changwoon.info/), [Dongsu Zhang](https://dszhang.me/about), [Olga Sorkine-Hornung](https://igl.ethz.ch/people/sorkine/), [Young Min Kim](https://3d.snu.ac.kr/members)

[[Webpage]](https://excellgen.github.io) · [[Paper]](https://arxiv.org/abs/2412.16253) · [[Supplemental Document]](assets/supplemental.pdf) · [[Supplemental Video]](https://drive.google.com/file/d/1Vk0-fi-O-M05PscU345jeNkryUGH0Xkw/view?usp=sharing)

## Requirements

* **Ubuntu** 20.04 or higher
* **NVIDIA GPU** [compatible](https://docs.nvidia.com/deploy/cuda-compatibility/) with CUDA 11.8
* **Miniconda** or Anaconda to manage Python virtual environments
* A version of **GCC** higher than 8

## Instructions

For clarity, instructions are provided in separate files:
* [Installation](docs/installation.md)
* [Gaussians, Post-processing and Training of GCAs](docs/training.md)
* [Generation](docs/generation.md)
* [Shortcuts Sheet](docs/shortcuts.md)
* [Codebase Overview](docs/overview.md)
* [Reproducing Benchmarks](docs/benchmarks.md)