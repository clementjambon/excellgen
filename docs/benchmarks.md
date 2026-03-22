# Benchmarks

## Downloading pre-processed data

The pre-processed data is available following [this link](https://drive.google.com/file/d/1y_tw2FUGAp6TfgRtSqLzg12Y5VS84Nj1/view?usp=sharing). 
Download the archive and extract it in anywhere you want. 
Then, set the environment variable `EXP_PRIMITIVES_ROOT` to the root of the extracted folder.

## Running the benchmarks

To run the receptive field experiments, execute
```sh
python scripts/gca_scheduler.py --gca_configs configs/benchmarks/grids/receptive_field.txt --prim_list configs/benchmarks/prims/wall_flowers.csv --results_prefix receptive_field
python scripts/gca_renderer.py --gca_configs configs/benchmarks/grids/receptive_field.txt --prim_list configs/benchmarks/prims/wall_flowers.csv --results_prefix receptive_field
```

To run the performance experiments, execute
```sh
python scripts/gca_scheduler.py --gca_configs configs/benchmarks/grids/performance.txt --prim_list configs/benchmarks/prims/performance3.csv --results_prefix performance3
python scripts/gca_renderer.py --gca_configs configs/benchmarks/grids/performance.txt --prim_list configs/benchmarks/prims/performance3.csv  --results_prefix performance3
```

Note that for each run, you can use `--test` to test whether benchmarks will run or render properly.

## Understanding and interpreting the outputs

### gca_scheduler

By default, `gca_scheduler` will run all the `gca_configs` against all the primitives in `prim_list` and store the results in `experiments/results/[results_prefix]`.
At the root, `report.txt` will store all the recorded metrics. Note that results will be appended to the file. This means that if you run it a second time, the corresponding entries will appear after the one of the previous runs.
The corresponding entries are (in order) `scene_name`/`exp_name`/`raw`(precomputed hierarchy of voxels)/`start_time`(in seconds)/`end_time`(in seconds)/`duration`(in seconds)/`iterations/sec`/`gca_size`(in bytes).

### gca_renderer

By default, `gca_renderer` will render all the `gca_configs` for all the primitives in `prim_list` and store the results in `experiments/results/[results_prefix]`.
At the root, `report_renderer.txt` will store all the recorded metrics. Note that results will be appended to the file. Note that it requires these to be pretrained using `gca_scheduler`.
The corresponding entries are (in order) `scene_name`/`exp_name`/`raw`(precomputed hierarchy of voxels)/`n_samples`(over which metrics are averaged)/`ref_voxels`(the number of voxels of the exemplar)/`gca_time`(averaged over `n_samples`)/`patch_time`(averaged over `n_samples`)/`update_gaussian_time`(remapping step averaged over `n_samples`)/`grown_voxels_gca`(grown voxels averaged over `n_samples`)/`grown_voxels_patch`(voxels processed by our patch consistency step averaged over `n_samples`).

### Reproducing the plots in the paper

We provide a notebook to reproduce the plots used in the paper given the corresponding `report.txt` and `report_renderer.txt`. The notebook can be found in `configs/benchmarks/plots.ipynb`.