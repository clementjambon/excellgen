# Gaussians, Post-processing and Training of GCAs

## (Optional) Getting your own datasets ready

If you want to process your own videos with nerfstudio and COLMAP, first make sure nerfstudio CLI is installed with
```sh
ns-install-cli
```

Then, run (by filling appropriately the placeholders)
```sh
ns-process-data video --data [INPUT_VIDEO] --output-dir [TARGET_FOLDER]
```

## Turning it into splats 

Now, we can finally let the magic go on 🪄 To do so, you'll need a config file 🗃️ You can start from the default one in `configs/default.yaml` and modify the `data_dir` with your own choice of dataset. If you run out of memory, consider using `configs/default_small.yaml` instead.

<!-- Then, you can visualize training with the viewer using
```sh
python scripts/viewer.py --input [CONFIG_FILE] 
``` -->

Train without viewer in headless-mode using
```sh
python scripts/run.py --input [CONFIG_FILE] (Optional: --data_dir [NERFSTUDIO_NERF_DATASET])
```

<!-- Note that the viewer almost shouldn't bring too much overhead because rasterization is quite fast. -->

I would recommend to run things for more than 20000 iterations as pruning really starts around 15000 iterations, but you should see cool things going on from 3000 iterations already.

By default, runs are saved in `log/[timestamp]`. If you want to override this, add `--log_dir [LOG_FOLDER]` to the initial command.

After training, you can visualize the scene by simply substituting the input path with the checkpoint file, for example:
```sh
python scripts/viewer.py --input log/05-25-23:38:45/ckpts/00005000.pt
```

<!-- **WARNING**: for now, you need to reload everything to start editing (sorry about that!) -->

If you want to play around with rendering, open the "*Rendering*" tab and play around with the sliders 🎚️. The most important thing you might be interested in is switching rendering to features. Note that there's a shortcut for this: press "*f*". If you have an active quantizer, features will be rendered with quantization by default. You can switch it on and off with "q".

## Preparing data for GCA

With a pretrained snapshot, you are almost ready to go! If you are satisfied by your pre-trained scene, **move the corresponding log folder to a subfolder of `primitives`**. This way, we'll be able to find primitives directly in the hierarchy. In other words, this will be tracked as part of a collection of primitives. 

**WARNING:** if you move a folder later on, paths might be broken, and you might have to fix things by yourself. As a consequence, please make sure to index things the way you want to use things later on. The convention used in our dataset is `[category]/[name]`.

**NOTE:** you can add a preview thumbnail within each primitive folder by giving it the name `preview.png`. This will be shown when you hover the corresponding primitive in the library within the viewer.

### (Optional) Prefilter Gaussians using quantized features

If you want to use (PCA-ed) features to filter out semantic regions of the scenes, switch to feature rendering (press "*f*") and quantized mode (press "*q*"). 

If your features are not quantized, it's because you don't have a quantizer yet! 
Don't stress out: open the "*Suggestive Selection*" tab, then the subsequent "*Quantization*" tab and finally click on "*Update Quantizer*". From there, you can play with quantization parameters like the number of k-means iterations or clusters. Note that with the same set of parameters, you'll get different segments every time you update the quantizer because k-means is non-deterministic (due to random seeds for initialization).

To select segments, hold "*Ctrl*" pressed and click on the target segment. It should turn gray. 
If you want, to discard a segment, click again and it should go back to its original color. If you switch back to RGB rendering (press "*f*"). You should see that unselected segments are not rendered.
You can easily re-run k-means by pressing "*Space*".

### Building a hierarchy of surface voxels for GCAs

Now, open the "*Latent Exporter*" tab. If the currently rendered bounding box doesn't fit your content, you can adjust it. In general, **you should always adjust it!** However, don't try to fit perfectly the content because you must remember that this will define the relative resolution over which GCA will operate! In other words, if it's too granular, it might be too complicated for it. If it is too coarse, it will copy chunks from the exemplar.

If you can see the bounding box and its gizmo, click on "*Limit bbox*" and move the box around with the gizmo. To adjust the scale of the box, press "*s*" to switch the gizmo mode (and back). 

Once you're done, click on "*Precompute*". This will capture all the Gaussians within the bounding box and take into account the semantically-aware segments that you optionally selected before. 

In general, you might want to filter out part of this selection (and you should!). This is made possible by the "*PcSelector*" that was created when you pressed "*Precompute*":
* Its parameters can be controlled in the small menu box that should have appeared. 
* The most important thing you might be interested in is the brush mode: by using the slider or **pressing "*s*"**, you can switch between adding or removing elements from the selection. 
* You should see that when you hover the current point cloud, points are highlighted within a sphere of a radius that you can adjust by pressing "*alt*" and using the mouse wheel. You can also change the selection to be a cube by selecting the corresponding checkbox in the panel.
* To Add (green)/Remove (red) points, maintain "*alt*" pressed and click with you mouse. 
* You can reset your selection at any time using "*Reset Selection*". Note that in "Add" mode, this will trigger an empty selection (fully black) and in "Remove" mode, this will set a full selection.
* **Black means unselected, White means selected**
* To hide/show the point cloud, press "*h*".
* You can also preview the current selection by pressing "*o*".

During editing, you can still move the limit bbox again (click on "*Limit bbox*") and "*Re-compute*". However, note that your previous selection will be lost!

After editing, you can either set a name to your future GCA primitive in the corresponding placeholder or directly press "*Export*".

By default, this creates everything you need to train the corresponding GCA and subsequently using it as a primitive. More precisely:
* The full hierarchy of voxels (at resolutions `[128, 64, 32, 16, 8]`) with the PCA-ed features will be stored at `primitives/[your_path]/gca_latents/raw/[your_export_name].npz`
* The pruned set of splats will be saved in `primitives/[your_primitive_path]/gaussian_ckpt/[your_export_name].pt`. After that, you should only use this one because memory usage is dramatically reduced!
* A config file for the GCA you will train is generated at `primitives/[your_primitive_path]/gca_config/[your_export_name].yaml`. Note that it is automatically generated from a template that you can find at `deps/fast-gca/configs/default/template_fine=1_coarse=3_geo=4_dino=4_s0_T=5.yaml`. If you want to play with GCA training parameters that's the file you should take a look at.
* Most importantly, an executable to launch training automatically is generated at `primitives/[your_path]/gca_config/[your_export_name].sh`.

**NOTE:** you can use the same selection system to simply create a "layer" of Gaussians by clicking on "*Layer*". You can also remove the corresponding Gaussians from the current layer by clicking on "*Inverted Layer*".

### Training a GCA

After following the procedure above, you can simply execute
```sh
./primitives/[your_path]/gca_config/raw/[your_export_name].sh
```

Note that this will automatically write into `primitives/[your_path]/gca_config/gca_logs/[your_export_name]`. Wait around 10 minutes (depending on you machine ⚙️).