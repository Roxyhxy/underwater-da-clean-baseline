# Underwater DA Clean Baseline

This project is a cleaned baseline extracted from `DA_16_lora_feature_distill_adapter`.

It keeps only the core path that is still worth building on:

- `Depth Anything V2` metric-depth backbone
- encoder LoRA with `aquadegrade` style routing
- optional decoder ConvLoRA
- main depth supervision in the aligned evaluation space
- weak-image-perturbation consistency hardness weighting

It intentionally removes the exploratory branches that made the old project hard to maintain:

- feature distillation teacher
- delta hinge
- mutual degradation coupling
- SeaErra dual/calibrated branches
- search and bilevel gate branches
- style/spatial hardness auxiliary predictors
- transmission input placeholders

## Folder layout

- `train.py`: clean FLSea training entry
- `eval.py`: validation on FLSea splits
- `models/depth_anything_lora.py`: minimal LoRA + AquaDegrade model
- `dataset/flsea.py`: FLSea loader from split text files
- `depth_anything_v2/`: copied from `Depth-Anything-V2/metric_depth`

## Expected FLSea split format

Each line in `train-list` / `val-list` should contain:

```text
/abs/path/to/image.png /abs/path/to/depth.npy
```

If your depth is stored as `png/tiff`, the loader also supports it. For 16-bit depth images it will divide by `1000.0`.

## Recommended baseline

The provided shell script reproduces the intended clean starting point:

- `aquadegrade`
- `depthdive_relative`
- `consistency hardness`
- `decoder tail ConvLoRA`

This is meant to be the new stable base before adding:

1. optional sparse-prior injection
2. transmission / medium branch
3. any teacher or distillation design

