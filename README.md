# Underwater DA Clean Baseline

This project is a cleaned baseline extracted from `DA_16_lora_feature_distill_adapter`.

It keeps only the core path that is still worth building on:

- `Depth Anything V2` backbone as the controllable base model
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
- `eval_baseline_legacy.py`: verified DA_0-compatible baseline evaluation
- `train_latent_prior.py`: latent-prior training entry
- `eval_latent_prior.py`: latent-prior validation entry
- `models/depth_anything_lora.py`: minimal LoRA + AquaDegrade model
- `models/depth_anything_latent_prior.py`: independent latent-prior depth model
- `models/underwater_latent_prior.py`: latent prior encoder + explicit deg-map generators
- `dataset/flsea.py`: FLSea loader from split text files
- `depth_anything_v2/`: synced with the verified `DA_0` baseline package used for FLSea reproduction

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

## Evaluation

Use these two scripts as the only primary evaluation entrypoints:

- `scripts/eval_flsea_baseline_legacy.sh`:
  verified baseline reproduction on FLSea with the same protocol as the strong `DA_0` result.
- `scripts/eval_flsea_latent_prior.sh`:
  latent-prior evaluation under the exact same FLSea protocol, so its numbers are directly comparable to the baseline above.

For research-one experiments, compare every model against `eval_flsea_baseline_legacy.sh`, then report the matching latent-prior result from `eval_flsea_latent_prior.sh`.

## Server Run

The training script is now server-oriented and expects explicit dataset and checkpoint paths.

Set the three required variables first:

```bash
export CKPT=/path/to/depth_anything_v2_vits.pth
export TRAIN_LIST=/path/to/flsea_train.txt
export VAL_LIST=/path/to/flsea_val.txt
```

Then launch:

```bash
bash scripts/train_flsea_baseline.sh
```

## Latent Prior Branch

The new latent-prior branch is separate from the LoRA baseline and does not modify `depth_anything_lora.py`.

Recommended first run:

```bash
export CKPT=/path/to/depth_anything_v2_vits.pth
export TRAIN_LIST=/path/to/flsea_train.txt
export VAL_LIST=/path/to/flsea_val.txt
bash scripts/train_flsea_latent_prior.sh
```

Default behavior of the latent-prior script:

- freezes the DINOv2 backbone first
- freezes the verified standard DPT depth head for the first ablation
- trains `UnderwaterLatentPriorEncoder` plus the new deg-map/global-modulation path
- uses FP32 and gradient clipping by default for numerical stability
- evaluates every epoch with the same original-resolution disparity-alignment protocol as the verified baseline
- saves full checkpoints because this branch is no longer LoRA-only

Recommended first-round research-one rule:

- keep `CONSISTENCY_HARDNESS_WEIGHT=0.0`
- keep `CONSISTENCY_AUG_PROB=0.0`
- freeze both the backbone and original DPT decoder
- do not add encoder/decoder LoRA until the latent-prior-only ablation is complete
- compare only against `scripts/eval_flsea_baseline_legacy.sh`
- use `scripts/eval_flsea_latent_prior.sh` for the final latent-prior number under the same FLSea protocol

Optional overrides:

```bash
export EXP_NAME=baseline_seed42
export BATCH_SIZE=2
export EPOCHS=1
export PYTHON_BIN=/path/to/python
```

If you want a fully explicit one-line command instead of environment variables:

```bash
CKPT=/path/to/depth_anything_v2_vits.pth \
TRAIN_LIST=/path/to/flsea_train.txt \
VAL_LIST=/path/to/flsea_val.txt \
EXP_NAME=baseline_seed42 \
bash scripts/train_flsea_baseline.sh
```
