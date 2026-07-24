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

### Unified raw-disparity visualization

Every `eval_flsea_*.sh` entry now uses the same visualization protocol for the
baseline, latent-prior variants, capacity controls, consistency model, and LoRA:

- save the dense raw output returned by `model.infer_image()`
- never use GT alignment, inverse-depth conversion, depth caps, or GT masks
- normalize each raw disparity map with its own finite min/max
- apply the `Spectral_r` colormap: near/high disparity is red and far/low disparity is purple
- save `*_raw_disp.npy` and `*_raw_disp.png` under `<save-dir>/raw_disparity`

The PNG is intended for relative geometric-shape comparison. The NPY retains the
actual model output and must be used when numerical output differences matter.
Metric computation remains unchanged and still uses disparity scale-shift
alignment against valid FLSea GT pixels.

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

### Legacy first-round structure ablation

These commands document the completed first round. Do not rerun `global_only`,
`full`, `no_fft`, or `no_deg_map` for final reporting: their controls were later
found invalid and are superseded by the fixed second-round scripts below.

The core latent-prior ablation keeps the backbone, original DPT head, loss, data split, and optimizer settings fixed. Only the enabled prior structure changes:

```bash
bash scripts/train_flsea_latent_prior_ablation.sh global_only
bash scripts/train_flsea_latent_prior_ablation.sh local_only
bash scripts/train_flsea_latent_prior_ablation.sh no_fft
bash scripts/train_flsea_latent_prior_ablation.sh full
```

Each run uses five epochs and writes to `runs/ablation_<variant>`. Evaluate the selected `best_abs_rel.pth` on the fixed test split with:

```bash
bash scripts/eval_flsea_latent_prior_ablation.sh global_only
bash scripts/eval_flsea_latent_prior_ablation.sh local_only
bash scripts/eval_flsea_latent_prior_ablation.sh no_fft
bash scripts/eval_flsea_latent_prior_ablation.sh full
```

### Research-one experiment matrix

All primary runs use the same checkpoint, FLSea train/validation split, five epochs,
relative-depth loss, FP32, seed 42, and legacy disparity scale-shift evaluation.

| Experiment | Command | Trainable parameters | Purpose |
| --- | --- | --- | --- |
| B0 Original DA V2 | `bash scripts/eval_flsea_baseline_legacy.sh` | none | frozen reference |
| B1 Decoder FT | `bash scripts/train_flsea_capacity_control.sh decoder_ft` | original DPT decoder | generic decoder adaptation |
| B2 Conv Adapter | `bash scripts/train_flsea_capacity_control.sh conv_adapter` | four plain residual adapters | parameter-matched capacity control |
| B3 AquaDegrade LoRA | `bash scripts/train_flsea_aquadegrade_lora_control.sh` | encoder/decoder LoRA and degradation encoder | previous method control |
| A1 Global only | `bash scripts/train_flsea_latent_prior_ablation.sh global_only` | global latent branch | global degradation descriptor |
| A2 Local only | `bash scripts/train_flsea_latent_prior_ablation.sh local_only` | local prior and deg-map branch | spatial degradation prior |
| A3 No FFT | `bash scripts/train_flsea_latent_prior_ablation.sh no_fft` | full prior except FFT branch | frequency prior contribution |
| A4 No deg map (invalid first-round control) | `bash scripts/train_flsea_latent_prior_ablation.sh no_deg_map` | constant-one gate | superseded by scalar and spatial-mean controls |
| A5 Full | `bash scripts/train_flsea_latent_prior_ablation.sh full` | latent encoder and prior injection | proposed core model |
| L1 Full + consistency | `bash scripts/train_flsea_latent_prior_consistency.sh` | same as A5 | loss contribution after structure is fixed |
| L2 Full + decoder | `bash scripts/train_flsea_capacity_control.sh full_decoder` | A5 plus original decoder | capacity upper bound, not the fair core comparison |

Run the primary structure experiments first in this order:

```bash
bash scripts/train_flsea_capacity_control.sh decoder_ft
bash scripts/train_flsea_capacity_control.sh conv_adapter
bash scripts/train_flsea_latent_prior_ablation.sh global_only
bash scripts/train_flsea_latent_prior_ablation.sh local_only
bash scripts/train_flsea_latent_prior_ablation.sh no_fft
bash scripts/train_flsea_latent_prior_ablation.sh full
```

Only after A5 is stable, run the loss and upper-bound experiments:

```bash
bash scripts/train_flsea_latent_prior_consistency.sh
bash scripts/train_flsea_capacity_control.sh full_decoder
```

Test-set evaluation uses the matching `eval_flsea_*` scripts. Each script loads
`best_abs_rel.pth`; do not select a checkpoint using test-set results.

For B3, use `bash scripts/eval_flsea_aquadegrade_lora_control.sh`; it deliberately
replaces the old loader-based validation with the same original-resolution legacy
alignment protocol used by every other row.

### Fixed second-round ablation

> Historical note: this matrix predates bounded prior injection, raw-disparity
> gauge anchoring, deterministic loaders, and local spectral degradation maps.
> Keep its logs only for debugging; do not use it for final reporting.

The first round exposed two invalid controls: the global FiLM MLP had all Linear
weights initialized to zero, and `no_deg_map` used a constant-one gate with a
larger effective injection magnitude. Both are fixed in the current model.

Run the corrected single-seed diagnostic matrix without overwriting old results:

```bash
bash scripts/train_flsea_fixed_ablation.sh global_only
bash scripts/train_flsea_fixed_ablation.sh full
bash scripts/train_flsea_fixed_ablation.sh no_fft
bash scripts/train_flsea_fixed_ablation.sh local_scalar
bash scripts/train_flsea_fixed_ablation.sh local_spatial_mean
bash scripts/train_flsea_fixed_ablation.sh local_consistency
```

`local_scalar` learns one scalar gate per scale, initialized to `0.5`.
`local_spatial_mean` retains the generated map's exact per-image, per-scale mean
but removes all spatial variation. Compare both against `local_only` to isolate
the value of explicit spatial degradation localization.

After selecting the structure using validation only, run three seeds:

```bash
for seed in 42 123 3407; do
  SEED=${seed} bash scripts/train_flsea_fixed_ablation.sh local_only
done
```

Matching evaluation example:

```bash
SEED=42 bash scripts/eval_flsea_fixed_ablation.sh local_only
```

### Current latent-prior ablation

The current implementation adapts DAIR's which-where-what reasoning to monocular
depth without replacing the pretrained DA2 geometry pathway:

- **which:** a multi-scale convolutional latent encoder provides stage-wise priors;
- **where:** magnitude/phase FFT cues and latent priors form spatial maps through
  linear element-wise attention;
- **what:** a global descriptor applies identity-anchored, bounded modulation at
  the decoder bottleneck;
- all prior corrections are zero-initialized and bounded, while a raw-disparity
  mean/std anchor prevents affine-invariant losses from drifting the output gauge.

This is a task-driven adaptation, not a reproduction of DAIR's separately
pretrained VAE. DA2's backbone and original DPT decoder remain frozen in the fair
structure ablation.

Run one-epoch smoke tests first:

```bash
EPOCHS=1 RUN_TAG=smoke bash scripts/train_flsea_latent_prior_clean_ablation.sh local_only
EPOCHS=1 RUN_TAG=smoke bash scripts/train_flsea_latent_prior_clean_ablation.sh local_spectral
EPOCHS=1 RUN_TAG=smoke bash scripts/train_flsea_latent_prior_clean_ablation.sh full
```

Then run the five-epoch matrix with a fresh tag:

```bash
RUN_TAG=main_seed42 bash scripts/train_flsea_latent_prior_clean_ablation.sh local_only
RUN_TAG=main_seed42 bash scripts/train_flsea_latent_prior_clean_ablation.sh local_spectral
RUN_TAG=main_seed42 bash scripts/train_flsea_latent_prior_clean_ablation.sh no_fft
RUN_TAG=main_seed42 bash scripts/train_flsea_latent_prior_clean_ablation.sh full
```

`local_spectral` isolates local frequency-aware where-reasoning; `no_fft`
isolates spatial/global latent reasoning without frequency cues; `full` combines
both. Use the matching evaluation entrypoint, for example:

```bash
RUN_TAG=main_seed42 bash scripts/eval_flsea_latent_prior_clean_ablation.sh full
```

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

### Wat3R privileged multi-view distillation

Wat3R is an offline training-only teacher. Each 24-frame package keeps depth,
confidence, intrinsics, and extrinsics from the same forward pass so that its
geometry gauge is internally consistent. The DA2 student still processes every
view independently and remains single-image-only at inference.

Generate one scene first as a pipeline pilot:

```bash
SCENE=flatiron \
IMAGE_DIR=/data1/hxy/DATASET/FLSeaVI/canyons/flatiron/imgs \
WAT3R_ROOT=/data1/hxy/Wat3R \
WAT3R_CKPT=/data1/hxy/Wat3R/checkpoints/wat3r.pth \
bash scripts/generate_flsea_wat3r_teacher.sh
```

For formal training, generate every scene represented in the training split:

```bash
TRAIN_LIST=/data1/hxy/DPV2_prompt_fusion/dataset/splits/flsea/train_half.txt \
DATASET_ROOT=/data1/hxy/DATASET/FLSeaVI \
WAT3R_ROOT=/data1/hxy/Wat3R \
WAT3R_CKPT=/data1/hxy/Wat3R/checkpoints/wat3r.pth \
OUTPUT_ROOT=/data1/hxy/flsea_wat3r_train_teacher \
bash scripts/generate_flsea_wat3r_all_train.sh
```

The all-scene command groups the training list by scene, sorts each scene by
frame name, generates its windows, and writes `manifest_all.csv` automatically.
It never scans validation/test RGB frames. Existing scene manifests are reused,
so the command can resume after interruption; pass `--overwrite` only when a
scene must be regenerated.

Merge completed scene manifests:

```bash
python tools/merge_wat3r_manifests.py \
  --output /data1/hxy/flsea_wat3r_teacher/manifest_all.csv \
  /data1/hxy/flsea_wat3r_teacher/*/manifest.csv
```

Audit paths, matching triplets, and static-mask coverage before training:

```bash
python tools/audit_wat3r_teacher.py \
  --manifest /data1/hxy/flsea_wat3r_train_teacher/manifest_all.csv \
  --train-list /data1/hxy/DPV2_prompt_fusion/dataset/splits/flsea/train_half.txt
```

Do not train directly from the raw Wat3R manifest. Build cross-window agreement
masks first. They align the relative disparities of repeated predictions for
the same frame and retain only pixels supported by at least two overlapping
windows.

For the completed `flatiron` pilot:

```bash
TRAIN_LIST=/data1/hxy/DPV2_prompt_fusion/dataset/splits/flsea/train.txt \
bash scripts/prepare_flsea_wat3r_pilot.sh
```

This writes `canyons/flatiron/manifest_overlap.csv`. Run a one-epoch,
single-scene smoke test before spending time on every scene:

```bash
INIT_FROM=runs/clean_hybrid_lora_aqua_seed42/best_abs_rel.pth \
TRAIN_LIST=/data1/hxy/DPV2_prompt_fusion/dataset/splits/flsea/train.txt \
bash scripts/train_flsea_wat3r_pilot.sh
```

After every training scene has been generated, build and audit the formal
all-scene overlap manifest:

```bash
TRAIN_LIST=/data1/hxy/DPV2_prompt_fusion/dataset/splits/flsea/train.txt \
bash scripts/prepare_flsea_wat3r_all.sh
```

The final teacher reliability rule is finite depth intersected with Wat3R
static consistency, the selected confidence quantile, cross-window agreement,
and the FLSea GT hole. Frames seen by only one window receive an empty overlap
mask instead of unverified pseudo-labels.

Start stage two from the matching research-one checkpoint:

```bash
INIT_FROM=runs/clean_hybrid_lora_aqua_seed42/best_abs_rel.pth \
WAT3R_MANIFEST=/data1/hxy/flsea_wat3r_train_teacher/manifest_all_overlap.csv \
bash scripts/train_flsea_wat3r_distill.sh
```

Supervision precedence is fixed: FLSea GT on observed pixels, reliable Wat3R
depth only in holes, multi-view reprojection only on static consistent pixels,
and frozen DA2 gradient preservation everywhere else in holes. Wat3R files are
never required by evaluation or inference.
