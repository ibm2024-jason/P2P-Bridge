# RODR Plug-and-Play Server Run

This branch keeps baseline MSE and RODR in the same code path. Use the same
script to run both experiments sequentially on a selected GPU.

## 1. Environment

```bash
cd P2P-Bridge
./scripts/setup_p2pb_env.sh
conda activate p2pb
sh install.sh
```

## 2. Download P2P Object Data

P2P-Bridge uses the ScoreDenoise object datasets for PUNet/PCNet. Download them
directly on the server into `data/objects`:

```bash
cd P2P-Bridge
./scripts/download_scored_objects.sh data/objects
```

Expected layout:

```text
data/objects/
├── examples/
├── PCNet/
└── PUNet/
```

If Google Drive blocks automated download, download the same ScoreDenoise files
manually and extract them into `data/objects` with the layout above.

## 3. Train Baseline and RODR

Run on GPU 0 with P2P's PUNet data:

```bash
STEPS=2000 BS=8 ./scripts/train_rodr_compare.sh punet 0 experiments/rodr_compare_punet
```

Run on GPU 1 with starter-code data:

```bash
STEPS=2000 BS=8 ./scripts/train_rodr_compare.sh starter 1 experiments/rodr_compare_starter
```

Tune RODR tangent weight:

```bash
RODR_WEIGHT=1.5 STEPS=2000 ./scripts/train_rodr_compare.sh punet 0 experiments/rodr_compare_punet_w15
```

## 4. Evaluate

Evaluate PUNet checkpoints:

```bash
STEP=2000 ./scripts/eval_rodr_compare.sh punet 0 experiments/rodr_compare_punet
```

Evaluate starter checkpoints:

```bash
STEP=2000 ./scripts/eval_rodr_compare.sh starter 0 experiments/rodr_compare_starter
```

## Notes

- `punet` uses P2P's native dataloader and `evaluate_objects.py`.
- `starter` trains from `starter_code/dataset_train` meshes and validates on
  `starter_code/local/clean` plus `starter_code/local/noisy`.
- `CUDA_VISIBLE_DEVICES` is set by the second argument of the train/eval scripts.
