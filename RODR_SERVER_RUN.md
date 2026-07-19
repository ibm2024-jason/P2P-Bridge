# RODR Plug-and-Play Server Run

This branch keeps baseline MSE and RODR in the same code path. Use the scripts
below to set up the environment, start both trainings in the background on two
GPUs, and evaluate both checkpoints.

## 1. Environment

```bash
cd P2P-Bridge
./scripts/run_env_setup.sh
conda activate p2pb
```

Optional: set `WITH_DATA=1` to download P2P object data during setup:

```bash
WITH_DATA=1 ./scripts/run_env_setup.sh
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

## 3. Start Two Background Trainings On Two GPUs

Run baseline on GPU 0 and RODR on GPU 1 with P2P's PUNet data:

```bash
STEPS=2000 BS=8 ./scripts/run_train_pair_bg.sh punet 0 1 experiments/rodr_compare_punet
```

Run baseline on GPU 2 and RODR on GPU 3 with starter-code data:

```bash
STEPS=2000 BS=8 ./scripts/run_train_pair_bg.sh starter 2 3 experiments/rodr_compare_starter
```

Tune RODR tangent weight:

```bash
RODR_WEIGHT=1.5 STEPS=2000 ./scripts/run_train_pair_bg.sh punet 0 1 experiments/rodr_compare_punet_w15
```

Check status and logs:

```bash
./scripts/run_train_status.sh experiments/rodr_compare_punet
```

## 4. Evaluate

Evaluate PUNet checkpoints:

```bash
STEP=2000 ./scripts/run_eval_pair.sh punet 0 experiments/rodr_compare_punet
```

Evaluate starter checkpoints:

```bash
STEP=2000 ./scripts/run_eval_pair.sh starter 0 experiments/rodr_compare_starter
```

## Notes

- `punet` uses P2P's native dataloader and `evaluate_objects.py`.
- `starter` trains from `starter_code/dataset_train` meshes and validates on
  `starter_code/local/clean` plus `starter_code/local/noisy`.
- `run_train_pair_bg.sh <dataset> <baseline_gpu> <rodr_gpu> <save_root>` starts
  both trainings with `nohup`-style background jobs and writes PID/log files.
- `run_eval_pair.sh <dataset> <gpu> <save_root>` evaluates both checkpoints.
