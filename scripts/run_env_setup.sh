#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${ENV_NAME:-p2pb}"
WITH_DATA="${WITH_DATA:-0}"
DATASET_ROOT="${DATASET_ROOT:-data/objects}"

conda create -n "${ENV_NAME}" python=3.10 -y
conda run -n "${ENV_NAME}" conda install pytorch==2.1.2 torchvision==0.16.2 pytorch-cuda=11.8 -c pytorch -c nvidia --yes
conda run -n "${ENV_NAME}" python -m pip install -r requirements.txt

conda run -n "${ENV_NAME}" bash install.sh

if [ "${WITH_DATA}" = "1" ]; then
  conda run -n "${ENV_NAME}" ./scripts/download_scored_objects.sh "${DATASET_ROOT}"
fi

echo "Ready. Use: conda activate ${ENV_NAME}"
