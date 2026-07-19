#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${ENV_NAME:-p2pb}"

conda create -n "${ENV_NAME}" python=3.10 -y
conda run -n "${ENV_NAME}" conda install pytorch==2.1.2 torchvision==0.16.2 pytorch-cuda=11.8 -c pytorch -c nvidia --yes
conda run -n "${ENV_NAME}" python -m pip install -r requirements.txt

echo "Environment ${ENV_NAME} is ready for the Python dependencies."
echo "Activate it with: conda activate ${ENV_NAME}"
echo "Then run sh install.sh inside P2P-Bridge to compile CUDA metric/point ops."
