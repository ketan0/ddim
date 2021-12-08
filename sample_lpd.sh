#!/usr/bin/env bash
set -euo pipefail
DATASET="lpd"
PROJECT_PATH="/home/ketanagrawal/ddim"
MODEL_NAME="lpd_improved"
STEPS=100
ETA=0
# python main.py --config ${DATASET}.yml --exp ${PROJECT_PATH} --doc ${MODEL_NAME} --sample --sequence --timesteps ${STEPS} --eta ${ETA}
python main.py --config ${DATASET}.yml --exp ${PROJECT_PATH} --doc ${MODEL_NAME} --sample --fid --timesteps ${STEPS} --eta ${ETA}
