#!/bin/bash
# Runner for exp_flowers102_difficulty.py on 5090 (lizhuo@115.190.153.37).
# Usage: bash run_flowers102_difficulty.sh  (run under nohup on the server)
set -e
cd ~/Work/Mayan/SharedVisual
source ~/miniconda3/etc/profile.d/conda.sh
conda activate llava

# Model cache dir on 5090
export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
# allow HF downloads on first run (server has network); script default is
# HF_HUB_OFFLINE=1 unless we override it here.
export HF_HUB_OFFLINE=0

mkdir -p results
echo "[$(date '+%F %T')] starting"
nohup python -u experiments/exp_flowers102_difficulty.py \
    > /tmp/flowers102_difficulty.log 2>&1 &
echo "[$(date '+%F %T')] launched PID $! log=/tmp/flowers102_difficulty.log"