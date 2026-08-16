#!/usr/bin/env bash
# Link B orchestration (V100x4). Per remote-execution discipline:
# background + full logs + resumable stages; poll with short ssh tails only.
# Each trainable arm gets one GPU; features extracted once on GPU 0.
# NOT YET RUN.
set -euo pipefail
cd "$(dirname "$0")/../.."
export PYTHONUNBUFFERED=1
LOG=results/linkb
mkdir -p "$LOG"

stage="${1:-all}"

if [[ "$stage" == "features" || "$stage" == "all" ]]; then
  CUDA_VISIBLE_DEVICES=0 python3 experiments/linkb/01_extract_features.py \
      > "$LOG/01_features.log" 2>&1
fi

if [[ "$stage" == "train" || "$stage" == "all" ]]; then
  # one arm per GPU, parallel, each idempotent/resumable
  CUDA_VISIBLE_DEVICES=0 nohup python3 experiments/linkb/02_train_projector.py \
      --arms clip_l_laion   > "$LOG/02_train_C.log" 2>&1 &
  CUDA_VISIBLE_DEVICES=1 nohup python3 experiments/linkb/02_train_projector.py \
      --arms siglip_so400m  > "$LOG/02_train_S.log" 2>&1 &
  CUDA_VISIBLE_DEVICES=2 nohup python3 experiments/linkb/02_train_projector.py \
      --arms dinov2_b14     > "$LOG/02_train_D.log" 2>&1 &
  CUDA_VISIBLE_DEVICES=3 nohup python3 experiments/linkb/02_train_projector.py \
      --arms random_vitb    > "$LOG/02_train_R.log" 2>&1 &
  wait
fi

if [[ "$stage" == "eval" || "$stage" == "all" ]]; then
  for i in 0 1 2 3; do
    arm=$(echo clip_l_laion siglip_so400m dinov2_b14 random_vitb | cut -d' ' -f$((i+1)))
    CUDA_VISIBLE_DEVICES=$i nohup python3 experiments/linkb/03_eval_swap_variants.py \
        --arms "$arm" > "$LOG/03_eval_${arm}.log" 2>&1 &
  done
  wait
fi

if [[ "$stage" == "analysis" || "$stage" == "all" ]]; then
  python3 experiments/linkb/analysis_linkb.py > "$LOG/04_analysis.log" 2>&1
fi

echo "Link B stage '$stage' complete. Logs in $LOG/"
