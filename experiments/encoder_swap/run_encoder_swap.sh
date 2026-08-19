#!/usr/bin/env bash
# Encoder-swap orchestration (V100x4). Remote-execution discipline: background +
# full logs + resumable stages; poll with short ssh tails only.
# Usage: bash run_encoder_swap.sh [features|train|eval|analysis|all]
# NOT for foreground use over ssh — launch via: nohup bash run_encoder_swap.sh all &
set -uo pipefail
cd "$(dirname "$0")/../.."
export PYTHONUNBUFFERED=1
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export TOKENIZERS_PARALLELISM=false
PY="${SHAREDVISUAL_PY:-python3}"
LOG=results/encoder_swap
mkdir -p "$LOG"

stage="${1:-all}"

if [[ "$stage" == "features" || "$stage" == "all" ]]; then
  echo "[encoder-swap] stage features $(date -Is)"
  CUDA_VISIBLE_DEVICES=0 $PY experiments/encoder_swap/01_extract_features.py \
      > "$LOG/01_features.log" 2>&1 || echo "[encoder-swap] features FAILED — see log"
fi

if [[ "$stage" == "train" || "$stage" == "all" ]]; then
  echo "[encoder-swap] stage train $(date -Is)"
  CUDA_VISIBLE_DEVICES=0 $PY experiments/encoder_swap/02_train_projector.py \
      --arms clip_l_laion   > "$LOG/02_train_C.log" 2>&1 &
  CUDA_VISIBLE_DEVICES=1 $PY experiments/encoder_swap/02_train_projector.py \
      --arms siglip_so400m  > "$LOG/02_train_S.log" 2>&1 &
  CUDA_VISIBLE_DEVICES=2 $PY experiments/encoder_swap/02_train_projector.py \
      --arms dinov2_b14     > "$LOG/02_train_D.log" 2>&1 &
  CUDA_VISIBLE_DEVICES=3 $PY experiments/encoder_swap/02_train_projector.py \
      --arms random_vitb    > "$LOG/02_train_R.log" 2>&1 &
  wait
fi

if [[ "$stage" == "eval" || "$stage" == "all" ]]; then
  echo "[encoder-swap] stage eval $(date -Is)"
  for i in 0 1 2 3; do
    arm=$(echo clip_l_laion siglip_so400m dinov2_b14 random_vitb | cut -d' ' -f$((i+1)))
    CUDA_VISIBLE_DEVICES=$i $PY experiments/encoder_swap/03_eval_swap_variants.py \
        --arms "$arm" > "$LOG/03_eval_${arm}.log" 2>&1 &
  done
  wait
fi

if [[ "$stage" == "analysis" || "$stage" == "all" ]]; then
  echo "[encoder-swap] stage analysis $(date -Is)"
  $PY experiments/encoder_swap/analysis_encoder_swap.py > "$LOG/04_analysis.log" 2>&1
fi

echo "[encoder-swap] stage '$stage' done $(date -Is). Logs in $LOG/"
