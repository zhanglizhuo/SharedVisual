#!/usr/bin/env bash
# Run HF eval for 2 CLIP-encoder MLLMs on SCB5, sequentially, in background.
# Run:  nohup bash run_hf_eval.sh > /tmp/run_hf_eval.log 2>&1 &
set -uo pipefail
PY=$HOME/miniconda3/envs/llava/bin/python
SCRIPT=$HOME/eval_hf.py
DATA=/tmp/scb5
OUT=$HOME/hf_eval_results
mkdir -p $OUT

export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DISABLE_XET=1

MODELS=(
    "llava-hf/llava-1.5-13b-hf"
    "llava-hf/llava-v1.6-mistral-7b-hf"
)

echo "=== [$(date)] start run_hf_eval.sh ==="

for MODEL in "${MODELS[@]}"; do
    SAFE=$(echo "$MODEL" | tr '/' '_')
    echo ""
    echo "=== [$(date)] MODEL: $MODEL ==="
    $PY $SCRIPT "$MODEL" --data-dir "$DATA" 2>&1 | tee "$OUT/${SAFE}.log" | tail -80
    # Move output JSON into OUT
    if [ -f "/tmp/${SAFE}_scb5.json" ]; then
        mv "/tmp/${SAFE}_scb5.json" "$OUT/"
        echo "[ok] saved $OUT/${SAFE}_scb5.json"
    fi
    echo "=== [$(date)] DONE: $MODEL ==="
done

echo ""
echo "=== [$(date)] ALL DONE ==="
ls -la $OUT/
