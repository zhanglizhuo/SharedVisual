#!/bin/bash
# Run 3 CLIP-encoder MLLMs on SCB5, sequentially.
# Usage (on V100): bash run_all_clip.sh
set -euo pipefail
SCRIPT_DIR="$HOME/eval_clip_mllms"
RESULT_DIR="$SCRIPT_DIR/results"
mkdir -p "$RESULT_DIR"
LOGDIR="$SCRIPT_DIR/logs"
mkdir -p "$LOGDIR"

MODELS=("llava:13b" "bakllava:7b" "llava-llama3:8b")

for MODEL in "${MODELS[@]}"; do
    SAFE=$(echo "$MODEL" | tr ':' '_')
    LOG="$LOGDIR/${SAFE}.log"
    echo "==== $(date) Starting $MODEL ====" | tee -a "$LOG"
    python3 "$SCRIPT_DIR/eval_clip_mllms_v100.py" "$MODEL" >> "$LOG" 2>&1
    # Move output to results
    if [ -f "$SCRIPT_DIR/${SAFE}_scb5.json" ]; then
        mv "$SCRIPT_DIR/${SAFE}_scb5.json" "$RESULT_DIR/"
    fi
    echo "==== $(date) Done $MODEL ====" | tee -a "$LOG"
done
echo "==== $(date) ALL DONE ===="
