#!/bin/bash
# Re-extract Qwen/ResNet linear-probe features with blur applied (2026-08-18).
# The superseded probe entries in the raw JSON are stripped first so the
# checkpoint logic does not skip them.
set -e
cd ~/works/lizhuo/SharedVisual
source ~/miniconda3/bin/activate dl_env_310 || source ~/miniconda3/bin/activate dl_env
export SHAREDVISUAL_DATA=~/works/lizhuo/AutoResearchClaw/datasets_scb
export SHAREDVISUAL_RESULTS=~/works/lizhuo/SharedVisual/results

python - << 'PYEOF'
import json
p = 'results/visual_perturbation_sync_raw.json'
d = json.load(open(p))
removed = [d.pop(k, None) for k in ['Qwen2-VL Vision probe', 'ResNet-50 (IN-1k) probe']]
json.dump(d, open(p, 'w'), indent=2)
print('stale probe keys stripped:', [r is not None for r in removed])
PYEOF

echo "=== ResNet probe (blur applied) ==="
python experiments/exp_visual_perturbation_sync.py --only-encoder "ResNet-50 (IN-1k) probe"
echo "=== Qwen probe (blur applied) ==="
python experiments/exp_visual_perturbation_sync.py --only-encoder "Qwen2-VL Vision probe"
echo "ALL DONE"
