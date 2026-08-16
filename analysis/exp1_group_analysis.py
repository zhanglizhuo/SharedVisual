#!/usr/bin/env python3
"""
Exp 1 (Approach C): Natural MLLM Groups by Encoder Family
===========================================================
Question: Do MLLMs grouped by visual encoder family show different
ρ with CLIP AnchorScore?

If ρ is similar across groups → shared vision encoder is NOT the mechanism.
If CLIP-encoder group shows higher ρ → supports shared vision hypothesis.

Groups:
  CLIP encoder:   LLaVA-1.5-7B, LLaVA-NeXT-Mistral-7B
  SigLIP encoder: Gemma-3-27B, Gemma4-31B, Gemma4-26B
  QwenViT:        Qwen3.5-27B, Qwen3.6-27B, Qwen3.5-35B, Qwen3.6-35B
  Qwen2-VL:       Qwen2-VL-7B

Uses VA's existing MLLM accuracy data. No GPU needed.
"""
import json
import numpy as np
from pathlib import Path
from scipy.stats import spearmanr

import sys
BASE = Path(__file__).resolve().parents[1]
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))
from sharedvisual.paths import va_results as _va
VA_RESULTS = _va()

# Load AnchorScore (SCB5 per-class)
with open(VA_RESULTS / "01_core" / "anchor_score_scb5" / "anchor_scores.json") as f:
    anchor = json.load(f)

# Load MLLM accuracy
with open(VA_RESULTS / "01_core" / "paper_data" / "mllm_full.json") as f:
    mllm = json.load(f)

CLASS_MAP = {
    "TeacherBehavior": [
        "guide", "answer", "On-stage interaction", "blackboard-writing",
        "teacher", "stand", "screen", "blackBoard",
    ],
    "HandriseReadWrite": ["hand-raising", "read", "write"],
    "BowTurnHead": ["BowHead", "TurnHead"],
}

ENCODER_GROUPS = {
    "CLIP ViT-L/14": {
        "models": ["LLaVA-1.5-7B"],
        "hypothesis": "shared CLIP encoder → higher ρ expected",
    },
    "SigLIP": {
        "models": ["Gemma4-31B", "Gemma4-26B", "Gemma-3-27B"],
        "hypothesis": "SigLIP encoder → lower ρ than CLIP group (if shared vision matters)",
    },
    "QwenViT (自研)": {
        "models": ["Qwen3.5-27B", "Qwen3.6-27B", "Qwen3.5-35B", "Qwen3.6-35B"],
        "hypothesis": "proprietary encoder → intermediate ρ",
    },
    "Qwen2-VL ViT": {
        "models": ["Qwen2-VL-7B"],
        "hypothesis": "smaller model, different encoder → low ρ",
    },
}

def get_anchor_scores():
    scores = {}
    for dataset, classes in CLASS_MAP.items():
        for cls in classes:
            acc = anchor[dataset]["per_class_acc"][cls]["acc"]
            scores[(dataset, cls)] = acc
    return scores

anchor_scores = get_anchor_scores()

print("=" * 70)
print("Exp 1 (Approach C): Natural MLLM Groups by Visual Encoder Family")
print("=" * 70)

for group_name, group_info in ENCODER_GROUPS.items():
    models = group_info["models"]
    rho_list = []
    all_accs = []

    for model in models:
        if model not in mllm.get("TeacherBehavior", {}):
            continue

        accs = []
        anchor_vals = []
        for dataset, classes in CLASS_MAP.items():
            for cls in classes:
                v = mllm.get(dataset, {}).get(model, {}).get(cls)
                if v is not None:
                    accs.append(v)
                    anchor_vals.append(anchor_scores[(dataset, cls)])

        if len(accs) >= 3:
            rho, p = spearmanr(anchor_vals, accs)
            rho_list.append(rho)
            all_accs.append(accs)

    mean_rho = np.mean(rho_list) if rho_list else float("nan")
    print(f"\nGroup: {group_name} ({len(models)} MLLMs)")
    print(f"  Individual ρ: {[f'{r:.3f}' for r in rho_list]}")
    print(f"  Mean ρ: {mean_rho:.3f}")
    print(f"  Hypothesis: {group_info['hypothesis']}")

print("\n" + "=" * 70)
print("CONCLUSION: Compare mean ρ across groups.")
print("If CLIP group >> SigLIP group → shared vision encoder is causal.")
print("If similar across groups → mechanism is not encoder-specific.")
print("=" * 70)
