"""Revision: four-group Kruskal-Wallis test on model-level rho.
Reviewer point: separation test only compared CLIP (n=3) vs SigLIP (n=3);
add a test including all four encoder groups (CLIP, SigLIP, QwenViT, Qwen2-VL-ViT).
Also recompute exact separation p-values for pairwise CLIP-vs-SigLIP.

Inputs: data/visualanchor_snapshot upstream JSONs + the three committed
LLaVA HF eval results (llava_hf_*_scb5.json).
Output: results/kruskal_4groups.json
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import kruskal, spearmanr

BASE = Path(__file__).resolve().parents[1]
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

from sharedvisual.paths import RESULTS_DIR
from sharedvisual.datasets import CLASS_MAP, load_anchor_scores, load_mllm_full

REV_RESULTS = RESULTS_DIR

GROUPS = {
    "CLIP": ["LLaVA-1.5-7B", "LLaVA-1.5-13B", "LLaVA-NeXT-Mistral-7B",
             "LLaVA-NeXT-Vicuna-7B"],
    "SigLIP": ["Gemma4-31B", "Gemma4-26B", "Gemma-3-27B"],
    "QwenViT": ["Qwen3.5-27B", "Qwen3.6-27B", "Qwen3.5-35B", "Qwen3.6-35B"],
    "Qwen2-VL-ViT": ["Qwen2-VL-7B"],
}


def load_new_llava(mllm_full, json_name, mname):
    """Merge a committed HF eval JSON into mllm_full under _hf_extra."""
    d = json.load(open(REV_RESULTS / json_name))
    row = {}
    for _, per_class in d.items():
        if not isinstance(per_class, dict):
            continue
        for cls, v in per_class.items():
            if isinstance(v, dict):
                v = v.get("acc", 0)
            row[cls] = v
    mllm_full.setdefault("_hf_extra", {})[mname] = row


MLLM_FULL = load_mllm_full()
for jf, mn in [("llava_hf_llava_1.5_13b_hf_scb5.json", "LLaVA-1.5-13B"),
               ("llava_hf_llava_v1.6_mistral_7b_hf_scb5.json", "LLaVA-NeXT-Mistral-7B"),
               ("llava_hf_llava_v1.6_vicuna_7b_hf_scb5.json", "LLaVA-NeXT-Vicuna-7B")]:
    load_new_llava(MLLM_FULL, jf, mn)

anchor_scores = load_anchor_scores()

model_rhos = {}
for mname in set(sum(GROUPS.values(), [])):
    pairs = []
    for dset, classes in CLASS_MAP.items():
        pc = MLLM_FULL.get(dset, {}).get(mname)
        if pc is None:
            pc = MLLM_FULL.get("_hf_extra", {}).get(mname)
        if pc is None:
            continue
        for cls in classes:
            v = pc.get(cls)
            if v is not None:
                pairs.append((anchor_scores[(dset, cls)], v))
    if len(pairs) >= 3:
        a = np.array([p[0] for p in pairs])
        y = np.array([p[1] for p in pairs])
        rho, p = spearmanr(a, y)
        model_rhos[mname] = {"rho": round(rho, 4), "p": round(p, 4), "n": len(pairs)}

print("Per-model rho (recomputed):")
for m, v in sorted(model_rhos.items(), key=lambda kv: -kv[1]["rho"]):
    print(f"  {m:24s} rho={v['rho']:+.4f}  p={v['p']:.4f}  n={v['n']}")

group_rhos = {g: [model_rhos[m]["rho"] for m in ms] for g, ms in GROUPS.items()}
for g, rhos in group_rhos.items():
    print(f"{g:14s} n={len(rhos)} mean={np.mean(rhos):.3f} individual={[round(r,3) for r in rhos]}")

stat, p = kruskal(*group_rhos.values())
print(f"\nKruskal-Wallis (4 groups): H={stat:.3f}, p={p:.4f}")

# exact separation test CLIP vs SigLIP (two-sided: probability that all of one
# group's values rank entirely above or below the other's under random assignment)
clip = group_rhos["CLIP"]
sig = group_rhos["SigLIP"]
n = len(clip) + len(sig)
exact = 2.0 / __import__("math").comb(n, len(clip))
print(f"CLIP vs SigLIP: max(CLIP)={max(clip):.3f} < min(SigLIP)={min(sig):.3f}; "
      f"exact p=2/C({n},{len(clip)})={exact:.4f}")

# CLIP vs QwenViT (Mann-Whitney, small-sample exact via permutation)
qwen = group_rhos["QwenViT"]
rng = np.random.default_rng(20260801)
obs = np.abs(np.mean(clip) - np.mean(qwen))
n_perm = 100000
combined = np.array(clip + qwen)
n_clip, n_qwen = len(clip), len(qwen)
cnt = 0
for _ in range(n_perm):
    perm = rng.permutation(combined)
    if np.abs(np.mean(perm[:n_clip]) - np.mean(perm[n_clip:])) >= obs - 1e-9:
        cnt += 1
print(f"CLIP vs QwenViT: mean {np.mean(clip):.3f} vs {np.mean(qwen):.3f}, "
      f"permutation p={cnt / n_perm:.4f}")

# also report Qwen2-VL-ViT value for completeness
print(f"Qwen2-VL-ViT single model rho = {group_rhos['Qwen2-VL-ViT'][0]:.3f}")

out = {
    "per_model_rho": model_rhos,
    "groups": {g: [round(r, 4) for r in rhos] for g, rhos in group_rhos.items()},
    "kruskal_4_groups": {"H": round(float(stat), 4), "p": round(float(p), 4)},
    "clip_vs_siglip": {
        "max_clip": round(float(max(clip)), 4),
        "min_siglip": round(float(min(sig)), 4),
        "exact_p": round(float(exact), 4),
    },
    "clip_vs_qwenvit_permutation": {
        "mean_clip": round(float(np.mean(clip)), 4),
        "mean_qwenvit": round(float(np.mean(qwen)), 4),
        "p": round(cnt / n_perm, 4),
    },
}
Path(BASE / "results" ).mkdir(exist_ok=True)
with open(BASE / "results" / "kruskal_4groups.json", "w") as f:
    json.dump(out, f, indent=1)
print("\nSaved results/kruskal_4groups.json")
