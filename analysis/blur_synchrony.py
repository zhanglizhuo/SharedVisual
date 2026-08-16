from pathlib import Path
"""Perturbation synchrony: gemma4:31b per-class blur drop vs CLIP per-class blur drop.

Direct geometry->behavior test: if SHARED confusability drives both models'
difficulty, then perturbing the shared visual signal (Gaussian blur) should
drop per-class accuracy in a correlated way across the two very different
models (an MLLM and a pure CLIP zero-shot classifier).

Inputs:
  results/exp_perturbation_blur.json  (gemma4:31b, sigmas 0..4)
  results/exp4_clip_blur.json                  (CLIP LAION L/14, sigmas 0,1,2,4,8)

Method (matching paper's Mantel/Spearman conventions):
  - common sigma grid {0,1,2,4}
  - per-class drop = acc(sigma=0) - acc(sigma=4) for each model
  - Spearman rho between gemma4 drops and CLIP drops across 8 classes
  - two-sided permutation p (10,000 shuffles, seed 42)
  - also report sigma-profile Spearman per class pair (concordance of full curves)

Output: results/blur_synchrony.json
"""
import json
import numpy as np
from scipy.stats import spearmanr, rankdata

BASE = Path(__file__).resolve().parents[1] / "results"
CLASSES = ["guide", "answer", "On-stage interaction", "blackboard-writing",
           "teacher", "stand", "screen", "blackBoard"]

gem = json.load(open(f"{BASE}/exp_perturbation_blur.json"))["results"]
clip = json.load(open(f"{BASE}/exp4_clip_blur.json"))

# CLIP uses percents and sigmas {0,1,2,4,8}; gemma4 uses fractions and {0..4}
clip_sig = {"0": "blur_sigma_0", "1": "blur_sigma_1", "2": "blur_sigma_2",
            "4": "blur_sigma_4"}
COMMON_SIGMAS = [0, 1, 2, 4]

clip_prof = {s: {c: clip[clip_sig[str(s)]][c] / 100.0 for c in CLASSES} for s in COMMON_SIGMAS}
gem_prof = {s: {c: gem[f"sigma_{s}"]["per_class_acc"][c] for c in CLASSES} for s in COMMON_SIGMAS}

# 1) end-to-end drop (sigma0 -> sigma4)
clip_drop = {c: clip_prof[0][c] - clip_prof[4][c] for c in CLASSES}
gem_drop = {c: gem_prof[0][c] - gem_prof[4][c] for c in CLASSES}

x = np.array([clip_drop[c] for c in CLASSES])
y = np.array([gem_drop[c] for c in CLASSES])
rho, p_asy = spearmanr(x, y)

# permutation test on rho (ties preserved by permuting y)
rng = np.random.default_rng(42)
perm = [spearmanr(x, rng.permutation(y)).statistic for _ in range(10000)]
p_perm = (1 + np.sum(np.abs(np.array(perm)) >= abs(rho))) / (1 + 10000)

# 2) full-profile concordance: per class, Spearman over the 4 sigmas between
#    gemma4 and CLIP curves -> per-class curve correlation; average as summary.
curve_rhos = {}
for c in CLASSES:
    g = [gem_prof[s][c] for s in COMMON_SIGMAS]
    k = [clip_prof[s][c] for s in COMMON_SIGMAS]
    curve_rhos[c] = float(spearmanr(g, k).statistic) if len(set(g)) > 1 and len(set(k)) > 1 else None
usable_curves = [v for v in curve_rhos.values() if v is not None]
mean_curve_rho = float(np.mean(usable_curves)) if usable_curves else None

out = {
    "task": "perturbation synchrony: gemma4:31b vs CLIP (LAION L/14), SCB5 TeacherBehavior 8 classes",
    "common_sigmas": COMMON_SIGMAS,
    "per_class_acc_sigma0": {c: round(gem_prof[0][c], 3) for c in CLASSES},
    "per_class_acc_sigma4": {c: round(gem_prof[4][c], 3) for c in CLASSES},
    "per_class_drop_gemma4": {c: round(gem_drop[c], 3) for c in CLASSES},
    "per_class_drop_clip": {c: round(clip_drop[c], 3) for c in CLASSES},
    "drop_spearman_rho": float(rho),
    "drop_spearman_p_asymptotic": float(p_asy),
    "drop_permutation_p_10k_seed42": float(p_perm),
    "per_class_curve_spearman": {c: (round(v, 3) if v is not None else None) for c, v in curve_rhos.items()},
    "mean_curve_spearman_over_usable": round(mean_curve_rho, 3) if mean_curve_rho is not None else None,
    "n_usable_curves": len(usable_curves),
}
with open(f"{BASE}/blur_synchrony.json", "w") as f:
    json.dump(out, f, indent=2)
print(json.dumps(out, indent=2))