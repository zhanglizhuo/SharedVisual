from pathlib import Path
"""Perturbation synchrony v2: uses RECREATED CLIP blur (same PIL blur pipeline
as gemma4 pilot) and DUAL-scored gemma4 results.

This supersedes blur_synchrony.py once the recreations complete. Two scoring
conventions for gemma4: display (prompt vocabulary) and raw-key (paper's
9-MLLM table scoring). Two CLIP regimes: 'all' (3240 val images, class-
imbalanced) and '50' (first 50/class, n=390, identical sample set as gemma4).

The methodologically clean comparison is (gemma4, 50-image sample) vs
(CLIP, 50-image sample) under the shared blur pipeline = exp4_blur_recreated_50.

Output: results/blur_synchrony_v2.json
"""
import json
import numpy as np
from scipy.stats import spearmanr

BASE = Path(__file__).resolve().parents[1] / "results"
CLASSES = ["guide", "answer", "On-stage interaction", "blackboard-writing",
           "teacher", "stand", "screen", "blackBoard"]
COMMON_SIGMAS = [0, 1, 2, 4]  # intersection of gemma4 {0..4} and CLIP {0,1,2,4,8}


def perm_p(rho, x, y, seed=42, n=10000):
    rng = np.random.default_rng(seed)
    perm = [spearmanr(x, rng.permutation(y)).statistic for _ in range(n)]
    return (1 + np.sum(np.abs(np.array(perm)) >= abs(rho))) / (1 + n)


def load_clip(regime):
    path = f"{BASE}/exp4_clip_blur_recreated_{regime}.json"
    d = json.load(open(path))
    return {s: d[f"blur_sigma_{s}"] for s in COMMON_SIGMAS}


def load_gemma(scoring):
    d = json.load(open(f"{BASE}/exp_perturbation_blur_dual.json"))["results"]
    key = "per_class_acc" if scoring == "display" else "per_class_acc_rawkey"
    return {s: d[f"sigma_{s}"][key] for s in COMMON_SIGMAS}


def analyze(clip_prof, gem_prof, tag):
    clip_drop = {c: clip_prof[0][c] - clip_prof[4][c] for c in CLASSES}
    gem_drop = {c: gem_prof[0][c] - gem_prof[4][c] for c in CLASSES}
    x = np.array([clip_drop[c] for c in CLASSES])
    y = np.array([gem_drop[c] for c in CLASSES])
    rho, p_asy = spearmanr(x, y)
    p_perm = perm_p(rho, x, y)
    curve_rhos = {}
    for c in CLASSES:
        g = [gem_prof[s][c] for s in COMMON_SIGMAS]
        k = [clip_prof[s][c] for s in COMMON_SIGMAS]
        curve_rhos[c] = float(spearmanr(g, k).statistic) if len(set(g)) > 1 and len(set(k)) > 1 else None
    usable = [v for v in curve_rhos.values() if v is not None]
    return {
        "tag": tag,
        "clip_drop": {c: round(clip_drop[c], 3) for c in CLASSES},
        "gemma_drop": {c: round(gem_drop[c], 3) for c in CLASSES},
        "drop_spearman_rho": float(rho),
        "drop_p_asymptotic": float(p_asy),
        "drop_p_perm_10k_seed42": float(p_perm),
        "per_class_curve_spearman": {c: (round(v, 3) if v is not None else None) for c, v in curve_rhos.items()},
        "mean_curve_spearman_usable": round(float(np.mean(usable)), 3) if usable else None,
        "n_usable_curves": len(usable),
    }


def main():
    out = {"task": "perturbation synchrony v2 (shared PIL blur pipeline, dual scoring)",
           "common_sigmas": COMMON_SIGMAS,
           "comparisons": []}
    for regime in ["all", "50"]:
        clip_prof = load_clip(regime)
        for scoring in ["display", "rawkey"]:
            gem_prof = load_gemma(scoring)
            out["comparisons"].append(
                analyze(clip_prof, gem_prof, f"CLIP_{regime}_vs_gemma_{scoring}"))
    path = f"{BASE}/blur_synchrony_v2.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()