"""
analysis/stanford40_geometry_permutation.py — Stanford40 cross-encoder
Mantel permutation test
========================================================================
The paper reports the Stanford40 CLIP-vs-QwenViT class-geometry Mantel
agreement as "permutation p<10^-4" (Section 4.6). The p-value stored in
results/exp6_stanford40_geometry.json is the asymptotic Spearman p
(1.56e-182); this script supplies the corresponding permutation p so the
"permutation" qualifier in the paper is literally supported.

Method: 10,000 random class-label permutations (seed 42) of one matrix's
class order; Spearman correlation between the upper triangles of the
CLIP and permuted Qwen matrices; p = fraction of permutations whose rho
meets or exceeds the observed value.

Input:  results/exp6_stanford40_geometry.json
Output: results/stanford40_geometry_permutation.json
Reproducibility: python analysis/stanford40_geometry_permutation.py
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS = REPO_ROOT / "results"
N_PERM = 10_000
SEED = 42


def upper_tri(m):
    iu = np.triu_indices_from(m, k=1)
    return m[iu]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_perm", type=int, default=N_PERM)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    geo = json.load(open(RESULTS / "exp6_stanford40_geometry.json"))
    class_names = geo["class_names"]
    clip = np.array(geo["similarity_matrices"]["CLIP_ViT_L_14_OpenAI"])
    qwen = np.array(geo["similarity_matrices"]["Qwen2_VL_Vision"])
    n = len(class_names)
    assert clip.shape == (n, n) and qwen.shape == (n, n)

    clip_ut = upper_tri(clip)
    qwen_ut = upper_tri(qwen)
    obs_rho, asym_p = spearmanr(clip_ut, qwen_ut)

    rng = np.random.RandomState(args.seed)
    count_ge = 0
    for _ in range(args.n_perm):
        perm = rng.permutation(n)
        qwen_p = qwen[perm][:, perm]
        rho_p, _ = spearmanr(clip_ut, upper_tri(qwen_p))
        if rho_p >= obs_rho:
            count_ge += 1
    perm_p = count_ge / args.n_perm

    report = {
        "pair": "CLIP ViT-L/14 (OpenAI) vs Qwen2-VL Vision",
        "dataset": "Stanford40",
        "n_classes": n,
        "n_pairs": int(len(clip_ut)),
        "observed_spearman_rho": float(obs_rho),
        "asymptotic_p": float(asym_p),
        "permutation_p": perm_p,
        "n_permutations": args.n_perm,
        "seed": args.seed,
        "note": "p = fraction of label permutations with rho >= observed; "
                "smallest attainable = 1/n_permutations",
    }
    out = RESULTS / "stanford40_geometry_permutation.json"
    with open(out, "w") as f:
        json.dump({"args": vars(args), "results": report}, f, indent=1)
    print(f"observed rho = {obs_rho:.6f}")
    print(f"asymptotic p = {asym_p:.3e}")
    print(f"permutation p = {perm_p:.6f} "
          f"({count_ge}/{args.n_perm} permutations >= observed)")
    print(f"written: {out}")


if __name__ == "__main__":
    sys.exit(main())
