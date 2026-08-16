"""Mantel permutation test for the Flowers102 cross-domain geometry result.

Flowers102 has 102 classes -> exact permutation (102!) is infeasible, so we use
10,000 random class-label permutations (seeded). Reads the matrices produced by
experiments/exp_flowers102_geometry.py and outputs Mantel rho + random-
permutation p-values for all C(5,2)=10 encoder pairs.

Run locally after V100 extraction:
  python analysis/flowers102_mantel.py
"""
import json
import itertools
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
IN_PATH = ROOT / "results" / "flowers102_geometry.json"
OUT_PATH = ROOT / "results" / "flowers102_mantel.json"
N_PERM = 10_000
SEED = 42


def mantel_stat(A, B):
    iu = np.triu_indices(A.shape[0], k=1)
    r, _ = spearmanr(A[iu], B[iu])
    return float(r)


def random_mantel_p(A, B, n_perm=N_PERM, seed=SEED):
    """Random-permutation Mantel p-value (right-tail)."""
    rng = np.random.RandomState(seed)
    rho_obs = mantel_stat(A, B)
    count = 0
    n = A.shape[0]
    for _ in range(n_perm):
        perm = rng.permutation(n)
        Ap = A[perm][:, perm]
        rho_p = mantel_stat(Ap, B)
        if rho_p >= rho_obs:
            count += 1
    return rho_obs, (count + 1) / (n_perm + 1), n_perm


def main():
    data = json.load(open(IN_PATH))
    mats = data["matrices"]
    names = list(mats.keys())
    arrs = {n: np.array(mats[n], dtype=float) for n in names}
    print(f"Flowers102 Mantel: {len(names)} encoders, "
          f"{arrs[names[0]].shape[0]} classes, {N_PERM} random permutations")

    results = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            rho, p, n_perm = random_mantel_p(arrs[a], arrs[b])
            tag = ("within-CLIP" if "CLIP" in a and "CLIP" in b
                   else "SigLIP-cross" if "SigLIP" in a or "SigLIP" in b
                   else "cross-encoder")
            print(f"  [{tag:14s}] {a[:25]:25s} vs {b[:25]:25s} "
                  f"rho={rho:.4f}  p={p:.3g}")
            results.append({
                "encoder_a": a, "encoder_b": b, "tag": tag,
                "mantel_rho": rho, "permutation_p": p,
                "n_permutations": n_perm,
                "method": f"random permutations (seed={SEED})",
            })

    out = {
        "dataset": "Flowers102 (102 fine-grained flower species)",
        "n_classes": arrs[names[0]].shape[0],
        "n_pairs": len(results),
        "results": results,
        "note": ("Random-permutation Mantel (102 classes, exact infeasible). "
                 "Compare against SCB5 8-class Mantel rho 0.90-0.98 (5 encoders) "
                 "and Stanford40 40-class rho ~0.81."),
    }
    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved -> {OUT_PATH}")


if __name__ == "__main__":
    main()
