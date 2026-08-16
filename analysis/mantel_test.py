"""Mantel test: cross-encoder agreement of class-similarity matrices.

For each pair of vision encoders we quantify how similar their
class-neighborhood geometry is using a Mantel test:
  - statistic: Spearman rank correlation between the upper triangles of the
    two class-similarity matrices (cosine similarity of per-class centroids).
  - significance: permutation p-value. We permute the class labels (rows and
    columns simultaneously) of one matrix and recompute the statistic; the
    p-value is the fraction of permutations whose statistic meets or exceeds
    the observed value.

For SCB5 (n=8 classes) we enumerate all 8! = 40320 permutations (exact test).
For Stanford40 (n=40 classes) we draw 10,000 random permutations.

Inputs (produced by experiments/exp2_mllm_cka*.py and
exp6_stanford40_geometry.py):
  - results/class_similarity_matrices.json   (SCB5, 8x8)
  - results/exp6_stanford40_geometry.json    (Stanford40, 40x40)

Output:
  - results/mantel_test_results.json
"""
import itertools
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

RESULTS = Path(__file__).resolve().parents[1] / "results" 
SCB5_MAT = RESULTS / "class_similarity_matrices.json"
STANFORD_MAT = RESULTS / "exp6_stanford40_geometry.json"
OUT_PATH = RESULTS / "mantel_test_results.json"

N_PERM_RANDOM = 10_000
SEED = 42


def mantel_statistic(sim_a, sim_b):
    iu = np.triu_indices(sim_a.shape[0], k=1)
    rho, _ = spearmanr(sim_a[iu], sim_b[iu])
    return float(rho)


def permute_matrix(sim, perm):
    return sim[perm][:, perm]


def mantel_permutation_test(sim_a, sim_b, exact=False, n_random=N_PERM_RANDOM):
    """Return (rho_obs, p_value, n_perm, method)."""
    n = sim_a.shape[0]
    iu = np.triu_indices(n, k=1)
    rho_obs = mantel_statistic(sim_a, sim_b)

    if exact:
        perms = itertools.permutations(range(n))
        count = 0
        total = 0
        for perm in perms:
            rho_p = mantel_statistic(permute_matrix(sim_a, list(perm)), sim_b)
            if rho_p >= rho_obs:
                count += 1
            total += 1
        pval = (count + 1) / (total + 1)
        return rho_obs, pval, total, "exact (all permutations)"
    else:
        rng = np.random.RandomState(SEED)
        count = 0
        for _ in range(n_random):
            perm = rng.permutation(n)
            rho_p = mantel_statistic(permute_matrix(sim_a, perm), sim_b)
            if rho_p >= rho_obs:
                count += 1
        pval = (count + 1) / (n_random + 1)
        return rho_obs, pval, n_random, f"random permutations (seed={SEED})"


def run_dataset(matrices_path, encoder_key, exact):
    with open(matrices_path) as f:
        data = json.load(f)
    if encoder_key == "matrices":
        mats = data["matrices"]
        names = list(mats.keys())
        arrays = {n: np.array(mats[n], dtype=float) for n in names}
    else:
        mats = data["similarity_matrices"]
        names = list(mats.keys())
        arrays = {n: np.array(mats[n], dtype=float) for n in names}

    pairs = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            n_classes = arrays[a].shape[0]
            do_exact = exact and n_classes <= 8
            rho, p, n_perm, method = mantel_permutation_test(
                arrays[a], arrays[b], exact=do_exact)
            pairs.append({
                "encoder_a": a,
                "encoder_b": b,
                "n_classes": int(n_classes),
                "n_pairs": int(n_classes * (n_classes - 1) // 2),
                "mantel_rho": rho,
                "permutation_p": p,
                "n_permutations": int(n_perm),
                "method": method,
            })
            tag = "within" if ("CLIP" in a and "CLIP" in b) else "cross"
            print(f"  [{tag}] {a[:28]:28s} vs {b[:28]:28s}  "
                  f"rho={rho:.4f}  p={p:.3g}  (n_perm={n_perm})")
    return pairs


def main():
    print("=" * 72)
    print("Mantel test: cross-encoder class-geometry agreement")
    print("=" * 72)

    print("\n[SCB5] 8 classes (exact permutation test, 8! = 40320)")
    scb5 = run_dataset(SCB5_MAT, "matrices", exact=True)

    print("\n[Stanford40] 40 classes (10,000 random permutations)")
    stanford = run_dataset(STANFORD_MAT, "similarity_matrices", exact=False)

    out = {
        "test": "Mantel (Spearman rank correlation of similarity-matrix upper triangles, permutation-based significance)",
        "scb5_8_classes": scb5,
        "stanford40_40_classes": stanford,
    }
    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved -> {OUT_PATH}")


if __name__ == "__main__":
    main()
