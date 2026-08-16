#!/usr/bin/env python3
"""Competence-adjusted Mantel baseline (review round, DA-MAJOR-1).

The trivial-competence alternative to shared geometry: encoders that merely
solve the same task could show elevated class-similarity agreement because all
classes share within-domain similarity. We quantify this baseline with the
permutation null: random class relabelings of one matrix against the other.
Report per-pair observed Spearman rho, the null mean, and the null maximum,
plus the observed-minus-null-max margin.

Input : results/class_similarity_matrices.json (7 encoders, 8x8 SCB5 matrices)
Output: results/mantel_null_baseline.json
"""
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

BASE = Path(__file__).resolve().parents[1]
N_PERM = 10_000
SEED = 42


def mantel(A, B):
    iu = np.triu_indices(A.shape[0], k=1)
    return float(spearmanr(A[iu], B[iu])[0])


def main():
    d = json.load(open(BASE / "results/class_similarity_matrices.json"))
    mats = d["matrices"]
    names = list(mats.keys())
    rng = np.random.RandomState(SEED)
    pairs = []
    global_null_max = 0.0
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            A = np.array(mats[names[i]])
            B = np.array(mats[names[j]])
            obs = mantel(A, B)
            n = A.shape[0]
            perm = np.empty(N_PERM)
            for k in range(N_PERM):
                order = rng.permutation(n)
                perm[k] = abs(mantel(A, B[np.ix_(order, order)]))
            null_max = float(perm.max())
            null_mean = float(perm.mean())
            global_null_max = max(global_null_max, null_max)
            pairs.append({
                "encoder_a": names[i],
                "encoder_b": names[j],
                "observed_rho": round(obs, 3),
                "null_mean_abs_rho": round(null_mean, 3),
                "null_max_abs_rho": round(null_max, 3),
                "margin_over_null_max": round(obs - null_max, 3),
            })
    out = {
        "n_perm": N_PERM,
        "seed": SEED,
        "n_pairs": len(pairs),
        "global_null_max_abs_rho": round(global_null_max, 3),
        "min_margin_over_null_max": round(min(p["margin_over_null_max"] for p in pairs), 3),
        "max_margin_over_null_max": round(max(p["margin_over_null_max"] for p in pairs), 3),
        "pairs": pairs,
    }
    (BASE / "results/mantel_null_baseline.json").write_text(json.dumps(out, indent=2))
    print(json.dumps({k: v for k, v in out.items() if k != "pairs"}, indent=2))


if __name__ == "__main__":
    main()
