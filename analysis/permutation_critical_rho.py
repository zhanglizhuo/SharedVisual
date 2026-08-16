"""Permutation null distribution for Spearman |rho| under class-label permutation.

Supplies the significance thresholds quoted in the paper (SS Statistics):
with 13 pooled classes, |rho| ~ 0.55 is the two-sided alpha=0.05 critical
value under permutation of class labels; with 8 classes it is ~0.71.
These match the standard Spearman critical-value tables (n=13: 0.560,
n=8: 0.738) to within sampling error.

Because the number of class labels is small, asymptotic Spearman critical
values (n >= 8 treated as normal) are unreliable; we estimate the null by
permuting labels and recomputing the correlation. 100,000 permutations are
used.

Usage:
  python analysis/permutation_critical_rho.py
"""
import json
import numpy as np
from pathlib import Path
from scipy.stats import spearmanr, rankdata

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results" 
N_PERM = 100_000
ALPHA = 0.05


def two_sided_critical_rho(n, rng):
    """Critical |rho| under label permutation, estimated from N_PERM draws.

    rho is recomputed on rank data (as in the paper's Spearman usage):
    x are fixed class-level ranks 1..n, y are the ranks of permuted labels.
    """
    x = np.arange(1, n + 1)
    samples = np.zeros(N_PERM)
    for i in range(N_PERM):
        y = rng.permutation(x)
        samples[i] = abs(np.corrcoef(x, y)[0, 1])
    crit = np.quantile(samples, 1 - ALPHA)
    return crit, samples


def main():
    rng = np.random.default_rng(20260731)
    out = {"alpha": ALPHA, "n_perm": N_PERM, "critical_rho": {}}
    for n in (8, 13, 40):
        crit, samples = two_sided_critical_rho(n, rng)
        out["critical_rho"][str(n)] = round(float(crit), 4)
        print(f"n={n:2d}: two-sided {ALPHA} critical |rho| = {crit:.4f}")

    out_path = RESULTS_DIR / "permutation_critical_rho.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
