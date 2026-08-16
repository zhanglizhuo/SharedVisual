#!/usr/bin/env python3
"""Revision: simulation check for the rank-z product-of-coefficients mediation.

Reviewer point 6: the "product-of-coefficients on standardized ranks" approach
is not a textbook mediation method. We verify, on simulated data with a known
indirect effect, that it (a) tracks the textbook OLS product-of-coefficients
(Baron-Kenny) estimator, and (b) agrees with Spearman partial-correlation
inference, at sample sizes used in the paper (n=13, n=40).

Output: results/mediation_simulation.json
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr, rankdata

BASE = Path(__file__).resolve().parents[1]
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

from sharedvisual.paths import RESULTS_DIR, SEED_SIMULATION
from sharedvisual.stats import rank_z as rankmean

OUT = RESULTS_DIR


def partial_corr_manual(x, y, covar):
    """Correlation of residuals: corr(x - beta_x*covar, y - beta_y*covar)."""
    def resid(v, c):
        c = np.column_stack([c, np.ones(len(c))])
        beta, *_ = np.linalg.lstsq(c, v, rcond=None)
        return v - c @ beta
    rx_ = resid(x, covar)
    ry_ = resid(y, covar)
    return np.corrcoef(rx_, ry_)[0, 1]


def rankz_paths(x, m, y):
    """Our estimator: product-of-coefficients on z-scored ranks."""
    rx, rm, ry = rankmean(x), rankmean(m), rankmean(y)
    r_xm, _ = pearsonr(rx, rm)
    r_xy, _ = pearsonr(rx, ry)
    r_my, _ = pearsonr(rm, ry)
    beta_b = (r_my - r_xm * r_xy) / (1.0 - r_xm ** 2)
    return r_xm * beta_b  # indirect effect


def ols_paths(x, m, y):
    """Textbook Baron-Kenny product-of-coefficients on original variables."""
    xc = x - x.mean(); mc = m - m.mean(); yc = y - y.mean()
    r_xm = xc @ mc / np.sqrt((xc @ xc) * (mc @ mc))
    r_xy = xc @ yc / np.sqrt((xc @ xc) * (yc @ yc))
    r_my = mc @ yc / np.sqrt((mc @ mc) * (yc @ yc))
    beta_b = (r_my - r_xm * r_xy) / (1.0 - r_xm ** 2)
    return r_xm * beta_b  # indirect effect (standardized)


def spearman_partial(x, m, y):
    """Spearman partial correlation r(X,Y|M): correlation of rank residuals."""
    rx, rm, ry = (rankdata(v, method="average") for v in (x, m, y))
    return partial_corr_manual(rx, ry, rm)


def simulate(n, a, b, cprime, n_rep=2000, seed=20260802):
    rng = np.random.RandomState(seed)
    d = {"n": n, "a": a, "b": b, "cprime": cprime, "n_rep": n_rep,
         "rankz_indirect": [], "ols_indirect": [],
         "spearman_partial_cp": [], "rankz_cp": []}
    for _ in range(n_rep):
        x = rng.normal(size=n)
        m = a * x + np.sqrt(1 - a ** 2) * rng.normal(size=n)
        y = cprime * x + b * m + rng.normal(size=n)
        # noise scaling to realistic accuracy-like correlation levels
        y = y / np.std(y)
        d["rankz_indirect"].append(rankz_paths(x, m, y))
        d["ols_indirect"].append(ols_paths(x, m, y))
        d["spearman_partial_cp"].append(spearman_partial(x, m, y))
        # our direct-effect estimator from mediation_paths
        rx, rm, ry = rankmean(x), rankmean(m), rankmean(y)
        r_xm, _ = pearsonr(rx, rm); r_xy, _ = pearsonr(rx, ry); r_my, _ = pearsonr(rm, ry)
        d["rankz_cp"].append((r_xy - r_xm * r_my) / (1.0 - r_xm ** 2))
    for k in ("rankz_indirect", "ols_indirect", "spearman_partial_cp", "rankz_cp"):
        v = np.array(d[k])
        d[k + "_mean"] = float(np.mean(v))
        d[k + "_se"] = float(np.std(v, ddof=1))
        d[k] = d[k][:50]  # keep a sample, not all 2000
    return d


def main():
    configs = [
        dict(n=13, a=0.70, b=0.60, cprime=0.20),
        dict(n=13, a=0.70, b=0.60, cprime=0.00),
        dict(n=40, a=0.70, b=0.60, cprime=0.20),
        dict(n=40, a=0.70, b=0.60, cprime=0.00),
    ]
    out = {"configs": []}
    for cfg in configs:
        res = simulate(**cfg)
        # agreement between our estimator and the textbook estimator
        diffs = np.array(res["rankz_indirect"][:50]) - np.array(res["ols_indirect"][:50])
        res["mean_abs_diff_rankz_vs_ols"] = float(np.mean(np.abs(diffs)))
        # bias vs true *standardized* indirect effect.
        # Estimators operate on correlations (scale-free), so they recover the
        # standardized indirect effect a*b / sd(Y_raw), NOT the structural a*b.
        # Var(Y_raw) = c'^2 + b^2 + 2*c'*b*a + Var(noise), with Var(X)=Var(M)=1.
        # For Corr(X,M)=a: Cov(X,Y)=c'+b*a, so Var(Y)=c'^2+b^2+2*c'*b*a+1.
        var_y = cfg["cprime"]**2 + cfg["b"]**2 + 2*cfg["cprime"]*cfg["b"]*cfg["a"] + 1.0
        true_ab = cfg["a"] * cfg["b"] / np.sqrt(var_y)
        res["true_indirect"] = true_ab
        res["bias_rankz"] = float(res["rankz_indirect_mean"] - true_ab)
        res["bias_ols"] = float(res["ols_indirect_mean"] - true_ab)
        res["mean_spearman_partial_cp"] = res.pop("spearman_partial_cp_mean")
        res["true_cprime"] = cfg["cprime"]
        out["configs"].append(res)
        print(f"n={cfg['n']} a*b(true)={true_ab:.2f} c'(true)={cfg['cprime']:.2f}")
        print(f"  rankz indirect mean={res['rankz_indirect_mean']:.3f} (se {res['rankz_indirect_se']:.3f})")
        print(f"  ols   indirect mean={res['ols_indirect_mean']:.3f} (se {res['ols_indirect_se']:.3f})")
        print(f"  bias rankz={res['bias_rankz']:+.3f} ols={res['bias_ols']:+.3f} |diff|={res['mean_abs_diff_rankz_vs_ols']:.4f}")
        print(f"  spearman partial c'(mean)={res['mean_spearman_partial_cp']:.3f}  rankz c'(mean)={res['rankz_cp_mean']:.3f} (true {cfg['cprime']:.2f})")
        print()
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "mediation_simulation.json", "w") as f:
        json.dump(out, f, indent=1)
    print("Saved results/mediation_simulation.json")


if __name__ == "__main__":
    main()
