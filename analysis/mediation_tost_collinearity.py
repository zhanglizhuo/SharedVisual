#!/usr/bin/env python3
"""Revision analyses A1 + A2 + A6.

Addresses reviewer Major Concerns #1 (absence-of-evidence vs evidence-of-absence),
#5 (mediation method), and the collinearity-artifact question.

A1 — Bootstrap mediation (Baron-Kenny product-of-coefficients on rank-z scores).
     For each MLLM: X=AnchorScore, M=leave-one-out cross-model mean difficulty,
     Y=target MLLM accuracy. Reports paths a, b, c (total), c' (direct), and the
     INDIRECT effect a*b with a 10,000-replicate class-resampling bootstrap CI,
     plus proportion mediated. This is the standard test the reviewer asked for
     (replacing the informal "raw rho -> partial rho drop").

A2 — TOST equivalence test on the residual (direct) correlation c'.
     H0: |rho| >= delta  vs  H1: |rho| < delta, with delta = 0.2 and 0.3.
     Uses Fisher-z; for partial r with 1 covariate SE_z = 1/sqrt(n-4).

A6 — Collinearity between X (AnchorScore) and M (LOO covariate): Pearson r,
     VIF = 1/(1-r^2), and the variance-inflation interpretation.

Runs on both SCB5 (13 pooled classes) and Stanford40 (40 classes).
Outputs: results/mediation_bootstrap.json  +  .txt summary
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, norm

BASE = Path(__file__).resolve().parents[1]
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

from sharedvisual.paths import REPO_ROOT, RESULTS_DIR, SEED_BOOTSTRAP
from sharedvisual.datasets import load_scb5_model_accs, load_stanford40 as load_stanford40_shared
from sharedvisual.stats import rank_z as rankmean, mediation_paths

OUT = RESULTS_DIR
OUT.mkdir(parents=True, exist_ok=True)

RNG_SEED = SEED_BOOTSTRAP
N_BOOT = 10000
EQ_DELTAS = (0.2, 0.3)  # equivalence bounds on |rho|


# ---------------------------------------------------------------- data loaders
def load_scb5():
    """Wrapper over shared loader returning (classes, anchor, [(model, accs)])."""
    return load_scb5_model_accs()


def load_stanford40():
    """Wrapper matching this script's original 4-tuple signature."""
    classes, anchor, model_accs = load_stanford40_shared()
    models = list(model_accs.keys())
    mvals = {m: model_accs[m] for m in models}
    return classes, anchor, models, mvals


# ----------------------------------------------------------------- statistics
def boot_indirect(x, m, y, n_boot=N_BOOT, seed=RNG_SEED):
    rng = np.random.RandomState(seed)
    n = len(x)
    out = {"indirect": [], "cprime": [], "a": [], "b": [], "c": [], "prop_mediated": []}
    for _ in range(n_boot):
        idx = rng.choice(n, n, replace=True)
        if len(np.unique(x[idx])) < 2 or len(np.unique(y[idx])) < 2:
            continue
        p = mediation_paths(x[idx], m[idx], y[idx])
        if not np.isfinite(p.get("indirect", float("nan"))):
            continue
        for k in out:
            out[k].append(p[k])
    ci = {}
    for k, v in out.items():
        v = np.array(v); v_f = v[np.isfinite(v)]
        if len(v_f) == 0:
            ci[k] = {"mean": float("nan"), "median": float("nan"),
                     "ci_lo": float("nan"), "ci_hi": float("nan"),
                     "n_valid": 0, "n_skipped": len(v)}
        else:
            ci[k] = {"mean": float(np.mean(v_f)), "median": float(np.median(v_f)),
                     "ci_lo": float(np.percentile(v_f, 2.5)),
                     "ci_hi": float(np.percentile(v_f, 97.5)),
                     # Fraction of bootstrap resamples with |effect| <= 1e-9
                     # (effect indistinguishable from zero). NOT a conventional
                     # two-sided p-value; named accordingly to avoid confusion.
                     "p_boot_at_zero": float(np.mean(np.abs(v_f) <= 1e-9)),
                     "n_valid": len(v_f), "n_skipped": len(v) - len(v_f)}
    return ci


def tost_partial(r, n, n_cov, deltas=EQ_DELTAS):
    """TOST for equivalence on a (partial) correlation via Fisher-z.
       H0: |rho|>=delta vs H1: |rho|<delta. Returns per-delta p and verdict."""
    z = np.arctanh(r)
    se = 1.0 / np.sqrt(n - 3 - n_cov)   # partial r with n_cov covariates
    res = {}
    for d in deltas:
        dz = np.arctanh(d)
        p_lo = norm.sf((z - (-dz)) / se)   # test H0: rho <= -delta
        p_hi = norm.cdf((z - dz) / se)     # test H0: rho >= +delta
        p_eq = max(p_lo, p_hi)
        res[f"delta_{d}"] = {
            "p_equiv": float(p_eq),
            "equivalent_at_0.05": bool(p_eq < 0.05),
            "ci_lo_zbased": float(np.tanh(z - norm.ppf(0.975) * se)),
            "ci_hi_zbased": float(np.tanh(z + norm.ppf(0.975) * se)),
        }
    return res


def collinearity(x, m):
    r, p = pearsonr(rankmean(x), rankmean(m))
    vif = 1.0 / (1.0 - r ** 2) if abs(r) < 0.999 else float("inf")
    return {"rho_X_M": float(r), "p": float(p), "VIF": float(vif)}


# --------------------------------------------------------------------- driver
def run_dataset(name, classes, anchor, model_accs):
    """model_accs: list of (model_name, acc_vector)."""
    n = len(classes)
    # cross-model mean per class (all models) for collinearity reference
    all_stack = np.stack([v for _, v in model_accs])
    out = {"dataset": name, "n_classes": n,
           "n_models": len(model_accs), "results": {}}
    lines = []
    for mm, mv in model_accs:
        # LOO covariate
        loo = np.mean(np.stack([v for o, v in model_accs if o != mm]), axis=0)
        p = mediation_paths(anchor, loo, mv)
        bc = boot_indirect(anchor, loo, mv)
        tost = tost_partial(p["cprime"], n=n, n_cov=1)
        col = collinearity(anchor, loo)
        out["results"][mm] = {
            "n": n,
            "paths": {k: round(float(v), 4) for k, v in p.items()},
            "bootstrap": {k: {kk: round(vv, 4) for kk, vv in vc.items()}
                          for k, vc in bc.items()},
            "tost_equivalence_delta0.2": tost[f"delta_{EQ_DELTAS[0]}"],
            "tost_equivalence_delta0.3": tost[f"delta_{EQ_DELTAS[1]}"],
            "collinearity_X_vs_LOOcovariate": {k: round(v, 4) for k, v in col.items()},
        }
        bi = bc["indirect"]
        lines.append(
            f"  {mm:20s} c={p['c']:+.3f} c'={p['cprime']:+.3f} "
            f"a*b={p['indirect']:+.3f}[{bi['ci_lo']:+.3f},{bi['ci_hi']:+.3f}] "
            f"prop={p['prop_mediated']:.2f} "
            f"TOST(d=.2)={'EQ' if tost['delta_0.2']['equivalent_at_0.05'] else 'no'} "
            f"VIF={col['VIF']:.2f}"
        )
    return out, lines


def main():
    print("=" * 78)
    print("REVISION ANALYSES: bootstrap mediation (a*b) + TOST + collinearity")
    print("=" * 78)
    full = {"seed": RNG_SEED, "n_boot": N_BOOT}

    # ---- SCB5 ----
    print("\n### SCB5 (13 pooled classes) ###")
    cls5, anc5, models5 = load_scb5()
    scb5, lines5 = run_dataset("SCB5", cls5, anc5, models5)
    full["SCB5"] = scb5
    for ln in lines5:
        print(ln)

    # ---- Stanford40 ----
    print("\n### Stanford40 (40 classes) ###")
    cls40, anc40, mods40, mv40 = load_stanford40()
    sf40, lines40 = run_dataset("Stanford40", cls40, anc40,
                                [(m, mv40[m]) for m in mods40])
    full["Stanford40"] = sf40
    for ln in lines40:
        print(ln)

    jpath = OUT / "mediation_bootstrap.json"
    json.dump(full, open(jpath, "w"), indent=2)
    print(f"\nSaved JSON -> {jpath}")


if __name__ == "__main__":
    main()
