"""Mediation analysis on Stanford40 (40 classes, 6 MLLMs).

Replicates the SCB5 partial-correlation/mediation analysis on Stanford40
to test whether shared difficulty fully mediates the AnchorScore-MLLM
correlation at higher statistical power (n=40 classes vs n=13 on SCB5).

For each MLLM we compute:
  - raw Spearman rho between AnchorScore and MLLM per-class accuracy
  - partial Spearman rho after controlling for the leave-one-out cross-model
    average per-class difficulty (mean accuracy across the OTHER 5 MLLMs;
    the target model's own difficulty is excluded to avoid circularity)
  - non-parametric bootstrap 95% CI (resampling the 40 classes)
  - proportion mediated

Additionally we report the aggregated phenomenon statistic: the Spearman
correlation between AnchorScore and the cross-model mean accuracy (all 6
MLLMs), with a bootstrap 95% CI (seed RNG_SEED, N_BOOT resamples).

Inputs (VisualAnchor repo):
  results/02_robustness/stanford40/anchor_scores.json
  results/02_robustness/stanford40/ollama_results.json
Output:
  results/stanford40_partial_correlation.json
"""
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

import sys
BASE = Path(__file__).resolve().parents[1]
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))
from sharedvisual.paths import va_results as _va
VA_RESULTS = _va()
ANCHOR_PATH = VA_RESULTS / "02_robustness" / "stanford40" / "anchor_scores.json"
OLLAMA_PATH = VA_RESULTS / "02_robustness" / "stanford40" / "ollama_results.json"
OUT_PATH = BASE / "results" / "stanford40_partial_correlation.json"

RNG_SEED = 20260731
N_BOOT = 10000


def resid(a, b):
    lr = np.polyfit(b, a, 1)
    return a - np.polyval(lr, b)


def partial_corr(x, y, cov):
    xr = resid(x, cov)
    yr = resid(y, cov)
    if len(np.unique(xr)) > 1 and len(np.unique(yr)) > 1:
        r, _ = spearmanr(xr, yr)
        return float(r)
    return float("nan")


def bootstrap_ci(x, y, cov, n_boot=N_BOOT, seed=RNG_SEED):
    rng = np.random.RandomState(seed)
    n = len(x)
    boot = np.full(n_boot, np.nan)
    for i in range(n_boot):
        idx = rng.choice(n, n, replace=True)
        bx, by, bc = x[idx], y[idx], cov[idx]
        br = partial_corr(bx, by, bc)
        boot[i] = br
    boot = boot[~np.isnan(boot)]
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return float(lo), float(hi)


def main():
    with open(ANCHOR_PATH) as f:
        anchor = json.load(f)
    with open(OLLAMA_PATH) as f:
        ollama = json.load(f)

    classes = sorted(anchor["per_class_acc"].keys())
    models = list(ollama["models"].keys())
    n_classes = len(classes)
    print(f"Classes: {n_classes}, Models: {len(models)}: {models}")

    anchor_vals = np.array([anchor["per_class_acc"][c]["acc"] for c in classes])

    mllm_vals = {}
    for m in models:
        vals = np.array([ollama["models"][m]["per_class_acc"][c]["acc"] for c in classes])
        mllm_vals[m] = vals

    # Aggregated phenomenon statistic: AnchorScore vs cross-model mean accuracy.
    mean_acc = np.mean(np.stack(list(mllm_vals.values())), axis=0)
    r_agg, p_agg = spearmanr(anchor_vals, mean_acc)
    rng_agg = np.random.RandomState(RNG_SEED)
    boot_agg = []
    for _ in range(N_BOOT):
        idx = rng_agg.choice(n_classes, n_classes, replace=True)
        r, _ = spearmanr(anchor_vals[idx], mean_acc[idx])
        boot_agg.append(r)
    boot_agg = np.array(boot_agg)
    ci_agg = (float(np.percentile(boot_agg, 2.5)), float(np.percentile(boot_agg, 97.5)))

    results = {}
    for m in models:
        mv = mllm_vals[m]
        r_raw, p_raw = spearmanr(anchor_vals, mv)
        # Leave-one-out covariate: mean over the other 5 MLLMs.
        loo = np.stack([mllm_vals[o] for o in models if o != m]).mean(axis=0)
        r_partial = partial_corr(anchor_vals, mv, loo)
        ci_lo, ci_hi = bootstrap_ci(anchor_vals, mv, loo)
        prop_med = (r_raw - r_partial) / r_raw if r_raw != 0 else float("nan")
        results[m] = {
            "raw_spearman_r": round(float(r_raw), 3),
            "raw_p": float(p_raw),
            "partial_spearman_r": round(float(r_partial), 3),
            "bootstrap_95_ci": [round(ci_lo, 3), round(ci_hi, 3)],
            "proportion_mediated": round(float(prop_med), 3),
            "n_classes": n_classes,
            "n_bootstrap": N_BOOT,
            "covariate": "leave-one-out cross-model mean accuracy (other 5 MLLMs)",
        }
        print(f"  {m:18s} raw={r_raw:+.3f}  partial={r_partial:+.3f}  "
              f"CI=[{ci_lo:+.3f}, {ci_hi:+.3f}]  prop_med={prop_med:.3f}")

    mean_raw = np.mean([results[m]["raw_spearman_r"] for m in models])
    mean_partial = np.mean([results[m]["partial_spearman_r"] for m in models])
    print(f"\n  Mean raw rho    = {mean_raw:+.3f}")
    print(f"  Mean partial rho= {mean_partial:+.3f}")
    print(f"  Aggregated rho (AnchorScore vs cross-model mean) = {r_agg:+.3f} "
          f"(p={p_agg:.2e}, bootstrap CI [{ci_agg[0]:.3f}, {ci_agg[1]:.3f}])")

    out = {
        "dataset": "Stanford40",
        "n_classes": n_classes,
        "n_mllms": len(models),
        "models": models,
        "method": "partial Spearman controlling for leave-one-out cross-model mean difficulty; non-parametric bootstrap (resample classes)",
        "aggregated_phenomenon": {
            "statistic": "Spearman rho(AnchorScore, cross-model mean accuracy over 6 MLLMs)",
            "spearman_r": round(float(r_agg), 3),
            "p_value": float(p_agg),
            "bootstrap_95_ci": [round(ci_agg[0], 3), round(ci_agg[1], 3)],
            "n_bootstrap": N_BOOT,
            "seed": RNG_SEED,
        },
        "results": results,
        "summary": {
            "mean_raw_rho": round(float(mean_raw), 3),
            "mean_partial_rho": round(float(mean_partial), 3),
            "all_ci_contain_zero": all(
                results[m]["bootstrap_95_ci"][0] <= 0 <= results[m]["bootstrap_95_ci"][1]
                for m in models),
        },
    }
    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved -> {OUT_PATH}")


if __name__ == "__main__":
    main()
