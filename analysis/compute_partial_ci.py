"""Bootstrap CI for partial correlations — mirrors exp1_deep_analysis.py data loading."""

import json
import numpy as np
from pathlib import Path
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression

RNG = np.random.RandomState(20260731)
N_BOOT = 10000
import sys
BASE = Path(__file__).resolve().parents[1]
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))
from sharedvisual.paths import va_results as _va
VA_RESULTS = _va()
RESULTS_DIR = BASE / "results"

CLASS_MAP = {
    "TeacherBehavior": [
        "guide", "answer", "On-stage interaction", "blackboard-writing",
        "teacher", "stand", "screen", "blackBoard",
    ],
    "HandriseReadWrite": ["hand-raising", "read", "write"],
    "BowTurnHead": ["BowHead", "TurnHead"],
}

# Load data
with open(f'{VA_RESULTS}/01_core/anchor_score_scb5/anchor_scores.json') as f:
    anchor_raw = json.load(f)
with open(f'{VA_RESULTS}/01_core/paper_data/mllm_full.json') as f:
    mllm_raw = json.load(f)

# Build aligned dicts
anchor_scores = {}
for ds, classes in CLASS_MAP.items():
    for cls in classes:
        anchor_scores[(ds, cls)] = anchor_raw[ds]['per_class_acc'][cls]['acc']

mllm_scores = {}
for model in mllm_raw.get('TeacherBehavior', {}):
    accs = {}
    for ds, classes in CLASS_MAP.items():
        if model in mllm_raw.get(ds, {}):
            for cls in classes:
                val = mllm_raw[ds][model].get(cls)
                if val is not None:
                    accs[(ds, cls)] = val
    if accs:
        mllm_scores[model] = accs

all_classes = sorted(anchor_scores.keys())
anchor_vals = np.array([anchor_scores[c] for c in all_classes])

def resid(a, b):
    mask = ~np.isnan(a) & ~np.isnan(b)
    a, b = a[mask], b[mask]
    if len(a) < 3 or len(np.unique(b)) < 2:
        return a
    lr = LinearRegression()
    lr.fit(b.reshape(-1, 1), a)
    return a - lr.predict(b.reshape(-1, 1))

def partial_corr_val(x, y, cov):
    xr = resid(x, cov)
    yr = resid(y, cov)
    if len(np.unique(xr)) > 1 and len(np.unique(yr)) > 1:
        r, _ = spearmanr(xr, yr)
        return float(r)
    return float('nan')

def bootstrap_partial_ci(x, y, cov, n_iter=N_BOOT):
    n = len(x)
    boot = np.zeros(n_iter)
    for i in range(n_iter):
        idx = RNG.choice(n, n, replace=True)
        bx, by, bc = x[idx], y[idx], cov[idx]
        # Partial Spearman (vectorized)
        bx_r = bx - np.polyval(np.polyfit(bc, bx, 1), bc)
        by_r = by - np.polyval(np.polyfit(bc, by, 1), bc)
        if len(np.unique(bx_r)) > 1 and len(np.unique(by_r)) > 1:
            boot[i], _ = spearmanr(bx_r, by_r)
        else:
            boot[i] = np.nan
    ci_lo, ci_hi = np.percentile(boot[~np.isnan(boot)], [2.5, 97.5])
    return float(round(ci_lo, 3)), float(round(ci_hi, 3))

# Leave-one-out cross-model mean difficulty covariate:
# for each target model the covariate averages over the OTHER models only,
# avoiding the circularity of including the target model's own difficulty.
all_model_vals = []
for m in mllm_scores:
    vals = np.array([mllm_scores[m].get(c, np.nan) for c in all_classes])
    all_model_vals.append((m, vals))

models = ['LLaVA-1.5-7B', 'Gemma4-31B', 'Gemma-3-27B', 'Qwen3.5-35B', 'Qwen2-VL-7B']

results = {}
for model in models:
    if model not in mllm_scores:
        continue
    vals = np.array([mllm_scores[model].get(c, np.nan) for c in all_classes])
    loo = np.array([v for m_, v in all_model_vals if m_ != model])
    mean_difficulty = np.nanmean(loo, axis=0)
    mask = ~np.isnan(vals) & ~np.isnan(mean_difficulty)
    av, mv, md = anchor_vals[mask], vals[mask], mean_difficulty[mask]
    n = len(av)

    r_raw, _ = spearmanr(av, mv)
    r_partial = partial_corr_val(av, mv, md)
    _, p_partial = spearmanr(resid(av, md), resid(mv, md))

    ci_lo, ci_hi = bootstrap_partial_ci(av, mv, md)
    prop_med = (r_raw - r_partial) / r_raw if r_raw != 0 else float('nan')

    results[model] = {
        'raw_spearman_r': round(float(r_raw), 3),
        'partial_spearman_r': round(float(r_partial), 3),
        'partial_p': round(float(p_partial), 4),
        'bootstrap_95_ci': [ci_lo, ci_hi],
        'proportion_mediated': round(float(prop_med), 3),
        'n_classes': n,
        'n_bootstrap': N_BOOT,
        'method': 'non-parametric bootstrap (resample classes with replacement)'
    }

out_path = f'{RESULTS_DIR}/partial_correlation_bootstrap.json'
with open(out_path, 'w') as f:
    json.dump({**results, '_metadata': {
        'date': '2026-07-31',
        'dataset': 'SCB5 (all sub-datasets: TeacherBehavior + HandriseReadWrite + BowTurnHead)',
        'n_bootstrap': N_BOOT,
        'n_classes_total': len(all_classes),
        'analysis_script': 'compute_partial_ci.py',
        'covariate': 'leave-one-out cross-model mean accuracy (target model excluded)',
        'method': 'Non-parametric bootstrap resampling classes with replacement',
        'note': 'CI computed from 10,000 bootstrap resamples of (Anchor, MLLM, covariate) triples, resampling classes with replacement'
    }}, f, indent=2)

print(f'Saved to {out_path}')
print(f'{"Model":20s} {"n":>3s} {"Raw ρ":>6s} {"Partial ρ":>10s} {"95% CI":>16s} {"Prop. mediated":>14s}')
print('-' * 72)
for m in models:
    r = results[m]
    ci = r['bootstrap_95_ci']
    print(f'{m:20s} {r["n_classes"]:3d} {r["raw_spearman_r"]:6.3f} {r["partial_spearman_r"]:+10.3f}  [{ci[0]:.3f}, {ci[1]:.3f}]  {r["proportion_mediated"]:>6.1%}')
