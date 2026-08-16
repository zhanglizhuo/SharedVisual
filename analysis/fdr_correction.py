#!/usr/bin/env python3
"""A5 — FDR (Benjamini-Hochberg) correction over all p-values in the paper.

Collects reported p-values from:
  (a) Raw AnchorScore-MLLM Spearman correlations (9 MLLM × 1 dataset SCB5, 6 × Stanford40)
  (b) Partial correlation p-values (Table 1, 5 SCB5 models)
  (c) Stanford40 partial correlation raw-p values
  (d) Mantel test p-values (6 encoder pairs)
  (e) CKA significance (none reported — skip, or note)

Reports BH-adjusted q-values and flags which survive at FDR=0.05.

Output: results/fdr_correction.json + .txt summary
"""
import json
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))
from sharedvisual.paths import RESULTS_DIR
OUT_DIR = RESULTS_DIR

import numpy as np


OUT_DIR.mkdir(parents=True, exist_ok=True)


def bh_q(pvals):
    """Benjamini-Hochberg FDR correction. Returns q-values (same order)."""
    pvals = np.array(pvals, dtype=float)
    n = len(pvals)
    order = np.argsort(pvals)
    ranks = np.empty(n, dtype=float)
    ranks[order] = np.arange(1, n + 1)
    qvals = pvals * n / ranks
    # Ensure monotonicity
    sorted_q = qvals[order]
    cummin = np.minimum.accumulate(sorted_q[::-1])[::-1]
    qvals[order] = np.minimum(cummin, 1.0)
    return qvals


# -- Raw AnchorScore-MLLM p-values (from analysis/kruskal_4groups.py) --
# SCB5 9-model set matches mllm_full.json (excludes the expanded-CLIP-group
# LLaVA-1.5-13B / LLaVA-NeXT-Mistral-7B, consistent with the paper's 15-test FDR)
with open(OUT_DIR / "kruskal_4groups.json") as f:
    _kruskal = json.load(f)
_SCB5_9 = {
    "Qwen2-VL-7B", "LLaVA-1.5-7B", "Qwen3.5-27B", "Qwen3.6-27B",
    "Qwen3.5-35B", "Qwen3.6-35B", "Gemma-3-27B", "Gemma4-31B", "Gemma4-26B",
}
raw_scb5 = {
    model: entry["p"]
    for model, entry in _kruskal["per_model_rho"].items()
    if model in _SCB5_9
}

raw_stanford40 = {
    "qwen3.5:27b":      8.72e-05,
    "qwen3.5:35b-a3b":  9.21e-06,
    "qwen3.6:27b":      1.36e-07,
    "qwen3.6:35b-a3b":  1.36e-07,
    "gemma4:26b":       1.53e-06,
    "gemma4:31b":       3.23e-06,
}

# partial correlation p (from analysis/compute_partial_ci.py)
with open(BASE / "results" / "partial_correlation_bootstrap.json") as f:
    _partial = json.load(f)
partial_scb5 = {
    model: entry["partial_p"]
    for model, entry in _partial.items()
    if model != "_metadata"
}

# Mantel test exact p-values (from mantel_test_results.json)
mantel_p = {
    "CLIP OpenAI L/14 vs CLIP LAION L/14":  4.96e-05,
    "CLIP OpenAI L/14 vs CLIP B/32":         4.96e-05,
    "CLIP OpenAI L/14 vs Qwen2-VL":          4.96e-05,
    "CLIP LAION L/14 vs CLIP B/32":          4.96e-05,
    "CLIP LAION L/14 vs Qwen2-VL":           4.96e-05,
    "CLIP B/32 vs Qwen2-VL":                 4.96e-05,
}

# Stanford40 Mantel
mantel_stanford = {
    "CLIP vs Qwen2-VL (Stanford40)": 9.999e-05,
}


def run(entries, label):
    """entries: {description: p_value}. Returns list with q and sig."""
    descs = list(entries.keys())
    pvals = np.array([entries[d] for d in descs])
    qvals = bh_q(pvals)
    res = []
    for d, p, q in zip(descs, pvals, qvals):
        sig = "✓" if q < 0.05 else ""
        res.append({"test": d, "p": float(p), "q_fdr": round(float(q), 6),
                    "fdr_0.05": sig})
    return res


def main():
    all_results = {}

    print("FDR (Benjamini-Hochberg) correction\n", flush=True)

    # 1. Raw correlations (SCB5 + Stanford40 together)
    print("--- Raw AnchorScore-MLLM ---")
    raw_all = {**raw_scb5, **{f"(S40) {k}": v for k, v in raw_stanford40.items()}}
    raw_res = run(raw_all, "raw")
    all_results["raw"] = raw_res
    for r in raw_res:
        print(f"  p={r['p']:.2e} q={r['q_fdr']:.4f} {r['fdr_0.05']} {r['test']}")

    # 2. Partial correlations (SCB5 only — the key mediation claim)
    print("\n--- Partial correlations (SCB5, Table 1) ---")
    part_res = run(partial_scb5, "partial_scb5")
    all_results["partial_scb5"] = part_res
    for r in part_res:
        print(f"  p={r['p']:.4f} q={r['q_fdr']:.4f} {r['fdr_0.05']} {r['test']}")

    # 3. Mantel tests
    print("\n--- Mantel tests (exact) ---")
    mantel_all = {**mantel_p, **mantel_stanford}
    mantel_res = run(mantel_all, "mantel")
    all_results["mantel"] = mantel_res
    for r in mantel_res:
        print(f"  p={r['p']:.2e} q={r['q_fdr']:.4f} {r['fdr_0.05']} {r['test']}")

    # 4. All combined
    print("\n--- ALL COMBINED ---")
    all_all = {**raw_all, **partial_scb5, **mantel_all}
    all_res = run(all_all, "combined")
    all_results["combined"] = all_res
    for r in all_res:
        print(f"  p={r['p']:.2e} q={r['q_fdr']:.4f} {r['fdr_0.05']} {r['test']}")

    # 5. Visual-layer perturbation grid (Section 4.8)
    print("\n--- Visual-layer perturbation grid (Sec 4.8) ---")
    with open(BASE / "results" / "visual_perturbation_sync_all.json") as f:
        _sync = json.load(f)
    ds = _sync["delta_sync"]["accuracy"]
    pairs = list(ds["sigma_2"].keys())
    sigma_names = ["sigma_1", "sigma_2", "sigma_4", "sigma_8"]

    per_level = {}
    for s in sigma_names:
        per_level[s] = run(
            {f"{s} | {pr}": ds[s][pr]["spearman_p"] for pr in pairs}, s)
    all_results["perturbation_per_level"] = per_level
    for s in sigma_names:
        surv = [r["test"] for r in per_level[s] if r["fdr_0.05"]]
        print(f"  {s}: {len(surv)}/{len(pairs)} survive level-wise BH")

    merged = run(
        {f"merged | {pr}": _sync["merged"]["accuracy"][pr]["p_perm_class_block"]
         for pr in pairs}, "merged")
    all_results["perturbation_merged"] = merged
    surv = [r["test"] for r in merged if r["fdr_0.05"]]
    print(f"  merged: {len(surv)}/{len(pairs)} survive BH")

    grid30 = {}
    for s in sigma_names:
        for pr in pairs:
            grid30[f"{s} | {pr}"] = ds[s][pr]["spearman_p"]
    for pr in pairs:
        grid30[f"merged | {pr}"] = _sync["merged"]["accuracy"][pr]["p_perm_class_block"]
    grid_res = run(grid30, "perturbation_grid_30")
    all_results["perturbation_grid_30"] = grid_res
    surv = [r["test"] for r in grid_res if r["fdr_0.05"]]
    print(f"  full 30-test grid: {len(surv)} survive BH")

    static = run(
        {f"static | {pr}": _sync["static_control"]["accuracy"][pr]["p_perm"]
         for pr in pairs}, "static")
    all_results["perturbation_static"] = static
    surv = [r["test"] for r in static if r["fdr_0.05"]]
    print(f"  static control: {len(surv)}/{len(pairs)} survive BH")

    path = OUT_DIR / "fdr_correction.json"
    json.dump(all_results, open(path, "w"), indent=2)
    print(f"\nSaved -> {path}")


if __name__ == "__main__":
    main()
