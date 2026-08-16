#!/usr/bin/env python3
"""A4 — Power analysis for Spearman correlation at sample sizes n=8, 13, 40.

Uses Fisher-z asymptotic approximation:
    SE_z = 1/sqrt(n-3)
    Power(ρ) = 1 - Φ(z_{1-α/2} - |atanh(ρ)|·sqrt(n-3))
             + Φ(-z_{1-α/2} - |atanh(ρ)|·sqrt(n-3))
             [two-sided test]

NOTE: asymptotic for Spearman — exact for large n, approximate for n=8,13.
We annotate ``n<=20: approximate`` to be conservative.

Also reports the analytic minimum detectable |ρ| at 80% power.

Output: results/power_analysis.json
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
from scipy.stats import norm


OUT_DIR.mkdir(parents=True, exist_ok=True)
ALPHA = 0.05
z_crit = norm.ppf(1 - ALPHA / 2)


def power_at_rho(n, rho):
    """Fisher-z asymptotic power for two-sided Spearman/Pearson test."""
    z_effect = abs(np.arctanh(rho)) * np.sqrt(n - 3)
    pw = (1.0 - norm.cdf(z_crit - z_effect)
          + norm.cdf(-z_crit - z_effect))
    return float(pw)


def min_detectable(n, target=0.80):
    """Analytic solution: solve for rho where power=target.
    Power = 1 - Φ(z_crit - z_eff) + Φ(-z_crit - z_eff), z_eff = atanh(ρ)·sqrt(n-3).
    For target >= 0.5 the second tail term is negligible, so:
    Φ(z_crit - atanh(ρ)·sqrt(n-3)) = 1 - target
    → z_crit - atanh(ρ)·sqrt(n-3) = Φ^{-1}(1 - target) = -norm.ppf(target)
    → atanh(ρ) = (z_crit + norm.ppf(target)) / sqrt(n-3)
    → ρ = tanh(...)"""
    z_target = norm.ppf(target)
    z_rho = (z_crit + z_target) / np.sqrt(n - 3)
    if z_rho <= 0:
        return (1.0, 1.0)   # always detectable
    return (float(np.tanh(z_rho)), target)


def main():
    print("Power analysis (Fisher-z asymptotic)", flush=True)
    print(f"  α = {ALPHA}, z_crit = {z_crit:.4f}", flush=True)

    results = {"alpha": ALPHA, "z_critical": float(z_crit),
               "note": "Fisher-z asymptotic; approximate for n<=20",
               "power_curves": [], "min_detectable_rho": [],
               "power_for_observed_cprime": []}

    for n in [8, 13, 20, 40]:
        approx = " (approx)" if n <= 20 else ""
        print(f"\n--- n = {n}{approx} ---")
        curve = {"n": n, "approximate": n <= 20, "power": []}
        for rho_val in np.arange(0.05, 1.05, 0.05):
            pw = power_at_rho(n, rho_val)
            curve["power"].append({"rho": round(rho_val, 3), "power": round(pw, 4)})
            if rho_val <= 0.85 or round(rho_val * 20) % 20 == 0:
                print(f"  ρ={rho_val:.2f} → power={pw:.4f}", flush=True)
        results["power_curves"].append(curve)

        rho_min, _ = min_detectable(n)
        results["min_detectable_rho"].append({"n": n, "min_rho_80pct": round(rho_min, 3)})
        print(f"  Min |ρ| at 80% power: {rho_min:.3f}", flush=True)

    # Power for observed c' magnitudes
    print("\n\nPower to detect residual (c') effects:")
    for label, n, rhos in [
        ("SCB5 n=13", 13, [0.10, 0.20, 0.30, 0.45, 0.55]),
        ("Stanford40 n=40", 40, [0.05, 0.10, 0.15, 0.20, 0.25])]:
        for rho_val in rhos:
            pw = power_at_rho(n, rho_val)
            results["power_for_observed_cprime"].append(
                {"dataset": label, "n": n, "rho": rho_val, "power": round(pw, 4)})
            print(f"  {label}: ρ={rho_val:.2f} → power={pw:.4f}", flush=True)

    path = OUT_DIR / "power_analysis.json"
    json.dump(results, open(path, "w"), indent=2)
    print(f"\nSaved -> {path}")


if __name__ == "__main__":
    main()
