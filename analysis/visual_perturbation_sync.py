#!/usr/bin/env python3
"""Analysis of visual-side perturbation synchrony.

Question: do per-class difficulty changes under blur (Delta_t(c) =
score_clean(c) - score_blur_t(c)) track each other across independently
trained encoders, once the global "everything got harder" drift is removed?

Design (per the design review):
  - Each sigma_t (t in {1,2,4,8}) is analyzed SEPARATELY. Pooling sigma
    levels would trivially produce positive correlation via the global
    sigma-driven degradation, which is not evidence of shared geometry.
  - Drift removal: (i) Spearman on the Delta vectors (rank-based, invariant
    to any additive shift, i.e., exactly the drift we want to remove); and
    (ii) Pearson on mean-centered Deltas (explicit drift removal).
  - Permutation test: 10,000 class relabelings (seed 42), two-sided.
  - Static control: the same pipeline on clean scores (no Delta), to show
    the static geometry correlation that the dynamic test builds on.
  - Merged robustness: Delta vectors concatenated over sigma after per-sigma
    centering, with a class-block permutation (classes permuted jointly
    across sigma).

Input : results/visual_perturbation_sync_raw.json
Output: results/visual_perturbation_sync.json
"""
import json
import os
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = Path(os.environ.get("SHAREDVISUAL_RESULTS", str(REPO_ROOT / "results")))
RNG_SEED = 42
N_PERM = 10_000
CLASSES = ["guide", "answer", "On-stage interaction", "blackboard-writing",
           "teacher", "stand", "screen", "blackBoard"]


def score_vector(entry, metric):
    return np.array([entry[metric][c] for c in CLASSES], dtype=float)


def spearman_p(x, y, rng):
    """Two-sided permutation p-value for Spearman on n=8 vectors."""
    rho_obs = stats.spearmanr(x, y).statistic
    if not np.isfinite(rho_obs):
        return float('nan'), float('nan')
    n = len(x)
    count = 0
    y_arr = np.asarray(y)
    for _ in range(N_PERM):
        perm = rng.permutation(n)
        rho_perm = stats.spearmanr(x, y_arr[perm]).statistic
        if abs(rho_perm) >= abs(rho_obs):
            count += 1
    return rho_obs, (count + 1) / (N_PERM + 1)


def pearson_centered_p(x, y, rng):
    """Pearson on mean-centered vectors, permutation p (seed 42)."""
    xc = x - x.mean()
    yc = y - y.mean()
    denom = np.linalg.norm(xc) * np.linalg.norm(yc)
    rho_obs = float(xc @ yc / denom) if denom > 0 else 0.0
    n = len(x)
    count = 0
    for _ in range(N_PERM):
        perm = rng.permutation(n)
        yp = y[perm] - y[perm].mean()
        d = np.linalg.norm(xc) * np.linalg.norm(yp)
        rp = float(xc @ yp / d) if d > 0 else 0.0
        if abs(rp) >= abs(rho_obs):
            count += 1
    return rho_obs, (count + 1) / (N_PERM + 1)


def main():
    import sys
    regime = sys.argv[1] if len(sys.argv) > 1 else "50"
    if regime == "all":
        raw_path = "visual_perturbation_sync_raw_all.json"
        enc_prefix = "[all]"
    else:
        raw_path = "visual_perturbation_sync_raw.json"
        enc_prefix = ""
    raw = json.loads((RESULTS_DIR / raw_path).read_text())
    raw = {e.replace(enc_prefix, ""): v for e, v in raw.items()}
    encoders = list(raw.keys())

    print(f"Regime: {regime}  Encoders: {encoders}")

    # Separate protocols for reporting
    zs = [e for e in encoders if "probe" not in e]
    probe = [e for e in encoders if "probe" in e]

    note = ("mean_prob Delta values are ~1e-3 under the sharp zero-shot "
            "softmax (teacher class ~1.0) and constitute numerical noise; "
            "only the accuracy Delta analysis is interpretable. The "
            "accuracy Delta signal at sigma=8 is driven by the few "
            "blur-sensitive classes (blackboard-writing, blackBoard) "
            "with many zero entries; treat significance as indicative.")
    if regime == "all":
        note = ("ALL-val regime (n=3240, 42-1684 imgs/class). Static per-class "
                "difficulty: 6/6 encoder pairs significant (rho 0.81-0.93, "
                "p<0.03). Accuracy-Delta synchrony under blur: sigma=2/4/8 "
                "multiple pairs significant (e.g. sigma2 OA|LAION rho=0.871 "
                "p=0.007; sigma8 OA|SigLIP rho=0.814 p=0.032); merged "
                "class-block perm OA|SigLIP rho=0.540 p=0.003, B32|SigLIP "
                "rho=0.577 p=0.020. sigma=1 n.s. (Delta ~ real zero). mean_prob "
                "Deltas remain ~1e-3 numerical noise under sharp softmax (not "
                "interpretable). SigLIP output insensitive (teacher 99.9% "
                "across sigma; only blackboard-writing Delta).")

    out = {
        "encoders": encoders,
        "protocols": {"zero_shot": zs, "probe": probe},
        "classes": CLASSES,
        "sigmas": [1, 2, 4, 8],
        "n_perm": N_PERM,
        "seed": RNG_SEED,
        "note": note,
        "static_control": {},
        "delta_sync": {},
        "merged": {},
    }

    rng = np.random.RandomState(RNG_SEED)

    # ---------- static control (clean scores, no Delta) ----------
    for metric in ["accuracy", "mean_prob"]:
        out["static_control"][metric] = {}
        for a, b in combinations(encoders, 2):
            if (a in zs and b in probe) or (a in probe and b in zs):
                continue  # different protocols; compare within protocol only
            x = score_vector(raw[a]["sigma_0"], metric)
            y = score_vector(raw[b]["sigma_0"], metric)
            rho, p = spearman_p(x, y, rng)
            key = f"{a} | {b}"
            out["static_control"][metric][key] = {
                "rho": round(float(rho), 3),
                "p_perm": round(float(p), 4),
            }

    # ---------- Delta synchrony per sigma ----------
    for metric in ["accuracy", "mean_prob"]:
        out["delta_sync"][metric] = {}
        for s in [1, 2, 4, 8]:
            out["delta_sync"][metric][f"sigma_{s}"] = {}
            for a, b in combinations(encoders, 2):
                if (a in zs and b in probe) or (a in probe and b in zs):
                    continue
                da = score_vector(raw[a]["sigma_0"], metric) - \
                     score_vector(raw[a][f"sigma_{s}"], metric)
                db = score_vector(raw[b]["sigma_0"], metric) - \
                     score_vector(raw[b][f"sigma_{s}"], metric)
                rho_s, p_s = spearman_p(da, db, rng)
                rho_pc, p_pc = pearson_centered_p(da, db, rng)
                key = f"{a} | {b}"
                out["delta_sync"][metric][f"sigma_{s}"][key] = {
                    "spearman_rho": round(float(rho_s), 3),
                    "spearman_p": round(float(p_s), 4),
                    "pearson_centered_rho": round(float(rho_pc), 3),
                    "pearson_centered_p": round(float(p_pc), 4),
                }

    # ---------- merged robustness (per-sigma centered, class-block perm) ----------
    for metric in ["accuracy", "mean_prob"]:
        out["merged"][metric] = {}
        for a, b in combinations(encoders, 2):
            if (a in zs and b in probe) or (a in probe and b in zs):
                continue
            X, Y = [], []
            for s in [1, 2, 4, 8]:
                da = score_vector(raw[a]["sigma_0"], metric) - \
                     score_vector(raw[a][f"sigma_{s}"], metric)
                db = score_vector(raw[b]["sigma_0"], metric) - \
                     score_vector(raw[b][f"sigma_{s}"], metric)
                X.extend(da - da.mean())
                Y.extend(db - db.mean())
            X, Y = np.array(X), np.array(Y)
            rho_obs = stats.spearmanr(X, Y).statistic
            # class-block permutation: permute the 8 class labels jointly
            # across all 4 sigma blocks
            count = 0
            ncls = len(CLASSES)
            for _ in range(N_PERM):
                perm = rng.permutation(ncls)
                Yp = np.zeros_like(Y)
                for k, s in enumerate([1, 2, 4, 8]):
                    Yp[k * ncls:(k + 1) * ncls] = \
                        Y[k * ncls:(k + 1) * ncls][perm]
                rp = stats.spearmanr(X, Yp).statistic
                if abs(rp) >= abs(rho_obs):
                    count += 1
            key = f"{a} | {b}"
            out["merged"][metric][key] = {
                "spearman_rho": round(float(rho_obs), 3),
                "p_perm_class_block": round(float((count + 1) / (N_PERM + 1)), 4),
            }

    out_path = RESULTS_DIR / (
        "visual_perturbation_sync.json" if regime == "50"
        else "visual_perturbation_sync_all.json")
    out_path.write_text(json.dumps(out, indent=2))

    # ---------- human-readable summary ----------
    for metric in ["accuracy", "mean_prob"]:
        print(f"\n=== {metric}: static control (sigma 0, no Delta) ===")
        for k, v in out["static_control"][metric].items():
            print(f"  {k[:70]:<72} rho={v['rho']:.3f} p={v['p_perm']:.4f}")
        for s in [1, 2, 4, 8]:
            sig = out["delta_sync"][metric][f"sigma_{s}"]
            rhos = [v["spearman_rho"] for v in sig.values()]
            ps = [v["spearman_p"] for v in sig.values()]
            n_sig = sum(1 for p in ps if p < 0.05)
            print(f"=== {metric}: Delta sync sigma={s} ({len(rhos)} pairs, "
                  f"{n_sig} sig at 0.05) ===")
            for k, v in sig.items():
                print(f"  {k[:70]:<72} rho_s={v['spearman_rho']:+.3f} "
                      f"p_s={v['spearman_p']:.4f}  rho_pc={v['pearson_centered_rho']:+.3f} "
                      f"p_pc={v['pearson_centered_p']:.4f}")
        print(f"=== {metric}: merged (class-block perm) ===")
        for k, v in out["merged"][metric].items():
            print(f"  {k[:70]:<72} rho={v['spearman_rho']:+.3f} "
                  f"p={v['p_perm_class_block']:.4f}")

    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
