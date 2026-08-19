#!/usr/bin/env python3
"""Link B analysis: preregistered predictions P1-P4 (patched per review).

Inputs : results/linkb/{arm}_summary.json (per-class accuracy per arm)
         results/linkb/features/{arm}_val.npz (cached features -> 13-class
         pooled geometry matrices, PRIMARY reference; TB-8 committed matrices
         secondary)
Output : results/linkb/analysis_output.json

Preregistered (locked before any training):
- P1 ordinal: ceiling disclosed — min one-sided exact p = 1/6; descriptive.
- P2 matrix: ceiling disclosed — min two-sided exact p = 2/6; descriptive.
- P3: 10k class-bootstrap 95% CI of rho(C,S) - rho(C,R) excludes zero.
- Floor criterion: arm flagged floor-affected if mean acc < 20% or
  single-class collapse > 0.80; flagged arms void the causal reading of P1/P2.
- D-arm learnability confound: C-S is the primary causal pair; D directional.

NOT YET RUN — smoke-tested on synthetic profiles only.
"""
import itertools
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS = REPO_ROOT / "results"
RESP = RESULTS / "linkb"
FEAT = RESULTS / "linkb" / "features"

ARMS = ["clip_l_laion", "siglip_so400m", "dinov2_b14", "random_vitb"]
SHORT = {"clip_l_laion": "C", "siglip_so400m": "S",
         "dinov2_b14": "D", "random_vitb": "R"}
GEOM_TB8 = {("C", "S"): 0.957, ("C", "D"): 0.719, ("S", "D"): 0.772}  # secondary
BOOT_B = 10_000
BOOT_SEED = 20260801  # paper convention
FLOOR_MEAN_ACC = 20.0   # 13-class chance 7.7%; ~2.6x
FLOOR_COLLAPSE = 0.80   # paper's strong-prior/attractor threshold


def load_summaries():
    summaries = {}
    for arm in ARMS:
        p = RESP / f"{arm}_summary.json"
        if not p.exists():
            raise FileNotFoundError(f"{p} missing — run 03_eval_swap_variants.py")
        summaries[SHORT[arm]] = json.load(open(p))
    classes = sorted(set.intersection(
        *(set(s["per_class_acc"]) for s in summaries.values())))
    return summaries, classes


def boot_ci(x, y, b=BOOT_B, seed=BOOT_SEED):
    rng = np.random.RandomState(seed)
    n = len(x)
    out = np.empty(b)
    for i in range(b):
        idx = rng.randint(0, n, n)
        out[i] = spearmanr(x[idx], y[idx])[0]
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def boot_diff_ci(x, y1, y2, b=BOOT_B, seed=BOOT_SEED):
    """CI of spearman(x,y1) - spearman(x,y2), resampling classes jointly."""
    rng = np.random.RandomState(seed)
    n = len(x)
    out = np.empty(b)
    for i in range(b):
        idx = rng.randint(0, n, n)
        out[i] = (spearmanr(x[idx], y1[idx])[0]
                  - spearmanr(x[idx], y2[idx])[0])
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def pooled_geometry_13(classes):
    """PRIMARY geometry reference: 13-class centroid cosine matrices from the
    cached full-frame val features (same features feeding behavior)."""
    iu = np.triu_indices(len(classes), k=1)
    mats = {}
    for arm in ["clip_l_laion", "siglip_so400m", "dinov2_b14"]:
        p = FEAT / f"{arm}_val.npz"
        if not p.exists():
            return None  # features absent -> fall back to TB-8 secondary only
        d = np.load(p, allow_pickle=True)
        labels = list(d["labels"])
        feats = d["features"]
        cents = {}
        for c in classes:
            idx = [i for i, y in enumerate(labels) if y == c]
            if not idx:
                return None
            v = np.stack([feats[i].astype(np.float32).mean(axis=0)
                          for i in idx]).mean(axis=0)
            cents[c] = v / (np.linalg.norm(v) + 1e-9)
        M = np.zeros((len(classes), len(classes)))
        for i, ci in enumerate(classes):
            for j, cj in enumerate(classes):
                M[i, j] = float(cents[ci] @ cents[cj])
        mats[SHORT[arm]] = M[iu]
    out = {}
    for a, b in itertools.combinations(mats, 2):
        out[(a, b)] = round(float(spearmanr(mats[a], mats[b])[0]), 3)
    return out


def analyze():
    summaries, classes = load_summaries()
    n = len(classes)
    prof = {a: np.array([summaries[a]["per_class_acc"][c] for c in classes])
            for a in summaries}

    behavior = {}
    for a, b in itertools.combinations(["C", "S", "D", "R"], 2):
        r = spearmanr(prof[a], prof[b])[0]
        ci = boot_ci(prof[a], prof[b])
        behavior[(a, b)] = {"rho": round(float(r), 3),
                            "ci95": [round(ci[0], 3), round(ci[1], 3)]}

    # ---- floor-affected flags (preregistered; voids causal reading of P1/P2)
    floor = {}
    for a, s in summaries.items():
        flagged = (s["mean_acc"] < FLOOR_MEAN_ACC
                   or s.get("collapse_top_class_frac", 0) > FLOOR_COLLAPSE)
        floor[a] = {"mean_acc": s["mean_acc"],
                    "collapse": s.get("collapse_top_class_frac"),
                    "floor_affected": bool(flagged)}
    any_floor = any(v["floor_affected"] for k, v in floor.items() if k != "R")
    all_floor = all(v["floor_affected"] for k, v in floor.items() if k != "R")

    # ---- P1 ordinal (ceiling 1/6, disclosed)
    obs = (behavior[("C", "S")]["rho"], behavior[("S", "D")]["rho"],
           behavior[("C", "D")]["rho"])
    orderings = list(itertools.permutations(obs))
    p_p1 = sum(1 for o in orderings
               if o[0] == max(o) and o[1] >= o[2]) / len(orderings)

    # ---- P2 matrix (ceiling 2/6 two-sided, disclosed); primary 13-class geom
    geom13 = pooled_geometry_13(classes)
    if geom13 is not None:
        g = [geom13[("C", "S")], geom13[("S", "D")], geom13[("C", "D")]]
        ref = "pooled_13_from_cached_features"
    else:
        g = [GEOM_TB8[("C", "S")], GEOM_TB8[("S", "D")], GEOM_TB8[("C", "D")]]
        ref = "tb8_committed_secondary"
    bvals = [obs[0], obs[1], obs[2]]
    rho_matrix = spearmanr(g, bvals)[0]
    perms = list(itertools.permutations(bvals))
    p_p2 = sum(1 for p in perms if abs(spearmanr(g, list(p))[0])
               >= abs(rho_matrix) - 1e-12) / len(perms)

    # ---- P3 operationalized: bootstrap difference CI excludes zero
    dci = boot_diff_ci(prof["C"], prof["S"], prof["R"])
    p3_pass = dci[0] > 0 or dci[1] < 0

    # ---- P4 validity
    r_acc = floor["R"]["mean_acc"]
    min_other = min(v["mean_acc"] for k, v in floor.items() if k != "R")
    # P4(b) (preregistered): R's pair agreements must be the lowest of the six
    r_pairs = {str(k): behavior[k]["rho"] for k in behavior if "R" in k}
    non_r = {str(k): behavior[k]["rho"] for k in behavior if "R" not in k}
    p4b_pairs_lowest = all(r <= min(non_r.values()) for r in r_pairs.values())
    p4 = {"random_markedly_lower": r_acc < 0.5 * min_other,
          "p4b_r_pairs_lowest": bool(p4b_pairs_lowest),
          "p4b_r_pair_rhos": r_pairs,
          "p4b_non_r_pair_rhos": non_r,
          "random_acc": r_acc, "min_other_acc": min_other,
          "collapse_R": floor["R"]["collapse"]}

    out = {
        "n_classes": n, "classes": classes,
        "behavior_agreement": {f"{a}-{b}": v for (a, b), v in behavior.items()},
        "geometry": {"primary_ref": ref,
                      "pooled_13": ({f"{a}-{b}": v for (a, b), v in geom13.items()}
                                    if geom13 is not None else None),
                      "tb8_secondary": {f"{a}-{b}": GEOM_TB8[(a, b)]
                                        for (a, b) in GEOM_TB8}},
        "floor_flags": floor,
        "any_trainable_floor_affected": bool(any_floor),
        "all_trainable_floor_affected": bool(all_floor),
        "P1_ordinal": {"observed": {"C-S": obs[0], "S-D": obs[1], "C-D": obs[2]},
                        "predicted_order_holds": bool(
                            obs[0] == max(obs) and obs[1] >= obs[2]),
                        "exact_p": round(p_p1, 4),
                        "ceiling_disclosed": "min one-sided exact p = 1/6; "
                        "perfect match is descriptive support"},
        "P2_matrix": {"rho_geometry_vs_behavior": round(float(rho_matrix), 3),
                       "exact_p": round(p_p2, 4),
                       "ceiling_disclosed": "min two-sided exact p = 2/6"},
        "P3_lm_floor": {"criterion": "95% CI of rho(C,S)-rho(C,R) excludes 0",
                          "diff_ci95": [round(dci[0], 3), round(dci[1], 3)],
                          "passes": bool(p3_pass)},
        "P4_validity": p4,
        "preregistration_note": "P1/P2 descriptive under disclosed ceilings; "
        "causal reading of P1/P2 void if any trainable arm floor-affected; "
        "C-S is the primary causal pair (D-arm learnability confound "
        "preregistered)",
        "bootstrap": {"B": BOOT_B, "seed": BOOT_SEED, "unit": "class"},
    }
    def _clean(o):
        if isinstance(o, float) and (np.isnan(o) or np.isinf(o)):
            return None
        if isinstance(o, dict):
            return {k: _clean(v) for k, v in o.items()}
        if isinstance(o, list):
            return [_clean(v) for v in o]
        return o
    out = _clean(out)
    (RESULTS / "linkb").mkdir(parents=True, exist_ok=True)
    (RESULTS / "linkb" / "analysis_output.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    analyze()
