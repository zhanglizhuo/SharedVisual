#!/usr/bin/env python3
"""Link B analysis: preregistered predictions P1-P4.

Inputs : results/linkb/responses/{arm}_summary.json (per-class accuracy per arm)
         results/mantel_test_results.json (encoder geometry, committed)
Output : results/linkb/linkb_analysis.json

P1 (ordinal): rho_diff(C,S) > rho_diff(S,D) ~/> rho_diff(C,D), mirroring
     geometry 0.957 > 0.772 ~/> 0.719.
P2 (matrix): behavior-agreement ranks correlate with geometry-agreement ranks
     (3 pairs, exact permutation).
P3 (LM floor): rho_diff(C,S) exceeds the shared-LM+task floor estimated by the
     random-arm agreements rho_diff(C,R), rho_diff(S,R).
P4 (validity): random arm markedly weaker and not geometry-ordered.

NOT YET RUN — code path validated against synthetic summaries only.
"""
import itertools
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS = REPO_ROOT / "results"
RESP = RESULTS / "linkb" / "responses"

ARMS = ["clip_l_laion", "siglip_so400m", "dinov2_b14", "random_vitb"]
SHORT = {"clip_l_laion": "C", "siglip_so400m": "S",
         "dinov2_b14": "D", "random_vitb": "R"}
GEOM_PAIRS = {("C", "S"): 0.957, ("C", "D"): 0.719, ("S", "D"): 0.772}
BOOT_B = 10_000
BOOT_SEED = 20260801  # paper convention


def load_profiles():
    profiles = {}
    for arm in ARMS:
        p = RESP / f"{arm}_summary.json"
        if not p.exists():
            raise FileNotFoundError(f"{p} missing — run 03_eval_swap_variants.py")
        profiles[SHORT[arm]] = json.load(open(p))["per_class_acc"]
    classes = sorted(set.intersection(*(set(v) for v in profiles.values())))
    return {a: np.array([profiles[a][c] for c in classes]) for a in profiles}, classes


def boot_ci(x, y, b=BOOT_B, seed=BOOT_SEED):
    rng = np.random.RandomState(seed)
    n = len(x)
    out = np.empty(b)
    for i in range(b):
        idx = rng.randint(0, n, n)
        out[i] = spearmanr(x[idx], y[idx])[0]
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def analyze():
    prof, classes = load_profiles()
    n = len(classes)
    behavior = {}
    for a, b in itertools.combinations(["C", "S", "D", "R"], 2):
        r = spearmanr(prof[a], prof[b])[0]
        ci = boot_ci(prof[a], prof[b])
        behavior[(a, b)] = {"rho": round(float(r), 3),
                            "ci95": [round(ci[0], 3), round(ci[1], 3)]}

    # P1 ordinal: exact permutation over the 3! orderings of the three values
    obs = (behavior[("C", "S")]["rho"], behavior[("S", "D")]["rho"],
           behavior[("C", "D")]["rho"])
    orderings = list(itertools.permutations(obs))
    predicted = (max(obs),)  # P1: C-S is the maximum; S-D >= C-D
    rank_ok = obs[0] == max(obs) and obs[1] >= obs[2]
    p_p1 = sum(1 for o in orderings
               if o[0] == max(o) and o[1] >= o[2]) / len(orderings)

    # P2 matrix: geometry vs behavior across the 3 shared pairs
    g = [GEOM_PAIRS[k] for k in [("C", "S"), ("S", "D"), ("C", "D")]]
    bvals = [obs[0], obs[1], obs[2]]
    rho_matrix = spearmanr(g, bvals)[0]
    # exact permutation over 3! pairings
    perms = list(itertools.permutations(bvals))
    p_p2 = sum(1 for p in perms if abs(spearmanr(g, list(p))[0])
               >= abs(rho_matrix) - 1e-12) / len(perms)

    # P3 LM floor
    floor = max(abs(behavior[("C", "R")]["rho"]),
                abs(behavior[("S", "R")]["rho"]),
                abs(behavior[("D", "R")]["rho"]))
    p3_pass = obs[0] > floor

    # P4 validity
    summaries = {a: json.load(open(RESP / f"{ARMS[i]}_summary.json"))
                 for i, a in enumerate(SHORT.values())}
    r_acc = summaries["R"]["mean_acc"]
    min_other = min(v["mean_acc"] for k, v in summaries.items() if k != "R")
    p4 = {"random_markedly_lower": r_acc < 0.5 * min_other,
          "random_acc": r_acc, "min_other_acc": min_other,
          "collapse_frac_R": summaries["R"].get("collapse_top_class_frac")}

    out = {
        "n_classes": n,
        "classes": classes,
        "behavior_agreement": {f"{a}-{b}": v for (a, b), v in behavior.items()},
        "geometry_reference": {f"{a}-{b}": GEOM_PAIRS[(a, b)]
                               for (a, b) in GEOM_PAIRS},
        "P1_ordinal": {"observed": {"C-S": obs[0], "S-D": obs[1], "C-D": obs[2]},
                        "predicted_order_holds": bool(rank_ok),
                        "exact_p": round(p_p1, 4)},
        "P2_matrix": {"rho_geometry_vs_behavior": round(float(rho_matrix), 3),
                       "exact_p": round(p_p2, 4)},
        "P3_lm_floor": {"floor_max_abs_random_agreement": round(floor, 3),
                         "C-S_exceeds_floor": bool(p3_pass)},
        "P4_validity": p4,
        "bootstrap": {"B": BOOT_B, "seed": BOOT_SEED, "unit": "class"},
    }
    (RESULTS / "linkb").mkdir(parents=True, exist_ok=True)
    (RESULTS / "linkb" / "linkb_analysis.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    analyze()
