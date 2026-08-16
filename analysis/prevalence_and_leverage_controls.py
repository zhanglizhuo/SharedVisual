#!/usr/bin/env python3
"""Prevalence (class-frequency) and leverage controls for the review round.

Two zero-new-training robustness checks requested by reviewers:

1. Class-frequency common-cause control (SCB5):
   - Spearman(AnchorScore, log n_c) and Spearman(mean MLLM acc, log n_c)
   - partial Spearman of AnchorScore vs cross-model mean accuracy
     controlling for log class count (raw vs partial)
   Reported on both the 8-class TeacherBehavior panel and the pooled
   13-class panel.

2. Stanford40 leverage analysis:
   - jackknife over the 40 classes (max |delta rho| and which class)
   - rho after removing the 3 and 5 lowest-AnchorScore classes
   - per-model rho with and without the bottom-3 anchor classes

Inputs:
  data/visualanchor_snapshot/01_core/paper_data/mllm_full.json
  data/visualanchor_snapshot/02_robustness/multi_backbone/backbone_results.json
  data/visualanchor_snapshot/02_robustness/stanford40/anchor_scores.json
  data/visualanchor_snapshot/02_robustness/stanford40/ollama_results.json
Output:
  results/prevalence_and_leverage_controls.json
"""
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

BASE = Path(__file__).resolve().parents[1]
MLLM_PATH = BASE / "data/visualanchor_snapshot/01_core/paper_data/mllm_full.json"
BACKBONE_PATH = BASE / "data/visualanchor_snapshot/02_robustness/multi_backbone/backbone_results.json"
S40_ANCHOR_PATH = BASE / "data/visualanchor_snapshot/02_robustness/stanford40/anchor_scores.json"
S40_OLLAMA_PATH = BASE / "data/visualanchor_snapshot/02_robustness/stanford40/ollama_results.json"
OUT_PATH = BASE / "results/prevalence_and_leverage_controls.json"


def resid(a, b):
    x = np.asarray(b, float)
    y = np.asarray(a, float)
    lr = np.polyfit(x, y, 1)
    return y - np.polyval(lr, x)


def partial_spearman(x, y, cov):
    rx = resid(x, cov)
    ry = resid(y, cov)
    r, p = spearmanr(rx, ry)
    return float(r), float(p)


def main():
    mllm = json.load(open(MLLM_PATH))
    bb = json.load(open(BACKBONE_PATH))
    s40_anchor = json.load(open(S40_ANCHOR_PATH))
    s40_ollama = json.load(open(S40_OLLAMA_PATH))

    out = {}

    # ---------- 1. SCB5 class-frequency controls ----------
    sub_datasets = ["TeacherBehavior", "HandriseReadWrite", "BowTurnHead"]
    freq = {"scb5_8_classes": {}, "scb5_13_classes": {}}

    for tag, subs in [("scb5_8_classes", ["TeacherBehavior"]),
                      ("scb5_13_classes", sub_datasets)]:
        classes = []
        anchor = {}
        mean_mllm = {}
        n = {}
        for sub in subs:
            laion = bb["laion_l14"]["results"][sub]["per_class_acc"]
            models = list(mllm[sub].keys())
            for c, v in laion.items():
                classes.append(c)
                anchor[c] = v["acc"]
                n[c] = v["n"]
                mean_mllm[c] = float(np.mean([mllm[sub][m][c] for m in models]))
        cs = classes
        logn = {c: math.log(n[c]) for c in cs}
        r_anchor_n, p_anchor_n = spearmanr([anchor[c] for c in cs],
                                           [logn[c] for c in cs])
        r_mllm_n, p_mllm_n = spearmanr([mean_mllm[c] for c in cs],
                                       [logn[c] for c in cs])
        r_raw, p_raw = spearmanr([anchor[c] for c in cs],
                                 [mean_mllm[c] for c in cs])
        r_part, p_part = partial_spearman(
            [anchor[c] for c in cs], [mean_mllm[c] for c in cs],
            [logn[c] for c in cs])
        freq[tag] = {
            "n_classes": len(cs),
            "n_range": [min(n[c] for c in cs), max(n[c] for c in cs)],
            "spearman_anchor_log_n": [round(r_anchor_n, 3), round(p_anchor_n, 3)],
            "spearman_mllm_log_n": [round(r_mllm_n, 3), round(p_mllm_n, 3)],
            "spearman_anchor_mean_mllm_raw": [round(r_raw, 3), round(p_raw, 3)],
            "spearman_anchor_mean_mllm_partial_log_n": [round(r_part, 3), round(p_part, 3)],
        }
    out["class_frequency"] = freq

    # ---------- 2. Stanford40 leverage ----------
    pa = s40_anchor["per_class_acc"]
    models = s40_ollama["models"]
    classes = list(pa.keys())
    a = np.array([pa[c]["acc"] for c in classes], float)
    mean_m = np.array([np.mean([models[m]["per_class_acc"][c]["acc"] for m in models])
                       for c in classes], float)
    r0, p0 = spearmanr(a, mean_m)

    jackknife = []
    for i, c in enumerate(classes):
        idx = [j for j in range(len(classes)) if j != i]
        r, _ = spearmanr(a[idx], mean_m[idx])
        jackknife.append({"class": c, "rho_without": round(float(r), 3),
                          "delta": round(abs(r - r0), 3)})
    jackknife.sort(key=lambda d: d["delta"], reverse=True)

    low = sorted(classes, key=lambda c: pa[c]["acc"])[:5]
    idx3 = [j for j, c in enumerate(classes) if c not in set(low[:3])]
    idx5 = [j for j, c in enumerate(classes) if c not in set(low)]

    def rho_of(idx):
        r, p = spearmanr(a[idx], mean_m[idx])
        return round(float(r), 3), round(float(p), 5)

    per_model = []
    for mname in models:
        acc = np.array([models[mname]["per_class_acc"][c]["acc"] for c in classes],
                       float)
        r_full, _ = spearmanr(a, acc)
        r_t3, _ = spearmanr(a[idx3], acc[idx3])
        per_model.append({"model": mname, "rho_full": round(float(r_full), 3),
                          "rho_no_bottom3": round(float(r_t3), 3)})

    out["stanford40_leverage"] = {
        "n_classes": len(classes),
        "rho_anchor_mean": [round(float(r0), 3), round(float(p0), 6)],
        "n_classes_below_60pct_anchor": int(sum(1 for c in classes if pa[c]["acc"] < 60)),
        "lowest_5_anchor_classes": {c: round(pa[c]["acc"], 1) for c in low},
        "jackknife_top5": jackknife[:5],
        "rho_removing_3_lowest_anchor": rho_of(idx3),
        "rho_removing_5_lowest_anchor": rho_of(idx5),
        "per_model_full_vs_no_bottom3": per_model,
    }

    OUT_PATH.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
