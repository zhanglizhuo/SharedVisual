#!/usr/bin/env python3
"""Reliability correction for class-level Spearman correlations.

Motivation (reviewer concern): per-class accuracy is a binomial proportion with
sampling variance p(1-p)/n_c; classes with tiny n_c are noisier, and unequal n_c
across classes attenuates the observed class-level correlation (unequal
measurement-error variance). We therefore report, alongside the raw Spearman
rho, a classical attenuation-corrected estimate.

Method (classical test theory, Spearman's correction for attenuation):
  rho_obs = correlation of observed per-class accuracies
  reliability of a per-class accuracy vector:
      rel = var_between(true class difficulty) / var(observed)
          = (var_obs - mean_c p_c(1-p_c)/n_c) / var_obs
  rho_corr = rho_obs / sqrt(rel_X * rel_Y)

Because Spearman operates on ranks, we apply the correction to the Pearson
correlation on the same data and additionally report the rank analogue using
ranks' observed variance and an estimate of rank-level noise variance obtained
by simulation under the binomial model (10,000 replicates: draw per-class
accuracies ~ Binomial(n_c, p_c)/n_c, compute Spearman with the anchor vector
held fixed, subtract the implied noise from the observed rank variance).

Also reported: a sample-size-weighted Spearman (weights proportional to the
harmonic-mean effective n of each class across the two variables), as a
robustness check that the correlation is not driven by the most precise
classes.

Applies to Stanford40 (n per class known for both sides) and, partially, to
SCB5 (anchor n known; MLLM n not persisted - six core models were evaluated on
the full validation crops, n=149-4,967 per class on TeacherBehavior; the four
LLaVA models at 100/class, with two classes restricted to 42-48 usable crops).
For SCB5 we report the anchor-side correction and a sensitivity band over the
reported MLLM n range.

Outputs: results/reliability_correction.json + printed summary.
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr, pearsonr

BASE = Path(__file__).resolve().parents[1]
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

from sharedvisual.paths import RESULTS_DIR, SEED_BOOTSTRAP
from sharedvisual.datasets import (
    load_scb5_model_accs,
    load_stanford40,
)

OUT = RESULTS_DIR
OUT.mkdir(parents=True, exist_ok=True)

RNG = np.random.default_rng(SEED_BOOTSTRAP)
N_SIM = 10000


def rel_from_noise(p, noise):
    """Reliability from observed vector p (percent) and per-class noise (%^2)."""
    p = np.asarray(p, float)
    var_obs = np.var(p, ddof=1)
    noise_mean = float(np.mean(np.asarray(noise, float)))
    rel = (var_obs - noise_mean) / var_obs
    return float(np.clip(rel, 1e-6, 1.0)), float(var_obs), noise_mean


def rel_from_counts(p, n):
    """Reliability of a per-class accuracy vector given per-class sample sizes.

    rel = (var_obs - mean_c p_c(1-p_c)/n_c) / var_obs, clipped to (0, 1].
    """
    p = np.asarray(p, float)
    n = np.asarray(n, float)
    pf = p / 100.0  # fraction scale; per-class accuracy is stored in percent
    noise = pf * (1 - pf) / np.where(n > 0, n, np.inf) * 1e4  # %^2 per class
    return rel_from_noise(p, noise)


def correct_pearson(x, y, nx, ny):
    """Spearman-correction applied to Pearson rho via attenuation formula."""
    r_obs = pearsonr(x, y)[0]
    rel_x, _, _ = rel_from_counts(x, nx)
    rel_y, _, _ = rel_from_counts(y, ny)
    r_corr = r_obs / np.sqrt(rel_x * rel_y)
    return {
        "pearson_raw": float(r_obs),
        "rel_x": rel_x,
        "rel_y": rel_y,
        "pearson_attenuation_corrected": float(np.clip(r_corr, -1, 1)),
    }


def simulate_rank_noise(x, y, nx, ny):
    """Bootstrap the class-level noise under the binomial model.

    For each replicate, redraw every class's accuracy for both variables from
    Binomial(n_c, p_c)/n_c and recompute Spearman rho. Returns the mean rho and
    the SD of the replicate rho (measurement-noise envelope).
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    nx = np.asarray(nx, float)
    ny = np.asarray(ny, float)
    if np.any(np.isnan(nx)) or np.any(np.isnan(ny)):
        return np.nan, np.nan
    rhos = np.empty(N_SIM)
    for i in range(N_SIM):
        xs = RNG.binomial(nx.astype(int), x / 100.0) / nx * 100.0
        ys = RNG.binomial(ny.astype(int), y / 100.0) / ny * 100.0
        rhos[i] = spearmanr(xs, ys).statistic
    return float(rhos.mean()), float(rhos.std(ddof=1))


def weighted_spearman(x, y, nx, ny):
    """Spearman with class weights ~ harmonic mean of effective n.

    Ranks are computed on the accuracy values, then the correlation is computed
    as the weighted Pearson correlation of the rank-transformed variables,
    with weights w_c = 1 / (1/nx_c + 1/ny_c) (i.e., precision-weighted).
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    nx = np.asarray(nx, float)
    ny = np.asarray(ny, float)
    from scipy.stats import rankdata
    rx = rankdata(x)
    ry = rankdata(y)
    w = 1.0 / (1.0 / nx + 1.0 / ny)
    w = w / w.sum()
    mx = np.sum(w * rx)
    my = np.sum(w * ry)
    cov = np.sum(w * (rx - mx) * (ry - my))
    vx = np.sqrt(np.sum(w * (rx - mx) ** 2))
    vy = np.sqrt(np.sum(w * (ry - my) ** 2))
    if vx == 0 or vy == 0:
        return None
    return float(cov / (vx * vy))


def build_result(model, x, y, nx, ny, extra=None):
    rho_obs, p_obs = spearmanr(x, y)
    pear = correct_pearson(x, y, nx, ny)
    mean_sim, sd_sim = simulate_rank_noise(x, y, nx, ny)
    wrho = weighted_spearman(x, y, nx, ny)
    res = {
        "model": model,
        "spearman_raw": float(rho_obs),
        "spearman_p": float(p_obs),
        "n_classes": int(len(x)),
        **pear,
        "rank_noise_sim_mean_rho": float(mean_sim) if not np.isnan(mean_sim) else None,
        "rank_noise_sim_sd": float(sd_sim) if not np.isnan(sd_sim) else None,
        "weighted_spearman": wrho,
    }
    if extra:
        res.update(extra)
    return res


def main():
    # ---------------- Stanford40 (both n known) ----------------
    classes, anchor, model_accs = load_stanford40()
    a = json.load(
        open(BASE / "data/visualanchor_snapshot/02_robustness/stanford40/anchor_scores.json")
    )
    anchor_n = np.array([a["per_class_acc"][c]["n"] for c in classes], float)
    o = json.load(
        open(BASE / "data/visualanchor_snapshot/02_robustness/stanford40/ollama_results.json")
    )
    mllm_n = {
        m: np.array([o["models"][m]["per_class_acc"][c]["n"] for c in classes], float)
        for m in model_accs
    }

    s40 = {}
    for m, accs in model_accs.items():
        s40[m] = build_result(m, anchor, accs, anchor_n, mllm_n[m])

    # aggregated: anchor vs cross-model mean accuracy
    # The mean of 6 MLLM accuracies has per-class sampling variance
    # (1/6^2) * sum_m p_m(1-p_m)/n_m (percent^2); use the exact effective
    # noise rather than a min-n shortcut.
    mean_acc = np.mean(np.stack([model_accs[m] for m in model_accs]), axis=0)
    agg_noise = np.zeros(mean_acc.shape)
    for m in model_accs:
        p = np.asarray(model_accs[m], float) / 100.0
        agg_noise += p * (1 - p) / mllm_n[m]
    agg_noise = agg_noise / len(model_accs) ** 2 * 1e4
    agg = build_result(
        "cross-model mean (6 MLLMs)", anchor, mean_acc, anchor_n,
        np.full(mean_acc.shape, np.nan),
    )
    rho_obs, p_obs = spearmanr(anchor, mean_acc)
    r_obs = pearsonr(anchor, mean_acc)[0]
    rel_x, _, _ = rel_from_counts(anchor, anchor_n)
    rel_y, var_y, noise_y = rel_from_noise(mean_acc, agg_noise)
    agg["spearman_raw"] = float(rho_obs)
    agg["spearman_p"] = float(p_obs)
    agg["pearson_raw"] = float(r_obs)
    agg["rel_x"] = rel_x
    agg["rel_y"] = rel_y
    agg["var_obs_y"] = var_y
    agg["noise_y"] = noise_y
    agg["pearson_attenuation_corrected"] = float(
        np.clip(r_obs / np.sqrt(rel_x * rel_y), -1, 1)
    )
    # rank-noise simulation with the exact per-class aggregated noise
    sim = np.empty(N_SIM)
    for i in range(N_SIM):
        ys = RNG.normal(mean_acc, np.sqrt(agg_noise))
        sim[i] = spearmanr(anchor, ys).statistic
    agg["rank_noise_sim_mean_rho"] = float(sim.mean())
    agg["rank_noise_sim_sd"] = float(sim.std(ddof=1))
    agg["weighted_spearman"] = weighted_spearman(
        anchor, mean_acc, anchor_n, np.full(mean_acc.shape, 50.0)
    )
    s40["__aggregated_anchor_vs_crossmodel_mean__"] = agg

    # ---------------- SCB5 (anchor n known, MLLM n not persisted) ----------------
    scb_classes, scb_anchor, scb_models = load_scb5_model_accs()
    sa = json.load(
        open(BASE / "data/visualanchor_snapshot/01_core/anchor_score_scb5/anchor_scores.json")
    )
    # anchor n per (ds, class) in the same order as SCB5_CLASSES
    from sharedvisual.datasets import SCB5_CLASSES
    anchor_n = np.array(
        [sa[ds]["per_class_acc"][cls]["n"] for ds, cls in SCB5_CLASSES], float
    )

    scb5 = {}
    for m, accs in scb_models:
        # n unknown: sensitivity over the reported range n=50 (lowest LLaVA
        # effective) to n=5000 (largest core-model class).
        low = build_result(m, scb_anchor, accs, anchor_n, np.full(13, 50.0),
                           extra={"mllm_n_assumption": "all classes n=50"})
        high = build_result(m, scb_anchor, accs, anchor_n, np.full(13, 5000.0),
                            extra={"mllm_n_assumption": "all classes n=5000"})
        # weighted only uses anchor n and treats mllm n as uninformative
        rho_obs, p_obs = spearmanr(scb_anchor, accs)
        scb5[m] = {
            "model": m,
            "spearman_raw": float(rho_obs),
            "spearman_p": float(p_obs),
            "n_classes": 13,
            "anchor_rel": low["rel_x"],
            "sensitivity": {
                "mllm_n_50": {k: low[k] for k in ("pearson_attenuation_corrected",
                                                  "rank_noise_sim_mean_rho",
                                                  "rank_noise_sim_sd")},
                "mllm_n_5000": {k: high[k] for k in ("pearson_attenuation_corrected",
                                                     "rank_noise_sim_mean_rho",
                                                     "rank_noise_sim_sd")},
            },
        }

    out = {
        "method": (
            "Classical attenuation correction (Spearman 1904): "
            "rho_corr = rho_obs / sqrt(rel_X * rel_Y), rel = (var_obs - "
            "mean_c p_c(1-p_c)/n_c) / var_obs; rank-noise simulation under "
            "the binomial model (10,000 replicates); precision-weighted "
            "Spearman (w_c = 1/(1/nx_c + 1/ny_c))."
        ),
        "seed": int(SEED_BOOTSTRAP),
        "n_sim": N_SIM,
        "stanford40": s40,
        "scb5": scb5,
    }
    path = OUT / "reliability_correction.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=1)

    print("=" * 78)
    print("Stanford40 (n=40 classes; MLLM n=50/class, anchor n=182-296/class)")
    print("=" * 78)
    hdr = f"{'model':<44} {'raw rho':>8} {'att-corr':>8} {'w-rs':>7} {'sim mean':>8} {'sim sd':>7}"
    print(hdr)
    for k, v in s40.items():
        if k.startswith("__"):
            continue
        print(f"{v['model']:<44} {v['spearman_raw']:>8.3f} "
              f"{v['pearson_attenuation_corrected']:>8.3f} "
              f"{v['weighted_spearman']:>7.3f} "
              f"{v['rank_noise_sim_mean_rho']:>8.3f} {v['rank_noise_sim_sd']:>7.3f}")
    agg = s40["__aggregated_anchor_vs_crossmodel_mean__"]
    print(f"{'aggregated (mean acc)':<44} {agg['spearman_raw']:>8.3f} "
          f"{agg['pearson_attenuation_corrected']:>8.3f} "
          f"{agg['weighted_spearman']:>7.3f} "
          f"{agg['rank_noise_sim_mean_rho']:>8.3f} {agg['rank_noise_sim_sd']:>7.3f}")
    print(f"anchor rel = {agg['rel_x']:.3f} (var_obs={agg['var_obs']:.1f}, noise={agg['noise']:.1f})" if 'var_obs' in agg else "", )

    print()
    print("=" * 78)
    print("SCB5 (n=13; anchor n=42-1684/class; MLLM n not persisted -> sensitivity)")
    print("=" * 78)
    for m, v in scb5.items():
        lo = v["sensitivity"]["mllm_n_50"]
        hi = v["sensitivity"]["mllm_n_5000"]
        print(f"{v['model']:<16} raw {v['spearman_raw']:>6.3f} "
              f"att-corr[n50->n5k] {lo['pearson_attenuation_corrected']:>6.3f}->{hi['pearson_attenuation_corrected']:>6.3f} "
              f"sim-mean {lo['rank_noise_sim_mean_rho']:>6.3f}->{hi['rank_noise_sim_mean_rho']:>6.3f} "
              f"(anchor rel={v['anchor_rel']:.3f})")

    print()
    print("saved:", path)


if __name__ == "__main__":
    main()
