#!/usr/bin/env python3
"""Counterfactual mediation controls.

The main mediation analysis (analysis/mediation_tost_collinearity.py,
analysis/mediation_combined_test.py) uses the leave-one-out cross-model mean
accuracy as the mediator M. A reviewer raised the concern that this mediator
is "internal to the MLLM system": X (AnchorScore) and M (other MLLMs' mean
accuracy) are measured on the SAME images, so their overlap could be an
artifact of image-level difficulty, and M could carry no geometry-specific
information beyond "easy classes are easy for everyone." This script runs
four counterfactual controls on the same bootstrap mediation pipeline:

  0. Baseline replication  M = LOO cross-model mean accuracy
     (must reproduce the existing combined-test result; sanity check).

  1. Class-permuted M      M = LOO mean with class labels permuted (seed 42).
     If the indirect effect were an artifact of the mediator variable choice,
     a class-scrambled mediator (which preserves marginal rank structure but
     destroys any true per-class signal) would still mediate. Expected: the
     indirect effect collapses toward 0 / is not significant.

  2. Random-noise M        M = pure noise (noise seeded per replicate).
     A pure-noise mediator with the same correlation scale cannot mediate the
     X-Y association; this checks the pipeline does not spuriously attribute.

  3. Independent-encoder M M = OpenAI ViT-L/14 per-class accuracy (SCB5 only:
     the one dataset where an out-of-system, non-MLLM, differently-trained
     encoder's per-class difficulty is available as a mediator). M now comes
     from OUTSIDE the 9-model MLLM system. If shared difficulty is a real
     cross-system structure (not an MLLM-internal loop), the indirect effect
     through this independent encoder should remain positive and mediation
     should still be observable. Stanford40 has no multi-backbone anchors, so
     this control runs on SCB5 only (low power is acknowledged).

Outputs: results/mediation_counterfactual_controls.json
"""
import json
import sys
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parents[1]
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

from sharedvisual.paths import RESULTS_DIR, SEED_BOOTSTRAP
from sharedvisual.datasets import load_scb5_model_accs, load_stanford40, va_results
from sharedvisual.stats import mediation_paths

OUT = RESULTS_DIR
RNG_SEED = SEED_BOOTSTRAP
N_BOOT = 10000


def loo_mediator(model_accs, target_idx):
    """Leave-one-out cross-model mean accuracy per class (existing definition)."""
    others = [v for i, (_, v) in enumerate(model_accs) if i != target_idx]
    return np.mean(others, axis=0)


def indirect_effect(x, m, y):
    """Product-of-coefficients indirect effect a*b on z-scored ranks (NaN if collinear)."""
    return mediation_paths(x, m, y)["indirect"]


def pooled_mean_indirect(x, model_accs, m_fn, rng, n_boot=N_BOOT):
    """Bootstrap the cross-model mean indirect effect (cluster = class).

    Matches analysis/mediation_combined_test.py: resample the n classes with
    replacement (preserving the cross-model dependency on shared classes),
    compute each model's indirect effect, average; report the mean, its 95%
    percentile CI, and the one-sided p (fraction of bootstrap means <= 0).
    Also returns the observed (full-data) cross-model mean indirect effect,
    which is the point-estimate convention of mediation_combined_test.py.
    """
    n = len(x)
    M = len(model_accs)
    observed = [indirect_effect(x, m_fn(j), acc)
                for j, (_name, acc) in enumerate(model_accs)]
    observed = [v for v in observed if np.isfinite(v)]
    boot_mean = []
    for _ in range(n_boot):
        idx = rng.choice(n, n, replace=True)
        if len(np.unique(x[idx])) < 2:
            continue
        ies = []
        for j, (_name, yacc) in enumerate(model_accs):
            m = m_fn(j)[idx]
            ie = indirect_effect(x[idx], m, yacc[idx])
            if np.isfinite(ie):
                ies.append(ie)
        if len(ies) == M:
            boot_mean.append(np.mean(ies))
    boot_mean = np.array(boot_mean)
    if len(boot_mean) == 0:
        return {"point": float("nan"), "ci": [float("nan"), float("nan")],
                "p_one_sided": float("nan"), "n_valid": 0}
    lo, hi = np.percentile(boot_mean, [2.5, 97.5])
    return {"point": float(np.mean(boot_mean)), "ci": [float(lo), float(hi)],
            "p_one_sided": float(np.mean(boot_mean <= 0)),
            "observed_mean_full_data": float(np.mean(observed)),
            "n_valid": int(len(boot_mean))}


def run_control(name, x, model_accs, m_fn, rng, extra=None):
    res = pooled_mean_indirect(x, model_accs, m_fn, rng)
    out = {
        "control": name,
        "point_estimate_mean_indirect": res["point"],
        "observed_mean_full_data": res.get("observed_mean_full_data"),
        "bootstrap_ci_95": res["ci"],
        "p_one_sided": res["p_one_sided"],
        "n_valid_bootstraps": res["n_valid"],
        "excludes_zero": res["ci"][0] > 0,
    }
    if extra:
        out.update(extra)
    return out


def load_scb5_anchor_variants():
    """Per-class accuracy of independent CLIP variants on SCB5.

    Values come from the committed multi-backbone snapshot (02_robustness /
    multi_backbone / backbone_results.json), i.e., the same evaluation
    pipeline as AnchorScore but with the OpenAI-trained ViT-L/14 and ViT-B/32
    checkpoints, which are NOT the AnchorScore checkpoint used as X.
    Returns dict mapping scaffold label -> np.ndarray of per-class accuracy
    over the pooled 13 SCB5 classes.
    """
    va = va_results()
    b = json.load(open(va / "02_robustness" / "multi_backbone" / "backbone_results.json"))
    # class order must match SCB5_CLASSES used by load_scb5_model_accs
    from sharedvisual.datasets import SCB5_CLASSES
    scaffolds = {
        "openai_l14": ("openai_l14", "ViT-L/14 OpenAI"),
        "openai_b32": ("openai_b32", "ViT-B/32 OpenAI"),
    }
    out = {}
    for key, (name, label) in scaffolds.items():
        vals = []
        ok = True
        for ds, cls in SCB5_CLASSES:
            v = b[name]["results"].get(ds, {}).get("per_class_acc", {}).get(cls)
            if v is None:
                ok = False
                break
            # entry is {"n": ..., "acc": ...}
            vals.append(v["acc"] if isinstance(v, dict) else v)
        if ok:
            out[key] = np.array(vals, dtype=float)
    return out


def main():
    result = {"dataset_scb5": None, "dataset_stanford40": None, "config": {
        "rng_seed": RNG_SEED, "n_boot": N_BOOT,
        "permutation_seed": 42,
        "method": (
            "Cluster bootstrap: resample classes with replacement, seed "
            f"{RNG_SEED}; indirect effect via product-of-coefficients on "
            "rank-z scores (mediation_paths). Control 3 (independent encoder) "
            "uses OpenAI ViT-L/14 / ViT-B/32 per-class accuracies from the "
            "committed multi-backbone snapshot as the mediator."
        )}}

    # ---- SCB5 (n=13): baseline + permutation + noise + independent encoder ----
    classes, anchor, model_accs = load_scb5_model_accs()
    n, M = len(classes), len(model_accs)
    rng_base = np.random.RandomState(RNG_SEED)
    rng_perm = np.random.RandomState(RNG_SEED)
    rng_noise = np.random.RandomState(RNG_SEED)

    def base_loo(j):
        return loo_mediator(model_accs, j)

    controls_scb5 = [
        run_control("baseline_loo_cross_model", anchor, model_accs, base_loo, rng_base),
    ]

    # permutation control: a single perm applied consistently (mediator class-scrambled)
    rng_perm_ctl = np.random.RandomState(42)
    perm_vec = rng_perm_ctl.permutation(n)

    def perm_loo(j):
        return loo_mediator(model_accs, j)[perm_vec]

    controls_scb5.append(
        run_control("permuted_class_mediator", anchor, model_accs, perm_loo, rng_perm)
    )

    # noise control: fixed seed-derived noise vector per target model
    def noise_loo(j):
        sub = np.random.RandomState(1000 + j)
        v = sub.rand(n)
        # keep the same marginal correlation magnitude as real mediators is
        # not required; we only verify noise does not mediate.
        return v

    controls_scb5.append(
        run_control("random_noise_mediator", anchor, model_accs, noise_loo, rng_noise)
    )

    # independent encoder (OpenAI checkpoints, outside the MLLM system)
    variants = load_scb5_anchor_variants()
    for key in ("openai_l14", "openai_b32"):
        def enc_loo(j, mvec=variants[key]):
            return mvec
        rng_enc = np.random.RandomState(RNG_SEED)
        controls_scb5.append(
            run_control(f"independent_encoder_{key}", anchor, model_accs, enc_loo, rng_enc,
                        extra={"mediator": "OpenAI {} per-class accuracy (outside MLLM set)".format(
                            "ViT-L/14" if key == "openai_l14" else "ViT-B/32")})
        )

    result["dataset_scb5"] = {
        "n_classes": n, "n_models": M, "controls": controls_scb5,
    }

    # ---- Stanford40 (n=40): baseline + permutation + noise ----
    classes, anchor, model_accs = load_stanford40()
    model_list = [(name, np.asarray(acc)) for name, acc in model_accs.items()]
    n, M = len(classes), len(model_list)
    rng_base = np.random.RandomState(RNG_SEED)
    rng_perm = np.random.RandomState(RNG_SEED)
    rng_noise = np.random.RandomState(RNG_SEED)

    def base_loo(j):
        return loo_mediator(model_list, j)

    rng_perm_ctl = np.random.RandomState(42)
    perm_vec = rng_perm_ctl.permutation(n)

    def perm_loo(j):
        return loo_mediator(model_list, j)[perm_vec]

    def noise_loo(j):
        return np.random.RandomState(2000 + j).rand(n)

    controls_s40 = [
        run_control("baseline_loo_cross_model", anchor, model_list, base_loo, rng_base),
        run_control("permuted_class_mediator", anchor, model_list, perm_loo, rng_perm),
        run_control("random_noise_mediator", anchor, model_list, noise_loo, rng_noise),
    ]
    result["dataset_stanford40"] = {
        "n_classes": n, "n_models": M, "controls": controls_s40,
    }

    out_path = OUT / "mediation_counterfactual_controls.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    print("\n=== Counterfactual Mediation Controls ===")
    for ds in ("dataset_scb5", "dataset_stanford40"):
        d = result[ds]
        print(f"\n[{ds}  n={d['n_classes']}, M={d['n_models']}]")
        for c in d["controls"]:
            excl = "EXCLUDES 0" if c["excludes_zero"] else "straddles 0"
            print(f"  {c['control']:38s} mean_indirect={c['point_estimate_mean_indirect']:+.3f} "
                  f"CI[{c['bootstrap_ci_95'][0]:+.3f},{c['bootstrap_ci_95'][1]:+.3f}] "
                  f"p={c['p_one_sided']:.4f}  [{excl}]")
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
