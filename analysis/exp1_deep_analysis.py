#!/usr/bin/env python3
"""
Exp 1 Deep Analysis: Beyond Surface-Level ρ
=============================================
Uncovers why SigLIP group shows higher ρ than CLIP group.
Controls for confounds, bootsraps CIs, identifies per-class drivers.

No GPU needed — uses existing JSON results only.
"""
import json
import numpy as np
from pathlib import Path
from scipy.stats import spearmanr, pearsonr
from sklearn.linear_model import LinearRegression

import sys
BASE = Path(__file__).resolve().parents[1]
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))
from sharedvisual.paths import va_results as _va
VA_RESULTS = _va()
OUTPUT = BASE / "results" 

CLASS_MAP = {
    "TeacherBehavior": [
        "guide", "answer", "On-stage interaction", "blackboard-writing",
        "teacher", "stand", "screen", "blackBoard",
    ],
    "HandriseReadWrite": ["hand-raising", "read", "write"],
    "BowTurnHead": ["BowHead", "TurnHead"],
}

ENCODER_GROUPS = {
    "CLIP ViT-L/14 (LLaVA)": ["LLaVA-1.5-7B"],
    "SigLIP (Gemma)":        ["Gemma4-31B", "Gemma4-26B", "Gemma-3-27B"],
    "QwenViT (Qwen3.x)":     ["Qwen3.5-27B", "Qwen3.6-27B", "Qwen3.5-35B", "Qwen3.6-35B"],
    "Qwen2-VL ViT":          ["Qwen2-VL-7B"],
}


def load_data():
    with open(VA_RESULTS / "01_core" / "anchor_score_scb5" / "anchor_scores.json") as f:
        anchor = json.load(f)
    with open(VA_RESULTS / "01_core" / "paper_data" / "mllm_full.json") as f:
        mllm = json.load(f)
    # backbone_results.json is not in the committed snapshot (multi_backbone
    # experiments live in the VisualAnchor repo). Fall back to a sibling
    # VisualAnchor checkout if the snapshot does not contain it.
    bb_path = VA_RESULTS / "02_robustness" / "multi_backbone" / "backbone_results.json"
    if not bb_path.exists():
        bb_path = BASE.parent / "VisualAnchor" / "results" / "02_robustness" / "multi_backbone" / "backbone_results.json"
    with open(bb_path) as f:
        backbones = json.load(f)
    return anchor, mllm, backbones


def get_per_class_dicts(anchor, mllm):
    """Return aligned dicts: {class_key: anchor_acc} and {model_name: {class_key: mllm_acc}}."""
    anchor_scores = {}
    for ds, classes in CLASS_MAP.items():
        for cls in classes:
            anchor_scores[(ds, cls)] = anchor[ds]["per_class_acc"][cls]["acc"]

    mllm_scores = {}
    for model in mllm.get("TeacherBehavior", {}):
        accs = {}
        for ds, classes in CLASS_MAP.items():
            if model in mllm.get(ds, {}):
                for cls in classes:
                    val = mllm[ds][model].get(cls)
                    if val is not None:
                        accs[(ds, cls)] = val
        if accs:
            mllm_scores[model] = accs
    return anchor_scores, mllm_scores


def bootstrap_rho(x, y, n_iter=10000, seed=20260801):
    """Bootstrap Spearman ρ and return mean, 2.5%, 97.5% CI."""
    rng = np.random.RandomState(seed)
    n = len(x)
    rhos = []
    for _ in range(n_iter):
        idx = rng.randint(0, n, size=n)
        if len(np.unique(x[idx])) > 1 and len(np.unique(y[idx])) > 1:
            r, _ = spearmanr(x[idx], y[idx])
            rhos.append(r)
    return float(np.mean(rhos)), float(np.percentile(rhos, 2.5)), float(np.percentile(rhos, 97.5))


def partial_corr_rho(x, y, cov):
    """Spearman partial correlation: ρ(x, y | cov)."""
    def resid(a, b):
        mask = ~np.isnan(a) & ~np.isnan(b)
        a, b = a[mask], b[mask]
        if len(a) < 3 or len(np.unique(b)) < 2:
            return a
        lr = LinearRegression()
        lr.fit(b.reshape(-1, 1), a)
        return a - lr.predict(b.reshape(-1, 1))

    x_res = resid(x, cov)
    y_res = resid(y, cov)
    if len(np.unique(x_res)) > 1 and len(np.unique(y_res)) > 1:
        r, p = spearmanr(x_res, y_res)
        return float(r), float(p)
    return float("nan"), 1.0


def main():
    anchor, mllm, backbones = load_data()
    anchor_scores, mllm_scores = get_per_class_dicts(anchor, mllm)

    print("=" * 80)
    print("EXP 1 DEEP ANALYSIS: Why does SigLIP Group ρ > CLIP Group ρ?")
    print("=" * 80)

    # ── 1. Bootstrap ρ per group ──
    print("\n" + "-" * 80)
    print("1. BOOTSTRAP ρ PER ENCODER GROUP")
    print("-" * 80)

    all_classes = sorted(anchor_scores.keys())
    anchor_vals = np.array([anchor_scores[c] for c in all_classes])

    results = {}
    for group_name, models in ENCODER_GROUPS.items():
        rho_single_list = []
        for model in models:
            if model not in mllm_scores:
                continue
            vals = np.array([mllm_scores[model].get(c, np.nan) for c in all_classes])
            mask = ~np.isnan(vals)
            if mask.sum() >= 3:
                r, p = spearmanr(anchor_vals[mask], vals[mask])
                rho_single_list.append(r)

        if rho_single_list:
            mean_rho = float(np.mean(rho_single_list))
            # Bootstrap using the first available model's data
            primary_model = next(m for m in models if m in mllm_scores)
            vals = np.array([mllm_scores[primary_model].get(c, np.nan) for c in all_classes])
            mask = ~np.isnan(vals)
            bs_mean, bs_lo, bs_hi = bootstrap_rho(anchor_vals[mask], vals[mask])
            results[group_name] = {"models": models, "mean_rho": mean_rho, "rho_list": rho_single_list,
                                   "bs_ci": (bs_lo, bs_hi), "n_models": len(rho_single_list)}
            print(f"  {group_name:30s}  mean ρ={mean_rho:.3f}  [{bs_lo:.3f}, {bs_hi:.3f}]  "
                  f"(n_models={len(rho_single_list)}, individual: {[f'{r:.3f}' for r in rho_single_list]})")

    # ── 2. Partial correlation controlling for MLLM overall accuracy ──
    print("\n" + "-" * 80)
    print("2. PARTIAL ρ (controlling for MLLM overall accuracy)")
    print("   Removes the confound: 'better models have higher ρ because they're better at everything'")
    print("-" * 80)

    # LOO covariate: mean accuracy across the OTHER models (paper convention)
    for model in ["LLaVA-1.5-7B", "Gemma4-31B", "Gemma-3-27B", "Qwen3.5-35B", "Qwen2-VL-7B"]:
        if model not in mllm_scores:
            continue
        others = [m for m in mllm_scores if m != model]
        all_model_vals = [np.array([mllm_scores[m].get(c, np.nan) for c in all_classes])
                          for m in others]
        mean_difficulty = np.nanmean(all_model_vals, axis=0)
        vals = np.array([mllm_scores[model].get(c, np.nan) for c in all_classes])
        mask = ~np.isnan(vals) & ~np.isnan(mean_difficulty)
        overall = np.nanmean(vals)
        r_raw, _ = spearmanr(anchor_vals[mask], vals[mask])
        r_partial, p_partial = partial_corr_rho(anchor_vals[mask], vals[mask], mean_difficulty[mask])
        print(f"  {model:25s} overall={overall:.1f}%  raw ρ={r_raw:.3f}  "
              f"partial ρ(cross-model avg)={r_partial:.3f} (p={p_partial:.4f})")

    # ── 3. Per-class residual analysis ──
    print("\n" + "-" * 80)
    print("3. PER-CLASS DRIVERS: Where do groups disagree?")
    print("-" * 80)

    # For each class, compute residual from the overall ρ line
    # Average MLLM accuracy per group
    group_accs = {}
    for gname, models in ENCODER_GROUPS.items():
        valid = [m for m in models if m in mllm_scores]
        if valid:
            accs = np.array([np.array([mllm_scores[m].get(c, np.nan) for c in all_classes]) for m in valid])
            group_accs[gname] = np.nanmean(accs, axis=0)

    # Compare per-class prediction errors: which classes does LLaVA get wrong that Gemma gets right?
    if "LLaVA-1.5-7B" in mllm_scores and "Gemma4-31B" in mllm_scores:
        llava_v = np.array([mllm_scores["LLaVA-1.5-7B"].get(c, np.nan) for c in all_classes])
        gemma_v = np.array([mllm_scores["Gemma4-31B"].get(c, np.nan) for c in all_classes])
        anchor_v = np.array([anchor_scores[c] for c in all_classes])
        mask = ~np.isnan(llava_v) & ~np.isnan(gemma_v)
        mask_idx = np.where(mask)[0]

        # Fit ρ lines
        lr_llava = LinearRegression()
        lr_llava.fit(anchor_v[mask].reshape(-1, 1), llava_v[mask])
        residual_llava = llava_v[mask] - lr_llava.predict(anchor_v[mask].reshape(-1, 1))

        lr_gemma = LinearRegression()
        lr_gemma.fit(anchor_v[mask].reshape(-1, 1), gemma_v[mask])
        residual_gemma = gemma_v[mask] - lr_gemma.predict(anchor_v[mask].reshape(-1, 1))
        residual_diff = residual_gemma - residual_llava  # positive = Gemma better than expected

        top_k = 5
        sorted_idx = np.argsort(-residual_diff)
        print(f"  Classes where Gemma4-31B outperforms LLaVA-1.5-7B (relative to AnchorScore prediction):")
        for ri in sorted_idx[:top_k]:
            ci = mask_idx[ri]
            c = all_classes[ci]
            print(f"    {c[0]:25s} {c[1]:20s}  Anchor={anchor_v[ci]:.1f}%  "
                  f"LLaVA={llava_v[ci]:.1f}%  Gemma={gemma_v[ci]:.1f}%  "
                  f"Gap(resid)={residual_diff[ri]:+.1f}pp")

        print(f"  Classes where LLaVA outperforms Gemma4-31B (relative to AnchorScore prediction):")
        sorted_idx = np.argsort(residual_diff)
        for ri in sorted_idx[:top_k]:
            ci = mask_idx[ri]
            c = all_classes[ci]
            print(f"    {c[0]:25s} {c[1]:20s}  Anchor={anchor_v[ci]:.1f}%  "
                  f"LLaVA={llava_v[ci]:.1f}%  Gemma={gemma_v[ci]:.1f}%  "
                  f"Gap(resid)={residual_diff[ri]:+.1f}pp")

    # ── 4. Cross-encoder similarity matrix ──
    print("\n" + "-" * 80)
    print("4. CROSS-MODEL SIMILARITY MATRIX (ρ between models' per-class accuracy)")
    print("-" * 80)

    all_models = []
    all_patterns = []
    for gname, models_list in ENCODER_GROUPS.items():
        for m in models_list:
            if m in mllm_scores:
                vals = np.array([mllm_scores[m].get(c, np.nan) for c in all_classes])
                mask = ~np.isnan(vals)
                all_models.append(f"{m} ({gname.split('(')[-1].rstrip(')')})")
                all_patterns.append(vals[mask] if mask.sum() == len(all_classes) else None)

    # Filter to complete models
    complete = [(n, v) for n, v in zip(all_models, all_patterns) if v is not None]
    n_models = len(complete)
    sim_matrix = np.zeros((n_models, n_models))
    for i in range(n_models):
        for j in range(n_models):
            if i == j:
                sim_matrix[i][j] = 1.0
            else:
                r, _ = spearmanr(complete[i][1], complete[j][1])
                sim_matrix[i][j] = float(r)

    header = "  " + " " * 28
    for n, _ in complete:
        header += f"{n[:12]:>12s}"
    print(header)
    for i in range(n_models):
        name = complete[i][0][:28]
        line = f"  {name:28s} "
        for j in range(n_models):
            line += f"{sim_matrix[i][j]:8.3f}  "
        print(line)

    # ── 5. CLIP backbone accuracy vs AnchorScore ρ ──
    print("\n" + "-" * 80)
    print("5. CLIP BACKBONE VARIANTS: zero-shot ρ with each other and MLLM")
    print("-" * 80)

    backbone_models = {"laion_l14": "ViT-L/14 LAION", "openai_l14": "OpenAI ViT-L/14 (AnchorScore)",
                       "openai_b32": "OpenAI ViT-B/32"}
    backbone_accs = {}
    for bk, bk_name in backbone_models.items():
        accs = []
        for c in all_classes:  # all_classes = sorted(anchor_scores.keys()) — same
            # ordering as mllm_v below; per-class arrays must be aligned.
            ds, cls = c
            acc = backbones[bk]["results"][ds]["per_class_acc"].get(cls, {}).get("acc")
            if acc is not None:
                accs.append(acc)
            else:
                accs.append(np.nan)
        backbone_accs[bk_name] = np.array(accs)

    # Correlate backbones with each other
    for b1 in backbone_models.values():
        for b2 in backbone_models.values():
            if b1 >= b2:
                continue
            r, _ = spearmanr(backbone_accs[b1], backbone_accs[b2])
            print(f"  {b1:35s} vs {b2:35s}  ρ = {r:.3f}")

    # Correlate backbone accuracy with each MLLM's accuracy
    print()
    for bk, bk_name in backbone_models.items():
        print(f"  {bk_name} zero-shot vs MLLM per-class (ρ):")
        for model_name in ["LLaVA-1.5-7B", "Gemma4-31B", "Gemma-3-27B", "Qwen3.5-35B", "Qwen2-VL-7B"]:
            if model_name not in mllm_scores:
                continue
            mllm_v = np.array([mllm_scores[model_name].get(c, np.nan) for c in all_classes])
            mask = ~np.isnan(mllm_v)
            if mask.sum() >= 3:
                r, p = spearmanr(backbone_accs[bk_name][mask], mllm_v[mask])
                print(f"    → {model_name:25s}  ρ={r:.3f}  (p={p:.4f})")

    # ── 6. Summary: what explains the difference? ──
    print("\n" + "-" * 80)
    print("6. SYNTHESIS & INTERPRETATION")
    print("-" * 80)
    print("""
  The Exp1C (natural groups) finding — SigLip ρ > CLIP ρ — is SURPRISING.
  It suggests the shared vision bottleneck hypothesis is INCOMPLETE.

  Possible explanations:
  A. LLM scale dominates: Larger LLMs (Gemma4-31B, 27B) learn better
     from any visual features than small LLMs (LLaVA-1.5-7B, Vicuna-7B).
     → Partial correlation controlling for overall accuracy tests this.

  B. AnchorScore captures general visual difficulty, not CLIP-specific:
     If ALL vision encoders find the same classes hard, any classifier
     (linear probe, MLLM) will show correlation.
     → Exp 2 (CKA/RSA) and Exp 4 (perturbation synchrony) test this.

  C. SigLIP and CLIP learn similar visual features despite different losses:
     Both are trained on Internet-scale image-text data.
     → Exp 2 (feature similarity) directly tests this.

  D. LLaVA-1.5-7B is a uniquely bad MLLM:
     Its low overall accuracy (TeacherBehavior: ~32.5%) dominates.
     → Check if fixing LLaVA-1.5's instruction prompt improves things.
""")
    print("=" * 80)


if __name__ == "__main__":
    main()
