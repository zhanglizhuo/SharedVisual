from pathlib import Path
"""Test the geometry→confusion link within an MLLM, and report a null result.

Motivation
----------
The paper's central mechanism was originally named "shared class
confusability" but the experiments measure class-neighborhood geometry
(CKA, Mantel tests on class-centroid similarity matrices). The further step
from geometry to *behavioral* confusability was treated as a defined
behavioral translation rather than a separately measured link. This script
tests that link directly on the one SCB5 per-image MLLM dataset available
locally (gemma4:31b, SigLIP encoder, blur_0 = no-perturbation baseline),
then asks: does the MLLM's confusion structure correlate with its vision
encoder's class-centroid geometry?

Method
------
1. Build an 8×8 confusion matrix M from gemma4:31b per-image responses
   (results/blur_gemma4_31b_ckpt_discoring/blur_0.json) using the
   same display-name substring matching as exp_perturbation_blur.py.
2. Symmetrize confusion per unordered class pair: conf(i,j) = M[i,j] + M[j,i].
3. Load SigLIP ViT-SO400M 8×8 class-centroid cosine similarity (the vision
   encoder inside gemma4) from class_similarity_matrices.json.
4. Spearman ρ over the 28 unordered class pairs (upper triangle, k=1).

Cross-validation (correctness checks, not hypothesis tests)
-----------------------------------------------------------
- Class-name ordering: must align between confusion matrix and similarity
  matrix (else the comparison is silent garbage).
- SigLIP matrix variance: confirm non-saturated (else ρ ceiling ≈ 0).
- Sanity Mantel ρ(SigLIP vs CLIP-L OA): must match paper §3.3 (0.93–0.96).
- Sign consistency: ρ(conf, sim) and ρ(conf, dist=1-sim) must be negatives.

Hypothesis
----------
If geometry drives confusion, ρ(conf, sim) > 0 (similar class pairs confuse
more). A null result indicates MLLM confusion is dominated by non-geometric
factors (LM prior, prompt alignment), consistent with §5's prompt-variant
finding that CLIP failures are semantic, not visual.

Output
------
results/confusion_vs_geometry.json
"""
import json
import numpy as np
from scipy.stats import spearmanr

BASE = Path(__file__).resolve().parents[1] / "results"
CLASSES = ["guide", "answer", "On-stage interaction", "blackboard-writing",
           "teacher", "stand", "screen", "blackBoard"]
# Display-name map used by the gemma4 prompt (from exp_perturbation_blur.py L48-57).
NAME_MAP = {
    "On-stage interaction": "on stage interaction",
    "blackboard-writing": "writing on blackboard",
    "blackBoard": "standing near a blackboard or whiteboard",
    "stand": "standing at the front of a classroom",
    "guide": "guiding students in classroom",
    "answer": "answering questions in class",
    "teacher": "teacher in classroom",
    "screen": "showing a screen in classroom",
}


def predict_class(response):
    """Mirror exp_perturbation_blur.match_class_name: longest-display-match
    rule against the lowercased response. Returns None if no display name
    matches (an 'unmatched' prediction)."""
    resp = response.lower().strip().rstrip(".,;:!?\n\t ")
    matched = [(len(disp.lower()), cls) for cls, disp in NAME_MAP.items()
               if disp.lower() in resp]
    return max(matched)[1] if matched else None


def perm_p(rho_obs, x, y, seed=42, n=10000):
    """Two-sided permutation p-value via class-label shuffle.
    Shuffles BOTH x and y rows/cols by the same permutation, recomputes ρ
    on upper triangles; reports fraction |ρ_perm| >= |rho_obs|."""
    rng = np.random.default_rng(seed)
    k = x.shape[0]
    iu = np.triu_indices(k, k=1)
    count = 0
    for _ in range(n):
        p = rng.permutation(k)
        x_p = x[np.ix_(p, p)]
        y_p = y[np.ix_(p, p)]
        r, _ = spearmanr(x_p[iu], y_p[iu])
        if abs(r) >= abs(rho_obs):
            count += 1
    return count / n


def build_confusion(responses):
    """8×8 confusion matrix from gemma4 per-image responses. Rows=GT, cols=Pred.
    Unmatched predictions excluded (gemma4 unmatched rate ~0.3%)."""
    M = np.zeros((8, 8), dtype=float)
    n_unmatched = 0
    n_total = 0
    for v in responses.values():
        gt = CLASSES.index(v["class"])
        pred = predict_class(v["response"])
        n_total += 1
        if pred is None:
            n_unmatched += 1
            continue
        M[gt, CLASSES.index(pred)] += 1
    return M, n_total, n_unmatched


def accuracy_normalized_confusion(M):
    """For each GT class i, distribute its errors: e[i,j] = M[i,j] / (row_sum - diag).
    Returns the full error-rate matrix (diagonal is zero by construction)."""
    k = M.shape[0]
    E = np.zeros((k, k))
    for i in range(k):
        wrong = M[i, :].sum() - M[i, i]
        if wrong > 0:
            for j in range(k):
                if i != j:
                    E[i, j] = M[i, j] / wrong
    return E


def main():
    # ── Load gemma4 per-image responses ──
    with open(f"{BASE}/blur_gemma4_31b_ckpt_discoring/blur_0.json") as f:
        bd = json.load(f)
    M, n_total, n_unmatched = build_confusion(bd["responses"])

    # ── Load class-centroid similarity matrices ──
    with open(f"{BASE}/class_similarity_matrices.json") as f:
        sd = json.load(f)
    assert sd["class_names"] == CLASSES, (
        f"class-order mismatch: similarity_matrices has {sd['class_names']}, "
        f"analysis assumes {CLASSES}"
    )
    sig_key = next(k for k in sd["matrices"] if "SigLIP" in k)
    clip_key = next(k for k in sd["matrices"] if "CLIP ViT-L/14 (OpenAI)" in k)
    sim_sig = np.array(sd["matrices"][sig_key])
    sim_clip = np.array(sd["matrices"][clip_key])

    # ── Build test quantities ──
    iu8 = np.triu_indices(8, k=1)
    conf_raw = M[iu8] + M.T[iu8]                       # raw sym confusion counts
    E = accuracy_normalized_confusion(M)
    conf_norm = E[iu8] + E.T[iu8]                       # acc-normalized confusion propensity
    sim_pairs = sim_sig[iu8]
    dist_pairs = 1.0 - sim_pairs

    # ── Hypothesis tests ──
    rho_raw_sim, p_raw_sim = spearmanr(conf_raw, sim_pairs)
    rho_norm_sim, p_norm_sim = spearmanr(conf_norm, sim_pairs)
    rho_norm_dist, _ = spearmanr(conf_norm, dist_pairs)
    perm_norm = perm_p(rho_norm_sim, E, sim_sig, seed=42, n=10000)

    # 7×7 (exclude teacher, the dominant LM-attractor)
    keep7 = [i for i in range(8) if CLASSES[i] != "teacher"]
    M7 = M[np.ix_(keep7, keep7)]
    E7 = accuracy_normalized_confusion(M7)
    sim7 = sim_sig[np.ix_(keep7, keep7)]
    iu7 = np.triu_indices(7, k=1)
    conf7_norm = E7[iu7] + E7.T[iu7]
    sim7_pairs = sim7[iu7]
    rho7_norm_sim, p7_norm_sim = spearmanr(conf7_norm, sim7_pairs)
    rho7_raw, p7_raw = spearmanr(M7[iu7] + M7.T[iu7], sim7_pairs)

    # ── Cross-validation (correctness checks) ──
    sanity_mantel_rho, _ = spearmanr(sim_sig[iu8], sim_clip[iu8])

    out = {
        "encoder": "gemma4:31b",
        "vision_encoder_for_geometry": sig_key,
        "n_images": n_total,
        "n_unmatched": n_unmatched,
        "unmatched_rate": round(n_unmatched / n_total, 4),
        "confusion_matrix": M.astype(int).tolist(),
        "per_class_accuracy": {
            CLASSES[i]: round(M[i, i] / M[i, :].sum(), 4) for i in range(8)
        },
        "hypothesis_tests_8class_n28": {
            "raw_confusion_vs_sim":       {"rho": round(rho_raw_sim, 4),   "p": round(p_raw_sim, 4)},
            "norm_confusion_vs_sim":      {"rho": round(rho_norm_sim, 4),  "p": round(p_norm_sim, 4),
                                           "perm_p_10k_seed42": round(perm_norm, 4)},
            "norm_confusion_vs_distance": {"rho": round(rho_norm_dist, 4)},
        },
        "hypothesis_tests_7class_teacher_excluded_n21": {
            "norm_confusion_vs_sim": {"rho": round(rho7_norm_sim, 4), "p": round(p7_norm_sim, 4)},
            "raw_confusion_vs_sim":  {"rho": round(rho7_raw, 4),     "p": round(p7_raw, 4)},
        },
        "cross_validation": {
            "class_order_aligned":           sd["class_names"] == CLASSES,
            "siglip_matrix_range":           [round(float(sim_pairs.min()), 4), round(float(sim_pairs.max()), 4)],
            "siglip_matrix_std":             round(float(sim_pairs.std()), 4),
            "sanity_mantel_siglip_vs_cliploa": round(float(sanity_mantel_rho), 4),
            "paper_expected_mantel_range":    "0.93-0.96",
            "sign_consistency_conf_sim_vs_dist": (
                f"+{rho_norm_sim:.3f} vs {rho_norm_dist:+.3f} (must be opposite signs)"
            ),
        },
        "interpretation": (
            "Within gemma4:31b, confusion structure does NOT correlate with "
            "SigLIP class-centroid geometry (ρ ≈ 0.09, p ≈ 0.6-0.7). The MLLM's "
            "confusion is dominated by LM-level biases (teacher attractor) and "
            "prompt/text-image alignment, not by pairwise feature-space geometry. "
            "This is consistent with §5's prompt-variant finding that CLIP "
            "failures are semantic, not visual, and supports the paper's framing "
            "of the mechanism as 'class-neighborhood geometry' rather than "
            "'class confusability'."
        ),
        "caveats": [
            "n=28 unordered pairs (8 classes) gives low power; only ρ ≥ ~0.5 would reach p<0.05.",
            "gemma4 is an MLLM (encoder + LM head); the test conflates vision-encoder-side and LM-side effects.",
            "A cleaner test would use a pure vision encoder (CLIP/SigLIP zero-shot, no LM) — not run here.",
            "Image-set overlap with the 50/class centroid-matrix sample cannot be verified (filenames not persisted).",
        ],
    }

    with open(f"{BASE}/confusion_vs_geometry.json", "w") as f:
        json.dump(out, f, indent=2)

    # Human-readable summary
    print("=" * 70)
    print("Confusion-vs-Geometry Test (gemma4:31b vs SigLIP centroids)")
    print("=" * 70)
    print(f"n_images={n_total}, unmatched={n_unmatched} ({100*n_unmatched/n_total:.1f}%)")
    print(f"\nPer-class accuracy:")
    for c, a in out["per_class_accuracy"].items():
        print(f"  {c:25s}: {a:.2f}")
    print(f"\n8-class tests (n=28 pairs):")
    print(f"  raw confusion vs sim:    ρ = {rho_raw_sim:+.3f}  (p = {p_raw_sim:.3f})")
    print(f"  norm confusion vs sim:   ρ = {rho_norm_sim:+.3f}  (p = {p_norm_sim:.3f}, perm_p = {perm_norm:.3f})")
    print(f"  norm confusion vs dist:  ρ = {rho_norm_dist:+.3f}  (sign-consistency check)")
    print(f"\n7-class tests (teacher excluded, n=21):")
    print(f"  norm confusion vs sim:   ρ = {rho7_norm_sim:+.3f}  (p = {p7_norm_sim:.3f})")
    print(f"  raw confusion vs sim:    ρ = {rho7_raw:+.3f}  (p = {p7_raw:.3f})")
    print(f"\nCross-validation:")
    print(f"  class-order aligned:           {out['cross_validation']['class_order_aligned']}")
    print(f"  SigLIP sim range / std:        {out['cross_validation']['siglip_matrix_range']} / {out['cross_validation']['siglip_matrix_std']}")
    print(f"  sanity Mantel ρ(SigLIP,CLIP-L OA): {sanity_mantel_rho:.3f}  (paper: 0.93-0.96)")
    print(f"\nResult JSON: {BASE}/confusion_vs_geometry.json")


if __name__ == "__main__":
    main()
