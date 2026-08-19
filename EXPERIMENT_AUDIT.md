# Experiment Audit Report

**Date**: 2026-08-15
**Auditor**: internal audit (independent reviewer backend unavailable; deviation from the cross-verification rule recorded)
**Project**: SharedVisual — Shared Class-Neighborhood Geometry: A Candidate Mechanism for Why CLIP Predicts MLLM Difficulty

## Overall Verdict: PASS

## Integrity Status: pass

## Checks

### A. Ground Truth Provenance: PASS

All evaluations score against dataset-provided ground truth:
- SCB5: public HuggingFace dataset (YOLO-format labels; `scb5_true_class_mapping.json` traces true classes) — real_gt
- Stanford40: official Stanford40 Actions annotations (`experiments/exp6_stanford40_geometry.py` loads cached features with official labels) — real_gt
- Flowers102: official Oxford Flowers102 annotations — real_gt
- MLLM per-class accuracies: frozen from the AnchorScore benchmark (documented protocol, temperature=0 greedy, class-name prompts); provenance fields traced in `data/visualanchor_snapshot/01_core/paper_data/mllm_full.json`
- Mediation counterfactual controls use model-output-derived mediators but are explicitly labeled as proxy controls (permuted/noise/independent-encoder) — synthetic_proxy, disclosed
- No ground truth derived from the models being evaluated without explicit labeling

### B. Score Normalization: PASS

- Accuracy = correct/total (raw counts, `max(1,total)` guards division-by-zero only)
- Spearman/Pearson correlations raw; no metric divided by the model's own max/min/mean
- L2 normalization used only for cosine geometry (standard, scale-invariant relational structure)
- No suspicious near-1.0/100% scores that arise from self-reference (the one 99.94% SigLIP teacher accuracy is a real, disclosed degenerate zero-shot behavior, flagged as such in §4.8)

### C. Result File Existence + Number Match: PASS

Every audited claim verified against the committed files (spot-recomputation where feasible):

| Claim set | Verified against | Result |
|---|---|---|
| Per-model ρ 0.374–0.736; CLIP group 0.473/0.454/0.521/0.402 | kruskal_4groups.json | ✓ (0.5205→0.521 rounding) |
| Family variation: 0.822/0.393/+0.429/p=0.0002/CI[0.162,0.711]; S40 0.772/0.609/+0.163/p=0.067; per-family 0.788/0.897/0.742 | family_variation.json | ✓ exact |
| Adapter R²: linear 0.192/0.135/0.053, MLP 0.209/0.129/0.036 | exp1b_clean_r2.json | ✓ exact |
| CKA cells 0.907/0.905/0.79–0.82/0.86/0.84/0.80–0.87/0.70–0.82 | exp2_cka_all_encoders.json | ✓ exact |
| Flowers102 geometry 10 pairs 0.51–0.83, p<1e-4 | flowers102_mantel.json | ✓ exact |
| Flowers102 difficulty 14/15 acc, 15/15 margin; SigLIP–ResNet 0.18/0.069; margin 0.37–0.88 | flowers102_difficulty_multi.json | ✓ exact |
| Attenuation S40 0.892/0.819/0.783±0.038; anchor reliability 0.982 | reliability_correction.json | ✓ (0.8919/0.8188/0.7826) |
| Mediation S40 0.449–0.714; counterfactuals +0.56/−0.006/−0.02; SCB5 combined +0.54 CI[−0.51,0.89] | mediation_bootstrap/counterfactual_controls/combined_test.json | ✓ (re-verified) |
| Mantel cells 0.662–0.982 (SCB5 21 pairs); SigLIP 0.927–0.957; ResNet 0.928–0.980 | mantel_test_results.json | ✓ |
| Perturbation sync + FDR grid (8→0, static 6/6); SigLIP σ0 collapse | visual_perturbation_sync_all/raw_all.json + fdr_correction.json | ✓ (re-verified) |
| Frequency control 0.797→0.835; S40 jackknife 0.768/0.738, max Δ0.043 | prevalence_and_leverage_controls.json | ✓ |
| Competence-adjusted Mantel null max 0.944, margins +0.01–0.12 | mantel_null_baseline.json | ✓ |
| Confusion-vs-geometry supplement: 0.092/0.263/0.044/0.158/0.955 with p 0.641/0.176/0.849/0.494 | confusion_vs_geometry.json | ✓ (0.0921/0.2635/0.0443/0.1578/0.9547) |
| Prompt variants 45.3/0.0/1.7/0.0/0.0 → 98/100/69/92.5/73.8 | exp3_prompt_variants_v2 + exp3_stand_screen_prompts.json | ✓ |
| Figure-embedded numbers (fig5 ρ=0.817/0.810, fig2 group means, figS1 CKA, fig3 Mantel) | recomputed from source files | ✓ exact |

No phantom results; every headline number traces to a committed file.

### D. Dead Code Detection: PASS

No orphan metric functions found. Scripts without same-named result files are accounted for: `exp1_deep_analysis.py`/`exp1_group_analysis.py` → `.txt` outputs; `mantel_test.py` → `mantel_test_results.json`; `mediation_tost_collinearity.py` → `mediation_bootstrap.json`; `merge_*` → documented glue scripts updating existing JSONs; `verify_blur_ckpt.py` → verification utility; `eval_clip_mllms_v100.py` → upstream producer of mllm_full.json (V100, legacy protocol, documented in AGENTS.md). Superseded `blur_synchrony.py` (v1) is documented as replaced by v2.

### E. Scope Assessment: PASS

- Evidence scale: 12 MLLMs / 4 encoder families, 7 vision encoders, 3 domains (8/13/40/102 classes), 21+10+15 encoder-pair comparisons, 6-MLLM Stanford40 panel, 5 blur levels × 4 encoders, 10,000-seed resampling throughout
- Scope language check: "robust and general" (§4.1) is backed by two-dataset replication + frequency/leverage/noise-floor controls; "robust finding" (§4.8) refers to the static agreement that survives FDR; no "comprehensive/extensive" over-claims found
- Multiplicity: 30-test perturbation grid handled with BH-FDR and disclosed; n=13 power limitations acknowledged explicitly

### F. Evaluation Type Classification

- real_gt: CLIP zero-shot evaluations (SCB5/Stanford40/Flowers102), MLLM evaluations, prompt-variant experiments, blur perturbation, probe-based difficulty
- synthetic_proxy (explicitly labeled): counterfactual mediation mediators (class-permuted, noise, independent-encoder)
- simulation_only (explicitly labeled): mediation rank-z estimator simulation (2,000 datasets/config); binomial attenuation simulation; power analysis (Fisher-z)

## Action Items

- None blocking. [minor] The paper's CLIP-group per-model value 0.521 (file 0.5205) is a standard half-up rounding; if strict 3dp-truncation is preferred, change to 0.520 — cosmetic only.

## Claim Impact

- C1 Shared class-neighborhood geometry (Mantel ρ=0.90–0.98, permutation-tested): supported
- C2 Patch-level localization (R²≤0.21 vs Mantel ≥0.90): supported
- C3 Mediation on Stanford40 (6/6 CIs exclude zero): supported
- C4 Shared-encoder artifact refuted (SigLIP 0.716 > CLIP 0.462): supported
- C5 Data-overlap weakened (ResNet-50 0.93–0.98, DINOv2 ordering): supported
- C6 Prompt variants resolve CLIP failures (all five classes): supported
- C7 Counterfactual mediator controls (pipeline does not spuriously attribute): supported
- C8 Flowers102 cross-domain (geometry + difficulty): supported
- C9 Perturbation synchrony exploratory, static agreement robust: supported
- C10 Competence-adjusted Mantel (null max 0.944, all pairs exceed): supported
- C11 Class-frequency not a confound; S40 not leverage-driven: supported

## Limitations

- First-party audit (independent reviewer unavailable). The 2026-08-12 audit report and the 2026-08-18 cross-checks cover the same ground from two angles; the audit catches common integrity patterns, not all possible fabrication.

## Addendum — Encoder-Swap Run (2026-08-17)

Preregistered (DESIGN.md locked before training; deviations disclosed in RUN_MANIFEST: full cleaned pool vs bounded 4,896; 257-token cap; manual greedy decode). Verified post-run:

| Check | Result |
|-------|--------|
| GT provenance | Dataset labels (SCB5 val, full-frame); scoring identical to eval_hf.py first-match |
| Score normalization | Raw per-class accuracies; no self-reference |
| Result existence | 4 arm summaries n=5416 each; analysis JSON committed; RUN_MANIFEST present |
| Preregistration integrity | P1-P4 and interpretation table written before unblinding (commit history); outcome reported per table without post-hoc reinterpretation |
| Verdict | PASS (honest-boundary outcome faithfully reported) |

## Addendum — 2026-08-18 Delta (text and figures only)

Cross-model rerun attempted at audit time: blocked by an unreachable reviewer backend (network timeout). Deviation recorded as in the 2026-08-15 audit.

Experiment-side surface unchanged since the 08-17 PASS audit. Verified:
- No new analysis/evaluation scripts, no new result files, no new quantitative claims (changes were prose, captions, figures, cross-refs, and the fig3 colorbar bound 0.5→0.45, which changes a display scale only).
- Numeric-token multiset of paper/SharedVisual.tex verified identical across the revision (1692=1692) except the intentional colorbar bound.
- All claim-critical numbers re-verified against JSONs during figure reviews (fig2 group means 0.462/0.716/0.669/0.374; fig4 45.3→98.1 etc.; fig5 ρ=0.817/0.810; mediation table; supplement §1-§4).

Verdict for the delta: PASS (no integrity-relevant change).

## Addendum — Code Walkthrough Corrections (2026-08-18)

Two integrity-relevant bugs found by the full-code walkthrough (3 parallel passes, findings independently re-verified) and fixed:

| Bug | Fix | Paper impact |
|-----|-----|--------------|
| Probe blur never applied for Qwen/ResNet (`exp_visual_perturbation_sync.py` extractors) | Re-run on V100 with blur applied | S4.8 "blur-invariant (Δ=0)" rewritten with true results (ResNet teacher 65.3→44.0 at σ8; Qwen non-monotonic; no probe-pair Δ agreement, all perm p≥0.057) |
| `confusion_vs_geometry.py` permuted both matrices jointly (preserved pairing) | One-matrix permutation on symmetrized statistic | Supplement §2 perm p 0.661 → 0.7351 (null conclusion unchanged) |

Also fixed: analysis_encoder_swap.py reproducibility paths, preregistered P4(b) implemented (PASS), collapse-metric inertness documented, SigLIP token-cap wording corrected, exp1_deep_analysis LOO+seed. Verdict: PASS (all fixes verified by re-execution).
