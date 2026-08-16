# Experiment Audit Report

**Date**: 2026-08-15
**Auditor**: local executor (cross-model reviewer backend unavailable — codex disabled by user, no alternative API keys; deviation from the cross-model rule recorded)
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

Every audited claim verified against committed files in this session (spot-recomputation where feasible):

| Claim set | Verified against | Result |
|---|---|---|
| Per-model ρ 0.374–0.736; CLIP group 0.473/0.454/0.521/0.402 | kruskal_4groups.json | ✓ (0.5205→0.521 rounding) |
| Family variation: 0.822/0.393/+0.429/p=0.0002/CI[0.162,0.711]; S40 0.772/0.609/+0.163/p=0.067; per-family 0.788/0.897/0.742 | family_variation.json | ✓ exact |
| Adapter R²: linear 0.192/0.135/0.053, MLP 0.209/0.129/0.036 | exp1b_clean_r2.json | ✓ exact |
| CKA cells 0.907/0.905/0.79–0.82/0.86/0.84/0.80–0.87/0.70–0.82 | exp2_cka_all_encoders.json | ✓ exact |
| Flowers102 geometry 10 pairs 0.51–0.83, p<1e-4 | flowers102_mantel.json | ✓ exact |
| Flowers102 difficulty 14/15 acc, 15/15 margin; SigLIP–ResNet 0.18/0.069; margin 0.37–0.88 | flowers102_difficulty_multi.json | ✓ exact |
| Attenuation S40 0.892/0.819/0.783±0.038; anchor reliability 0.982 | reliability_correction.json | ✓ (0.8919/0.8188/0.7826) |
| Mediation S40 0.449–0.714; counterfactuals +0.56/−0.006/−0.02; SCB5 combined +0.54 CI[−0.51,0.89] | mediation_bootstrap/counterfactual_controls/combined_test.json | ✓ (session-verified) |
| Mantel cells 0.662–0.982 (SCB5 21 pairs); SigLIP 0.927–0.957; ResNet 0.928–0.980 | mantel_test_results.json | ✓ |
| Perturbation sync + FDR grid (8→0, static 6/6); SigLIP σ0 collapse | visual_perturbation_sync_all/raw_all.json + fdr_correction.json | ✓ (session-verified) |
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

- Same-family audit (executor; cross-model reviewer unavailable by user instruction). The 2026-08-12 audit report and this session's cross-checks cover the same ground from two angles; the audit catches common integrity patterns, not all possible fabrication.
