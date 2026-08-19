# Code Walkthrough Report — SharedVisual (2026-08-18)

Three parallel walkthrough passes (analysis/, experiments/, encoder-swap + figures), each finding independently re-verified of every CRITICAL/MAJOR claim against source and committed JSONs. ~12k lines / 70 files.

## Verified CRITICAL (paper-facing)

### C1. §4.8 "QwenViT and ResNet-50 features are essentially blur-invariant (Δ = 0 exactly)" is a pipeline bug, not an observation
- `experiments/exp_visual_perturbation_sync.py`: `extract_qwen` (L272) and `extract_resnet` (L294) open raw images with no `blur_bytes` call; only the CLIP probe branch (L344) applies blur.
- Committed evidence: `results/visual_perturbation_sync_raw.json` — Qwen teacher=29.33 and ResNet teacher=65.33 **byte-identical at σ0/σ1/σ2** (verified).
- Consequence: the Δ=0 reported in paper §4.8 is an artifact; the blur-invariance claim is unsupported. The CLIP-probe branch (blur applied) is valid.
- Remediation: re-run Qwen/ResNet probes WITH blur (ResNet: CPU-feasible minutes; Qwen2-VL-7B: needs V100 ~1-2 h), or amend the §4.8 sentence to drop the Qwen/ResNet invariance claim. **Pending decision (GPU time + text change).**

### C2. Supplement §2 permutation p=0.661 is not a valid label-permutation test
- `analysis/confusion_vs_geometry.py` `perm_p` (L75-90) shuffles **both** matrices with the **same** permutation — pairing between (confusion, similarity) per class pair is preserved, so the null varies only through directed-orientation flips; additionally the observed statistic uses symmetrized confusion (L146 `E[iu]+E.T[iu]`) while the permutation statistic uses directed entries.
- The asymptotic p=0.641 remains valid and carries the null conclusion; the published "permutation p=0.661 over 10,000 class-label shuffles" number/method is wrong.
- Remediation: fix `perm_p` to permute one matrix on the symmetrized statistic, re-run (seconds, CPU), update supplement §2 with the corrected number. **CPU, doable now.**

## Verified MAJOR

### M1. `experiments/encoder_swap/analysis_encoder_swap.py` (cited in paper Reproducibility) cannot reproduce committed artifacts
- Reads `results/encoder_swap/responses/{arm}_summary.json` (dir absent) and writes `encoder_swap_analysis.json`; committed artifacts are `results/encoder_swap/{arm}_summary.json` and `analysis_output.json` (verified via ls).
- Fix: correct the two paths, re-run, verify output == committed analysis_output.json. **CPU, doable now.**

### M2. Preregistered P4(b) was never computed
- DESIGN.md §3: random arm must "(b) NOT show geometry-ordered agreement (its pair agreements should be the lowest or noise-level)". `analysis_encoder_swap.py` computes only `random_markedly_lower` = mean-acc 0.5× threshold (L157-161), absent from DESIGN.
- Data check: R's pair agreements (C-R 0.508, S-R 0.666, D-R 0.491) ARE the three lowest vs C-S 0.740 / S-D 0.839 / C-D 0.791 — so P4(b) actually PASSES; it is just unreported.
- Fix: implement P4(b) from committed summaries, re-run, update RUN_MANIFEST + paper §4.9 sentence. **CPU, doable now.**

### M3. Collapse floor metric is inert (exact string equality)
- `03_eval_swap_variants.py` L132-138: `collapse_top_class_frac` counts responses byte-equal to a class name → 0.0 for every arm (verified pattern in summaries). The preregistered floor (>0.80) can never fire.
- Corroboration: R mean acc 8.44% ≈ 7.7% chance level is consistent with no single-class collapse, and no arm shows per-class dominance > 40%.
- Fix: response files are V100-only (not committed) → cannot recompute now. Amend RUN_MANIFEST with the metric limitation + corroboration note. **Doc-only now; optional V100 recompute later.**

### M4. Encoder-swap feature-shape documentation wrong (2 places)
- SigLIP 257-cap is `feats[:, :257, :]` = **prefix/top-left crop** (02:138-142), RUN_MANIFEST says "center-cropped to first 257" — wording false.
- DESIGN §6 lists R as "257×768"; ViT-B-16@224 yields 197 tokens.
- Fix: doc-only (RUN_MANIFEST + DESIGN.md).

### M5. `exp1_deep_analysis.py` exploratory hygiene (stdout-only, not paper-facing)
- Partial-ρ covariate includes the target model (L156-159) vs LOO convention elsewhere; unseeded `np.random.randint` (L83) breaks determinism.
- Fix: cheap, doable now.

### M6. Label-source split between pipelines (documented, under-attributed)
- Blur pipelines (exp4 recreate, exp_perturbation_blur, exp_visual_perturbation_sync) read raw YOLO labels; geometry/exp3 read `scb5_true_class_mapping.json`. Measured σ0 divergence (bbw 46.0 vs 45.3) is attributed in AGENTS.md to "different blur impl" but the actual cause is label source.
- Within-pipeline consistency holds (paper numbers are per-pipeline self-consistent). Fix: doc-only (AGENTS/RUN_MANIFEST attribution correction).

### M7. `eval_hf.py` protocol hazards (inherited from VisualAnchor legacy)
- 336×336 pre-squash for all models; first-found substring scoring (dead `match_class_name` L58-64); inference exceptions silently counted as wrong (L211-213). Numbers are "comparable" within the legacy family but carry unquantified class-confusion artifacts (e.g., "writing on the blackboard" → blackBoard).
- The paper already discloses strong class priors for these controls; recommend a Limitations footnote at most. **Pending decision.**

## Verified MINOR (selected)
- fdr_correction.py hard-coded p copies (L62-92, currently verified consistent; fragile).
- fig2 hard-coded "sep. p=0.029" vs JSON two-sided 0.0571 (one-sided value is paper-defensible; store in JSON).
- figure_audit.py audits code not rendered PDFs; paper/audit_figures.py crashes (literal-string path bug, L276/301).
- verify_blur_ckpt.py stale (points at deleted ckpt dir); blur_synchrony.py reads legacy json (documented).
- Seeds 20260731 (compute_partial_ci, stanford40_partial_correlation) vs canonical 20260801.
- README stale output names (exp1_deep_analysis, mediation_combined_test).
- exp2_mllm_cka within-sim includes diagonal (self-sim=1).

## Verified PASS (adversarially recomputed)
Mediation rank-z formulas; Mantel 8! enumeration (p=4.96e-05 identity-only); Kruskal-Wallis p=0.0303; Mantel null baseline 0.944; permutation critical ρ; power 16%/33%/80%; family variation (0.822/0.393/+0.429/p=0.0002); prevalence/leverage (LAION anchor correct); blur_synchrony_v2 (all four ρ/p pairs); flowers102 difficulty multi (all paper numbers); exp3 prompt numbers; encoder-swap P1/P3 arithmetic + bootstrap (C-S 0.740, S-D 0.839, C-D 0.791, diff CI [-0.17, 0.693]); generate_figures.py all six figures' numbers vs JSONs.
