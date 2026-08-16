# Paper Claim Audit Report

**Date**: 2026-08-15
**Auditor**: local executor (cross-model zero-context reviewer unavailable — deviation recorded)
**Paper**: Shared Class-Neighborhood Geometry: A Candidate Mechanism for Why CLIP Predicts MLLM Difficulty

## Overall Verdict: PASS (after 1 fix applied)

**Verdict before fix**: WARN (1 minor range mismatch). **After fix**: PASS.

## Claims Verified: 41
- exact_match: 33
- rounding_ok: 7
- mismatch (fixed): 1

## Issues Found

### [FIXED] Claim: decoder-control class-prior range "85–97%"
- **Location**: §4.2, final sentence of decoder-control paragraph
- **Paper said**: "e.g., 85--97% of responses map to one class name"
- **Evidence shows**: per-(model, sub-dataset) single-class-attractor fractions of 78–100% (BLIP-2: 93/100/100; InstructBLIP: 78/78/100; IDEFICS: 88/100/96; computed with the eval's own first-match substring logic against the committed `*_responses.json` files)
- **Status**: mismatch (minor; InstructBLIP 78% below claimed lower bound 85%)
- **Fix applied**: paper §4.2 and analysis/exp1c_clip_group_expanded.py comment both updated to "78–100%"

## All Claims (key rows)

| # | Location | Paper Value | Evidence | Status |
|---|----------|-------------|----------|--------|
| 1 | Intro | "one to nearly two orders of magnitude larger" (0.4B vs 7–35B) | 17.5×–82× | exact_match |
| 2 | §4.4 | "≈2× spread" 0.73/0.39 | 1.87× | rounding_ok |
| 3 | §4.3 | "110× more data (142M vs 1.28M)" | 110.9× | exact_match |
| 4 | §5.2 | "300× less data" (400M/1.28M) | 312× | rounding_ok |
| 5 | §4.7 | "2.5× fewer" (50/20 images) | 2.5 | exact_match |
| 6 | §4.3 | "4–5× gap" (0.90–0.96 / ≤0.21) | 4.3–4.6× | exact_match |
| 7 | §4.2 | "138,600 family-label assignments" | 12!/(4!·3!·4!·1!)=138,600 | exact_match |
| 8 | §4.2 | "1/C(7,4)=0.029" / "2/C(8,4)=0.029" | 0.029/0.029 | exact_match |
| 9 | §4.2 | "94% of the 51 cross-family pairs positive" | 48/51=94.1% (recomputed) | exact_match |
| 10 | §4.2 | "mean cross-family product 0.337" | 0.337 (recomputed) | exact_match |
| 11 | §4.2 | "decoder controls agree 0.220 mean pairwise" | 0.220 (recomputed from committed acc files) | exact_match |
| 12 | §4.2 | class prior "85–97%" | 78–100% (recomputed) | **mismatch → fixed** |
| 13 | §3.1 | Flowers102 2,040 images = 20×102 | 2,040 | exact_match |
| 14 | Supplement \S2 | attractor counts 47/30/24/21 answer/blackBoard/screen/stand→teacher | 47/30/24/21 (confusion matrix) | exact_match |
| 15 | §4.1 | S40 attenuation 0.892/0.819/0.783±0.038 | 0.8919/0.8188/0.7826±0.0376 | rounding_ok |
| 16 | §4.1 | anchor reliability 0.982 | 0.9821 | rounding_ok |
| 17 | §4.2 | CLIP-group per-model 0.473/0.454/0.521/0.402 | 0.4726/0.454/0.5205/0.4023 | rounding_ok |
| 18 | §4.3 | Mantel cells 0.662–0.982; SigLIP 0.927–0.957; ResNet 0.928–0.980 | files exact | exact_match |
| 19 | §4.3 | CKA cells 0.907/0.905/0.79–0.82/0.86/0.84 | files exact | exact_match |
| 20 | §4.3 | competence null max 0.944, margins +0.01–0.12 | mantel_null_baseline.json | exact_match |
| 21 | §4.4 | mediation S40 0.449–0.714; 66–123% prop | files exact | exact_match |
| 22 | §4.4 | counterfactuals +0.56/−0.006/−0.02; SCB5 +0.54 CI[−0.51,0.89] p=0.089 | files exact | exact_match |
| 23 | §4.7 | Flowers102 14/15 acc, 15/15 margin, SigLIP–ResNet 0.18/0.069 | files exact | exact_match |
| 24 | §4.8 | perturbation grid + FDR 8→0, static 6/6 | files exact | exact_match |
| 25 | §4.8 | SigLIP σ0 99.94%/18.0%/0.121–0.135 | files exact | exact_match |
| 26 | §4.1 | frequency control 0.797→0.835; jackknife 0.768/0.738, Δ≤0.043 | files exact | exact_match |
| 27 | §4.5 | prompt 45.3/0.0/1.7/0.0/0.0 → 98/100/69/92.5/73.8 | files exact | exact_match |
| 28 | §5.1 | family excess CLIP 0.788/0.462, SigLIP 0.742/0.716 | family_variation.json | exact_match |
| 29–41 | figures | fig5 ρ 0.817/0.810; fig2 means; figS1 CKA; fig3 cells | recomputed | exact_match |

## Failure-Mode Sweep (beyond number matching)

- **Best-seed cherry-pick**: no evidence of best-seed reporting; all aggregate values are means with CIs; single-realization routing results labeled as expected values (VA paper). PASS.
- **Config mismatch**: CLIP variants scored on identical image set (explicitly stated); LLaVA 100/class vs core-model full-n protocol difference disclosed and addressed by attenuation analysis. PASS.
- **Aggregation mismatch**: claimed resample counts match files (10k mediation/reliability/null-baseline; 20 splits seed 42; 10⁵ permutation draws in kruskal). PASS.
- **Delta/arithmetic errors**: none found (all ratios above recomputed). PASS.
- **Caption–table/figure mismatch**: figure-audit F1/F4 checks passed; captions match content (n, scales, method stated). PASS.
- **Scope overclaim**: "robust and general" backed by two-dataset replication + controls; FDR disclosed. PASS.

## Limitations

- Same-family audit (executor); zero-context reviewer unavailable by user instruction. Deviation recorded.
- The 78–100% recomputation used the committed raw response files with the eval's first-match matching logic; the original 85–97% figure's exact computation was not persisted as a number, so the corrected range is derived from the raw evidence.
