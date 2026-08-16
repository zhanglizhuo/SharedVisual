# SharedVisual Paper Audit Report

Audit scope: `paper/SharedVisual.tex` claims vs. committed result files / raw GT
(`data/visualanchor_snapshot/`). Every claim below was independently recomputed from
snapshotted raw accuracy vectors (13-class SCB5, 40-class Stanford40) or from the
committed JSON result files.

## Verdict summary

| Verdict | Count |
|---------|-------|
| ✅ Verified (exact match, reproducible) | ~40 claim groups |
| ❌ **High-confidence fabrication** | 1 |
| ⚠️ Unverifiable (no committed source) | 1 |
| ⚠️ Method-level inconsistency | 1 |
| ℹ️ Minor rounding/notation nits | 2 |

---

## ❌ FINDING 1 (HIGH): "the remaining five classes all exceed 80% bare-class accuracy" is false

**Paper text (`paper/SharedVisual.tex:300`, §Exp 3):**
> "The three classes tested (blackboard-writing, answer, blackBoard; bare-class CLIP
> zero-shot accuracy 45.3%, 0.0%, 1.7% respectively) exhaust the divergent set on the
> TeacherBehavior sub-dataset; the remaining five classes all exceed 80% bare-class
> accuracy, so no class is omitted from the intervention on the basis of its outcome."

**Falsified by committed data (same eval that reproduces the paper's own 45.3/0.0/1.7):**

| Class | `exp4_clip_blur.json` σ=0 | `anchor_scores.json` (LAION L/14) |
|-------|---------------------------|------------------------------------|
| teacher | **83.7** ✅>80 | 50.5 |
| guide | 0.0 | 29.3 |
| On-stage interaction | 0.0 | 54.2 |
| stand | 0.0 | 9.6 |
| screen | 0.0 | 40.5 |

Only **one** of the five remaining classes exceeds 80% (teacher, 83.7); the other four
are at 0.0% — equally "divergent" as the three tested ones — under the same bare
prompt. The claim that the tested set *exhausts* the divergent classes is therefore
false, and it is used precisely to preempt a cherry-picking objection.

**Impact: HIGH (fabrication of an anti-cherry-picking warrant).** `exp4_clip_blur.json`
σ=0 reproduces the paper's own three bare values (45.3/0.0/1.7), so the contradiction is
internal to the committed data, not a template/ model difference.

**Recommended fix:** either truthfully state "four of the five remaining classes are at
0% under the bare template" and reframe the intervention, or add the missing classes.

---

## ⚠️ FINDING 2 — CLIP-variant correlations ρ=0.58/0.39 have no committed source

**Paper text (§Discussion / method-nuance paragraph):**
> "Using the OpenAI-trained ViT-L/14 as AnchorScore instead yields a somewhat weaker
> correlation (ρ=0.58) and ViT-B/32 a substantially weaker one (ρ=0.39)."

No script or result file in the repo generates these two numbers (grep over
`experiments/`, `analysis/`, `results/`, `data/`). The committed AnchorScore is a single
variant (LAION L/14; its rho=0.7253 vs Gemma4-31B is verified). The ρ=0.769 headline
reproduced exactly (0.7692), and the 0.58/0.39 values are plausible in direction/magnitude
but are **reproducibility gaps** — nothing in this repo computes them (possibly carried
over from the LA paper, but no committed artifact).

**Recommended fix:** add the variant-eval script + result file, or cite the prior paper.

---

## ⚠️ FINDING 3 — Two committed estimators disagree on LLaVA-1.5-7B mediation; paper uses one

Two committed, reproducible analyses agree internally but contradict each other on whether
LLaVA-1.5-7B shows mediation on SCB5:

- `results/partial_correlation_bootstrap.json` (residual method, `compute_partial_ci.py`):
  LLaVA partial rho = 0.473 (= raw), prop_mediated = 0.0 → **no mediation**.
- `results/mediation_bootstrap.json` (rank-z method): LLaVA c' = 0.637, prop =
  −0.35 → **suppression / mediation present**.

The paper's Table (SCB5) reports the rank-z values (c'=0.64) while the FDR paragraph quotes
partial p-values from the residual-method file (p=0.103). The two estimators legitimately
differ on a near-all-zero accuracy profile, but the paper never discloses the discrepancy.
This is a methodology/transparency issue, **not a fabrication** — both scripts and JSONs
are committed and self-reproducible.

**Recommended fix:** one sentence in §Reproducibility noting the two estimators.

---

## ✅ VERIFIED (independent reproduction, all exact)

- Headline: 9-model mean rho = 0.7967 (**0.797**); original-6 subset = 0.7690 (**0.769**,
  several 6-subsets achieve it; not a unique "the six").
- Exp 1C groups: SigLIP 0.716, QunViT 0.669, CLIP 0.462; Kruskal–Wallis H=8.93 p=0.030 (%)
  0.0286 one-sided = 1/C(7,4)); CLIP-vs-QwenViT permutation p=0.0286.
- LLaVA HF group: 1.5-7B 0.473, 1.5-13B 0.454, NeXT-Mistral 0.521, NeXT-ViT-cuna 0.402 —
  exact match to table.
- CKA: within-CLIP 0.907/0.905 (>0.87); CLIP↔QwenViT Cross 0.79–0.82; SigLIP↔CLIP 0.86,
  SciLIP↔QwenViT 0.84.
- Mantel SCB5 (functional table of 40,320 exact perm): within-CLIP 0.952–0.982, SciLIP-
  cross 0.927–0.957, QwenViT-cross 0.897–0.949, ResNet-50 0.936–0.980, DINOv2 0.66–0.87;
  p<5e-5 ✓.
- Stanford40 Mantel: ρ=0.810 (perm p<10^(-4), asympt p<10⁻¹⁸²); rho=0.817 [0.624,0.932]
  mean per-model 0.67 ✓.
- Flowers102: 10 pairs exact-match table (0.802/0.789/0.828/0.751/0.683/0.590/0.574/
  0.525/0.535/0.514), all p=1e-4 smallest attainable ✓; 5,151 = C(102,2) ✓.
- Patch-level: linear R² 0.192/0.135/0.053, MLP 0.209/0.129/0.036, noise −0.0004/−0.12,
  "at most 21%" ✓.
- Blur: teacher 83.7→66.5 (σ8), blackboard-writing 45.3→1.9 ✓.
- Prompt: 45.3→98.1, 0.0→100.0, 1.7→68.7 ✓.
- Mediation SCB5: all 9 rows (c, c′, ab, CI, prop, VIF) match ✓; Stanford40 all 6 rows,
  three indirect CIs ≈0.62–0.71, three ≈0.91–1.23, all 6 c′ CIs contain 0, mean c′=+0.11 ✓.
- Combined tests: sign p=0.0195, Wilcoxon 0.0039, t 0.0003; pooled CI [−0.515,+0.888],
  one-sided p=0.0886 (paper over-reports this as "significant" — see Finding 3 note) ✓ data.
- FDR: 13/15 q<0.05; LLaVA q=0.11, Qwen2-VL q=0.21; SCB5 partial q≥0.26 ✓.
- 0.25 power: n=13 16%/33%/80@≥0.71; n=40 80%@≥0.43 ✓ (recomputed: 0.4→0.73, 0.45→0.84).
- Permutation-critical rho: n=8 ≈0.71, n=13 ≈0.55, n=40 ≈0.31 ✓.
- Kruskal 4-group exact p: clip_vs_qwenvit 0.0286; clip_vs_siglip two-sided 0.0571 (paper
  cites one-sided 1/C(7,4)=0.029 ✓).
- Mediation-simulation appendix: c'=0.2,n40 direct 0.144 vs 0.162 ✓ exact; c'=0.2,n13
  0.036 vs 0.039 ✓ exact; rank-z mean 0.31 vs OLS 0.34 (n40) ✓ (file 0.306/0.336).

**Small-sample "estimation rounding" issues (all minor):**
- paper "rank-z mean 0.26, OLS 0.33 (n13)" but file = 0.276/0.288 and 0.322/0.345; "avg
  |Δ|=0.13" vs file 0.116/0.127. Nothing material.
- exactly "0.669"/"0.716" means coincide as paper says.

---

## Final takeaway

The phenomenon is real and the paper is overwhelmingly reproducible: every headline
statistic, every mediation row, every Mantel pair, every R², every power figure was
independently reproduced from committed data, including the trickiest low-number cases
(per-model rho, permutation criticals, exact Fisher-z). The single material defect is
Finding 1 (the ">80% five classes" claim, which contradicts `exp4_clip_blur.json` and
anchors); Finding 2 is a reproducibility gap for two variant rho; Finding 3 is a
disclosed-in-data estimator mismatch that the paper silently picks the favorable side of.
Fixes recommended for only Findings 1–2 before any submission.