# Link B — Encoder-Swap Causal Experiment (DESIGN, preregistered, NOT YET RUN)

**Status**: design + code committed; execution deferred until V100 window.
**Goal**: fill the causal gap (Link B of the paper's roadmap, §5.6): does the vision
encoder's class geometry causally shape MLLM per-class difficulty, holding the
language model fixed?

**This document preregisters hypotheses, arms, budgets, and test statistics
BEFORE any training run.** Deviations after unblinding must be reported as such.

---

## 1. Scientific question

The paper's evidence is observational: encoders share geometry AND models share
difficulty. Naturally occurring MLLMs confound the vision encoder with the LM
(each family changes both). This experiment isolates the encoder contribution:

> With the LM held fixed (frozen), does swapping the vision encoder change the
> per-class difficulty profile in the direction its geometry predicts?

## 2. Design

### Arms (3 + 1 control)

| Arm | Encoder | Why | Geometry to CLIP-L-LAION (SCB5 TB 8×8, exact) |
|-----|---------|-----|-----------------------------------------------|
| C | CLIP ViT-L/14 LAION-2B | the paper's AnchorScore backbone (anchor arm) | — |
| S | SigLIP ViT-SO400M-14 (webli) | strong encoder, highest geometry agreement, different alignment quality (its own zero-shot is weak, ρ=0.201 — alignment ≠ geometry) | **0.957** |
| D | DINOv2 ViT-B/14 | self-supervised, no text supervision, geometry farther | **0.719** (S–D: **0.772**) |
| R (control) | randomly initialized ViT (seed 42) | validity check: pipeline must NOT produce difficulty agreement from nothing | n/a (no trained geometry) |

ResNet-50 was considered and dropped: its geometry agreement to CLIP (0.928–0.980)
is too close to arm C to add discriminative contrast.

### Fixed components

- **LM**: Mistral-7B-v0.1 (Apache-2.0), frozen. (Vicuna-7B alternative rejected
  for LLaMA-2 license friction.)
- **Projector**: LLaVA-1.5-style 2-layer MLP per arm (Linear→GELU→Linear),
  mapping each encoder's per-patch token dim → LM hidden dim (4096). Only the
  projector trains.
- **Training data**: SCB5 train pool, 13 classes pooled, ALL train images
  minus cross-dataset filename collisions with val (the exp1b leakage
  convention; ~104 collisions). Note: the plan draft cited the exp1b bounded
  pool of 4,896; the implementation uses the full leakage-cleaned train pool
  (larger), recorded in RUN_MANIFEST as a pre-registered deviation — more
  training data, identical exclusion rule, equal budget per arm unchanged.
  Class-balanced sampling.
- **Task**: classification-style instruction tuning. Prompt:
  `Classify the person's behavior in this classroom image into one of: {classes}. Answer with the class name only.`
  Target = gold class name; loss on answer tokens only.
- **Evaluation**: SCB5 validation, full-frame (deployment condition; bbox crops
  are unavailable at prediction time — consistent with the VA realized-run
  protocol), class-name prompt, greedy decoding, first-match class-name scoring
  (`match_class_name`, identical to `experiments/eval_hf.py`).

### Equal budget (preregistered)

All three trainable arms share: identical optimizer (AdamW), LR (1e-4, cosine to
0), warmup (3%), **fixed 2,000 optimizer steps**, batch 8 (grad-accum as needed),
fp16, seed 42, class-balanced sampler. The random arm R trains with the same
budget. "Tuned better" is thereby excluded as an explanation for differences:
any arm advantage cannot come from extra compute.

## 3. Predictions (preregistered, ordinal)

Let ρ_diff(X,Y) = Spearman between arms X,Y's per-class accuracy vectors
(difficulty profiles; 13 classes pooled; TB-8 secondary).

The paper's account (H3) predicts the difficulty-profile agreement tracks
geometry agreement:

- **P1 (primary, ordinal)**: ρ_diff(C,S) > ρ_diff(S,D) ≈/> ρ_diff(C,D),
  mirroring geometry 0.957 > 0.772 ≈/> 0.719.
- **P2 (matrix)**: across the 3×3 matrix, behavior-agreement ranks correlate
  with geometry-agreement ranks (Mantel/Spearman of upper triangles; exact
  permutation p).
- **P3 (LM-buffering boundary, operationalized)**: all arms may converge toward
  the LM's own prior difficulty profile (LM-attractor, cf. gemma4 in Supplement
  §1–2). P3 passes iff the 10k class-bootstrap (seed 20260801) 95% CI of the
  DIFFERENCE ρ_diff(C,S) − ρ_diff(C,R) excludes zero — a preregistered
  inferential criterion, not a point-estimate comparison. Power caveat applies:
  a CI including zero is reported as inconclusive-with-direction, not evidence
  of absence.
- **P4 (validity)**: the random arm R must (a) have markedly lower mean
  accuracy, and (b) NOT show geometry-ordered agreement (its pair agreements
  should be the lowest or noise-level). If R produces the same ordinal pattern,
  the pipeline manufactures agreement and the experiment is uninformative.

### Interpretation table (pre-committed)

| Outcome | Reading | Action for the paper |
|---------|---------|----------------------|
| P1+P2 pass, P3 passes above floor | encoder geometry causally shapes difficulty, LM fixed | upgrade §4 narrative: "first direct causal test passed"; still keep "candidate" (single experiment) but add the result |
| P3 fails: all arms equal (LM-attractor dominates) | honest boundary: geometry shared in the visual layer, but a frozen LM + small projector buffers behavior | report as boundary; reframe §5.2; do NOT overclaim |
| P4 fails | pipeline artifact | do not use; debug and rerun with disclosure |
| Any of C/S/D floor-affected (mean acc < 20% or collapse > 0.80) | projector-only training insufficient at this budget | report with floor flag; ordinal tests carry no causal reading; budget changes require a NEW preregistered run |
| D arm deviates from C/S more than geometry predicts | AMBIGUOUS by preregistration: mixes "geometry farther" with "projector harder to learn" (DINOv2 features carry no text alignment; the 2000-step projector must also learn the vision-language bridge) | D-arm evidence preregistered as weaker than the C-S pair; C-S is the primary causal comparison, D a directional check only |

Negative/positive, the run is reported. No post-hoc arm substitution.

## 4. Statistics

- Difficulty profiles: per-class accuracy per arm (13 classes pooled; also TB-8).
- **Geometry reference (primary)**: 13-class pooled class-centroid cosine
  matrices computed from THIS experiment's cached full-frame val features (the
  same features feeding behavior), pairwise Spearman of upper triangles. This
  aligns the geometry reference's class set with the behavior evaluation
  exactly. The committed TB-8 matrices (C-S 0.957, S-D 0.772, C-D 0.719) are
  the preregistered SECONDARY reference; divergence between the two references
  is reported, not adjudicated post hoc.
- ρ_diff: Spearman; CIs via class bootstrap (10k, seed 20260801 — paper convention).
- Ordinal test P1: exact permutation over the 3 pair values against the
  preregistered ordering. **Ceiling disclosed in advance**: with 3 pair values,
  exactly 1 of the 3! = 6 permutations satisfies the predicted partial order, so
  the smallest attainable one-sided p is **1/6 ≈ 0.167**; a perfect match is
  preregistered as *descriptive support*, not inferential significance (same
  disclosure convention as the paper's "smallest attainable" p=0.029 tests).
- Matrix test P2: Spearman of upper triangles (3 pairs) vs the geometry
  reference (primary: 13-class pooled geometry from this experiment's cached
  features; secondary: committed TB-8 matrices). **Ceiling disclosed in
  advance**: with 3 points, the two-sided exact test over 6 permutations has
  smallest attainable p = **2/6 = 0.333**; P2 is likewise descriptive.
- **Preregistered floor criterion (trainable arms; locked before unblinding) —
  the failure mode that killed both prior perturbation attempts**: an arm is
  flagged *floor-affected* if mean val accuracy **< 20%** (13-class chance
  7.7%; 20% ≈ 2.6× chance) OR single-class response fraction (collapse)
  **> 0.80** (the paper's strong-prior/attractor threshold). If ANY of C/S/D is
  floor-affected, P1/P2 are reported with the flag and carry no causal reading;
  if ALL are, the run is underpowered and reported as such.
- Power caveat (preregistered): at n=13, 80% power requires |ρ|≥0.71 (paper's
  Fisher-z analysis). We therefore pre-commit to interpreting CIs + ordinal
  consistency, not point-significance alone. TB-8 (n=8) is descriptive only.
- LM-attractor collapse check: per-arm fraction of val responses mapping to the
  single most-common class (if >80%, report collapse rate alongside accuracy).

## 5. Compute plan (V100×4, 32GB each)

1. `01_extract_features.py` — per-arm per-image features (train + val, full
   frame), cached as fp16 npz per (arm, split). ~10–30 min total.
2. `02_train_projector.py` — 4 arms × 2,000 steps, frozen 7B. DataParallel or
   single-GPU-per-arm (4 arms = 4 GPUs in parallel; each arm fits one V100 with
   grad-checkpointing + fp16). ~3–6 h/arm.
3. `03_eval_swap_variants.py` — val inference per arm (5,416 images full-frame),
   greedy; ~1 h/arm on 1 GPU.
4. `analysis/linkb_analysis.py` — CPU, seconds.
   Total wall-clock ≈ one V100×4 day. Checkpointed/resumable at every stage
   (remote-execution discipline: background + log + poll).

## 6. Implementation notes

- Feature shapes: C: 257×768 (224/14²+1); S: 730×1152 (384/14²+1); D: 257×768;
  R: 257×768 (random ViT-B, seed 42).
- Longest-sequence arm (S, 730 tokens) governs LM context; truncation guard at
  1024 tokens total (prompt + answer).
- Determinism: torch/np/random seeds fixed; fp16 nondeterminism acknowledged
  (batch order seeded; matmul nondeterminism noted in report).
- Leakage: reuse the exp1b collision list logic verbatim (md5+filename) — no
  train image may appear in any arm's val set.
- Scoring: first-match substring `match_class_name` copied from
  `experiments/eval_hf.py` (protocol continuity with the paper's LLaVA evals).

## 7. File map

- `01_extract_features.py` — encoder features (4 arms), cached
- `02_train_projector.py` — equal-budget projector training, resumable
- `03_eval_swap_variants.py` — per-class accuracy on val
- `analysis_linkb.py` — predictions P1–P4, stats, JSON output
- `run_linkb.sh` — V100 orchestration (nohup + logs, per-stage idempotency)

All scripts carry `--dry-run` for pipeline validation without GPU work.
Nothing here has been executed; first run must record actual configs into
`results/linkb/RUN_MANIFEST.json`.
