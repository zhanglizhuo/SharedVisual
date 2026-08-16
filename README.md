# SharedVisual — Shared Class-Neighborhood Geometry: A Candidate Mechanism for Why CLIP Predicts MLLM Difficulty

Why does a frozen CLIP model's per-class zero-shot accuracy predict the per-class accuracy of much larger multimodal language models (MLLMs) with different vision encoders? We test a candidate mechanism: vision encoders trained on diverse large-scale data exhibit a broadly shared *class-neighborhood geometry*, which is associated with shared per-class difficulty across models. The paper evaluates this across twelve MLLMs (four encoder families), seven vision encoders (CLIP variants, SigLIP, QwenViT, ResNet-50, DINOv2), and three datasets (SCB5, Stanford40, Flowers102), using Mantel tests, CKA, bootstrap mediation with counterfactual controls, and prompt-ablation. All analyses are observational; causal confirmation awaits perturbation-based validation.

## Quick Start

```bash
git clone https://github.com/<org>/SharedVisual.git
cd SharedVisual
pip install -r requirements.txt          # CPU-only is enough for all analyses

# Every analysis script reads committed results and reproduces the paper's numbers:
python analysis/exp1c_clip_group_expanded.py        # Exp 1C encoder-group comparison
python analysis/mediation_tost_collinearity.py      # Mediation + TOST + VIF
python analysis/mediation_combined_test.py          # SCB5 combined mediation test
python analysis/mediation_counterfactual_controls.py # Counterfactual mediation controls
python analysis/kruskal_4groups.py                  # Kruskal-Wallis 4-group test
python analysis/mantel_bootstrap.py                 # Mantel class-resampling bootstrap
python analysis/power_analysis.py                   # Statistical power analysis
python analysis/fdr_correction.py                   # FDR correction (incl. perturbation grid)
python analysis/prevalence_and_leverage_controls.py # Class-frequency partial control + S40 jackknife
python paper/generate_figures.py                             # Regenerate all paper figures
```

No GPU, no external data downloads, no separate repositories required — every analysis reads the committed JSON snapshots under `results/` and `data/visualanchor_snapshot/`.

## Results Summary

| Experiment | Finding | Script | Output |
|-----------|---------|--------|--------|
| Exp 1C (encoder groups) | SigLIP ρ=0.716 > CLIP group ρ=0.462 (n=4); complete separation, Kruskal-Wallis H=8.93 p=0.030; CLIP-vs-QwenViT permutation p=0.029 | `analysis/exp1c_clip_group_expanded.py` | `results/exp1c_clip_group_expanded.json` |
| Exp 2 (CKA/RSA geometry) | CKA>0.87 within CLIP; 0.79–0.82 across encoders; class-geometry ρ=0.90–0.96 | `experiments/exp2_class_similarity_matrices.py`, `experiments/exp2_mllm_cka.py` | `results/exp2_cka_all_encoders.json`, `results/class_similarity_matrices.json` |
| Exp 2 (patch-level R²) | Linear ridge R²≤0.19 vs class-centroid Mantel 0.90–0.96 | `experiments/exp1b_linear_r2.py` | `results/exp1b_clean_r2.json` |
| Exp 3 (semantic vs visual) | CLIP failures resolved by prompts across all five low-accuracy classes (answer 0%→100%; blackboard-writing 45%→98%) | `experiments/exp3_prompt_variants.py` | `results/exp3_prompt_variants_v2.json` |
| Exp 3 (stand/screen extension) | All five low-accuracy classes rescued: stand 0%→92.5%, screen 0%→73.8% with descriptive prompts | `experiments/exp3_stand_screen_prompts.py` | `results/exp3_stand_screen_prompts.json` |
| Mediation (bootstrap a×b) | Stanford40: 6/6 indirect-effect CIs exclude 0; SCB5 combined sign test p=0.020 | `analysis/mediation_tost_collinearity.py`, `analysis/mediation_combined_test.py` | `results/mediation_bootstrap.json`, `results/mediation_combined_test.json` |
| Mediation counterfactual controls | Permuted/noise mediators collapse (S40 baseline +0.56 CI[0.37,0.71] p<0.0001 vs −0.006/−0.02); pipeline does not spuriously attribute | `analysis/mediation_counterfactual_controls.py` | `results/mediation_counterfactual_controls.json` |
| Mantel (class resampling) | Stanford40 CI [0.74, 0.87]; Flowers102 10/10 pairs p<0.0001 | `analysis/mantel_bootstrap.py`, `analysis/flowers102_mantel.py` | `results/mantel_bootstrap.json`, `results/flowers102_mantel.json` |
| Power analysis | n=13: 33% power for ρ=0.45 | `analysis/power_analysis.py` | `results/power_analysis.json` |
| FDR correction | raw + Mantel survive; partial ρ do not; perturbation grid 8 sig → 0 (static 6/6 survive) | `analysis/fdr_correction.py` | `results/fdr_correction.json` |
| Class-frequency control | ρ(anchor, log n)=−0.07, ρ(MLLM, log n)=−0.04 n.s.; partial ρ unchanged 0.797→0.835 | `analysis/prevalence_and_leverage_controls.py` | `results/prevalence_and_leverage_controls.json` |
| S40 leverage (jackknife) | max \|Δρ\|=0.043; removing 3/5 lowest-anchor classes 0.817→0.768/0.738 (p<0.001) | `analysis/prevalence_and_leverage_controls.py` | `results/prevalence_and_leverage_controls.json` |

## Datasets

- **SCB5** (Smart Classroom Behavior v5; 13 classes across 3 sub-datasets): download from [`huggingface.co/wintonYF/SCB-Dataset`](https://huggingface.co/wintonYF/SCB-Dataset) (~5.4 GB). Layout after download: `datasets_scb/{SCB5_TeacherBehavior,SCB5_HandriseReadWrite,SCB_BowTurnHead}/`.
- **Stanford40 Actions** (40 classes): download from [`cv.stanford.edu/projects/ActionCategories`](http://cv.stanford.edu/projects/ActionCategories/) — the per-class AnchorScores and MLLM accuracies used in the paper are already committed under `data/visualanchor_snapshot/02_robustness/stanford40/`, so re-running the Stanford40 analyses needs no download.
- **Flowers102** (102 classes): download from [`robots.ox.ac.uk/~vgg/data/flowers`](https://www.robots.ox.ac.uk/~vgg/data/flowers/). Only needed to re-run the geometric validation of §Flowers102.

The per-class CLIP AnchorScores and the six core MLLMs' per-class accuracies on SCB5 are committed under `data/visualanchor_snapshot/01_core/`, so all analyses in `analysis/` run without downloading anything.

## Re-running MLLM Evaluation (GPU required)

The paper's CLIP-encoder MLLM results were produced with `experiments/eval_hf.py` (HF transformers, class-name prompt, 100 images/class, seed 42, greedy decoding, first-match substring scoring). To re-evaluate any of the four CLIP-group models:

```bash
python experiments/eval_hf.py llava-hf/llava-1.5-7b-hf       --data-dir /path/to/datasets_scb
python experiments/eval_hf.py llava-hf/llava-1.5-13b-hf      --data-dir /path/to/datasets_scb
python experiments/eval_hf.py llava-hf/llava-v1.6-mistral-7b-hf --data-dir /path/to/datasets_scb
python experiments/eval_hf.py llava-hf/llava-v1.6-vicuna-7b-hf  --data-dir /path/to/datasets_scb
```

Outputs (per-class accuracy JSONs) are saved to `results/llava_hf_*_scb5.json` and feed into `analysis/exp1c_clip_group_expanded.py` and `analysis/kruskal_4groups.py`.

> **Note:** LLaVA-NeXT-Llama3-8B was evaluated but excluded: its Llama3 chat template is incompatible with the uniform `"USER: ... ASSISTANT:"` prompt, yielding all-zero accuracy. See the analysis script comments for details.

## Requirements

- Python 3.10+
- See `requirements.txt` for the full pinned list
- A GPU (any CUDA-capable card, ≥16 GB) is required **only** to re-run MLLM evaluation (`experiments/eval_hf.py`) and feature extraction (`experiments/exp*.py`). All statistical analyses under `analysis/` are CPU-only and run in seconds.

## Contents

| Directory | Description |
|-----------|-------------|
| `paper/` | Manuscript (tex/PDF), figures, bibliography |
| `analysis/` | Statistical analysis scripts (CPU-only) |
| `experiments/` | Evaluation and feature-extraction scripts (GPU) |
| `results/` | Committed JSON outputs for every figure and table |
| `data/visualanchor_snapshot/` | Upstream per-class accuracy snapshots (CLIP AnchorScores + MLLM accuracies) that feed the analyses |
| `sharedvisual/` | Shared Python package: centralized paths (`paths.py`), dataset definitions (`datasets.py`), and statistical helpers (`stats.py`) |
| `dev/` | One-off diagnostic / connectivity-test scripts (not part of the paper pipeline) |

## Configuration (for GPU scripts only)

GPU experiment scripts under `experiments/` read dataset locations from environment variables (all optional; defaults use `__file__`-relative paths or `~/data`):

```bash
export SHAREDVISUAL_ROOT=/path/to/SharedVisual          # repo root (auto-detected if omitted)
export SHAREDVISUAL_DATA=/path/to/datasets_scb          # SCB5 / Stanford40 data
export SHAREDVISUAL_RESULTS=$SHAREDVISUAL_ROOT/results  # outputs
export SHAREDVISUAL_WORK=/tmp/exp1b                      # scratch for feature caches
```

All CPU analysis scripts under `analysis/` need no configuration — they read committed snapshots under `data/` and `results/`.

## Reproducibility

Every number in the paper traces to a committed, seeded script. The full provenance is documented in the paper's *Reproducibility* section. The analyses are deterministic (fixed seeds; bootstrap/mediation use seed `20260801`, Mantel permutations use seed `42`).
