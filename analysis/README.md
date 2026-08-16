# analysis/

Post-processing scripts that consume `results/*.json` (produced by
`experiments/*.py`) and emit derived statistics, bootstrap CIs, permutation
p-values, and supplementary result JSONs. All scripts are pure-Python
(numpy / scipy) and run on CPU; none require a GPU. Each script has a module
docstring describing what it does and which result file it produces.

## Entry point

`run_all.sh` regenerates the core set of no-GPU analyses in sequence. Run
individual scripts directly for the rest:

```
bash analysis/run_all.sh            # core 6 analyses (permutation, groups,
                                    #   deep, Mantel, partial CI, Stanford40)
python3 analysis/<script>.py        # any other analysis, standalone
```

All scripts are deterministic (seeded) and idempotent.

## Inventory

### Statistical infrastructure (paper-wide)
| Script | Purpose | Output |
|--------|---------|--------|
| `permutation_critical_rho.py` | Permutation null for Spearman \|ρ\| at n=8/13/40 | `results/permutation_critical_rho.json` |
| `reliability_correction.py` | Attenuation correction + precision-weighted + binomial resampling | `results/reliability_*.json` |
| `power_analysis.py` | Fisher-z power analysis at the paper's sample sizes | stdout |
| `fdr_correction.py` | Benjamini–Hochberg FDR over all reported p-values (incl. §4.8 perturbation grid) | `results/fdr_correction.json` |
| `prevalence_and_leverage_controls.py` | Class-frequency partial control (SCB5) + Stanford40 jackknife/leverage | `results/prevalence_and_leverage_controls.json` |

### Step 1 — Phenomenon & encoder-group comparison
| Script | Purpose | Output |
|--------|---------|--------|
| `exp1_group_analysis.py` | Natural MLLM groups by encoder family (Kruskal–Wallis, permutation) | stdout |
| `exp1_deep_analysis.py` | LOO partial correlation, SCB5 | `results/exp1_deep_analysis_output.txt` |
| `exp1c_clip_group_expanded.py` | CLIP-encoder group expansion (n=1 → n=5) | `results/exp1c_clip_group_expanded.json` |
| `kruskal_4groups.py` | Four-group Kruskal–Wallis across encoder families | `results/kruskal_4groups.json` |
| `compute_partial_ci.py` | Bootstrap CI for partial correlations | stdout |

### Step 3 — Geometry (CKA, Mantel, cross-encoder)
| Script | Purpose | Output |
|--------|---------|--------|
| `mantel_test.py` | Exact Mantel tests (SCB5 8×8, Stanford40 40×40) | `results/mantel_test_results.json` |
| `mantel_bootstrap.py` | Stanford40 class-resampling bootstrap CI | `results/mantel_bootstrap.json` |
| `flowers102_mantel.py` | Flowers102 cross-domain geometry Mantel (102 classes, 10 encoder pairs) | `results/flowers102_mantel.json` |
| `merge_siglip_matrix.py` | Glue: merge SigLIP centroid matrix into `class_similarity_matrices.json` | updates `results/class_similarity_matrices.json` |
| `merge_clean_encoders.py` | Glue: merge ResNet-50 / DINOv2 centroid matrices | updates `results/class_similarity_matrices.json` |

### Step 4 — Mediation (Stanford40, SCB5)
| Script | Purpose | Output |
|--------|---------|--------|
| `stanford40_partial_correlation.py` | Stanford40 mediation + aggregated phenomenon | `results/stanford40_partial_correlation.json` |
| `mediation_combined_test.py` | Pooled mediation test for SCB5 (low-power concern) | `results/mediation_bootstrap.json` (SCB5 section) |
| `mediation_simulation.py` | Rank-z product-of-coefficients estimator consistency check | stdout |
| `mediation_tost_collinearity.py` | TOST equivalence + mediator-covariate collinearity | stdout |
| `mediation_counterfactual_controls.py` | Counterfactual mediation controls (class-permuted / noise / independent-encoder mediators) showing the indirect effect is a pipeline artifact only when real | `results/mediation_counterfactual_controls.json` |

### Step 4 supplement — Confusion-vs-geometry null test
| Script | Purpose | Output |
|--------|---------|--------|
| `confusion_vs_geometry.py` | Tests the geometry→confusion link within an MLLM (gemma4:31b); null result supports the "geometry" framing | `results/confusion_vs_geometry.json` |

### Step 4 supplement — Perturbation synchrony (blur)
| Script | Purpose | Output |
|--------|---------|--------|
| `blur_synchrony_v2.py` | CLIP-vs-MLLM per-class blur-drop ρ under shared PIL blur pipeline (dual-scored) | `results/blur_synchrony_v2.json` |
| `blur_synchrony.py` | Earlier version of the above (legacy, superseded by `_v2`) | `results/blur_synchrony.json` |
| `verify_blur_ckpt.py` | Reproducibility check for blur-pilot checkpoint files | stdout |

## Conventions

- **Inputs**: every script reads from `results/*.json` (committed). Path
  constants use `Path(__file__).resolve().parents[1] / "results"`, so the
  repo can be cloned anywhere.
- **Outputs**: each script writes its result directly to `results/`.
- **Determinism**: permutation and bootstrap seeds are set explicitly
  (typically `seed=42`); reruns produce identical numbers.
- **No GPU**: this entire directory is CPU-only. GPU experiments live in
  `experiments/`.
