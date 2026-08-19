# Figure and Table Audit Report

**Date**: 2026-08-18
**Scope**: SharedVisual submission package — paper/SharedVisual.tex (28 pp) + paper/supplemental.tex (5 pp), 6 figures (5 main + S1), 6 tables (4 main + 2 SI)
**Inputs checked**: both .tex sources, compiled PDFs, figures/*.pdf, generate_figures.py, results/*.json (source data)
**Build/source status**: all figures script-generated (generate_figures.py, deterministic, Type42 vector PDF+PNG+SVG+TIFF); tables inline LaTeX from committed JSONs
**Target venue**: Neurocomputing (Elsevier, cas-sc)

**Summary**: 0 blocking, 2 recommended, 1 minor, 0 author-verification

## Inventory

| ID | Path/location | Caption/title | First callout | Source/script |
|----|---------------|---------------|---------------|---------------|
| Fig 1 | paper/figures/fig1_phenomenon_overview.pdf | Hypothesized two-level model | §1 (L116) | generate_figures.py |
| Fig 2 | paper/figures/fig2_not_shared_encoder.pdf | Group-level ρ by encoder family | §4.2 (L201) | generate_figures.py ← kruskal_4groups.json |
| Fig 3 | paper/figures/fig3_class_geometry.pdf | 7 class-similarity matrices + Mantel matrix | §4.3 (L247) | generate_figures.py ← class_similarity_matrices.json, mantel_test_results.json |
| Table 1 | main §4.3.3 (tab:patchlevel) | Per-patch transfer vs prototype geometry | §4.3.3 (L268) | adapter runs (results committed) |
| Table 2 | main §4.4 (tab:mediation) | Bootstrap mediation c/c′/a×b | §4.4 (L296) | mediation_tost_collinearity.py |
| Fig 4 | paper/figures/fig4_prompt_variants.pdf | Prompt variants resolve CLIP failures | §4.5 (L371) | exp3_prompt_variants_v2.json, exp3_stand_screen_prompts.json |
| Fig 5 | paper/figures/fig5_stanford40.pdf | Stanford40 4-panel | §4.6 (L376) | stanford40_external/*, exp6_stanford40_geometry.json |
| Table 3 | main §4.7 (tab:flowers102) | Mantel, 5 encoders | §4.7 (L390) | flowers102_mantel.py |
| Table 4 | main §4.7 (tab:flowers102b) | 1-shot probe, 15 pairs | §4.7 (L420) | exp_flowers102_difficulty_multi.py |
| S-Fig 1 | paper/figures/figS1_cka_heatmap.pdf | Pairwise CKA, 7 encoders | main §4.3 ("Supplement Fig.~1") | generate_figures.py ← exp2_cka_all_encoders.json |
| S-Table 1 | supplement §3 (tab:mediation_controls) | Counterfactual mediator controls | — (no in-supplement callout) | mediation_counterfactual_controls.py |
| S-Table 2 | supplement §4 (tab:per_model_rho) | Per-model ρ, SCB5 | supplement §4 (L122) | kruskal_4groups.py |

## Blocking Issues

None. Verified: every in-text callout resolves (0 undefined refs, full bibtex pipeline); numbering gap-free and non-colliding (main 1–5 / 1–4; SI S1 / S1–S2); callout order = number order; all values in this session's figure reviews traced to raw JSONs (fig2 means 0.462/0.716/0.669/0.374; fig4 45.3→98.1, 0→100, 1.7→68.7, 0→92.5, 0→73.8; fig5 ρ=0.817/0.810; Table 2 rows; Table 4 exception row SigLIP–ResNet ρ=0.18/p=0.069; S-Table 2 values = kruskal JSON).

## Recommended Fixes

| Location | Figure/table | Issue | Fix |
|----------|-------------|-------|-----|
| supplement §3 | S-Table 1 (tab:mediation_controls) | Table appears under "Results." with no textual `\ref` callout | Add "Table~\ref{tab:mediation_controls} reports the full numbers." |
| supplement §5 | S-Fig 1 (fig:cka_matrix) | Figure appears with no in-supplement `\ref` | Add "\Cref{fig:cka_matrix} reports..." (main paper's manual "Supplement Fig.~1" already matches S1 numbering) |

## Minor / Production Issues

| Figure/table | Issue | Fix |
|--------------|-------|-----|
| Supplement tables | Decimal precision 3dp consistent; table S2 p-values 3dp vs main text 2–3dp mix — cosmetic only | leave as-is |

## Author Verification Needed

None — all plotted values are either labeled in-figure or sourced from committed JSONs; no eyeball readings were required.

## Readiness Checklist

| Dimension | PASS/FAIL/PARTIAL/NA | Notes |
|-----------|----------------------|-------|
| Callouts resolve | PASS | 0 undefined refs |
| Numbering gap-free | PASS | main 5 figs + 4 tables; SI S1 fig + S1/S2 tables |
| Text↔table values | PASS | session-wide numeric verification (1692=1692 multiset; per-figure data checks) |
| Captions self-contained | PASS | units/sample/uncertainty/panels/axis-truncation disclosed (fig2, fig4, fig5, figS1) |
| Table notes complete | PASS | definitions, CIs, suppression dagger, VIF, seeds |
| Colorblind/grayscale safe | PASS | Blues_r single-hue + grays; red reserved for signal |
| Vector production quality | PASS | Type42 (searchable), effective fonts 8.1–11.7pt, no overfull in figures |
| SI↔main linkage | PASS | Supplement §1–§4 references verified; all main "Supplement \S*" pointers land correctly |
| Script traceability | PASS | every float traced to generate_figures.py or named analysis script + committed JSON |
| Source data archived | PASS | results/*.json committed; README maps outputs |
