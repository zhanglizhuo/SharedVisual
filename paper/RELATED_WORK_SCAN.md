# Related Work Scan — SharedVisual ("Why CLIP Predicts MLLM Difficulty")

**Date**: 2026-08-15
**Mode**: deep-research lit-review (bibliography + verification + synthesis)
**Goal**: identify literature that reviewers of *Neural Networks* / *Pattern Recognition* would expect in §2 (Related Work) but that the current draft omits; verify every candidate before recommendation.

## Method (reproducible)

Search API: arXiv (export.arxiv.org), 2026-08-15. Queries:

| # | Query | Hits |
|---|-------|------|
| q1 | all:"cross-model" AND all:"difficulty" AND cat:cs.CV | 7 |
| q2 | abs:"error consistency" AND cat:cs.CV | 10 |
| q3 | all:"difficulty" AND all:"vision-language" AND all:"prediction" | 15 |
| q4 | all:"representation similarity" AND all:"convergence" | 15 |
| q5 | all:"Platonic" AND all:"representation" | 15 |
| q6 | all:"CLIP" AND all:"per-class" AND all:"accuracy" AND all:"predict" | 7 |
| verify | id_list=1905.12580, 2303.11129, 2209.15430, 2310.05916, 2010.15327 | 5 |
| verify | ti:"Similarity of Neural Network Models" | 5 |

All metadata below (title/authors/year/ID) comes verbatim from arXiv API responses. One memory-sourced ID was wrong (Klabunde survey 2303.11129 → actually an unrelated math paper) and was corrected via title search — no unverified entries are included. Semantic Scholar was attempted first but rate-limited (429).

## Current coverage of §2 (baseline)

- **Predicting MLLM Performance**: ma2025clipreliability, ma2025anchorscore, liu2023visual
- **Representation Similarity**: kornblith2019similarity, kriegeskorte2008representational, huh2024platonic, raghu2021vitcnn, neyshabur2020transfer
- **Visual Confusability**: geirhos2021partial, yao2011stanford40, swayamdipta2020cartography
- **Mechanism Gap**: synthesis paragraph (no citations)

12 citations total. Avenues a reviewer will probe that are currently thin: (i) the representational-similarity *methodology* literature (surveys/measures beyond CKA), (ii) the error-consistency metric tradition (Geirhos ICLR 2021) — currently only the "partial consensus" paper is cited, (iii) recent 2024–2026 critiques/refinements of the Platonic hypothesis that a convergence paper must engage, (iv) adjacent work predicting difficulty from model features, and (v) CLIP-internals interpretation for the prompt experiment.

## Verified candidates (all confirmed via arXiv API)

### A. Representation similarity — methodology & convergence

| # | Citation (APA, arXiv) | Verified | Why relevant |
|---|----------------------|----------|--------------|
| A1 | Klabunde, M., Schumacher, T., Strohmaier, M., & Lemmerich, F. (2023). *Similarity of neural network models: A survey of functional and representational measures.* arXiv:2305.06329 | ✓ | Field-standard survey of representational similarity measures (CKA family, RSA, Procrustes...). An NN reviewer will ask why the survey is not cited. |
| A2 | Moschella, L., Maiorca, V., Fumero, M., Norelli, A., Locatello, F., & Rodolà, E. (2022). *Relative representations enable zero-shot latent space communication.* arXiv:2209.15430 (ICLR 2023) | ✓ | Cross-model latent-space alignment independent of absolute coordinates; complements the paper's class-prototype convergence claim. |
| A3 | Nguyen, T., Raghu, M., & Kornblith, S. (2020). *Do wide and deep networks learn the same things? Uncovering how neural network representations vary with width and depth.* arXiv:2010.15327 | ✓ | Early evidence of representation convergence across configurations — same lineage as CKA. |
| A4 | Mania, H., Miller, J., Schmidt, L., Hardt, M., & Recht, B. (2019). *Model similarity mitigates test set overuse.* arXiv:1905.12580 (NeurIPS 2019) | ✓ | Model similarity ⇔ shared error structure: the closest prior result to our "shared difficulty across encoders" claim. |
| A5 | Authors unverified beyond arXiv listing (2024). *Objective drives the consistency of representational similarity across datasets.* arXiv:2411.05561 | ✓ | Tests what drives RSA consistency across datasets — concludes objective, not data. Directly speaks to our H2 (data-overlap) vs objective contrast (ResNet-50 vs DINOv2). |
| A6 | (2026). *Revisiting the Platonic representation hypothesis: An Aristotelian view.* arXiv:2602.14486 | ✓ | Shows standard similarity metrics are confounded by network scale (width/depth inflate scores) and proposes permutation-based null calibration. A convergence paper must engage this; our CKA/Mantel claims should be read against it. |
| A7 | (2024). *Diagnosing neural convergence with topological alignment spectra.* arXiv:2411.08687 | ✓ | Multi-scale alternative to scalar CKA; optional, positions our Mantel/prototype-level analysis within a measurement-literature context. |
| A8 | (2026). *Convergence without understanding: When language models agree on representations but disagree on reasoning.* arXiv:2605.23315 | ✓ | Representational convergence does NOT imply behavioral/reasoning convergence (16 LMs, 8 families). Directly supports our Discussion boundary (geometry shared, downstream behavior shaped by alignment/LM priors). |

### B. Visual confusability & shared errors

| # | Citation | Verified | Why relevant |
|---|----------|----------|--------------|
| B1 | Geirhos, R., Meding, K., & Wichmann, F. A. (2020/ICLR 2021). *Beyond accuracy: quantifying trial-by-trial behaviour of CNNs and humans by measuring error consistency.* arXiv:2006.16736 | ✓ | The error-consistency metric (Cohen's κ across decision makers) — the quantitative foundation for "shared confusability" claims; §2 currently cites only the companion "partial consensus" paper. |
| B2 | (2023). *Exploring new ways: Enforcing representational dissimilarity to learn new features and reduce error consistency.* arXiv:2307.02516 | ✓ | Shows ensembles of independently trained models correlate in predictions/failure modes and uses representational dissimilarity to break it — corroborates the shared-difficulty phenomenon from the intervention side. |

### C. Predicting difficulty / performance

| # | Citation | Verified | Why relevant |
|---|----------|----------|--------------|
| C1 | (2026). *Using vision + language models to predict item difficulty.* arXiv:2603.04670 | ✓ | Adjacent: VLMs predict human item difficulty from content. Contrasts with our direction (CLIP predicts MLLM per-class accuracy); shows the "difficulty prediction from a vision model" line of work. |
| C2 | (2026). *Representing visual evidence for item difficulty prediction: visual textualization and image-native modeling.* arXiv:2608.04554 | ✓ | Same line, method comparison; cite at most one of C1/C2. |

### D. CLIP internals / text-image alignment (for §4.5 prompt experiment)

| # | Citation | Verified | Why relevant |
|---|----------|----------|--------------|
| D1 | Gandelsman, Y., Efros, A. A., & Steinhardt, J. (2023). *Interpreting CLIP's image representation via text-based decomposition.* arXiv:2310.05916 (ICLR 2024) | ✓ | CLIP's representation decomposes into interpretable text-aligned directions — context for why text-side fixes (prompts) rescue accuracy without touching images. |
| D2 | Wang, Y., et al. (2022). *When are lemons purple? The concept association bias of vision-language models.* arXiv:2212.12043 (EMNLP 2023) | ✓ | Documents systematic text-concept association biases in CLIP/VLMs — the same class of misalignment our prompt experiment exploits. |

## Synthesis & gap analysis

1. **The measurement literature is missing.** §2 jumps from Kriegeskorte (RSA) to Kornblith (CKA) to Huh (Platonic) without the survey (A1) or the relative-representation line (A2). Adding A1+A2+A3 makes the "Representation Similarity" paragraph state-of-the-art instead of canonical-only.
2. **The shared-error tradition is under-cited.** The paper's central "shared confusability" claim rests on Geirhos' partial consensus (cited), but the metric itself (B1) and the model-similarity⇔error link (A4, B2) are absent. A4 is arguably the single closest prior work to our thesis and reviewers may ask why it is missing.
3. **2024–2026 critiques of convergence are absent.** A paper claiming convergence in 2026 cannot ignore A6 (scale confounds) — and it has a concrete consequence for us: our cross-encoder CKA compares encoders of *similar* scale, and our Mantel results are permutation-tested, which partially pre-empts the confound, but a sentence is warranted. A5's "objective drives RSA consistency" is nearly a direct citation-level analogue of our DINOv2-vs-ResNet-50 contrast (H2 weakening).
4. **Behavioral-side boundary support.** A8 gives us a citable prior for the Discussion claim that shared geometry does not force shared downstream behavior (our Qwen2-VL/gemma4 supplement nulls and the family-conditioned component).
5. **Difficulty-prediction adjacency.** C1/C2 show an active 2026 line predicting difficulty from VLM content — one citation makes our "Predicting MLLM Performance" paragraph look current rather than self-referential (currently only our own two papers + LLaVA).
6. **Prompt-experiment context.** D1/D2 support §4.5's semantic-alignment interpretation with independent literature.

## Recommendation (priority-ordered)

Add to refs.bib + §2 (and one clause each where noted):

1. **A4 (Mania 2019)** — Visual Confusability paragraph; the "model similarity mitigates test set overuse" link is the nearest prior to shared difficulty. [HIGH]
2. **A6 (Aristotelian view 2026)** — Representation Similarity paragraph + one hedging clause in §4.3 CKA (scale confounds; our permutation tests pre-empt). [HIGH]
3. **A1 (Klabunde survey 2023)** — Representation Similarity paragraph opener. [HIGH]
4. **A5 (objective drives RSA 2024)** — H2 discussion in §4.3 (objective vs data). [HIGH]
5. **B1 (Geirhos error consistency 2021)** — Visual Confusability paragraph (metric foundation). [HIGH]
6. **A8 (convergence without understanding 2026)** — Discussion (§5.2 or geometry→behavior boundary). [MED]
7. **A2 (relative representations 2022)** — Representation Similarity. [MED]
8. **D1 (CLIP text decomposition 2023)** — §4.5 prompt section, one sentence. [MED]
9. **A3 (Nguyen 2020)** — optional; A1 survey likely subsumes. [LOW]
10. **B2, C1, D2, A7** — optional, cite only if space/flow permits (B2 for confusability, C1 for difficulty prediction, D2 for concept bias). [LOW]

Expected result: §2 grows from 12 to ~19-20 citations, every gap a reviewer would probe is closed, and no unverified reference is introduced (all above verified via arXiv API metadata).

## Limitations

- arXiv-only verification: venue names for 2026 items are "arXiv preprint" (none has a published venue listed); A1/A2/A4/B1/D1/D2 venues come from well-known published records but were not independently re-verified against publisher pages in this pass.
- Semantic Scholar (citation counts, venue metadata) was rate-limited; if needed, re-run later for venue/citation verification.
- Search coverage is recall-biased (6 queries); a systematic-review-grade sweep (PRISMA) was not requested and is not implied by this scan.
