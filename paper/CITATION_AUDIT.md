# Citation Audit Report

**Date**: 2026-08-15
**Bib file(s)**: refs.bib
**Citing files**: SharedVisual.tex, supplemental.tex
**Auditor**: local executor, no external reviewer tools (per user instruction; deviation from the cross-model rule recorded)
**Verification channels**: arXiv API (export.arxiv.org), DBLP API — existence/metadata lookups only

## Summary

| Verdict | Count |
|---------|-------|
| KEEP    | 29 |
| FIX     | 1  |
| REPLACE | 0  |
| REMOVE  | 1  |

**Overall verdict: FAIL** (reason: `hallucinated` — one bib entry is a fabricated mashup).

## Critical Findings

### REMOVE: maini2023tmars (fabricated mashup)

Bib entry claims:
- title: "Improving CLIP training with data rewards"
- authors: "Maini, Pratyush and Girdhar, Rishabh and Lu, David and Aneja, Yutong and Segal, Shreyas and Lim, Serena N."

Verification result: **no such paper exists**. arXiv title search for "Improving CLIP training with data rewards" → empty; DBLP "Data Rewards CLIP" → empty; arXiv author search for Pratyush Maini returns exactly one CLIP-related paper:

> Maini, P., Goyal, S., Lipton, Z. C., Kolter, J. Z., & Raghunathan, A. (2023/ICLR 2024). *T-MARS: Improving Visual Representations by Circumventing Text Feature Learning*. arXiv:2307.03132.

That real paper is about **circumventing text feature learning**, not about data curation — the title and author list in our bib are a mashup of unrelated fragments (classic vibe-citation pattern).

Cited in: SharedVisual.tex §5.5 "For vision representation learning" (data-curation sentence).
ACTION: remove the entry from refs.bib; remove the inline cite; the sentence remains supported by gadre2023datacomp alone.

### FIX: gadre2023datacomp (author typo)

- Bib says "Hayashi, Jonathan" — canonical record (arXiv:2304.14108, NeurIPS 2023) says **Hayase, Jonathan**.
- Entry lacks the arXiv identifier used to verify it.
ACTION: correct the author name; add `eprint = {2304.14108}`.

### Context tightening (§5.5, same sentence)

"...filter noisy web-scale image--text corpora by per-class or per-caption quality estimates"
- DataComp filters by caption quality / image-text alignment; "per-class" is not a DataComp filtering signal and is our own framing. Tighten to "by caption-quality estimates" to keep the citation context clean.

## Uncited Entries (detected, advisory)

9 bib entries have no `\cite` in either tex file:

| Key | Status | Recommendation |
|-----|--------|----------------|
| klabunde2023similarity, moschella2022relative, mania2019model, geirhos2020error, gandelsman2023interpreting, ciernik2024objective, groger2026aristotelian, usama2026convergence | Metadata verified against arXiv API (all 8 correct) | **Add to §2** — these were approved by the user ("HIGH+MED 全加") in the related-work scan but the in-text citations were never inserted (task interrupted). |
| chen2018sensitivity | "Sensitivity analysis for information criteria in nonlinear structural equation models" (statistics paper, unrelated) | **Prune** — unrelated leftover. |

## All-Clean Entries (29 KEEP)

All other cited entries were audited on 2026-08-10 (cross-model, full three-axis) with all FIXes applied at that time; no changes to those entries or their citation contexts since, except:
- §5.5 data-curation sentence (new; findings above)
- Two §4.5/§4.8 sentences citing only pre-existing keys (verified contexts unchanged).

## Verification Trace (this run)

- arXiv API `id_list` lookups: 1905.12580, 2209.15430, 2310.05916, 2010.15327 (✓); 2304.08118, 2306.14101 (both mismatched my memory — correct IDs found by title search instead)
- arXiv API title searches: "Similarity of Neural Network Models" → 2305.06329 ✓; "DataComp" → 2304.14108 ✓ (author typo found); "Overcoming common flaws..." → 2407.01032 ✓; "T-MARS"/"data rewards"/au:"Pratyush Maini" → only 2307.03132 exists (mashup confirmed)
- Local: bib/cite key set diffs; context read of every new cite site

## Limitations

- Executor-side audit (no cross-model reviewer per user instruction); layers 1–2 (existence/metadata) rely on arXiv/DBLP API records, which do not confirm final published venues for 2026 preprints.
- Context layer reviewed by the executor for the 2 changed sites only; 29 pre-existing sites carry over the 2026-08-10 cross-model verdicts.
