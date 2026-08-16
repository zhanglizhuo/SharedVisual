# Reviewer Panel — SharedVisual (2026-08-15)

**Paper**: Shared Class-Neighborhood Geometry: A Candidate Mechanism for Why CLIP Predicts MLLM Difficulty (27 pp, targeted at *Neural Networks*)
**Mode**: full review; 5 independent reviewers (EIC + R1 methodology + R2 domain + R3 perspective + Devil's Advocate); Phase 1 reports are independent (no cross-referencing).

---

## Report 1 — Editor-in-Chief

**Recommendation**: Major Revision

**Strengths.** This manuscript asks a genuinely interesting question — why does a frozen 2021 CLIP predict per-class difficulty of much larger MLLMs — and structures it as four falsifiable hypotheses (H1–H4), each with a dedicated control. The disclosure culture is exemplary: negative MLLM perturbation results are reported in full, the perturbation grid's multiplicity is FDR-handled (§4.8), the causal roadmap is concrete rather than hand-waving, and the "supportive rather than decisive" phrasing matches the evidence. For *Neural Networks*, a representation-learning paper with a well-specified empirical question, cross-encoder geometry measurements (CKA/Mantel across seven encoders), and a quantified behavioral consequence (mediation) is squarely in scope. The abstract's five-line organization mapping to §4.1–4.5 is unusually clean.

**Concerns.**
1. **Mechanism strength vs. framing.** The title correctly says "candidate mechanism", but the Introduction and Conclusion oscillate between "provides convergent evidence" and "observational; causal chain awaits validation". For an NN reader the honest middle is fine, but the paper should decide whether the geometry→difficulty link is presented as *the* account or *a* structural correlate. §4.3's "consistent with broadly shared class geometry" is the right register; the Conclusion's "provides convergent evidence for this account" is slightly ahead of it.
2. **Length and density.** 27 pages with a Discussion of five subsections plus a roadmap paragraph. The within/between-family decomposition (§4.2) is the conceptually heaviest passage in the paper and will lose non-expert readers; consider a one-sentence plain-language gloss when first introduced.
3. **Companion-paper dependency.** The manuscript leans on ma2025anchorscore/ma2025clipreliability for the frozen MLLM evaluations. The provenance is documented and the reproduction ρ=0.769 is reassuring, but the paper should be readable standalone — the Setup section handles this well; only minor polish remains.

**Journal-fit verdict**: in scope. **Priority**: address the two DA-MAJOR items (trivial-competence control; shared-protocol confound discussion) before acceptance; R1/R2/R3 items are minor.

---

## Report 2 — Reviewer 1 (Methodology)

**Recommendation**: Minor Revision

The statistical machinery is unusually careful for this literature. Specific comments:

1. **Small-n mediation (§4.4)** — The paper's own power analysis (33% power for |ρ|=0.45 at n=13) and the admission that SCB5 CIs span zero are exactly right. The Stanford40-powered test (6/6 indirect CIs exclude zero) carries the claim, and SCB5 is used only directionally (sign test p=0.020). I verified the reported numbers against the committed results (mediation_bootstrap.json): Stanford40 indirect effects 0.449–0.714 with CIs excluding zero — consistent. No change needed, but consider adding one sentence stating explicitly that the mediation is not a causal decomposition (already in §4.4 "association-based decomposition" — good).
2. **Multiplicity (§4.8)** — The 30-test perturbation grid with BH-FDR (8 raw → 0 surviving across the grid; static 6/6 survive) is handled better than most submissions. The paper draws the right conclusion (static agreement robust, perturbation exploratory).
3. **Permutation nulls** — Exact enumeration for 8-class Mantel (8!), 10⁴–10⁵ permutations elsewhere, class-block permutation for repeated measures. All appropriate. One point: the within/between-family exact test over 138,600 family-label assignments — the permutation unit is the family label; this is the right null for the claim, but please state it in one clause in the main text (currently only implied).
4. **Attenuation correction (§4.1)** — The binomial-noise correction (reliability 0.982 for anchor) and the precision-weighted check are appropriate; reporting the simulation mean ± SD (0.783 ± 0.038) is good practice.
5. **MINOR** — The S40 CI [0.624, 0.932] is wide; the jackknife (max |Δρ|=0.043) and 3/5-lowest-removal checks (0.768/0.738) now pre-empt the leverage concern. State the bootstrap is percentile-based (10k resamples) in the Reproducibility section for completeness.
6. **MINOR** — "one-sided exact probability 1/C(7,4)=0.029": the combinatorics are correct for complete separation, but the one-sided/two-sided framing in §4.2 should be unified with the later "smallest attainable two-sided value 2/C(8,4)=0.029" (the 7-model vs 8-model version). One clarifying sentence.

**Verdict**: no methodological error found; fixes are cosmetic.

---

## Report 3 — Reviewer 2 (Domain)

**Recommendation**: Minor Revision

1. **Literature coverage** — Now current. The survey (Klabunde et al.), error-consistency metric (Geirhos et al. ICLR 2021), model-similarity/shared-errors (Mania et al.), objective-driven RSA consistency (Ciernik et al.), the scale-confound critique (Gröger et al.), and the convergence-vs-reasoning decoupling (Usama et al.) close the gaps a 2026 reviewer would probe. The Visual Confusability paragraph now has its metric foundation, and the Representation Similarity paragraph engages the 2024–2026 critiques rather than only citing Huh et al. favorably.
2. **Encoder independence claims** — The seven-encoder panel (3 CLIP variants, SigLIP, QwenViT, ImageNet-only ResNet-50, DINOv2) is adequate. One factual check: the paper uses the Qwen2-VL vision encoder as the geometric proxy for the Qwen3.x MLLM group; the Setup text discloses this ("QwenViT denotes the Qwen2-VL vision encoder used to represent this family in the geometric analyses") — good, but add one sentence acknowledging that Qwen3.x models may use a newer encoder internally, so the geometric proxy is approximate for that family. This does not threaten the family-level behavioral result, which is measured on the actual models.
3. **Differentiation from prior work** — The paper is clearly distinct from the AnchorScore companion papers (diagnostic utility vs. mechanism). The SigLIP tension paragraph (§5.1) is one of the better pieces of scientific writing in the paper.
4. **MINOR** — The MLLM roster (Qwen3.5/3.6, Gemma4) are served via Ollama; a sentence stating model sizes/quantization of served checkpoints in the Reproducibility section would preempt a reviewer question about eval fidelity (temperature=0 greedy is stated).

**Verdict**: domain contribution is genuine; only disclosure polish remains.

---

## Report 4 — Reviewer 3 (Cross-disciplinary / Practical)

**Recommendation**: Minor Revision

1. **Cross-disciplinary resonance** — The paper's "shared confusability" framing now connects properly to the error-consistency tradition (Geirhos, Cohen's κ across decision makers) and to cross-species representational geometry (Kriegeskorte). The three-level distinction (geometry → confusability → difficulty) is a useful conceptual export for cognitive-model comparison work; a reader from computational neuroscience will want one sentence on whether class-level confusability in *humans* follows the same difficulty ranking — as future work, this is fine.
2. **Practical deployment cost (the real issue here)** — The Introduction states CLIP evaluation takes "3 minutes and $0.12", but AnchorScore requires **labeled validation images per class** (42–1,684 on SCB5), and labels are precisely the scarce resource in annotation workflows. The cost claim should be qualified in §5.4 "Practical Implications": the compute is 3 minutes, but the label requirement is N_c labeled images per class. This is the kind of over-claim a deployment-oriented reader will catch immediately.
3. **Domain specificity** — Flowers102 (fine-grained, natural) and Stanford40 (actions) support transfer; the perturbation result is concentrated in blur-sensitive classes. A practitioner should be told that the diagnostic is validated on recognition-style benchmarks and not yet on document/medical/remote-sensing domains (the paper's H2 future-work sentence partially covers this).
4. **Ethical note** — Classroom-behavior surveillance applications: the data is public, and the paper is diagnostic; a one-line responsible-use statement in the Limitations or Data Availability would be sufficient for a journal of this profile.

**Verdict**: practically valuable and appropriately scoped once the label-cost qualification lands.

---

## Report 5 — Devil's Advocate

**Strongest Counter-Argument** (≈250 words)

*The geometric agreement may be trivially induced by shared semantic competence rather than by shared class-neighborhood geometry.* All audited encoders are strong classifiers of the same eight semantic classes; two classifiers that distinguish the same categories will necessarily produce similar class-centroid configurations — centroids of different classes are pushed apart in every competent encoder, so the pairwise similarity structure is largely determined by the class identity structure itself, not by any fine-grained shared geometry. On this account the Mantel ρ=0.90–0.98 (and CKA 0.79–0.87) is what any pair of adequate models would achieve, and the paper's geometric evidence reduces to the observation that all encoders solve the same task. The paper has partial responses: (i) DINOv2 — trained on *more* data with corpus overlap — agrees *less* (0.66–0.87), which a pure competence account would not predict; (ii) the patch-level adapter transfers at most 21% of variance while centroids agree at 0.90–0.96, localizing the agreement. But the cleanest test has not been run and is nearly free: compare each encoder's class-similarity matrix against a *label-only reference* (e.g., the identity/one-hot structure or a random competent classifier's expected matrix). If encoder–encoder agreement exceeds encoder–reference agreement, the shared geometry carries information beyond competence. This control belongs in the paper.

**Issue List**

- **CRITICAL** — None that would invalidate a core conclusion *given* the existing partial defenses (DINOv2 ordering, patch-level localization). I record this explicitly; the trivial-competence alternative is the strongest remaining competitor to H3 and must be confronted head-on, but the paper's current evidence is not consistent with it being the whole story.
- **MAJOR-1** — The trivial-competence baseline control described above is missing (§4.3). Add the label-reference comparison or an equivalent falsifier.
- **MAJOR-2** — Shared evaluation-protocol confound is never discussed: all models are evaluated zero-shot with class-name-style prompts on the same benchmark vocabulary. A reviewer will ask whether the observed alignment is partly induced by the shared prompt/benchmark protocol. The paper has implicit defenses (family-conditioned component in §4.2, prompt-variant robustness in §4.5, and the fact that AnchorScore and MLLM protocols use different prompt forms), but they are scattered — one explicit paragraph would settle it.
- **MINOR-1** — The mediation mediator (leave-one-out cross-model mean) is constructed to be the shared signal; the mediation therefore quantifies the shared-difficulty component almost by construction, and its evidentiary role is confirmatory rather than diagnostic. The paper's own §4.4 caution already covers most of this; one sentence aligning it with MAJOR-2's protocol point would help.
- **MINOR-2** — "So what" for the practitioner does not depend on the mechanism (the diagnostic works either way). The paper should state plainly that the scientific contribution is the mechanism account; the practical contribution stands independently.

**Ignored Alternatives** — (a) shared task/competence (above); (b) shared protocol/vocabulary (above); (c) class-frequency (refuted empirically, §4.1 — good); (d) label noise/annotation artifacts shared across benchmark classes (not discussed; the SCB5 YOLO-derived crops and label mappings could carry systematic noise correlated with class difficulty).

**Observations (non-defects)** — The transparency of negative results (supplement floor/null) is above the norm of the venue; the FDR handling is better than most accepted papers; the causal roadmap is honest.

---

## Editorial Synthesis (Phase 2)

**Consensus**: All five reviewers rate the paper as strong; no reviewer identified a fatal flaw. R1 confirms methodology is clean (minor polish). R2 confirms literature coverage is now current. EIC flags framing consistency. R3 flags the label-cost qualification. DA records no CRITICAL, two MAJOR (both addressable with zero new training), four MINOR.

**Disagreements**: None substantive. EIC's framing concern vs DA's mechanism emphasis are the same issue from two angles (the paper should state the mechanism is the *scientific* contribution while the diagnostic stands independently — DA MINOR-2 resolves EIC's point 1).

**Editorial Decision: Major Revision** (borderline minor — MAJOR-1/MAJOR-2 are conceptual additions requiring ~2 paragraphs + 1 zero-cost analysis, no new training, no new models).

### Revision Roadmap (prioritized)

1. **[DA-MAJOR-1]** §4.3: add the label-only reference control (compare encoder class-similarity matrices vs. the one-hot label structure; report whether encoder–encoder agreement exceeds encoder–reference agreement) — ~1 hour of analysis, no training.
2. **[DA-MAJOR-2]** §4.2 or §5: one paragraph on the shared evaluation-protocol confound, consolidating the existing scattered defenses (family-conditioned component, prompt-variant robustness, differing prompt forms between AnchorScore and MLLM evals).
3. **[R3-1]** §5.4 Practical Implications: qualify "3 minutes / $0.12" with the labeled-validation-image requirement (N_c per class).
4. **[R2-1]** §3.2: one sentence that Qwen3.x may use a newer internal encoder than the Qwen2-VL geometric proxy.
5. **[R2-2]** Reproducibility: state served-checkpoint quantization for the Ollama roster.
6. **[R1-1]** Reproducibility: state bootstrap type (percentile, 10k).
7. **[R1-2]** §4.2: unify the one-sided/two-sided exact-probability phrasing (7-model 1/C(7,4) vs 8-model 2/C(8,4)).
8. **[EIC-1]** Conclusion: align "provides convergent evidence for this account" with "supportive structural correlate" register (one clause).
9. **[EIC-2 / R3-3]** Add a one-line responsible-use note (classroom surveillance data) in Limitations.
10. **[R3-2 / DA-MINOR-1]** Optional: one sentence each on benchmark label-noise as an untested alternative, and on the confirmatory role of the mediation mediator.

**Expected outcome after revision**: Minor Revision / Accept.

*(Phase 2.5 Revision Coaching offered; user may skip with "just fix it".)*
