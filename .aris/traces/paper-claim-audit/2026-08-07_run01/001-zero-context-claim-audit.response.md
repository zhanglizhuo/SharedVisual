# Reviewer response (see PAPER_CLAIM_AUDIT.md for the finalized report)

Full claim-by-claim table (57 claims) was produced by the fresh zero-context
reviewer thread (id ses_0268ffcd5ffeDj1iErbhem16Vr). Key findings verbatim:

- C1/C3/C4/C5: headline rho 0.769/0.797, range 0.374-0.736, mean 0.630 — exact_match
- C8-C12: group means SigLIP 0.716, QwenViT 0.669, CLIP 0.462, per-model rhos — exact_match
- C13/C14: Kruskal H=8.93 p=0.030; perm p=0.029 = 2/C(8,4) — rounding_ok
- C30: linear/MLP/board R2 — exact_match
- C32/C33: mediation all 15 rows — exact match
- C42 (FAIL): "bottom-three under both CLIP variants" — scope_overclaim; stand=9.6 between answer and blackboard under LAION; OpenAI bottom three are screen/stand/answer
- C44: appendix "rank-z 0.26 / OLS 0.33" — raw is 0.276/0.322 — number_mismatch
- C19: fig3 caption "SigLIP 0.84-0.86 with all other encoders" — DINOv2 0.824, ResNet 0.869 — scope_overclaim
- C27: "ResNet-50 highest in entire panel" — within-CLIP 0.9815 > 0.9797 — number_mismatch
- C55: reproducibility "5-encoder 10 pairs" — mantel file has 21 pairs — provenance staleness

Overall verdict from reviewer: WARN.
