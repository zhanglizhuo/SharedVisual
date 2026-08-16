# SharedVisual: Experimental Findings

## Core Question
Why does CLIP's per-class zero-shot accuracy predict MLLM per-class accuracy?

## Proposed Mechanism: Shared Class-Neighborhood Geometry

> Different Internet-scale vision encoders independently learn nearly identical **class-neighborhood geometry**, producing a shared **class-confusability structure**. This shared confusability determines zero-shot visual difficulty for any encoder, which is why CLIP AnchorScore predicts MLLM accuracy regardless of the MLLM's vision encoder.

**Causal chain:**
- Internet-scale training → Shared representation geometry
- → Shared class neighborhoods (which classes look similar)
- → Shared class confusability (which classes are hard to separate)
- → Shared visual difficulty → AnchorScore predicts MLLMs

## Evidence Chain

### Level 1: Observation — AnchorScore predicts MLLM accuracy

**SCB5 (13 pooled classes):** ρ=0.47–0.74 across 9 MLLMs (mean 0.64)
**Stanford40 (human actions, 40 classes):** ρ=0.817 aggregated (AnchorScore vs cross-model mean, 6 Ollama MLLMs, p<10⁻¹⁰, 95% CI: [0.624, 0.932]; mean of per-model ρ=0.67)

The phenomenon holds across domains (classroom → daily actions) and model families.

### Level 2: Rule-out — Not the shared encoder

**Finding 1: SigLIP ε > CLIP ρ (Exp 1C)**

| Group | Encoder | Mean ρ | n |
|-------|---------|--------|---|
| CLIP (LLaVA) | CLIP ViT-L/14 | 0.473 | 1 |
| SigLIP (Gemma) | SigLIP | **0.716** | 3 |
| QwenViT (Qwen3.x) | QwenViT | **0.669** | 4 |
| Qwen2-VL | QwenViT | 0.479 | 1 |

Models with DIFFERENT vision encoders show HIGHER correlation with CLIP AnchorScore.
→ The mechanism is NOT sharing the CLIP encoder.

**Finding 2: Partial ρ attenuated (Deep Analysis, leave-one-out covariate)**

Controlling for leave-one-out cross-model average per-class difficulty (target model excluded from the mean):
- LLaVA: raw ρ=0.473 → partial ρ=+0.445 (p=0.128)
- Gemma4-31B: raw ρ=0.725 → partial ρ=+0.258 (p=0.394)
- Gemma-3-27B: raw ρ=0.687 → partial ρ=+0.214 (p=0.482)
- Qwen3.5-35B: raw ρ=0.736 → partial ρ=+0.055 (p=0.859)
- Qwen2-VL-7B: raw ρ=0.479 → partial ρ=-0.181 (p=0.553)
- All models: p>0.1, all bootstrap 95% CIs contain zero

→ AnchorScore's correlation is largely (though not provably completely) explained by shared class-level difficulty.

**Finding 2b: Replication on Stanford40 (n=40, 6 MLLMs, `analysis/stanford40_partial_correlation.py`)**

The SCB5 mediation uses only 13 classes (wide bootstrap CIs). On Stanford40 (40 classes), the result is sharper:

| Model | raw ρ | partial ρ | 95% CI |
|-------|-------|-----------|--------|
| qwen3.5:27b | 0.580 | -0.177 | [-0.619, 0.407] |
| qwen3.5:35b-a3b | 0.639 | -0.132 | [-0.495, 0.281] |
| qwen3.6:27b | 0.723 | 0.319 | [-0.348, 0.699] |
| qwen3.6:35b-a3b | 0.723 | 0.459 | [0.087, 0.738] |
| gemma4:26b | 0.678 | 0.291 | [-0.151, 0.606] |
| gemma4:31b | 0.662 | 0.182 | [-0.238, 0.603] |

Mean raw ρ=**0.667** → mean partial ρ=**0.157**; **five of six CIs contain zero** (exception: qwen3.6:35b-a3b, CI [0.087, 0.738], marginally above zero). Substantial, though not complete, mediation at 2.5× the sample size (n=40 vs n=13). Note: with the earlier circular covariate (target model included in the mean), mean partial ρ was 0.002; the leave-one-out covariate removes this artifact.

### Level 3: Representation Similarity

**Finding 3: CKA across encoder families (Exp 2B)**

| Pair | CKA | RSA |
|------|-----|-----|
| CLIP L/14 vs CLIP L/14 (LAION) | **0.907** | **0.976** |
| CLIP L/14 vs CLIP B/32 | **0.874** | **0.976** |
| **CLIP L/14 vs Qwen2-VL Vision** | **0.792** | **0.810** |
| **CLIP L/14 (LAION) vs Qwen2-VL Vision** | **0.809** | **0.786** |
| **CLIP B/32 vs Qwen2-VL Vision** | **0.820** | **0.667** |

Different architectures (ViT vs QwenViT), training paradigms (CLIP vs language-model-pretrained), and data sources converge to similar feature spaces (CKA≈0.8) and per-class difficulty structure (RSA≈0.7–0.8).

### Level 4: Shared Class Geometry — THE NEW EVIDENCE

**Finding 4: Class Similarity Matrices are nearly identical across encoders**

For each encoder: compute class centroid → 8×8 cosine similarity matrix → compare upper triangles.

| Pair | ρ (class geometry) | p |
|------|-------------------|---|
| CLIP L/14 vs CLIP L/14 (LAION) | **0.982** | <0.0001 |
| CLIP L/14 vs CLIP B/32 | **0.955** | <0.0001 |
| **CLIP L/14 vs Qwen2-VL Vision** | **0.897** | <0.0001 |
| **CLIP L/14 (LAION) vs Qwen2-VL Vision** | **0.913** | <0.0001 |
| **CLIP B/32 vs Qwen2-VL Vision** | **0.949** | <0.0001 |

All p < 0.0001. Cross-encoder ρ=0.90–0.95 for the ENTIRE class-neighborhood structure (50 images/class, ~390 total).

**Mantel test (exact permutation, `analysis/mantel_test.py`):**

Because SCB5 has only 8 classes (28 pairs), asymptotic p-values are unreliable. We use an exact Mantel test enumerating all 8!=40,320 class relabelings. For every encoder pair, **no random relabeling produced agreement as high as the observed value** (all exact p < 5×10⁻⁵, the minimum possible):

| Pair | Mantel ρ | exact perm p |
|------|----------|--------------|
| CLIP L/14 vs CLIP L/14 (LAION) | **0.982** | <5×10⁻⁵ |
| CLIP L/14 vs CLIP B/32 | **0.955** | <5×10⁻⁵ |
| **CLIP L/14 vs Qwen2-VL Vision** | **0.897** | <5×10⁻⁵ |
| **CLIP L/14 (LAION) vs Qwen2-VL Vision** | **0.913** | <5×10⁻⁵ |
| **CLIP B/32 vs Qwen2-VL Vision** | **0.949** | <5×10⁻⁵ |

That is, **not just are representations broadly similar (CKA), but the specific pattern of which classes are close to which other classes is preserved across encoder families.**

This directly explains shared confusability: if two classes are close in one encoder's space, they are also close in another encoder's space → both encoders find them hard to separate → both have similar per-class accuracy.

**Finding 4b: Cross-dataset validation on Stanford40 (Exp 6, 40 classes, 9,532 images)**

Real feature-based 40×40 class-similarity matrices (per-class centroid cosine sim), computed from independently trained encoders on an unrelated domain (human actions):

| Pair | ρ (class geometry) | p | n_pairs |
|------|-------------------|---|---------|
| **CLIP ViT-L/14 vs Qwen2-VL Vision** | **0.810** | 1.6×10⁻¹⁸² | 780 |

Reproduced via `experiments/exp6_stanford40_geometry.py` (feature caches in `exp6_{clip,qwen}_features.npz`). Confirms the shared geometry is not a small-$n$ artifact of SCB5's 8 classes and generalizes across datasets and encoder families.

### Level 5: Understanding CLIP's failures

**Finding 5: CLIP failures are semantic, not visual (Exp 3)**

| Class | Original | Best Prompt | Δ |
|-------|----------|-------------|---|
| blackboard-writing | 45% | "a teacher writing on the board" → **98%** | +53pp |
| answer | 0% | "a student raising hand to answer" → **100%** | +100pp |
| blackBoard | 2% | "a teacher at the blackboard" → **69%** | +67pp |

Prompt variants resolve CLIP's failure on composition classes. The vision encoder CAN see these concepts; the failure is in text-image alignment.

→ Explains why CLIP's zero-shot has noise, but the underlying visual difficulty still correlates with MLLM performance.

### Supplementary

**Blur perturbation (Exp 4):** σ=0/1/2/4/8 on CLIP → teacher robust 84→67%, blackboard-writing 45→2%. Informative only for high-accuracy classes.

**MLLM experiments (Qwen2-VL-7B):** Floor effects prevent analysis — the 8-class teacher behavior classification is too difficult for 7B MLLMs. LLaVA-1.5-7B perplexity-based classification also fails (all class losses nearly identical).

### Finding 7: MLLM Prompt Sensitivity (Exp 3 - Qwen2-VL-7B)

Qwen2-VL-7B shows near-random classification accuracy on this task (~15.6%, vs 12.5% random):

| Condition | Overall | teacher | guide | other classes |
|-----------|---------|---------|-------|---------------|
| Bare class names | 15.6% | 95% | 20% | 0-5% |
| Enriched descriptions | 15.6% | 100% | 20% | 0-5% |

**Key observations:**
- Only "teacher" is classified accurately (95-100%)
- Enriched descriptions do NOT improve accuracy for any class (unlike CLIP, where prompts boosted 0→100%)
- MCQ format causes complete failure (0/20), confirming Qwen2-VL-7B does not follow classification instructions well
- Floor effects limit interpretability — the task is too hard for this MLLM

**Conclusion: MLLMs (Qwen2-VL-7B) do NOT benefit from better class descriptions. This contrasts sharply with CLIP, suggesting different failure modes: CLIP fails from semantic misalignment, while MLLMs fail from task difficulty.**

### Finding 8: MLLM Blur Degradation (Exp 4B - Qwen2-VL-7B)

All blur levels show near-chance accuracy on Qwen2-VL-7B:

| σ | Accuracy |
|---|----------|
| 0 | 13.1% |
| 1 | 13.1% |
| 2 | 12.5% |
| 4 | 12.5% |
| 8 | 11.9% |

**Inconclusive due to floor effects** — the MLLM's low baseline accuracy (~13%) makes perturbation analysis impossible on this task.

---

## REFINED HYPOTHESIS

**The Universal Visual Difficulty Hypothesis:**

> All vision encoders trained on Internet-scale data learn essentially the same per-class visual difficulty structure. This structure determines how easily each class can be distinguished from others based on visual features alone. CLIP AnchorScore captures this universal visual difficulty, which is why it predicts MLLM accuracy — regardless of whether the MLLM uses CLIP, SigLIP, or a proprietary vision encoder.

**Why LLaVA-1.5-7B has lower ρ:** It's a weaker model with unique failure modes, so its accuracy pattern deviates from the shared visual difficulty signal.

**Why Cross-model ρ largely disappears with partial correlation:** The shared difficulty signal is the common factor. Once you control for it (leave-one-out), residual CLIP-specific predictive power is small and mostly indistinguishable from zero (SCB5: all CIs contain 0; Stanford40: 5/6 CIs contain 0).

**Why prompt variants help CLIP:** The difficulty is in semantic alignment (mapping class name → visual concept), not in visual processing. Both CLIP and MLLMs share this alignment challenge.

---

## Remaining Questions

1. ~~**MLLM prompt sensitivity**: Would MLLMs also improve with better class descriptions?~~ → **No (at least for Qwen2-VL-7B)**. Floor effects prevent strong conclusions; a different MLLM or dataset might yield different results.
2. ~~**Perturbation synchrony (MLLM)**~~ → **Inconclusive due to floor effects**. Qwen2-VL-7B cannot classify these images above chance.
3. ~~**CKA with MLLM visual features**~~ → **Done**. CKA = 0.79–0.82 between CLIP variants and Qwen2-VL vision encoder.
4. **Other MLLMs**: Would LLaVA-1.5-7B or Gemma-3-4B show similar patterns? LLaVA-1.5 uses CLIP encoder (may show prompt sensitivity), Gemma uses SigLIP (may be better at this task).
