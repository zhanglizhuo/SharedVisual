"""Within- vs between-encoder-family variation in class-difficulty rankings.

Question (reviewer-facing): if shared class difficulty is primarily visual,
models sharing an encoder family should show similar per-class accuracy
rankings despite different language models; if difficulty were family-
specific, within-family agreement should exceed between-family agreement.

Design (deliberately descriptive, not a formal variance decomposition:
group sizes are 4/3/4/1 on SCB5):
  1. Pairwise Spearman rho between per-class accuracy vectors for all model
     pairs; partition into within-family and between-family pairs.
  2. Exact enumeration of family-label assignments where tractable
     (SCB5: 12!/(4!3!4!1!) = 138,600; Stanford40: C(6,4) = 15), else
     10,000 random permutations (seed 42). Two-sided p on the
     within-minus-between mean difference.
  3. Bootstrap over classes (2,000 resamples, seed 42) for a 95% CI of the
     same difference.

Naming discipline: "within-family model variation", NOT "LM effect" ---
models within a family differ in more than the decoder (projector,
resolution, instruction tuning), so the analysis cannot isolate the LM.

Inputs (all committed locally):
  - data/visualanchor_snapshot/01_core/paper_data/mllm_full.json (9 core MLLMs)
  - results/llava_hf_{llava_1.5_13b,llava_v1.6_mistral_7b,llava_v1.6_vicuna_7b}_hf_scb5.json (3 HF MLLMs)
  - data/visualanchor_snapshot/02_robustness/stanford40/ollama_results.json (6 MLLMs, 40 classes)

Output: results/family_variation.json
"""
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

BASE = Path(__file__).resolve().parents[1]
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

from sharedvisual.datasets import CLASS_MAP, load_mllm_full
from sharedvisual.paths import RESULTS_DIR

SEED = 42
N_BOOT = 2000

SCB5_GROUPS = {
    "CLIP": ["LLaVA-1.5-7B", "LLaVA-1.5-13B", "LLaVA-NeXT-Mistral-7B",
             "LLaVA-NeXT-Vicuna-7B"],
    "SigLIP": ["Gemma4-31B", "Gemma4-26B", "Gemma-3-27B"],
    "QwenViT": ["Qwen3.5-27B", "Qwen3.6-27B", "Qwen3.5-35B", "Qwen3.6-35B"],
    "Qwen2-VL-ViT": ["Qwen2-VL-7B"],
}

HF_FILES = [
    ("llava_hf_llava_1.5_13b_hf_scb5.json", "LLaVA-1.5-13B"),
    ("llava_hf_llava_v1.6_mistral_7b_hf_scb5.json", "LLaVA-NeXT-Mistral-7B"),
    ("llava_hf_llava_v1.6_vicuna_7b_hf_scb5.json", "LLaVA-NeXT-Vicuna-7B"),
]

STANFORD40_GROUPS = {
    "QwenViT": ["qwen3.5:27b", "qwen3.5:35b-a3b", "qwen3.6:27b", "qwen3.6:35b-a3b"],
    "SigLIP": ["gemma4:26b", "gemma4:31b"],
}


def load_scb5_matrix():
    """12 models x 13 classes per-class accuracy matrix (SCB5)."""
    mllm_full = load_mllm_full()
    for jf, mn in HF_FILES:
        d = json.load(open(RESULTS_DIR / jf))
        row = {}
        for _, per_class in d.items():
            if not isinstance(per_class, dict):
                continue
            for cls, v in per_class.items():
                if isinstance(v, dict):
                    v = v.get("acc", 0)
                row[cls] = float(v)
        mllm_full.setdefault("_hf_extra", {})[mn] = row

    models = sorted(set(sum(SCB5_GROUPS.values(), [])))
    n_models = len(models)
    classes = [cls for _, cls_list in CLASS_MAP.items() for cls in cls_list]
    M = np.full((n_models, len(classes)), np.nan)
    hf = mllm_full.get("_hf_extra", {})
    for i, m in enumerate(models):
        for j, cls in enumerate(classes):
            if m in hf:
                M[i, j] = hf[m].get(cls, np.nan)
            else:
                # find the sub-dataset containing this class
                for ds, cls_list in CLASS_MAP.items():
                    if cls in cls_list and m in mllm_full.get(ds, {}):
                        M[i, j] = mllm_full[ds][m].get(cls, np.nan)
    assert not np.isnan(M).any(), "missing per-class values in SCB5 matrix"
    return models, classes, M


def load_stanford40_matrix():
    d = json.load(open(BASE / "data/visualanchor_snapshot/02_robustness/stanford40/ollama_results.json"))
    models = sorted(set(sum(STANFORD40_GROUPS.values(), [])))
    classes = sorted(next(iter(d["models"].values()))["per_class_acc"].keys())
    M = np.full((len(models), len(classes)), np.nan)
    for i, m in enumerate(models):
        pc = d["models"][m]["per_class_acc"]
        for j, cls in enumerate(classes):
            v = pc[cls]
            if isinstance(v, dict):
                v = v.get("acc", 0)
            M[i, j] = float(v)
    assert not np.isnan(M).any(), "missing per-class values in Stanford40 matrix"
    return models, classes, M


def pairwise_rhos(M):
    """All pairwise Spearman correlations between rows of M."""
    n = M.shape[0]
    out = {}
    for i, j in combinations(range(n), 2):
        out[(i, j)] = spearmanr(M[i], M[j])[0]
    return out


def partition(pairs_rhos, family_of, n):
    within, between = [], []
    for (i, j), r in pairs_rhos.items():
        (within if family_of[i] == family_of[j] else between).append(r)
    return np.array(within), np.array(between)


def group_sizes(family_of):
    sizes = {}
    for f in family_of.values():
        sizes[f] = sizes.get(f, 0) + 1
    return sizes


def enumerate_partitions(n, sizes):
    """Yield all assignments of n items to labeled groups with given sizes.

    Nested combinations: C(n, k1) * C(n-k1, k2) * ...  Multiset-permutation
    count without duplicates. SCB5: 495*56*5 = 138,600; Stanford40: 15.
    """
    fams = list(sizes.keys())
    ks = [sizes[f] for f in fams]

    def rec(remaining, level, assign):
        if level == len(fams):
            yield list(assign)
            return
        k = ks[level]
        for combo in combinations(sorted(remaining), k):
            nxt = [x for x in remaining if x not in combo]
            yield from rec(nxt, level + 1, assign + [(x, fams[level]) for x in combo])

    yield from rec(list(range(n)), 0, [])


def analyze(name, models, classes, M, groups, exact=True, n_perm=10000):
    family_of = {}
    for fam, ms in groups.items():
        for m in ms:
            family_of[models.index(m)] = fam
    fam_arr = np.array([family_of[i] for i in range(len(models))])
    sizes = group_sizes(family_of)

    pairs_rhos = pairwise_rhos(M)
    within, between = partition(pairs_rhos, family_of, len(models))
    obs = within.mean() - between.mean()

    # per-family within-family means
    per_family = {}
    for fam in groups:
        idx = [i for i in range(len(models)) if family_of[i] == fam]
        if len(idx) >= 2:
            rr = [pairs_rhos[(min(a, b), max(a, b))]
                  for a, b in combinations(idx, 2)]
            per_family[fam] = {"n_pairs": len(rr),
                               "mean_rho": float(np.mean(rr)),
                               "min": float(np.min(rr)),
                               "max": float(np.max(rr))}

    # permutation / enumeration on the within-minus-between difference
    all_idx = list(range(len(models)))
    if exact:
        n_assign = 1
        from math import factorial
        denom = 1
        for k in sizes.values():
            denom *= factorial(k)
        n_assign = factorial(len(models)) // denom
        diffs = np.empty(n_assign)
        c = 0
        for assign in enumerate_partitions(len(models), sizes):
            fo = dict(assign)
            w, b = partition(pairs_rhos, fo, len(models))
            diffs[c] = w.mean() - b.mean()
            c += 1
        mode = f"exact enumeration ({n_assign} assignments)"
    else:
        rng = np.random.RandomState(SEED)
        diffs = np.empty(n_perm)
        pool = sum(([f] * k for f, k in sizes.items()), [])
        for c in range(n_perm):
            perm = rng.permutation(pool)
            fo = {i: perm[i] for i in all_idx}
            w, b = partition(pairs_rhos, fo, len(models))
            diffs[c] = w.mean() - b.mean()
        mode = f"{n_perm} random permutations (seed {SEED})"
    p_two = float(np.mean(np.abs(diffs) >= abs(obs) - 1e-12))

    # bootstrap over classes for CI of the difference
    rng = np.random.RandomState(SEED)
    boot = np.empty(N_BOOT)
    n_cls = M.shape[1]
    for c in range(N_BOOT):
        cols = rng.randint(0, n_cls, n_cls)
        Mb = M[:, cols]
        pr = pairwise_rhos(Mb)
        w, b = partition(pr, family_of, len(models))
        boot[c] = w.mean() - b.mean()
    ci = [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]

    return {
        "dataset": name,
        "n_models": len(models),
        "n_classes": len(classes),
        "groups": {f: len(m) for f, m in groups.items()},
        "within": {"n_pairs": len(within), "mean_rho": float(np.mean(within)),
                   "min": float(np.min(within)), "max": float(np.max(within))},
        "between": {"n_pairs": len(between), "mean_rho": float(np.mean(between)),
                    "min": float(np.min(between)), "max": float(np.max(between))},
        "per_family_within": per_family,
        "difference": {"within_minus_between": float(obs),
                       "ci95_bootstrap_classes": ci,
                       "p_two_sided": p_two,
                       "test_mode": mode},
        "pairwise_rhos": {f"{models[i]} | {models[j]}": float(r)
                          for (i, j), r in sorted(pairs_rhos.items())},
    }


def decoder_control_pairs():
    """Pairwise rank agreement among the three non-LLaVA CLIP-encoder controls.

    These models exhibit strong class-prior collapse (near-deterministic
    single-attractor outputs), so their per-class accuracies carry limited
    information; reported as a supporting observation only.
    """
    files = {
        "BLIP-2 OPT-2.7B": "blip2_opt_2_7b_scb5.json",
        "InstructBLIP-Flan-T5-XL": "instructblip_flan_t5_xl_scb5.json",
        "IDEFICS-9B": "idefics_9b_instruct_scb5.json",
    }
    classes = [cls for _, cl in CLASS_MAP.items() for cls in cl]
    vecs = {}
    for m, f in files.items():
        d = json.load(open(RESULTS_DIR / f))
        vecs[m] = [float(d[ds][cls]["acc"]) for ds, cl in CLASS_MAP.items() for cls in cl]
    out = {}
    for (a, va), (b, vb) in combinations(vecs.items(), 2):
        out[f"{a} | {b}"] = float(spearmanr(va, vb)[0])
    return {"pairwise_rhos": out,
            "mean_rho": float(np.mean(list(out.values()))),
            "caveat": ("all three models show class-prior collapse "
                       "(near-deterministic single-attractor outputs), "
                       "limiting the information content of their "
                       "per-class accuracies")}


def main():
    out = {}
    models, classes, M = load_scb5_matrix()
    out["scb5"] = analyze("SCB5 (13 classes, 12 MLLMs, 4 families)",
                          models, classes, M, SCB5_GROUPS, exact=True)
    models2, classes2, M2 = load_stanford40_matrix()
    out["stanford40"] = analyze("Stanford40 (40 classes, 6 MLLMs, 2 families)",
                                models2, classes2, M2, STANFORD40_GROUPS, exact=True)
    out["decoder_control"] = decoder_control_pairs()

    out["note"] = (
        "Within- vs between-encoder-family agreement of per-class accuracy "
        "rankings (pairwise Spearman). 'Within-family model variation', not "
        "'LM effect': models within a family differ in decoder, projector, "
        "resolution, and instruction tuning. SCB5 exact enumeration "
        "12!/(4!3!4!1!)=138600 label assignments; Stanford40 C(6,4)=15 "
        "(p resolution 1/15). Bootstrap: 2000 class resamples, seed 42. "
        "Data: mllm_full.json + 3 LLaVA HF evals (SCB5); "
        "stanford40 ollama_results.json."
    )
    path = RESULTS_DIR / "family_variation.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"Saved -> {path}")
    for k in ("scb5", "stanford40"):
        r = out[k]
        print(f"\n== {r['dataset']} ==")
        print(f"  within  ({r['within']['n_pairs']:2d} pairs): mean rho = {r['within']['mean_rho']:.3f}  [{r['within']['min']:.3f}, {r['within']['max']:.3f}]")
        print(f"  between ({r['between']['n_pairs']:2d} pairs): mean rho = {r['between']['mean_rho']:.3f}  [{r['between']['min']:.3f}, {r['between']['max']:.3f}]")
        print(f"  difference = {r['difference']['within_minus_between']:+.3f}, CI95 [{r['difference']['ci95_bootstrap_classes'][0]:+.3f}, {r['difference']['ci95_bootstrap_classes'][1]:+.3f}], p = {r['difference']['p_two_sided']:.4f} ({r['difference']['test_mode']})")
        for fam, v in r["per_family_within"].items():
            print(f"    within {fam}: {v['n_pairs']} pairs, mean {v['mean_rho']:.3f}  [{v['min']:.3f}, {v['max']:.3f}]")
    dc = out["decoder_control"]
    print(f"\n== decoder-control (3 non-LLaVA CLIP-encoder models) ==")
    for k, v in dc["pairwise_rhos"].items():
        print(f"  {k}: {v:.3f}")
    print(f"  mean = {dc['mean_rho']:.3f} (vs LLaVA within-family 0.788)")


if __name__ == "__main__":
    main()
