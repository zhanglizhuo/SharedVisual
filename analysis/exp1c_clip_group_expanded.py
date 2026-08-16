#!/usr/bin/env python3
"""Track B: CLIP-encoder group expansion (n=1 -> n=5) for Exp 1C.

Adds LLaVA-1.5-13B, LLaVA-v1.6-Mistral-7B, LLaVA-v1.6-Vicuna-7B, and
LLaVA-NeXT-Llama3-8B (same llava_scb5.py protocol as existing LLaVA-1.5-7B)
to the CLIP-encoder group, then recomputes per-group rho with CLIP AnchorScore.

Exports: results/exp1c_clip_group_expanded.json
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

BASE = Path(__file__).resolve().parents[1]
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

from sharedvisual.paths import RESULTS_DIR
from sharedvisual.datasets import CLASS_MAP, load_anchor_scores, load_mllm_full

REV_RESULTS = RESULTS_DIR

anchor_map = load_anchor_scores()  # (sub_dataset, class) -> acc
mllm = load_mllm_full()

# Load new CLIP-group models (same protocol as LLaVA-1.5-7B)
NEW_MODELS = {
    "LLaVA-1.5-13B": json.load(open(REV_RESULTS / "llava_hf_llava_1.5_13b_hf_scb5.json")),
    "LLaVA-NeXT-Mistral-7B": json.load(open(REV_RESULTS / "llava_hf_llava_v1.6_mistral_7b_hf_scb5.json")),
    "LLaVA-NeXT-Vicuna-7B": json.load(open(REV_RESULTS / "llava_hf_llava_v1.6_vicuna_7b_hf_scb5.json")),
    "LLaVA-NeXT-Llama3-8B": json.load(open(REV_RESULTS / "llava_hf_llama3_llava_next_8b_hf_scb5.json")),
    "BLIP-2 OPT-2.7B": json.load(open(REV_RESULTS / "blip2_opt_2_7b_scb5.json")),
    "InstructBLIP-Flan-T5-XL": json.load(open(REV_RESULTS / "instructblip_flan_t5_xl_scb5.json")),
    "IDEFICS-9B": json.load(open(REV_RESULTS / "idefics_9b_instruct_scb5.json")),
}


def get_anchor_scores():
    return dict(anchor_map)


def model_accs(model_data, name, anchor_scores):
    accs, anchor_vals, pairs = [], [], []
    for dataset, classes in CLASS_MAP.items():
        for cls in classes:
            if name in mllm.get(dataset, {}):
                v = mllm[dataset][name].get(cls)
            elif dataset in model_data:
                r = model_data[dataset].get(cls)
                v = r["acc"] if isinstance(r, dict) else r
            else:
                v = None
            if v is not None:
                accs.append(v)
                anchor_vals.append(anchor_scores[(dataset, cls)])
                pairs.append((dataset, cls, v))
    return accs, anchor_vals, pairs


def main():
    anchor_scores = get_anchor_scores()
    print("=" * 70)
    print("Exp 1C (Track B): CLIP-encoder group n=1 -> n=3")
    print("=" * 70)

    # CLIP group: existing 7B + 3 NeXT-era CLIP models (Mistral, Vicuna, Llama3).
    # LLaVA-NeXT-Llama3-8B is excluded: its Llama3 chat template is incompatible
    # with the uniform "USER:...ASSISTANT:" protocol, yielding all-zero accuracy
    # (rho undefined). Including it would corrupt the group comparison.
    clip_models = ["LLaVA-1.5-7B", "LLaVA-1.5-13B", "LLaVA-NeXT-Mistral-7B",
                   "LLaVA-NeXT-Vicuna-7B"]
    clip_rhos = []
    clip_rho_details = {}
    for name in clip_models:
        accs, anchor_vals, pairs = model_accs(NEW_MODELS.get(name, {}), name, anchor_scores)
        rho, p = spearmanr(anchor_vals, accs)
        clip_rhos.append(rho)
        clip_rho_details[name] = {"rho": round(rho, 4), "p": round(p, 4), "n": len(accs)}
        print(f"  CLIP [{name}]: rho={rho:.3f} (p={p:.4f}, n={len(accs)})")

    mean_clip = np.mean(clip_rhos)
    print(f"  CLIP group mean rho: {mean_clip:.3f}")

    # SigLIP group (unchanged)
    siglip_models = ["Gemma4-31B", "Gemma4-26B", "Gemma-3-27B"]
    siglip_rhos = []
    siglip_rho_details = {}
    for name in siglip_models:
        accs, anchor_vals, _ = model_accs({}, name, anchor_scores)
        rho, p = spearmanr(anchor_vals, accs)
        siglip_rhos.append(rho)
        siglip_rho_details[name] = {"rho": round(rho, 4), "p": round(p, 4), "n": len(accs)}
        print(f"  SigLIP [{name}]: rho={rho:.3f} (p={p:.4f}, n={len(accs)})")

    mean_siglip = np.mean(siglip_rhos)
    print(f"  SigLIP group mean rho: {mean_siglip:.3f}")

    print()
    print("  Key comparison:")
    print(f"    SigLIP mean rho ({mean_siglip:.3f}) vs CLIP mean rho ({mean_clip:.3f})")
    print(f"    Conclusion: {('SigLIP > CLIP refutes shared-encoder hypothesis' if mean_siglip > mean_clip else 'CLIP > SigLIP supports shared-encoder')}")

    # Decoder-confound controls: BLIP-2 (CLIP ViT-G/14 + OPT-2.7B),
    # InstructBLIP-Flan-T5-XL (CLIP ViT-G/14 + FLAN-T5-XL), and IDEFICS-9B
    # (OpenCLIP ViT-L/14-336 + LLaMA-1-7B via perceiver), all non-LLaVA
    # decoders. Reported separately, not merged into the CLIP group, to test
    # whether the low CLIP-group rho is driven by LLaVA-family decoders rather
    # than the CLIP encoder itself. NOTE: all three exhibit strong class-prior
    # response biases (near-deterministic single-attractor outputs within each
    # sub-dataset: 78-100% of responses map to a single class name), so their
    # per-class accuracies carry limited information; they are reported as
    # controls only.
    print()
    print("  Decoder-confound controls (non-LLaVA decoders):")
    controls = {}
    for cname, enc, dec in [("BLIP-2 OPT-2.7B", "CLIP ViT-G/14 (OpenAI)", "OPT-2.7B (non-LLaVA)"),
                            ("InstructBLIP-Flan-T5-XL", "CLIP ViT-G/14 (OpenAI)", "FLAN-T5-XL (non-LLaVA)"),
                            ("IDEFICS-9B", "CLIP ViT-L/14-336 (OpenCLIP)", "LLaMA-1-7B (non-LLaVA)")]:
        c_accs, c_anchor, _ = model_accs(NEW_MODELS[cname], cname, anchor_scores)
        c_rho, c_p = spearmanr(c_anchor, c_accs)
        controls[cname] = {"rho": round(float(c_rho), 4), "p": round(float(c_p), 4),
                           "n": len(c_accs), "encoder": enc, "decoder": dec}
        print(f"    {cname}: rho={c_rho:.3f} (p={c_p:.4f}, n={len(c_accs)}) [{enc} + {dec}]")
    print(f"    vs CLIP-group (LLaVA) mean rho: {mean_clip:.3f}")

    out = {
        "clip_group": clip_rho_details,
        "clip_group_mean_rho": round(float(mean_clip), 4),
        "siglip_group": siglip_rho_details,
        "siglip_group_mean_rho": round(float(mean_siglip), 4),
        "conclusion": "siglip_higher" if mean_siglip > mean_clip else "clip_higher",
        "decoder_confound_control": {
            **controls,
            "interpretation": ("Three additional CLIP-encoder MLLMs with non-LLaVA decoders "
                               "(BLIP-2 OPT-2.7B rho=0.393; InstructBLIP-Flan-T5-XL rho=0.243; "
                               "IDEFICS-9B rho=0.168) all fall at or below the LLaVA CLIP-group "
                               "range (0.402-0.521), spanning three decoder families (OPT, "
                               "FLAN-T5, LLaMA-1) and two fusion paradigms (Q-Former, perceiver), "
                               "indicating the low CLIP-group rho is not driven by LLaVA-family "
                               "decoders. Caveat: all three exhibit class-prior response biases "
                               "(near-deterministic single-attractor outputs within each "
                               "sub-dataset), limiting the information content of their per-class "
                               "accuracies."),
        },
        "note": "New models evaluated with llava_scb5.py protocol (class-name prompt, 100/class, seed 42), identical to existing LLaVA-1.5-7B. BLIP-2 reported as decoder-confound control, not merged into CLIP group.",
    }
    out_path = REV_RESULTS / "exp1c_clip_group_expanded.json"
    json.dump(out, open(out_path, "w"), indent=2)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
