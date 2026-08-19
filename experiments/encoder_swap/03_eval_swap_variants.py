#!/usr/bin/env python3
"""Encoder-swap stage 3: evaluate swap variants on SCB5 val (full-frame, greedy).

Per arm: load trained projector + frozen Mistral-7B, classify each val image
(class-name prompt, greedy, first-match scoring identical to eval_hf.py),
compute per-class accuracy. Resumable: per-(arm, class) checkpoints of raw
responses. NOT YET RUN — validate with --dry-run.
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np

REPO_ROOT = Path(os.environ.get(
    "SHAREDVISUAL_ROOT", str(Path(__file__).resolve().parents[2])))
RESULTS = Path(os.environ.get("SHAREDVISUAL_RESULTS", str(REPO_ROOT / "results")))
FEAT_DIR = RESULTS / "encoder_swap" / "features"
CKPT_DIR = RESULTS / "encoder_swap" / "ckpt"
RESP_DIR = RESULTS / "encoder_swap" / "responses"

from pathlib import Path as _P
import importlib.util as _iu
_spec = _iu.spec_from_file_location(
    "eval_hf", REPO_ROOT / "experiments" / "eval_hf.py")
_eval = _iu.module_from_spec(_spec)
try:
    _spec.loader.exec_module(_eval)
    match_class_name = _eval.match_class_name
except Exception:
    def match_class_name(response, true_class, all_classes):
        """Fallback copy of eval_hf.py's first-match scorer (protocol-identical)."""
        resp = response.lower().strip()
        for c in all_classes:
            if c.lower().strip() in resp:
                return c == true_class
        return False

LM_ID = "mistralai/Mistral-7B-v0.1"
INSTRUCTION = ("Classify the person's behavior in this classroom image into "
               "one of: {classes}. Answer with the class name only.")
CLASSES = {
    "TeacherBehavior": ["guide", "answer", "On-stage interaction",
                        "blackboard-writing", "teacher", "stand", "screen",
                        "blackBoard"],
    "HandriseReadWrite": ["hand-raising", "read", "write"],
    "BowTurnHead": ["BowHead", "TurnHead"],
}
GEN_KW = dict(max_new_tokens=16, do_sample=False)


def evaluate_arm(arm, device="cuda", dry_run=False):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    resp_path = RESP_DIR / f"{arm}_val_responses.json"
    resp_path.parent.mkdir(parents=True, exist_ok=True)
    out = json.load(open(resp_path)) if resp_path.exists() else {}

    feat_path = FEAT_DIR / f"{arm}_val.npz"
    if dry_run:
        n_expected = 5416
        print(f"[dry-run] {arm}: would evaluate val images "
              f"(features {'present' if feat_path.exists() else 'NOT extracted yet'}); "
              f"{len(out)} cached responses")
        return {"cached": len(out)}

    data = np.load(feat_path, allow_pickle=True)
    feats, labels, img_ids = data["features"], list(data["labels"]), list(data["img_ids"])

    ckpt = torch.load(CKPT_DIR / arm / "latest.pt", map_location=device)
    D_i = feats.shape[2]
    from torch import nn
    LM_HIDDEN = 4096
    proj = nn.Sequential(
        nn.Linear(D_i, (D_i + LM_HIDDEN) // 2), nn.GELU(),
        nn.Linear((D_i + LM_HIDDEN) // 2, LM_HIDDEN)).to(device)
    proj.load_state_dict(ckpt["proj"])
    proj = proj.eval().to(torch.float16)

    tok = AutoTokenizer.from_pretrained(LM_ID)
    tok.pad_token = tok.pad_token or tok.eos_token
    lm = AutoModelForCausalLM.from_pretrained(
        LM_ID, torch_dtype=torch.float16).to(device).eval()
    all_classes = [c for ids in CLASSES.values() for c in ids]
    prompt_ids = tok.encode(
        INSTRUCTION.format(classes=", ".join(all_classes)),
        add_special_tokens=False)

    T_cap = 257  # must match 02_train_projector's token cap
    for i, (iid, y) in enumerate(zip(img_ids, labels)):
        if iid in out:
            continue
        with torch.no_grad():
            v = torch.from_numpy(feats[i][:T_cap].astype(np.float16)) \
                    .to(device).unsqueeze(0)
            emb = proj(v)
            e = lm.get_input_embeddings()
            ids = torch.tensor(prompt_ids, device=device)
            # manual greedy loop: generate(inputs_embeds=...) returns empty
            # in this transformers version; the plain forward path is
            # verified correct on trained checkpoints (top-5 = class names)
            cur = torch.cat([emb, e(ids).unsqueeze(0).to(emb.dtype)], dim=1)
            gen_ids = []
            eos = tok.eos_token_id
            for _ in range(8):
                logits = lm(inputs_embeds=cur).logits[0, -1]
                nxt = int(logits.argmax())
                if nxt == eos:
                    break
                gen_ids.append(nxt)
                cur = torch.cat(
                    [cur,
                     e(torch.tensor([[nxt]], device=device)).to(emb.dtype)],
                    dim=1)
            txt = tok.decode(gen_ids, skip_special_tokens=True)
        out[iid] = {"class": y, "response": txt,
                    "correct": match_class_name(txt, y, all_classes)}
        if (i + 1) % 200 == 0:
            resp_path.write_text(json.dumps(out))
            print(f"[{arm}] {i+1}/{len(img_ids)}")
    resp_path.write_text(json.dumps(out))

    # per-class accuracy
    per_class = {}
    for iid, r in out.items():
        per_class.setdefault(r["class"], []).append(r["correct"])
    acc = {c: round(100 * sum(v) / len(v), 2) for c, v in per_class.items()}
    result = {"arm": arm, "n": len(out),
              "per_class_acc": acc,
              "mean_acc": round(sum(acc.values()) / len(acc), 2),
              "collapse_top_class_frac": round(
                  max(
                      (lambda cnt: cnt)(
                          sum(1 for r in out.values()
                              if r["response"].strip().lower() == c.lower()))
                      for c in all_classes
                  ) / max(1, len(out)), 3)}
    (RESP_DIR / f"{arm}_summary.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+",
                    default=["clip_l_laion", "siglip_so400m",
                             "dinov2_b14", "random_vitb"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    for arm in args.arms:
        evaluate_arm(arm, args.device, args.dry_run)


if __name__ == "__main__":
    main()
