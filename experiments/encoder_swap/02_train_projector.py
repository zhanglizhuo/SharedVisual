#!/usr/bin/env python3
"""Encoder-swap stage 2: equal-budget projector training (frozen LM).

LLaVA-style: frozen Mistral-7B-v0.1 LM + per-arm 2-layer MLP projector mapping
vision tokens -> LM hidden dim. Classification-style instruction tuning on the
leakage-cleaned SCB5 train pool (13 classes). All arms share the preregistered
budget: AdamW, lr 1e-4 cosine->0, 3% warmup, 2000 optimizer steps, batch 8,
fp16, seed 42, class-balanced sampler.

Resumable: checkpoints per (arm, step) in results/encoder_swap/ckpt/{arm}/; a rerun
resumes from the latest checkpoint. NOT YET RUN — validate with --dry-run.
"""
import argparse
import json
import math
import os
import random
from pathlib import Path

import numpy as np

REPO_ROOT = Path(os.environ.get(
    "SHAREDVISUAL_ROOT", str(Path(__file__).resolve().parents[2])))
RESULTS = Path(os.environ.get("SHAREDVISUAL_RESULTS", str(REPO_ROOT / "results")))
FEAT_DIR = RESULTS / "encoder_swap" / "features"
CKPT_DIR = RESULTS / "encoder_swap" / "ckpt"

LM_ID = "mistralai/Mistral-7B-v0.1"
LM_HIDDEN = 4096
CLASSES = {
    "TeacherBehavior": ["guide", "answer", "On-stage interaction",
                        "blackboard-writing", "teacher", "stand", "screen",
                        "blackBoard"],
    "HandriseReadWrite": ["hand-raising", "read", "write"],
    "BowTurnHead": ["BowHead", "TurnHead"],
}
INSTRUCTION = ("Classify the person's behavior in this classroom image into "
               "one of: {classes}. Answer with the class name only.")
TOTAL_STEPS = 2000
BATCH_SIZE = 8
LR = 1e-4
WARMUP_FRAC = 0.03
SEED = 42
MAX_POS = 1024  # vision tokens + prompt + answer guard


def build_projector(in_dim, hidden, out_dim):
    import torch.nn as nn
    return nn.Sequential(
        nn.Linear(in_dim, hidden),
        nn.GELU(),
        nn.Linear(hidden, out_dim),
    )


def class_balanced_batches(labels, batch_size, steps, rng):
    """Class-balanced sampling: each batch draws classes round-robin then
    samples within class. Deterministic given rng."""
    by_class = {}
    for i, y in enumerate(labels):
        by_class.setdefault(y, []).append(i)
    classes = sorted(by_class)
    order = []
    for _ in range(steps * batch_size):
        c = classes[len(order) % len(classes)]
        order.append(rng.choice(by_class[c]))
    for s in range(steps):
        yield order[s * batch_size:(s + 1) * batch_size]


def train_arm(arm, device="cuda", dry_run=False):
    ckpt = CKPT_DIR / arm
    if dry_run:
        feat_probe = FEAT_DIR / f"{arm}_train.npz"
        have = feat_probe.exists()
        print(f"[dry-run] {arm}: plan = {TOTAL_STEPS} steps, batch {BATCH_SIZE}, "
              f"lr {LR} cosine, LM {LM_ID} frozen; "
              f"train features {'present' if have else 'NOT extracted yet'}")
        return None
    import torch
    import torch.nn.functional as F
    from transformers import AutoModelForCausalLM, AutoTokenizer

    ckpt.mkdir(parents=True, exist_ok=True)
    feat_path = FEAT_DIR / f"{arm}_train.npz"
    if not feat_path.exists():
        raise FileNotFoundError(f"{feat_path} missing — run 01_extract_features.py")

    data = np.load(feat_path, allow_pickle=True)
    feats = data["features"]            # (N, T_i, D_i) fp16
    labels_raw = data["labels"]         # per-dataset class ids or None
    img_ids = data["img_ids"]           # dataset/filename

    # leakage discipline (exp1b convention): drop any TRAIN image whose
    # filename collides with a val filename (cross-dataset duplication by
    # md5 in the exp1b audit; filename collision is the operative rule).
    mapping = json.load(open(RESULTS / "scb5_true_class_mapping.json"))
    val_names = {Path(p).name for p in mapping}
    keep = [i for i, iid in enumerate(img_ids)
            if Path(str(iid)).name not in val_names]
    n_dropped = len(img_ids) - len(keep)
    print(f"[leakage] {arm}: dropped {n_dropped} filename-colliding train "
          f"images (exp1b convention); pool {len(keep)}")
    feats, labels_raw = feats[keep], [labels_raw[i] for i in keep]
    img_ids = [img_ids[i] for i in keep]
    dataset_of = np.array([i.split("/")[0] for i in img_ids])

    # resolve class names: val npz stores names; train stores YOLO ids
    def resolve(i):
        ds = dataset_of[i]
        ids = CLASSES[ds]
        raw = labels_raw[i]
        if isinstance(raw, str) and raw in ids:
            return raw
        try:
            return ids[int(raw)]
        except (ValueError, TypeError, IndexError):
            return None
    labels = [resolve(i) for i in range(len(img_ids))]
    keep = [i for i, y in enumerate(labels) if y is not None]
    feats, labels = feats[keep], [labels[i] for i in keep]

    rng = random.Random(SEED)
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    tok = AutoTokenizer.from_pretrained(LM_ID)
    tok.pad_token = tok.pad_token or tok.eos_token
    lm = AutoModelForCausalLM.from_pretrained(
        LM_ID, torch_dtype=torch.float16, use_safetensors=True).to(device).eval()
    for p in lm.parameters():
        p.requires_grad_(False)
    lm.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False})
    lm.config.use_cache = False

    T_i, D_i = feats.shape[1], feats.shape[2]
    # cap vision tokens at 257 for all arms (equal-budget fairness + memory):
    # SigLIP's 730 tokens are center-cropped to the first 257
    if T_i > 257:
        feats = feats[:, :257, :]
        T_i = 257
    proj = build_projector(D_i, (D_i + LM_HIDDEN) // 2, LM_HIDDEN).to(device)
    proj = proj.to(torch.float32)
    opt = torch.optim.AdamW(proj.parameters(), lr=LR, weight_decay=0.0)

    steps_done = 0
    state_path = ckpt / "latest.pt"
    if state_path.exists():
        st = torch.load(state_path, map_location=device)
        proj.load_state_dict(st["proj"])
        opt.load_state_dict(st["opt"])
        steps_done = st["step"]
        print(f"[resume] {arm} from step {steps_done}")

    def lr_at(step):
        w = int(TOTAL_STEPS * WARMUP_FRAC)
        if step < w:
            return LR * step / max(1, w)
        t = (step - w) / max(1, TOTAL_STEPS - w)
        return LR * 0.5 * (1 + math.cos(math.pi * t))

    all_classes = [c for ids in CLASSES.values() for c in ids]
    prompt_txt = INSTRUCTION.format(classes=", ".join(all_classes))
    answer_ids = {c: tok.encode(" " + c, add_special_tokens=False)
                  for c in all_classes}
    prompt_ids = tok.encode(prompt_txt, add_special_tokens=False)

    batches = class_balanced_batches(labels, BATCH_SIZE,
                                     TOTAL_STEPS - steps_done, rng)
    log = open(ckpt / "train.log", "a")
    for bi, idxs in enumerate(batches, start=steps_done):
        for g in opt.param_groups:
            g["lr"] = lr_at(bi)
        loss = 0.0
        for i in idxs:
            v = torch.from_numpy(feats[i].astype(np.float32)).to(device) \
                    .unsqueeze(0)                                    # (1,T,D)
            emb = proj(v).to(torch.float16)                          # (1,T,H)
            ans = answer_ids[labels[i]]
            ids = torch.tensor(
                prompt_ids + ans, device=device).unsqueeze(0)
            if ids.shape[1] + emb.shape[1] > MAX_POS:
                v = v[:, :MAX_POS - ids.shape[1] - 8, :]
                emb = proj(v).to(torch.float16)
            inputs_embeds = torch.cat([
                emb,
                lm.get_input_embeddings()(ids)], dim=1)
            out = lm(inputs_embeds=inputs_embeds, use_cache=False)
            A = len(ans)
            # rows L-A-1 .. L-2 predict the A answer tokens
            logits = out.logits[0, -(A + 1):-1, :]
            tgt = torch.tensor(ans, device=device)
            loss = loss + F.cross_entropy(logits.float(), tgt, reduction="sum")
        loss = loss / len(idxs)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(proj.parameters(), 1.0)
        opt.step()
        if bi % 50 == 0 or bi == TOTAL_STEPS - 1:
            msg = f"step {bi} loss {float(loss):.4f} lr {lr_at(bi):.2e}"
            print(f"[{arm}] {msg}", flush=True)
            log.write(msg + "\n")
            log.flush()
        if bi % 250 == 0 or bi == TOTAL_STEPS - 1:
            torch.save({"proj": proj.state_dict(), "opt": opt.state_dict(),
                        "step": bi + 1}, state_path)
    log.close()
    return str(state_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+",
                    default=["clip_l_laion", "siglip_so400m",
                             "dinov2_b14", "random_vitb"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    manifest = {}
    for arm in args.arms:
        manifest[arm] = train_arm(arm, args.device, args.dry_run)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    (CKPT_DIR / "train_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
