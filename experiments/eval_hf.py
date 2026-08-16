#!/usr/bin/env python3
"""Eval CLIP-encoder MLLMs on SCB5 via HuggingFace (hf-mirror.com).

Protocol: IDENTICAL to VisualAnchor llava_scb5.py (the protocol that
produced mllm_full.json LLaVA-1.5-7B data):
  - Class-name prompt:
    "This is a classroom behavior image. Classify the dominant behavior: "
    + ", ".join(class_names) + ". Output only the behavior name."
  - Same random seed 42, same per-class sampling cap (--samples, default 100)
  - Same simple substring matching (first class found in response)
  - max_new_tokens=15, greedy decoding

This guarantees CLIP-encoder group (LLaVA-1.5-7B / 13B / Mistral-7B)
comparability within mllm_full.json.

Usage on 5090:
  ~/miniconda3/envs/llava/bin/python eval_hf.py llava-hf/llava-1.5-13b-hf
  ~/miniconda3/envs/llava/bin/python eval_hf.py llava-hf/llava-v1.6-mistral-7b-hf

Output: /tmp/{model_name}_scb5.json  (mllm_full-compatible per-class accuracy)
"""
import argparse, json, os, random, time
from pathlib import Path

import torch
from PIL import Image

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

SUBDATASETS = [
    {
        "key": "TeacherBehavior",
        "classes": ["guide", "answer", "On-stage interaction", "blackboard-writing",
                    "teacher", "stand", "screen", "blackBoard"],
        "subdir": "SCB5_TeacherBehavior/SCB5_Teacher_Behavior_Stand_BlackBoard_Sreen_20250406-2",
    },
    {
        "key": "HandriseReadWrite",
        "classes": ["hand-raising", "read", "write"],
        "subdir": "SCB5_HandriseReadWrite",
    },
    {
        "key": "BowTurnHead",
        "classes": ["BowHead", "TurnHead"],
        "subdir": "SCB_BowTurnHead",
    },
]


def read_label(path):
    with open(path) as f:
        line = f.readline().strip()
    if not line:
        return None
    return int(line.split()[0])


def match_class_name(response, true_class, all_classes):
    """First-class-found substring matching, exactly like llava_scb5.py."""
    resp = response.lower().strip()
    for i, c in enumerate(all_classes):
        if c.lower().strip() in resp:
            return all_classes[i] == true_class
    return False


def load_model(model_name):
    if "idefics" in model_name.lower():
        from transformers import IdeficsForVisionText2Text, IdeficsProcessor
        print(f"Loading {model_name} (IDEFICS)...", flush=True)
        model = IdeficsForVisionText2Text.from_pretrained(
            model_name, torch_dtype=torch.float16, device_map="auto",
        )
        processor = IdeficsProcessor.from_pretrained(model_name)
        model.eval()
        return model, processor
    elif "instructblip" in model_name.lower():
        from transformers import InstructBlipForConditionalGeneration, InstructBlipProcessor
        print(f"Loading {model_name} (InstructBLIP)...", flush=True)
        model = InstructBlipForConditionalGeneration.from_pretrained(
            model_name, torch_dtype=torch.float16, device_map="auto",
        )
        processor = InstructBlipProcessor.from_pretrained(model_name)
        model.eval()
        return model, processor
    elif "blip2" in model_name.lower():
        from transformers import Blip2ForConditionalGeneration, Blip2Processor
        print(f"Loading {model_name} (BLIP-2)...", flush=True)
        model = Blip2ForConditionalGeneration.from_pretrained(
            model_name, torch_dtype=torch.float16, device_map="auto",
        )
        processor = Blip2Processor.from_pretrained(model_name)
        model.eval()
        return model, processor
    elif "1.6" in model_name or "next" in model_name.lower():
        from transformers import LlavaNextForConditionalGeneration, AutoProcessor
        print(f"Loading {model_name} (LlavaNext)...", flush=True)
        model = LlavaNextForConditionalGeneration.from_pretrained(
            model_name, torch_dtype=torch.float16, device_map="auto",
        )
    else:
        from transformers import LlavaForConditionalGeneration, AutoProcessor
        print(f"Loading {model_name} (Llava)...", flush=True)
        model = LlavaForConditionalGeneration.from_pretrained(
            model_name, torch_dtype=torch.float16, device_map="auto",
        )
    processor = AutoProcessor.from_pretrained(model_name)
    model.eval()
    return model, processor


def load_scb5_dataset(sd, data_dir, samples_per_class):
    key = sd["key"]
    classes = sd["classes"]
    subdir = sd.get("subdir", key)
    sd_name = f"{data_dir}/{subdir}"

    img_dir = f"{sd_name}/images/val"
    lbl_dir = f"{sd_name}/labels/val"
    if not os.path.isdir(img_dir):
        print(f"  SKIP: {img_dir} not found", flush=True)
        return [], [], classes

    idx_per_class = {i: [] for i in range(len(classes))}
    for fname in sorted(os.listdir(img_dir)):
        if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        lp = os.path.join(lbl_dir, os.path.splitext(fname)[0] + ".txt")
        if not os.path.exists(lp):
            continue
        cid = read_label(lp)
        if cid is not None and cid in idx_per_class:
            idx_per_class[cid].append(os.path.join(img_dir, fname))

    images, labels = [], []
    for ci in range(len(classes)):
        indices = idx_per_class[ci]
        k = min(samples_per_class, len(indices))
        chosen = random.sample(indices, k) if k < len(indices) else indices
        for fpath in chosen:
            try:
                img = Image.open(fpath).convert("RGB").resize((336, 336), Image.BILINEAR)
                images.append(img)
                labels.append(ci)
            except Exception:
                pass
        print(f"  {classes[ci]}: {k}/{len(indices)} sampled", flush=True)
    return images, labels, classes


def run_inference(model, processor, images, labels, class_names, ds_name, save_responses=False):
    prompt = ("This is a classroom behavior image. Classify the dominant behavior: "
              + ", ".join(class_names) + ". Output only the behavior name.")
    is_blip2 = "Blip2" in type(model).__name__
    is_instructblip = "InstructBlip" in type(model).__name__
    is_idefics = "Idefics" in type(model).__name__
    results = {c: {"correct": 0, "total": 0} for c in class_names}
    responses = []
    total = len(images)
    t0 = time.time()

    for idx, (img, tl) in enumerate(zip(images, labels)):
        try:
            if is_idefics:
                text = (f"User: <fake_token_around_image><image><fake_token_around_image>"
                        f"{prompt}\nAssistant:")
                inputs = processor(text=[text], images=[img], return_tensors="pt")
                inputs = {k: (v.to(model.device).to(model.dtype)
                              if v.dtype.is_floating_point else v.to(model.device))
                          for k, v in inputs.items()}
            elif is_blip2 or is_instructblip:
                text = f"Question: {prompt} Answer:"
                inputs = processor(text=text, images=img, return_tensors="pt")
                inputs = {k: (v.to(model.device).to(model.dtype)
                              if v.dtype.is_floating_point else v.to(model.device))
                          for k, v in inputs.items()}
            else:
                text = f"USER: <image>\n{prompt}\nASSISTANT:"
                inputs = processor(text=text, images=img, return_tensors="pt").to(model.device)
            with torch.inference_mode():
                outputs = model.generate(**inputs, max_new_tokens=15, do_sample=False)
            resp = processor.decode(outputs[0], skip_special_tokens=True)
            if is_idefics:
                resp = resp.split("Assistant:")[-1].strip() if "Assistant:" in resp else resp
            elif is_blip2 or is_instructblip:
                resp = resp.split("Answer:")[-1].strip() if "Answer:" in resp else resp
            else:
                resp = resp.split("ASSISTANT:")[-1].strip() if "ASSISTANT:" in resp else resp

            if save_responses:
                responses.append({"class": class_names[tl], "response": resp[:500]})

            pred = None
            for i, c in enumerate(class_names):
                if c.lower().strip() in resp.lower().strip():
                    pred = i
                    break

            cname = class_names[tl]
            results[cname]["total"] += 1
            if pred == tl:
                results[cname]["correct"] += 1

            done = idx + 1
            if done % 20 == 0 or done == total:
                el = time.time() - t0
                rate = done / el if el > 0 else 0
                acc = sum(r["correct"] for r in results.values()) / max(1, sum(r["total"] for r in results.values()))
                eta = (total - done) / rate if rate > 0 else 0
                print(f"  [{ds_name} {done}/{total}] acc={acc:.3f} {rate:.1f}/s ETA={eta:.0f}s", flush=True)
        except Exception as e:
            print(f"  ERR: {e}", flush=True)
            results[class_names[tl]]["total"] += 1

    acc_dict = {c: {"correct": r["correct"], "total": r["total"],
                    "acc": round(r["correct"] / r["total"] * 100, 1) if r["total"] > 0 else 0}
                for c, r in results.items()}
    if save_responses:
        return acc_dict, responses
    return acc_dict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", default="llava-hf/llava-1.5-13b-hf", nargs="?")
    ap.add_argument("--data-dir", default="/tmp/scb5")
    ap.add_argument("--samples", type=int, default=100)
    ap.add_argument("--save-responses", action="store_true",
                    help="also save every raw generation response to /tmp/{model}_scb5_responses.json")
    args = ap.parse_args()

    random.seed(42)
    print(f"Model: {args.model}", flush=True)
    print(f"Data:  {args.data_dir}  samples/class: {args.samples}", flush=True)

    model, processor = load_model(args.model)
    all_res = {}
    all_raw = {}
    for sd in SUBDATASETS:
        key = sd["key"]
        print(f"\n=== {key} ({len(sd['classes'])} classes) ===", flush=True)
        images, labels, classes = load_scb5_dataset(sd, args.data_dir, args.samples)
        if len(images) == 0:
            print(f"  No images, skipping", flush=True)
            continue
        if args.save_responses:
            all_res[key], all_raw[key] = run_inference(model, processor, images, labels,
                                                       classes, key, save_responses=True)
        else:
            all_res[key] = run_inference(model, processor, images, labels, classes, key)

    safe_name = args.model.replace("/", "_").replace("-", "_")
    out_path = f"/tmp/{safe_name}_scb5.json"
    json.dump(all_res, open(out_path, "w"), indent=2)
    print(f"\nSaved -> {out_path}", flush=True)
    if all_raw:
        resp_path = f"/tmp/{safe_name}_scb5_responses.json"
        json.dump(all_raw, open(resp_path, "w"), indent=1)
        print(f"Saved responses -> {resp_path}", flush=True)

    # Per-class acc for mllm_full.json insertion
    print("\n=== PER-CLASS ACC (%) ===")
    for key, res in all_res.items():
        accs = {c: r["acc"] for c, r in res.items()}
        print(f'  "{key}": {json.dumps(accs)}')


if __name__ == "__main__":
    main()
