"""#4 Perturbation synchrony pilot: gemma4:31b under Gaussian blur.

Goal: test if the geometry -> behavior link can be causally probed by
perturbation. Previous attempt (Qwen2-VL-7B) floored at ~12-13% even at sigma=0.
This pilot uses gemma4:31b (SigLIP encoder, SCB5 rho=0.73) which has high
baseline accuracy and should escape floor at moderate blur.

If gemma4:31b gives a non-degenerate accuracy profile across sigma in
{0,1,2,3,4}, we compute per-class accuracy drops for both gemma4:31b and CLIP
(LAION L/14), and test whether the per-class drops are synchronized (Spearman
rho across the 8 classes) -- a direct geometry -> behavior perturbation test.

Protocol: identical to eval_clip_mllms_v100.py (class-name prompt via
CLASS_NAME_MAP, temperature 0, /api/chat, match_class_name scoring), so results
are directly comparable to the paper's existing MLLM accuracy numbers.

CRITICAL (2026-08-10): gemma4:31b is a THINKING model. With num_predict=32 the
whole budget is consumed by the CoT and message.content is empty -> 0% accuracy
for every class. Also Ollama must be >= 0.20.x with gemma4 RENDERER (v0.32.6
installed on V100 after 0.20.0 produced empty completions). Fix: "think": False
must be TOP-LEVEL in the request body (NOT inside options -- options.think is
silently ignored for multimodal requests and the CoT eats the whole budget);
num_predict=256. Verified: content='teacher in classroom', ~20s/img.

Idempotent: per (sigma, class, image) checkpoints; resume skips done work.
Run on V100 (GPU 0, 1 Ollama server; gemma4:31b is 19GB, fits 1 GPU):
  python experiments/exp_perturbation_blur.py
"""
import os, sys, json, time, base64, subprocess, signal, atexit, io
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests

# Optional Gaussian blur via scipy/PIL
from PIL import Image, ImageFilter

DATA_ROOT = Path(os.environ.get("SHAREDVISUAL_DATA", str(Path.home() / "data")))
SCB5_BASE = (DATA_ROOT / "SCB5_TeacherBehavior"
             / "SCB5_Teacher_Behavior_Stand_BlackBoard_Sreen_20250406-2")
IMG_DIR = SCB5_BASE / "images" / "val"
LBL_DIR = SCB5_BASE / "labels" / "val"
RESULTS_DIR = Path(os.environ.get("SHAREDVISUAL_RESULTS", str(Path(__file__).resolve().parents[1] / "results")))
CKPT_DIR = RESULTS_DIR / "blur_gemma4_31b_ckpt_discoring"
OUT_PATH = RESULTS_DIR / "exp_perturbation_blur_dual.json"

CLASSES = ["guide", "answer", "On-stage interaction", "blackboard-writing",
           "teacher", "stand", "screen", "blackBoard"]
CLASS_NAME_MAP = {
    "On-stage interaction": "on stage interaction",
    "blackboard-writing": "writing on blackboard",
    "blackBoard": "standing near a blackboard or whiteboard",
    "stand": "standing at the front of a classroom",
    "guide": "guiding students in classroom",
    "answer": "answering questions in class",
    "teacher": "teacher in classroom",
    "screen": "showing a screen in classroom",
}
BLUR_SIGMAS = [0, 1, 2, 3, 4]
N_PER_CLASS = 50  # match existing geometry analysis
MODEL = "gemma4:31b"
OLLAMA_BIN = "/usr/local/bin/ollama"
PORT = int(os.environ.get("BLUR_PORT", "11450"))  # dedicated port for this pilot
GPU_ID = int(os.environ.get("BLUR_GPU", "0"))
CKPT_EVERY = 25

PROMPT = ("Classify the classroom activity in this image. "
          "Choose exactly one category from the list: {classes}. "
          "Output only the category name, nothing else.").format(
              classes=", ".join(CLASS_NAME_MAP.get(c, c) for c in CLASSES))


def read_label(path):
    with open(path) as f:
        line = f.readline().strip()
    return int(line.split()[0]) if line else None


def match_class_name(response, true_class, all_classes, name_map=None):
    """Substring match against the DISPLAY names (name_map values) used in the
    prompt. The model answers in prompt vocabulary (e.g. 'guiding students in
    classroom'), so matching raw CLASSES keys ('guide') misses everything except
    'teacher'. This mirrors eval_hf.py first-class-found matching."""
    resp = response.lower().strip().rstrip(".,;:!?\n\t ")
    true_disp = (name_map or {}).get(true_class, true_class).lower()
    if resp == true_disp:
        return True
    valid = [(name_map or {}).get(c, c).lower() for c in all_classes]
    matched = [c for c in valid if c in resp]
    if len(matched) == 1:
        return matched[0] == true_disp
    elif len(matched) > 1:
        return max(matched, key=len) == true_disp
    return False


def match_class_name_raw_key(response, true_class, all_classes):
    """Exact legacy scoring from eval_clip_mllms_v100.py / VisualAnchor
    mllm_4gpu_scb5.py: substring match against the raw CLASSES keys
    (guide, answer, ...). Kept feature-identical so the pilot's sigma=0
    numbers are directly comparable to the paper's 9-MLLM tables."""
    resp = response.lower().strip().rstrip(".,;:!?\n\t ")
    true_lower = true_class.lower()
    if resp == true_lower:
        return True
    valid = [c.lower() for c in all_classes]
    matched = [c for c in valid if c in resp]
    if len(matched) == 1:
        return matched[0] == true_lower
    elif len(matched) > 1:
        return max(matched, key=len) == true_lower
    return False


def dual_match(response, true_class, all_classes, name_map):
    """Return (match_display, match_raw_key) for one response."""
    return (match_class_name(response, true_class, all_classes, name_map),
            match_class_name_raw_key(response, true_class, all_classes))


def collect_samples():
    """First N_PER_CLASS images per class (sorted, deterministic)."""
    counts = {c: 0 for c in CLASSES}
    samples = []  # (img_path, fname, true_class)
    for fname in sorted(os.listdir(IMG_DIR)):
        if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        lbl_path = LBL_DIR / (os.path.splitext(fname)[0] + ".txt")
        if not lbl_path.exists():
            continue
        cid = read_label(lbl_path)
        if cid is None or not (0 <= cid < len(CLASSES)):
            continue
        cls = CLASSES[cid]
        if counts[cls] >= N_PER_CLASS:
            continue
        samples.append((str(IMG_DIR / fname), fname, cls))
        counts[cls] += 1
    return samples, counts


def blur_image_b64(img_path, sigma):
    """Load image, apply Gaussian blur with given sigma (0 = no blur), return base64."""
    with open(img_path, "rb") as f:
        if sigma == 0:
            return base64.b64encode(f.read()).decode()
        img = Image.open(img_path).convert("RGB")
        # PIL GaussianBlur radius ~ sigma (visually equivalent for our purposes)
        blurred = img.filter(ImageFilter.GaussianBlur(radius=sigma))
        buf = io.BytesIO()
        blurred.save(buf, format="JPEG", quality=90)
        return base64.b64encode(buf.getvalue()).decode()


def classify(port, img_b64, max_retries=2):
    for attempt in range(max_retries + 1):
        try:
            resp = requests.post(f"http://127.0.0.1:{port}/api/chat", json={
                "model": MODEL,
                "messages": [{"role": "user", "content": PROMPT,
                              "images": [img_b64]}],
                "stream": False,
                "think": False,
                "options": {"temperature": 0, "num_predict": 256},
            }, timeout=120)
            return resp.json().get("message", {}).get("content", "").strip().lower()
        except Exception as e:
            if attempt < max_retries:
                time.sleep(3)
            else:
                print(f"    classify fail: {e}", flush=True)
                return ""


def run_sigma(sigma, samples, ckpt_dir):
    """Idempotent: skip images already in checkpoint for this sigma.
    Records raw model responses so both scoring conventions (display-name
    vs raw-key) can be recomputed offline."""
    sig_key = f"blur_{sigma}"
    ckpt_path = ckpt_dir / f"{sig_key}.json"
    ckpt = {"done": [], "per_class": {}, "responses": {}}
    if ckpt_path.exists():
        ckpt = json.loads(ckpt_path.read_text())
    done = set(ckpt.get("done", []))
    print(f"\n=== sigma={sigma} ({len(done)}/{len(samples)} done) ===", flush=True)

    t0 = time.time()
    for idx, (img_path, fname, true_class) in enumerate(samples):
        if fname in done:
            continue
        img_b64 = blur_image_b64(img_path, sigma)
        content = classify(PORT, img_b64)
        ckpt["responses"][fname] = {"class": true_class, "response": content}
        match_d, match_r = dual_match(content, true_class, CLASSES, CLASS_NAME_MAP)

        pc = ckpt["per_class"].setdefault(true_class, {"correct": 0, "total": 0})
        pc["total"] += 1
        if match_d:
            pc["correct"] += 1
        # legacy raw-key counts kept separately
        rk = ckpt.setdefault("per_class_rawkey", {}).setdefault(
            true_class, {"correct": 0, "total": 0})
        rk["total"] += 1
        if match_r:
            rk["correct"] += 1
        ckpt["done"].append(fname)

        if (len(ckpt["done"]) % CKPT_EVERY) == 0 or len(ckpt["done"]) == len(samples):
            elapsed = time.time() - t0
            print(f"  sigma={sigma}: {len(ckpt['done'])}/{len(samples)}"
                  f"  ({len(ckpt['done'])/max(elapsed,1):.2f}/s)", flush=True)
            ckpt_path.write_text(json.dumps(ckpt))

    ckpt_path.write_text(json.dumps(ckpt))
    # Compute per-class accuracy under both scorings
    acc = {}
    acc_rk = {}
    for cls in CLASSES:
        pc = ckpt["per_class"].get(cls, {"correct": 0, "total": 0})
        acc[cls] = pc["correct"] / max(pc["total"], 1)
        rk = ckpt.get("per_class_rawkey", {}).get(cls, {"correct": 0, "total": 0})
        acc_rk[cls] = rk["correct"] / max(rk["total"], 1)
    overall = sum(c["correct"] for c in ckpt["per_class"].values()) / max(
        sum(c["total"] for c in ckpt["per_class"].values()), 1)
    overall_rk = sum(c["correct"] for c in ckpt.get("per_class_rawkey", {}).values()) / max(
        sum(c["total"] for c in ckpt.get("per_class_rawkey", {}).values()), 1)
    print(f"  sigma={sigma} overall acc (display): {overall:.3f} | (raw-key): {overall_rk:.3f}",
          flush=True)
    return acc, acc_rk, overall, overall_rk


# ── Ollama lifecycle ────────────────────────────────────────────────────────
server_proc = None


def start_ollama():
    global server_proc
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(GPU_ID)
    env["OLLAMA_HOST"] = f"127.0.0.1:{PORT}"
    env["OLLAMA_MODELS"] = "/usr/share/ollama/.ollama/models"
    env["OLLAMA_KEEP_ALIVE"] = "10m"
    server_proc = subprocess.Popen(
        [OLLAMA_BIN, "serve"], env=env,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print(f"  Ollama on GPU{GPU_ID} port {PORT} (PID {server_proc.pid})", flush=True)
    for _ in range(60):
        try:
            r = requests.get(f"http://127.0.0.1:{PORT}/api/tags", timeout=2)
            if r.status_code == 200:
                break
        except Exception:
            pass
        time.sleep(1)
    else:
        raise RuntimeError(f"Ollama port {PORT} failed to start")
    # warmup (loads model weights)
    t0 = time.time()
    r = requests.post(f"http://127.0.0.1:{PORT}/api/generate", json={
        "model": MODEL, "prompt": "warmup", "stream": False,
        "options": {"temperature": 0, "num_predict": 1},
    }, timeout=300)
    r.raise_for_status()
    print(f"  model loaded in {time.time()-t0:.1f}s", flush=True)


def cleanup(*_):
    if server_proc:
        try:
            server_proc.terminate()
            server_proc.wait(timeout=5)
        except Exception:
            try:
                server_proc.kill()
            except Exception:
                pass


atexit.register(cleanup)
signal.signal(signal.SIGTERM, cleanup)
signal.signal(signal.SIGINT, cleanup)


def main():
    print("=" * 60)
    print(f"Perturbation blur pilot: {MODEL} on SCB5 TeacherBehavior")
    print("=" * 60)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    samples, counts = collect_samples()
    print(f"Samples: {len(samples)} images, per class: {counts}")
    print(f"Blur sigmas: {BLUR_SIGMAS}")

    start_ollama()

    results = {}
    for sigma in BLUR_SIGMAS:
        acc, acc_rk, overall, overall_rk = run_sigma(sigma, samples, CKPT_DIR)
        results[f"sigma_{sigma}"] = {
            "per_class_acc": acc,
            "per_class_acc_rawkey": acc_rk,
            "overall_acc": overall,
            "overall_acc_rawkey": overall_rk,
        }

    out = {
        "model": MODEL,
        "task": "SCB5 TeacherBehavior 8-class classification under Gaussian blur",
        "n_per_class": N_PER_CLASS,
        "blur_sigmas": BLUR_SIGMAS,
        "prompt": PROMPT,
        "scoring": {
            "display": "match_class_name (display names in prompt vocabulary)",
            "rawkey": "match_class_name_raw_key (legacy VisualAnchor raw CLASSES keys, "
                      "identical to eval_clip_mllms_v100.py scoring for paper comparability)",
        },
        "results": results,
    }
    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to {OUT_PATH}")
    print("\n=== Accuracy profile ===")
    for sigma in BLUR_SIGMAS:
        r = results[f"sigma_{sigma}"]
        print(f"  sigma={sigma}: display={r['overall_acc']:.3f}  rawkey={r['overall_acc_rawkey']:.3f}")


if __name__ == "__main__":
    main()
