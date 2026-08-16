#!/usr/bin/env python3
"""Quick test: verify bakllava:7b works on V100."""
import base64, time, requests, os, subprocess, sys

subprocess.run(["pkill", "-9", "ollama"], capture_output=True)
time.sleep(2)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["OLLAMA_HOST"] = "127.0.0.1:11434"
os.environ["OLLAMA_MODELS"] = "/usr/share/ollama/.ollama/models"
os.environ["OLLAMA_KEEP_ALIVE"] = "0"
p = subprocess.Popen(["/usr/local/bin/ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
time.sleep(5)

MODEL = sys.argv[1] if len(sys.argv) > 1 else "bakllava:7b"

# warmup
r = requests.post("http://127.0.0.1:11434/api/generate", json={
    "model": MODEL, "prompt": "hi", "stream": False,
    "options": {"temperature": 0, "num_predict": 1}
}, timeout=30)
print(f"warmup: {r.elapsed.total_seconds():.1f}s status={r.status_code}", flush=True)

# build prompt
CM = {
    "guide": "guiding students in classroom",
    "answer": "answering questions in class",
    "On-stage interaction": "on stage interaction",
    "blackboard-writing": "writing on blackboard",
    "stand": "standing at the front of a classroom",
    "blackBoard": "standing near a blackboard or whiteboard",
}
classes = ["guide", "answer", "On-stage interaction", "blackboard-writing",
           "teacher", "stand", "screen", "blackBoard"]
classes_str = ", ".join(CM.get(c, c) for c in classes)
prompt = (
    "Classify the classroom activity in this image. "
    f"Choose exactly one category from the list: {classes_str}. "
    "Output only the category name, nothing else."
)

IMG = "/home/broadsense/works/lizhuo/AutoResearchClaw/datasets_scb/SCB5_TeacherBehavior/SCB5_Teacher_Behavior_Stand_BlackBoard_Sreen_20250406-2/images/val/0001001.jpg"
LBL = IMG.replace("images", "labels").replace(".jpg", ".txt")
with open(LBL) as f:
    true_cid = int(f.readline().split()[0])
true_class = classes[true_cid]
with open(IMG, "rb") as f:
    img_b64 = base64.b64encode(f.read()).decode()

t0 = time.time()
r = requests.post("http://127.0.0.1:11434/api/chat", json={
    "model": MODEL,
    "messages": [{"role": "user", "content": prompt, "images": [img_b64]}],
    "stream": False,
    "options": {"temperature": 0, "num_predict": 32},
}, timeout=120)
dt = time.time() - t0
print(f"dt={dt:.1f}s status={r.status_code}", flush=True)
if r.status_code == 200:
    content = r.json().get("message", {}).get("content", "").strip()
    print(f"true={true_class} resp=\"{content[:100]}\"", flush=True)
else:
    print(f"ERROR: {r.text[:300]}", flush=True)

# Test image 2 (different class)
IMG2 = IMG.replace("0001001", "0001002")
LBL2 = IMG2.replace("images", "labels").replace(".jpg", ".txt")
if os.path.exists(LBL2):
    with open(LBL2) as f:
        cid2 = int(f.readline().split()[0])
    true2 = classes[cid2]
    with open(IMG2, "rb") as f:
        img2 = base64.b64encode(f.read()).decode()
    t0 = time.time()
    r = requests.post("http://127.0.0.1:11434/api/chat", json={
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt, "images": [img2]}],
        "stream": False,
        "options": {"temperature": 0, "num_predict": 32},
    }, timeout=120)
    dt = time.time() - t0
    print(f"img2: dt={dt:.1f}s status={r.status_code} true={true2}", flush=True)
    if r.status_code == 200:
        content = r.json().get("message", {}).get("content", "").strip()
        print(f"resp=\"{content[:100]}\"", flush=True)

p.terminate()
print("DONE", flush=True)
