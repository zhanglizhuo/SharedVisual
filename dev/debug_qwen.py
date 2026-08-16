import sys, torch
from transformers import AutoProcessor
from PIL import Image
import numpy as np

MODEL = "Qwen/Qwen2-VL-7B-Instruct"
proc = AutoProcessor.from_pretrained(MODEL, trust_remote_code=True)
img = Image.fromarray(np.random.randint(0, 255, (336, 336, 3), dtype=np.uint8))
out = proc.image_processor(images=[img], return_tensors="pt")
pv = out["pixel_values"]
print("pixel_values type:", type(pv))
if isinstance(pv, (list, tuple)):
    print("list of", len(pv), "each", tuple(pv[0].shape))
    print("min", pv[0].min(), "max", pv[0].max())
    pv_t = torch.cat(pv, dim=0)
else:
    print("tensor shape:", tuple(pv.shape))
    pv_t = pv
print("merged pixels shape:", tuple(pv_t.shape))
from transformers import Qwen2VLConfig
print("image_size cfg:", Qwen2VLConfig.from_pretrained(MODEL).vision_config.image_size)