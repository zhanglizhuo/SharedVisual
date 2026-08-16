#!/usr/bin/env python3
"""Test which PyPI mirror can actually serve torch wheels from the 5090."""
import urllib.request, time
urls = [
    ("tsinghua", "https://pypi.tuna.tsinghua.edu.cn/simple/torch/"),
    ("aliyun",   "https://mirrors.aliyun.com/pypi/simple/torch/"),
    ("douban",   "https://pypi.doubanio.com/simple/torch/"),
    ("ustc",     "https://pypi.mirrors.ustc.edu.cn/simple/torch/"),
    ("pypi",     "https://pypi.org/simple/torch/"),
    ("hf-mirror","https://hf-mirror.com"),  # just connectivity
]
for name, url in urls:
    try:
        t0 = time.time()
        req = urllib.request.Request(url, headers={"User-Agent": "curl/8"})
        r = urllib.request.urlopen(req, timeout=10)
        dt = time.time() - t0
        body = r.read(2000)
        print(f"{name:10s} OK   {r.status} {dt:.2f}s  {len(body)}B")
    except Exception as e:
        print(f"{name:10s} FAIL {type(e).__name__}: {str(e)[:80]}")
