#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ImageHub 全功能组合压测：所有模型/后端/设计功能轮番调用，记录耗时与错误
用法: python full_stress.py [key]
"""
import base64
import io
import json
import os
import ssl
import sys
import time
import urllib.request

BASE = os.environ.get("IMAGEHUB_BASE_URL", "http://127.0.0.1:5569")
CA = os.environ.get("IMAGEHUB_CA", "/etc/imagehub/tls/imagehub.crt")
SSL_CTX = (ssl.create_default_context(cafile=CA)
           if BASE.startswith("https://") and os.path.exists(CA) else None)
KEY = sys.argv[1] if len(sys.argv) > 1 else ""
H = {"Content-Type": "application/json"}
LOG = "/var/log/full_stress.log"


def post(path, payload, timeout=2400):
    if KEY and "apikey" not in payload:
        payload["apikey"] = KEY
    req = urllib.request.Request(BASE + path, data=json.dumps(payload).encode(), headers=H)
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout, context=SSL_CTX) as r:
            d = json.loads(r.read())
        return True, round(time.time() - t0, 1), d
    except urllib.error.HTTPError as e:
        body = ""
        try:
            body = e.read().decode()[:180]
        except Exception:
            pass
        return False, round(time.time() - t0, 1), body
    except Exception as e:
        return False, round(time.time() - t0, 1), str(e)[:180]


def log(step, ok, dur, info=""):
    line = f"[{time.strftime('%H:%M:%S')}] {step:22s} {'OK ' if ok else 'ERR'} {dur:7.1f}s {str(info)[:110]}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def img_b64():
    import glob
    p = sorted(glob.glob("/opt/vLLM/imagehub/output/*.png"), key=os.path.getmtime)[-1]
    return base64.b64encode(open(p, "rb").read()).decode()


def mask_b64():
    from PIL import Image, ImageDraw
    m = Image.new("L", (1024, 1024), 0)
    ImageDraw.Draw(m).ellipse((300, 300, 724, 724), fill=255)
    b = io.BytesIO(); m.convert("RGB").save(b, "PNG")
    return base64.b64encode(b.getvalue()).decode()


IMG = img_b64()

# A. ComfyUI 文生图：4 个模型轮换（每次都是模型切换）
for m in ["zimage", "qwen", "h3", "moss"]:
    ok, d, r = post("/v1/images/generations",
                    {"model": m, "prompt": "全功能压测-文生图", "size": "1024x1024",
                     "seed": 7, "response_format": "url", "backend": "comfyui"})
    log(f"comfy t2i {m}", ok, d, r if not ok else (r.get("data") or [{}])[0].get("url"))

# B. 直连文生图（触发 ComfyUI 停 + 直连起）
for m in ["zimage", "qwen"]:
    ok, d, r = post("/v1/images/generations",
                    {"model": m, "prompt": "全功能压测-直连", "size": "1024x1024",
                     "seed": 8, "response_format": "url", "backend": "direct"})
    log(f"direct t2i {m}", ok, d, r if not ok else (r.get("data") or [{}])[0].get("url"))

# C. 设计：ControlNet
ok, d, r = post("/v1/design/render", {"prompt": "全功能压测-CN", "control_image": IMG,
                                      "control_type": "canny", "size": "1024x1024", "seed": 3})
log("design render", ok, d, r if not ok else r.get("image"))

# D. 白模
ok, d, r = post("/v1/design/clay2render", {"image": IMG, "materials": "清水混凝土",
                                           "lighting": "黄昏暖光", "size": "1024x1024", "seed": 11})
log("clay2render", ok, d, r if not ok else r.get("image"))

# E. 线稿
ok, d, r = post("/v1/design/sketch2render", {"sketch": IMG, "style": "写实建筑",
                                             "size": "1024x1024", "seed": 12})
log("sketch2render", ok, d, r if not ok else r.get("image"))

# F. 多角度
ok, d, r = post("/v1/design/views", {"image": IMG, "base_prompt": "现代办公楼",
                                     "views": ["人视", "鸟瞰"], "size": "1024x1024", "seed": 13})
log("views x2", ok, d, r if not ok else f"{r.get('count')} 张")

# G. 批量
ok, d, r = post("/v1/design/batch", {"prompt": "批量方案", "n": 2, "seed_start": 20,
                                     "size": "1024x1024"})
log("batch n=2", ok, d, r if not ok else f"{r.get('count')} 张")

# H. 局部重绘
ok, d, r = post("/v1/design/render", {"prompt": "把中间改成天空", "control_image": IMG,
                                      "control_type": "raw", "mask_image": mask_b64(),
                                      "size": "1024x1024", "seed": 31})
log("inpaint", ok, d, r if not ok else r.get("image"))

# I. 扩图
ok, d, r = post("/v1/design/outpaint", {"image": IMG, "prompt": "补全边缘",
                                        "expand": 256, "seed": 9})
log("outpaint", ok, d, r if not ok else r.get("size"))

# J. 异步 + 轮询
ok, d, r = post("/v1/design/async", {"prompt": "异步压测", "control_image": IMG,
                                     "control_type": "raw", "size": "1024x1024", "seed": 77})
if ok:
    jid = r.get("job_id")
    t0 = time.time()
    while time.time() - t0 < 900:
        time.sleep(4)
        try:
            j = json.loads(urllib.request.urlopen(f"{BASE}/v1/jobs/{jid}", timeout=10, context=SSL_CTX).read())
        except Exception:
            continue
        if j.get("status") in ("done", "error"):
            log("async job", j.get("status") == "done", round(time.time() - t0, 1), j.get("status"))
            break
else:
    log("async job", False, d, r)

# K. 图片编辑（20 步）
ok, d, r = post("/api/edit", {"prompt": "把背景改成夜晚", "image_b64": IMG, "seed": 7, "steps": 12})
log("edit", ok, d, r if not ok else r.get("image"))

# L. 多模型对比
ok, d, r = post("/api/compare", {"prompt": "全功能压测-对比", "models": ["h3", "qwen", "zimage"],
                                 "width": 1024, "height": 1024, "seed": 7, "backend": "comfyui"})
if ok:
    jid = r.get("job_id")
    t0 = time.time()
    while time.time() - t0 < 1800:
        time.sleep(5)
        try:
            j = json.loads(urllib.request.urlopen(f"{BASE}/api/jobs/{jid}", timeout=10, context=SSL_CTX).read())
        except Exception:
            continue
        if j.get("status") == "done":
            log("compare x3", True, round(time.time() - t0, 1),
                [(x.get("model"), x.get("status"), x.get("duration")) for x in j.get("results", [])])
            break
else:
    log("compare x3", False, d, r)

print("DONE", flush=True)
