#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ImageHub BFF：统一调用 ComfyUI，支持 4 个文生图模型切换与对比
  POST /api/compare  {prompt, models:[...], width?, height?, seed?}
  GET  /api/jobs/{id}
  GET  /api/models
  GET  /api/history
  GET  /image/{filename}
"""
import json
import logging
import os
import sqlite3
import subprocess
import threading
import time
import uuid
import urllib.request
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, HTMLResponse, RedirectResponse
from starlette.middleware.sessions import SessionMiddleware
from urllib.parse import quote
from pydantic import BaseModel

APP_VERSION = "1.3.5"          # ImageHub 版本号
APP_BUILD = "2026-11-20"

logger = logging.getLogger("imagehub")
logging.basicConfig(level=logging.INFO)

COMFY = "http://127.0.0.1:5568"
COMFY_OUTPUT = "/opt/ComfyUI/output"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "imagehub.db")

# ---- 读取 /opt/vLLM/imagehub/.env（不打印密钥）----
def _load_env():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if not os.path.isfile(path):
        return
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())


_load_env()
YZ_LOGIN_URL = os.environ.get("YZ_LOGIN_URL", "http://192.168.0.8")
YZ_APP_ID = os.environ.get("YZ_APP_ID", "125")
YZ_APP_SECRET = os.environ.get("YZ_APP_SECRET", "")
SESSION_SECRET = os.environ.get("SESSION_SECRET", "imagehub-dev-secret-change-me")
SSO_ENABLED = bool(YZ_APP_SECRET)

app = FastAPI(title="ImageHub")
app.add_middleware(SessionMiddleware, secret_key=SESSION_SECRET,
                   session_cookie="imagehub_session", max_age=8 * 3600)
_jobs = {}
_lock = threading.Lock()
_STATE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "state.json")


def _load_state():
    try:
        with open(_STATE_FILE, encoding="utf-8") as f:
            d = json.load(f)
        if d.get("loaded") in MODELS:
            return d["loaded"]
    except Exception:
        pass
    return None


def _save_state(model_key):
    try:
        with open(_STATE_FILE, "w", encoding="utf-8") as f:
            json.dump({"loaded": model_key, "ts": time.time()}, f)
    except Exception:
        pass


_state = {"loaded": None, "active": 0, "switching": False, "target": None}
_cond = threading.Condition()


# ==================== 工作流定义 ====================
def wf_h3(prompt, w, h, seed):
    return {
        "1": {"class_type": "UnetLoaderGGUFMultiGPU",
              "inputs": {"unet_name": "MiniMax-H3-Image-Pruned-Q5_K_M.gguf", "device": "cuda:0"}},
        "2": {"class_type": "CLIPLoaderGGUFMultiGPU",
              "inputs": {"clip_name": "qwen3vl_32b_minimax_h3-Q4_K_M.gguf", "type": "minimax", "device": "cuda:1"}},
        "3": {"class_type": "VAELoaderMultiGPU",
              "inputs": {"vae_name": "minimax_h3_vae.safetensors", "device": "cuda:0"}},
        "4": {"class_type": "MiniMaxH3ImageConditioning",
              "inputs": {"clip": ["2", 0], "prompt": prompt, "width": w, "height": h}},
        "5": {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["4", 0]}},
        "6": {"class_type": "KSampler",
              "inputs": {"model": ["1", 0], "positive": ["4", 0], "negative": ["5", 0],
                         "latent_image": ["4", 1], "seed": seed, "steps": 6, "cfg": 1.0,
                         "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0}},
        "7": {"class_type": "VAEDecode", "inputs": {"samples": ["6", 0], "vae": ["3", 0]}},
        "8": {"class_type": "SaveImage", "inputs": {"images": ["7", 0], "filename_prefix": "hub_h3"}},
    }


def wf_qwen(prompt, w, h, seed):
    return {
        "1": {"class_type": "UNETLoaderMultiGPU",
              "inputs": {"unet_name": "qwen_image_2.1_turbo_bf16.safetensors", "weight_dtype": "default", "device": "cuda:0"}},
        "2": {"class_type": "QwenImage21Cache", "inputs": {"model": ["1", 0], "device": "auto", "dtype": "default"}},
        "3": {"class_type": "CLIPLoaderMultiGPU",
              "inputs": {"clip_name": "qwen3vl_8b_bf16.safetensors", "type": "qwen_image", "device": "cuda:1"}},
        "4": {"class_type": "VAELoaderMultiGPU",
              "inputs": {"vae_name": "qwen_image_2.1_vae_bf16.safetensors", "device": "cuda:0"}},
        "5": {"class_type": "TextEncodeQwenImage21",
              "inputs": {"clip": ["3", 0], "prompt": prompt, "negative_prompt": "",
                         "resolution": 1024, "images": []}},
        "6": {"class_type": "EmptyLatentImage", "inputs": {"width": w, "height": h, "batch_size": 1}},
        "7": {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}},
        "8": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}},
        "9": {"class_type": "ManualSigmas",
              "inputs": {"sigmas": "1.0, 0.978453, 0.95418, 0.926626, 0.89508, 0.845148, 0.704534, 0.414568, 0"}},
        "10": {"class_type": "CFGGuider",
               "inputs": {"model": ["2", 0], "positive": ["5", 0], "negative": ["5", 1], "cfg": 1.0}},
        "11": {"class_type": "SamplerCustomAdvanced",
               "inputs": {"noise": ["7", 0], "guider": ["10", 0], "sampler": ["8", 0],
                          "sigmas": ["9", 0], "latent_image": ["6", 0]}},
        "12": {"class_type": "VAEDecode", "inputs": {"samples": ["11", 0], "vae": ["4", 0]}},
        "13": {"class_type": "SaveImage", "inputs": {"images": ["12", 0], "filename_prefix": "hub_qwen"}},
    }


def wf_zimage(prompt, w, h, seed):
    return {
        "1": {"class_type": "UNETLoaderMultiGPU",
              "inputs": {"unet_name": "z_image_turbo_bf16.safetensors", "weight_dtype": "default", "device": "cuda:0"}},
        "2": {"class_type": "CLIPLoaderMultiGPU",
              "inputs": {"clip_name": "qwen_3_4b.safetensors", "type": "lumina2", "device": "cuda:1"}},
        "3": {"class_type": "VAELoaderMultiGPU",
              "inputs": {"vae_name": "ae.safetensors", "device": "cuda:0"}},
        "4": {"class_type": "ModelSamplingAuraFlow", "inputs": {"model": ["1", 0], "shift": 3}},
        "5": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["2", 0], "text": prompt}},
        "6": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["2", 0], "text": ""}},
        "7": {"class_type": "EmptySD3LatentImage", "inputs": {"width": w, "height": h, "batch_size": 1}},
        "8": {"class_type": "KSampler",
              "inputs": {"model": ["4", 0], "positive": ["5", 0], "negative": ["6", 0],
                         "latent_image": ["7", 0], "seed": seed, "steps": 8, "cfg": 1.0,
                         "sampler_name": "res_multistep", "scheduler": "simple", "denoise": 1.0}},
        "9": {"class_type": "VAEDecode", "inputs": {"samples": ["8", 0], "vae": ["3", 0]}},
        "10": {"class_type": "SaveImage", "inputs": {"images": ["9", 0], "filename_prefix": "hub_zimage"}},
    }


def wf_moss(prompt, w, h, seed):
    return {
        "1": {"class_type": "UNETLoaderMultiGPU",
              "inputs": {"unet_name": "moss_image_sft_bf16.safetensors", "weight_dtype": "default", "device": "cuda:0"}},
        "2": {"class_type": "CLIPLoaderMultiGPU",
              "inputs": {"clip_name": "qwen3.5_9b_bf16.safetensors", "type": "moss_image", "device": "cuda:1"}},
        "3": {"class_type": "VAELoaderMultiGPU",
              "inputs": {"vae_name": "flux2-vae.safetensors", "device": "cuda:0"}},
        "4": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["2", 0], "text": prompt}},
        "5": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["2", 0], "text": ""}},
        "6": {"class_type": "EmptyFlux2LatentImage", "inputs": {"width": w, "height": h, "batch_size": 1}},
        "7": {"class_type": "KSampler",
              "inputs": {"model": ["1", 0], "positive": ["4", 0], "negative": ["5", 0],
                         "latent_image": ["6", 0], "seed": seed, "steps": 25, "cfg": 2.5,
                         "sampler_name": "er_sde", "scheduler": "simple", "denoise": 1.0}},
        "8": {"class_type": "VAEDecode", "inputs": {"samples": ["7", 0], "vae": ["3", 0]}},
        "9": {"class_type": "SaveImage", "inputs": {"images": ["8", 0], "filename_prefix": "hub_moss"}},
    }


def wf_flux(prompt, w, h, seed):
    return {
        "1": {"class_type": "UnetLoaderGGUFMultiGPU",
              "inputs": {"unet_name": "flux-3-image-klein-Q8_0.gguf", "device": "cuda:0"}},
        "2": {"class_type": "CLIPLoaderMultiGPU",
              "inputs": {"clip_name": "qwen3-vl-4b-heretic.safetensors", "type": "flux3", "device": "cuda:1"}},
        "3": {"class_type": "VAELoaderMultiGPU",
              "inputs": {"vae_name": "flux2-vae.safetensors", "device": "cuda:0"}},
        "4": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["2", 0], "text": prompt}},
        "5": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["2", 0], "text": ""}},
        "6": {"class_type": "EmptyFlux2LatentImage", "inputs": {"width": w, "height": h, "batch_size": 1}},
        "7": {"class_type": "KSampler",
              "inputs": {"model": ["1", 0], "positive": ["4", 0], "negative": ["5", 0],
                         "latent_image": ["6", 0], "seed": seed, "steps": 15, "cfg": 1.0,
                         "sampler_name": "er_sde", "scheduler": "simple", "denoise": 1.0}},
        "8": {"class_type": "VAEDecode", "inputs": {"samples": ["7", 0], "vae": ["3", 0]}},
        "9": {"class_type": "SaveImage", "inputs": {"images": ["8", 0], "filename_prefix": "hub_flux"}},
    }


def wf_qwen_uc(prompt, w, h, seed):
    """无限制版：与 Qwen-Image-2.1 相同的 DiT，仅替换为 abliterated 文本编码器"""
    wf = wf_qwen(prompt, w, h, seed)
    wf["3"]["inputs"]["clip_name"] = "qwen3vl_8b_uncensored.safetensors"
    return wf


MODEL_NEEDS = {
    "qwen_uc": ["/opt/ComfyUI/models/text_encoders/qwen3vl_8b_uncensored.safetensors"],
}

MODELS = {
    "h3": {"name": "h3", "label": "MiniMax H3-Image", "size": (2048, 2048), "warm": (1024, 1024), "wf": wf_h3,
           "note": "质量第一 Elo1066，6步，中文文字最强",
           "gpus": "GPU2+GPU3 · DiT@2 编码器@3"},
    "qwen": {"name": "qwen", "label": "Qwen-Image-2.1-Turbo", "size": (1024, 1024), "warm": (640, 640), "wf": wf_qwen,
             "note": "Elo1047，8步，中文排版强",
             "gpus": "GPU2+GPU3 · DiT@2 编码器@3"},
    "qwen_uc": {"name": "qwen_uc", "label": "Qwen-Image-2.1 (无限制)", "size": (1024, 1024),
                "warm": (640, 640), "wf": wf_qwen_uc,
                "note": "abliterated 编码器，画质同原版，限制更少"},
    "zimage": {"name": "zimage", "label": "Z-Image-Turbo", "size": (1024, 1024), "warm": (640, 640), "wf": wf_zimage,
               "note": "6B 极速模型，8步",
               "gpus": "GPU2+GPU3 · DiT@2 编码器@3"},
    "moss": {"name": "moss", "label": "MOSS-Image-SFT", "size": (1024, 1024), "warm": (640, 640), "wf": wf_moss,
             "note": "布局/海报专精，25步",
             "gpus": "GPU2+GPU3 · DiT@2 编码器@3"},
}


# 恢复上次驻留模型（BFF 重启后 auto 仍可用）
try:
    _state["loaded"] = _load_state()
    if _state["loaded"]:
        logger.info(f"恢复驻留模型状态: {_state['loaded']}")
except Exception:
    pass


# ==================== 工具函数 ====================
def db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""CREATE TABLE IF NOT EXISTS jobs(
        id TEXT PRIMARY KEY, prompt TEXT, models TEXT, created REAL,
        results TEXT)""")
    try:
        conn.execute("ALTER TABLE jobs ADD COLUMN status TEXT DEFAULT 'done'")
        conn.commit()
    except Exception:
        pass
    # 启动时把上次未完成的任务标记为中断
    conn.execute("UPDATE jobs SET status='interrupted' WHERE status='running'")
    conn.commit()
    return conn


def comfy_post(path, payload):
    req = urllib.request.Request(COMFY + path, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())


def _comfy_alive(timeout=2):
    try:
        with urllib.request.urlopen(f"{COMFY}/system_stats", timeout=timeout) as r:
            return r.status == 200
    except Exception:
        return False


def ensure_comfy(timeout=240):
    """确保 ComfyUI 在运行；未运行时通过 systemd 拉起（ImageHub 统一管理图像服务）"""
    if _comfy_alive():
        return True
    logger.info("ComfyUI 未运行，正在启动 comfyui-h3.service ...")
    try:
        subprocess.run(["systemctl", "start", "comfyui-h3.service"],
                       capture_output=True, timeout=60)
    except Exception as e:
        logger.warning(f"启动 ComfyUI 失败: {e}")
        return False
    t0 = time.time()
    while time.time() - t0 < timeout:
        time.sleep(2)
        if _comfy_alive():
            logger.info(f"ComfyUI 已就绪（{time.time()-t0:.0f}s）")
            return True
    logger.error("ComfyUI 启动超时")
    return False


def comfy_free():
    """卸载 ComfyUI 已加载模型（仅在切换模型时调用）"""
    try:
        comfy_post("/free", {"unload_models": True, "free_memory": True})
    except Exception:
        pass
    time.sleep(3)


def _submit_wait(wf, timeout=1800):
    """提交工作流并等待完成，返回 (ok, images, error, duration)"""
    t0 = time.time()
    try:
        pid = comfy_post("/prompt", {"prompt": wf})["prompt_id"]
    except Exception as e:
        return False, [], f"提交失败: {e}", 0
    while time.time() - t0 < timeout:
        time.sleep(2)
        try:
            with urllib.request.urlopen(f"{COMFY}/history/{pid}", timeout=10) as r:
                hist = json.loads(r.read())
        except Exception:
            continue
        if pid in hist:
            st = hist[pid].get("status", {})
            if st.get("status_str") == "error":
                msg = ""
                for m in st.get("messages", []):
                    if m[0] == "execution_error":
                        msg = m[1].get("exception_message", "")[:300]
                return False, [], msg or "执行错误", round(time.time() - t0, 1)
            imgs = [im.get("filename") for n in hist[pid].get("outputs", {}).values()
                    for im in n.get("images", [])]
            return True, imgs, "", round(time.time() - t0, 1)
    return False, [], "timeout", timeout


def _warm_load(model_key):
    """加载模型（用小尺寸 1 步跑通），不生成用户可见图"""
    spec = MODELS[model_key]
    ww, wh = spec.get("warm", (640, 640))
    wf = spec["wf"]("warmup", ww, wh, 1)
    ok, _, err, _ = _submit_wait(wf, timeout=900)
    if not ok:
        raise RuntimeError(f"加载 {model_key} 失败: {err}")


def acquire_model(model_key):
    """获取模型使用权：同模型并行、跨模型排队；仅在没有生成任务时切换模型"""
    with _cond:
        while True:
            if _state["switching"]:
                _cond.wait(); continue
            if _state["loaded"] == model_key:
                _state["active"] += 1
                return
            if _state["active"] > 0:
                _cond.wait(); continue
            _state["switching"] = True
            _state["loaded"] = None
            _state["target"] = model_key
            break
    ok = False
    try:
        if not ensure_comfy():
            raise RuntimeError("ComfyUI 未就绪")
        comfy_free()
        _warm_load(model_key)
        ok = True
    finally:
        with _cond:
            _state["switching"] = False
            _state["target"] = None
            if ok:
                _state["loaded"] = model_key
                _state["active"] += 1
            _cond.notify_all()
        if ok:
            _save_state(model_key)
    if not ok:
        raise RuntimeError(f"模型 {model_key} 加载失败")


def release_model():
    with _cond:
        _state["active"] = max(0, _state["active"] - 1)
        _cond.notify_all()


def run_model(model_key, prompt, width, height, seed):
    spec = MODELS[model_key]
    w, h = (width or spec["size"][0]), (height or spec["size"][1])
    try:
        acquire_model(model_key)
    except Exception as e:
        return {"model": model_key, "status": "error", "error": str(e)[:300], "duration": 0}
    try:
        logger.info(f"[{model_key}] prompt={prompt[:60]!r} size={w}x{h} seed={seed}")
        wf = spec["wf"](prompt, w, h, seed)
        ok, imgs, err, dur = _submit_wait(wf)
        if not ok:
            return {"model": model_key, "status": "error", "error": err, "duration": dur}
        return {"model": model_key, "status": "ok",
                "image": imgs[0] if imgs else None,
                "duration": dur, "size": f"{w}x{h}"}
    finally:
        release_model()


def run_job(job_id, prompt, model_keys, width, height, seed):
    results = []
    for mk in model_keys:
        try:
            results.append(run_model(mk, prompt, width, height, seed))
        except Exception as e:
            results.append({"model": mk, "status": "error", "error": str(e)[:300]})
        with _lock:
            _jobs[job_id]["results"] = list(results)
        try:
            conn = db()
            conn.execute("INSERT OR REPLACE INTO jobs VALUES (?,?,?,?,?,?)",
                         (job_id, prompt, json.dumps(model_keys), time.time(),
                          json.dumps(results), "running"))
            conn.commit()
            conn.close()
        except Exception:
            pass
    with _lock:
        _jobs[job_id]["status"] = "done"
        conn = db()
        conn.execute("INSERT OR REPLACE INTO jobs VALUES (?,?,?,?,?,?)",
                     (job_id, prompt, json.dumps(model_keys), time.time(),
                      json.dumps(results), "done"))
        conn.commit()
        conn.close()



# ==================== GPU 功耗控制 ====================
def _power_info():
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,power.limit,power.default_limit,power.min_limit,power.max_limit",
         "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10).stdout.strip()
    rows = []
    for line in out.split("\n"):
        p = [x.strip() for x in line.split(",")]
        if len(p) >= 5:
            rows.append({"index": int(p[0]), "limit": int(float(p[1])), "default": int(float(p[2])),
                         "min": int(float(p[3])), "max": int(float(p[4]))})
    return rows


class PowerReq(BaseModel):
    gpu: str = "all"          # "all" 或 "0".."3"
    limit: Optional[int] = None   # None = 恢复默认


@app.get("/api/power")
def api_power():
    return _power_info()


@app.post("/api/power")
def api_set_power(req: PowerReq):
    info = {r["index"]: r for r in _power_info()}
    targets = list(info.keys()) if str(req.gpu) == "all" else [int(req.gpu)]
    changed = []
    for i in targets:
        if i not in info:
            continue
        g = info[i]
        val = g["default"] if req.limit is None else int(req.limit)
        if not (g["min"] <= val <= g["max"]):
            raise HTTPException(400, f"GPU{i} 功耗需在 {g['min']}~{g['max']}W 之间")
        r = subprocess.run(["nvidia-smi", "-i", str(i), "-pl", str(val)],
                           capture_output=True, text=True, timeout=20)
        if r.returncode != 0:
            raise HTTPException(500, f"GPU{i} 设置失败: {(r.stderr or r.stdout)[:200]}")
        changed.append({"index": i, "limit": val})
    logger.info(f"功耗调整: {changed}")
    return {"ok": True, "changed": changed, "current": _power_info()}


# ==================== OpenAI 兼容接口 ====================
MODEL_ALIASES = {
    "h3": "h3", "h3-image": "h3", "minimax-h3": "h3", "minimax_h3": "h3",
    "qwen": "qwen", "qwen-image": "qwen", "qwen-image-2.1": "qwen",
    "qwen-image-2.1-turbo": "qwen",
    "qwen_uc": "qwen_uc", "qwen-image-2.1-uncensored": "qwen_uc", "qwen-uncensored": "qwen_uc",
    "zimage": "zimage", "z-image": "zimage", "z-image-turbo": "zimage",
    "moss": "moss", "moss-image": "moss",
}


class OAIImageRequest(BaseModel):
    prompt: str
    model: Optional[str] = "h3"
    only_loaded: bool = False   # true=只用当前已加载模型，不触发切换
    n: int = 1
    size: str = "1024x1024"
    response_format: str = "b64_json"
    seed: Optional[int] = None


@app.get("/v1/models")
def v1_models():
    return {"object": "list",
            "data": [{"id": k, "object": "model", "owned_by": "imagehub",
                      "label": v["label"], "default_size": list(v["size"])}
                     for k, v in MODELS.items()]}


@app.post("/v1/images/generations")
def v1_generate(req: OAIImageRequest):
    raw_key = (req.model or "auto").strip()
    lowered = raw_key.lower()
    # model=auto/loaded/current：直接使用当前已加载模型
    if lowered in ("auto", "loaded", "current"):
        with _cond:
            cur = _state["loaded"]
        if cur:
            key = cur                      # 有已加载模型 → 直接用
        elif req.only_loaded:
            raise HTTPException(503, {
                "error": "当前没有已加载的模型（only_loaded=true 不加载）",
                "hint": "去掉 only_loaded 或显式指定 model",
                "loaded": None,
            })
        else:
            key = "h3"                     # 无加载 → 默认用 h3
            logger.info("auto: 无已加载模型，回落到 h3")
    else:
        key = MODEL_ALIASES.get(lowered, raw_key)
        if key not in MODELS:
            raise HTTPException(400, f"未知模型 {raw_key}，可用: {list(MODELS)} + auto")
        if not all(os.path.exists(p) for p in MODEL_NEEDS.get(key, [])):
            raise HTTPException(503, {"error": f"模型 {key} 权重尚未就绪",
                                      "hint": "下载完成后即可使用"})
        if req.only_loaded:
            with _cond:
                cur = _state["loaded"]
            if cur != key:
                raise HTTPException(409, {
                    "error": f"当前已加载的是 {cur or '无'}，未加载 {key}",
                    "hint": "去掉 only_loaded 会自动切换（约 20~40 秒）",
                    "loaded": cur,
                    "requested": key,
                })
    try:
        w, h = [int(x) for x in req.size.lower().split("x")]
    except Exception:
        raise HTTPException(400, "size 格式应为 1024x1024")
    res = run_model(key, req.prompt, w, h, req.seed if req.seed is not None else 42)
    if res.get("status") != "ok" or not res.get("image"):
        raise HTTPException(500, res.get("error", "生成失败"))
    import base64 as _b64
    data = []
    if req.response_format == "b64_json":
        path = os.path.join(COMFY_OUTPUT, res["image"])
        with open(path, "rb") as f:
            data.append({"b64_json": _b64.b64encode(f.read()).decode(),
                         "revised_prompt": req.prompt})
    else:
        data.append({"url": f"/image/{res['image']}", "revised_prompt": req.prompt})
    return {"created": int(time.time()), "data": data,
            "model": key, "duration_s": res.get("duration"),
            "size": res.get("size")}


# ==================== 图片编辑（Qwen-Image-Edit-2511） ====================
COMFY_INPUT = "/opt/ComfyUI/input"
EDIT_LABEL = "Qwen-Image-Edit-2511"
EDIT_NEEDS = ["/opt/ComfyUI/models/diffusion_models/qwen-image-edit-2511-Q6_K.gguf"]


def wf_qwen_edit(prompt, image_name, w, h, seed, steps=40, cfg=4.0):
    return {
        "1": {"class_type": "UnetLoaderGGUFMultiGPU",
              "inputs": {"unet_name": "qwen-image-edit-2511-Q6_K.gguf", "device": "cuda:0"}},
        "2": {"class_type": "CLIPLoaderMultiGPU",
              "inputs": {"clip_name": "qwen_2.5_vl_7b_fp8_scaled.safetensors",
                         "type": "qwen_image", "device": "cuda:1"}},
        "3": {"class_type": "VAELoaderMultiGPU",
              "inputs": {"vae_name": "wan_2.1_vae.safetensors", "device": "cuda:0"}},
        "4": {"class_type": "LoadImage", "inputs": {"image": image_name}},
        "5": {"class_type": "ImageScaleToTotalPixels",
              "inputs": {"image": ["4", 0], "upscale_method": "lanczos",
                         "megapixels": 2.0, "resolution_steps": 16}},
        "6": {"class_type": "TextEncodeQwenImageEditPlus",
              "inputs": {"clip": ["2", 0], "prompt": prompt, "vae": ["3", 0],
                         "image1": ["5", 0]}},
        "7": {"class_type": "TextEncodeQwenImageEditPlus",
              "inputs": {"clip": ["2", 0], "prompt": "", "vae": ["3", 0],
                         "image1": ["5", 0]}},
        "8": {"class_type": "ModelSamplingAuraFlow", "inputs": {"model": ["1", 0], "shift": 3}},
        "9": {"class_type": "EmptyLatentImage", "inputs": {"width": w, "height": h, "batch_size": 1}},
        "10": {"class_type": "KSampler",
               "inputs": {"model": ["8", 0], "positive": ["6", 0], "negative": ["7", 0],
                          "latent_image": ["9", 0], "seed": seed, "steps": steps, "cfg": cfg,
                          "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0}},
        "11": {"class_type": "VAEDecode", "inputs": {"samples": ["10", 0], "vae": ["3", 0]}},
        "12": {"class_type": "SaveImage", "inputs": {"images": ["11", 0], "filename_prefix": "hub_edit"}},
    }


class EditReq(BaseModel):
    prompt: str
    image_b64: str
    seed: Optional[int] = None
    steps: int = 40


@app.post("/api/edit")
def api_edit(req: EditReq):
    if not all(os.path.exists(p) for p in EDIT_NEEDS):
        raise HTTPException(503, {"error": "编辑模型权重尚未就绪", "hint": "GGUF 下载完成后可用"})
    import base64
    raw = req.image_b64.split(",")[-1]
    try:
        data = base64.b64decode(raw)
    except Exception:
        raise HTTPException(400, "图片 base64 解析失败")
    os.makedirs(COMFY_INPUT, exist_ok=True)
    name = f"hub_edit_{uuid.uuid4().hex[:8]}.png"
    with open(os.path.join(COMFY_INPUT, name), "wb") as f:
        f.write(data)
    if not ensure_comfy():
        raise HTTPException(503, "ComfyUI 未就绪")
    comfy_free()
    t0 = time.time()
    wf = wf_qwen_edit(req.prompt, name, 1024, 1024, req.seed if req.seed is not None else 42, req.steps)
    try:
        pid = comfy_post("/prompt", {"prompt": wf})["prompt_id"]
    except Exception as e:
        raise HTTPException(500, f"提交失败: {e}")
    while time.time() - t0 < 900:
        time.sleep(3)
        try:
            with urllib.request.urlopen(f"{COMFY}/history/{pid}", timeout=10) as r:
                hist = json.loads(r.read())
        except Exception:
            continue
        if pid in hist:
            st = hist[pid].get("status", {})
            if st.get("status_str") == "error":
                msg = ""
                for m in st.get("messages", []):
                    if m[0] == "execution_error":
                        msg = m[1].get("exception_message", "")[:300]
                return {"status": "error", "error": msg, "duration": round(time.time() - t0, 1)}
            imgs = []
            for node in hist[pid].get("outputs", {}).values():
                for im in node.get("images", []):
                    imgs.append(im.get("filename"))
            _state["loaded"] = None
            return {"status": "ok", "image": imgs[0] if imgs else None,
                    "duration": round(time.time() - t0, 1), "model": EDIT_LABEL}
    return {"status": "error", "error": "timeout"}


@app.get("/api/edit/model")
def api_edit_model():
    return {"key": "qwen_edit", "label": EDIT_LABEL,
            "note": "上传图片 + 指令进行编辑（40 步）",
            "ready": all(os.path.exists(p) for p in EDIT_NEEDS)}


@app.get("/api/edit/ready")
def api_edit_ready():
    return {"ready": all(os.path.exists(p) for p in EDIT_NEEDS)}


# ==================== API ====================
class CompareReq(BaseModel):
    prompt: str
    models: List[str]
    width: Optional[int] = None
    height: Optional[int] = None
    seed: Optional[int] = None


@app.get("/api/models")
def api_models():
    out = []
    for k, v in MODELS.items():
        needs = MODEL_NEEDS.get(k, [])
        ready = all(os.path.exists(p) for p in needs)
        out.append({"key": k, "label": v["label"], "default_size": list(v["size"]),
                    "note": v["note"], "gpus": v.get("gpus", "GPU2+GPU3"),
                    "ready": ready})
    return out


@app.post("/api/compare")
def api_compare(req: CompareReq):
    keys = [m for m in req.models if m in MODELS]
    if not keys:
        raise HTTPException(400, "未选择有效模型")
    skipped = [k for k in keys if not all(os.path.exists(p) for p in MODEL_NEEDS.get(k, []))]
    keys = [k for k in keys if k not in skipped]
    if not keys:
        raise HTTPException(503, {"error": "所选模型均未就绪", "skipped": skipped})
    job_id = uuid.uuid4().hex[:12]
    with _lock:
        _jobs[job_id] = {"id": job_id, "status": "running", "prompt": req.prompt,
                         "models": keys, "results": [], "created": time.time(),
                         "skipped": skipped}
    seed = req.seed if req.seed is not None else 42
    threading.Thread(target=run_job, daemon=True,
                     args=(job_id, req.prompt, keys, req.width, req.height, seed)).start()
    return {"job_id": job_id}


@app.get("/api/jobs/{job_id}")
def api_job(job_id: str):
    j = _jobs.get(job_id)
    if not j:
        conn = db()
        row = conn.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
        conn.close()
        if not row:
            raise HTTPException(404, "job not found")
        return {"id": row[0], "prompt": row[1], "models": json.loads(row[2]),
                "created": row[3], "results": json.loads(row[4]), "status": "done"}
    return j


class LaunchReq(BaseModel):
    model: str


@app.post("/api/launch")
def api_launch(req: LaunchReq):
    if req.model not in MODELS:
        raise HTTPException(400, "未知模型")
    if not all(os.path.exists(p) for p in MODEL_NEEDS.get(req.model, [])):
        raise HTTPException(503, {"error": f"模型 {req.model} 权重尚未就绪",
                                  "hint": "下载完成后可用"})
    spec = MODELS[req.model]
    ww, wh = spec.get("warm", (640, 640))
    threading.Thread(target=run_model, daemon=True,
                     args=(req.model, "warmup", ww, wh, 1)).start()
    return {"status": "launching", "model": req.model, "size": f"{ww}x{wh}"}


@app.get("/api/gpu")
def api_gpu():
    import subprocess
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.used,memory.total,power.draw,power.limit,temperature.gpu", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5).stdout.strip()
        rows = []
        for line in out.split("\n"):
            parts = [x.strip() for x in line.split(",")]
            if len(parts) >= 6 and int(parts[0]) in (2, 3):
                rows.append({"index": int(parts[0]), "used": int(parts[1]), "total": int(parts[2]),
                             "power_draw": int(float(parts[3])), "power_limit": int(float(parts[4])),
                             "temp": int(parts[5])})
        return rows
    except Exception:
        return []


@app.get("/api/version")
def api_version():
    return {"name": "ImageHub", "version": APP_VERSION, "build": APP_BUILD}


@app.get("/api/status")
def api_status():
    with _cond:
        return {"loaded": _state["loaded"], "active": _state["active"],
                "switching": _state["switching"], "target": _state["target"],
                "version": APP_VERSION}


@app.get("/api/history")
def api_history(limit: int = 30):
    conn = db()
    rows = conn.execute("SELECT id,prompt,models,created,results,status FROM jobs ORDER BY created DESC LIMIT ?",
                        (limit,)).fetchall()
    conn.close()
    out = []
    for r in rows:
        try:
            results = json.loads(r[4]) if r[4] else []
        except Exception:
            results = []
        out.append({"id": r[0], "prompt": r[1], "models": json.loads(r[2]),
                    "created": r[3], "status": r[5] or "done",
                    "images": [{"model": x.get("model"), "image": x.get("image"),
                                "duration": x.get("duration")}
                               for x in results if x.get("status") == "ok" and x.get("image")]})
    return out


@app.delete("/api/history")
def api_clear_history():
    """清空全部历史记录及其图片"""
    conn = db()
    rows = conn.execute("SELECT results FROM jobs").fetchall()
    conn.execute("DELETE FROM jobs")
    conn.commit()
    conn.close()
    removed = 0
    for (raw,) in rows:
        try:
            results = json.loads(raw or "[]")
        except Exception:
            continue
        for r in results:
            img = r.get("image") if isinstance(r, dict) else None
            if img:
                path = os.path.join(COMFY_OUTPUT, os.path.basename(img))
                try:
                    os.remove(path)
                    removed += 1
                except Exception:
                    pass
    _jobs.clear()
    logger.info(f"清空历史：{len(rows)} 条记录，{removed} 张图片")
    return {"ok": True, "deleted_jobs": len(rows), "deleted_images": removed}


class BatchDeleteReq(BaseModel):
    ids: List[str]


@app.post("/api/history/batch-delete")
def api_batch_delete(req: BatchDeleteReq):
    """批量删除选中的历史记录及图片"""
    conn = db()
    removed = 0
    deleted = 0
    for jid in req.ids:
        row = conn.execute("SELECT results FROM jobs WHERE id=?", (jid,)).fetchone()
        if not row:
            continue
        try:
            results = json.loads(row[0] or "[]")
        except Exception:
            results = []
        conn.execute("DELETE FROM jobs WHERE id=?", (jid,))
        deleted += 1
        for r in results:
            img = r.get("image") if isinstance(r, dict) else None
            if img:
                path = os.path.join(COMFY_OUTPUT, os.path.basename(img))
                try:
                    os.remove(path)
                    removed += 1
                except Exception:
                    pass
        _jobs.pop(jid, None)
    conn.commit()
    conn.close()
    logger.info(f"批量删除：{deleted} 条记录，{removed} 张图片")
    return {"ok": True, "deleted_jobs": deleted, "deleted_images": removed}


@app.delete("/api/history/{job_id}")
def api_delete_history(job_id: str):
    """删除一条对比记录（同时删除其生成的图片文件）"""
    conn = db()
    row = conn.execute("SELECT results FROM jobs WHERE id=?", (job_id,)).fetchone()
    if not row:
        conn.close()
        raise HTTPException(404, "记录不存在")
    try:
        results = json.loads(row[0] or "[]")
    except Exception:
        results = []
    conn.execute("DELETE FROM jobs WHERE id=?", (job_id,))
    conn.commit()
    conn.close()
    removed = 0
    for r in results:
        img = r.get("image") if isinstance(r, dict) else None
        if img:
            path = os.path.join(COMFY_OUTPUT, os.path.basename(img))
            try:
                os.remove(path)
                removed += 1
            except Exception:
                pass
    _jobs.pop(job_id, None)
    logger.info(f"删除记录 {job_id}，图片 {removed} 张")
    return {"ok": True, "deleted_images": removed}


@app.get("/image/{filename}")
def image(filename: str):
    path = os.path.join(COMFY_OUTPUT, os.path.basename(filename))
    if not os.path.isfile(path):
        raise HTTPException(404, "not found")
    return FileResponse(path)


def _verify_ticket(ticket: str):
    """调用 yz-login 验证 ticket（带应用身份）"""
    url = (f"{YZ_LOGIN_URL}/api/ticket/verify?ticket={quote(ticket)}"
           f"&app_id={YZ_APP_ID}&app_secret={quote(YZ_APP_SECRET)}")
    try:
        with urllib.request.urlopen(url, timeout=10) as r:
            data = json.loads(r.read())
        if data.get("ok"):
            return {"id": data.get("id"), "username": data.get("username"),
                    "display_name": data.get("display_name"),
                    "is_admin": data.get("is_admin")}
    except Exception as e:
        logger.warning(f"ticket 验证失败: {e}")
    return None


def _do_login(request: Request):
    user = _verify_ticket(request.query_params.get("ticket", ""))
    if not user:
        return HTMLResponse("<h3>登录验证失败，请重试</h3><a href='/'>返回</a>", status_code=403)
    request.session["user"] = user
    return RedirectResponse("/")


@app.get("/callback")
def callback(request: Request):
    """yz-login 回调（兼容注册 URL 配成 / 或 /callback）"""
    return _do_login(request)


@app.get("/logout")
def logout(request: Request):
    request.session.clear()
    return RedirectResponse(f"{YZ_LOGIN_URL}/logout?from=id:{YZ_APP_ID}")


@app.get("/api/whoami")
def whoami(request: Request):
    """当前登录用户（网页用，API 无需登录）"""
    return {"sso_enabled": SSO_ENABLED, "user": request.session.get("user")}


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    # 陷阱一：注册 URL 可能是 / 或 /callback，这里都处理 ticket
    if request.query_params.get("ticket"):
        return _do_login(request)
    user = request.session.get("user")
    # 陷阱二：只在页面入口做登录跳转，/api 与 /image 完全放行
    if SSO_ENABLED and not user:
        return RedirectResponse(f"{YZ_LOGIN_URL}/login?from=id:{YZ_APP_ID}")
    with open(os.path.join(BASE_DIR, "index.html"), encoding="utf-8") as f:
        html = f.read()
    if user:
        chip = (f'<span class="userchip">👤 {user.get("display_name") or user.get("username")}'
                f' · <a href="/logout">退出</a></span>')
    else:
        chip = ""
    html = html.replace("<!--USER-->", chip)
    return HTMLResponse(html, headers={
        "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
        "Pragma": "no-cache",
        "Expires": "0",
    })
