#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ImageHub BFF：直连 diffusers 图像服务（不再经过 ComfyUI）
  POST /api/compare  {prompt, models:[...], width?, height?, seed?}
  GET  /api/jobs/{id}
  GET  /api/models
  GET  /api/history
  GET  /image/{filename}

后端服务（同一时间只驻留一个，共享 GPU2/3）：
  qwen   5566  Qwen-Image-2.1-Turbo   (CUDA 2,3)
  zimage 5565  Z-Image-Turbo          (CUDA 2)
  moss   5567  MOSS-Image-SFT         (CUDA 2,3 split)
  edit   5564  Qwen-Image-Edit-2511   (CUDA 2,3 split)
"""
import base64
import json
import logging
import os
import re
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

APP_VERSION = "1.4.0"          # ImageHub 版本号（1.4.0：直连后端，移除 ComfyUI）
APP_BUILD = "2026-11-20"

logger = logging.getLogger("imagehub")
logging.basicConfig(level=logging.INFO)

OUTPUT_DIR = os.environ.get("IMAGEHUB_OUTPUT", "/opt/vLLM/imagehub/output")
os.makedirs(OUTPUT_DIR, exist_ok=True)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "imagehub.db")
RUN_DIR = os.path.join(BASE_DIR, "run")
os.makedirs(RUN_DIR, exist_ok=True)
PY_BIN = os.environ.get("IMAGEHUB_PY", "/opt/conda3/envs/imagegen/bin/python")

# ComfyUI 后端（可切换）
COMFY = "http://127.0.0.1:5568"
COMFY_OUTPUT = "/opt/ComfyUI/output"
COMFY_INPUT = "/opt/ComfyUI/input"

SETTINGS_FILE = os.path.join(BASE_DIR, "settings.json")
_DEF_SETTINGS = {"backend": "direct", "gate": False, "clocks": False}


def _load_settings():
    try:
        with open(SETTINGS_FILE, encoding="utf-8") as f:
            d = json.load(f)
        out = dict(_DEF_SETTINGS)
        for k in _DEF_SETTINGS:
            if k in d:
                out[k] = d[k]
        return out
    except Exception:
        return dict(_DEF_SETTINGS)


def _save_settings():
    try:
        with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
            json.dump(_settings, f, ensure_ascii=False)
    except Exception:
        pass


def _clocks_set(on):
    """锁定/恢复 GPU 频率（约基础频率 1410MHz），平滑电流阶跃"""
    for i in range(4):
        cmd = (["nvidia-smi", "-i", str(i), "-lgc", "1410"] if on
               else ["nvidia-smi", "-i", str(i), "-rgc"])
        try:
            subprocess.run(cmd, capture_output=True, timeout=10)
        except Exception:
            pass


_settings = _load_settings()

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


_state = {"loaded": None, "comfy_loaded": None, "active": 0, "switching": False, "target": None}
_cond = threading.Condition()


# ==================== 直连后端服务 ====================
# 每个服务是一个独立 uvicorn 进程（FastAPI + diffusers），端口固定。
# GPU2/3 显存有限，同一时间只驻留一个服务，切换 = 停旧进程 + 起新进程。
SERVICES = {
    "qwen": {
        "port": 5566, "cwd": "/opt/models/Qwen/Qwen-Image-2.1-Turbo",
        "app": "qwenimage_server:app",
        "env": {"CUDA_VISIBLE_DEVICES": "2,3", "QIMAGE_VAE_TILING": "1"},
        "needs": ["/opt/models/Qwen/Qwen-Image-2.1-Turbo/model_index.json"],
    },
    "zimage": {
        "port": 5565, "cwd": "/opt/models/Tongyi-MAI/Z-Image-Turbo",
        "app": "zimage_server:app", "env": {"CUDA_VISIBLE_DEVICES": "2"},
        "needs": ["/opt/models/Tongyi-MAI/Z-Image-Turbo/model_index.json"],
    },
    "moss": {
        "port": 5567, "cwd": "/opt/models/OpenMOSS/MOSS-Image-SFT",
        "app": "moss_server:app", "env": {"CUDA_VISIBLE_DEVICES": "2,3", "MOSS_SPLIT": "1"},
        "needs": ["/opt/models/OpenMOSS/MOSS-Image-SFT/model_index.json"],
    },
    "edit": {
        "port": 5564, "cwd": "/opt/models/Qwen/Qwen-Image-Edit-2511",
        "app": "qwen_edit_server:app",
        "env": {"CUDA_VISIBLE_DEVICES": "2,3", "EDIT_MODE": "split"},
        "needs": [
            "/opt/models/Qwen/Qwen-Image-Edit-2511/model_index.json",
            "/opt/models/Comfy-Org/Qwen-Image-Edit-2511-gguf/qwen-image-edit-2511-Q6_K.gguf",
        ],
    },
    "h3": {
        "port": 5568, "cwd": "/opt/sd.cpp",
        "cmd": ["/opt/sd.cpp/build/bin/sd-server", "--listen-ip", "0.0.0.0",
                "--listen-port", "5568",
                "--diffusion-model", "/opt/models/Unsloth/MiniMax-H3-Image-GGUF/h3_ud_q6k_xl.gguf",
                "--llm", "/opt/models/Unsloth/MiniMax-H3-GGUF/qwen3vl_32b_minimax_h3-Q4_K_M.gguf",
                "--vae", "/opt/models/MiniMax/MiniMax-H3-Image/vae/diffusion_pytorch_model.safetensors",
                "--backend", "diffusion=CUDA0,llm=CUDA1,vae=CUDA1"],
        "env": {"CUDA_VISIBLE_DEVICES": "2,3"}, "health_path": "/v1/models", "api": "sdcpp",
        "needs": ["/opt/models/Unsloth/MiniMax-H3-Image-GGUF/h3_ud_q6k_xl.gguf",
                  "/opt/models/Unsloth/MiniMax-H3-GGUF/qwen3vl_32b_minimax_h3-Q4_K_M.gguf",
                  "/opt/models/MiniMax/MiniMax-H3-Image/vae/diffusion_pytorch_model.safetensors"],
    },
}

def _model_ready(model_key):
    """模型是否可用：服务已接入且权重文件存在"""
    spec = MODELS.get(model_key)
    if not spec:
        return False
    svc = SERVICES.get(spec["svc"], {})
    if svc.get("pending") or not (svc.get("app") or svc.get("cmd")):
        return False
    return all(os.path.exists(p) for p in svc.get("needs", []))


MODELS = {
    "h3": {"name": "h3", "svc": "h3", "label": "MiniMax H3-Image",
           "size": (1024, 1024), "max_size": 2048, "warm": (768, 768),
           "note": "sd.cpp GGUF UD-Q6，6步，中文文字最强",
           "gpus": "GPU2+GPU3 · DiT@2 编码器@3"},
    "qwen": {"name": "qwen", "svc": "qwen", "label": "Qwen-Image-2.1-Turbo",
             "size": (1024, 1024), "max_size": 2048, "warm": (640, 640),
             "note": "Elo1047，8步，中文排版强",
             "gpus": "GPU2+GPU3 · DiT@2 编码器@3"},
    "zimage": {"name": "zimage", "svc": "zimage", "label": "Z-Image-Turbo",
               "size": (1024, 1024), "max_size": 2048, "warm": (640, 640),
               "note": "6B 极速模型，8步",
               "gpus": "GPU2 · 单卡全驻留"},
    "moss": {"name": "moss", "svc": "moss", "label": "MOSS-Image-SFT",
             "size": (1024, 1024), "max_size": 2048, "warm": (640, 640),
             "note": "布局/海报专精，25步",
             "gpus": "GPU2+GPU3 · 编码器@2 DiT@3"},
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


def http_post(url, payload, timeout=1800):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def svc_alive(svc_key, timeout=3):
    svc = SERVICES.get(svc_key)
    if not svc or svc.get("pending"):
        return False
    try:
        hp = svc.get("health_path", "/health")
        with urllib.request.urlopen(f"http://127.0.0.1:{svc['port']}{hp}", timeout=timeout) as r:
            return r.status == 200
    except Exception:
        return False


def _pidfile(svc_key):
    return os.path.join(RUN_DIR, f"{svc_key}.pid")


def _read_pid(svc_key):
    try:
        with open(_pidfile(svc_key)) as f:
            return int(f.read().strip())
    except Exception:
        return None


def _pid_on_port(port):
    """兜底：找出监听指定端口的进程（处理非本进程启动的旧服务）"""
    try:
        out = subprocess.run(["ss", "-lptnH", f"sport = :{port}"],
                             capture_output=True, text=True, timeout=5).stdout
        m = re.search(r"pid=(\d+)", out)
        return int(m.group(1)) if m else None
    except Exception:
        return None


def stop_service(svc_key):
    """停止服务进程组（uvicorn 可能会拉起子进程）"""
    import signal
    pid = _read_pid(svc_key)
    if pid:
        try:
            os.killpg(os.getpgid(pid), signal.SIGTERM)
        except Exception:
            try:
                os.kill(pid, signal.SIGTERM)
            except Exception:
                pass
        t0 = time.time()
        while time.time() - t0 < 20:
            if not svc_alive(svc_key):
                break
            time.sleep(1)
        if svc_alive(svc_key):
            try:
                os.killpg(os.getpgid(pid), signal.SIGKILL)
            except Exception:
                pass
            time.sleep(3)
    # 兜底：端口仍在监听说明是旧进程（非本进程启动），按端口找 PID 杀掉
    if svc_alive(svc_key):
        port = SERVICES[svc_key]["port"]
        pid2 = _pid_on_port(port)
        if pid2:
            try:
                os.kill(pid2, signal.SIGTERM)
                time.sleep(5)
                if svc_alive(svc_key):
                    os.kill(pid2, signal.SIGKILL)
            except Exception:
                pass
    try:
        os.remove(_pidfile(svc_key))
    except Exception:
        pass


def start_service(svc_key, timeout=600):
    """启动直连服务并等待 /health 就绪"""
    svc = SERVICES[svc_key]
    if svc.get("pending") or not (svc.get("app") or svc.get("cmd")):
        raise RuntimeError(f"服务 {svc_key} 尚未接入")
    if svc_alive(svc_key):
        return True
    env = os.environ.copy()
    env.update({k: str(v) for k, v in svc.get("env", {}).items()})
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    log = open(f"/var/log/imagehub_{svc_key}.log", "ab", buffering=0)
    if svc.get("cmd"):
        cmd = list(svc["cmd"])
    else:
        cmd = [PY_BIN, "-m", "uvicorn", svc["app"], "--host", "0.0.0.0",
               "--port", str(svc["port"])]
    logger.info(f"启动服务 {svc_key}: {' '.join(cmd)} (cwd={svc['cwd']})")
    p = subprocess.Popen(cmd, cwd=svc["cwd"], env=env, stdout=log, stderr=log,
                         start_new_session=True)
    with open(_pidfile(svc_key), "w") as f:
        f.write(str(p.pid))
    t0 = time.time()
    while time.time() - t0 < timeout:
        time.sleep(3)
        if svc_alive(svc_key):
            logger.info(f"服务 {svc_key} 就绪（{time.time()-t0:.0f}s）")
            return True
        if p.poll() is not None:
            raise RuntimeError(f"服务 {svc_key} 启动失败，退出码 {p.returncode}，"
                               f"详见 /var/log/imagehub_{svc_key}.log")
    raise RuntimeError(f"服务 {svc_key} 启动超时")


VLLM_METRICS = os.environ.get("VLLM_METRICS", "http://127.0.0.1:5563/metrics")


def _vllm_running():
    """当前 vLLM 正在执行的请求数；查不到返回 0（不阻塞）"""
    try:
        with urllib.request.urlopen(VLLM_METRICS, timeout=3) as r:
            for line in r.read().decode(errors="ignore").splitlines():
                if line.startswith("vllm:num_requests_running"):
                    return float(line.rsplit(" ", 1)[-1])
    except Exception:
        pass
    return 0.0


def _wait_vllm_idle(max_wait=25):
    """等 3.6 空闲（连续 3 次采样为 0）再动 GPU2/3，避免整机冻结。
    超时返回 False，由调用方快速失败，绝不硬闯。"""
    t0 = time.time()
    calm = 0
    while time.time() - t0 < max_wait:
        if _vllm_running() <= 0:
            calm += 1
            if calm >= 3:
                return True
        else:
            calm = 0
            logger.info(f"等待 3.6 空闲（running={_vllm_running():.0f}）")
        time.sleep(2)
    logger.warning(f"等待 3.6 空闲超时（{max_wait}s），放弃本次切换")
    return False


def acquire_model(model_key):
    """获取模型使用权：同模型并行、跨模型排队；切换 = 停旧服务 + 起新服务"""
    svc_key = MODELS[model_key]["svc"]
    with _cond:
        while True:
            if _state["switching"]:
                _cond.wait(); continue
            if _state["loaded"] == svc_key:
                if svc_alive(svc_key):
                    _state["active"] += 1
                    return
                # 状态残留但服务已不在：按未加载处理，走切换流程
                logger.warning(f"状态显示 {svc_key} 已加载，但服务无响应，重新启动")
                _state["loaded"] = None
            if _state["active"] > 0:
                _cond.wait(); continue
            _state["switching"] = True
            _state["loaded"] = None
            _state["target"] = model_key
            break
    ok = False
    try:
        # 可选门控（默认关闭）：IMAGEHUB_GATE=1 时等 3.6 空闲再切换
        if _settings.get("gate") and not _wait_vllm_idle():
            raise RuntimeError("对话服务 3.6 正在繁忙，暂停图像模型切换，请稍后再试")
        # GPU2/3 只能驻留一个图像服务
        for k in SERVICES:
            if k != svc_key:
                stop_service(k)
        start_service(svc_key)
        ok = True
    finally:
        with _cond:
            _state["switching"] = False
            _state["target"] = None
            if ok:
                _state["loaded"] = svc_key
                _state["active"] += 1
            _cond.notify_all()
        if ok:
            _save_state(svc_key)
    if not ok:
        raise RuntimeError(f"模型 {model_key} 启动失败")


def release_model():
    with _cond:
        _state["active"] = max(0, _state["active"] - 1)
        _cond.notify_all()


# ==================== ComfyUI 后端（可切换） ====================
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



def wf_qwen_edit(prompt, image_name, w, h, seed, steps=40, cfg=4.0):
    return {
        "1": {"class_type": "UnetLoaderGGUFMultiGPU",
              "inputs": {"unet_name": "qwen-image-edit-2511-Q6_K.gguf", "device": "cuda:0"}},
        "2": {"class_type": "CLIPLoaderMultiGPU",
              "inputs": {"clip_name": "qwen_2.5_vl_7b_bf16.safetensors",
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




WF_BUILDERS = {"h3": wf_h3, "qwen": wf_qwen, "zimage": wf_zimage, "moss": wf_moss}


def _sdcpp_generate(port, prompt, w, h, seed, timeout=1800):
    """sd-server 原生异步接口（h3）：提交任务 → 轮询结果，返回 b64"""
    payload = {"prompt": prompt, "negative_prompt": "低质量，模糊",
               "width": w, "height": h, "steps": 6, "cfg_scale": 1.0,
               "sampling_method": "euler", "scheduler": "simple",
               "batch_count": 1}
    if seed is not None:
        payload["seed"] = int(seed)
    job = http_post(f"http://127.0.0.1:{port}/sdcpp/v1/img_gen", payload, timeout=60)
    jid = job.get("id")
    if not jid:
        raise RuntimeError(f"sd-server 提交失败: {job}")
    t0 = time.time()
    while time.time() - t0 < timeout:
        time.sleep(2)
        try:
            with urllib.request.urlopen(
                    f"http://127.0.0.1:{port}/sdcpp/v1/jobs/{jid}", timeout=10) as r:
                st = json.loads(r.read())
        except Exception:
            continue
        if st.get("error"):
            raise RuntimeError(f"sd-server 错误: {st['error'].get('message')}")
        if st.get("status") == "completed":
            imgs = (st.get("result") or {}).get("images") or []
            if not imgs:
                raise RuntimeError("sd-server 未返回图片")
            return imgs[0].get("b64_json")
    raise RuntimeError("sd-server 生成超时")


_comfy_lock = threading.Lock()


def _req_backend(backend=None, legacy=None):
    """请求级后端：comfyui|comfy → ComfyUI；direct → 直连；留空 → 全局设置（默认 comfyui）"""
    v = (backend or legacy or "").strip().lower()
    if v in ("comfyui", "comfy", "com"):
        return "comfy"
    if v in ("direct", "直接", "直连"):
        return "direct"
    return _settings.get("backend", "comfy")


def _comfy_running():
    try:
        with urllib.request.urlopen(f"{COMFY}/system_stats", timeout=2) as r:
            return r.status == 200
    except Exception:
        return False


def _stop_comfy():
    """直连模式下停掉 ComfyUI，避免显存冲突"""
    if not _comfy_running():
        return
    try:
        subprocess.run(["systemctl", "stop", "comfyui-h3"], capture_output=True, timeout=60)
    except Exception:
        pass
    t0 = time.time()
    while time.time() - t0 < 60 and _comfy_alive(2):
        time.sleep(2)


def _run_comfy(model_key, prompt, w, h, seed):
    """ComfyUI 后端：停直连服务 → 起 ComfyUI → 提交工作流 → 读输出图"""
    with _comfy_lock:
        t0 = time.time()
        for k in SERVICES:
            stop_service(k)
        with _cond:
            _state["loaded"] = None
        _save_state(None)
        if not ensure_comfy():
            return {"model": model_key, "status": "error", "error": "ComfyUI 未就绪", "duration": 0}
        comfy_free()
        builder = WF_BUILDERS.get(model_key)
        if not builder:
            return {"model": model_key, "status": "error",
                    "error": f"ComfyUI 后端不支持 {model_key}", "duration": 0}
        ok, imgs, err, dur = _submit_wait(builder(prompt, w, h, seed))
        if not ok:
            return {"model": model_key, "status": "error", "error": err, "duration": dur}
        name = imgs[0] if imgs else None
        with _cond:
            _state["comfy_loaded"] = model_key if name else None
        return {"model": model_key, "status": "ok", "image": name,
                "duration": dur, "size": f"{w}x{h}", "seed": seed}


def run_model(model_key, prompt, width, height, seed, backend=None):
    spec = MODELS[model_key]
    w, h = (width or spec["size"][0]), (height or spec["size"][1])
    cap = spec.get("max_size")
    if cap and max(w, h) > cap:
        return {"model": model_key, "status": "error",
                "error": f"{model_key} 最大支持 {cap}×{cap}（当前 {w}×{h}）", "duration": 0}
    backend = backend or _settings.get("backend", "comfy")
    if backend == "comfy":
        try:
            logger.info(f"[comfy][{model_key}] prompt={prompt[:60]!r} size={w}x{h}")
            return _run_comfy(model_key, prompt, w, h, seed)
        except Exception as e:
            return {"model": model_key, "status": "error", "error": str(e)[:300], "duration": 0}
    try:
        _stop_comfy()
        acquire_model(model_key)
    except Exception as e:
        return {"model": model_key, "status": "error", "error": str(e)[:300], "duration": 0}
    try:
        logger.info(f"[{model_key}] prompt={prompt[:60]!r} size={w}x{h} seed={seed}")
        svc = SERVICES[spec["svc"]]
        port = svc["port"]
        t0 = time.time()
        if svc.get("api") == "sdcpp":
            b64 = _sdcpp_generate(port, prompt, w, h, seed)
        else:
            d = http_post(f"http://127.0.0.1:{port}/v1/images/generations",
                          {"prompt": prompt, "size": f"{w}x{h}", "seed": seed,
                           "n": 1, "response_format": "b64_json"}, timeout=1800)
            item = (d.get("data") or [{}])[0]
            b64 = item.get("b64_json")
        if not b64:
            return {"model": model_key, "status": "error",
                    "error": "服务未返回图片", "duration": round(time.time() - t0, 1)}
        name = f"hub_{model_key}_{uuid.uuid4().hex[:8]}.png"
        with open(os.path.join(OUTPUT_DIR, name), "wb") as f:
            f.write(base64.b64decode(b64))
        return {"model": model_key, "status": "ok", "image": name,
                "duration": round(time.time() - t0, 1), "size": f"{w}x{h}",
                "seed": seed}
    except Exception as e:
        return {"model": model_key, "status": "error",
                "error": str(e)[:300], "duration": 0}
    finally:
        release_model()


def record_job(prompt, model_keys, results, status="done"):
    """把单次生成写入历史（供 /v1 与 /api/edit 使用）"""
    jid = uuid.uuid4().hex[:12]
    try:
        conn = db()
        conn.execute("INSERT OR REPLACE INTO jobs VALUES (?,?,?,?,?,?)",
                     (jid, prompt, json.dumps(model_keys), time.time(),
                      json.dumps(results), status))
        conn.commit()
        conn.close()
    except Exception as e:
        logger.warning(f"写入历史失败: {e}")
    return jid


def run_job(job_id, prompt, model_keys, width, height, seed, backend=None):
    results = []
    for mk in model_keys:
        try:
            results.append(run_model(mk, prompt, width, height, seed, backend=backend))
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
    "zimage": "zimage", "z-image": "zimage", "z-image-turbo": "zimage",
    "moss": "moss", "moss-image": "moss",
}


class OAIImageRequest(BaseModel):
    prompt: str
    model: Optional[str] = "qwen"
    backend: Optional[str] = None  # 生成后端：comfyui | direct；留空用全局设置（默认 comfyui）
    com: Optional[str] = None      # 兼容旧参数（等同 backend），后续可移除
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
                     for k, v in MODELS.items() if not v.get("hidden")]}


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
            key = "qwen"                   # 无加载 → 默认用 qwen
            logger.info("auto: 无已加载模型，回落到 qwen")
    else:
        key = MODEL_ALIASES.get(lowered, raw_key)
        if key not in MODELS:
            raise HTTPException(400, f"未知模型 {raw_key}，可用: {list(MODELS)} + auto")
        if not _model_ready(key):
            raise HTTPException(503, {"error": f"模型 {key} 尚未就绪",
                                      "hint": "权重或服务配置未完成"})
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
    res = run_model(key, req.prompt, w, h, req.seed if req.seed is not None else 42,
                    backend=_req_backend(getattr(req, "backend", None),
                                        getattr(req, "com", None)))
    if res.get("status") != "ok" or not res.get("image"):
        raise HTTPException(500, res.get("error", "生成失败"))
    data = []
    if req.response_format == "b64_json":
        path = os.path.join(OUTPUT_DIR, res["image"])
        with open(path, "rb") as f:
            data.append({"b64_json": base64.b64encode(f.read()).decode(),
                         "revised_prompt": req.prompt})
    else:
        data.append({"url": f"/image/{res['image']}", "revised_prompt": req.prompt})
    job_id = record_job(req.prompt, [key],
                        [{"model": key, "status": "ok", "image": res["image"],
                          "duration": res.get("duration"), "size": res.get("size"),
                          "seed": res.get("seed")}])
    return {"created": int(time.time()), "data": data,
            "model": key, "duration_s": res.get("duration"),
            "size": res.get("size"), "job_id": job_id}


# ==================== 图片编辑（Qwen-Image-Edit-2511 直连服务） ====================
EDIT_LABEL = "Qwen-Image-Edit-2511"
EDIT_NEEDS = SERVICES["edit"]["needs"]

MODELS["edit"] = {"name": "edit", "svc": "edit", "label": EDIT_LABEL,
                  "size": (1024, 1024), "warm": (1024, 1024), "hidden": True,
                  "note": "上传图片 + 指令编辑", "gpus": "GPU2+GPU3 · DiT@2 编码器@3"}


class EditReq(BaseModel):
    prompt: str
    image_b64: str
    seed: Optional[int] = None
    steps: int = 20


@app.post("/api/edit")
def api_edit(req: EditReq):
    if not all(os.path.exists(p) for p in EDIT_NEEDS):
        raise HTTPException(503, {"error": "编辑模型权重尚未就绪",
                                  "hint": "需要 diffusers 组件 + GGUF DiT"})
    raw = req.image_b64.split(",")[-1]
    try:
        base64.b64decode(raw)
    except Exception:
        raise HTTPException(400, "图片 base64 解析失败")
    if _settings.get("backend") == "comfy":
        # ComfyUI 编辑路径
        os.makedirs(COMFY_INPUT, exist_ok=True)
        name = f"hub_edit_{uuid.uuid4().hex[:8]}.png"
        with open(os.path.join(COMFY_INPUT, name), "wb") as f:
            f.write(base64.b64decode(raw))
        with _comfy_lock:
            if not ensure_comfy():
                raise HTTPException(503, "ComfyUI 未就绪")
            comfy_free()
            t0 = time.time()
            ok, imgs, err, dur = _submit_wait(
                wf_qwen_edit(req.prompt, name, 1024, 1024,
                             req.seed if req.seed is not None else 42, req.steps))
        if not ok:
            return {"status": "error", "error": err, "duration": dur}
        record_job(req.prompt, ["edit"],
                   [{"model": "edit", "status": "ok",
                     "image": imgs[0] if imgs else None, "duration": dur}])
        return {"status": "ok", "image": imgs[0] if imgs else None,
                "duration": dur, "model": EDIT_LABEL}
    try:
        acquire_model("edit")
    except Exception as e:
        raise HTTPException(503, f"编辑服务启动失败: {str(e)[:200]}")
    try:
        port = SERVICES["edit"]["port"]
        t0 = time.time()
        d = http_post(f"http://127.0.0.1:{port}/v1/images/generations",
                      {"prompt": req.prompt, "image": raw, "size": "1024x1024",
                       "seed": req.seed if req.seed is not None else 42,
                       "num_inference_steps": req.steps,
                       "response_format": "b64_json"}, timeout=1800)
        item = (d.get("data") or [{}])[0]
        b64 = item.get("b64_json")
        if not b64:
            return {"status": "error", "error": "编辑服务未返回图片",
                    "duration": round(time.time() - t0, 1)}
        name = f"hub_edit_{uuid.uuid4().hex[:8]}.png"
        with open(os.path.join(OUTPUT_DIR, name), "wb") as f:
            f.write(base64.b64decode(b64))
        record_job(req.prompt, ["edit"],
                   [{"model": "edit", "status": "ok", "image": name,
                     "duration": round(time.time() - t0, 1)}])
        return {"status": "ok", "image": name,
                "duration": round(time.time() - t0, 1), "model": EDIT_LABEL}
    except Exception as e:
        return {"status": "error", "error": str(e)[:300],
                "duration": round(time.time() - t0, 1)}
    finally:
        release_model()


@app.get("/api/edit/model")
def api_edit_model():
    return {"key": "edit", "label": EDIT_LABEL,
            "note": "上传图片 + 指令进行编辑（20 步）",
            "ready": all(os.path.exists(p) for p in EDIT_NEEDS)}


@app.get("/api/edit/ready")
def api_edit_ready():
    return {"ready": all(os.path.exists(p) for p in EDIT_NEEDS)}


# ==================== API ====================
class CompareReq(BaseModel):
    prompt: str
    models: List[str]
    backend: Optional[str] = None
    width: Optional[int] = None
    height: Optional[int] = None
    seed: Optional[int] = None


@app.get("/api/models")
def api_models():
    out = []
    for k, v in MODELS.items():
        if v.get("hidden"):
            continue
        out.append({"key": k, "label": v["label"], "default_size": list(v["size"]),
                    "max_size": v.get("max_size"), "note": v["note"],
                    "gpus": v.get("gpus", "GPU2+GPU3"), "ready": _model_ready(k)})
    return out


@app.post("/api/compare")
def api_compare(req: CompareReq):
    keys = [m for m in req.models if m in MODELS]
    if not keys:
        raise HTTPException(400, "未选择有效模型")
    skipped = [k for k in keys if not _model_ready(k)]
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
                     args=(job_id, req.prompt, keys, req.width, req.height, seed,
                           _req_backend(getattr(req, "backend", None)))).start()
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
    if not _model_ready(req.model):
        raise HTTPException(503, {"error": f"模型 {req.model} 尚未就绪",
                                  "hint": "权重或服务配置未完成"})
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


# ==================== 3.6 压测 ====================
VLLM_CHAT = os.environ.get("VLLM_CHAT", "http://127.0.0.1:5563/v1/chat/completions")
VLLM_MODEL = os.environ.get("VLLM_MODEL", "qwen-3")
_s36 = {"running": False, "ok": 0, "err": 0, "start": 0, "conc": 4, "duration": 0}
_s36_stop = threading.Event()


def _s36_worker():
    prompt = ("请详细解释大语言模型推理中的 KV 缓存、连续批处理与投机解码，并举例说明。") * 3
    payload = json.dumps({"model": VLLM_MODEL,
                          "messages": [{"role": "user", "content": prompt}],
                          "max_tokens": 256, "temperature": 0.8}).encode()
    while not _s36_stop.is_set():
        if _s36["duration"] and time.time() - _s36["start"] > _s36["duration"]:
            break
        try:
            req = urllib.request.Request(VLLM_CHAT, data=payload,
                                         headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=300) as r:
                r.read()
            with _lock:
                _s36["ok"] += 1
        except Exception:
            with _lock:
                _s36["err"] += 1
            time.sleep(1)


def _s36_monitor():
    while not _s36_stop.is_set():
        if _s36["duration"] and time.time() - _s36["start"] > _s36["duration"]:
            break
        time.sleep(1)
    _s36_stop.set()
    time.sleep(1)
    with _lock:
        _s36["running"] = False
    logger.info(f"3.6 压测结束 ok={_s36['ok']} err={_s36['err']}")


class Stress36Req(BaseModel):
    conc: int = 4
    duration: int = 300


@app.post("/api/stress36/start")
def api_stress36_start(req: Stress36Req):
    global _s36
    if _s36["running"]:
        raise HTTPException(409, "压测正在进行中")
    _s36_stop.clear()
    with _lock:
        _s36 = {"running": True, "ok": 0, "err": 0, "start": time.time(),
                "conc": max(1, min(req.conc, 32)), "duration": max(0, req.duration)}
    for _ in range(_s36["conc"]):
        threading.Thread(target=_s36_worker, daemon=True).start()
    threading.Thread(target=_s36_monitor, daemon=True).start()
    logger.info(f"3.6 压测开始 conc={_s36['conc']} duration={_s36['duration']}s")
    return {"ok": True, **_s36}


@app.post("/api/stress36/stop")
def api_stress36_stop():
    _s36_stop.set()
    with _lock:
        _s36["running"] = False
    return {"ok": True, **_s36}


@app.get("/api/stress36/status")
def api_stress36_status():
    d = dict(_s36)
    d["elapsed"] = round(time.time() - d["start"], 0) if d["start"] else 0
    return d


class SettingsReq(BaseModel):
    backend: Optional[str] = None
    gate: Optional[bool] = None
    clocks: Optional[bool] = None


@app.get("/api/settings")
def api_get_settings():
    s = dict(_settings)
    s["clocks_now"] = _clocks_locked()
    return s


@app.post("/api/settings")
def api_set_settings(req: SettingsReq):
    if req.backend in ("direct", "comfy"):
        if req.backend != _settings.get("backend"):
            with _cond:
                _state["loaded"] = None
                _state["comfy_loaded"] = None
            _save_state(None)
        _settings["backend"] = req.backend
    if req.gate is not None:
        _settings["gate"] = bool(req.gate)
    if req.clocks is not None and bool(req.clocks) != _settings["clocks"]:
        _settings["clocks"] = bool(req.clocks)
        _clocks_set(_settings["clocks"])
    _save_settings()
    logger.info(f"设置已保存: {_settings}")
    return {"ok": True, **_settings, "clocks_now": _clocks_locked()}


def _clocks_locked():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=clocks.applications.graphics", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5).stdout.strip().split("\n")
        return all(int(x) and int(x) < 1600 for x in out if x.strip())
    except Exception:
        return False


@app.get("/api/version")
def api_version():
    return {"name": "ImageHub", "version": APP_VERSION, "build": APP_BUILD}


@app.get("/api/status")
def api_status():
    backend = _settings.get("backend", "comfy")
    with _cond:
        loaded = _state["comfy_loaded"] if backend == "comfy" else _state["loaded"]
        return {"loaded": loaded, "active": _state["active"],
                "switching": _state["switching"], "target": _state["target"],
                "backend": backend, "version": APP_VERSION}


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
                                "duration": x.get("duration"), "size": x.get("size"),
                                "seed": x.get("seed")}
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
                path = os.path.join(OUTPUT_DIR, os.path.basename(img))
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
                path = os.path.join(OUTPUT_DIR, os.path.basename(img))
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
            path = os.path.join(OUTPUT_DIR, os.path.basename(img))
            try:
                os.remove(path)
                removed += 1
            except Exception:
                pass
    _jobs.pop(job_id, None)
    logger.info(f"删除记录 {job_id}，图片 {removed} 张")
    return {"ok": True, "deleted_images": removed}


LEGACY_DIRS = ["/opt/ComfyUI/output"]


@app.get("/image/{filename}")
def image(filename: str):
    name = os.path.basename(filename)
    for d in [OUTPUT_DIR] + LEGACY_DIRS:
        path = os.path.join(d, name)
        if os.path.isfile(path):
            return FileResponse(path)
    raise HTTPException(404, "not found")


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
