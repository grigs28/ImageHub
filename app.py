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
import collections
import contextlib
import io
import json
import logging
import os
import queue
import re
import signal
import sqlite3
import subprocess
import tempfile
import threading
import time
import uuid
import urllib.error
import urllib.request
import wave
import zipfile
from typing import List, Optional

from fastapi import FastAPI, File, Form, HTTPException, Request, Response, UploadFile
from fastapi.responses import FileResponse, JSONResponse, HTMLResponse, RedirectResponse
from starlette.middleware.sessions import SessionMiddleware
from urllib.parse import quote
from pydantic import BaseModel

APP_VERSION = "1.7.3"          # ImageHub 版本号（1.7.3：轮播只播勾选项）
APP_BUILD = "2026-11-27"

logger = logging.getLogger("imagehub")
logging.basicConfig(level=logging.INFO)

OUTPUT_DIR = os.environ.get("IMAGEHUB_OUTPUT", "/opt/vLLM/imagehub/output")
os.makedirs(OUTPUT_DIR, exist_ok=True)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.environ.get("IMAGEHUB_DB", os.path.join(BASE_DIR, "imagehub.db"))
RUN_DIR = os.environ.get("IMAGEHUB_RUN", os.path.join(BASE_DIR, "run"))
os.makedirs(RUN_DIR, exist_ok=True)
LEGACY_DIRS = ["/opt/ComfyUI/output"]
THUMB_DIR = os.environ.get("IMAGEHUB_THUMBS", os.path.join(BASE_DIR, "thumbs"))
os.makedirs(THUMB_DIR, exist_ok=True)
PY_BIN = os.environ.get("IMAGEHUB_PY", "/opt/conda3/envs/imagegen/bin/python")

# ComfyUI 后端（可切换）
COMFY = "http://127.0.0.1:5568"
COMFY_OUTPUT = "/opt/ComfyUI/output"
COMFY_INPUT = "/opt/ComfyUI/input"

# TTS 服务（Qwen3-TTS 1.7B 三变体，内部 127.0.0.1:5570）
TTS_PORT = int(os.environ.get("IMAGEHUB_TTS_PORT", "5570"))
TTS_PY = os.environ.get("IMAGEHUB_TTS_PY", "/opt/conda3/envs/qwen-tts/bin/python")
TTS_MODELS = {
    "custom": os.environ.get("IMAGEHUB_TTS_CUSTOM", "/opt/models/Qwen3-TTS/CustomVoice"),
    "design": os.environ.get("IMAGEHUB_TTS_DESIGN", "/opt/models/Qwen3-TTS/VoiceDesign"),
    "clone": os.environ.get("IMAGEHUB_TTS_CLONE", "/opt/models/Qwen3-TTS/Base"),
}
_tts = {"proc": None, "state": "stopped", "error": None, "active": 0, "pending": 0}
_tts_lock = threading.RLock()
_arbiter_q = collections.deque()
_arbiter_lock = threading.Lock()
TTS_MAX_UPLOAD = int(os.environ.get("IMAGEHUB_TTS_MAX_UPLOAD",
                                    str(20 * 1024 * 1024)))
TTS_OUT_DIR = os.environ.get("IMAGEHUB_TTS_OUT", os.path.join(OUTPUT_DIR, "tts"))
os.makedirs(TTS_OUT_DIR, exist_ok=True)

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


def _tls_enabled():
    cert = os.environ.get("IMAGEHUB_TLS_CERT")
    key = os.environ.get("IMAGEHUB_TLS_KEY")
    return bool(cert and key and os.path.isfile(cert) and os.path.isfile(key))


app = FastAPI(title="ImageHub")
app.add_middleware(SessionMiddleware, secret_key=SESSION_SECRET,
                   session_cookie="imagehub_session", max_age=8 * 3600,
                   https_only=_tls_enabled())
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


_state = {"loaded": None, "comfy_loaded": None, "active": 0, "switching": False,
          "target": None, "image_pending": 0}
_cond = threading.Condition()


# ==================== 直连后端服务 ====================
# 每个服务是一个独立 uvicorn 进程（FastAPI + diffusers），端口固定。
# GPU2/3 显存有限，同一时间只驻留一个服务，切换 = 停旧进程 + 起新进程。
SERVICES = {
    "qwen": {
        "port": 5566, "cwd": "/opt/models/Qwen/Qwen-Image-2.1-Turbo",
        "app": "qwenimage_server:app",
        "env": {"CUDA_VISIBLE_DEVICES": "0,1", "QIMAGE_VAE_TILING": "1"},
        "needs": ["/opt/models/Qwen/Qwen-Image-2.1-Turbo/model_index.json"],
    },
    "zimage": {
        "port": 5565, "cwd": "/opt/models/Tongyi-MAI/Z-Image-Turbo",
        "app": "zimage_server:app", "env": {"CUDA_VISIBLE_DEVICES": "0,1", "ZIMAGE_BALANCED": "1"},
        "needs": ["/opt/models/Tongyi-MAI/Z-Image-Turbo/model_index.json"],
    },
    "moss": {
        "port": 5567, "cwd": "/opt/models/OpenMOSS/MOSS-Image-SFT",
        "app": "moss_server:app", "env": {"CUDA_VISIBLE_DEVICES": "0,1", "MOSS_SPLIT": "1"},
        "needs": ["/opt/models/OpenMOSS/MOSS-Image-SFT/model_index.json"],
    },
    "edit": {
        "port": 5564, "cwd": "/opt/models/Qwen/Qwen-Image-Edit-2511",
        "app": "qwen_edit_server:app",
        "env": {"CUDA_VISIBLE_DEVICES": "0,1", "EDIT_MODE": "split"},
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
        "env": {"CUDA_VISIBLE_DEVICES": "0,1"}, "health_path": "/v1/models", "api": "sdcpp",
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
    if spec.get("comfy_only"):
        return all(os.path.exists(p) for p in spec.get("needs", []))
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
    "flux2klein": {
        "name": "flux2klein", "svc": "comfy", "label": "FLUX.2 Klein 4B",
        "size": (1024, 1024), "max_size": 2048, "warm": (1024, 1024),
        "note": "4 步蒸馏，速度最快，英文/写实强",
        "gpus": "ComfyUI · GPU0/1",
        "comfy_only": True,
        "needs": [
            "/opt/ComfyUI/models/diffusion_models/flux-2-klein-4b-fp8.safetensors",
            "/opt/ComfyUI/models/text_encoders/qwen_3_4b.safetensors",
            "/opt/ComfyUI/models/vae/flux2-vae.safetensors",
        ],
    },
    "nemotron": {
        "name": "nemotron", "svc": "comfy", "label": "Nemotron-30B-A3B",
        "size": (1024, 1024), "max_size": 2048, "warm": (1024, 1024),
        "note": "自回归 30B/3B W4A8，写实/长提示词强",
        "gpus": "ComfyUI · GPU0/1",
        "comfy_only": True,
        "needs": [
            "/opt/ComfyUI/models/diffusion_models/nemotron-image/"
            "nemotron_labs_image_30b_a3b_w4a8.safetensors",
            "/opt/ComfyUI/models/vae/nemotron-image/"
            "nemotron_labs_image_tokenizer_bf16.safetensors",
            "/opt/ComfyUI/models/text_encoders/qwen_3_4b.safetensors",
        ],
    },
}


# 恢复上次驻留模型（BFF 重启后 auto 仍可用）
try:
    _state["loaded"] = _load_state()
    if _state["loaded"]:
        logger.info(f"恢复驻留模型状态: {_state['loaded']}")
except Exception:
    pass


# ==================== 工具函数 ====================
_schema_lock = threading.Lock()
_schema_ready = False


def _ensure_schema():
    """建表、WAL、迁移、索引、启动中断标记，只执行一次。"""
    global _schema_ready
    if _schema_ready:
        return
    with _schema_lock:
        if _schema_ready:
            return
        conn = sqlite3.connect(DB_PATH, timeout=5)
        try:
            conn.execute("""CREATE TABLE IF NOT EXISTS jobs(
                id TEXT PRIMARY KEY, prompt TEXT, models TEXT, created REAL,
                results TEXT)""")
            conn.execute("""CREATE TABLE IF NOT EXISTS users(
                id TEXT PRIMARY KEY, username TEXT, display_name TEXT,
                is_admin INTEGER DEFAULT 0, source TEXT DEFAULT 'yz',
                synced_at REAL, updated_at REAL)""")
            conn.execute("""CREATE TABLE IF NOT EXISTS api_keys(
                id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT, key_hash TEXT UNIQUE,
                role TEXT DEFAULT 'user', enabled INTEGER DEFAULT 1,
                created REAL, last_used REAL, quota_json TEXT)""")
            conn.execute("""CREATE TABLE IF NOT EXISTS audit(
                id INTEGER PRIMARY KEY AUTOINCREMENT, actor TEXT, action TEXT,
                detail TEXT, created REAL)""")
            conn.execute("""CREATE TABLE IF NOT EXISTS presets(
                id INTEGER PRIMARY KEY AUTOINCREMENT, kind TEXT, value TEXT, created REAL,
                UNIQUE(kind, value))""")
            conn.execute("""CREATE TABLE IF NOT EXISTS projects(
                id TEXT PRIMARY KEY, name TEXT, owner TEXT, created REAL)""")
            conn.execute("""CREATE TABLE IF NOT EXISTS assets(
                id TEXT PRIMARY KEY, project_id TEXT, kind TEXT, path TEXT,
                meta TEXT, created REAL)""")
            conn.execute("""CREATE TABLE IF NOT EXISTS tts_audio(
                id TEXT PRIMARY KEY, created REAL NOT NULL, variant TEXT, text TEXT,
                language TEXT, speaker TEXT, instruct TEXT, file TEXT,
                duration REAL, elapsed REAL, size INTEGER,
                status TEXT DEFAULT 'ok', error TEXT)""")
            conn.execute("PRAGMA journal_mode=WAL")
            try:
                conn.execute("ALTER TABLE jobs ADD COLUMN status TEXT DEFAULT 'done'")
                conn.commit()
            except Exception:
                pass
            try:
                conn.execute("ALTER TABLE api_keys ADD COLUMN key_plain TEXT")
                conn.commit()
            except Exception:
                pass
            try:
                conn.execute("ALTER TABLE jobs ADD COLUMN project TEXT")
                conn.commit()
            except Exception:
                pass
            conn.execute("CREATE INDEX IF NOT EXISTS idx_jobs_created ON jobs(created DESC)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_jobs_project_created "
                         "ON jobs(project, created DESC)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_tts_audio_created "
                         "ON tts_audio(created DESC)")
            conn.execute("UPDATE jobs SET status='interrupted' WHERE status='running'")
            conn.commit()
        finally:
            conn.close()
        _schema_ready = True


def db(readonly=False):
    _ensure_schema()
    if readonly:
        conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True, timeout=5)
    else:
        conn = sqlite3.connect(DB_PATH, timeout=5)
    conn.execute("PRAGMA busy_timeout=5000")
    return conn


# ==================== 缩略图 ====================
_thumb_locks = {}
_thumb_locks_guard = threading.Lock()
_thumb_q = queue.Queue(maxsize=2000)
_thumb_workers_started = False
_thumb_workers_lock = threading.Lock()


def _thumb_path(filename, w):
    name = os.path.basename(filename)
    w = max(64, min(int(w), 1024))
    return os.path.join(THUMB_DIR, f"{os.path.splitext(name)[0]}_{w}.jpg")


def _find_source(filename):
    name = os.path.basename(filename)
    for d in [OUTPUT_DIR] + LEGACY_DIRS:
        p = os.path.join(d, name)
        if os.path.isfile(p):
            return p
    return None


def _thumb_lock(out):
    with _thumb_locks_guard:
        return _thumb_locks.setdefault(out, threading.Lock())


def _ensure_thumb(filename, w):
    from PIL import Image
    w = max(64, min(int(w), 1024))
    name = os.path.basename(filename)
    out = _thumb_path(name, w)
    if os.path.isfile(out):
        return out
    with _thumb_lock(out):
        if os.path.isfile(out):
            return out
        src = _find_source(name)
        if not src:
            return None
        im = Image.open(src).convert("RGB")
        im.thumbnail((w, w * 8))
        fd, tmp = tempfile.mkstemp(prefix=".thumb_", suffix=".jpg", dir=THUMB_DIR)
        os.close(fd)
        try:
            im.save(tmp, "JPEG", quality=82)
            os.replace(tmp, out)
        except Exception:
            try:
                os.remove(tmp)
            except OSError:
                pass
            raise
    return out


def _thumb_worker():
    while True:
        image, w = _thumb_q.get()
        try:
            _ensure_thumb(image, w)
        except Exception as e:
            logger.warning(f"缩略图预生成失败 {image} w={w}: {e}")
        finally:
            _thumb_q.task_done()


def _start_thumb_workers():
    global _thumb_workers_started
    with _thumb_workers_lock:
        if _thumb_workers_started:
            return
        _thumb_workers_started = True
        for _ in range(2):
            threading.Thread(target=_thumb_worker, daemon=True,
                             name="thumb-worker").start()


def _prewarm_thumbs(results):
    _start_thumb_workers()
    for r in results or []:
        if not isinstance(r, dict) or r.get("status") != "ok":
            continue
        image = r.get("image")
        if not image:
            continue
        for w in (160, 480, 1024):
            try:
                _thumb_q.put_nowait((image, w))
            except queue.Full:
                logger.warning("缩略图预生成队列已满，跳过剩余任务")
                return


def _prewarm_history_thumbs(limit=200):
    try:
        conn = db(readonly=True)
        rows = conn.execute("SELECT results FROM jobs ORDER BY created DESC LIMIT ?",
                            (limit,)).fetchall()
        conn.close()
    except Exception as e:
        logger.warning(f"读取历史缩略图回填失败: {e}")
        return
    for (raw,) in rows:
        try:
            results = json.loads(raw or "[]")
        except Exception:
            continue
        _prewarm_thumbs(results)


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


def _tts_url(path):
    return f"http://127.0.0.1:{TTS_PORT}{path}"


def _tts_json(path, timeout=5):
    """GET TTS 服务 JSON；不可达返回 None。"""
    try:
        with urllib.request.urlopen(_tts_url(path), timeout=timeout) as r:
            return json.loads(r.read())
    except Exception:
        return None


def _tts_call(method, path, body=None, data=None, content_type=None, timeout=600):
    """调用 TTS 服务，返回 (status, bytes, content_type)；不抛网络异常。"""
    headers = {}
    if body is not None:
        data = json.dumps(body).encode()
        headers["Content-Type"] = "application/json"
    elif content_type:
        headers["Content-Type"] = content_type
    req = urllib.request.Request(_tts_url(path), data=data, headers=headers,
                                 method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read(), r.headers.get("content-type", "")
    except urllib.error.HTTPError as e:
        return e.code, e.read(), e.headers.get("content-type", "application/json")
    except Exception as e:
        return 503, json.dumps({"detail": f"TTS 服务不可用: {e}"}).encode(), "application/json"


def _tts_detail(body):
    try:
        return json.loads(body).get("detail") or body.decode(errors="replace")[:200]
    except Exception:
        return body.decode(errors="replace")[:200]


def _tts_start(timeout=300):
    """启动 TTS 子进程并等待三个模型就绪。"""
    with _tts_lock:
        health = _tts_json("/health", timeout=2)
        if health and health.get("state") == "ready":
            _tts["state"] = "ready"
            _tts["error"] = None
            return True
        if health and health.get("state") == "error":
            _tts["state"] = "error"
            _tts["error"] = health.get("error") or "TTS 加载失败"
            _tts_stop(reset_error=False)
            return False
        proc = _tts.get("proc")
        if not proc or proc.poll() is not None:
            if not os.path.exists(TTS_PY):
                _tts["state"] = "error"
                _tts["error"] = f"TTS Python 不存在: {TTS_PY}"
                return False
            for path in TTS_MODELS.values():
                if not os.path.isdir(path):
                    _tts["state"] = "error"
                    _tts["error"] = f"模型目录不存在: {path}"
                    return False
            env = os.environ.copy()
            env.update({"CUDA_VISIBLE_DEVICES": "0,1", "PYTHONPATH": BASE_DIR,
                        "HF_HUB_OFFLINE": "1"})
            cmd = [TTS_PY, "-m", "tts_service.app", "--host", "127.0.0.1",
                   "--port", str(TTS_PORT),
                   "--custom", TTS_MODELS["custom"],
                   "--design", TTS_MODELS["design"],
                   "--clone", TTS_MODELS["clone"],
                   "--devices", "cuda:0,cuda:0,cuda:1"]
            log_path = os.environ.get("IMAGEHUB_TTS_LOG", "/var/log/imagehub_tts.log")
            try:
                log = open(log_path, "ab")
            except OSError:
                log = open(os.path.join(RUN_DIR, "imagehub_tts.log"), "ab")
            try:
                proc = subprocess.Popen(cmd, cwd=BASE_DIR, env=env,
                                        stdout=log, stderr=subprocess.STDOUT)
            except Exception as e:  # noqa: BLE001
                _tts["state"] = "error"
                _tts["error"] = f"TTS 进程启动失败: {e}"
                return False
            finally:
                log.close()
            _tts["proc"] = proc
        _tts["state"] = "loading"
        _tts["error"] = None
        t0 = time.time()
        while time.time() - t0 < timeout:
            health = _tts_json("/health", timeout=2)
            if health and health.get("state") == "ready":
                _tts["state"] = "ready"
                return True
            if health and health.get("state") == "error":
                _tts["state"] = "error"
                _tts["error"] = health.get("error") or "TTS 加载失败"
                _tts_stop(reset_error=False)
                return False
            if proc.poll() is not None:
                _tts["state"] = "error"
                _tts["error"] = f"TTS 进程退出（code={proc.returncode}）"
                return False
            time.sleep(2)
        _tts["state"] = "error"
        _tts["error"] = "TTS 加载超时"
        _tts_stop(reset_error=False)
        return False


def _tts_stop(timeout=60, reset_error=True):
    """停止 TTS 子进程，并清理仍占用端口的残留进程。"""
    with _tts_lock:
        proc = _tts.get("proc")
        if proc and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout)
            except subprocess.TimeoutExpired:
                proc.kill()
                try:
                    proc.wait(10)
                except subprocess.TimeoutExpired:
                    pass
        pid = _pid_on_port(TTS_PORT)
        if pid:
            try:
                os.kill(pid, signal.SIGTERM)
            except Exception:
                pass
            t0 = time.time()
            while time.time() - t0 < timeout and _pid_on_port(TTS_PORT):
                time.sleep(1)
            pid = _pid_on_port(TTS_PORT)
            if pid:
                try:
                    os.kill(pid, signal.SIGKILL)
                except Exception:
                    pass
        _tts["proc"] = None
        _tts["state"] = "stopped"
        if reset_error:
            _tts["error"] = None


def _tts_adopt():
    """BFF 重启后接管仍在运行的 TTS 服务。"""
    health = _tts_json("/health", timeout=2)
    if health and health.get("ok"):
        _tts["state"] = "ready"
        _tts["error"] = None
    elif health:
        _tts["state"] = health.get("state", "error")
        _tts["error"] = health.get("error")


def _wav_duration_bytes(data):
    try:
        with wave.open(io.BytesIO(data), "rb") as w:
            return round(w.getnframes() / float(w.getframerate()), 2)
    except Exception:
        return 0.0


def _tts_save_result(variant, text, language, speaker, instruct, data, elapsed):
    """把合成结果写入 TTS 专用语音库（不写 jobs、不进图库）。"""
    name = f"tts_{uuid.uuid4().hex}.wav"
    path = os.path.join(TTS_OUT_DIR, name)
    os.makedirs(TTS_OUT_DIR, exist_ok=True)
    with open(path, "wb") as f:
        f.write(data)
    jid = uuid.uuid4().hex
    conn = db()
    try:
        conn.execute(
            "INSERT INTO tts_audio(id,created,variant,text,language,speaker,instruct,"
            "file,duration,elapsed,size,status) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
            (jid, time.time(), variant, text, language or "", speaker or "",
             instruct or "", name, _wav_duration_bytes(data),
             round(elapsed, 1), len(data), "ok"))
        conn.commit()
    finally:
        conn.close()
    return jid


def _tts_history_audio(filename):
    name = os.path.basename(filename or "")
    if not name or name != filename:
        return None
    path = os.path.join(TTS_OUT_DIR, name)
    return path if os.path.isfile(path) else None


_MULTIPART_FILENAME_RE = re.compile(r"[^A-Za-z0-9._-]+")
_MULTIPART_ALLOWED_TYPES = {
    "audio/wav", "audio/x-wav", "audio/wave", "audio/mpeg", "audio/mp3",
    "audio/mp4", "audio/x-m4a", "audio/aac", "audio/flac", "audio/x-flac",
    "audio/ogg", "audio/opus", "audio/webm", "audio/amr", "audio/x-ms-wma",
    "application/octet-stream",
}


def _safe_multipart_filename(name):
    """只保留基础名与安全字符，阻止 CRLF/引号注入与路径穿越。"""
    base = os.path.basename((name or "").replace("\\", "/"))
    base = base.replace("\r", "").replace("\n", "").replace('"', "")
    base = _MULTIPART_FILENAME_RE.sub("_", base).strip("._")
    return (base or "file")[:120]


def _safe_multipart_ctype(ctype):
    c = (ctype or "").split(";")[0].strip().lower()
    return c if c in _MULTIPART_ALLOWED_TYPES else "application/octet-stream"


def _multipart_encode(fields, files):
    """构造 multipart/form-data，避免给 BFF 增加新依赖。"""
    boundary = "----imagehub" + uuid.uuid4().hex
    out = []
    for key, value in fields.items():
        out.append((f"--{boundary}\r\nContent-Disposition: form-data; "
                    f"name=\"{key}\"\r\n\r\n{value}\r\n").encode())
    for key, (filename, data, ctype) in files.items():
        out.append((f"--{boundary}\r\nContent-Disposition: form-data; "
                    f"name=\"{key}\"; filename=\"{_safe_multipart_filename(filename)}\"\r\n"
                    f"Content-Type: {_safe_multipart_ctype(ctype)}\r\n\r\n").encode())
        out.append(data + b"\r\n")
    out.append(f"--{boundary}--\r\n".encode())
    return b"".join(out), f"multipart/form-data; boundary={boundary}"


# ==================== GPU 资源仲裁（图像 ↔ TTS 互斥） ====================
def _arbiter_acquire(timeout: float = 600):
    """FIFO 票号：所有后端切换按到达顺序串行，避免图像/TTS 同时抢显存。"""
    ev = threading.Event()
    with _arbiter_lock:
        _arbiter_q.append(ev)
        if _arbiter_q[0] is ev:
            ev.set()
    if ev.wait(timeout):
        return True
    with _arbiter_lock:
        if ev in _arbiter_q:
            was_head = _arbiter_q[0] is ev
            _arbiter_q.remove(ev)
            if was_head and _arbiter_q:
                _arbiter_q[0].set()
    return False


def _arbiter_release():
    with _arbiter_lock:
        if _arbiter_q:
            _arbiter_q.popleft()
        if _arbiter_q:
            _arbiter_q[0].set()


def _image_busy():
    """图像侧是否有正在运行或即将运行的请求（不含“已加载但空闲”）。"""
    with _cond:
        if (_state.get("active", 0) or _state.get("switching")
                or _state.get("image_pending", 0)):
            return True
    with _lock:
        running_job = any(j.get("status") == "running" for j in _jobs.values())
    with _dq["lock"]:
        dq = bool(_dq.get("running") or _dq.get("queue"))
    return bool(running_job or dq or _comfy_lock.locked())


def _wait_image_idle(timeout):
    deadline = time.time() + timeout
    while _image_busy():
        left = deadline - time.time()
        if left <= 0:
            return False
        with _cond:
            _cond.wait(min(left, 2))
    return True


def _tts_begin_request(timeout=600):
    """TTS 请求入口：确保 TTS 已加载；必要时等图像任务结束并停掉图像后端。

    等图像空闲时**不持有仲裁票**，否则多模型对比的下一张图会拿不到票而死锁。
    """
    deadline = time.time() + timeout
    while True:
        left = deadline - time.time()
        if left <= 0 or not _wait_image_idle(left):
            _tts["state"] = "error"
            _tts["error"] = "等待图像任务结束超时"
            return False
        left = deadline - time.time()
        if left <= 0 or not _arbiter_acquire(left):
            _tts["state"] = "error"
            _tts["error"] = "等待 TTS 切换排队超时"
            return False
        try:
            health = _tts_json("/health", timeout=2)
            if health and health.get("state") == "ready":
                with _cond:
                    _tts["state"] = "ready"
                    _tts["active"] += 1
                return True
            if _image_busy():
                continue
            for k in SERVICES:
                try:
                    stop_service(k)
                except Exception:
                    pass
            try:
                _stop_comfy()
            except Exception:
                pass
            with _cond:
                _state["loaded"] = None
                _state["comfy_loaded"] = None
            _save_state(None)
            if not _tts_start(timeout=300):
                return False
            with _cond:
                _tts["active"] += 1
            return True
        finally:
            _arbiter_release()


def _tts_end_request():
    with _cond:
        _tts["active"] = max(0, _tts["active"] - 1)
        _cond.notify_all()


def _tts_release_for_image(timeout=900):
    """图像请求入口：等 TTS 生成结束，停掉整个 TTS 服务，再让图像加载。"""
    if not _arbiter_acquire(timeout):
        raise RuntimeError("等待图像任务排队超时")
    try:
        deadline = time.time() + timeout
        while True:
            with _cond:
                active = _tts["active"]
                state = _tts["state"]
            if active > 0:
                left = deadline - time.time()
                if left <= 0:
                    raise RuntimeError("等待 TTS 生成结束超时")
                with _cond:
                    _cond.wait(min(left, 2))
                continue
            if state != "stopped" or _tts_json("/health", timeout=2):
                _tts_stop()
            with _cond:
                _state["image_pending"] = _state.get("image_pending", 0) + 1
            return
    finally:
        _arbiter_release()


def _image_done():
    with _cond:
        _state["image_pending"] = max(0, _state.get("image_pending", 0) - 1)
        _cond.notify_all()


@contextlib.contextmanager
def _image_session():
    _tts_release_for_image()
    try:
        yield
    finally:
        _image_done()


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




def wf_qwen_cn(control_name, control_type, strength, prompt, w, h, seed, mask_name=None):
    """Qwen-Image-2.1 + Fun ControlNet Union（控制图已在 ComfyUI input 目录）"""
    wf = wf_qwen(prompt, w, h, seed)
    if control_type in ("canny", "lineart"):
        wf["20"] = {"class_type": "LoadImage", "inputs": {"image": control_name}}
        wf["21"] = {"class_type": "Canny",
                    "inputs": {"image": ["20", 0], "low_threshold": 0.4, "high_threshold": 0.8}}
        ctrl_src = ["21", 0]
    else:  # depth / normal / raw：用户上传的就是控制图
        wf["20"] = {"class_type": "LoadImage", "inputs": {"image": control_name}}
        ctrl_src = ["20", 0]
    wf["22"] = {"class_type": "ModelPatchLoader",
                "inputs": {"name": "qwen_image_2.1_fun_controlnet_union_int8_convrot.safetensors"}}
    wf["23"] = {"class_type": "QwenImageDiffsynthControlnet",
                "inputs": {"model": ["2", 0], "model_patch": ["22", 0], "vae": ["4", 0],
                           "image": ctrl_src, "strength": float(strength)}}
    if mask_name:
        wf["30"] = {"class_type": "LoadImageMask",
                    "inputs": {"image": mask_name, "channel": "red"}}
        wf["23"]["inputs"]["mask"] = ["30", 0]
    wf["10"]["inputs"]["model"] = ["23", 0]
    wf["13"]["inputs"]["filename_prefix"] = "hub_cn"
    return wf


# 工作流注册表：同一模型可按功能选用不同工作流
def wf_flux2klein(prompt, w, h, seed):
    """FLUX.2 Klein 4B（ComfyUI，4 步蒸馏）。"""
    return {
        "70": {"class_type": "UNETLoader",
               "inputs": {"unet_name": "flux-2-klein-4b-fp8.safetensors",
                          "weight_dtype": "default"}},
        "71": {"class_type": "CLIPLoader",
               "inputs": {"clip_name": "qwen_3_4b.safetensors",
                          "type": "flux2", "device": "default"}},
        "72": {"class_type": "VAELoader",
               "inputs": {"vae_name": "flux2-vae.safetensors"}},
        "74": {"class_type": "CLIPTextEncode",
               "inputs": {"clip": ["71", 0], "text": prompt}},
        "76": {"class_type": "ConditioningZeroOut",
               "inputs": {"conditioning": ["74", 0]}},
        "63": {"class_type": "CFGGuider",
               "inputs": {"model": ["70", 0], "positive": ["74", 0],
                          "negative": ["76", 0], "cfg": 1.0}},
        "66": {"class_type": "EmptyFlux2LatentImage",
               "inputs": {"width": w, "height": h, "batch_size": 1}},
        "62": {"class_type": "Flux2Scheduler",
               "inputs": {"steps": 4, "width": w, "height": h}},
        "61": {"class_type": "KSamplerSelect",
               "inputs": {"sampler_name": "euler"}},
        "73": {"class_type": "RandomNoise",
               "inputs": {"noise_seed": seed}},
        "64": {"class_type": "SamplerCustomAdvanced",
               "inputs": {"noise": ["73", 0], "guider": ["63", 0],
                          "sampler": ["61", 0], "sigmas": ["62", 0],
                          "latent_image": ["66", 0]}},
        "65": {"class_type": "VAEDecode",
               "inputs": {"samples": ["64", 0], "vae": ["72", 0]}},
        "9": {"class_type": "SaveImage",
              "inputs": {"images": ["65", 0],
                         "filename_prefix": "hub_flux2klein"}},
    }


def wf_nemotron(prompt, w, h, seed):
    """Nemotron-Labs-Image-30B-A3B（ComfyUI 自定义节点）。"""
    return {
        "1": {"class_type": "NemotronImageLoader",
              "inputs": {
                  "model_name": "nemotron-image/"
                                "nemotron_labs_image_30b_a3b_w4a8.safetensors",
                  "experts": "auto", "engine": "auto"}},
        "2": {"class_type": "NemotronImageGenerate",
              "inputs": {"model": ["1", 0], "prompt": prompt,
                         "width": w, "height": h, "seed": seed,
                         "batch_size": 1, "temperature": 0.9,
                         "top_p": 1.0, "top_k": 0}},
        "3": {"class_type": "NemotronImageDecoderLoader",
              "inputs": {"vae_name": "nemotron-image/"
                                    "nemotron_labs_image_tokenizer_bf16.safetensors"}},
        "4": {"class_type": "VAEDecode",
              "inputs": {"samples": ["2", 0], "vae": ["3", 0]}},
        "5": {"class_type": "SaveImage",
              "inputs": {"images": ["4", 0],
                         "filename_prefix": "hub_nemotron"}},
    }


WF_BUILDERS = {"h3": wf_h3, "qwen": wf_qwen, "zimage": wf_zimage, "moss": wf_moss,
               "flux2klein": wf_flux2klein, "nemotron": wf_nemotron}
MODEL_WORKFLOWS = {
    "qwen": {"t2i": wf_qwen, "controlnet": wf_qwen_cn},
    "h3": {"t2i": wf_h3},
    "zimage": {"t2i": wf_zimage},
    "moss": {"t2i": wf_moss},
    "flux2klein": {"t2i": wf_flux2klein},
    "nemotron": {"t2i": wf_nemotron},
}


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
_comfy_state = {"model": None}


def _comfy_prepare(model_key):
    """仅在切换底座模型时卸载；同模型（含 ControlNet patch）直接复用"""
    if _comfy_state["model"] != model_key:
        logger.info(f"ComfyUI 切换底座模型 {_comfy_state['model']} -> {model_key}，卸载后重载")
        comfy_free()
        _comfy_state["model"] = model_key


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
    _comfy_state["model"] = None
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
        _comfy_prepare(model_key)
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


def _run_model_impl(model_key, prompt, width, height, seed, backend=None):
    spec = MODELS[model_key]
    w, h = (width or spec["size"][0]), (height or spec["size"][1])
    cap = spec.get("max_size")
    if cap and max(w, h) > cap:
        return {"model": model_key, "status": "error",
                "error": f"{model_key} 最大支持 {cap}×{cap}（当前 {w}×{h}）", "duration": 0}
    backend = backend or _settings.get("backend", "comfy")
    if spec.get("comfy_only"):
        backend = "comfy"
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


def run_model(model_key, prompt, width, height, seed, backend=None):
    with _image_session():
        return _run_model_impl(model_key, prompt, width, height, seed, backend)


def _admin_seed():
    raw = os.environ.get("IMAGEHUB_ADMINS", "grigs")
    return {x.strip() for x in raw.split(",") if x.strip()}


def _hash_key(k):
    import hashlib
    return hashlib.sha256(k.encode()).hexdigest()


def _api_authed(apikey, request=None):
    """外部调用必须有 Key；网页登录会话可直接调用"""
    if apikey:
        ok, _ = _key_valid(apikey)
        return ok
    if request is not None and (request.session.get("user") or {}):
        return True
    return False


def _key_valid(raw):
    """校验请求体中的 apikey；返回 (ok, info)"""
    if not raw:
        return True, {"name": "anonymous", "role": "user", "is_admin": False}
    h = _hash_key(raw.strip())
    try:
        conn = db()
        row = conn.execute("SELECT name, role, enabled FROM api_keys WHERE key_hash=?",
                           (h,)).fetchone()
        if row and row[2]:
            conn.execute("UPDATE api_keys SET last_used=? WHERE key_hash=?", (time.time(), h))
            conn.commit()
        conn.close()
    except Exception:
        return False, None
    if not row or not row[2]:
        return False, None
    return True, {"name": row[0], "role": row[1], "is_admin": row[1] == "admin"}


def _api_key_user(request):
    """Bearer key → {name, role, is_admin}"""
    auth = request.headers.get("authorization", "")
    if not auth.lower().startswith("bearer "):
        return None
    h = _hash_key(auth[7:].strip())
    try:
        conn = db()
        row = conn.execute("SELECT name, role, enabled FROM api_keys WHERE key_hash=?",
                           (h,)).fetchone()
        if row and row[2]:
            conn.execute("UPDATE api_keys SET last_used=? WHERE key_hash=?", (time.time(), h))
            conn.commit()
        conn.close()
    except Exception:
        return None
    if not row or not row[2]:
        return None
    return {"name": row[0], "role": row[1], "is_admin": row[1] == "admin"}


def _is_admin(request):
    tok = os.environ.get("IMAGEHUB_ADMIN_TOKEN", "")
    if tok and request.headers.get("x-admin-token") == tok:
        return True
    u = _api_key_user(request)
    if u and u.get("is_admin"):
        return True
    sess = request.session.get("user") or {}
    return bool(sess.get("is_admin"))


def _require_admin(request):
    if not _is_admin(request):
        raise HTTPException(403, "需要管理员权限")


def _audit(actor, action, detail=""):
    try:
        conn = db()
        conn.execute("INSERT INTO audit(actor,action,detail,created) VALUES(?,?,?,?)",
                     (actor, action, json.dumps(detail, ensure_ascii=False)[:2000], time.time()))
        conn.commit()
        conn.close()
    except Exception:
        pass


def _upsert_user(u):
    """登录时同步 yz-login 用户；is_admin 以本地为准（新用户按种子/票据初始化）"""
    uid = str(u.get("id") or u.get("username") or "")
    if not uid:
        return 0
    now = time.time()
    conn = db()
    row = conn.execute("SELECT is_admin FROM users WHERE id=?", (uid,)).fetchone()
    if row is None:
        seed_admin = 1 if (u.get("username") in _admin_seed() or u.get("is_admin")) else 0
        conn.execute("INSERT INTO users VALUES(?,?,?,?,?,?,?)",
                     (uid, u.get("username"), u.get("display_name"), seed_admin, "yz", now, now))
        is_admin = seed_admin
    else:
        conn.execute("UPDATE users SET username=?, display_name=?, synced_at=? WHERE id=?",
                     (u.get("username"), u.get("display_name"), now, uid))
        is_admin = row[0]
    conn.commit()
    conn.close()
    return is_admin


def _upsert_job_result(job_id, prompt, models, created, results, status):
    conn = db()
    try:
        conn.execute(
            "INSERT INTO jobs(id,prompt,models,created,results,status) "
            "VALUES(?,?,?,?,?,?) ON CONFLICT(id) DO UPDATE SET "
            "results=excluded.results, status=excluded.status",
            (job_id, prompt, json.dumps(models), created,
             json.dumps(results), status),
        )
        conn.commit()
    finally:
        conn.close()


def record_job(prompt, model_keys, results, status="done", project=None):
    """把单次生成写入历史（供 /v1 与 /api/edit 使用）"""
    jid = uuid.uuid4().hex[:12]
    try:
        conn = db()
        conn.execute("INSERT OR REPLACE INTO jobs(id,prompt,models,created,results,status,project) "
                     "VALUES (?,?,?,?,?,?,?)",
                     (jid, prompt, json.dumps(model_keys), time.time(),
                      json.dumps(results), status, project))
        conn.commit()
        conn.close()
        _prewarm_thumbs(results)
    except Exception as e:
        logger.warning(f"写入历史失败: {e}")
    return jid


def run_job(job_id, prompt, model_keys, width, height, seed, backend=None):
    results = []
    created = _jobs.get(job_id, {}).get("created", time.time())
    for mk in model_keys:
        try:
            results.append(run_model(mk, prompt, width, height, seed, backend=backend))
        except Exception as e:
            results.append({"model": mk, "status": "error", "error": str(e)[:300]})
        with _lock:
            _jobs[job_id]["results"] = list(results)
        try:
            _upsert_job_result(job_id, prompt, model_keys, created, results, "running")
        except Exception:
            pass
    with _lock:
        _jobs[job_id]["status"] = "done"
    _upsert_job_result(job_id, prompt, model_keys, created, results, "done")
    _prewarm_thumbs(results)



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
def api_set_power(req: PowerReq, request: Request):
    _require_admin(request)
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
    "flux2klein": "flux2klein", "flux-2-klein": "flux2klein",
    "flux2-klein": "flux2klein", "flux-klein": "flux2klein",
    "flux": "flux2klein", "flux2": "flux2klein",
    "nemotron": "nemotron", "nemotron-image": "nemotron",
    "nemotron-labs-image": "nemotron", "nemotron-30b": "nemotron",
}


class OAIImageRequest(BaseModel):
    prompt: str
    model: Optional[str] = "qwen"
    backend: Optional[str] = None  # 生成后端：comfyui | direct；留空用全局设置（默认 comfyui）
    com: Optional[str] = None      # 兼容旧参数（等同 backend），后续可移除
    apikey: Optional[str] = None   # 可选：调用密钥，留空匿名调用
    only_loaded: bool = False   # true=只用当前已加载模型，不触发切换
    n: int = 1
    size: str = "1024x1024"
    response_format: str = "b64_json"
    seed: Optional[int] = None


@app.get("/v1/models")
def v1_models():
    return {"object": "list",
            "data": [{"id": k, "object": "model", "owned_by": "imagehub",
                      "label": v["label"], "default_size": list(v["size"]),
                      "max_size": v.get("max_size"),
                      "capabilities": (["t2i", "edit"] if k == "edit" else
                                       (["t2i", "controlnet"] if k == "qwen" else ["t2i"]))}
                     for k, v in MODELS.items() if not v.get("hidden")]}


@app.post("/v1/images/generations")
def v1_generate(req: OAIImageRequest, request: Request):
    if not _api_authed(getattr(req, "apikey", None), request):
        raise HTTPException(401, "需要 API Key（网页登录状态可直接调用）")
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
        err = res.get("error", "生成失败")
        if "最大支持" in err:
            raise HTTPException(400, err)
        raise HTTPException(500, err)
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


def _api_edit_impl(req: EditReq):
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


@app.post("/api/edit")
def api_edit(req: EditReq):
    with _image_session():
        return _api_edit_impl(req)


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
        conn = db(readonly=True)
        row = conn.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
        conn.close()
        if not row:
            raise HTTPException(404, "job not found")
        status = row[5] if len(row) > 5 and row[5] else "done"
        return {"id": row[0], "prompt": row[1], "models": json.loads(row[2]),
                "created": row[3], "results": json.loads(row[4]),
                "status": status, "project": row[6] if len(row) > 6 else None}
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


_gpu_cache = {"t": 0, "data": None}


@app.get("/api/gpu")
def api_gpu():
    import subprocess
    now = time.time()
    if _gpu_cache["data"] is not None and now - _gpu_cache["t"] < 2:
        return _gpu_cache["data"]
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
        _gpu_cache["t"] = now
        _gpu_cache["data"] = rows
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
def api_stress36_start(req: Stress36Req, request: Request):
    _require_admin(request)
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
def api_stress36_stop(request: Request):
    _require_admin(request)
    _s36_stop.set()
    with _lock:
        _s36["running"] = False
    return {"ok": True, **_s36}


@app.get("/api/stress36/status")
def api_stress36_status():
    d = dict(_s36)
    d["elapsed"] = round(time.time() - d["start"], 0) if d["start"] else 0
    return d


class DesignRenderReq(BaseModel):
    prompt: str
    control_image: Optional[str] = None    # base64 控制图（线稿/深度/白模截图）
    control_type: str = "canny"            # canny | lineart | depth | normal | raw
    control_strength: float = 0.8
    mask_image: Optional[str] = None       # base64 蒙版（局部重绘）
    size: str = "1024x1024"
    seed: Optional[int] = 42
    model: str = "qwen"
    apikey: Optional[str] = None
    backend: Optional[str] = None
    project: Optional[str] = None


def _save_b64(b64, prefix):
    os.makedirs(COMFY_INPUT, exist_ok=True)
    raw = (b64 or "").split(",")[-1]
    try:
        data = base64.b64decode(raw)
    except Exception:
        raise HTTPException(400, "图片 base64 解析失败")
    name = f"{prefix}_{uuid.uuid4().hex[:8]}.png"
    with open(os.path.join(COMFY_INPUT, name), "wb") as f:
        f.write(data)
    return name


def _design_generate_impl(prompt, control_b64, control_type, strength, w, h, seed,
                          mask_b64=None, model="qwen"):
    """ControlNet 结构化生成（ComfyUI），返回 (image, duration)"""
    if control_type not in ("canny", "lineart", "depth", "normal", "raw"):
        raise HTTPException(400, "control_type 仅支持 canny/lineart/depth/normal/raw")
    cap = MODELS.get(model, {}).get("max_size", 2048)
    if max(w, h) > cap:
        raise HTTPException(400, f"{model} 最大支持 {cap}×{cap}（当前 {w}×{h}）")
    name = _save_b64(control_b64, "design")
    mask_name = _save_b64(mask_b64, "mask") if mask_b64 else None
    if not _comfy_lock.acquire(timeout=600):
        raise HTTPException(503, "图像服务繁忙（队列已满），请稍后重试")
    try:
        # 设计走 ComfyUI：先停掉直连服务，避免两边抢 GPU2/3 显存
        for k in SERVICES:
            try:
                stop_service(k)
            except Exception:
                pass
        with _cond:
            _state["loaded"] = None
        _save_state(None)
        if not ensure_comfy():
            raise HTTPException(503, "ComfyUI 未就绪（ControlNet 目前依赖 ComfyUI 后端）")
        _comfy_prepare("qwen")
        wf = wf_qwen_cn(name, control_type, strength, prompt, w, h,
                        seed if seed is not None else 42, mask_name)
        ok, imgs, err, dur = _submit_wait(wf, timeout=1800)
    finally:
        _comfy_lock.release()
    if not ok:
        raise HTTPException(500, err)
    return (imgs[0] if imgs else None), dur


def _design_generate(prompt, control_b64, control_type, strength, w, h, seed,
                     mask_b64=None, model="qwen"):
    with _image_session():
        return _design_generate_impl(prompt, control_b64, control_type, strength,
                                     w, h, seed, mask_b64, model)


def _design_response(prompt, image, dur, w, h, seed, control_type, strength, workflow,
                     model="qwen", project=None):
    jid = record_job(prompt, [model],
                     [{"model": model, "status": "ok", "image": image,
                       "duration": dur, "size": f"{w}x{h}", "seed": seed,
                       "workflow": workflow, "control_type": control_type}],
                     project=project)
    return {"created": int(time.time()), "job_id": jid, "image": image,
            "url": f"/image/{image}", "duration_s": dur, "size": f"{w}x{h}",
            "seed": seed, "control_type": control_type,
            "control_strength": strength, "workflow": workflow}


@app.post("/v1/design/render")
def api_design_render(req: DesignRenderReq, request: Request):
    """结构化生成：控制图 + 提示词（ComfyUI + Qwen-Image-2.1 Fun ControlNet）"""
    if not _api_authed(req.apikey, request):
        raise HTTPException(401, "需要 API Key（网页登录状态可直接调用）")
    if not req.control_image:
        raise HTTPException(400, "需要 control_image（base64）")
    w, h = _parse_size(req.size)
    image, dur = _design_generate(req.prompt, req.control_image, req.control_type,
                                  req.control_strength, w, h, req.seed, req.mask_image)
    return _design_response(req.prompt, image, dur, w, h, req.seed,
                            req.control_type, req.control_strength,
                            "inpaint" if req.mask_image else "controlnet",
                            project=req.project)


class ClayReq(BaseModel):
    image: str                              # 白模截图（base64）
    materials: str = "清水混凝土、超白玻璃、深灰金属"
    lighting: str = "黄昏暖光，柔和阴影"
    style: str = "写实建筑摄影"
    prompt: Optional[str] = None
    size: str = "1024x1024"
    seed: Optional[int] = 42
    strength: float = 0.9
    apikey: Optional[str] = None


@app.post("/v1/design/clay2render")
def api_clay2render(req: ClayReq, request: Request):
    """白模 → 效果图（预设提示词模板，raw 控制）"""
    if not _api_authed(req.apikey, request):
        raise HTTPException(401, "需要 API Key（网页登录状态可直接调用）")
    prompt = req.prompt or (
        f"以输入的白模渲染图为结构参考，生成{req.style}。"
        f"材质：{req.materials}；光照：{req.lighting}。"
        "严格保持原有建筑体量、开窗位置、透视与构图不变，只替换材质与环境。")
    w, h = _parse_size(req.size)
    image, dur = _design_generate(prompt, req.image, "raw", req.strength, w, h, req.seed)
    return _design_response(prompt, image, dur, w, h, req.seed, "raw", req.strength, "clay2render")


class SketchReq(BaseModel):
    sketch: str                             # 线稿/SU 截图（base64）
    style: str = "写实建筑效果图"
    prompt: Optional[str] = None
    size: str = "1024x1024"
    seed: Optional[int] = 42
    strength: float = 0.8
    apikey: Optional[str] = None


@app.post("/v1/design/sketch2render")
def api_sketch2render(req: SketchReq, request: Request):
    """线稿 / SU 截图 → 效果图（lineart 控制）"""
    if not _api_authed(req.apikey, request):
        raise HTTPException(401, "需要 API Key（网页登录状态可直接调用）")
    prompt = req.prompt or (
        f"以输入线稿为结构参考，生成{req.style}。"
        "保持建筑轮廓、体块关系与构图不变，补充合理的材质、光影与环境。")
    w, h = _parse_size(req.size)
    image, dur = _design_generate(prompt, req.sketch, "lineart", req.strength, w, h, req.seed)
    return _design_response(prompt, image, dur, w, h, req.seed, "lineart", req.strength, "sketch2render")


class BatchReq(BaseModel):
    prompt: str
    n: int = 4
    seed_start: int = 42
    size: str = "1024x1024"
    control_image: Optional[str] = None
    control_type: str = "canny"
    control_strength: float = 0.8
    apikey: Optional[str] = None


@app.post("/v1/design/batch")
def api_design_batch(req: BatchReq, request: Request):
    """批量方案：同一提示词多 seed 出图（n ≤ 4）"""
    if not _api_authed(req.apikey, request):
        raise HTTPException(401, "需要 API Key（网页登录状态可直接调用）")
    n = max(1, min(int(req.n), 4))
    w, h = _parse_size(req.size)
    out = []
    t0 = time.time()
    for i in range(n):
        seed = req.seed_start + i
        if req.control_image:
            image, dur = _design_generate(req.prompt, req.control_image, req.control_type,
                                          req.control_strength, w, h, seed)
            workflow, ctype, strength = "batch-controlnet", req.control_type, req.control_strength
        else:
            res = run_model("qwen", req.prompt, w, h, seed, backend="comfy")
            image, dur = res.get("image"), res.get("duration")
            workflow, ctype, strength = "batch-t2i", None, None
        out.append(_design_response(req.prompt, image, dur, w, h, seed, ctype, strength, workflow))
    return {"created": int(time.time()), "count": len(out),
            "jobs": out, "total_s": round(time.time() - t0, 1)}


def _parse_size(size):
    try:
        return [int(x) for x in size.lower().split("x")]
    except Exception:
        raise HTTPException(400, "size 格式应为 1024x1024")


@app.get("/v1/jobs/{jid}")
def v1_job(jid: str):
    return api_job(jid)


# ==================== 设计预设 / 多角度 / 异步 ====================
DESIGN_PRESETS = {
    "materials": ["清水混凝土", "超白玻璃", "玻璃幕墙", "红砖", "木格栅", "铝板",
                  "石材", "白色涂料", "穿孔金属", "陶土板"],
    "lighting": ["清晨柔光", "正午硬光", "黄昏暖光", "夜景灯光", "阴天散射",
                 "雨后反光", "室内暖光"],
    "styles": ["写实建筑摄影", "建筑竞赛效果图", "水彩手绘", "轴测图",
               "室内表现图", "极简插画"],
    "views": ["人视", "鸟瞰", "正立面", "左侧", "右侧", "室内"],
}


@app.get("/v1/design/presets")
def api_design_presets(request: Request, apikey: Optional[str] = None):
    if not _api_authed(apikey, request):
        raise HTTPException(401, "需要 API Key")
    out = {k: list(v) for k, v in DESIGN_PRESETS.items()}
    try:
        conn = db(readonly=True)
        for kind, value in conn.execute("SELECT kind,value FROM presets ORDER BY id").fetchall():
            if kind in out and value not in out[kind]:
                out[kind].append(value)
        conn.close()
    except Exception:
        pass
    return out


class PresetReq(BaseModel):
    kind: str
    value: str


@app.get("/api/admin/presets")
def admin_presets(request: Request):
    _require_admin(request)
    conn = db(readonly=True)
    rows = conn.execute("SELECT id,kind,value FROM presets ORDER BY kind,id").fetchall()
    conn.close()
    return [{"id": r[0], "kind": r[1], "value": r[2]} for r in rows]


@app.post("/api/admin/presets")
def admin_add_preset(req: PresetReq, request: Request):
    _require_admin(request)
    if req.kind not in DESIGN_PRESETS:
        raise HTTPException(400, f"kind 只能是 {list(DESIGN_PRESETS)}")
    v = (req.value or "").strip()
    if not v:
        raise HTTPException(400, "value 不能为空")
    conn = db()
    try:
        conn.execute("INSERT OR IGNORE INTO presets(kind,value,created) VALUES(?,?,?)",
                     (req.kind, v, time.time()))
        conn.commit()
    finally:
        conn.close()
    _audit((request.session.get("user") or {}).get("username", "admin"),
           "preset.add", {"kind": req.kind, "value": v})
    return {"ok": True}


@app.delete("/api/admin/presets/{pid}")
def admin_del_preset(pid: int, request: Request):
    _require_admin(request)
    conn = db()
    conn.execute("DELETE FROM presets WHERE id=?", (pid,))
    conn.commit()
    conn.close()
    return {"ok": True}


# ---- 深度 / 法线预处理（CPU）----
_depth_pipe = None


def _get_depth_pipe():
    global _depth_pipe
    if _depth_pipe is None:
        from transformers import pipeline as _hf_pipeline
        _depth_pipe = _hf_pipeline(
            "depth-estimation",
            model="/opt/models/depth-anything/Depth-Anything-V2-Small-hf",
            device=-1)
    return _depth_pipe


class PreprocessReq(BaseModel):
    image: str
    type: str = "depth"        # depth | normal
    apikey: Optional[str] = None


@app.post("/v1/design/preprocess")
def api_design_preprocess(req: PreprocessReq, request: Request):
    """自动生成控制图：深度图或法线图（CPU 推理，不占显存）"""
    if not _api_authed(req.apikey, request):
        raise HTTPException(401, "需要 API Key")
    import io as _io
    from PIL import Image as _Img
    raw = (req.image or "").split(",")[-1]
    try:
        img = _Img.open(_io.BytesIO(base64.b64decode(raw))).convert("RGB")
    except Exception:
        raise HTTPException(400, "image base64 解析失败")
    try:
        depth = _get_depth_pipe()(img)["depth"]
    except Exception as e:
        raise HTTPException(500, f"深度推理失败: {str(e)[:200]}")
    if req.type == "normal":
        import numpy as np
        d = np.asarray(depth.resize(img.size)).astype(np.float32) / 255.0
        gy, gx = np.gradient(d)
        nz = np.ones_like(d) * 0.6
        nx, ny = -gx * 8.0, -gy * 8.0
        ln = np.sqrt(nx ** 2 + ny ** 2 + nz ** 2)
        rgb = np.stack([(nx / ln + 1) / 2, (ny / ln + 1) / 2, (nz / ln)], axis=-1)
        out = _Img.fromarray((rgb * 255).clip(0, 255).astype(np.uint8), "RGB")
    else:
        out = depth.convert("RGB") if getattr(depth, "mode", None) != "RGB" else depth
    buf = _io.BytesIO(); out.save(buf, "PNG")
    return {"type": req.type, "width": out.width, "height": out.height,
            "image_b64": base64.b64encode(buf.getvalue()).decode()}


class ViewsReq(BaseModel):
    image: str
    base_prompt: str = "现代简约建筑，清水混凝土与超白玻璃"
    views: List[str] = ["人视", "鸟瞰"]
    size: str = "1024x1024"
    seed: Optional[int] = 42
    strength: float = 0.9
    apikey: Optional[str] = None
    project: Optional[str] = None


@app.post("/v1/design/views")
def api_design_views(req: ViewsReq, request: Request):
    """多角度：同一参考图，按视角批量生成（最多 4 个视角）"""
    if not _api_authed(req.apikey, request):
        raise HTTPException(401, "需要 API Key")
    views = [v for v in req.views if v in DESIGN_PRESETS["views"]][:4] or ["人视"]
    w, h = _parse_size(req.size)
    out = []
    t0 = time.time()
    for i, v in enumerate(views):
        prompt = (f"{req.base_prompt}。视角：{v}。保持建筑体量、材质与整体风格一致，"
                  "只改变观察角度，输出写实效果图。")
        image, dur = _design_generate(prompt, req.image, "raw", req.strength,
                                      w, h, (req.seed or 42) + i)
        out.append(_design_response(prompt, image, dur, w, h, (req.seed or 42) + i,
                                    "raw", req.strength, f"views-{v}", project=req.project))
    return {"created": int(time.time()), "count": len(out), "views": views,
            "jobs": out, "total_s": round(time.time() - t0, 1)}


# ---- 异步任务队列（单 worker，webhook 可选）----
_dq = {"queue": [], "running": False, "lock": threading.Lock()}


def _dq_worker():
    while True:
        with _dq["lock"]:
            if not _dq["queue"]:
                _dq["running"] = False
                return
            item = _dq["queue"].pop(0)
        jid, payload, callback = item
        logger.info(f"异步任务开始 {jid}")
        try:
            w, h = _parse_size(payload.get("size", "1024x1024"))
            image, dur = _design_generate(payload["prompt"], payload["control_image"],
                                          payload.get("control_type", "raw"),
                                          float(payload.get("control_strength", 0.9)),
                                          w, h, payload.get("seed"),
                                          payload.get("mask_image"))
            conn = db()
            conn.execute("UPDATE jobs SET status='done', results=? WHERE id=?",
                         (json.dumps([{"model": "qwen", "status": "ok", "image": image,
                                       "duration": dur, "size": f"{w}x{h}",
                                       "seed": payload.get("seed"),
                                       "workflow": payload.get("workflow", "async")}]),
                          jid))
            conn.commit()
            conn.close()
            logger.info(f"异步任务完成 {jid} -> {image}")
            _prewarm_thumbs([{"status": "ok", "image": image}])
            if callback:
                try:
                    http_post(callback, {"job_id": jid, "status": "done",
                                         "url": f"/image/{image}", "duration_s": dur},
                              timeout=10)
                except Exception:
                    pass
        except Exception as e:
            logger.warning(f"异步任务失败 {jid}: {e}")
            conn = db()
            conn.execute("UPDATE jobs SET status='error', results=? WHERE id=?",
                         (json.dumps([{"model": "qwen", "status": "error",
                                       "error": str(e)[:300]}]), jid))
            conn.commit()
            conn.close()


class AsyncReq(DesignRenderReq):
    callback_url: Optional[str] = None


@app.post("/v1/design/async")
def api_design_async(req: AsyncReq, request: Request):
    """异步提交：立即返回 job_id，后台排队执行；可选 callback_url 回调"""
    if not _api_authed(req.apikey, request):
        raise HTTPException(401, "需要 API Key")
    if not req.control_image:
        raise HTTPException(400, "需要 control_image（base64）")
    jid = uuid.uuid4().hex[:12]
    conn = db()
    conn.execute("INSERT OR REPLACE INTO jobs(id,prompt,models,created,results,status,project) "
                 "VALUES (?,?,?,?,?,?,?)",
                 (jid, req.prompt, json.dumps(["qwen"]), time.time(), "[]",
                  "queued", req.project))
    conn.commit()
    conn.close()
    with _dq["lock"]:
        _dq["queue"].append((jid, req.dict(), req.callback_url))
        start = not _dq["running"]
        _dq["running"] = True
    if start:
        threading.Thread(target=_dq_worker, daemon=True).start()
    return {"ok": True, "job_id": jid, "status": "queued",
            "queue_len": len(_dq["queue"])}


class OutpaintReq(BaseModel):
    image: str
    prompt: str = ""
    expand: int = 256          # 每边扩展像素（16 的倍数）
    seed: Optional[int] = 42
    strength: float = 1.0
    apikey: Optional[str] = None
    project: Optional[str] = None


@app.post("/v1/design/outpaint")
def api_design_outpaint(req: OutpaintReq, request: Request):
    """扩图：四边扩展并重新生成边缘内容（蒙版=新增区域）"""
    if not _api_authed(req.apikey, request):
        raise HTTPException(401, "需要 API Key")
    import io as _io
    from PIL import Image as _Img, ImageDraw as _Draw
    raw = (req.image or "").split(",")[-1]
    try:
        im = _Img.open(_io.BytesIO(base64.b64decode(raw))).convert("RGB")
    except Exception:
        raise HTTPException(400, "image base64 解析失败")
    e = max(16, min(int(req.expand) // 16 * 16, 512))
    W, H = im.width + 2 * e, im.height + 2 * e
    if W > 2048 or H > 2048:
        raise HTTPException(400, f"扩图后尺寸 {W}x{H} 超过 2048 上限")
    canvas = _Img.new("RGB", (W, H))
    # 边缘拉伸填充，避免纯色边界
    canvas.paste(im.resize((W, H)), (0, 0))
    canvas.paste(im, (e, e))
    mask = _Img.new("L", (W, H), 0)
    _Draw.Draw(mask).rectangle([0, 0, W, H], fill=255)
    _Draw.Draw(mask).rectangle([e, e, e + im.width, e + im.height], fill=0)
    def _b64(img):
        buf = _io.BytesIO(); img.save(buf, "PNG")
        return base64.b64encode(buf.getvalue()).decode()
    prompt = req.prompt or "延续原图的建筑风格、材质与光影，自然补全扩展区域"
    image, dur = _design_generate(prompt, _b64(canvas), "raw", req.strength,
                                  W, H, req.seed, _b64(mask.convert("RGB")))
    return _design_response(prompt, image, dur, W, H, req.seed, "raw",
                            req.strength, "outpaint", project=req.project)


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
def api_set_settings(req: SettingsReq, request: Request):
    _require_admin(request)
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


def _preload_default():
    """启动后自动加载默认模型（qwen）；等 3.6 就绪且空闲后再加载，避免并发分配显存"""
    if os.environ.get("IMAGEHUB_PRELOAD", "qwen").strip() in ("", "0", "off"):
        return
    if _tts.get("state") in ("ready", "loading") or _tts_json("/health", timeout=2):
        logger.info("TTS 已加载/加载中，跳过图像预加载")
        return
    target = os.environ.get("IMAGEHUB_PRELOAD", "qwen").strip() or "qwen"
    time.sleep(10)
    try:
        t0 = time.time()
        while time.time() - t0 < 300:
            ready = False
            try:
                with urllib.request.urlopen("http://127.0.0.1:5563/v1/models", timeout=3) as r:
                    ready = r.status == 200
            except Exception:
                ready = False
            if ready and _vllm_running() <= 0:
                break
            time.sleep(5)
        if target not in MODELS:
            return
        if _settings.get("backend") == "direct":
            with _image_session():
                acquire_model(target)
                release_model()
        else:
            # ComfyUI 后端：跑一次小尺寸预热
            ww, wh = MODELS[target].get("warm", (640, 640))
            run_model(target, "warmup", ww, wh, 42)
        logger.info(f"启动预加载完成: {target}")
    except Exception as e:
        logger.warning(f"启动预加载失败: {e}")


@app.on_event("startup")
def _on_startup_schema():
    _ensure_schema()
    threading.Thread(target=_prewarm_history_thumbs, daemon=True).start()


@app.on_event("startup")
def _adopt_running_services():
    """接管重启前已在运行的直连服务，避免状态丢失"""
    for k in SERVICES:
        try:
            if svc_alive(k):
                with _cond:
                    _state["loaded"] = k
                _save_state(k)
                logger.info(f"接管已在运行的服务: {k}")
                return
        except Exception:
            pass
    # 没有服务在跑：清掉 state.json 里残留的“已驻留”，避免状态页谎报
    with _cond:
        _state["loaded"] = None
    _save_state(None)


@app.on_event("startup")
def _adopt_tts_service():
    try:
        _tts_adopt()
    except Exception:
        pass


@app.on_event("startup")
def _on_startup_preload():
    threading.Thread(target=_preload_default, daemon=True).start()


# ==================== 管理后台 ====================
class UserAdminReq(BaseModel):
    is_admin: bool


class KeyReq(BaseModel):
    name: str = ""
    role: str = "user"      # user | admin


@app.get("/api/admin/users")
def admin_users(request: Request):
    _require_admin(request)
    conn = db(readonly=True)
    rows = conn.execute("SELECT id,username,display_name,is_admin,source,synced_at "
                        "FROM users ORDER BY is_admin DESC, username").fetchall()
    conn.close()
    return [{"id": r[0], "username": r[1], "display_name": r[2],
             "is_admin": bool(r[3]), "source": r[4], "synced_at": r[5]} for r in rows]


PROTECTED_USERS = {"grigs"}


@app.delete("/api/admin/users/{uid}")
def admin_del_user(uid: str, request: Request):
    _require_admin(request)
    conn = db()
    row = conn.execute("SELECT username FROM users WHERE id=?", (uid,)).fetchone()
    if not row:
        conn.close()
        raise HTTPException(404, "用户不存在")
    if (row[0] or "") in PROTECTED_USERS:
        conn.close()
        raise HTTPException(403, "该用户受保护，不可删除")
    conn.execute("DELETE FROM users WHERE id=?", (uid,))
    conn.commit()
    conn.close()
    _audit((request.session.get("user") or {}).get("username", "admin"), "user.delete", {"uid": uid})
    return {"ok": True}


@app.post("/api/admin/users/{uid}")
def admin_set_user(uid: str, req: UserAdminReq, request: Request):
    _require_admin(request)
    conn = db()
    row = conn.execute("SELECT username FROM users WHERE id=?", (uid,)).fetchone()
    if not row:
        conn.close()
        raise HTTPException(404, "用户不存在（需先登录过一次）")
    if (row[0] or "") in PROTECTED_USERS:
        conn.close()
        raise HTTPException(403, "该用户受保护，不可修改")
    cur = conn.execute("UPDATE users SET is_admin=?, updated_at=? WHERE id=?",
                       (1 if req.is_admin else 0, time.time(), uid))
    conn.commit()
    ok = cur.rowcount > 0
    conn.close()
    if not ok:
        raise HTTPException(404, "用户不存在（需先登录过一次）")
    _audit((request.session.get("user") or {}).get("username", "admin"),
           "user.set_admin", {"uid": uid, "is_admin": req.is_admin})
    return {"ok": True}


@app.get("/api/admin/keys")
def admin_keys(request: Request):
    _require_admin(request)
    conn = db(readonly=True)
    rows = conn.execute("SELECT id,name,role,enabled,created,last_used,key_plain "
                        "FROM api_keys ORDER BY id DESC").fetchall()
    conn.close()
    return [{"id": r[0], "name": r[1], "role": r[2], "enabled": bool(r[3]),
             "created": r[4], "last_used": r[5], "key": r[6]} for r in rows]


@app.post("/api/admin/keys")
def admin_create_key(req: KeyReq, request: Request):
    _require_admin(request)
    if req.role not in ("user", "admin"):
        raise HTTPException(400, "role 只能是 user 或 admin")
    name = (req.name or "").strip() or f"key-{time.strftime('%m%d-%H%M%S')}"
    raw = "ih-" + uuid.uuid4().hex + uuid.uuid4().hex[:8]
    conn = db()
    conn.execute("INSERT INTO api_keys(name,key_hash,role,enabled,created,key_plain) "
                 "VALUES(?,?,?,1,?,?)",
                 (name, _hash_key(raw), req.role, time.time(), raw))
    conn.commit()
    conn.close()
    _audit((request.session.get("user") or {}).get("username", "admin"),
           "key.create", {"name": name, "role": req.role})
    return {"ok": True, "key": raw, "note": "只显示一次，请立即保存"}


@app.delete("/api/admin/keys/{kid}")
def admin_delete_key(kid: int, request: Request):
    _require_admin(request)
    conn = db()
    conn.execute("DELETE FROM api_keys WHERE id=?", (kid,))
    conn.commit()
    conn.close()
    _audit((request.session.get("user") or {}).get("username", "admin"), "key.delete", {"id": kid})
    return {"ok": True}


@app.get("/api/admin/audit")
def admin_audit(request: Request, limit: int = 100):
    _require_admin(request)
    conn = db(readonly=True)
    rows = conn.execute("SELECT id,actor,action,detail,created FROM audit "
                        "ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    conn.close()
    return [{"id": r[0], "actor": r[1], "action": r[2], "detail": r[3], "created": r[4]}
            for r in rows]


# ==================== 项目 / 素材 ====================
class ProjectReq(BaseModel):
    name: str


@app.get("/api/projects")
def api_projects():
    conn = db(readonly=True)
    rows = conn.execute("SELECT id,name,owner,created FROM projects ORDER BY created DESC").fetchall()
    conn.close()
    return [{"id": r[0], "name": r[1], "owner": r[2], "created": r[3]} for r in rows]


@app.post("/api/projects")
def api_create_project(req: ProjectReq):
    name = (req.name or "").strip()
    if not name:
        raise HTTPException(400, "项目名不能为空")
    pid = uuid.uuid4().hex[:8]
    conn = db()
    conn.execute("INSERT INTO projects VALUES(?,?,?,?)", (pid, name, "", time.time()))
    conn.commit()
    conn.close()
    return {"ok": True, "id": pid, "name": name}


@app.get("/api/projects/{pid}/assets")
def api_project_assets(pid: str):
    conn = db(readonly=True)
    rows = conn.execute("SELECT id,prompt,results,created FROM jobs WHERE project=? ORDER BY created DESC LIMIT 100",
                        (pid,)).fetchall()
    conn.close()
    out = []
    for r in rows:
        try:
            res = json.loads(r[2] or "[]")
        except Exception:
            res = []
        out.append({"job": r[0], "prompt": r[1], "created": r[3],
                    "images": [x.get("image") for x in res if x.get("image")]})
    return out


# ==================== TTS 路由 ====================
def _tts_authed(apikey, request):
    return _api_authed(apikey, request) or bool(_api_key_user(request))


class TTSSpeakReq(BaseModel):
    variant: str = "custom"
    text: str = ""
    language: str = "Auto"
    speaker: Optional[str] = None
    instruct: Optional[str] = None
    apikey: Optional[str] = None


class TTSSpeechReq(BaseModel):
    input: str = ""
    voice: str = "Vivian"
    language: str = "Auto"
    response_format: str = "wav"
    apikey: Optional[str] = None


@app.get("/api/tts/status")
def api_tts_status():
    remote = _tts_json("/api/status")
    with _cond:
        loaded = _state.get("loaded") or _state.get("comfy_loaded")
    r = remote or {}
    state = r.get("state") or _tts["state"]
    if _tts["state"] in ("loading", "error"):
        state = _tts["state"]
    if remote and state == "ready":
        # TTS 服务自己知道真实的生成/排队数；BFF 的 active 只是“已受理”
        active = int(r.get("active", 0) or 0)
        pending = int(r.get("pending", 0) or 0)
    else:
        active = int(_tts.get("active", 0) or 0)
        pending = int(_tts.get("pending", 0) or 0)
    return {"loaded": bool(remote) and state == "ready", "state": state,
            "active": active, "pending": pending,
            "models": r.get("models", {}),
            "error": r.get("error") or _tts.get("error"),
            "image": {"loaded": loaded, "busy": _image_busy()}}


@app.post("/api/tts/load")
def api_tts_load(request: Request, apikey: Optional[str] = None):
    if not _tts_authed(apikey, request):
        raise HTTPException(401, "需要登录或 API Key")
    ok = _tts_begin_request(timeout=600)
    if ok:
        _tts_end_request()
    if not ok:
        raise HTTPException(503, _tts.get("error") or "TTS 启动失败")
    return {"ok": True, "state": _tts["state"]}


@app.post("/api/tts/unload")
def api_tts_unload(request: Request, apikey: Optional[str] = None):
    if not _tts_authed(apikey, request):
        raise HTTPException(401, "需要登录或 API Key")
    _tts_stop()
    return {"ok": True, "state": _tts["state"]}


@app.get("/api/tts/voices")
def api_tts_voices():
    code, body, _ = _tts_call("GET", "/api/voices", timeout=10)
    if code != 200:
        raise HTTPException(code if code >= 400 else 503, _tts_detail(body))
    return JSONResponse(json.loads(body))


def _tts_audio_response(body):
    return Response(content=body, media_type="audio/wav",
                    headers={"Content-Disposition": 'attachment; filename="tts.wav"'})


def _tts_generation(call):
    """TTS 生成统一入口：pending 只统计“还在等待”的请求。"""
    with _cond:
        _tts["pending"] += 1
    waiting = True
    try:
        if not _tts_begin_request(timeout=600):
            raise HTTPException(503, _tts.get("error") or "TTS 资源繁忙，请稍后重试")
        waiting = False
        with _cond:
            _tts["pending"] = max(0, _tts["pending"] - 1)
        try:
            return call()
        finally:
            _tts_end_request()
    finally:
        if waiting:
            with _cond:
                _tts["pending"] = max(0, _tts["pending"] - 1)


def _validate_tts_speak(variant, text, language, speaker, instruct):
    """BFF 侧先校验，避免非法请求触发停图像 + 加载 TTS 的 30~120s 切换。"""
    if variant not in ("custom", "design"):
        raise HTTPException(400, "variant 必须是 custom 或 design")
    if not (text or "").strip():
        raise HTTPException(400, "text 不能为空")
    if len(text) > 2000:
        raise HTTPException(400, "text 不能超过 2000 字")
    if not (language or "").strip():
        raise HTTPException(400, "language 不能为空")
    if variant == "custom" and not (speaker or "").strip():
        raise HTTPException(400, "custom 需要 speaker")
    if variant == "design" and not (instruct or "").strip():
        raise HTTPException(400, "design 需要 instruct")


@app.post("/api/tts/speak")
def api_tts_speak(req: TTSSpeakReq, request: Request):
    if not _tts_authed(req.apikey, request):
        raise HTTPException(401, "需要登录或 API Key")
    _validate_tts_speak(req.variant, req.text, req.language, req.speaker, req.instruct)
    payload = {"variant": req.variant, "text": req.text, "language": req.language,
               "speaker": req.speaker, "instruct": req.instruct}
    t0 = time.time()
    code, body, _ = _tts_generation(
        lambda: _tts_call("POST", "/api/tts/speak", body=payload, timeout=1800))
    elapsed = time.time() - t0
    if code != 200:
        raise HTTPException(code if code >= 400 else 503, _tts_detail(body))
    try:
        _tts_save_result(req.variant, req.text, req.language, req.speaker or "",
                         req.instruct or "", body, elapsed)
    except Exception:
        logger.warning("TTS 记录写入失败", exc_info=True)
    return _tts_audio_response(body)


@app.post("/api/tts/clone")
def api_tts_clone(request: Request,
                  text: str = Form(""), language: str = Form("Auto"),
                  ref_text: str = Form(""), x_vector_only_mode: str = Form("false"),
                  apikey: str = Form(""),
                  ref_audio: UploadFile = File(...)):
    if not _tts_authed(apikey, request):
        raise HTTPException(401, "需要登录或 API Key")
    if not (text or "").strip():
        raise HTTPException(400, "text 不能为空")
    if len(text) > 2000:
        raise HTTPException(400, "text 不能超过 2000 字")
    if not (language or "").strip():
        raise HTTPException(400, "language 不能为空")
    xv = (x_vector_only_mode or "").lower() in ("1", "true", "yes", "on")
    if not xv and not (ref_text or "").strip():
        raise HTTPException(400, "克隆需要 ref_text，或启用 x_vector_only_mode")
    data = ref_audio.file.read(TTS_MAX_UPLOAD + 1)
    if len(data) > TTS_MAX_UPLOAD:
        raise HTTPException(400, f"参考音频超过 {TTS_MAX_UPLOAD // (1024 * 1024)}MB")
    body, ctype = _multipart_encode(
        {"text": text, "language": language, "ref_text": ref_text,
         "x_vector_only_mode": x_vector_only_mode},
        {"ref_audio": (ref_audio.filename or "ref.wav", data,
                       ref_audio.content_type or "application/octet-stream")})
    t0 = time.time()
    code, resp, _ = _tts_generation(
        lambda: _tts_call("POST", "/api/tts/clone", data=body,
                          content_type=ctype, timeout=1800))
    elapsed = time.time() - t0
    if code != 200:
        raise HTTPException(code if code >= 400 else 503, _tts_detail(resp))
    try:
        _tts_save_result("clone", text, language, "", ref_text or "", resp, elapsed)
    except Exception:
        logger.warning("TTS 记录写入失败", exc_info=True)
    return _tts_audio_response(resp)


@app.post("/v1/audio/speech")
def v1_audio_speech(req: TTSSpeechReq, request: Request):
    if not _tts_authed(req.apikey, request):
        raise HTTPException(401, "需要 API Key（网页登录状态可直接调用）")
    if req.response_format not in ("wav",):
        raise HTTPException(400, "只支持 response_format=wav")
    if not (req.input or "").strip():
        raise HTTPException(400, "input 不能为空")
    if len(req.input) > 2000:
        raise HTTPException(400, "input 不能超过 2000 字")
    payload = {"variant": "custom", "text": req.input,
               "language": req.language, "speaker": req.voice, "instruct": None}
    t0 = time.time()
    code, body, _ = _tts_generation(
        lambda: _tts_call("POST", "/api/tts/speak", body=payload, timeout=1800))
    elapsed = time.time() - t0
    if code != 200:
        raise HTTPException(code if code >= 400 else 503, _tts_detail(body))
    try:
        _tts_save_result("custom", req.input, req.language, req.voice, "", body, elapsed)
    except Exception:
        logger.warning("TTS 记录写入失败", exc_info=True)
    return _tts_audio_response(body)


@app.get("/api/tts/history")
def api_tts_history(request: Request, limit: int = 10, offset: int = 0,
                    apikey: Optional[str] = None):
    if not _tts_authed(apikey, request):
        raise HTTPException(401, "需要登录或 API Key")
    limit = max(1, min(int(limit), 50))
    offset = max(0, int(offset))
    conn = db(readonly=True)
    try:
        total = conn.execute("SELECT COUNT(*) FROM tts_audio").fetchone()[0]
        rows = conn.execute(
            "SELECT id,created,variant,text,language,speaker,instruct,file,"
            "duration,elapsed,size FROM tts_audio ORDER BY created DESC "
            "LIMIT ? OFFSET ?", (limit, offset)).fetchall()
    finally:
        conn.close()
    keys = ["id", "created", "variant", "text", "language", "speaker", "instruct",
            "file", "duration", "elapsed", "size"]
    return {"rows": [dict(zip(keys, r)) for r in rows], "total": total,
            "limit": limit, "offset": offset}


@app.get("/api/tts/audio/{filename}")
def api_tts_audio(filename: str, request: Request,
                  apikey: Optional[str] = None):
    if not _tts_authed(apikey, request):
        raise HTTPException(401, "需要登录或 API Key")
    path = _tts_history_audio(filename)
    if not path:
        raise HTTPException(404, "音频不存在")
    return FileResponse(path, media_type="audio/wav")


@app.delete("/api/tts/history/{jid}")
def api_tts_history_delete(jid: str, request: Request,
                           apikey: Optional[str] = None):
    if not _tts_authed(apikey, request):
        raise HTTPException(401, "需要登录或 API Key")
    conn = db()
    try:
        row = conn.execute("SELECT file FROM tts_audio WHERE id=?", (jid,)).fetchone()
        if not row:
            raise HTTPException(404, "记录不存在")
        conn.execute("DELETE FROM tts_audio WHERE id=?", (jid,))
        conn.commit()
    finally:
        conn.close()
    path = _tts_history_audio(row[0])
    if path:
        try:
            os.remove(path)
        except OSError:
            pass
    return {"ok": True}


class TTSDeleteReq(BaseModel):
    ids: List[str] = []
    apikey: Optional[str] = None


@app.post("/api/tts/history/batch-delete")
def api_tts_history_batch_delete(req: TTSDeleteReq, request: Request):
    if not _tts_authed(req.apikey, request):
        raise HTTPException(401, "需要登录或 API Key")
    ids = list(dict.fromkeys(req.ids or []))[:200]
    if not ids:
        return {"ok": True, "deleted": 0}
    conn = db()
    try:
        q = ",".join("?" * len(ids))
        rows = conn.execute(f"SELECT file FROM tts_audio WHERE id IN ({q})",
                            ids).fetchall()
        conn.execute(f"DELETE FROM tts_audio WHERE id IN ({q})", ids)
        conn.commit()
    finally:
        conn.close()
    for row in rows:
        path = _tts_history_audio(row[0])
        if path:
            try:
                os.remove(path)
            except OSError:
                pass
    return {"ok": True, "deleted": len(rows)}


@app.post("/api/tts/history/batch-download")
def api_tts_history_batch_download(req: TTSDeleteReq, request: Request):
    """把选中的语音打包成 zip；单条请直接用 /api/tts/audio/{file}。"""
    if not _tts_authed(req.apikey, request):
        raise HTTPException(401, "需要登录或 API Key")
    ids = list(dict.fromkeys(req.ids or []))[:200]
    if not ids:
        raise HTTPException(400, "请选择要下载的记录")
    conn = db(readonly=True)
    try:
        q = ",".join("?" * len(ids))
        rows = conn.execute(
            f"SELECT id,created,variant,file FROM tts_audio WHERE id IN ({q})",
            ids).fetchall()
    finally:
        conn.close()
    buf = io.BytesIO()
    count = 0
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for jid, created, variant, name in rows:
            path = _tts_history_audio(name)
            if not path:
                continue
            ts = time.strftime("%Y%m%d-%H%M%S", time.localtime(created or 0))
            arc = f"{ts}_{variant or 'tts'}_{jid[:8]}.wav"
            z.write(path, arcname=arc)
            count += 1
    if not count:
        raise HTTPException(404, "没有可下载的音频")
    fn = f"tts-audio-{time.strftime('%Y%m%d-%H%M%S')}.zip"
    return Response(content=buf.getvalue(), media_type="application/zip",
                    headers={"Content-Disposition": f'attachment; filename="{fn}"'})


@app.get("/api/docs")
def api_docs():
    """下载完整的 API 调用说明（Markdown）。"""
    path = os.path.join(BASE_DIR, "docs", "ImageHub调用说明.md")
    if not os.path.isfile(path):
        raise HTTPException(404, "说明文档不存在")
    return FileResponse(path, media_type="text/markdown; charset=utf-8",
                        filename="ImageHub-API.md")


@app.get("/api/tls/cert")
def api_tls_cert():
    """下载 HTTPS 公钥证书；私钥永远不对外。"""
    path = os.environ.get("IMAGEHUB_TLS_CERT") or "/etc/imagehub/tls/imagehub.crt"
    if not os.path.isfile(path):
        raise HTTPException(404, "证书不存在")
    return FileResponse(path, media_type="application/x-x509-ca-cert",
                        filename="imagehub.crt")


@app.get("/api/version")
def api_version():
    return {"name": "ImageHub", "version": APP_VERSION, "build": APP_BUILD}


def _busy():
    with _cond:
        active = _state.get("active", 0)
        switching = _state.get("switching", False)
        image_pending = _state.get("image_pending", 0)
    with _lock:
        running_job = any(j.get("status") == "running" for j in _jobs.values())
    with _dq["lock"]:
        dq_running = _dq.get("running")
        dq_queued = bool(_dq.get("queue"))
    tts_busy = (_tts.get("state") == "loading" or _tts.get("active", 0) > 0
                or _tts.get("pending", 0) > 0)
    with _arbiter_lock:
        switch_waiting = bool(_arbiter_q)
    return bool(
        active or switching or image_pending or running_job or dq_running or dq_queued
        or _comfy_lock.locked() or tts_busy or switch_waiting
    )


@app.get("/api/status")
def api_status():
    backend = _settings.get("backend", "comfy")
    with _cond:
        direct = _state["loaded"]
        comfy = _state["comfy_loaded"]
        if direct and svc_alive(direct):
            loaded = direct          # 直连服务确实在跑，优先如实报告
            backend_now = "direct"
        elif comfy:
            loaded = comfy
            backend_now = "comfy"
        else:
            loaded = None
            backend_now = backend
        return {"loaded": loaded, "active": _state["active"],
                "switching": _state["switching"], "target": _state["target"],
                "backend": backend_now, "version": APP_VERSION,
                "tls": _tls_enabled(),
                "tts": {"state": _tts["state"], "active": _tts["active"],
                        "pending": _tts["pending"],
                        "error": _tts.get("error")},
                "busy": _busy()}


@app.get("/api/history")
def api_history(response: Response, limit: int = 30, offset: int = 0):
    limit = max(1, min(int(limit), 200))
    offset = max(0, int(offset))
    conn = db(readonly=True)
    total = conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0]
    rows = conn.execute("SELECT id,prompt,models,created,results,status FROM jobs "
                        "ORDER BY created DESC LIMIT ? OFFSET ?",
                        (limit, offset)).fetchall()
    conn.close()
    response.headers["X-Total-Count"] = str(total)
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
    with _lock:
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
        with _lock:
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
    with _lock:
        _jobs.pop(job_id, None)
    logger.info(f"删除记录 {job_id}，图片 {removed} 张")
    return {"ok": True, "deleted_images": removed}


@app.get("/thumb/{filename}")
def thumb(filename: str, w: int = 160):
    """缩略图：服务端缩放并缓存，避免浏览器解码几十 MB 原图"""
    try:
        out = _ensure_thumb(filename, w)
    except Exception as e:
        raise HTTPException(500, f"缩略图生成失败: {str(e)[:120]}")
    if not out:
        raise HTTPException(404, "not found")
    return FileResponse(out, media_type="image/jpeg",
                        headers={"Cache-Control": "public, max-age=86400"})


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
    is_admin = _upsert_user(user)
    user["is_admin"] = bool(is_admin)
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
