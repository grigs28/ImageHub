#!/usr/bin/env python3
"""三模型同载 spike：Qwen3-TTS 1.7B CustomVoice / VoiceDesign / Base。

在 GPU0/1 空闲时运行（不要有图像任务）：
    CUDA_VISIBLE_DEVICES=0,1 /opt/conda3/envs/qwen-tts/bin/python tools/tts_spike.py

输出每个模型加载后的显存、三次生成耗时，并把 wav 写到 /tmp/tts_spike/。
这是 spike 工具，不是运行时依赖。
"""
import json
import os
import subprocess
import time
import traceback
import urllib.request

import torch
import soundfile as sf

BASE = "/opt/models/Qwen3-TTS"
OUT = "/tmp/tts_spike"
os.makedirs(OUT, exist_ok=True)


def snap(tag):
    try:
        rows = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.used,memory.total",
             "--format=csv,noheader"], capture_output=True, text=True,
            timeout=10).stdout.strip().replace("\n", " | ")
    except Exception as e:
        rows = f"nvidia-smi failed: {e}"
    alloc = {}
    try:
        for i in range(torch.cuda.device_count()):
            alloc[i] = round(torch.cuda.memory_allocated(i) / 2**30, 2)
    except Exception as e:
        alloc = {"err": str(e)}
    print(f"[mem] {tag} torch_alloc_GiB={json.dumps(alloc)}", flush=True)
    print(f"[mem] {tag} {rows}", flush=True)


def load(path, dev, tag):
    t = time.perf_counter()
    try:
        m = Qwen3TTSModel.from_pretrained(path, device_map=dev, dtype=torch.bfloat16)
    except Exception:
        print(f"[load] {tag} device={dev} FAILED\n{traceback.format_exc()}", flush=True)
        snap(f"after failed load {tag}")
        return None
    dt = time.perf_counter() - t
    print(f"[load] {tag} device={dev} {dt:.1f}s", flush=True)
    snap(f"after load {tag}")
    return m


print("[start]", flush=True)
from qwen_tts import Qwen3TTSModel  # noqa: E402

m_custom = load(f"{BASE}/CustomVoice", "cuda:0", "custom")
m_design = load(f"{BASE}/VoiceDesign", "cuda:0", "design")
m_clone = load(f"{BASE}/Base", "cuda:1", "clone")

try:
    if m_custom:
        print("[info] speakers", m_custom.get_supported_speakers(), flush=True)
        print("[info] languages", m_custom.get_supported_languages(), flush=True)
except Exception as e:
    print("[info] failed", e, flush=True)

results = {}


def gen(tag, fn):
    t = time.perf_counter()
    try:
        wavs, sr = fn()
        dt = time.perf_counter() - t
        dur = sum(len(w) for w in wavs) / sr
        path = f"{OUT}/{tag}.wav"
        sf.write(path, wavs[0], sr)
        results[tag] = {"ok": True, "seconds": round(dt, 1),
                        "sr": sr, "dur": round(dur, 1), "file": path}
        print(f"[gen] {tag} ok {dt:.1f}s sr={sr} dur={dur:.1f}s", flush=True)
    except Exception:
        results[tag] = {"ok": False, "error": traceback.format_exc()[-1200:]}
        print(f"[gen] {tag} FAILED\n{traceback.format_exc()}", flush=True)
    snap(f"after gen {tag}")


if m_custom:
    gen("custom", lambda: m_custom.generate_custom_voice(
        text="你好，欢迎使用 ImageHub 语音合成。今天天气很好，适合出门散步。",
        language="Chinese", speaker="Vivian", instruct="用温柔平静的语气"))
else:
    print("[gen] custom skipped (load failed)", flush=True)

if m_design:
    gen("design", lambda: m_design.generate_voice_design(
        text="你好，这是声音设计测试。希望这个声音听起来自然、可信。",
        language="Chinese", instruct="一位年轻女性的声音，温柔、清晰、语速稍慢。"))
else:
    print("[gen] design skipped (load failed)", flush=True)

ref = f"{OUT}/ref.wav"
ref_text = ("Okay. Yeah. I resent you. I love you. I respect you. "
            "But you know what? You blew it! And thanks to you.")
if not os.path.exists(ref):
    try:
        urllib.request.urlretrieve(
            "https://qianwen-res.oss-cn-beijing.aliyuncs.com/"
            "Qwen3-TTS-Repo/clone.wav", ref)
        print("[ref] downloaded", flush=True)
    except Exception as e:
        print(f"[ref] download failed: {e}", flush=True)

if m_clone and os.path.exists(ref):
    gen("clone", lambda: m_clone.generate_voice_clone(
        text="Hello, this is a voice clone test for ImageHub.",
        language="English", ref_audio=ref, ref_text=ref_text))
else:
    print("[gen] clone skipped (load failed or no ref)", flush=True)

print("[summary] " + json.dumps(results, ensure_ascii=False), flush=True)
print("[done]", flush=True)
