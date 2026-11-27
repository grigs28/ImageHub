# ImageHub TTS + HTTPS Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 ImageHub 增加原生 TTS 页面，用一个内部服务同时加载 Qwen3-TTS 1.7B 三个变体，与图像模型严格互斥排队；并把 ImageHub 切到 HTTPS。

**Architecture:** 新增独立 Python 包 `tts_service/`（FastAPI，127.0.0.1:5570，独立 `qwen-tts` 环境），BFF `app.py` 负责子进程起停、反向代理、鉴权和 GPU 资源仲裁；`index.html` 增加原生 TTS tab（不使用 Gradio）；`run.py` 直接给 uvicorn 加 TLS。

**Tech Stack:** FastAPI、Pydantic、uvicorn、Qwen3-TTS 官方 `qwen-tts==0.1.1`、PyTorch（cu12x）、ffmpeg 7.1、SQLite（仅图像历史）、原生 JS/HTML。

**Spec:** `docs/superpowers/specs/2026-11-27-imagehub-tts-design.md`

## Global Constraints

- 当前工作区：`/opt/vLLM/ImageHub/.claude/worktrees/imagehub-library-perf`；**不提交 git**（沿用当前约定），以测试和工作区 diff 作为交付证据。
- 界面和文档用中文；TTS 输出只经 Web UI/API 返回，**不写 `jobs` 表、不生成缩略图、不进图库**。
- BFF 用 `/opt/conda3/bin/python`（测试环境）或 `/opt/conda3/envs/imagegen/bin/python`（生产 BFF）；TTS 服务用 `/opt/conda3/envs/qwen-tts/bin/python`。
- 端口：ImageHub 5569（HTTPS）、TTS 服务 127.0.0.1:5570、ComfyUI 5568、直连 5564–5567、vLLM 5563。
- 模型目录：`/opt/models/Qwen3-TTS/{CustomVoice,VoiceDesign,Base}`。
- GPU：图像和 TTS 都只用物理 GPU0/1；GPU2/3 归 vLLM。TTS 放置 `cuda:0, cuda:0, cuda:1`。
- HTTPS 证书：`/etc/imagehub/tls/imagehub.crt|key`，自签，`SAN IP:192.168.0.70,IP:127.0.0.1`；私钥 `600 root`；内部健康检查用 `--cacert` 校验，不关闭全局 TLS 校验。
- yz-login 不修改；`YZ_LOGIN_URL` 保持 `http://192.168.0.8`。
- 所有新代码先写失败测试，再实现；测试命令统一 `python -m unittest discover -s tests -p "<file>" -v`。

## Review Focus

1. TTS 三个模型加载失败或超时：用户点生成后必须得到明确错误并回到可重试状态，不能永久卡在“加载中”。
2. 图像任务正在生成时用户点 TTS：必须排队等待，不能停掉正在生成图像的服务。
3. TTS 正在生成时图像 API 请求：必须等 TTS 完成再切换，不能同时加载导致显存 OOM。
4. 参考音频损坏、超长（>60s）、超大（>20MB）：返回 400，临时文件必须清理。
5. 自签 HTTPS 下登录回跳和 `deploy-guard.sh` 健康检查：不能因证书校验失败而误判 busy 或误重启。

---

### Task 1: TTS 环境与三模型同载 spike

**Files:**
- Create: `tools/tts_spike.py`
- Create: `.superpowers/sdd/2026-11-27-imagehub-tts/tts-spike.md`（结果记录，git-ignored）

**Interfaces:**
- Consumes: 设计文档 §4.3 的模型仓库与放置。
- Produces: 实测结论（每模型显存、总显存、加载耗时、三变体生成耗时）；后续任务的模型路径和 `devices` 常量以此为准。

- [ ] **Step 1: 建独立环境并安装**

Run:
```bash
/usr/bin/python3 -m venv /opt/conda3/envs/qwen-tts
/opt/conda3/envs/qwen-tts/bin/pip install --upgrade pip setuptools wheel
/opt/conda3/envs/qwen-tts/bin/pip install torch torchaudio
/opt/conda3/envs/qwen-tts/bin/pip install qwen-tts==0.1.1
```
Expected: `torch.cuda.is_available()` 为 True，`transformers==4.57.3`。

- [ ] **Step 2: 下载三个模型 + tokenizer**

Run:
```bash
export HF_HUB_DISABLE_XET=1
HF=/opt/conda3/envs/imagegen/bin/hf
D=/opt/models/Qwen3-TTS
$HF download Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice --local-dir $D/CustomVoice
$HF download Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign --local-dir $D/VoiceDesign
$HF download Qwen/Qwen3-TTS-12Hz-1.7B-Base        --local-dir $D/Base
$HF download Qwen/Qwen3-TTS-Tokenizer-12Hz        --local-dir $D/Tokenizer
```
Expected: 每个模型目录约 4.5GB（含自带 speech_tokenizer）。

- [ ] **Step 3: 运行 spike**

Run:
```bash
CUDA_VISIBLE_DEVICES=0,1 /opt/conda3/envs/qwen-tts/bin/python tools/tts_spike.py
```
Expected: 三个模型都加载成功；打印每步显存；`/tmp/tts_spike/{custom,design,clone}.wav` 生成；总显存 < 40 GiB。

- [ ] **Step 4: 记录结论并做决策**

把实测数字写入 `.superpowers/sdd/2026-11-27-imagehub-tts/tts-spike.md`：
- 每模型加载后 `torch_alloc_GiB` 与 `nvidia-smi`；
- 三个加载完成后的总占用；
- 三次生成耗时和音频时长；
- 结论：三模型同载可行 / 需降级为“两个常驻 + 一个按需”。

若不可行：**停止后续任务，向用户报告实测数字和降级建议**，不要擅自改设计。

---

### Task 2: TTS 服务骨架（FakeEngine + API + 队列）

**Files:**
- Create: `tts_service/__init__.py`
- Create: `tts_service/audio.py`（仅 `wav_bytes`）
- Create: `tts_service/models.py`（`BaseEngine`、`FakeEngine`、`SPEAKER_META`、`LANGUAGES`）
- Create: `tts_service/app.py`（`create_app`、`main`）
- Create: `tts_service/requirements.txt`
- Create: `tests/test_tts_service.py`

**Interfaces:**
- Produces:
  - `FakeEngine(delay: float = 0.0)`，属性 `max_parallel: int`、`SPEAKERS`、`LANGUAGES`。
  - `create_app(engine: BaseEngine) -> FastAPI`；`app.state.ready` 是 `threading.Event`；`app.state.tts` 是 `{"state","error","active","pending"}`。
  - `BaseEngine.speak_custom(text, language, speaker, instruct=None) -> tuple[list[float], int]`；
    `speak_design(text, language, instruct) -> tuple[list[float], int]`；
    `speak_clone(text, language, ref_audio_path, ref_text, x_vector_only_mode) -> tuple[list[float], int]`。
  - `audio.wav_bytes(samples: list[float], sr: int) -> bytes`（stdlib `wave`，16-bit PCM）。

- [ ] **Step 1: 写失败测试**

`tests/test_tts_service.py`：
```python
import io, threading, unittest, wave
from fastapi.testclient import TestClient
from tts_service.app import create_app
from tts_service.models import FakeEngine


class TTSServiceTests(unittest.TestCase):
    def make(self, delay=0.0):
        engine = FakeEngine(delay=delay)
        app = create_app(engine)
        self.assertTrue(app.state.ready.wait(3))
        return engine, TestClient(app)

    def test_health_and_status_ready(self):
        _, c = self.make()
        self.assertEqual(c.get("/health").json()["state"], "ready")
        d = c.get("/api/status").json()
        self.assertEqual(d["active"], 0)
        self.assertEqual(d["pending"], 0)
        self.assertEqual(set(d["models"]), {"custom", "design", "clone"})

    def test_voices(self):
        _, c = self.make()
        d = c.get("/api/voices").json()
        self.assertIn("Vivian", [s["name"] for s in d["speakers"]])
        self.assertIn("Chinese", d["languages"])

    def test_speak_custom_returns_wav(self):
        _, c = self.make()
        r = c.post("/api/tts/speak", json={"variant": "custom", "text": "你好",
                                           "language": "Chinese", "speaker": "Vivian"})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.headers["content-type"], "audio/wav")
        self.assertTrue(r.content.startswith(b"RIFF"))

    def test_speak_validation_400(self):
        _, c = self.make()
        self.assertEqual(c.post("/api/tts/speak", json={"variant": "custom", "text": ""}).status_code, 400)
        self.assertEqual(c.post("/api/tts/speak", json={"variant": "custom", "text": "x"}).status_code, 400)
        self.assertEqual(c.post("/api/tts/speak", json={"variant": "design", "text": "x"}).status_code, 400)

    def test_queue_serializes(self):
        engine, c = self.make(delay=0.25)
        threads = [threading.Thread(target=lambda: TestClient(c.app).post(
            "/api/tts/speak", json={"variant": "custom", "text": "你好",
                                    "language": "Chinese", "speaker": "Vivian"}))
                   for _ in range(2)]
        [t.start() for t in threads]
        [t.join(5) for t in threads]
        self.assertEqual(engine.max_parallel, 1)

    def test_clone_requires_ref_text_unless_xvector(self):
        _, c = self.make()
        wav = io.BytesIO()
        with wave.open(wav, "wb") as w:
            w.setnchannels(1); w.setsampwidth(2); w.setframerate(24000)
            w.writeframes(b"\x00\x00" * 2400)
        data = {"text": "你好", "language": "Chinese"}
        r = c.post("/api/tts/clone", data=data,
                   files={"ref_audio": ("a.wav", wav.getvalue(), "audio/wav")})
        self.assertEqual(r.status_code, 400)
        r = c.post("/api/tts/clone", data={**data, "x_vector_only_mode": "true"},
                   files={"ref_audio": ("a.wav", wav.getvalue(), "audio/wav")})
        self.assertEqual(r.status_code, 200)
```
Run: `python -m unittest discover -s tests -p "test_tts_service.py" -v`
Expected: FAIL（`tts_service` 不存在）。

- [ ] **Step 2: 实现 `tts_service/audio.py` 的 `wav_bytes`**

`wav_bytes(samples: list[float], sr: int) -> bytes`：用 `wave` + `struct` 写 16-bit PCM 单声道；把 `[-1,1]` 裁剪后乘 32767。

- [ ] **Step 3: 实现 `tts_service/models.py`**

`SPEAKER_META` 按官方表写 9 项：
`Vivian`（Bright, slightly edgy young female voice. / Chinese）、`Serena`（Warm, gentle young female voice. / Chinese）、`Uncle_Fu`（Seasoned male voice with a low, mellow timbre. / Chinese）、`Dylan`（Youthful Beijing male voice with a clear, natural timbre. / Chinese (Beijing Dialect)）、`Eric`（Lively Chengdu male voice with a slightly husky brightness. / Chinese (Sichuan Dialect)）、`Ryan`（Dynamic male voice with strong rhythmic drive. / English）、`Aiden`（Sunny American male voice with a clear midrange. / English）、`Ono_Anna`（Playful Japanese female voice with a light, nimble timbre. / Japanese）、`Sohee`（Warm Korean female voice with rich emotion. / Korean）。

`LANGUAGES = ["Auto","Chinese","English","Japanese","Korean","German","French","Russian","Portuguese","Spanish","Italian"]`。

`FakeEngine`：
- `load()` 直接返回；
- `status()` 返回 `{"state":"ready","models":{"custom":"ready","design":"ready","clone":"ready"}}`；
- `voices()` 返回 `{"speakers": SPEAKER_META, "languages": LANGUAGES}`；
- 三个 `speak_*` 用 `threading.Lock` 维护 `max_parallel`，`time.sleep(delay)`，返回 0.5s 440Hz 正弦波。

- [ ] **Step 4: 实现 `tts_service/app.py`**

`create_app(engine)`：
- `app.state.tts = {"state":"loading","error":None,"active":0,"pending":0}`；`app.state.ready = threading.Event()`；`app.state.gen_lock = threading.Lock()`；`app.state.cond = threading.Condition()`。
- 启动 daemon 线程执行 `engine.load()`，成功置 `state="ready"`、失败置 `"error"` 并记录 `error`，最后 `ready.set()`。
- `GET /health` → `{"ok": state=="ready", "state": ..., "error": ...}`。
- `GET /api/status` → `{"state","error","active","pending","models"}`，`models` 来自 `engine.status()`。
- `GET /api/voices` → `engine.voices()`，非 ready 返回 503。
- 生成入口统一封装：
  - 非 ready → 503；
  - `pending += 1`；`gen_lock.acquire()`；`pending -= 1`、`active += 1`；
  - 生成；`audio.wav_bytes`；返回 `Response(media_type="audio/wav")`；
  - `finally`：`active -= 1`、`cond.notify_all()`、释放锁。
- `POST /api/tts/speak`：校验 `variant in {custom,design}`、`text` 非空且 ≤2000 字、`language` 非空；custom 需要 `speaker`；design 需要 `instruct`。失败 `HTTPException(400, 中文原因)`。
- `POST /api/tts/clone`：`UploadFile ref_audio`、`Form text/language/ref_text/x_vector_only_mode`；`x_vector_only_mode` 为 `"true"` 时允许无 `ref_text`，否则必须提供。
- `POST /v1/audio/speech`：`model/input/voice/response_format/language`，映射到 `speak_custom`。
- `main()`：`argparse` 接受 `--fake --host --port --custom --design --clone --devices`；构造 `FakeEngine` 或（Task 3 的）`QwenEngine`；`uvicorn.run(create_app(engine), ...)`。

- [ ] **Step 5: 跑测试**

Run: `python -m unittest discover -s tests -p "test_tts_service.py" -v`
Expected: PASS（6 个测试）。

---

### Task 3: 真实 Qwen 引擎与参考音频处理

**Files:**
- Modify: `tts_service/models.py`（增加 `QwenEngine`、`create_engine`）
- Modify: `tts_service/audio.py`（增加 `convert_to_wav`、`wav_duration`、`validate_ref_audio`）
- Create: `tests/test_tts_audio.py`
- Create: `tests/test_tts_engine.py`

**Interfaces:**
- Consumes: Task 2 的 `BaseEngine`、`wav_bytes`。
- Produces:
  - `QwenEngine(custom_path, design_path, clone_path, devices=("cuda:0","cuda:0","cuda:1"))`
  - `create_engine(fake: bool, paths: dict, devices: tuple[str,str,str]) -> BaseEngine`
  - `audio.convert_to_wav(src: str, dst: str, sr: int = 24000, max_seconds: float = 60.0, max_bytes: int = 20*1024*1024) -> None`
  - `audio.wav_duration(path: str) -> float`
  - `audio.validate_ref_audio(path: str, max_seconds=60.0, max_bytes=20*1024*1024) -> None`

- [ ] **Step 1: 写失败测试**

`tests/test_tts_audio.py`：
```python
import math, os, struct, tempfile, unittest, wave
from tts_service import audio


def make_wav(path, seconds=1.0, sr=16000, channels=2):
    with wave.open(path, "wb") as w:
        w.setnchannels(channels); w.setsampwidth(2); w.setframerate(sr)
        n = int(seconds * sr)
        w.writeframes(b"\x00\x00" * channels * n)


class AudioTests(unittest.TestCase):
    def test_wav_bytes_roundtrip(self):
        data = audio.wav_bytes([0.0, 0.5, -0.5], 24000)
        with wave.open(__import__("io").BytesIO(data)) as w:
            self.assertEqual((w.getnchannels(), w.getframerate()), (1, 24000))

    def test_convert_to_24k_mono(self):
        d = tempfile.mkdtemp()
        src = os.path.join(d, "in.wav"); dst = os.path.join(d, "out.wav")
        make_wav(src, 1.0, 16000, 2)
        audio.convert_to_wav(src, dst)
        with wave.open(dst) as w:
            self.assertEqual((w.getnchannels(), w.getframerate()), (1, 24000))
            self.assertAlmostEqual(w.getnframes() / 24000, 1.0, places=1)

    def test_convert_rejects_long_audio(self):
        d = tempfile.mkdtemp()
        src = os.path.join(d, "long.wav"); dst = os.path.join(d, "out.wav")
        make_wav(src, 61.0, 1000, 1)
        with self.assertRaises(ValueError):
            audio.convert_to_wav(src, dst)

    def test_convert_rejects_large_file(self):
        d = tempfile.mkdtemp()
        src = os.path.join(d, "big.wav")
        with open(src, "wb") as f:
            f.write(b"\0" * (20 * 1024 * 1024 + 1))
        with self.assertRaises(ValueError):
            audio.convert_to_wav(src, os.path.join(d, "out.wav"))
```
`tests/test_tts_engine.py`：
```python
import unittest
from tts_service.models import QwenEngine


class FakeModel:
    def __init__(self):
        self.calls = []
    def get_supported_speakers(self): return ["Vivian"]
    def get_supported_languages(self): return ["Chinese"]
    def generate_custom_voice(self, **kw): self.calls.append(("custom", kw)); return [0.1, 0.2], 24000
    def generate_voice_design(self, **kw): self.calls.append(("design", kw)); return [0.1], 24000
    def generate_voice_clone(self, **kw): self.calls.append(("clone", kw)); return [0.1], 24000


class QwenEngineTests(unittest.TestCase):
    def engine(self):
        e = QwenEngine("c", "d", "b")
        e._custom = FakeModel(); e._design = FakeModel(); e._clone = FakeModel()
        return e

    def test_speak_custom_passes_speaker_and_instruct(self):
        e = self.engine()
        e.speak_custom("你好", "Chinese", "Vivian", "温柔")
        kind, kw = e._custom.calls[0]
        self.assertEqual((kind, kw["text"], kw["speaker"], kw["instruct"]),
                         ("custom", "你好", "Vivian", "温柔"))

    def test_speak_design_requires_instruct(self):
        e = self.engine()
        with self.assertRaises(ValueError):
            e.speak_design("你好", "Chinese", "")
        e.speak_design("你好", "Chinese", "年轻女性")
        self.assertEqual(e._design.calls[0][0], "design")

    def test_speak_clone_passes_ref_audio(self):
        e = self.engine()
        e.speak_clone("你好", "Chinese", "/tmp/a.wav", "参考文本", False)
        kind, kw = e._clone.calls[0]
        self.assertEqual((kind, kw["ref_audio"], kw["ref_text"]), ("clone", "/tmp/a.wav", "参考文本"))
```
Run: `python -m unittest discover -s tests -p "test_tts_audio.py" -v` 与 `... -p "test_tts_engine.py" -v`
Expected: FAIL。

- [ ] **Step 2: 实现音频处理**

- `wav_duration(path)`：用 `wave` 读 `getnframes()/getframerate()`。
- `validate_ref_audio`：文件大小 > `max_bytes` 或时长 > `max_seconds` → `ValueError("参考音频超过 60 秒")` / `ValueError("参考音频超过 20MB")`。
- `convert_to_wav`：先 `validate_ref_audio`；`subprocess.run(["ffmpeg","-y","-i",src,"-ac","1","-ar",str(sr),dst], check=True, capture_output=True)`；失败抛 `ValueError("无法解析音频")`。

- [ ] **Step 3: 实现 `QwenEngine`**

- `load()` 内延迟导入：`import torch; from qwen_tts import Qwen3TTSModel`；三次 `from_pretrained(path, device_map=dev, dtype=torch.bfloat16)`；保存到 `_custom/_design/_clone`。
- `status()` 返回三个模型 ready 和 `devices`。
- `voices()` 返回 `{"speakers": SPEAKER_META, "languages": self._custom.get_supported_languages()}`。
- `speak_custom` 调 `self._custom.generate_custom_voice(text=..., language=..., speaker=..., instruct=...)`；`speak_design` 校验 `instruct` 非空；`speak_clone` 调 `generate_voice_clone(text=..., language=..., ref_audio=path, ref_text=..., x_vector_only_mode=...)`。
- `create_engine(fake, paths, devices)`：fake 返回 `FakeEngine()`，否则 `QwenEngine(paths["custom"], paths["design"], paths["clone"], devices)`。

- [ ] **Step 4: 跑测试**

Run: `python -m unittest discover -s tests -p "test_tts_audio.py" -v`；`... -p "test_tts_engine.py" -v`
Expected: PASS。

- [ ] **Step 5: 真实 GPU 冒烟（非单元测试）**

Run:
```bash
CUDA_VISIBLE_DEVICES=0,1 /opt/conda3/envs/qwen-tts/bin/python -m tts_service.app --port 5570 \
  --custom /opt/models/Qwen3-TTS/CustomVoice \
  --design /opt/models/Qwen3-TTS/VoiceDesign \
  --clone /opt/models/Qwen3-TTS/Base
curl -s http://127.0.0.1:5570/health
curl -s -X POST http://127.0.0.1:5570/api/tts/speak -H 'Content-Type: application/json' \
  -d '{"variant":"custom","text":"你好，这是 ImageHub TTS 冒烟测试。","language":"Chinese","speaker":"Vivian"}' \
  -o /tmp/tts_smoke.wav -w '%{http_code} %{size_download}\n'
```
Expected: health ready；HTTP 200；`/tmp/tts_smoke.wav` 可播放。

---

### Task 4: BFF 代理、状态、鉴权、无历史

**Files:**
- Modify: `app.py`
- Create: `tests/test_tts_bff.py`

**Interfaces:**
- Consumes: Task 2/3 的 TTS 服务 API。
- Produces:
  - `TTS_PORT = int(os.environ.get("IMAGEHUB_TTS_PORT", "5570"))`
  - `TTS_PY = os.environ.get("IMAGEHUB_TTS_PY", "/opt/conda3/envs/qwen-tts/bin/python")`
  - `_tts = {"proc": None, "state": "stopped", "error": None, "active": 0, "pending": 0}`
  - `_tts_json(path, timeout=5) -> dict | None`
  - `_tts_call(method, path, body=None, files=None, timeout=600) -> tuple[int, bytes, str]`
  - `_tts_start(timeout=300) -> bool`
  - `_tts_stop(timeout=60) -> None`
  - `_tts_adopt() -> None`
  - 路由：`GET /api/tts/status`、`GET /api/tts/voices`、`POST /api/tts/load`、`POST /api/tts/unload`、`POST /api/tts/speak`、`POST /api/tts/clone`、`POST /v1/audio/speech`。

- [ ] **Step 1: 写失败测试**

`tests/test_tts_bff.py` 先按 `tests/test_library_perf.py` 的方式设置 `IMAGEHUB_*` 临时目录，再 `import app as imagehub`。加一个 stub TTS 服务：

```python
class StubTTS(BaseHTTPRequestHandler):
    def do_GET(self):
        body = {"health": {"ok": True, "state": "ready"},
                "api/status": {"state": "ready", "active": 0, "pending": 0,
                               "models": {"custom": "ready", "design": "ready", "clone": "ready"}},
                "api/voices": {"speakers": [{"name": "Vivian"}], "languages": ["Chinese"]}}
        d = body.get(self.path.lstrip("/"), {"error": "not found"})
        self.send_response(200); self.send_header("Content-Type", "application/json")
        self.end_headers(); self.wfile.write(json.dumps(d).encode())
```
测试：
- `test_tts_status_merges_remote`：`GET /api/tts/status` → `loaded is True`、`state=="ready"`、`image` 字段存在。
- `test_tts_voices_proxies`：返回 Vivian。
- `test_speak_without_auth_401`：`POST /api/tts/speak` 无 key、无 session → 401。
- `test_speak_with_key_returns_wav`：插入 `api_keys` 行（`_hash_key("k")`），请求体 `apikey="k"` → 200 + `audio/wav`。
- `test_speak_does_not_write_jobs`：调用前后 `SELECT COUNT(*) FROM jobs` 不变。
- `test_v1_audio_speech_openai_shape`：`POST /v1/audio/speech` + Bearer key → 200，且 stub 收到 `voice`/`input` 映射。
- `test_load_unload_calls_start_stop`：monkeypatch `imagehub._tts_start` / `_tts_stop` 记录调用。

Run: `python -m unittest discover -s tests -p "test_tts_bff.py" -v`
Expected: FAIL。

- [ ] **Step 2: 实现代理与状态**

- `_tts_json`：`urllib.request.urlopen(f"http://127.0.0.1:{TTS_PORT}{path}", timeout=...)`，异常返回 `None`。
- `_tts_call`：用 `urllib.request.Request`，`body` 为 dict 时 JSON，`files` 为 multipart 时手工拼 boundary；返回 `(status, bytes, content_type)`。
- `GET /api/tts/status`：
  - 远程 status = `_tts_json("/api/status")`；
  - 返回 `{"loaded": bool(remote), "state": remote.get("state") if remote else _tts["state"],
    "active": remote.get("active", 0), "pending": remote.get("pending", 0),
    "models": remote.get("models", {}), "error": _tts.get("error"),
    "image": {"loaded": _state["loaded"] or _state["comfy_loaded"], "busy": _image_busy()}}`。
- `GET /api/tts/voices` → 代理。
- `_tts_start`：`subprocess.Popen([TTS_PY, "-m", "tts_service.app", "--port", str(TTS_PORT), ...], cwd=BASE_DIR, env={...,"CUDA_VISIBLE_DEVICES":"0,1","PYTHONPATH":BASE_DIR,"HF_HUB_OFFLINE":"1"}, stdout=log, stderr=log)`；轮询 `/health` 直到 ready 或超时；失败 kill 并置 `state="error"`。
- `_tts_stop`：`proc.terminate()` → `wait(30)` → `kill()`；把 `state` 置 `"stopped"`；等待 5570 端口关闭。
- `_tts_adopt`：启动时若 `/health` 正常则置 `state="ready"`；否则读 `run/tts.pid` 清理残留。
- 生成路由：鉴权 `_tts_authed(req, request)`：`_api_authed(req.apikey, request)` 或 `_api_key_user(request)` 非空；否则 401。
  - `/api/tts/speak`：转 JSON 到 `/api/tts/speak`，返回 `Response(content, media_type="audio/wav")`。
  - `/api/tts/clone`：接收 `UploadFile` + `Form`，转 multipart。
  - `/v1/audio/speech`：映射 `input→text`、`voice→speaker`、`language` 默认 `Auto`，转 `/api/tts/speak`。
- 不调用 `record_job()`。
- `api_status` 增加 `"tts": {"state": _tts["state"], "active": _tts["active"], "pending": _tts["pending"]}`。

- [ ] **Step 3: 跑测试**

Run: `python -m unittest discover -s tests -p "test_tts_bff.py" -v`
Expected: PASS。

---

### Task 5: GPU 资源仲裁（图像 ↔ TTS 互斥 + FIFO + busy）

**Files:**
- Modify: `app.py`
- Create: `tests/test_tts_arbiter.py`
- Modify: `tests/test_deploy_guard.py`（若 busy 字段变化）

**Interfaces:**
- Consumes: Task 4 的 `_tts`、`_tts_start`、`_tts_stop`、`_tts_json`。
- Produces:
  - `_arbiter_acquire(timeout=600) -> bool`、`_arbiter_release() -> None`（FIFO 票号队列）。
  - `_image_busy() -> bool`：`_state["active"]>0 or _state["switching"] or _state["image_pending"]>0 or _comfy_lock.locked() or _dq running/queue`。
  - `_tts_begin_request(timeout=600) -> bool`：仲裁 → 等图像空闲 → 停图像后端 → 起 TTS → `_tts["active"] += 1`。
  - `_tts_end_request() -> None`：`_tts["active"] -= 1`、`cond.notify_all()`。
  - `_tts_release_for_image(timeout=900) -> None`：仲裁 → 等 TTS active 归零 → 停 TTS → `_state["image_pending"] += 1`。
  - `_image_done() -> None`：`_state["image_pending"] -= 1`、`cond.notify_all()`。
  - `_image_session()`：contextmanager，进 `_tts_release_for_image`，出 `_image_done`。

- [ ] **Step 1: 写失败测试**

`tests/test_tts_arbiter.py`：
- `test_image_waits_for_tts_active`：`_tts["active"]=1`；线程调用 `_tts_release_for_image()`；0.2s 后仍未返回；置 active 0 + notify；返回，且 `_tts_stop` 被调用。
- `test_image_stops_idle_tts`：`_tts["state"]="ready"`、active 0；调用后 `_tts_stop` 被调用、`_state["image_pending"]==1`；再 `_image_done()`。
- `test_tts_waits_for_active_image`：`_state["active"]=1`；线程调用 `_tts_begin_request()`；0.2s 未返回；active 0 + notify 后返回、`_tts_start` 被调用、`_tts["active"]==1`；`_tts_end_request()`。
- `test_tts_stops_idle_image`：`_state["loaded"]="qwen"`；monkeypatch `stop_service`、`_stop_comfy`、`_tts_start`；调用后全部 direct 服务与 ComfyUI 被停。
- `test_arbiter_fifo_order`：三个线程按顺序 acquire/release，记录进入顺序，断言 0→1→2。
- `test_busy_includes_tts_loading`：`_tts["state"]="loading"` → `_busy()` True；`ready`/active 0 → 不因 TTS 变 True。

Run: `python -m unittest discover -s tests -p "test_tts_arbiter.py" -v`
Expected: FAIL。

- [ ] **Step 2: 实现仲裁**

- `_state` 增加 `"image_pending": 0`。
- FIFO 队列：`collections.deque` + `threading.Event`（见设计 §4.6）；超时移除自身并把下一个置位。
- `_image_busy()` 用 `_cond` 读 `_state`，`_dq` 用 `_dq["lock"]`，`_comfy_lock.locked()`。
- 等待循环用 `_cond.wait(2)` 轮询，避免漏通知；超时返回 False → 上层返回 503「排队超时」。
- `_tts_release_for_image`：`_arbiter_acquire` → 等 `_tts["active"]==0` → `_tts_stop()` → `_state["image_pending"] += 1` → release。
- `_tts_begin_request`：`_arbiter_acquire` → 若 TTS ready 直接 active+1；否则等 `_image_busy()` 为 False → 对所有 `SERVICES` 调 `stop_service`、`_stop_comfy()` → `_tts_start()` → active+1 → release。
- 接入点：
  - `run_model` 整体包 `with _image_session():`；
  - `api_edit` 在参数校验后包 `with _image_session():`；
  - `_design_generate` 整体包 `with _image_session():`；
  - 避免嵌套：`run_model` 内部不重复开 session；`api_design_batch` 的 controlnet 分支只经 `_design_generate`/`run_model` 之一。
- `_busy()` 增加 `_tts["state"]=="loading"`、`_tts["active"]>0`、`_tts["pending"]>0`、`_arbiter_busy`。
- BFF 启动时调用 `_tts_adopt()`。

- [ ] **Step 3: 跑测试（含现有回归）**

Run: `python -m unittest discover -s tests -v`
Expected: 全部 PASS（含图库 14 个 + TTS 新增）。

---

### Task 6: 原生 TTS Web UI

**Files:**
- Modify: `index.html`
- Create: `tools/verify_tts_ui.py`

**Interfaces:**
- Consumes: `/api/tts/status`、`/api/tts/voices`、`/api/tts/speak`、`/api/tts/clone`、`/api/tts/load`、`/api/tts/unload`、`/api/status`。
- Produces: 导航项 `TTS`；页面 `data-page="tts"`；JS 函数 `loadTTS()`、`ttsStatus()`、`ttsSpeak()`、`ttsClone()`、`ttsLoad()`、`ttsUnload()`、`renderTTSStatus()`。

- [ ] **Step 1: 写 Playwright 验收脚本（先失败）**

`tools/verify_tts_ui.py` 复用 `tools/verify_library_perf.py` 的 mock server 方式，加载 `index.html`，mock 以上 API。断言：
- A：导航有 `TTS`，点击后 `.page[data-page="tts"]` 可见；
- B：三个子 tab 文本为“自定义音色 / 声音设计 / 声音克隆”，切换后对应字段可见；
- C：`/api/tts/voices` 的 `Vivian` 出现在音色下拉；
- D：自定义音色点生成，请求体 `variant=custom`、`speaker=Vivian`，页面出现 `<audio>` 和下载链接；
- E：声音设计不填指令点生成，不发出请求且显示中文错误；
- F：克隆 tab 有文件上传；`navigator.mediaDevices` 被删除时录音按钮隐藏；
- G：`/api/tts/status` 返回 `state=loading` 时页面显示“正在加载三个模型”，返回 `pending=1` 时显示“排队中”；
- H：`/api/status` 的 `image.busy=true` 时页面提示“等待图像任务完成”。

Run: `python3 tools/verify_tts_ui.py`
Expected: FAIL（页面还没有 TTS）。

- [ ] **Step 2: 加导航和页面骨架**

- 导航在“图库”后加 `<a href="#" data-page="tts">TTS</a>`。
- 新增 `<div class="page" data-page="tts">`，包含：
  - 状态卡 `#ttsStatus`：TTS 状态、active/pending、图像侧状态、按钮「加载 TTS」「释放 TTS」；
  - 子 tab 按钮 `data-tts-tab="custom|design|clone"`；
  - 三个表单容器；
  - 输出区 `#ttsOut`（audio + 下载链接 + 耗时）。
- `switchPage` 增加 `else if(name==="tts"){ loadTTS(); }`；`pollWhenVisible` 增加 `pollTTS`（5s，仅 TTS 页）。

- [ ] **Step 3: 实现表单与请求**

- 自定义音色：`#ttsSpeaker`（来自 `/api/tts/voices`）、`#ttsLangCustom`、`#ttsInstructCustom`、`#ttsTextCustom`。
- 声音设计：`#ttsInstructDesign`（必填）、`#ttsLangDesign`、`#ttsTextDesign`。
- 声音克隆：`#ttsRefFile`、`#ttsRecord`（`mediaDevices` 存在才显示）、`#ttsRefText`、`#ttsLangClone`、`#ttsTextClone`。
- 生成：`fetch("/api/tts/speak", {method:"POST", headers, body})`；克隆用 `FormData` 发 `/api/tts/clone`；
- 响应 `blob()` → `URL.createObjectURL` → `<audio controls>` + `<a download>`；
- 状态：生成前 `ttsStatus()` 显示“加载中/排队中/等待图像任务”；失败显示 JSON `detail`；
- 表单值存 `localStorage`（键 `imagehub.tts.custom` 等）。

- [ ] **Step 4: 跑 Playwright**

Run: `python3 tools/verify_tts_ui.py`
Expected: ALL PASS A–H。

---

### Task 7: HTTPS（5569）

**Files:**
- Create: `tlsutil.py`
- Modify: `run.py`
- Modify: `app.py`（`_tls_enabled()`、SessionMiddleware）
- Modify: `imagehub.service`
- Modify: `deploy-guard.sh`
- Modify: `tools/full_stress.py`
- Modify: `index.html`（API 示例的 http → https）
- Create: `tests/test_run_tls.py`
- Modify: `tests/test_deploy_guard.py`

**Interfaces:**
- Produces:
  - `tlsutil.build_uvicorn_cmd(python: str, port: int, cert: str | None, key: str | None) -> list[str]`：只有 `cert`、`key` 都存在且文件可读时才追加 `--ssl-certfile/--ssl-keyfile`。
  - `app._tls_enabled() -> bool`：`IMAGEHUB_TLS_CERT` 和 `IMAGEHUB_TLS_KEY` 都存在且文件存在。
  - `deploy-guard.sh` 读取 `IMAGEHUB_BASE_URL`（默认：证书存在时 `https://127.0.0.1:5569`，否则 `http://127.0.0.1:5569`）；https 时给 curl 加 `--cacert /etc/imagehub/tls/imagehub.crt`，不关闭校验。

- [ ] **Step 1: 写失败测试**

`tests/test_run_tls.py`：
```python
import os, tempfile, unittest
import tlsutil


class TLSTests(unittest.TestCase):
    def test_http_when_no_cert(self):
        cmd = tlsutil.build_uvicorn_cmd("python", 5569, None, None)
        self.assertNotIn("--ssl-certfile", cmd)

    def test_https_when_files_exist(self):
        d = tempfile.mkdtemp()
        cert = os.path.join(d, "c.crt"); key = os.path.join(d, "c.key")
        open(cert, "w").close(); open(key, "w").close()
        cmd = tlsutil.build_uvicorn_cmd("python", 5569, cert, key)
        self.assertIn("--ssl-certfile", cmd)
        self.assertIn(cert, cmd)
```
`tests/test_deploy_guard.py` 扩展：curl shim 把 `"$@"` 写入文件；`run_guard(..., base_url="https://127.0.0.1:5569")`；断言参数含 `--cacert` 和 https URL，且不含 `-k`。

Run: `python -m unittest discover -s tests -p "test_run_tls.py" -v`；`... -p "test_deploy_guard.py" -v`
Expected: FAIL。

- [ ] **Step 2: 实现 `tlsutil.py` 和 `run.py`**

`build_uvicorn_cmd` 返回：
```python
[python, "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", str(port)]
```
有证书时追加 `--ssl-certfile`, cert, `--ssl-keyfile`, key。
`run.py` 的 `subprocess.Popen` 改用 `build_uvicorn_cmd(...)`，证书从 `os.environ.get("IMAGEHUB_TLS_CERT")` / `IMAGEHUB_TLS_KEY` 读取。

- [ ] **Step 3: 实现 app 侧与部署脚本**

- `app.py`：`def _tls_enabled(): ...`；`SessionMiddleware(..., https_only=_tls_enabled())`。
- `imagehub.service` 增加：
  ```
  Environment="IMAGEHUB_TLS_CERT=/etc/imagehub/tls/imagehub.crt"
  Environment="IMAGEHUB_TLS_KEY=/etc/imagehub/tls/imagehub.key"
  ```
- `deploy-guard.sh`：
  ```bash
  CERT=${IMAGEHUB_TLS_CERT:-/etc/imagehub/tls/imagehub.crt}
  URL=${IMAGEHUB_BASE_URL:-}
  if [ -z "$URL" ]; then
    if [ -f "$CERT" ]; then URL=https://127.0.0.1:5569/api/status; else URL=http://127.0.0.1:5569/api/status; fi
  fi
  CURL_OPTS="-fsS -m 2"
  case "$URL" in https://*) [ -f "$CERT" ] && CURL_OPTS="$CURL_OPTS --cacert $CERT";; esac
  B=$(curl $CURL_OPTS "$URL" | ...)
  ```
- `tools/full_stress.py` 的 base URL 改成 `IMAGEHUB_BASE_URL`，https 时用 `ssl.create_default_context(cafile=IMAGEHUB_CA)`（默认 `/etc/imagehub/tls/imagehub.crt`），不关闭校验。
- `index.html` API 示例里的 `http://192.168.0.70:5569` 改成 `https://`。

- [ ] **Step 4: 跑测试**

Run: `python -m unittest discover -s tests -p "test_run_tls.py" -v`；`... -p "test_deploy_guard.py" -v`
Expected: PASS。

- [ ] **Step 5: 生成证书（生产，非仓库）**

Run:
```bash
mkdir -p /etc/imagehub/tls && chmod 700 /etc/imagehub/tls
openssl req -x509 -newkey rsa:2048 -nodes -days 825 \
  -keyout /etc/imagehub/tls/imagehub.key \
  -out /etc/imagehub/tls/imagehub.crt \
  -subj "/CN=192.168.0.70" \
  -addext "subjectAltName=IP:192.168.0.70,IP:127.0.0.1"
chmod 600 /etc/imagehub/tls/imagehub.key
```
Expected: 两个文件存在；`openssl x509 -in ... -noout -text | grep -A1 SAN` 显示 `IP Address:192.168.0.70, IP Address:127.0.0.1`。

---

### Task 8: 版本、README、NOTICE

**Files:**
- Modify: `app.py`（`APP_VERSION = "1.6.0"`）
- Modify: `README.md`
- Modify: `NOTICE`

- [ ] **Step 1: 更新版本与文档**

- `APP_VERSION` → `1.6.0`；`APP_BUILD` 同步。
- README 增加：TTS tab、三个变体、API `POST /v1/audio/speech`、`/api/tts/*`、互斥规则、HTTPS 访问与证书信任、模型路径、`qwen-tts` 环境安装、回滚。
- NOTICE 增加 1.6.0 条目（TTS + HTTPS）。

- [ ] **Step 2: 回归**

Run: `python -m unittest discover -s tests -v`
Expected: 全部 PASS；`grep 1.6.0 README.md NOTICE app.py` 均命中。

---

### Task 9: 生产部署、验收、回滚

**Files:**
- Deploy: `app.py`、`index.html`、`run.py`、`tlsutil.py`、`tts_service/`、`deploy-guard.sh`、`imagehub.service`、`NOTICE`、`README.md`
- Backup: `/opt/vLLM/imagehub/*.bak.20261127-tts`

- [ ] **Step 1: 部署前检查**

Run:
```bash
ssh root@192.168.0.70 'curl -s http://127.0.0.1:5569/api/status; nvidia-smi --query-gpu=index,memory.used --format=csv,noheader'
```
Expected: busy=false；GPU0/1 空闲或只有已知服务；记录当前版本。

- [ ] **Step 2: 备份并部署**

- 备份 `app.py`、`index.html`、`run.py`、`deploy-guard.sh`、`imagehub.service`、`imagehub.db`（`.backup`）。
- 复制新文件；`chmod +x deploy-guard.sh`；`systemctl daemon-reload`。
- 用 `deploy-guard.sh` 重启（它会用 https + `--cacert`）。

- [ ] **Step 3: 验收**

- `curl --cacert /etc/imagehub/tls/imagehub.crt https://192.168.0.70:5569/api/version` → `1.6.0`；
- 浏览器访问 https，接受证书，SSO 登录成功；
- TTS 页加载三个模型（记录耗时/显存），三个变体各生成一次；
- 图像生成一次（确认 TTS 被停）；再回 TTS（确认重新加载）；
- `deploy-guard.sh` 在 TTS 生成时不会重启（busy=true）；
- 日志无 `database is locked`、无 OOM、无 5570 端口残留。

- [ ] **Step 4: 回滚预案**

- 恢复备份的 `app.py`/`index.html`/`run.py`/`deploy-guard.sh`/`imagehub.service`；
- 删除 `IMAGEHUB_TLS_*` 环境变量，`systemctl daemon-reload` 并重启；
- 证书和 `qwen-tts` 环境保留（无害）；确认 HTTP 恢复、TTS tab 消失。

---

## Self-Review

- **Spec coverage:** HTTPS（Task 7）、TTS 服务（2–3）、三模型放置（1）、API（2/4）、原生 UI（6）、互斥队列（5）、无历史（4）、busy/deploy-guard（5/7）、版本/NOTICE（8）、部署（9）。
- **Type consistency:** `_tts["state"]`、`_tts_start/_tts_stop`、`_arbiter_acquire/_arbiter_release`、`_image_session`、`build_uvicorn_cmd` 在任务间一致。
- **Review Focus:** 五个风险分别由 Task 5（1/2/3）、Task 3（4）、Task 7（5）的测试覆盖。
- **Proportion:** 任务以接口和测试为主，未逐行写实现。
