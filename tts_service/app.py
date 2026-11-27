"""ImageHub TTS 服务：三个 Qwen3-TTS 1.7B 变体，内部端口 5570。"""
import argparse
import logging
import os
import re
import tempfile
import threading

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import Response
from pydantic import BaseModel

from . import audio
from .models import BaseEngine, create_engine

logger = logging.getLogger("tts_service")


class SpeakReq(BaseModel):
    variant: str = ""
    text: str = ""
    language: str = "Auto"
    speaker: str | None = None
    instruct: str | None = None


class OpenAISpeechReq(BaseModel):
    model: str = "tts-1"
    input: str = ""
    voice: str = "Vivian"
    response_format: str = "wav"
    language: str = "Auto"


def create_app(engine: BaseEngine) -> FastAPI:
    app = FastAPI(title="ImageHub TTS")
    app.state.engine = engine
    app.state.ready = threading.Event()
    app.state.tts = {"state": "loading", "error": None, "active": 0, "pending": 0}
    app.state.gen_lock = threading.Lock()
    app.state.cond = threading.Condition()

    def _load() -> None:
        try:
            engine.load()
            with app.state.cond:
                app.state.tts["state"] = "ready"
                app.state.cond.notify_all()
        except Exception as e:  # noqa: BLE001 - 加载失败要暴露给 /health
            logger.exception("TTS 模型加载失败")
            with app.state.cond:
                app.state.tts["state"] = "error"
                app.state.tts["error"] = str(e)[:500]
                app.state.cond.notify_all()
        finally:
            app.state.ready.set()

    threading.Thread(target=_load, daemon=True, name="tts-load").start()

    def _require_ready() -> None:
        st = app.state.tts
        if st["state"] == "loading":
            raise HTTPException(503, "TTS 模型正在加载")
        if st["state"] != "ready":
            raise HTTPException(503, f"TTS 不可用: {st.get('error') or st['state']}")

    def _run(fn):
        _require_ready()
        with app.state.cond:
            app.state.tts["pending"] += 1
        with app.state.gen_lock:
            with app.state.cond:
                app.state.tts["pending"] -= 1
                app.state.tts["active"] += 1
            try:
                wavs, sr = fn()
                data = audio.wav_bytes(wavs, sr)
            finally:
                with app.state.cond:
                    app.state.tts["active"] -= 1
                    app.state.cond.notify_all()
        return Response(content=data, media_type="audio/wav",
                        headers={"Content-Disposition": 'attachment; filename="tts.wav"'})

    def _validate_speak(req: SpeakReq) -> None:
        if req.variant not in ("custom", "design"):
            raise HTTPException(400, "variant 必须是 custom 或 design")
        if not (req.text or "").strip():
            raise HTTPException(400, "text 不能为空")
        if len(req.text) > 2000:
            raise HTTPException(400, "text 不能超过 2000 字")
        if not (req.language or "").strip():
            raise HTTPException(400, "language 不能为空")
        if req.variant == "custom" and not (req.speaker or "").strip():
            raise HTTPException(400, "custom 需要 speaker")
        if req.variant == "design" and not (req.instruct or "").strip():
            raise HTTPException(400, "design 需要 instruct")

    @app.get("/health")
    def health():
        st = app.state.tts
        return {"ok": st["state"] == "ready", "state": st["state"],
                "error": st.get("error")}

    @app.get("/api/status")
    def status():
        st = app.state.tts
        models = {}
        if st["state"] == "ready":
            try:
                models = engine.status().get("models", {})
            except Exception:  # noqa: BLE001 - 状态查询不能拖垮服务
                models = {}
        return {"state": st["state"], "error": st.get("error"),
                "active": st["active"], "pending": st["pending"],
                "models": models}

    @app.get("/api/voices")
    def voices():
        _require_ready()
        return engine.voices()

    @app.post("/api/tts/speak")
    def speak(req: SpeakReq):
        _validate_speak(req)
        if req.variant == "custom":
            return _run(lambda: engine.speak_custom(
                req.text, req.language, req.speaker, req.instruct))
        return _run(lambda: engine.speak_design(
            req.text, req.language, req.instruct))

    @app.post("/api/tts/clone")
    def clone(text: str = Form(""), language: str = Form("Auto"),
              ref_text: str = Form(""), x_vector_only_mode: str = Form("false"),
              ref_audio: UploadFile = File(...)):
        if not (text or "").strip():
            raise HTTPException(400, "text 不能为空")
        if len(text) > 2000:
            raise HTTPException(400, "text 不能超过 2000 字")
        if not (language or "").strip():
            raise HTTPException(400, "language 不能为空")
        xv = (x_vector_only_mode or "").lower() in ("1", "true", "yes", "on")
        if not xv and not (ref_text or "").strip():
            raise HTTPException(400, "克隆需要 ref_text，或启用 x_vector_only_mode")

        suffix = os.path.splitext(ref_audio.filename or "ref.wav")[1].lower()
        if not re.fullmatch(r"\.[a-z0-9]{1,8}", suffix):
            suffix = ".wav"
        fd, path = tempfile.mkstemp(prefix="imagehub-tts-ref-", suffix=suffix)
        converted = path + ".wav"
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(ref_audio.file.read())
            try:
                audio.convert_to_wav(path, converted)
            except ValueError as e:
                raise HTTPException(400, str(e))
            return _run(lambda: engine.speak_clone(
                text, language, converted, ref_text or None, xv))
        finally:
            for p in (path, converted):
                try:
                    os.remove(p)
                except OSError:
                    pass

    @app.post("/v1/audio/speech")
    def openai_speech(req: OpenAISpeechReq):
        if req.response_format not in ("wav",):
            raise HTTPException(400, "只支持 response_format=wav")
        if not (req.input or "").strip():
            raise HTTPException(400, "input 不能为空")
        if len(req.input) > 2000:
            raise HTTPException(400, "input 不能超过 2000 字")
        return _run(lambda: engine.speak_custom(
            req.input, req.language, req.voice))

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="ImageHub TTS service")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5570)
    parser.add_argument("--fake", action="store_true")
    parser.add_argument("--custom")
    parser.add_argument("--design")
    parser.add_argument("--clone")
    parser.add_argument("--devices", default="cuda:0,cuda:0,cuda:1")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s - %(levelname)s - %(message)s")
    if args.fake:
        engine: BaseEngine = create_engine(True, {})
    else:
        engine = create_engine(False, {
            "custom": args.custom, "design": args.design, "clone": args.clone,
        }, tuple(args.devices.split(",")))
    uvicorn.run(create_app(engine), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
