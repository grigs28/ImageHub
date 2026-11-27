"""音频编码与参考音频处理。"""
import array
import io
import os
import subprocess
import sys
import wave

MAX_REF_SECONDS = float(os.environ.get("IMAGEHUB_TTS_MAX_SECONDS", "60"))
MAX_REF_BYTES = int(os.environ.get("IMAGEHUB_TTS_MAX_UPLOAD",
                                   str(20 * 1024 * 1024)))


def wav_bytes(samples, sr: int) -> bytes:
    """把 [-1, 1] 的浮点采样编码为 16-bit PCM 单声道 WAV。"""
    pcm = array.array("h")
    for s in samples:
        v = max(-1.0, min(1.0, float(s)))
        pcm.append(int(round(v * 32767)))
    if sys.byteorder == "big":
        pcm.byteswap()
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(int(sr))
        w.writeframes(pcm.tobytes())
    return buf.getvalue()


def wav_duration(path: str) -> float:
    """返回 WAV 文件时长（秒）。非 WAV 文件会抛异常，由调用方决定是否忽略。"""
    with wave.open(path, "rb") as w:
        return w.getnframes() / float(w.getframerate())


def validate_ref_audio(path: str, max_seconds: float = MAX_REF_SECONDS,
                       max_bytes: int = MAX_REF_BYTES) -> None:
    """校验参考音频大小；WAV 可直接校验时长，其它格式等转码后再校验。"""
    if os.path.getsize(path) > max_bytes:
        raise ValueError("参考音频超过 20MB")
    try:
        duration = wav_duration(path)
    except Exception:  # noqa: BLE001 - 非 WAV 由 ffmpeg 解析
        return
    if duration > max_seconds:
        raise ValueError("参考音频超过 60 秒")


def convert_to_wav(src: str, dst: str, sr: int = 24000,
                   max_seconds: float = MAX_REF_SECONDS,
                   max_bytes: int = MAX_REF_BYTES) -> None:
    """把任意音频转成 24kHz 单声道 WAV；超长/超大/无法解析抛 ValueError。"""
    validate_ref_audio(src, max_seconds, max_bytes)
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", src,
             "-ac", "1", "-ar", str(sr), dst],
            check=True, capture_output=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        try:
            os.remove(dst)
        except OSError:
            pass
        raise ValueError("无法解析音频")
    try:
        duration = wav_duration(dst)
    except Exception:  # noqa: BLE001 - 转码产物异常，按不可解析处理
        duration = 0
    if duration > max_seconds:
        try:
            os.remove(dst)
        except OSError:
            pass
        raise ValueError("参考音频超过 60 秒")
