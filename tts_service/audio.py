"""音频编码与参考音频处理。"""
import array
import io
import json
import os
import subprocess
import sys
import wave

MAX_REF_SECONDS = float(os.environ.get("IMAGEHUB_TTS_MAX_SECONDS", "60"))
MAX_REF_BYTES = int(os.environ.get("IMAGEHUB_TTS_MAX_UPLOAD",
                                   str(20 * 1024 * 1024)))

# 参考音频只接受常见音频容器；播放列表/网络流/图像序列一律拒绝。
_ALLOWED_FORMATS = {
    "wav", "mp3", "flac", "ogg", "opus", "matroska", "webm",
    "mov", "mp4", "m4a", "3gp", "3g2", "mj2", "aac", "amr",
    "asf", "aiff", "caf", "wv", "ape",
}
_BLOCKED_FORMAT_PARTS = (
    "concat", "hls", "dash", "image2", "lavfi", "rtsp", "rtmp",
    "http", "https", "ftp", "sftp", "smb", "tcp", "udp", "sdp",
)


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


def _probe_audio(path: str) -> tuple[str, float]:
    """用 ffprobe 读取容器与时长；只允许本地文件协议。"""
    cmd = ["ffprobe", "-v", "error", "-protocol_whitelist", "file",
           "-show_entries", "format=format_name,duration",
           "-of", "json", path]
    try:
        proc = subprocess.run(cmd, check=True, capture_output=True,
                              text=True, timeout=20)
        data = json.loads(proc.stdout or "{}")
        meta = data.get("format") or {}
        fmt = str(meta.get("format_name") or "")
        duration = float(meta.get("duration") or 0)
    except (subprocess.CalledProcessError, FileNotFoundError,
            subprocess.TimeoutExpired, ValueError, TypeError):
        raise ValueError("无法解析音频")
    if not fmt:
        raise ValueError("无法解析音频")
    return fmt, duration


def _format_allowed(fmt: str) -> bool:
    names = {n.strip().lower() for n in fmt.split(",") if n.strip()}
    if not names:
        return False
    if any(part in name for name in names for part in _BLOCKED_FORMAT_PARTS):
        return False
    return bool(names & _ALLOWED_FORMATS)


def validate_ref_audio(path: str, max_seconds: float = MAX_REF_SECONDS,
                       max_bytes: int = MAX_REF_BYTES) -> tuple[str, float]:
    """校验参考音频大小、容器与时长；返回 (format_name, duration)。"""
    if os.path.getsize(path) > max_bytes:
        raise ValueError("参考音频超过 20MB")
    fmt, duration = _probe_audio(path)
    if not _format_allowed(fmt):
        raise ValueError("不支持的音频格式")
    if duration > max_seconds:
        raise ValueError("参考音频超过 60 秒")
    return fmt, duration


def convert_to_wav(src: str, dst: str, sr: int = 24000,
                   max_seconds: float = MAX_REF_SECONDS,
                   max_bytes: int = MAX_REF_BYTES) -> None:
    """把任意音频转成 24kHz 单声道 WAV；超长/超大/无法解析抛 ValueError。"""
    validate_ref_audio(src, max_seconds, max_bytes)
    max_decoded = int(sr) * 2 * int(max_seconds) + 65536
    cmd = ["ffmpeg", "-nostdin", "-y", "-loglevel", "error",
           "-protocol_whitelist", "file",
           "-i", src,
           "-t", f"{max_seconds:.3f}",
           "-ac", "1", "-ar", str(sr),
           "-f", "wav", dst]
    try:
        subprocess.run(cmd, check=True, capture_output=True, timeout=180)
    except (subprocess.CalledProcessError, FileNotFoundError,
            subprocess.TimeoutExpired):
        try:
            os.remove(dst)
        except OSError:
            pass
        raise ValueError("无法解析音频")
    try:
        size = os.path.getsize(dst)
    except OSError:
        size = 0
    if size <= 0 or size > max_decoded:
        try:
            os.remove(dst)
        except OSError:
            pass
        raise ValueError("参考音频超过限制")
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
