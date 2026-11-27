"""TTS 引擎抽象与假引擎（测试/无 GPU 环境用）。"""
import math
import threading
import time

SPEAKER_META = [
    {"name": "Vivian", "description": "Bright, slightly edgy young female voice.",
     "native_language": "Chinese"},
    {"name": "Serena", "description": "Warm, gentle young female voice.",
     "native_language": "Chinese"},
    {"name": "Uncle_Fu", "description": "Seasoned male voice with a low, mellow timbre.",
     "native_language": "Chinese"},
    {"name": "Dylan", "description": "Youthful Beijing male voice with a clear, natural timbre.",
     "native_language": "Chinese (Beijing Dialect)"},
    {"name": "Eric", "description": "Lively Chengdu male voice with a slightly husky brightness.",
     "native_language": "Chinese (Sichuan Dialect)"},
    {"name": "Ryan", "description": "Dynamic male voice with strong rhythmic drive.",
     "native_language": "English"},
    {"name": "Aiden", "description": "Sunny American male voice with a clear midrange.",
     "native_language": "English"},
    {"name": "Ono_Anna", "description": "Playful Japanese female voice with a light, nimble timbre.",
     "native_language": "Japanese"},
    {"name": "Sohee", "description": "Warm Korean female voice with rich emotion.",
     "native_language": "Korean"},
]

LANGUAGES = ["Auto", "Chinese", "English", "Japanese", "Korean", "German",
             "French", "Russian", "Portuguese", "Spanish", "Italian"]


class BaseEngine:
    """三个变体的统一接口；返回值统一为 (samples, sample_rate)。"""

    def load(self) -> None:
        raise NotImplementedError

    def status(self) -> dict:
        return {"state": "ready",
                "models": {"custom": "ready", "design": "ready", "clone": "ready"}}

    def voices(self) -> dict:
        raise NotImplementedError

    def speak_custom(self, text: str, language: str, speaker: str,
                     instruct: str | None = None):
        raise NotImplementedError

    def speak_design(self, text: str, language: str, instruct: str):
        raise NotImplementedError

    def speak_clone(self, text: str, language: str, ref_audio_path: str,
                    ref_text: str | None, x_vector_only_mode: bool):
        raise NotImplementedError


class FakeEngine(BaseEngine):
    """不占显存的假引擎：固定正弦波，用于接口/队列测试。"""

    def __init__(self, delay: float = 0.0):
        self.delay = delay
        self.max_parallel = 0
        self._now = 0
        self._mu = threading.Lock()

    def load(self) -> None:
        return None

    def voices(self) -> dict:
        return {"speakers": [dict(s) for s in SPEAKER_META],
                "languages": list(LANGUAGES)}

    def _work(self) -> None:
        with self._mu:
            self._now += 1
            self.max_parallel = max(self.max_parallel, self._now)
        try:
            if self.delay:
                time.sleep(self.delay)
        finally:
            with self._mu:
                self._now -= 1

    def _tone(self, sr: int = 24000, seconds: float = 0.5):
        n = int(sr * seconds)
        return [0.2 * math.sin(2 * math.pi * 440 * i / sr) for i in range(n)], sr

    def speak_custom(self, text, language, speaker, instruct=None):
        self._work()
        return self._tone()

    def speak_design(self, text, language, instruct):
        self._work()
        return self._tone()

    def speak_clone(self, text, language, ref_audio_path, ref_text,
                    x_vector_only_mode):
        self._work()
        return self._tone()


class QwenEngine(BaseEngine):
    """官方 qwen-tts 包装：三个 1.7B 变体分别加载到指定 GPU。"""

    def __init__(self, custom_path: str, design_path: str, clone_path: str,
                 devices: tuple[str, str, str] = ("cuda:0", "cuda:0", "cuda:1")):
        if len(devices) != 3:
            raise ValueError("devices 需要三个值，例如 cuda:0,cuda:0,cuda:1")
        self.custom_path = custom_path
        self.design_path = design_path
        self.clone_path = clone_path
        self.devices = tuple(devices)
        self._custom = None
        self._design = None
        self._clone = None

    def load(self) -> None:
        import torch
        from qwen_tts import Qwen3TTSModel

        self._custom = Qwen3TTSModel.from_pretrained(
            self.custom_path, device_map=self.devices[0], dtype=torch.bfloat16)
        self._design = Qwen3TTSModel.from_pretrained(
            self.design_path, device_map=self.devices[1], dtype=torch.bfloat16)
        self._clone = Qwen3TTSModel.from_pretrained(
            self.clone_path, device_map=self.devices[2], dtype=torch.bfloat16)

    def status(self) -> dict:
        def state(model):
            return "ready" if model is not None else "missing"
        return {"state": "ready",
                "models": {"custom": state(self._custom),
                           "design": state(self._design),
                           "clone": state(self._clone)},
                "devices": {"custom": self.devices[0],
                            "design": self.devices[1],
                            "clone": self.devices[2]}}

    def voices(self) -> dict:
        return {"speakers": [dict(s) for s in SPEAKER_META],
                "languages": self._custom.get_supported_languages()}

    @staticmethod
    def _first(wavs):
        # 官方返回 [np.ndarray]；测试里的假模型可能直接返回一维浮点列表
        if isinstance(wavs, (list, tuple)) and wavs and hasattr(wavs[0], "__len__"):
            return wavs[0]
        return wavs

    def speak_custom(self, text, language, speaker, instruct=None):
        wavs, sr = self._custom.generate_custom_voice(
            text=text, language=language, speaker=speaker, instruct=instruct)
        return self._first(wavs), sr

    def speak_design(self, text, language, instruct):
        if not (instruct or "").strip():
            raise ValueError("design 需要 instruct")
        wavs, sr = self._design.generate_voice_design(
            text=text, language=language, instruct=instruct)
        return self._first(wavs), sr

    def speak_clone(self, text, language, ref_audio_path, ref_text,
                    x_vector_only_mode):
        wavs, sr = self._clone.generate_voice_clone(
            text=text, language=language, ref_audio=ref_audio_path,
            ref_text=ref_text, x_vector_only_mode=x_vector_only_mode)
        return self._first(wavs), sr


def create_engine(fake: bool, paths: dict,
                  devices: tuple[str, str, str] = ("cuda:0", "cuda:0", "cuda:1")
                  ) -> BaseEngine:
    if fake:
        return FakeEngine()
    return QwenEngine(paths["custom"], paths["design"], paths["clone"], devices)

