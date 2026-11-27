import io
import threading
import unittest
import wave

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
        self.assertEqual(c.post("/api/tts/speak",
                                json={"variant": "custom", "text": ""}).status_code, 400)
        self.assertEqual(c.post("/api/tts/speak",
                                json={"variant": "custom", "text": "x"}).status_code, 400)
        self.assertEqual(c.post("/api/tts/speak",
                                json={"variant": "design", "text": "x"}).status_code, 400)

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
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(24000)
            w.writeframes(b"\x00\x00" * 2400)
        data = {"text": "你好", "language": "Chinese"}
        r = c.post("/api/tts/clone", data=data,
                   files={"ref_audio": ("a.wav", wav.getvalue(), "audio/wav")})
        self.assertEqual(r.status_code, 400)
        r = c.post("/api/tts/clone", data={**data, "x_vector_only_mode": "true"},
                   files={"ref_audio": ("a.wav", wav.getvalue(), "audio/wav")})
        self.assertEqual(r.status_code, 200)


if __name__ == "__main__":
    unittest.main()
