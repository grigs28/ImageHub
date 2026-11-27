import json
import os
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

TMP = tempfile.mkdtemp(prefix="imagehub-tts-bff-")
os.environ.setdefault("IMAGEHUB_DB", os.path.join(TMP, "imagehub.db"))
os.environ.setdefault("IMAGEHUB_THUMBS", os.path.join(TMP, "thumbs"))
os.environ.setdefault("IMAGEHUB_RUN", os.path.join(TMP, "run"))
os.environ.setdefault("IMAGEHUB_OUTPUT", os.path.join(TMP, "output"))

import app as imagehub  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):
        pass

    def _send(self, code, body, ctype="application/json"):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            self._send(200, json.dumps({"ok": True, "state": "ready"}).encode())
        elif self.path == "/api/status":
            self._send(200, json.dumps({"state": "ready", "active": 0, "pending": 0,
                                        "models": {"custom": "ready", "design": "ready",
                                                   "clone": "ready"}}).encode())
        elif self.path == "/api/voices":
            self._send(200, json.dumps({"speakers": [{"name": "Vivian"}],
                                        "languages": ["Chinese"]}).encode())
        else:
            self._send(404, b'{"detail":"not found"}')

    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        self.server.last_path = self.path
        self.server.last_body = self.rfile.read(n)
        if self.path in ("/api/tts/speak", "/api/tts/clone"):
            self._send(200, b"RIFF....WAVE", "audio/wav")
        else:
            self._send(404, b'{"detail":"not found"}')


class TTSBFFTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.old_port = imagehub.TTS_PORT
        imagehub.TTS_PORT = cls.server.server_address[1]

    @classmethod
    def tearDownClass(cls):
        imagehub.TTS_PORT = cls.old_port
        cls.server.shutdown()
        cls.server.server_close()

    def setUp(self):
        imagehub._ensure_schema()
        imagehub._tts.update({"proc": None, "state": "stopped", "error": None,
                              "active": 0, "pending": 0})
        conn = imagehub.db()
        conn.execute("DELETE FROM api_keys WHERE name='tts-test'")
        conn.execute("INSERT INTO api_keys(name,key_hash,role,enabled,created) "
                     "VALUES(?,?,?,?,?)",
                     ("tts-test", imagehub._hash_key("tts-key"), "user", 1, time.time()))
        conn.commit()
        conn.close()
        self.c = TestClient(imagehub.app)

    def speak_body(self, apikey=None):
        return {"variant": "custom", "text": "你好", "language": "Chinese",
                "speaker": "Vivian", "apikey": apikey}

    def test_tts_status_merges_remote(self):
        d = self.c.get("/api/tts/status").json()
        self.assertTrue(d["loaded"])
        self.assertEqual(d["state"], "ready")
        self.assertIn("image", d)

    def test_tts_voices_proxies(self):
        d = self.c.get("/api/tts/voices").json()
        self.assertEqual(d["speakers"][0]["name"], "Vivian")

    def test_speak_without_auth_401(self):
        r = self.c.post("/api/tts/speak", json=self.speak_body())
        self.assertEqual(r.status_code, 401)

    def test_speak_with_key_returns_wav(self):
        r = self.c.post("/api/tts/speak", json=self.speak_body("tts-key"))
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.headers["content-type"], "audio/wav")
        self.assertTrue(r.content.startswith(b"RIFF"))

    def test_speak_does_not_write_jobs(self):
        conn = imagehub.db(readonly=True)
        before = conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0]
        conn.close()
        r = self.c.post("/api/tts/speak", json=self.speak_body("tts-key"))
        self.assertEqual(r.status_code, 200)
        conn = imagehub.db(readonly=True)
        after = conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0]
        conn.close()
        self.assertEqual(before, after)

    def test_v1_audio_speech_with_bearer(self):
        r = self.c.post("/v1/audio/speech",
                        headers={"Authorization": "Bearer tts-key"},
                        json={"input": "你好", "voice": "Vivian"})
        self.assertEqual(r.status_code, 200)
        body = json.loads(self.server.last_body)
        self.assertEqual((body["variant"], body["text"], body["speaker"]),
                         ("custom", "你好", "Vivian"))

    def test_clone_forwards_multipart(self):
        r = self.c.post("/api/tts/clone",
                        data={"text": "你好", "language": "Chinese",
                              "ref_text": "参考", "apikey": "tts-key"},
                        files={"ref_audio": ("a.wav", b"RIFF....WAVE", "audio/wav")})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(self.server.last_path, "/api/tts/clone")
        self.assertIn(b'name="ref_audio"', self.server.last_body)

    def test_disabled_bearer_key_rejected_on_tts(self):
        conn = imagehub.db()
        conn.execute("DELETE FROM api_keys WHERE name='tts-disabled'")
        conn.execute("INSERT INTO api_keys(name,key_hash,role,enabled,created) "
                     "VALUES(?,?,?,?,?)",
                     ("tts-disabled", imagehub._hash_key("disabled-key"),
                      "user", 0, time.time()))
        conn.commit()
        conn.close()
        r = self.c.post("/v1/audio/speech",
                        headers={"Authorization": "Bearer disabled-key"},
                        json={"input": "你好", "voice": "Vivian"})
        self.assertEqual(r.status_code, 401)

    def test_multipart_sanitizes_filename_and_content_type(self):
        body, _ = imagehub._multipart_encode(
            {},
            {"ref_audio": ('../../x"\r\nX-Evil: 1.wav', b"data",
                           "text/html\r\nX-Evil: 1")})
        self.assertNotIn(b"\r\nX-Evil:", body)
        self.assertNotIn(b"Content-Type: text/html", body)
        self.assertNotIn(b"../", body)
        self.assertIn(b"Content-Type: application/octet-stream", body)

    def test_load_unload_calls_start_stop(self):
        old = {k: getattr(imagehub, k) for k in
               ("_tts_authed", "_tts_start", "_tts_stop", "_tts_json",
                "stop_service", "_stop_comfy")}
        called = []
        imagehub._tts_authed = lambda apikey, request: True
        imagehub._tts_json = lambda path, timeout=5: None
        imagehub._tts_start = lambda timeout=300: called.append("start") or True
        imagehub._tts_stop = lambda *a, **kw: called.append("stop")
        imagehub.stop_service = lambda k: None
        imagehub._stop_comfy = lambda: None
        try:
            self.assertEqual(self.c.post("/api/tts/load").status_code, 200)
            self.assertEqual(self.c.post("/api/tts/unload").status_code, 200)
        finally:
            for k, v in old.items():
                setattr(imagehub, k, v)
        self.assertEqual(called, ["start", "stop"])


    def test_load_stops_idle_image_before_start(self):
        old = {k: getattr(imagehub, k) for k in
               ("_tts_authed", "_tts_json", "_tts_start", "stop_service", "_stop_comfy")}
        calls = {"stop": [], "comfy": [], "start": []}
        imagehub._tts_authed = lambda apikey, request: True
        imagehub._tts_json = lambda path, timeout=5: None
        imagehub._tts_start = lambda timeout=300: calls["start"].append(1) or True
        imagehub.stop_service = lambda k: calls["stop"].append(k)
        imagehub._stop_comfy = lambda: calls["comfy"].append(1)
        with imagehub._cond:
            imagehub._state["loaded"] = "qwen"
        try:
            r = self.c.post("/api/tts/load")
        finally:
            for k, v in old.items():
                setattr(imagehub, k, v)
            with imagehub._cond:
                imagehub._state["loaded"] = None
        self.assertEqual(r.status_code, 200)
        self.assertEqual(set(calls["stop"]), set(imagehub.SERVICES))
        self.assertTrue(calls["comfy"])
        self.assertTrue(calls["start"])

    def test_status_image_busy_excludes_tts(self):
        imagehub._tts["state"] = "loading"
        try:
            d = self.c.get("/api/tts/status").json()
        finally:
            imagehub._tts["state"] = "stopped"
        self.assertFalse(d["image"]["busy"])


    def test_invalid_speak_does_not_start_tts(self):
        old = imagehub._tts_begin_request
        calls = []
        imagehub._tts_begin_request = lambda timeout=600: calls.append(1) or True
        try:
            r = self.c.post("/api/tts/speak",
                            json={"variant": "custom", "text": "", "apikey": "tts-key"})
        finally:
            imagehub._tts_begin_request = old
        self.assertEqual(r.status_code, 400)
        self.assertEqual(calls, [], "非法请求不应触发停图像/加载 TTS")

    def test_openai_speech_empty_input_does_not_start_tts(self):
        old = imagehub._tts_begin_request
        calls = []
        imagehub._tts_begin_request = lambda timeout=600: calls.append(1) or True
        try:
            r = self.c.post("/v1/audio/speech",
                            headers={"Authorization": "Bearer tts-key"},
                            json={"input": ""})
        finally:
            imagehub._tts_begin_request = old
        self.assertEqual(r.status_code, 400)
        self.assertEqual(calls, [])

    def test_clone_missing_ref_text_does_not_start_tts(self):
        old = imagehub._tts_begin_request
        calls = []
        imagehub._tts_begin_request = lambda timeout=600: calls.append(1) or True
        try:
            r = self.c.post("/api/tts/clone",
                            data={"text": "你好", "language": "Chinese",
                                  "apikey": "tts-key"},
                            files={"ref_audio": ("a.wav", b"RIFF", "audio/wav")})
        finally:
            imagehub._tts_begin_request = old
        self.assertEqual(r.status_code, 400)
        self.assertEqual(calls, [])

    def test_clone_large_upload_does_not_start_tts(self):
        old_begin, old_max = imagehub._tts_begin_request, imagehub.TTS_MAX_UPLOAD
        calls = []
        imagehub._tts_begin_request = lambda timeout=600: calls.append(1) or True
        imagehub.TTS_MAX_UPLOAD = 10
        try:
            r = self.c.post("/api/tts/clone",
                            data={"text": "你好", "language": "Chinese",
                                  "ref_text": "参考", "apikey": "tts-key"},
                            files={"ref_audio": ("a.wav", b"x" * 11, "audio/wav")})
        finally:
            imagehub._tts_begin_request, imagehub.TTS_MAX_UPLOAD = old_begin, old_max
        self.assertEqual(r.status_code, 400)
        self.assertEqual(calls, [])


    def test_speak_records_tts_history(self):
        conn = imagehub.db(readonly=True)
        before = conn.execute("SELECT COUNT(*) FROM tts_audio").fetchone()[0]
        conn.close()
        r = self.c.post("/api/tts/speak", json=self.speak_body("tts-key"))
        self.assertEqual(r.status_code, 200)
        conn = imagehub.db(readonly=True)
        after = conn.execute("SELECT COUNT(*) FROM tts_audio").fetchone()[0]
        row = conn.execute("SELECT variant,language,speaker,file FROM tts_audio "
                           "ORDER BY created DESC LIMIT 1").fetchone()
        conn.close()
        self.assertEqual(after, before + 1)
        self.assertEqual(row[0], "custom")
        self.assertIn(row[3], os.listdir(imagehub.TTS_OUT_DIR))


    def test_v1_audio_speech_body_apikey(self):
        r = self.c.post("/v1/audio/speech",
                        json={"input": "你好", "voice": "Vivian",
                              "apikey": "tts-key"})
        self.assertEqual(r.status_code, 200)
        self.assertTrue(r.content.startswith(b"RIFF"))

    def test_status_prefers_service_counts_when_ready(self):
        imagehub._tts.update({"active": 2, "pending": 1, "state": "ready"})
        try:
            d = self.c.get("/api/tts/status").json()
        finally:
            imagehub._tts.update({"active": 0, "pending": 0, "state": "stopped"})
        # 服务 ready 时以服务端 active/pending 为准（stub 返回 0/0）
        self.assertEqual((d["active"], d["pending"]), (0, 0))

    def test_status_reports_local_loading_and_pending(self):
        imagehub._tts.update({"state": "loading", "pending": 1, "active": 0})
        try:
            d = self.c.get("/api/tts/status").json()
        finally:
            imagehub._tts.update({"state": "stopped", "pending": 0})
        self.assertEqual(d["state"], "loading")
        self.assertGreaterEqual(d["pending"], 1)
        self.assertFalse(d["loaded"])


    def test_pending_clears_when_generation_starts(self):
        imagehub._tts.update({"active": 0, "pending": 0})
        seen = {}

        def fake_begin(timeout=600):
            with imagehub._cond:
                imagehub._tts["active"] += 1
            return True

        def fake_call():
            with imagehub._cond:
                seen["pending"] = imagehub._tts["pending"]
                seen["active"] = imagehub._tts["active"]
            return 200, b"RIFF", "audio/wav"

        old = imagehub._tts_begin_request
        imagehub._tts_begin_request = fake_begin
        try:
            imagehub._tts_generation(fake_call)
        finally:
            imagehub._tts_begin_request = old
        self.assertEqual(seen, {"pending": 0, "active": 1})
        self.assertEqual(imagehub._tts["active"], 0)
        self.assertEqual(imagehub._tts["pending"], 0)


if __name__ == "__main__":
    unittest.main()
