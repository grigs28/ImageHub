import io
import os
import tempfile
import time
import unittest
import zipfile

TMP = tempfile.mkdtemp(prefix="imagehub-tts-hist-")
os.environ.setdefault("IMAGEHUB_DB", os.path.join(TMP, "imagehub.db"))
os.environ.setdefault("IMAGEHUB_THUMBS", os.path.join(TMP, "thumbs"))
os.environ.setdefault("IMAGEHUB_RUN", os.path.join(TMP, "run"))
os.environ.setdefault("IMAGEHUB_OUTPUT", os.path.join(TMP, "output"))

import app as imagehub  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from tts_service.audio import wav_bytes  # noqa: E402

WAV = wav_bytes([0.0] * 2400, 24000)  # 0.1s


class TTSHistoryTests(unittest.TestCase):
    def setUp(self):
        imagehub._ensure_schema()
        conn = imagehub.db()
        conn.execute("DELETE FROM tts_audio")
        conn.execute("DELETE FROM api_keys WHERE name='tts-hist'")
        conn.execute("INSERT INTO api_keys(name,key_hash,role,enabled,created) "
                     "VALUES(?,?,?,?,?)",
                     ("tts-hist", imagehub._hash_key("hist-key"), "user", 1, time.time()))
        conn.commit()
        conn.close()
        self.client = TestClient(imagehub.app)
        self.h = {"Authorization": "Bearer hist-key"}

    def save(self, variant="custom", text="你好", speaker="Vivian"):
        return imagehub._tts_save_result(variant, text, "Chinese", speaker, "", WAV, 1.2)

    def list_rows(self):
        r = self.client.get("/api/tts/history", headers=self.h)
        self.assertEqual(r.status_code, 200, r.text)
        return r.json()

    def test_schema(self):
        conn = imagehub.db(readonly=True)
        try:
            cols = {r[1] for r in conn.execute("PRAGMA table_info(tts_audio)")}
            indexes = {r[1] for r in conn.execute("PRAGMA index_list(tts_audio)")}
        finally:
            conn.close()
        self.assertTrue({"id", "created", "variant", "text", "language", "speaker",
                         "instruct", "file", "duration", "elapsed", "size"} <= cols)
        self.assertIn("idx_tts_audio_created", indexes)

    def test_save_and_list(self):
        jid = self.save()
        d = self.list_rows()
        self.assertEqual(d["total"], 1)
        row = d["rows"][0]
        self.assertEqual(row["id"], jid)
        self.assertEqual(row["variant"], "custom")
        self.assertEqual(row["speaker"], "Vivian")
        self.assertAlmostEqual(row["duration"], 0.1, places=2)
        self.assertAlmostEqual(row["elapsed"], 1.2, places=1)
        self.assertEqual(row["size"], len(WAV))
        self.assertTrue(os.path.isfile(os.path.join(imagehub.TTS_OUT_DIR, row["file"])))

    def test_audio_served(self):
        self.save()
        row = self.list_rows()["rows"][0]
        r = self.client.get("/api/tts/audio/" + row["file"], headers=self.h)
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.headers["content-type"], "audio/wav")
        self.assertEqual(r.content, WAV)

    def test_delete_one(self):
        jid = self.save()
        row = self.list_rows()["rows"][0]
        path = os.path.join(imagehub.TTS_OUT_DIR, row["file"])
        r = self.client.delete("/api/tts/history/" + jid, headers=self.h)
        self.assertEqual(r.status_code, 200)
        self.assertFalse(os.path.exists(path))
        self.assertEqual(self.list_rows()["total"], 0)

    def test_batch_delete(self):
        ids = [self.save(text="a"), self.save(text="b")]
        r = self.client.post("/api/tts/history/batch-delete", headers=self.h,
                             json={"ids": ids})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["deleted"], 2)
        self.assertEqual(self.list_rows()["total"], 0)

    def test_requires_auth(self):
        self.assertEqual(self.client.get("/api/tts/history").status_code, 401)
        self.assertEqual(self.client.get(
            "/api/tts/history", headers={"Authorization": "Bearer bad"}).status_code, 401)

    def test_batch_download_zip(self):
        ids = [self.save(text="a"), self.save(text="b")]
        r = self.client.post("/api/tts/history/batch-download", headers=self.h,
                             json={"ids": ids})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.headers["content-type"], "application/zip")
        self.assertIn("attachment", r.headers.get("content-disposition", ""))
        z = zipfile.ZipFile(io.BytesIO(r.content))
        names = z.namelist()
        self.assertEqual(len(names), 2)
        self.assertTrue(all(n.endswith(".wav") for n in names))
        self.assertEqual(z.read(names[0]), WAV)

    def test_batch_download_requires_auth(self):
        r = self.client.post("/api/tts/history/batch-download", json={"ids": ["x"]})
        self.assertEqual(r.status_code, 401)

    def test_batch_download_missing_404(self):
        r = self.client.post("/api/tts/history/batch-download", headers=self.h,
                             json={"ids": ["nope"]})
        self.assertEqual(r.status_code, 404)

    def test_history_accepts_query_apikey(self):
        r = self.client.get("/api/tts/history?apikey=hist-key")
        self.assertEqual(r.status_code, 200)

    def test_audio_accepts_query_apikey(self):
        self.save()
        row = self.list_rows()["rows"][0]
        r = self.client.get(f"/api/tts/audio/{row['file']}?apikey=hist-key")
        self.assertEqual(r.status_code, 200)

    def test_batch_delete_accepts_body_apikey(self):
        ids = [self.save()]
        r = self.client.post("/api/tts/history/batch-delete",
                             json={"ids": ids, "apikey": "hist-key"})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["deleted"], 1)

    def test_batch_download_accepts_body_apikey(self):
        ids = [self.save()]
        r = self.client.post("/api/tts/history/batch-download",
                             json={"ids": ids, "apikey": "hist-key"})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.headers["content-type"], "application/zip")

    def test_no_image_jobs_written(self):
        conn = imagehub.db(readonly=True)
        before = conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0]
        conn.close()
        self.save()
        conn = imagehub.db(readonly=True)
        after = conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0]
        conn.close()
        self.assertEqual(before, after)


if __name__ == "__main__":
    unittest.main()
