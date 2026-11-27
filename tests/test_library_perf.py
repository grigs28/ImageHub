import json, os, shutil, sqlite3, tempfile, threading, time, unittest

from PIL import Image

TMP = tempfile.mkdtemp(prefix="imagehub-test-")
os.environ["IMAGEHUB_DB"] = os.path.join(TMP, "imagehub.db")
os.environ["IMAGEHUB_THUMBS"] = os.path.join(TMP, "thumbs")
os.environ["IMAGEHUB_RUN"] = os.path.join(TMP, "run")
os.environ["IMAGEHUB_OUTPUT"] = os.path.join(TMP, "output")

import app as imagehub


def reset_db():
    for suffix in ("", "-wal", "-shm"):
        try:
            os.remove(os.environ["IMAGEHUB_DB"] + suffix)
        except FileNotFoundError:
            pass
    imagehub._schema_ready = False
    imagehub._ensure_schema()


class SchemaTests(unittest.TestCase):
    def setUp(self):
        reset_db()

    def test_indexes_and_query_plan(self):
        conn = imagehub.db(readonly=True)
        self.assertEqual(conn.execute("PRAGMA journal_mode").fetchone()[0].lower(), "wal")
        names = {row[1] for row in conn.execute("PRAGMA index_list(jobs)")}
        self.assertIn("idx_jobs_created", names)
        self.assertIn("idx_jobs_project_created", names)
        plan = conn.execute(
            "EXPLAIN QUERY PLAN SELECT id,prompt,models,created,results,status "
            "FROM jobs ORDER BY created DESC LIMIT 20 OFFSET 0"
        ).fetchall()
        self.assertTrue(any("USING INDEX idx_jobs_created" in row[3] for row in plan))
        conn.close()

    def test_readonly_blocks_write(self):
        conn = imagehub.db(readonly=True)
        with self.assertRaises(sqlite3.OperationalError):
            conn.execute(
                "INSERT INTO jobs(id,prompt,models,created,results) "
                "VALUES('x','x','[]',0,'[]')"
            )
        conn.close()

    def test_interrupted_only_on_schema_init(self):
        conn = imagehub.db()
        conn.execute(
            "INSERT INTO jobs(id,prompt,models,created,results,status) "
            "VALUES('r1','p','[]',1,'[]','running')"
        )
        conn.commit(); conn.close()
        imagehub._schema_ready = False
        imagehub._ensure_schema()
        conn = imagehub.db(readonly=True)
        self.assertEqual(conn.execute("SELECT status FROM jobs WHERE id='r1'").fetchone()[0], "interrupted")
        conn.close()

        conn = imagehub.db()
        conn.execute(
            "INSERT INTO jobs(id,prompt,models,created,results,status) "
            "VALUES('r2','p','[]',2,'[]','running')"
        )
        conn.commit(); conn.close()
        conn = imagehub.db(); conn.close()
        conn = imagehub.db(readonly=True)
        self.assertEqual(conn.execute("SELECT status FROM jobs WHERE id='r2'").fetchone()[0], "running")
        conn.close()

    def test_upsert_preserves_created(self):
        imagehub._upsert_job_result(
            "j1", "p", ["qwen"], 100.0,
            [{"model": "qwen", "status": "ok", "image": "a.png"}], "running",
        )
        imagehub._upsert_job_result(
            "j1", "p", ["qwen"], 100.0,
            [{"model": "qwen", "status": "ok", "image": "b.png"}], "running",
        )
        imagehub._upsert_job_result(
            "j1", "p", ["qwen"], 100.0,
            [{"model": "qwen", "status": "ok", "image": "b.png"}], "done",
        )
        conn = imagehub.db(readonly=True)
        created, status, results = conn.execute(
            "SELECT created,status,results FROM jobs WHERE id='j1'"
        ).fetchone()
        conn.close()
        self.assertEqual(created, 100.0)
        self.assertEqual(status, "done")
        self.assertIn("b.png", results)


class ThumbTests(unittest.TestCase):
    def setUp(self):
        reset_db()
        shutil.rmtree(imagehub.THUMB_DIR, ignore_errors=True)
        os.makedirs(imagehub.THUMB_DIR, exist_ok=True)
        os.makedirs(imagehub.OUTPUT_DIR, exist_ok=True)
        self.src = os.path.join(imagehub.OUTPUT_DIR, "thumb_src.png")
        Image.new("RGB", (200, 200), (120, 30, 30)).save(self.src)

    def test_ensure_thumb_creates_and_caches(self):
        out = imagehub._ensure_thumb("thumb_src.png", 160)
        self.assertTrue(os.path.isfile(out))
        Image.open(out).verify()
        mtime = os.path.getmtime(out)
        imagehub._ensure_thumb("thumb_src.png", 160)
        self.assertEqual(os.path.getmtime(out), mtime)
        self.assertEqual([f for f in os.listdir(imagehub.THUMB_DIR) if f.endswith(".tmp")], [])

    def test_missing_source_returns_none(self):
        self.assertIsNone(imagehub._ensure_thumb("not_here.png", 160))

    def test_prewarm_three_widths(self):
        imagehub._prewarm_thumbs([{"status": "ok", "image": "thumb_src.png"}])
        imagehub._thumb_q.join()
        for w in (160, 480, 1024):
            self.assertTrue(os.path.isfile(imagehub._thumb_path("thumb_src.png", w)))

    def test_prewarm_history_backfill(self):
        imagehub._upsert_job_result(
            "h1", "p", ["qwen"], time.time(),
            [{"model": "qwen", "status": "ok", "image": "thumb_src.png"}], "done",
        )
        imagehub._prewarm_history_thumbs()
        imagehub._thumb_q.join()
        for w in (160, 480, 1024):
            self.assertTrue(os.path.isfile(imagehub._thumb_path("thumb_src.png", w)))


class BusyTests(unittest.TestCase):
    def setUp(self):
        reset_db()
        imagehub._state["active"] = 0
        imagehub._state["switching"] = False
        imagehub._dq["running"] = False
        imagehub._dq["queue"].clear()
        imagehub._jobs.clear()

    def test_busy_flags(self):
        self.assertFalse(imagehub._busy())
        imagehub._state["active"] = 1
        self.assertTrue(imagehub._busy())
        imagehub._state["active"] = 0

        imagehub._state["switching"] = True
        self.assertTrue(imagehub._busy())
        imagehub._state["switching"] = False

        imagehub._dq["running"] = True
        self.assertTrue(imagehub._busy())
        imagehub._dq["running"] = False

        imagehub._dq["queue"].append(("x", {}, None))
        self.assertTrue(imagehub._busy())
        imagehub._dq["queue"].clear()

        with imagehub._comfy_lock:
            self.assertTrue(imagehub._busy())

        imagehub._jobs["j"] = {"status": "running"}
        self.assertTrue(imagehub._busy())
        imagehub._jobs.clear()

    def test_busy_waits_for_jobs_lock(self):
        result = []

        def probe():
            result.append(imagehub._busy())

        with imagehub._lock:
            t = threading.Thread(target=probe)
            t.start()
            time.sleep(0.1)
            self.assertTrue(t.is_alive(), "_busy should wait for _jobs lock")
        t.join(1)
        self.assertFalse(t.is_alive())
        self.assertEqual(result, [False])


class VersionTests(unittest.TestCase):
    def test_version(self):
        self.assertEqual(imagehub.APP_VERSION, "1.7.3")


class StatusRouteTests(unittest.TestCase):
    def setUp(self):
        reset_db()
        imagehub._state["active"] = 0
        imagehub._state["switching"] = False
        imagehub._state["loaded"] = None
        imagehub._state["comfy_loaded"] = None
        imagehub._dq["running"] = False
        imagehub._dq["queue"].clear()
        imagehub._jobs.clear()
        old = imagehub.svc_alive
        imagehub.svc_alive = lambda *a, **k: False
        self.addCleanup(setattr, imagehub, "svc_alive", old)

    def test_status_route_returns_dict_with_busy(self):
        route = next(r for r in imagehub.app.routes
                     if getattr(r, "path", None) == "/api/status")
        self.assertIs(route.endpoint, imagehub.api_status)
        body = route.endpoint()
        self.assertIn("busy", body)
        self.assertFalse(body["busy"])
        imagehub._state["active"] = 1
        self.assertTrue(route.endpoint()["busy"])
