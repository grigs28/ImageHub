import os
import tempfile
import threading
import time
import unittest
from unittest import mock

TMP = tempfile.mkdtemp(prefix="imagehub-tts-arb-")
os.environ.setdefault("IMAGEHUB_DB", os.path.join(TMP, "imagehub.db"))
os.environ.setdefault("IMAGEHUB_THUMBS", os.path.join(TMP, "thumbs"))
os.environ.setdefault("IMAGEHUB_RUN", os.path.join(TMP, "run"))
os.environ.setdefault("IMAGEHUB_OUTPUT", os.path.join(TMP, "output"))

import app as imagehub  # noqa: E402


class TTSArbiterTests(unittest.TestCase):
    def setUp(self):
        imagehub._ensure_schema()
        imagehub._tts.update({"proc": None, "state": "stopped", "error": None,
                              "active": 0, "pending": 0})
        with imagehub._cond:
            imagehub._state.update({"loaded": None, "comfy_loaded": None, "active": 0,
                                    "switching": False, "target": None,
                                    "image_pending": 0})
        with imagehub._lock:
            imagehub._jobs.clear()
        with imagehub._dq["lock"]:
            imagehub._dq["running"] = None
            imagehub._dq["queue"].clear()
        with imagehub._arbiter_lock:
            imagehub._arbiter_q.clear()

    def patch(self, **names):
        old = {k: getattr(imagehub, k) for k in names}
        for k, v in names.items():
            setattr(imagehub, k, v)
        self.addCleanup(lambda: [setattr(imagehub, k, v) for k, v in old.items()])

    def test_image_waits_for_tts_active(self):
        imagehub._tts["state"] = "ready"
        imagehub._tts["active"] = 1
        stopped = []
        self.patch(_tts_stop=lambda: stopped.append(1))
        done = threading.Event()

        def worker():
            imagehub._tts_release_for_image(timeout=5)
            done.set()

        t = threading.Thread(target=worker)
        t.start()
        self.assertFalse(done.wait(0.3), "图像请求不应该在 TTS 生成中切换")
        with imagehub._cond:
            imagehub._tts["active"] = 0
            imagehub._cond.notify_all()
        self.assertTrue(done.wait(3))
        self.assertTrue(stopped)
        with imagehub._cond:
            self.assertEqual(imagehub._state["image_pending"], 1)
        imagehub._image_done()
        with imagehub._cond:
            self.assertEqual(imagehub._state["image_pending"], 0)

    def test_image_stops_idle_tts(self):
        imagehub._tts["state"] = "ready"
        stopped = []
        self.patch(_tts_stop=lambda: stopped.append(1))
        imagehub._tts_release_for_image(timeout=5)
        self.assertTrue(stopped)
        with imagehub._cond:
            self.assertEqual(imagehub._state["image_pending"], 1)
        imagehub._image_done()

    def test_tts_waits_for_active_image(self):
        with imagehub._cond:
            imagehub._state["active"] = 1
        started = []
        self.patch(_tts_start=lambda timeout=300: started.append(1) or True,
                   stop_service=lambda k: None,
                   _stop_comfy=lambda: None)
        result = []
        done = threading.Event()

        def worker():
            result.append(imagehub._tts_begin_request(timeout=5))
            done.set()

        t = threading.Thread(target=worker)
        t.start()
        self.assertFalse(done.wait(0.3), "TTS 请求不应该在图像生成中切换")
        with imagehub._cond:
            imagehub._state["active"] = 0
            imagehub._cond.notify_all()
        self.assertTrue(done.wait(3))
        self.assertEqual(result, [True])
        self.assertTrue(started)
        self.assertEqual(imagehub._tts["active"], 1)
        imagehub._tts_end_request()
        self.assertEqual(imagehub._tts["active"], 0)

    def test_tts_stops_idle_image(self):
        with imagehub._cond:
            imagehub._state["loaded"] = "qwen"
        stopped = []
        comfy = []
        self.patch(_tts_start=lambda timeout=300: True,
                   stop_service=lambda k: stopped.append(k),
                   _stop_comfy=lambda: comfy.append(1))
        self.assertTrue(imagehub._tts_begin_request(timeout=5))
        self.assertEqual(set(stopped), set(imagehub.SERVICES))
        self.assertTrue(comfy)
        with imagehub._cond:
            self.assertIsNone(imagehub._state["loaded"])
        self.assertEqual(imagehub._tts["active"], 1)
        imagehub._tts_end_request()

    def test_arbiter_fifo_order(self):
        self.assertTrue(imagehub._arbiter_acquire(timeout=1))
        order = []

        def waiter(i):
            imagehub._arbiter_acquire(timeout=3)
            order.append(i)
            time.sleep(0.05)
            imagehub._arbiter_release()

        threads = []
        for i in range(3):
            t = threading.Thread(target=waiter, args=(i,))
            t.start()
            threads.append(t)
            time.sleep(0.05)
        imagehub._arbiter_release()
        for t in threads:
            t.join(5)
        self.assertEqual(order, [0, 1, 2])

    def test_busy_includes_tts_loading(self):
        imagehub._tts["state"] = "loading"
        self.assertTrue(imagehub._busy())
        imagehub._tts["state"] = "ready"
        imagehub._tts["active"] = 0
        imagehub._tts["pending"] = 0
        self.assertFalse(imagehub._busy())


    def test_tts_wait_does_not_block_image_switch(self):
        with imagehub._lock:
            imagehub._jobs["j1"] = {"status": "running"}
        self.patch(_tts_json=lambda path, timeout=5: None,
                   _tts_start=lambda timeout=300: True,
                   stop_service=lambda k: None,
                   _stop_comfy=lambda: None,
                   _tts_stop=lambda: None)
        result = []
        done = threading.Event()

        def tts_worker():
            result.append(imagehub._tts_begin_request(timeout=3))
            done.set()

        t = threading.Thread(target=tts_worker)
        t.start()
        time.sleep(0.2)
        t0 = time.time()
        imagehub._tts_release_for_image(timeout=1)
        imagehub._image_done()
        elapsed = time.time() - t0
        self.assertLess(elapsed, 0.8, "TTS 等待图像时不应占着仲裁票")
        with imagehub._lock:
            imagehub._jobs["j1"]["status"] = "done"
        with imagehub._cond:
            imagehub._cond.notify_all()
        self.assertTrue(done.wait(2))
        self.assertEqual(result, [True])
        imagehub._tts_end_request()

    def test_preload_skips_when_tts_ready(self):
        imagehub._tts["state"] = "ready"
        called = []
        self.patch(acquire_model=lambda k: called.append(k),
                   _tts_json=lambda path, timeout=5: None)
        with mock.patch.object(imagehub.time, "sleep",
                               side_effect=AssertionError("preload slept")):
            imagehub._preload_default()
        self.assertEqual(called, [])
        self.assertEqual(called, [])

    def test_stop_preserves_error_when_requested(self):
        imagehub._tts.update({"proc": None, "state": "error", "error": "boom"})
        imagehub._tts_stop(reset_error=False)
        self.assertEqual(imagehub._tts["state"], "stopped")
        self.assertEqual(imagehub._tts["error"], "boom")
        imagehub._tts["error"] = None

    def test_start_error_keeps_message(self):
        self.patch(_tts_json=lambda path, timeout=5:
                   {"state": "error", "error": "CUDA OOM"})
        ok = imagehub._tts_start()
        self.assertFalse(ok)
        self.assertEqual(imagehub._tts["state"], "stopped")
        self.assertIn("CUDA OOM", imagehub._tts["error"] or "")
        imagehub._tts["error"] = None


    def test_adopt_clears_stale_state_when_no_service(self):
        with imagehub._cond:
            imagehub._state["loaded"] = "qwen"
        self.patch(svc_alive=lambda k, timeout=3: False,
                   _save_state=lambda k: None)
        imagehub._adopt_running_services()
        with imagehub._cond:
            self.assertIsNone(imagehub._state["loaded"])

    def test_busy_includes_image_pending(self):
        with imagehub._cond:
            imagehub._state["image_pending"] = 1
        self.assertTrue(imagehub._busy())


if __name__ == "__main__":
    unittest.main()
