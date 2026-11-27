import os
import tempfile
import unittest
from unittest import mock

TMP = tempfile.mkdtemp(prefix="imagehub-api-docs-")
os.environ.setdefault("IMAGEHUB_DB", os.path.join(TMP, "imagehub.db"))
os.environ.setdefault("IMAGEHUB_THUMBS", os.path.join(TMP, "thumbs"))
os.environ.setdefault("IMAGEHUB_RUN", os.path.join(TMP, "run"))
os.environ.setdefault("IMAGEHUB_OUTPUT", os.path.join(TMP, "output"))

import app as imagehub  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402


class ApiDocsTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(imagehub.app)

    def test_download_markdown(self):
        r = self.client.get("/api/docs")
        self.assertEqual(r.status_code, 200)
        self.assertTrue(r.headers["content-type"].startswith("text/markdown"))
        self.assertIn("ImageHub-API.md", r.headers.get("content-disposition", ""))
        text = r.content.decode("utf-8")
        self.assertIn("语音合成", text)
        self.assertIn("/v1/audio/speech", text)
        self.assertIn("apikey", text)

    def test_missing_doc_404(self):
        with mock.patch.object(imagehub, "BASE_DIR", tempfile.mkdtemp()):
            self.assertEqual(self.client.get("/api/docs").status_code, 404)


if __name__ == "__main__":
    unittest.main()
