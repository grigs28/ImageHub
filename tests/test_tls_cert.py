import os
import tempfile
import unittest

TMP = tempfile.mkdtemp(prefix="imagehub-tls-cert-")
os.environ.setdefault("IMAGEHUB_DB", os.path.join(TMP, "imagehub.db"))
os.environ.setdefault("IMAGEHUB_THUMBS", os.path.join(TMP, "thumbs"))
os.environ.setdefault("IMAGEHUB_RUN", os.path.join(TMP, "run"))
os.environ.setdefault("IMAGEHUB_OUTPUT", os.path.join(TMP, "output"))

import app as imagehub  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

PEM = ("-----BEGIN CERTIFICATE-----\n"
       "MIIBfakecertificatecontent\n"
       "-----END CERTIFICATE-----\n")


class TLSCertTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="imagehub-cert-")
        self.cert = os.path.join(self.dir, "imagehub.crt")
        self.key = os.path.join(self.dir, "imagehub.key")
        with open(self.cert, "w", encoding="utf-8") as f:
            f.write(PEM)
        open(self.key, "w").close()
        self.old = {k: os.environ.get(k)
                    for k in ("IMAGEHUB_TLS_CERT", "IMAGEHUB_TLS_KEY")}
        os.environ["IMAGEHUB_TLS_CERT"] = self.cert
        os.environ["IMAGEHUB_TLS_KEY"] = self.key
        self.client = TestClient(imagehub.app)

    def tearDown(self):
        for k, v in self.old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def test_cert_download(self):
        r = self.client.get("/api/tls/cert")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.headers["content-type"], "application/x-x509-ca-cert")
        self.assertIn("imagehub.crt", r.headers.get("content-disposition", ""))
        self.assertEqual(r.content.decode(), PEM)

    def test_cert_missing_404(self):
        os.environ["IMAGEHUB_TLS_CERT"] = os.path.join(self.dir, "missing.crt")
        self.assertEqual(self.client.get("/api/tls/cert").status_code, 404)

    def test_status_tls_flag(self):
        self.assertTrue(self.client.get("/api/status").json()["tls"])
        os.environ.pop("IMAGEHUB_TLS_KEY", None)
        self.assertFalse(self.client.get("/api/status").json()["tls"])


if __name__ == "__main__":
    unittest.main()
