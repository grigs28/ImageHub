import os
import tempfile
import unittest

import tlsutil


class TLSTests(unittest.TestCase):
    def test_http_when_no_cert(self):
        cmd = tlsutil.build_uvicorn_cmd("python", 5569, None, None)
        self.assertNotIn("--ssl-certfile", cmd)
        self.assertEqual(cmd[-2:], ["--port", "5569"])

    def test_https_when_files_exist(self):
        d = tempfile.mkdtemp()
        cert = os.path.join(d, "c.crt")
        key = os.path.join(d, "c.key")
        open(cert, "w").close()
        open(key, "w").close()
        cmd = tlsutil.build_uvicorn_cmd("python", 5569, cert, key)
        self.assertIn("--ssl-certfile", cmd)
        self.assertIn(cert, cmd)
        self.assertIn("--ssl-keyfile", cmd)
        self.assertIn(key, cmd)

    def test_missing_key_falls_back_to_http(self):
        d = tempfile.mkdtemp()
        cert = os.path.join(d, "c.crt")
        open(cert, "w").close()
        cmd = tlsutil.build_uvicorn_cmd("python", 5569, cert, None)
        self.assertNotIn("--ssl-certfile", cmd)


if __name__ == "__main__":
    unittest.main()
