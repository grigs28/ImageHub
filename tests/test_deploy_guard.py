import os
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "deploy-guard.sh"


class DeployGuardTests(unittest.TestCase):
    def run_guard(self, curl_body, tries=2, base_url=None, cert=None,
                  https_probe_fails=False):
        tmp = tempfile.mkdtemp(prefix="imagehub-guard-")
        bin_dir = Path(tmp) / "bin"
        bin_dir.mkdir()
        marker = Path(tmp) / "restarted"
        args_file = Path(tmp) / "curl_args"
        curl = bin_dir / "curl"
        pre = 'case "$*" in *https://*) exit 7;; esac\n' if https_probe_fails else ""
        curl.write_text(
            "#!/bin/sh\n"
            + pre +
            f"printf '%s\\n' \"$@\" > '{args_file}'\n"
            f"printf '%s' '{curl_body}'\n"
            "exit 0\n")
        curl.chmod(0o755)
        systemctl = bin_dir / "systemctl"
        systemctl.write_text(f"#!/bin/sh\ntouch {marker}\nexit 0\n")
        systemctl.chmod(0o755)
        env = os.environ.copy()
        env["PATH"] = str(bin_dir) + os.pathsep + env["PATH"]
        env["IMAGEHUB_GUARD_TRIES"] = str(tries)
        env["IMAGEHUB_GUARD_SLEEP"] = "0"
        if base_url:
            env["IMAGEHUB_BASE_URL"] = base_url
        if cert:
            env["IMAGEHUB_TLS_CERT"] = cert
        proc = subprocess.run(
            ["bash", str(SCRIPT)], cwd=ROOT, env=env,
            capture_output=True, text=True, timeout=20)
        args = args_file.read_text() if args_file.exists() else ""
        return proc, marker.exists(), args

    def test_missing_busy_field_does_not_restart(self):
        proc, restarted, _ = self.run_guard('{"detail":"Internal Server Error"}')
        self.assertNotEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        self.assertFalse(restarted, "guard must fail closed when busy is unknown")

    def test_idle_busy_restarts(self):
        proc, restarted, _ = self.run_guard('{"busy":false}', tries=2)
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        self.assertTrue(restarted)

    def test_https_uses_cacert_not_insecure_flag(self):
        d = tempfile.mkdtemp(prefix="imagehub-guard-cert-")
        cert = Path(d) / "imagehub.crt"
        cert.write_text("test cert")
        proc, restarted, args = self.run_guard(
            '{"busy":false}', base_url="https://127.0.0.1:5569", cert=str(cert))
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        self.assertTrue(restarted)
        self.assertIn("--cacert", args)
        self.assertIn(str(cert), args)
        self.assertNotIn("-k", args)
        self.assertIn("https://127.0.0.1:5569/api/status", args)


    def test_cert_with_http_service_falls_back(self):
        d = tempfile.mkdtemp(prefix="imagehub-guard-cert-")
        cert = Path(d) / "imagehub.crt"
        cert.write_text("test cert")
        proc, restarted, args = self.run_guard(
            '{"busy":false}', cert=str(cert), https_probe_fails=True)
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        self.assertTrue(restarted)
        self.assertIn("http://127.0.0.1:5569/api/status", args)


if __name__ == "__main__":
    unittest.main()
