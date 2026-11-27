"""uvicorn 启动参数：有证书才启用 TLS。"""
import os


def build_uvicorn_cmd(python: str, port: int, cert: str | None,
                      key: str | None) -> list[str]:
    cmd = [python, "-m", "uvicorn", "app:app",
           "--host", "0.0.0.0", "--port", str(port)]
    if cert and key and os.path.isfile(cert) and os.path.isfile(key):
        cmd += ["--ssl-certfile", cert, "--ssl-keyfile", key]
    return cmd
