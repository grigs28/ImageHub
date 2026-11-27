#!/usr/bin/env python3
"""Dev-only Playwright acceptance for the ImageHub library list.

Run from the repo root: python3 tools/verify_library_perf.py
Not a runtime dependency; requires Playwright + a local Chrome.
"""
import io
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from PIL import Image
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
INDEX = (ROOT / "index.html").read_bytes()
BUF = io.BytesIO()
Image.new("RGB", (1, 1), (200, 20, 20)).save(BUF, "JPEG")
JPEG = BUF.getvalue()

LOCK = threading.Lock()
STATE = {"history": [], "history_requests": 0}
SLOW_OFFSET = 20
SLOW_DELAY = 0.5


def job(i, prompt=None, images=None):
    return {
        "id": f"job-{i:02d}",
        "prompt": prompt or f"prompt {i}",
        "models": ["qwen"],
        "created": 1700000000 - i,
        "status": "done",
        "images": images or [{
            "model": "qwen", "image": f"img-{i}.png",
            "duration": 1, "size": "1024x1024", "seed": i,
        }],
    }


BASE20 = [job(i) for i in range(20)]
BASE40 = [job(i) for i in range(40)]
FULL80 = [job(i, images=[
    {"model": "qwen", "image": f"img-{i}-{j}.png", "duration": 1,
     "size": "1024x1024", "seed": i * 4 + j}
    for j in range(4)
]) for i in range(20)]

with LOCK:
    STATE["history"] = list(BASE20)


def set_history(rows):
    with LOCK:
        STATE["history"] = list(rows)


def history_requests():
    with LOCK:
        return STATE["history_requests"]


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, format, *args):
        pass

    def _send(self, body, content_type="application/json", status=200, headers=None):
        if isinstance(body, (dict, list)):
            body = json.dumps(body, ensure_ascii=False).encode()
        elif isinstance(body, str):
            body = body.encode()
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        for k, v in (headers or {}).items():
            self.send_header(k, v)
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        if path in ("/", "/index.html"):
            self._send(INDEX, "text/html; charset=utf-8",
                       headers={"Cache-Control": "no-store"})
            return
        if path == "/api/history":
            qs = parse_qs(parsed.query)
            limit = max(1, min(int(qs.get("limit", ["20"])[0]), 200))
            offset = max(0, int(qs.get("offset", ["0"])[0]))
            with LOCK:
                STATE["history_requests"] += 1
                rows = list(STATE["history"][offset:offset + limit])
                total = len(STATE["history"])
            if offset == SLOW_OFFSET:
                time.sleep(SLOW_DELAY)
            self._send(rows, headers={
                "X-Total-Count": str(total),
                "Cache-Control": "no-store",
            })
            return
        if path == "/api/models":
            self._send([{"key": "qwen", "label": "Qwen", "default_size": [1024, 1024],
                         "max_size": 2048, "note": "test", "gpus": "test",
                         "ready": True}])
            return
        if path == "/api/status":
            self._send({"loaded": None, "active": 0, "switching": False,
                        "target": None, "backend": "direct", "version": "test"})
            return
        if path == "/api/gpu":
            self._send([])
            return
        if path == "/api/power":
            self._send([])
            return
        if path == "/api/settings":
            self._send({"backend": "direct", "gate": False, "clocks": False,
                        "clocks_now": False})
            return
        if path == "/api/projects":
            self._send([])
            return
        if path == "/api/whoami":
            self._send({"sso_enabled": False, "user": None})
            return
        if path == "/api/edit/model":
            self._send({"ready": True})
            return
        if path == "/api/version":
            self._send({"name": "ImageHub", "version": "test", "build": "test"})
            return
        if path.startswith("/thumb/") or path.startswith("/image/"):
            self._send(JPEG, "image/jpeg",
                       headers={"Cache-Control": "public, max-age=86400"})
            return
        if path.startswith("/api/"):
            self._send({})
            return
        self._send({}, status=404)

    def do_POST(self):
        self._send({})

    def do_DELETE(self):
        self._send({})


def start_server():
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


class Checks:
    def __init__(self):
        self.failed = 0

    def check(self, name, cond, detail=""):
        print(("PASS" if cond else "FAIL") + f": {name}" + (f" — {detail}" if detail else ""))
        if not cond:
            self.failed += 1

    def run(self, name, fn):
        try:
            fn()
        except Exception as e:
            self.check(name, False, f"{type(e).__name__}: {e}")


def main():
    server = start_server()
    base = f"http://127.0.0.1:{server.server_address[1]}/"
    checks = Checks()
    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path="/usr/bin/google-chrome",
                                    headless=True, args=["--no-sandbox"])
        page = browser.new_page(viewport={"width": 1440, "height": 900})
        page.goto(base, wait_until="domcontentloaded", timeout=30000)
        page.wait_for_timeout(500)

        checks.check("A. 启动不自动加载历史", history_requests() == 0,
                     f"requests={history_requests()}")

        page.click('a[data-page="library"]')
        page.wait_for_selector(".histitem", state="attached", timeout=8000)
        checks.check("A. 首次点图库只请求 1 次", history_requests() == 1,
                     f"requests={history_requests()}")

        def b_work_poll():
            page.evaluate("CURRENT_PAGE='work'")
            before = history_requests()
            page.evaluate("pollHistory()")
            page.wait_for_timeout(250)
            checks.check("B. 工作台不轮询历史", history_requests() == before,
                         f"before={before}, after={history_requests()}")
        checks.run("B. 工作台不轮询历史", b_work_poll)

        def c_no_rerender():
            marks = page.evaluate("""() => {
              const els=[...document.querySelectorAll('.histitem')];
              els.forEach((e,i)=>e.dataset.mark='m'+i);
              return els.map(e=>e.dataset.mark);
            }""")
            page.evaluate("loadHistory(0)")
            page.wait_for_timeout(400)
            after = page.evaluate(
                "() => [...document.querySelectorAll('.histitem')].map(e=>e.dataset.mark)")
            checks.check("C. 数据未变不重建 DOM", after == marks,
                         f"before={marks[:3]}..., after={after[:3]}...")
        checks.run("C. 数据未变不重建 DOM", c_no_rerender)

        def d_full_page():
            new_job = job(100, prompt="<img src=x onerror=alert(1)>")
            new_job["id"] = "new-job"
            set_history([new_job] + BASE20[:19])
            anchor = page.evaluate("""() => {
              const box = document.getElementById('historyList');
              box.scrollTop = 300;
              const br = box.getBoundingClientRect();
              const el = [...box.querySelectorAll('.histitem')]
                .find(e => e.getBoundingClientRect().bottom > br.top);
              return el ? {id: el.dataset.id,
                           top: el.getBoundingClientRect().top - br.top} : null;
            }""")
            page.evaluate("loadHistory(0)")
            page.wait_for_timeout(500)
            count = page.evaluate(
                "() => document.querySelectorAll('.histitem').length")
            first = page.evaluate(
                "() => document.querySelector('.histitem').dataset.id")
            kept = page.evaluate(
                "() => [...document.querySelectorAll('.histitem')]"
                ".filter(e=>e.dataset.mark).length")
            top = page.evaluate(
                "() => document.getElementById('historyList').scrollTop")
            after_top = page.evaluate(
                """(id) => {
                  const box = document.getElementById('historyList');
                  const br = box.getBoundingClientRect();
                  const el = [...box.querySelectorAll('.histitem')]
                    .find(e => e.dataset.id === id);
                  return el ? el.getBoundingClientRect().top - br.top : null;
                }""", anchor["id"] if anchor else "")
            checks.check("D. 满页新增保留 19 个旧节点", count == 20 and kept == 19,
                         f"count={count}, kept={kept}")
            checks.check("D. 新任务在顶部", first == "new-job", f"first={first}")
            checks.check("D. 滚动位置保留", top > 0, f"scrollTop={top}")
            checks.check("D. 滚动锚点保持",
                         bool(anchor) and after_top is not None
                         and abs(after_top - anchor["top"]) <= 2,
                         f"anchor={anchor}, after_top={after_top}")
        checks.run("D. 满页新增", d_full_page)

        def g_escape():
            txt = page.evaluate(
                "() => document.querySelector('.histitem .pp').textContent")
            imgs = page.evaluate(
                "() => document.querySelectorAll('.histitem .pp img').length")
            checks.check("G. prompt HTML 被转义", txt.startswith("<img") and imgs == 0,
                         f"text={txt[:30]!r}, imgs={imgs}")
        checks.run("G. prompt HTML 被转义", g_escape)

        def e_abort():
            set_history(BASE40)
            page.evaluate("loadHistory(0)")
            page.wait_for_timeout(300)
            page.evaluate("() => { loadHistory(1); }")
            page.wait_for_timeout(50)
            page.evaluate("loadHistory(0)")
            page.wait_for_timeout(800)
            hist_page = page.evaluate("HIST_PAGE")
            first = page.evaluate(
                "() => document.querySelector('.histitem').dataset.id")
            page1 = page.evaluate(
                "() => document.querySelectorAll('.histitem[data-id^=\"job-2\"]').length")
            checks.check("E. 最新请求优先", hist_page == 0 and first == "job-00" and page1 == 0,
                         f"HIST_PAGE={hist_page}, first={first}, page1={page1}")
        checks.run("E. 最新请求优先", e_abort)

        def f_80_images():
            set_history(FULL80)
            t0 = time.perf_counter()
            page.evaluate("loadHistory(0)")
            page.wait_for_selector(".histitem", state="attached", timeout=8000)
            page.wait_for_timeout(600)
            elapsed = (time.perf_counter() - t0) * 1000
            loaded = page.evaluate(
                "() => [...document.querySelectorAll('#historyList img')]"
                ".filter(i=>i.complete && i.naturalWidth>0).length")
            total = page.evaluate(
                "() => document.querySelectorAll('#historyList img').length")
            checks.check("F. 80 图页面可渲染", total == 80,
                         f"elapsed={elapsed:.0f}ms, loaded={loaded}/{total}")
        checks.run("F. 80 图页面", f_80_images)

        def h_lightbox():
            first_image = page.eval_on_selector(".hthumb img", "el=>el.src")
            name = first_image.split("/thumb/")[1].split("?")[0]
            page.click(".hthumb img")
            src = page.eval_on_selector("#lbImg", "el=>el.src")
            checks.check("H. 灯箱先 1024", "?w=1024" in src, f"src={src}")
            disp = page.eval_on_selector(
                "#lbOrig", "el=>getComputedStyle(el).display")
            checks.check("H. 显示查看原图", disp != "none", f"display={disp}")
            page.evaluate(
                "window._editUrl=null; useForEdit=(u)=>{window._editUrl=u}")
            page.click("#lbEdit")
            edit_url = page.evaluate("window._editUrl")
            checks.check("H. 编辑使用原图",
                         bool(edit_url) and edit_url.endswith("/image/" + name),
                         f"edit_url={edit_url}")
            page.click("#lbOrig")
            src = page.eval_on_selector("#lbImg", "el=>el.src")
            checks.check("H. 查看原图切换",
                         src.endswith("/image/" + name), f"src={src}")
        checks.run("H. 灯箱 1024/原图", h_lightbox)

        browser.close()
    server.shutdown()
    print(f"\n{'ALL PASS' if checks.failed == 0 else f'{checks.failed} FAILED'}")
    return 0 if checks.failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
