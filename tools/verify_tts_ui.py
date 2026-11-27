#!/usr/bin/env python3
"""Dev-only Playwright acceptance for the ImageHub TTS tab.

Run from the repo root: python3 tools/verify_tts_ui.py
Not a runtime dependency; requires Playwright + a local Chrome.
"""
import io
import json
import threading
import time
import wave
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
INDEX = (ROOT / "index.html").read_bytes()
LOCK = threading.Lock()
STATE = {
    "status": {"loaded": True, "state": "ready", "active": 0, "pending": 0,
               "models": {"custom": "ready", "design": "ready", "clone": "ready"},
               "image": {"loaded": None, "busy": False}},
    "speak": [],
    "clone": 0,
    "speak_delay": 0,
    "history": [
        {"id": "h1", "created": 1700000000, "variant": "custom", "text": "第一条语音记录",
         "language": "Chinese", "speaker": "Vivian", "instruct": "",
         "file": "tts_h1.wav", "duration": 9.1, "elapsed": 10.9, "size": 207404},
        {"id": "h2", "created": 1700000100, "variant": "design", "text": "第二条语音记录",
         "language": "Chinese", "speaker": "", "instruct": "年轻女性",
         "file": "tts_h2.wav", "duration": 6.6, "elapsed": 6.9, "size": 172844},
    ],
}
HIST_ROWS = [dict(h) for h in STATE["history"]]


def wav_bytes(seconds=5.0, sr=24000):
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(b"\x00\x00" * int(seconds * sr))
    return buf.getvalue()


WAV = wav_bytes()
ZIP_BUF = io.BytesIO()
with zipfile.ZipFile(ZIP_BUF, "w") as _z:
    _z.writestr("a.wav", WAV)
    _z.writestr("b.wav", WAV)
ZIP_BYTES = ZIP_BUF.getvalue()


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):
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
        if path == "/api/tts/history":
            qs = parse_qs(parsed.query)
            limit = int(qs.get("limit", ["10"])[0])
            offset = int(qs.get("offset", ["0"])[0])
            with LOCK:
                rows = list(STATE["history"])[offset:offset + limit]
                total = len(STATE["history"])
            self._send({"rows": rows, "total": total, "limit": limit, "offset": offset})
            return
        if path.startswith("/api/tts/audio/"):
            self._send(WAV, "audio/wav")
            return
        if path == "/api/whoami":
            self._send({"sso_enabled": False,
                        "user": {"username": "tester", "display_name": "Tester",
                                 "is_admin": False}})
            return
        if path == "/api/models":
            self._send([{"key": "qwen", "label": "Qwen", "default_size": [1024, 1024],
                         "max_size": 2048, "note": "test", "gpus": "test",
                         "ready": True}])
            return
        if path == "/api/status":
            self._send({"loaded": None, "active": 0, "switching": False,
                        "target": None, "backend": "direct", "version": "test",
                        "tls": True, "busy": False})
            return
        if path == "/api/tts/status":
            self._send(STATE["status"])
            return
        if path == "/api/tts/voices":
            self._send({"speakers": [
                {"name": "Vivian", "description": "Bright young female",
                 "native_language": "Chinese"}],
                "languages": ["Auto", "Chinese", "English"]})
            return
        if path == "/api/settings":
            self._send({"backend": "direct", "gate": False, "clocks": False,
                        "clocks_now": False})
            return
        if path == "/api/edit/model":
            self._send({"ready": True})
            return
        if path == "/api/version":
            self._send({"name": "ImageHub", "version": "test", "build": "test"})
            return
        if path == "/api/history":
            self._send([], headers={"X-Total-Count": "0"})
            return
        if path in ("/api/gpu", "/api/power", "/api/projects"):
            self._send([])
            return
        if path.startswith("/thumb/") or path.startswith("/image/"):
            self._send(b"", "image/jpeg")
            return
        self._send({}, status=404)

    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(n)
        path = self.path.split("?")[0]
        if path == "/api/tts/speak":
            with LOCK:
                delay = STATE.get("speak_delay", 0)
            if delay:
                time.sleep(delay)
            with LOCK:
                STATE["speak"].append(json.loads(raw or b"{}"))
            self._send(WAV, "audio/wav")
            return
        if path == "/api/tts/clone":
            with LOCK:
                STATE["clone"] += 1
            self._send(WAV, "audio/wav")
            return
        if path == "/api/tts/load":
            with LOCK:
                STATE["status"]["state"] = "ready"
                STATE["status"]["loaded"] = True
            self._send({"ok": True, "state": "ready"})
            return
        if path == "/api/tts/unload":
            with LOCK:
                STATE["status"]["state"] = "stopped"
                STATE["status"]["loaded"] = False
            self._send({"ok": True, "state": "stopped"})
            return
        if path == "/api/tts/history/batch-delete":
            data = json.loads(raw or b"{}")
            ids = set(data.get("ids") or [])
            with LOCK:
                before = len(STATE["history"])
                STATE["history"] = [h for h in STATE["history"] if h["id"] not in ids]
                deleted = before - len(STATE["history"])
            self._send({"ok": True, "deleted": deleted})
            return
        if path == "/api/tts/history/batch-download":
            self._send(ZIP_BYTES, "application/zip",
                       headers={"Content-Disposition": 'attachment; filename="tts-audio.zip"'})
            return
        self._send({})

    def do_DELETE(self):
        path = self.path.split("?")[0]
        if path.startswith("/api/tts/history/"):
            jid = path.rsplit("/", 1)[-1]
            with LOCK:
                STATE["history"] = [h for h in STATE["history"] if h["id"] != jid]
            self._send({"ok": True})
            return
        self._send({})


class Checks:
    def __init__(self):
        self.failed = 0

    def check(self, name, cond, detail=""):
        print(("PASS" if cond else "FAIL") + f": {name}" + (f" — {detail}" if detail else ""),
              flush=True)
        if not cond:
            self.failed += 1

    def run(self, name, fn):
        try:
            fn()
        except Exception as e:
            self.check(name, False, f"{type(e).__name__}: {e}")


def main():
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_address[1]}/"
    checks = Checks()
    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path="/usr/bin/google-chrome",
                                    headless=True, args=["--no-sandbox"])
        page = browser.new_page(viewport={"width": 1440, "height": 1000})
        page.set_default_timeout(4000)
        page.add_init_script(
            "try{Object.defineProperty(Navigator.prototype,'mediaDevices',"
            "{get(){return undefined},configurable:true});}catch(e){}")
        page.goto(base, wait_until="domcontentloaded", timeout=30000)
        page.wait_for_timeout(400)

        def a_nav():
            page.click('a[data-page="tts"]')
            page.wait_for_selector('div.page[data-page="tts"].active', timeout=5000)
            checks.check("A. TTS 导航和页面", True)
        checks.run("A. TTS 导航和页面", a_nav)

        def b_tabs():
            tabs = page.eval_on_selector_all(".tts-tab", "els=>els.map(e=>e.textContent.trim())")
            checks.check("B. 三个子 tab",
                         tabs == ["自定义音色", "声音设计", "声音克隆"], f"tabs={tabs}")
            page.click('.tts-tab[data-tts-tab="design"]')
            checks.check("B. 声音设计字段可见",
                         page.is_visible("#ttsInstructDesign"))
            page.click('.tts-tab[data-tts-tab="clone"]')
            checks.check("B. 声音克隆字段可见", page.is_visible("#ttsRefFile"))
            page.click('.tts-tab[data-tts-tab="custom"]')
        checks.run("B. 子 tab 切换", b_tabs)

        def c_voices():
            page.wait_for_function(
                "document.querySelectorAll('#ttsSpeaker option').length > 0",
                timeout=5000)
            names = page.eval_on_selector_all(
                "#ttsSpeaker option", "els=>els.map(e=>e.value)")
            checks.check("C. 音色下拉加载", "Vivian" in names, f"names={names}")
        checks.run("C. 音色下拉", c_voices)

        def d_speak():
            page.fill("#ttsTextCustom", "你好，这是 TTS 页面测试。")
            page.click("#ttsGo")
            page.wait_for_selector("#ttsOut audio", timeout=8000)
            with LOCK:
                last = STATE["speak"][-1] if STATE["speak"] else {}
            checks.check("D. 自定义音色请求",
                         last.get("variant") == "custom"
                         and last.get("speaker") == "Vivian"
                         and last.get("text") == "你好，这是 TTS 页面测试。",
                         f"body={last}")
            href = page.eval_on_selector("#ttsOut a", "el=>el.getAttribute('href')")
            checks.check("D. 播放器和下载链接", bool(href) and href.startswith("blob:"),
                         f"href={href}")
        checks.run("D. 自定义音色生成", d_speak)

        def e_design_validation():
            page.click('.tts-tab[data-tts-tab="design"]')
            page.fill("#ttsTextDesign", "测试声音设计")
            before = page.eval_on_selector_all("#ttsOut audio", "els=>els.length")
            page.click("#ttsGo")
            page.wait_for_timeout(300)
            msg = page.text_content("#ttsMsg") or ""
            after = page.eval_on_selector_all("#ttsOut audio", "els=>els.length")
            checks.check("E. 声音设计缺指令被拦截",
                         "声音描述" in msg and before == after, f"msg={msg}")
            page.click('.tts-tab[data-tts-tab="custom"]')
        checks.run("E. 声音设计校验", e_design_validation)

        def f_clone_inputs():
            page.click('.tts-tab[data-tts-tab="clone"]')
            checks.check("F. 克隆有文件上传", page.is_visible("#ttsRefFile"))
            disp = page.eval_on_selector(
                "#ttsRecordBtn", "el=>getComputedStyle(el).display")
            checks.check("F. 无麦克风时隐藏录音", disp == "none", f"display={disp}")
            page.click('.tts-tab[data-tts-tab="custom"]')
        checks.run("F. 克隆输入", f_clone_inputs)

        def g_loading_pending():
            with LOCK:
                STATE["status"] = {"loaded": False, "state": "loading",
                                   "active": 0, "pending": 0, "models": {},
                                   "image": {"busy": False}}
            page.evaluate("loadTTS()")
            page.wait_for_function(
                "document.getElementById('ttsStatus').textContent.includes('正在加载三个模型')",
                timeout=5000)
            checks.check("G. 加载中状态", True)
            with LOCK:
                STATE["status"] = {"loaded": True, "state": "ready",
                                   "active": 0, "pending": 1,
                                   "models": {"custom": "ready"},
                                   "image": {"busy": False}}
            page.evaluate("loadTTS()")
            page.wait_for_function(
                "document.getElementById('ttsStatus').textContent.includes('排队中')",
                timeout=5000)
            checks.check("G. 排队状态", True)
        checks.run("G. 加载/排队状态", g_loading_pending)

        def h_image_busy():
            with LOCK:
                STATE["status"] = {"loaded": True, "state": "ready",
                                   "active": 0, "pending": 0,
                                   "models": {"custom": "ready"},
                                   "image": {"loaded": "qwen", "busy": True}}
            page.evaluate("loadTTS()")
            page.wait_for_function(
                "document.getElementById('ttsStatus').textContent.includes('等待图像任务完成')",
                timeout=5000)
            checks.check("H. 图像任务提示", True)
        checks.run("H. 图像互斥提示", h_image_busy)

        def i_cert_button():
            page.evaluate("refreshStatus()")
            page.wait_for_function(
                "getComputedStyle(document.getElementById('certBtn')).display !== 'none'",
                timeout=5000)
            page.click("#certBtn")
            checks.check("I. 证书弹窗", page.is_visible("#certModal"))
            href = page.eval_on_selector("#certModal a", "el=>el.getAttribute('href')")
            checks.check("I. 证书下载链接", href == "/api/tls/cert", f"href={href}")
            page.click("#certModal button")
            checks.check("I. 弹窗关闭", not page.is_visible("#certModal"))
        checks.run("I. 证书按钮", i_cert_button)

        def j_history():
            page.evaluate("loadTTSHistory(0)")
            page.wait_for_selector(".tts-histitem", timeout=5000)
            count = page.eval_on_selector_all(".tts-histitem", "els=>els.length")
            checks.check("J. 记录列表", count == 2, f"count={count}")
            text = page.text_content("#ttsHistList") or ""
            checks.check("J. 显示时长/用时", "时长" in text and "生成" in text)
            src = page.eval_on_selector(".tts-histitem audio", "el=>el.getAttribute('src')")
            checks.check("J. 可播放", src.startswith("/api/tts/audio/"), f"src={src}")
            dl = page.eval_on_selector(
                ".tts-histitem a[download]",
                "el=>({href:el.getAttribute('href'), name:el.getAttribute('download')})")
            checks.check("J. 单条下载链接",
                         dl["href"].startswith("/api/tts/audio/")
                         and dl["name"].endswith(".wav"), f"{dl}")
            page.click("#ttsAutoBtn")
            page.wait_for_function(
                "document.getElementById('toastText').textContent.includes('请先选择')",
                timeout=3000)
            checks.check("J. 未选不轮播",
                         "自动轮播" in (page.text_content("#ttsAutoBtn") or "")
                         and page.eval_on_selector_all(
                             ".tts-histitem.tts-playing", "els=>els.length") == 0)
            page.check(".tts-histitem .tts-sel")
            with page.expect_download(timeout=5000) as di:
                page.click("#ttsDownloadSel")
            single = di.value
            checks.check("J. 单文件直接下载",
                         single.suggested_filename.endswith(".wav"),
                         single.suggested_filename)
            page.check("#ttsSelAll")
            with page.expect_download(timeout=5000) as dz:
                page.click("#ttsDownloadSel")
            multi = dz.value
            checks.check("J. 多文件打包下载",
                         multi.suggested_filename.endswith(".zip"),
                         multi.suggested_filename)
            page.click("#ttsAutoBtn")
            page.wait_for_function(
                "document.getElementById('ttsAutoBtn').textContent.includes('停止轮播')",
                timeout=3000)
            checks.check("J. 轮播开始",
                         page.eval_on_selector_all(
                             ".tts-histitem.tts-playing", "els=>els.length") == 1)
            page.click("#ttsAutoBtn")
            checks.check("J. 轮播停止",
                         "自动轮播" in (page.text_content("#ttsAutoBtn") or ""))
            page.on("dialog", lambda d: d.accept())
            page.check("#ttsSelAll")
            checks.check("J. 全选计数", page.text_content("#ttsSelCount").strip() == "2")
            page.click("#ttsDeleteSel")
            page.wait_for_function(
                "document.querySelectorAll('.tts-histitem').length === 0", timeout=5000)
            page.wait_for_function(
                "document.getElementById('toastText').textContent.includes('删除')",
                timeout=3000)
            checks.check("J. 删除飘窗", True)
            with LOCK:
                STATE["history"] = [dict(h) for h in HIST_ROWS]
            page.evaluate("loadTTSHistory(0)")
            page.wait_for_selector(".tts-histitem", timeout=5000)
            page.click(".tts-histitem button:has-text('删除')")
            page.wait_for_function(
                "document.querySelectorAll('.tts-histitem').length === 1", timeout=5000)
            checks.check("J. 单条删除", True)
        checks.run("J. 生成记录列表", j_history)

        def k_api_docs():
            page.click('a[data-page="api"]')
            page.wait_for_selector('div.page[data-page="api"].active', timeout=5000)
            tabs = page.eval_on_selector_all(
                ".apitab", "els=>els.map(e=>e.textContent.trim())")
            checks.check("K. API 子 tab",
                         tabs == ["文生图", "图片编辑", "多模型对比", "TTS 语音",
                                  "语音记录", "鉴权与证书"], f"tabs={tabs}")
            checks.check("K. 默认文生图", page.is_visible('.api-pane[data-api-pane="img"]'))
            page.click('.apitab[data-apitab="tts"]')
            tts_text = page.text_content('.api-pane[data-api-pane="tts"]') or ""
            checks.check("K. TTS 参数",
                         all(k in tts_text for k in
                             ("input", "voice", "ref_audio", "x_vector_only_mode")))
            page.click('.apitab[data-apitab="ttshist"]')
            hist_text = page.text_content('.api-pane[data-api-pane="ttshist"]') or ""
            checks.check("K. 语音记录接口",
                         "/api/tts/history" in hist_text and "/api/tts/audio" in hist_text)
            page.click('.apitab[data-apitab="auth"]')
            auth_text = page.text_content('.api-pane[data-api-pane="auth"]') or ""
            checks.check("K. 证书说明", "--cacert" in auth_text and "401" in auth_text)
            rows = page.eval_on_selector_all(".api-table tr", "els=>els.length")
            checks.check("K. 参数表", rows >= 20, f"rows={rows}")
            origin = page.evaluate("location.origin")
            base = (page.text_content("#apiBase") or "").strip()
            checks.check("K. 当前地址", base == origin, f"base={base}, origin={origin}")
            note = page.text_content("#apiAddrNote") or ""
            checks.check("K. IP/端口", "IP " in note and "端口 " in note, f"note={note}")
            tts_curl = page.text_content("#api-tts-curl") or ""
            checks.check("K. 示例使用当前地址", origin in tts_curl, f"origin={origin}")
            doc = page.eval_on_selector(
                'a[href="/api/docs"]',
                "el=>({href:el.getAttribute('href'), name:el.getAttribute('download'),"
                " text:el.textContent.trim()})")
            checks.check("K. 下载说明按钮",
                         doc["href"] == "/api/docs" and "下载" in doc["text"]
                         and doc["name"].endswith(".md"), f"{doc}")
        checks.run("K. API 文档", k_api_docs)

        def l_model_chips():
            page.wait_for_selector("#modelChips .mchip", timeout=5000)
            chips = page.eval_on_selector_all(
                "#modelChips .mchip", "els=>els.map(e=>e.textContent.trim())")
            checks.check("L. 顶部状态",
                         len(chips) == 2 and any("图像" in c for c in chips)
                         and any("TTS" in c for c in chips), f"{chips}")
            page.click('a[data-page="library"]')
            checks.check("L. 图库页仍显示", page.is_visible("#modelChips"))
            page.click('a[data-page="api"]')
            checks.check("L. API 页仍显示", page.is_visible("#modelChips"))
            page.click('a[data-page="tts"]')
        checks.run("L. 顶部模型状态", l_model_chips)

        def m_toast():
            page.click('a[data-page="tts"]')
            page.fill("#ttsTextCustom", "飘窗测试")
            with LOCK:
                STATE["speak_delay"] = 1.2
            try:
                page.click("#ttsGo")
                page.wait_for_selector("#toast", state="visible", timeout=3000)
                txt = page.text_content("#toastText") or ""
                checks.check("M. 飘窗提示",
                             ("TTS" in txt) or ("合成" in txt) or ("加载" in txt),
                             f"text={txt}")
                page.wait_for_selector("#ttsOut audio", timeout=8000)
                page.wait_for_function(
                    "document.getElementById('toastText').textContent.includes('合成完成')",
                    timeout=4000)
                checks.check("M. 完成提示", True)
            finally:
                with LOCK:
                    STATE["speak_delay"] = 0
        checks.run("M. 生成飘窗", m_toast)

        browser.close()
    server.shutdown()
    print(f"\n{'ALL PASS' if checks.failed == 0 else f'{checks.failed} FAILED'}")
    return 0 if checks.failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
