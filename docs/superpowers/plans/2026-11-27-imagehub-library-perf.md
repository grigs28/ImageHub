# ImageHub 图库性能优化 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 让图库列表点击即开、停留不卡，并消除历史查询写库、无索引、缩略图冷启动和灯箱原图带来的放大效应。

**Architecture:** 前端用“最新请求优先 + 签名对比 + 通用 DOM diff”替代每 20 秒全量 `innerHTML`；后端把 schema/迁移收敛到启动一次，历史查询走只读连接并加 `created` 索引；任务完成后后台预生成 160/480/1024 缩略图；灯箱先 1024 后原图；新增 `busy` 判定供安全重启使用。

**Tech Stack:** Python 3.12 / FastAPI / SQLite 3.45 / Pillow / 原生 JS / Playwright（仅开发验收）。

**Spec:** `docs/superpowers/specs/2026-11-27-imagehub-library-perf-design.md`

## Global Constraints

- 版本号：`APP_VERSION = "1.5.1"`，`APP_BUILD = "2026-11-27"`。
- 不新增运行时 Python 依赖；缩略图继续只用 Pillow，队列/线程用标准库 `queue`/`threading`。
- 不改现有 API 路径和响应结构；`/api/status` 只新增 `busy` 字段（加法变更）。
- 不引入 nginx/HTTP2、不改成多 worker uvicorn、不引入前端框架。
- 缩略图宽度固定为 `160`（列表）、`480`（结果网格）、`1024`（灯箱预览）；`/image` 原图只在用户点「查看原图」后加载。
- 历史读取使用 `db(readonly=True)`；`_ensure_schema()` 只执行一次；索引名为 `idx_jobs_created`、`idx_jobs_project_created`。
- `/api/compare` 的 `run_job()` 必须保留首次 `created`，只更新 `results/status`。
- 新增环境变量：`IMAGEHUB_DB`、`IMAGEHUB_THUMBS`、`IMAGEHUB_RUN`（测试隔离用，默认值不变）。
- 不要把 SSH 密码/Token 写进仓库或计划；生产部署时用 `IMAGEHUB_SSH_PASS` 环境变量。
- 不执行 `git commit`/`git push`：用户没有要求提交；当前分支是 `main`。每个任务结束只做测试与 `git diff` 报告；若用户后来要求提交，先建分支。
- 生产部署必须：先向用户再次确认重启窗口；备份 `app.py`/`index.html`/`imagehub.db`；部署后逐项验收；失败按备份回滚。
- 测试命令：`python -m unittest discover -s tests -v`；前端验收：`python3 tools/verify_library_perf.py`（仅开发机，不加入 `requirements.txt`）。

## Review Focus

1. **满页 20 条时新增一条**：尾部移除、顶部插入后，未变化的 19 个 `.histitem` 必须保持原 DOM 对象，滚动位置按高度差补偿；不能退化成全量重建。
2. **并发历史请求/翻页**：慢的旧请求被新请求 abort 后不得落地；`HIST_PAGE`、列表内容和分页按钮必须对应最新一次请求。
3. **对比任务状态与排序**：`run_job` 运行中反复写 `results/status` 时，`created` 不得改变，历史不得反复跳到最前；启动时只标记一次 `interrupted`。
4. **只读连接与 WAL**：schema 只建一次，`db(readonly=True)` 在 WAL 下可读、写会抛 `OperationalError`，后续读请求不得再把 running 任务改成 interrupted。
5. **忙碌判定**：ComfyUI 设计锁、`_dq` 队列/运行、直连 active、内存中 running 对比任务任一存在时，`busy` 必须为 true，避免重启打断任务。

---

### Task 1: 后端 schema、只读连接、索引、run_job 保留 created

**Files:**
- Modify: `app.py`（顶部路径常量、`db()` 约 230 行、`run_job()` 约 990 行、startup 事件、只读端点）
- Create: `tests/test_library_perf.py`

**Interfaces:**
- Consumes: 现有 SQLite 表结构和 `jobs` 字段。
- Produces:
  - `_ensure_schema() -> None`
  - `db(readonly: bool = False) -> sqlite3.Connection`
  - `_upsert_job_result(job_id: str, prompt: str, models: list[str], created: float, results: list[dict], status: str) -> None`
  - `run_job()` 调用 `_upsert_job_result()`，`created` 取 `_jobs[job_id]["created"]`。

- [ ] **Step 1: 写失败的单元测试**

创建 `tests/test_library_perf.py`：

```python
import json, os, shutil, sqlite3, tempfile, time, unittest

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
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m unittest discover -s tests -v`
Expected: FAIL，出现 `AttributeError: module 'app' has no attribute '_ensure_schema'` 或 `db() got an unexpected keyword argument 'readonly'`。

- [ ] **Step 3: 实现 schema、只读连接、索引、UPSERT**

在 `app.py` 顶部把路径改为可覆盖：

```python
DB_PATH = os.environ.get("IMAGEHUB_DB", os.path.join(BASE_DIR, "imagehub.db"))
RUN_DIR = os.environ.get("IMAGEHUB_RUN", os.path.join(BASE_DIR, "run"))
```

在 `db()` 前新增：

```python
_schema_lock = threading.Lock()
_schema_ready = False

def _ensure_schema():
    global _schema_ready
    if _schema_ready:
        return
    with _schema_lock:
        if _schema_ready:
            return
        conn = sqlite3.connect(DB_PATH, timeout=5)
        try:
            # 原有 7 个 CREATE TABLE IF NOT EXISTS
            # PRAGMA journal_mode=WAL
            # 原有 3 个 ALTER TABLE（try/except 吞掉重复列）
            conn.execute("CREATE INDEX IF NOT EXISTS idx_jobs_created ON jobs(created DESC)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_jobs_project_created ON jobs(project, created DESC)")
            conn.execute("UPDATE jobs SET status='interrupted' WHERE status='running'")
            conn.commit()
        finally:
            conn.close()
        _schema_ready = True

def db(readonly=False):
    _ensure_schema()
    if readonly:
        conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True, timeout=5)
    else:
        conn = sqlite3.connect(DB_PATH, timeout=5)
    conn.execute("PRAGMA busy_timeout=5000")
    return conn
```

新增 UPSERT helper（放在 `record_job` 前）：

```python
def _upsert_job_result(job_id, prompt, models, created, results, status):
    conn = db()
    try:
        conn.execute(
            "INSERT INTO jobs(id,prompt,models,created,results,status) "
            "VALUES(?,?,?,?,?,?) ON CONFLICT(id) DO UPDATE SET "
            "results=excluded.results, status=excluded.status",
            (job_id, prompt, json.dumps(models), created,
             json.dumps(results), status),
        )
        conn.commit()
    finally:
        conn.close()
```

把 `run_job()` 中两处 `INSERT OR REPLACE` 替换为 `_upsert_job_result(...)`，并使用：

```python
created = _jobs[job_id].get("created", time.time())
```

在现有 startup 事件之前新增：

```python
@app.on_event("startup")
def _on_startup_schema():
    _ensure_schema()
```

把以下只读端点中的 `conn = db()` 改成 `conn = db(readonly=True)`：
`api_history`、`api_job` 的 DB 兜底、`api_projects`、`api_project_assets`、`admin_users`、`admin_keys`、`admin_audit`、`api_design_presets`、`admin_presets`。写端点保持 `db()`。

- [ ] **Step 4: 运行测试确认通过**

Run: `python -m unittest discover -s tests -v`
Expected: PASS（4 个测试）。

- [ ] **Step 5: 检查差异，不提交**

Run: `git diff --stat`
Expected: 只看到 `app.py` 和 `tests/test_library_perf.py`。不要执行 `git commit`。

---

### Task 2: 缩略图 helper、预生成队列、启动回填

**Files:**
- Modify: `app.py`（`LEGACY_DIRS`/`THUMB_DIR` 上移、`thumb()` 约 2308 行、`record_job()`、`run_job()`、`_dq_worker()`、startup 事件）
- Modify: `tests/test_library_perf.py`

**Interfaces:**
- Consumes: Task 1 的 `db(readonly=True)`、`_ensure_schema()`。
- Produces:
  - `_thumb_path(filename: str, w: int) -> str`
  - `_find_source(filename: str) -> str | None`
  - `_ensure_thumb(filename: str, w: int) -> str | None`
  - `_prewarm_thumbs(results: list[dict]) -> None`
  - `_prewarm_history_thumbs(limit: int = 200) -> None`
  - `_thumb_q: queue.Queue`

- [ ] **Step 1: 追加失败测试**

在 `tests/test_library_perf.py` 增加（使用 Pillow）：

```python
from PIL import Image

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
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m unittest discover -s tests -v`
Expected: FAIL，出现 `_ensure_thumb`/`THUMB_DIR`/`_thumb_path`/`_prewarm_thumbs` 不存在。

- [ ] **Step 3: 实现缩略图 helper 与队列**

- 把 `LEGACY_DIRS`、`THUMB_DIR` 移到文件顶部路径区，并改为：

```python
LEGACY_DIRS = ["/opt/ComfyUI/output"]
THUMB_DIR = os.environ.get("IMAGEHUB_THUMBS", os.path.join(BASE_DIR, "thumbs"))
os.makedirs(THUMB_DIR, exist_ok=True)
```

- 新增 `_thumb_path`、`_find_source`、`_ensure_thumb`。`_ensure_thumb` 必须：
  - `w = max(64, min(int(w), 1024))`；缓存命中直接返回。
  - 用每文件锁避免请求和预生成重复解码；二次检查缓存。
  - 用 `tempfile.mkstemp(prefix=".thumb_", suffix=".jpg", dir=THUMB_DIR)` 写临时文件，`os.replace` 到目标；异常时删除临时文件。
  - 源文件不存在返回 `None`。
- 新增 `queue.Queue(maxsize=2000)`、2 个 daemon worker、`_start_thumb_workers()`；`_prewarm_thumbs()` 对 `status == "ok"` 且有 `image` 的结果投递 160/480/1024，`queue.Full` 时 warning 并返回。
- 新增 `_prewarm_history_thumbs()`：`db(readonly=True)` 读取最近 200 条 `results`，解析后调用 `_prewarm_thumbs`。
- 在 startup 事件中启动一个 daemon 线程执行 `_prewarm_history_thumbs`。
- `thumb()` 改为：

```python
try:
    out = _ensure_thumb(filename, w)
except Exception as e:
    raise HTTPException(500, f"缩略图生成失败: {str(e)[:120]}")
if not out:
    raise HTTPException(404, "not found")
return FileResponse(out, media_type="image/jpeg",
                    headers={"Cache-Control": "public, max-age=86400"})
```

- 在 `record_job()` 插入成功后、`run_job()` 最终 done 后、`_dq_worker()` 更新 done 后调用 `_prewarm_thumbs(results)`。

- [ ] **Step 4: 运行测试确认通过**

Run: `python -m unittest discover -s tests -v`
Expected: PASS（Task 2 新增测试全部通过）。

- [ ] **Step 5: 检查差异，不提交**

Run: `git diff --stat`
Expected: `app.py`、`tests/test_library_perf.py`。不要提交。

---

### Task 3: 前端历史列表增量渲染、最新请求优先、轮询条件

**Files:**
- Modify: `index.html`（`loadHistory` 约 683 行、`switchPage` 约 1425 行、底部初始化约 1461 行、历史项模板约 709 行）
- Create: `tools/verify_library_perf.py`

**Interfaces:**
- Consumes: `/api/history` 现有响应结构（`id/prompt/models/created/status/images`）。
- Produces（JS）：
  - `CURRENT_PAGE`、`HIST_SIG`、`HIST_ABORT`
  - `loadHistory(page?)`
  - `renderHistoryRows(rows)`
  - `buildHistoryItem(h)`
  - `itemSig(h)`
  - `esc(value)`
  - `pollHistory()`

- [ ] **Step 1: 写失败的 Playwright 验收脚本**

创建 `tools/verify_library_perf.py`，结构如下（仅开发机运行，不加入 `requirements.txt`）：

```python
# 1. 启动本地 ThreadingHTTPServer（127.0.0.1:0），同时提供：
#    - GET /          -> index.html
#    - GET /api/*     -> 固定 JSON；HISTORY 是可变的全局列表
#    - GET /thumb/*   -> 1x1 JPEG bytes
#    - /api/history 的 offset=20 请求 sleep 0.5s（用于 abort 测试）
# 2. 不用 page.route；所有请求都走本地 HTTP 服务器，线程安全、可延迟。
# 3. 用 sync_playwright + /usr/bin/google-chrome 打开本地页面。
```

必须包含这些断言：

```python
# A. 启动不自动加载；首次点图库只 1 次 /api/history
assert history_requests == 0
page.click('a[data-page="library"]')
page.wait_for_selector(".histitem", state="attached")
assert history_requests == 1

# B. 工作台不轮询
page.evaluate("CURRENT_PAGE='work'")
page.evaluate("pollHistory()")
page.wait_for_timeout(200)
assert history_requests == 1

# C. 数据未变不重渲染
marks = page.evaluate("""() => {
  const els=[...document.querySelectorAll('.histitem')];
  els.forEach((e,i)=>e.dataset.mark='m'+i);
  return els.map(e=>e.dataset.mark);
}""")
page.evaluate("loadHistory(0)")
page.wait_for_timeout(300)
assert page.evaluate("""() => [...document.querySelectorAll('.histitem')].map(e=>e.dataset.mark)""") == marks

# D. 满页新增：保留 19 个旧节点、删尾部、顶部插入
# HISTORY = [new_job] + HISTORY[:19]
page.evaluate("document.getElementById('historyList').scrollTop = 200")
page.evaluate("loadHistory(0)")
page.wait_for_timeout(300)
assert page.evaluate("""() => document.querySelectorAll('.histitem').length""") == 20
assert page.evaluate("""() => document.querySelector('.histitem').dataset.id""") == "new-job"
assert page.evaluate("""() => [...document.querySelectorAll('.histitem')].filter(e=>e.dataset.mark).length""") == 19
assert page.evaluate("() => document.getElementById('historyList').scrollTop") > 0

# E. 并发：慢的旧请求被 abort，最新请求落地
page.evaluate("loadHistory(1)")   # offset=20，mock 会 sleep
page.wait_for_timeout(50)
page.evaluate("loadHistory(0)")
page.wait_for_timeout(700)
assert page.evaluate("HIST_PAGE") == 0

# F. 80 张图模拟：记录点击到可见图片完成的耗时（不设硬性门槛）

# G. prompt 中的 HTML 被转义
# mock 一条 prompt 为 "<img src=x onerror=alert(1)>" 的历史
assert page.evaluate("""() => document.querySelector('.histitem .pp').textContent""").startswith("<img")
assert page.evaluate("""() => document.querySelectorAll('.histitem .pp img').length""") == 0
```

Run: `python3 tools/verify_library_perf.py`
Expected: FAIL（启动阶段就会请求历史；首次点击出现 2 次；没有 abort）。

- [ ] **Step 2: 实现前端状态与 loadHistory**

在 `index.html` 顶部脚本区新增：

```js
let CURRENT_PAGE = "work";
let HIST_SIG = "";
let HIST_ABORT = null;
```

`switchPage(name)`：

```js
CURRENT_PAGE = name;
...
if (name === "library") {
  loadHistory();
} else if (!_pageLoaded.has(name)) {
  _pageLoaded.add(name);
  if (name === "admin") { ... } else if (name === "design") { loadPresets(); }
}
```

删除页面底部无条件的 `loadHistory();`。

`loadHistory(page)` 必须：

- 先 `HIST_ABORT?.abort()`，创建新 `AbortController`，`fetch(url, {cache:"no-store", signal})`；捕获 `AbortError` 直接返回。
- 先设置 `HIST_PAGE`，再发请求；只有当前 controller 的响应可以更新 UI。
- 读取 `X-Total-Count`，更新 `HIST`、分页按钮、`histPage`、`histTotal`。
- `const sig = JSON.stringify(rows)`；`sig === HIST_SIG` 时直接返回，不碰 DOM。
- 生成完成后的 `loadHistory()` 调用保留；不在图库页时签名未变则不重建 DOM，签名变了只更新隐藏列表。
- `HIST` 对象在每次响应后更新，供 `copyPrompt` 使用。
- 否则调用 `renderHistoryRows(rows)`。

- [ ] **Step 3: 实现通用 DOM diff 与历史项构建**

`renderHistoryRows(rows)`：

- 空列表显示“暂无记录”，清空 `HIST_SIG` 后返回。
- 记录 `oldTop = box.scrollTop`、`oldHeight = box.scrollHeight`。
- 现有 `.histitem` 按 `data-id` 放入 `Map`。
- 按新顺序逐条：签名相同复用节点；不同 `replaceWith(buildHistoryItem(h))`；没有则创建。处理过的从 `Map` 删除。
- `Map` 剩余节点 remove。
- 所有节点按顺序放入 `DocumentFragment`，再 `box.replaceChildren(fragment)`（同时清掉“暂无记录”等占位节点）。
- 恢复滚动：记录首个可见 `.histitem` 的 `data-id` 及其相对滚动容器顶部的偏移；重建后找到同一节点，按其新偏移恢复 `scrollTop`；找不到锚点时才回退到高度差；在顶部则保持 0。

`buildHistoryItem(h)`：

- `document.createElement("div")`，设置 `className="histitem"`、`dataset.id`、`dataset.sig=itemSig(h)`、`onclick=()=>showJob(h.id)`。
- 用现有模板生成 `innerHTML`；对 `h.prompt.slice(0,80)`、`meta`、`title` 中的文本使用 `esc()`。
- 图片标签加 `decoding="async"`，保留 `loading="lazy"`。

`itemSig(h)` 返回 `JSON.stringify(h)`。

`esc(value)`：

```js
function esc(s){
  return String(s ?? "").replace(/[&<>"']/g, c => ({
    "&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"
  }[c]));
}
```

删除 `box.style.maxHeight`/`offsetHeight` 测高逻辑（`index.html:727-734`）。

- [ ] **Step 4: 实现 pollHistory 并验证**

```js
function pollHistory(){
  if (CURRENT_PAGE === "library" && HIST_PAGE === 0) loadHistory();
}
pollWhenVisible(pollHistory, 20000);
```

Run: `python3 tools/verify_library_perf.py`
Expected: A~E 全部 PASS；F 打印 80 张图的耗时。

- [ ] **Step 5: 检查差异，不提交**

Run: `git diff --stat`
Expected: `index.html`、`tools/verify_library_perf.py`。不要提交。

---

### Task 4: 灯箱 1024 预览 + 按需原图

**Files:**
- Modify: `index.html`（灯箱 HTML 约 441 行、`openLightbox` 约 1063 行、三个调用点约 613/710/1176 行）
- Modify: `tools/verify_library_perf.py`

**Interfaces:**
- Consumes: Task 2 的 1024 缩略图；Task 3 的历史列表 DOM。
- Produces:
  - `openLightbox(src, title, info, orig)`
  - `lbOrig` 按钮、`lbOrigUrl`
  - `lbCurrent` 始终是原图 URL（供 `lbEdit`）。

- [ ] **Step 1: 追加失败断言**

在 `tools/verify_library_perf.py` 增加：

```python
# 打开历史缩略图：先 1024，不请求原图
first_image = page.eval_on_selector(".hthumb img", "el=>el.src")
name = first_image.split("/thumb/")[1].split("?")[0]
page.click(".hthumb img")
assert "?w=1024" in page.eval_on_selector("#lbImg", "el=>el.src")
assert page.eval_on_selector("#lbOrig", "el=>getComputedStyle(el).display") != "none"

# 编辑使用原图
page.evaluate("window._editUrl=null; useForEdit=(u)=>{window._editUrl=u}")
page.click("#lbEdit")
assert page.evaluate("window._editUrl").endswith("/image/" + name)

# 点查看原图才切到原图
page.click("#lbOrig")
assert page.eval_on_selector("#lbImg", "el=>el.src").endswith("/image/" + name)
```

Run: `python3 tools/verify_library_perf.py`
Expected: FAIL（当前 src 是 `/image/...`，没有 `#lbOrig`）。

- [ ] **Step 2: 实现灯箱**

HTML 在 `#lbEdit` 后加：

```html
<button class="mini" id="lbOrig" style="font-size:11px;display:none">查看原图</button>
```

`openLightbox` 改为：

```js
function openLightbox(src, title, info, orig){
  lbCurrent = orig || src;
  lbImg.src = src;
  lbOrigUrl = orig && orig !== src ? orig : "";
  document.getElementById("lbOrig").style.display = lbOrigUrl ? "" : "none";
  ...
}
```

增加：

```js
document.getElementById("lbOrig").onclick = () => {
  if (!lbOrigUrl) return;
  lbImg.src = lbOrigUrl;
  lbCurrent = lbOrigUrl;
  lbOrigUrl = "";
  document.getElementById("lbOrig").style.display = "none";
  scale = 1; tx = 0; ty = 0; applyTransform();
};
```

`closeLightbox` 清空 `lbOrigUrl` 并隐藏按钮。

三个调用点改为 1024 预览 + 原图：

- 结果网格：`openLightbox('/thumb/${r.image}?w=1024', ..., '/image/${r.image}')`
- 历史列表：`openLightbox('/thumb/${im.image}?w=1024', ..., '/image/${im.image}')`
- 编辑结果：`openLightbox('/thumb/${d.image}?w=1024', ..., '/image/${d.image}')`

- [ ] **Step 3: 运行验收**

Run: `python3 tools/verify_library_perf.py`
Expected: Task 3 的 A~E + 灯箱断言全部 PASS。

- [ ] **Step 4: 检查差异，不提交**

Run: `git diff --stat`
Expected: `index.html`、`tools/verify_library_perf.py`。不要提交。

---

### Task 5: busy 判定 + deploy-guard.sh

**Files:**
- Modify: `app.py`（`api_status` 约 2162 行附近）
- Modify: `deploy-guard.sh`
- Modify: `tests/test_library_perf.py`

**Interfaces:**
- Consumes: `_state`、`_comfy_lock`、`_dq`、`_jobs`。
- Produces: `_busy() -> bool`；`GET /api/status` 增加 `busy`；`deploy-guard.sh` 优先读 `busy`。

- [ ] **Step 1: 写失败的 busy 测试**

```python
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
```

Run: `python -m unittest discover -s tests -v`
Expected: FAIL（`_busy` 不存在）。

- [ ] **Step 2: 实现 `_busy` 与 `/api/status`**

在 `api_status` 前新增：

```python
def _busy():
    with _cond:
        active = _state.get("active", 0)
        switching = _state.get("switching", False)
    return bool(
        active or switching
        or _comfy_lock.locked()
        or _dq.get("running")
        or _dq.get("queue")
        or any(j.get("status") == "running" for j in list(_jobs.values()))
    )
```

`api_status` 返回中增加 `"busy": _busy()`。

- [ ] **Step 3: 更新 deploy-guard.sh**

在循环中先读 `busy`，没有该字段时回退 `active`；连续两次采样为 0 才 break（循环前 `Z=0`）：

```bash
B=$(curl -s -m 2 "$URL" | python3 -c "import json,sys;d=json.load(sys.stdin);print(int(d.get('busy',d.get('active',0))))" 2>/dev/null || echo 1)
if [ "$B" = "0" ]; then Z=$((Z+1)); else Z=0; fi
[ "$Z" -ge 2 ] && break
```

保留原有 300 次、2 秒间隔的上限。修改后运行 `bash -n deploy-guard.sh`，Expected: 无输出（语法正确）。

- [ ] **Step 4: 运行测试**

Run: `python -m unittest discover -s tests -v`
Expected: PASS（Task 5 新增测试通过）。
Run: `bash -n deploy-guard.sh`
Expected: 无输出。

- [ ] **Step 5: 检查差异，不提交**

Run: `git diff --stat`
Expected: `app.py`、`deploy-guard.sh`、`tests/test_library_perf.py`。不要提交。

---

### Task 6: 版本、README、NOTICE

**Files:**
- Modify: `app.py`（`APP_VERSION`/`APP_BUILD`）
- Modify: `README.md`
- Create: `NOTICE`

**Interfaces:**
- Consumes: 无。
- Produces: `/api/version` 返回 `1.5.1`；README/NOTICE 记录本次变更。

- [ ] **Step 1: 写版本断言**

在 `tests/test_library_perf.py` 增加：

```python
def test_version(self):
    self.assertEqual(imagehub.APP_VERSION, "1.5.1")
```

Run: `python -m unittest discover -s tests -v`
Expected: FAIL（当前 1.5.0）。

- [ ] **Step 2: 更新版本与文档**

- `app.py`：`APP_VERSION = "1.5.1"`，`APP_BUILD = "2026-11-27"`。
- `README.md`：版本徽章改为 `1.5.1`；在 1.5.0 段落后新增：

```markdown
## 1.5.1 图库性能

- 历史列表增量更新：数据未变不重渲染，新增任务只插新行，保留滚动位置
- 图库页可见时才轮询，首次点击不再重复加载
- 生成完成后后台预生成 160/480/1024 缩略图
- 灯箱先加载 1024 预览，原图按需加载；编辑仍使用原图
- 历史查询改只读连接；建表/迁移只执行一次；`jobs(created)` 加索引
- 对比任务保留首次 `created`，运行中不再跳到历史最前
```

- `NOTICE`（新建）：

```text
ImageHub

变更记录：
2026-11-27  v1.5.1  图库列表性能优化：增量渲染、缩略图预生成、
灯箱按需原图、只读历史查询、jobs(created) 索引、busy 安全重启判定。
```

- [ ] **Step 3: 验证**

Run: `python -m unittest discover -s tests -v`
Expected: PASS（Task 6 新增测试通过）。
Run: `grep -n "1.5.1" README.md NOTICE app.py`
Expected: 三处都能看到 1.5.1。

- [ ] **Step 4: 检查差异，不提交**

Run: `git diff --stat`
Expected: `app.py`、`README.md`、`NOTICE`、`tests/test_library_perf.py`。不要提交。

---

### Task 7: 生产部署、验收、回滚

**Files:**
- 操作：生产 `192.168.0.70:/opt/vLLM/imagehub`
- 备份：`app.py`、`index.html`、`deploy-guard.sh`、`imagehub.db`

**Interfaces:**
- Consumes: Task 1~6 的代码与测试。
- Produces: 生产运行 1.5.1，`/api/status.busy` 可用，图库验收通过。

- [ ] **Step 1: 向用户确认重启窗口（必须）**

执行前在当前 shell 设置 `IMAGEHUB_SSH_PASS`（用 `read -s IMAGEHUB_SSH_PASS` 输入），**不要把密码写进仓库、计划或命令历史**。在开始前明确询问并等待用户确认；不要用“用户选了部署 A”替代本次窗口确认。确认后检查：

```bash
curl -s http://192.168.0.70:5569/api/status
curl -s http://192.168.0.70:5569/api/version
```

若旧版 `/api/status` 没有 `busy`，以 `active=0` + 用户确认无设计/异步任务为准；新版 `busy` 从下一次重启开始生效。

- [ ] **Step 2: 备份生产文件与数据库**

```bash
sshpass -p "$IMAGEHUB_SSH_PASS" ssh root@192.168.0.70 '
  cd /opt/vLLM/imagehub
  cp -a app.py app.py.bak.20261127
  cp -a index.html index.html.bak.20261127
  cp -a deploy-guard.sh deploy-guard.sh.bak.20261127 2>/dev/null || true
  sqlite3 imagehub.db ".backup imagehub.db.bak.20261127"
  ls -l app.py.bak.20261127 index.html.bak.20261127 imagehub.db.bak.20261127
'
```

Expected: 三个备份文件存在。

- [ ] **Step 3: 同步文件到生产**

```bash
sshpass -p "$IMAGEHUB_SSH_PASS" scp /opt/vLLM/ImageHub/app.py \
  /opt/vLLM/ImageHub/index.html \
  /opt/vLLM/ImageHub/deploy-guard.sh \
  root@192.168.0.70:/opt/vLLM/imagehub/
sshpass -p "$IMAGEHUB_SSH_PASS" ssh root@192.168.0.70 '
  cd /opt/vLLM/imagehub && chmod +x deploy-guard.sh && md5sum app.py index.html
'
```

Expected: 生产 `md5sum` 与仓库一致。

- [ ] **Step 4: 重启**

```bash
sshpass -p "$IMAGEHUB_SSH_PASS" ssh root@192.168.0.70 'cd /opt/vLLM/imagehub && ./deploy-guard.sh'
```

Expected: 等待到 busy/active 为 0 后重启；命令输出“已重启 ImageHub”。

- [ ] **Step 5: 生产验收**

```bash
curl -s http://192.168.0.70:5569/api/version
curl -s http://192.168.0.70:5569/api/status
curl -s -o /dev/null -w '%{time_total}s %{size_download}B\n' \
  'http://192.168.0.70:5569/api/history?limit=20&offset=0'
sshpass -p "$IMAGEHUB_SSH_PASS" ssh root@192.168.0.70 '
  cd /opt/vLLM/imagehub
  sqlite3 -readonly imagehub.db "PRAGMA index_list(jobs);"
  journalctl -u imagehub --since "-3min" --no-pager | tail -40
'
```

Expected:
- `/api/version` 为 `1.5.1`；
- `/api/status` 含 `busy:false`（空闲时），模型驻留状态保持；
- `/api/history` 200，`X-Total-Count` 正确，耗时毫秒级；
- 索引存在；日志无 `database is locked`、无缩略图异常；
- 等几秒后 `ls thumbs | grep -c _1024` 大于 0（启动回填或新任务预生成）。

- [ ] **Step 6: 0.7 浏览器验收**

请用户从 `192.168.0.7` 打开图库：

- 点击“图库”列表立即出现；
- 停留 1 分钟不出现每 20 秒的整表闪烁/卡顿；
- 滚动时不跳动；
- 点缩略图先看到 1024 预览，点“查看原图”才加载原图；
- “编辑此图”仍使用原图。

- [ ] **Step 7: 回滚（仅在验收失败时）**

```bash
sshpass -p "$IMAGEHUB_SSH_PASS" ssh root@192.168.0.70 '
  cd /opt/vLLM/imagehub
  cp -a app.py.bak.20261127 app.py
  cp -a index.html.bak.20261127 index.html
  cp -a deploy-guard.sh.bak.20261127 deploy-guard.sh 2>/dev/null || true
  systemctl restart imagehub.service
'
```

回滚后验证 `/api/version` 回到 `1.5.0`，`/api/history` 正常。索引和新增缩略图是无害增量，可保留。

---

## Self-Review

- **Spec coverage:** 设计 §4.1 → Task 1；§4.2 → Task 2；§4.3 → Task 3；§4.4 → Task 4；§4.5 → Task 6；§5 测试 → Task 1~5 + Task 3/4 脚本 + Task 7 验收；§6 风险 → Global Constraints/Task 7；§7 busy/部署 → Task 5/Task 7；§8 顺序 → Task 1~7。
- **Step scan:** 每个测试步骤有断言与运行命令；每个实现步骤给出签名/关键 SQL/算法，不写完整函数体；每个验证步骤有命令与预期。
- **Type consistency:** `_ensure_schema()`、`db(readonly=False)`、`_upsert_job_result(...)`、`_ensure_thumb(...)`、`_prewarm_thumbs(...)`、`_busy()`、`openLightbox(src,title,info,orig)` 在各任务中名称一致。
- **Review Focus:** 满页新增 → Task 3 测试 D；并发请求 → Task 3 测试 E；created/状态 → Task 1 `test_upsert_preserves_created` + `test_interrupted_only_on_schema_init`；只读 WAL → Task 1 `test_readonly_blocks_write`；busy → Task 5 `test_busy_flags`。
- **Proportion:** 计划以测试/接口/命令为主，未复写 `app.py` 的完整实现；实现细节以设计文档和签名约束为准。
