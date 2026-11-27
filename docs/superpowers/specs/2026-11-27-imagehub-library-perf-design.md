# ImageHub 图库列表性能优化 · 设计

> 日期：2026-11-27
> 版本：1.5.1（计划）
> 范围：`app.py`、`index.html`、`README.md`、`NOTICE`
> 目标环境：生产 `192.168.0.70:/opt/vLLM/imagehub`，开发仓库 `/opt/vLLM/ImageHub`

## 1. 背景与问题

用户从 `192.168.0.7` 的浏览器点击「图库」时，历史列表明显卡顿。

已完成的只读排查结论：

- 点击图库时服务端 `/api/history` 约 **1.7 ms**，响应约 **6.9 KB**（12 条历史）。
- 同一时间段内 **没有任何 `/thumb` 请求**，说明缩略图全部命中浏览器缓存。
- `192.168.0.7 → 192.168.0.70:5569` 的 HTTP TCP RTT 约 **0.3 ms**，无重传。
- 0.7 在 18:48~19:10 之间发生 **11 次整页刷新**。
- 0.7 上还有两条从 70 下载大文件的 SSH 会话（约 100 MB / 196 MB，RTT 40~50 ms，有重传）；0.7 的 5900 端口开放（VNC）。

结论：**慢点不在 FastAPI、SQLite 查询或缩略图下载，而在浏览器端列表重渲染与客户端显示环境。** 代码里会放大这个问题的点：

1. 每 20 秒 `loadHistory()` 用 `innerHTML` 全量重建整张列表（`index.html:1468`、`index.html:683`）。
2. 页面启动时一次 `loadHistory()`，首次点击图库又调用一次（`index.html:1461`、`index.html:1440`）。
3. 每行 4 张图，一页最多 20 行 = 80 个 `<img>`；重建时全部销毁重插。
4. 历史灯箱直接加载 1~3 MB 原图（`index.html:709`、`index.html:613`）。
5. 后端 `db()` 每次调用都执行建表、`PRAGMA`、失败的 `ALTER TABLE` 和 `UPDATE interrupted`（`app.py:230-273`），历史读请求实际是写请求。
6. `jobs.created` 无索引，查询计划为 `SCAN jobs` + `USE TEMP B-TREE FOR ORDER BY`。

## 2. 目标与非目标

### 目标

- 点击「图库」后列表立即显示；停留/滚动时不再每 20 秒卡一下。
- 数据未变化时轮询不产生 DOM 重建；新增任务时只插入新行，保留已有 DOM 和滚动位置。
- 不在非图库页轮询历史，不在首次点击时重复加载。
- 新生成任务的 160/480/1024 缩略图在后台预生成，首次查看不等待 PIL 解码。
- 历史灯箱先加载 1024px 缩略图，原图按需加载；编辑仍使用原图。
- 历史查询不再写库；建表/迁移只执行一次；`jobs.created` 有索引；启动时只标记一次中断任务。
- 保持现有 API 路径、响应结构、缩略图 URL 和页面交互不变。

### 非目标

- 不引入 nginx/HTTP/2 反向代理（当前生产没有运行 nginx，且近期 nginx 有未修复风险）。
- 不改成多 worker uvicorn（模型调度状态在进程内）。
- 不重写前端框架、不拆单文件页面。
- 不改变模型生成/调度/后端切换语义。
- 不引入新的运行时 Python 依赖（缩略图继续用 Pillow；后台队列用标准库）。

### 反目标

- 为了“看起来快”而减少历史条数或隐藏图片。
- 用 base64 批量缩略图把 `/api/history` 响应从 7 KB 膨胀到数百 KB。
- 用前端假数据掩盖真实的慢查询或后台锁。

## 3. 方案对比

### 方案 A：只改前端

- 单飞请求、签名对比、增量插入、仅图库页轮询、去掉首次双加载。
- 优点：零重启、风险最低、直接解决 0.7 的卡顿。
- 缺点：不解决冷缓存缩略图、DB 读时写、无索引、灯箱原图。

### 方案 B：前端 + 后端低风险加固

- 方案 A + `_ensure_schema()` 一次性建表/迁移、`db(readonly=True)`、`jobs(created)` 索引、`interrupted` 仅启动时执行。
- 优点：解决潜在的锁竞争和未来历史增长问题。
- 缺点：需要重启一次 `imagehub.service`。

### 方案 C：完整优化（本设计采用）

- 方案 B + 缩略图预生成队列 + 启动回填 + 灯箱先 480 后原图。
- 优点：覆盖“冷缓存首次打开”和“点大图慢”；不改变 API。
- 缺点：改动面最大，需要重启、需要浏览器验收。

**不采用批量缩略图接口**：实测 0.7 点击图库时缩略图请求为 0，瓶颈是 DOM 重建/显示，不是请求数量。批量 base64 会增加响应体积和内存，对 VNC 显示也没有帮助。预生成 + 增量 DOM 更直接。

## 4. 设计细节

### 4.1 后端：SQLite 连接与迁移

新增模块级状态：

```python
_schema_lock = threading.Lock()
_schema_ready = False

def _ensure_schema():
    """建表、WAL、迁移、索引、启动中断标记，只执行一次。"""
```

- `_ensure_schema()` 内部完成：
  - 现有 7 张表的 `CREATE TABLE IF NOT EXISTS`。
  - `PRAGMA journal_mode=WAL`。
  - 现有 3 个 `ALTER TABLE` 迁移（仍用 try/except 吞掉重复列）。
  - `CREATE INDEX IF NOT EXISTS idx_jobs_created ON jobs(created DESC)`。
  - `CREATE INDEX IF NOT EXISTS idx_jobs_project_created ON jobs(project, created DESC)`。
  - `UPDATE jobs SET status='interrupted' WHERE status='running'`（仅此处执行）。
- `db()` 改造：

```python
def db(readonly=False):
    _ensure_schema()
    if readonly:
        conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True, timeout=5)
    else:
        conn = sqlite3.connect(DB_PATH, timeout=5)
    conn.execute("PRAGMA busy_timeout=5000")
    return conn
```

- 新增 `@app.on_event("startup")`：最先调用 `_ensure_schema()`，保证服务开始处理请求前 schema 就绪。
- 只读端点改为 `db(readonly=True)`：
  - `api_history`
  - `api_job` 的 DB 兜底
  - `api_projects`
  - `api_project_assets`
  - `admin_users`、`admin_keys`、`admin_audit`、`admin_presets`
  - `api_design_presets`、`admin_presets` 读取
- 写端点继续使用 `db()`：`record_job`、`run_job`、`_audit`、`_upsert_user`、`_api_key_user`/`_key_valid`（更新 `last_used`）、历史删除、预设增删、异步队列更新。
- 为测试与隔离，新增环境变量：
  - `IMAGEHUB_DB` 覆盖 `DB_PATH`（默认仍是仓库下 `imagehub.db`）。
  - `IMAGEHUB_THUMBS` 覆盖 `THUMB_DIR`。
  - `IMAGEHUB_RUN` 覆盖 `RUN_DIR`（避免测试在源码目录创建 `run/`）。
- 验证：`EXPLAIN QUERY PLAN SELECT ... ORDER BY created DESC LIMIT 20` 应出现 `USING INDEX idx_jobs_created`。

#### 4.1.1 `run_job` 保留首次 `created`

`/api/compare` 的 `run_job()` 当前每完成一个模型就用 `INSERT OR REPLACE` 重写整行，并把 `created` 设为当前时间，导致运行中的对比任务在历史里反复跳到最前。这会持续破坏列表顺序，使增量渲染退化为全量重建。

改为 UPSERT，保留首次创建时间：

```sql
INSERT INTO jobs(id,prompt,models,created,results,status)
VALUES(?,?,?,?,?,'running')
ON CONFLICT(id) DO UPDATE SET results=excluded.results, status=excluded.status
```

- 首次插入使用 `_jobs[job_id]["created"]`（`api_compare` 创建任务时的时间）。
- 后续模型完成只更新 `results`/`status`，`created` 不变。
- 最终 done 同样使用 UPSERT。
- 该修复属于本次范围：它直接决定“只插新行”能否稳定成立。

### 4.2 后端：缩略图预生成

把现有 `thumb()` 中的逻辑抽成可复用函数：

```python
def _thumb_path(filename, w) -> str
def _find_source(filename) -> Optional[str]   # OUTPUT_DIR + LEGACY_DIRS
def _ensure_thumb(filename, w) -> Optional[str]
```

- `_ensure_thumb`：
  - 缓存命中直接返回。
  - 未命中时用 PIL 解码并缩放到 `(w, w*8)`，保存为 JPEG quality 82。
  - 写临时文件（唯一后缀）后 `os.replace` 到目标，避免并发/中断产生半张图。
  - 源文件不存在返回 `None`，由调用方决定 404 或跳过。
- `thumb()` 端点改为调用 `_ensure_thumb`，保留现有 `Cache-Control`、`FileResponse` 行为。
- 后台预生成：
  - 模块级 `queue.Queue(maxsize=2000)` + 2 个 daemon worker 线程。
  - `_prewarm_thumbs(results)` 遍历 `status == "ok"` 且有 `image` 的结果，投递 160、480、1024 三个宽度。
  - 队列满时 `put_nowait` 捕获 `Full` 并记录 warning，绝不阻塞生成主流程。
- 接入点：
  - `record_job()` 成功写入后：覆盖 `/v1/images/generations`、`/api/edit`、设计类 `_design_response`。
  - `run_job()` 全部模型结束后：覆盖 `/api/compare`。
  - `_dq_worker()` 异步任务更新为 done 后。
  - 启动后台回填：读取最近 200 条 `jobs.results`，对缺失缩略图投递 160/480/1024。仅处理 `status == "ok"` 的图片。
- 不删除现有缩略图缓存；预生成只补充缺失文件。

### 4.3 前端：历史列表增量更新

新增模块级状态：

```js
let CURRENT_PAGE = "work";
let HIST_SIG = "";
let HIST_ABORT = null;
```

`switchPage(name)`：

- 设置 `CURRENT_PAGE = name`。
- `name === "library"` 时调用 `loadHistory()`；不再用 `_pageLoaded` 阻止图库刷新。
- 其他页面保持现有 `_pageLoaded` 懒加载逻辑。
- 删除页面底部无条件的 `loadHistory();`（`index.html:1461`）。

`loadHistory(page)`：

- **最新请求优先**：每次调用先 `HIST_ABORT?.abort()`，再创建新的 `AbortController`；`fetch(..., {cache: "no-store", signal})`。
- 被 abort 的旧请求捕获 `AbortError` 后直接返回，不更新分页、不渲染；只有当前 controller 的响应可以落地。
- 先更新 `HIST_PAGE`，再发请求；响应后读取 `X-Total-Count`，更新 `HIST`（供复制提示词）和分页/总数。
- 用 `const sig = JSON.stringify(rows)` 作为数据签名（包含 prompt、models、status、images）。
- `sig === HIST_SIG`：不碰 `historyList`，直接返回。
- `sig !== HIST_SIG`：调用 `renderHistoryRows(rows)`。
- `renderHistoryRows(rows)`：
  - `rows` 为空时显示“暂无记录”，清空 `HIST_SIG` 后返回。
  - 把历史项构建抽成 `buildHistoryItem(h)`，输出带 `data-id`、`data-sig` 的 `.histitem`；`buildHistoryItem` 内对 prompt、meta、title 使用 `esc()` 转义，避免提示词里的 `<`/`&` 破坏 DOM。
  - **通用增量 diff**（不依赖“新序列是旧序列后缀”）：
    1. 记录 `box.scrollTop` 与 `box.scrollHeight`，把现有 `.histitem` 按 `data-id` 放进 `Map`。
    2. 按新顺序逐条处理：签名相同则复用原节点；签名不同则 `replaceWith` 新节点；不存在则创建。处理过的从 `Map` 删除。
    3. `Map` 剩余节点从 DOM 移除。
    4. 所有节点按新顺序放入 `DocumentFragment`，再一次性 `box.replaceChildren(fragment)`；移动的是已有节点，图片不会重新下载。
    5. 恢复滚动：记录首个可见 `.histitem` 的 `data-id` 及其相对滚动容器顶部的偏移；重建后找到同一节点，按其新偏移恢复 `scrollTop`；找不到锚点时才回退到高度差；在顶部则保持 0。
  - 满页（20 条）新增任务时，尾部被移除、顶部插入新节点；未变化节点保持原对象，不触发全量重建。
- 移除 `box.querySelectorAll(".histitem")` + `offsetHeight` 的测量逻辑（`index.html:727-734`）；高度交给 `.hist { max-height: 540px; overflow-y: auto; }`。
- 列表图片加 `decoding="async"`，保留 `loading="lazy"`。
- 轮询抽成具名函数，便于测试：

```js
function pollHistory(){
  if(CURRENT_PAGE === "library" && HIST_PAGE === 0) loadHistory();
}
pollWhenVisible(pollHistory, 20000);
```

- 生成完成后的 `loadHistory()` 调用保留；若不在图库页，签名未变时不会重建 DOM；若变了，只更新隐藏列表，开销很小。

### 4.4 前端：灯箱按需加载原图

HTML 增加按钮：

```html
<button class="mini" id="lbOrig" style="font-size:11px;display:none">查看原图</button>
```

JS：

```js
function openLightbox(src, title, info, orig) {
  lbCurrent = orig || src;   // 编辑始终使用原图
  lbImg.src = src;           // 默认显示 1024 预览图
  lbOrigUrl = orig && orig !== src ? orig : "";
  document.getElementById("lbOrig").style.display = lbOrigUrl ? "" : "none";
  ...
}
document.getElementById("lbOrig").onclick = () => {
  if (!lbOrigUrl) return;
  lbImg.src = lbOrigUrl;
  lbCurrent = lbOrigUrl;
  lbOrigUrl = "";
  document.getElementById("lbOrig").style.display = "none";
};
```

调用点：

- `render(job)` 结果网格：`openLightbox('/thumb/...?w=1024', ..., '/image/...')`。
- 历史列表：`openLightbox('/thumb/...?w=1024', ..., '/image/...')`。
- 编辑结果：同样 1024 + 原图。
- `lbCurrent` 保持原图，`lbEdit` 不会拿到低分辨率缩略图。

### 4.5 版本与文档

- `APP_VERSION`：`1.5.0` → `1.5.1`。
- `README.md`：更新版本徽章/迭代说明，新增 1.5.1 图库性能条目。
- `NOTICE`：仓库目前没有该文件。按全局约定新增一条本次变更记录（若不希望创建，部署前告知，改为只更新 README）。

## 5. 测试与验收

### 后端单元测试（标准库 `unittest`）

新增 `tests/test_library_perf.py`：

- 使用 `IMAGEHUB_DB`/`IMAGEHUB_THUMBS`/`IMAGEHUB_RUN` 指向临时目录，导入/重载 `app`。
- `_ensure_schema()`：
  - 7 张表存在；
  - `PRAGMA index_list(jobs)` 包含 `idx_jobs_created`、`idx_jobs_project_created`；
  - 查询计划包含 `USING INDEX idx_jobs_created`。
- 中断标记：
  - 先插入 `status='running'`，调用 `_ensure_schema()` 后变 `interrupted`；
  - 再插入 `status='running'`，调用 `db()`（读写）后仍为 `running`，证明不再每次连接都写。
- 只读连接：
  - `db(readonly=True)` 执行 `INSERT` 抛出 `sqlite3.OperationalError`。
- `run_job` 保留 `created`：
  - 插入一条 `status='running'` 的 compare 任务，记录 `created`；
  - 模拟两次模型完成后的 UPSERT，最后 done；
  - 断言 `created` 始终等于首次值，且历史排序不变。
- `busy` 判定：
  - `/api/status` 返回 `busy` 字段；空闲时 false，设计锁/异步队列/直连 active/内存 running 任务任一存在时 true。
  - `deploy-guard.sh` 优先读 `busy`，连续两次为 0 才重启。
- 缩略图：
  - 临时生成一张 PNG，`_ensure_thumb(name,160)` 生成有效 JPEG；
  - 第二次调用不改变 mtime；
  - `_prewarm_thumbs` 投递 160/480/1024，`_thumb_q.join()` 后三个文件都存在。
- 所有测试用 `python -m unittest discover -s tests -v` 执行。

### 前端验收（开发机 Playwright，非运行时依赖）

新增 `tools/verify_library_perf.py`（仅开发验收，不加入 `requirements.txt`）：

- 本地 mock `/api/history`、`/thumb`，加载 `index.html`。
- 断言：
  - 启动后不自动请求历史；首次点击图库只发起 **1 次** `/api/history`。
  - 切到工作台后调用 `pollHistory()` 不请求历史。
  - 相同数据再次 `loadHistory()` 不替换 `.histitem` 节点（用 DOM 节点标识验证）。
  - 新增一条历史时，已有节点不被销毁，新节点出现在顶部，`scrollTop` 按增量补偿。
  - 满页 20 条时新增一条（尾部移除、顶部插入）：未变化的 19 个节点保持原对象。
  - 并发请求：先发出的慢请求被后发出的请求 abort，最终只渲染最新响应，分页状态一致。
  - 灯箱打开时先请求 `?w=1024`，点「查看原图」后才请求 `/image/...`。
  - 80 张图的模拟页在可接受时间内完成（记录数据，不设硬性门槛）。
- 脚本输出每项 PASS/FAIL 和耗时。

### 生产验收

- 备份：`app.py`、`index.html`、`imagehub.db`（`.backup`）。
- 部署后：
  - `GET /api/version` 返回 `1.5.1`；
  - `GET /api/history?limit=20` 仍为 200，`X-Total-Count` 正确；
  - `EXPLAIN QUERY PLAN` 使用索引；
  - `/api/status` 正常，模型驻留状态保持；
  - 日志无 `database is locked`、无缩略图异常；
  - 用 0.7 浏览器点图库，列表立即出现，停留 1 分钟不再周期性卡顿；
  - 点缩略图，灯箱先显示 1024px，点「查看原图」再加载原图。
- 回滚：恢复备份的 `app.py`/`index.html`，重启 `imagehub.service`；新增索引和缩略图是无害的增量，可保留。

## 6. 风险与缓解

| 风险 | 缓解 |
|---|---|
| 重启打断正在生成的任务 | `/api/status` 增加 `busy`，`deploy-guard.sh` 连续两次 busy=0 再重启；重启前再次向用户确认窗口 |
| DB 迁移/索引阻塞 | 12 条数据，毫秒级；备份 DB；索引为 `IF NOT EXISTS` |
| 预生成抢占 CPU | 2 个 daemon worker、有界队列、仅处理缺失缩略图、失败只告警 |
| 增量 DOM 逻辑引入错位 | 通用 diff 保留未变化节点；满页新增/删除/状态变化均覆盖；Playwright 验收 |
| 灯箱改用 1024 后用户看不到细节 | 保留「查看原图」按钮；编辑始终用原图 |
| 0.7 的 VNC/大流量 SSH 仍然导致显示慢 | 本次只能减少重绘和请求；VNC/带宽问题需环境侧处理，验收时说明 |
| 无自动化测试基线 | 新增后端 unittest + 前端 Playwright 验收脚本，先让测试失败再实现 |

## 7. 部署与已定决策

- 代码先改开发仓库 `/opt/vLLM/ImageHub`。
- 生产部署需要重启 `imagehub.service`，会打断进行中的生成任务。
- **部署方式：A（已确认）**：由我同步到生产 `/opt/vLLM/imagehub` 并重启；重启前必须再次向用户确认窗口，部署前备份 `app.py`/`index.html`/`imagehub.db`。
- **忙碌判定：A（已确认）**：本次一并增加 `busy` 判定并更新 `deploy-guard.sh`。
  - `/api/status` 增加 `busy`：
    - `_state["active"] > 0`（直连模型生成）；
    - `_comfy_lock.locked()`（ComfyUI 设计/对比任务）；
    - `_dq["running"]` 或 `_dq["queue"]` 非空（异步队列）；
    - `_jobs` 中存在 `status == "running"` 的对比任务。
  - `deploy-guard.sh` 改为优先读 `busy`，没有该字段时回退 `active`；连续两次采样为 0 才重启，避免刚好在任务开始/结束的窗口。
- 已定（实现者负责）：`NOTICE` 新增一条变更记录；测试目录 `tests/`；Playwright 脚本放 `tools/`；新增 `IMAGEHUB_DB`/`IMAGEHUB_THUMBS`/`IMAGEHUB_RUN` 环境变量。

## 8. 实施顺序（供计划阶段拆分）

1. 后端 schema/只读连接/索引 + `run_job` 保留 `created` + 单元测试。
2. 缩略图 helper + 预生成队列 + 启动回填 + 单元测试。
3. 前端历史签名/增量渲染/最新请求优先/轮询条件/去双加载。
4. 灯箱 1024 + 原图按钮。
5. `busy` 判定 + `deploy-guard.sh`（已确认纳入）。
6. 版本、README、NOTICE。
7. 开发机 Playwright 验收。
8. 生产备份、部署、重启、验收、回滚预案。
