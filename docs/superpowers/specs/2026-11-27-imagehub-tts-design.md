# ImageHub TTS（Qwen3-TTS 1.7B 三变体 + HTTPS）· 设计

> 日期：2026-11-27
> 版本：1.6.0（计划）
> 范围：`app.py`、`index.html`、`run.py`、`deploy-guard.sh`、新增 `tts_service/`、`README.md`、`NOTICE`
> 目标环境：生产 `192.168.0.70:/opt/vLLM/imagehub`

## 1. 背景与目标

在 ImageHub 新增 TTS tab，使用 Qwen3-TTS 1.7B 的三个变体：

- **CustomVoice**：预设音色 + 文本；
- **VoiceDesign**：用文字描述设计音色；
- **Base**：参考音频克隆音色。

官方 `qwen-tts-demo` 一次只能加载一个模型、没有 API、没有登录，满足不了需求。因此：

- 自研一个 **TTS 服务**（FastAPI，内部 `127.0.0.1:5570`），一个进程同时加载三个 1.7B 模型；
- **原生重写 Web UI**，放进现有 `index.html`，不用官方 Gradio；
- 推理仍然用官方 `qwen-tts` Python 包，不重写 tokenizer/codec；
- TTS 与图像模型**不同时加载**，双方忙就 FIFO 排队；
- TTS 输出只在 Web UI/API 返回，**不进 ImageHub 历史、不落库、不做缩略图**；
- ImageHub 从 HTTP 切到 **HTTPS（`https://192.168.0.70:5569`）**，让浏览器麦克风可用于克隆录音。

## 2. 已定决策（用户）

1. 推理用官方包；只重写服务和 Web UI。
2. TTS 1.7B 三个变体同时加载，不与图像模型同时加载。
3. TTS 输出只在 Web UI/API 返回。
4. 图像 API 和 TTS API 互相检查空闲；对方忙就排队。
5. ImageHub 改 HTTPS，`https://192.168.0.70:5569`。
6. 不修改 yz-login。

## 3. 方案对比

### 3.1 Web UI：官方 Gradio vs 原生重写

| 方案 | 优点 | 缺点 |
|---|---|---|
| 挂官方 Gradio | 代码少 | 三套 UI、样式割裂、反代 WebSocket/iframe 麻烦、无法统一 SSO/队列状态 |
| **原生重写（采用）** | 统一风格、单页、可控队列/状态、无 iframe/WS 反代 | 前端工作量较大，录音/上传/播放要自己写 |

原生 UI 采用：三个子 tab（自定义音色 / 声音设计 / 声音克隆），共享文本区和播放器。

### 3.2 TTS 服务形态：一个进程 vs 三个进程

| 方案 | 优点 | 缺点 |
|---|---|---|
| **一个 FastAPI 加载三个模型（采用）** | 一个端口、一份队列/状态、一个进程好起停 | 需要自己写多模型加载和路由 |
| 三个官方 demo 进程 | 改动小 | 三个端口、三套队列/状态、整体释放复杂 |

### 3.3 HTTPS：uvicorn 直接 TLS vs 反向代理

| 方案 | 优点 | 缺点 |
|---|---|---|
| **uvicorn 直接 TLS（采用）** | 不新增组件、单进程模型调度不变 | 5569 只能 HTTPS，旧 HTTP 调用方要改 |
| Caddy/nginx 反代 | 可同时保留 HTTP/HTTPS、证书集中 | 新增组件；nginx 近期 CVE 风险；多一层排障 |
| 另起一个 HTTP 兼容实例 | 旧客户端不动 | 两个 uvicorn 进程状态不共享，会破坏模型调度，**不可行** |

## 4. 设计

### 4.1 HTTPS（5569）

- 生成自签证书，含 `SAN IP:192.168.0.70`，放到 `/etc/imagehub/tls/imagehub.crt|key`，私钥 `600 root`。
- `run.py` 读取 `IMAGEHUB_TLS_CERT` / `IMAGEHUB_TLS_KEY`，存在时给 uvicorn 追加 `--ssl-certfile` / `--ssl-keyfile`；不存在时维持 HTTP。
- `systemd` 的 `imagehub.service` 增加这两个环境变量。
- 浏览器首次访问会提示证书不受信任，点“继续访问”后 `isSecureContext=true`，麦克风可用。
- `SessionMiddleware` 在 TLS 开启时使用 `https_only=True`。
- **SSO 不需要改代码**：
  - yz-login 目前只监听 `http://192.168.0.8:80`（gunicorn），没有 443/8443 证书；
  - 但 yz-login 里 app 125 的 `from` 已经是 `https://192.168.0.70:5569`；
  - 流程：HTTPS 首页 → 302 到 `http://192.168.0.8/login?from=id:125`（顶层跳转，浏览器允许）→ 登录后回跳 `https://192.168.0.70:5569?ticket=...` → ImageHub 服务端仍用 HTTP 调 yz-login 验 ticket；
  - 不改 yz-login 的代价：yz-login 登录页和密码仍是局域网 HTTP 明文（现状不变）。
- 受影响的内部调用要改 HTTPS + `--cacert /etc/imagehub/tls/imagehub.crt`（证书含 `IP:192.168.0.70` 和 `IP:127.0.0.1`，用它做 CA 校验，不全局关闭 TLS 校验）：
  - `deploy-guard.sh` 的 `/api/status`；
  - `tools/full_stress.py`；
  - README 中的 curl 示例。
- 外部 API 调用方必须改 `https://` 并处理自签证书；不提供同端口 HTTP 兼容。

### 4.2 TTS 服务（127.0.0.1:5570）

一个独立的 FastAPI 进程，使用**独立 conda 环境** `qwen-tts`（官方包要求 `transformers==4.57.3`，与 `imagegen` 的 5.23 冲突）：

- 依赖：`qwen-tts==0.1.1`、`torch`（cu12x，匹配 3090）、`torchaudio`、`librosa`、`soundfile`、`sox`、`onnxruntime`、`einops`；不需要 gradio。
- 启动时加载三个模型：
  - `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice` → `cuda:0`
  - `Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign` → `cuda:0`
  - `Qwen/Qwen3-TTS-12Hz-1.7B-Base` → `cuda:1`
  - tokenizer `Qwen/Qwen3-TTS-Tokenizer-12Hz` 预下载；若每个模型各加载一份，按实测调整。
- 只绑 `127.0.0.1:5570`，不对外。
- 提供一个 `--fake` 测试模式，用假模型跑接口/队列测试，不占显存。
- 生成串行（全局并发 1），避免三模型同时打满 2×24GB；API 请求 FIFO，`/api/tts/status` 暴露 `active`、`pending`、`state`。

### 4.3 显存与加载

- 每个 1.7B 权重约 3.9GB，运行显存约 6~8GB；三个合计约 18~24GB，2×24GB 理论足够。
- 放置：GPU0 两个（CustomVoice + VoiceDesign），GPU1 一个（Base）。
- 加载三个模型预计 30~120 秒，取决于磁盘与 HF 缓存；UI 显示“正在加载三个模型”。
- **必须做真实 spike**（见 §5.4）确认显存、加载时间和三路生成；若超限，退化为“常驻两个 + 按需加载第三个”，但优先满足用户“三个同时加载”。

### 4.4 TTS API（内部 5570，由 BFF 转发）

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/health` | 进程/模型加载状态 |
| GET | `/api/status` | `state`、`models`、`active`、`pending` |
| GET | `/api/voices` | CustomVoice 预设音色、支持语言 |
| POST | `/api/tts/speak` | JSON：`variant=custom`，`text`、`language`、`speaker`、`instruct?` |
| POST | `/api/tts/speak` | JSON：`variant=design`，`text`、`language`、`instruct` |
| POST | `/api/tts/clone` | multipart：`text`、`language`、`ref_text?`、`file` |
| POST | `/v1/audio/speech` | OpenAI 兼容，映射到 CustomVoice；`model`、`input`、`voice`、`response_format=wav` |

- 成功返回 `audio/wav`（24kHz 单声道），失败返回 JSON 错误。
- 参考音频：服务端用 ffmpeg 统一转 24kHz 单声道 wav；建议 10~30 秒，限制 60 秒/20MB。
- 不写文件到长期目录；临时文件在 `/tmp`，请求结束即删。

BFF 对外：

- `/api/tts/*` → 反代到 `127.0.0.1:5570`；
- `/v1/audio/speech` → 反代到 TTS 服务，鉴权与 `/v1/images/generations` 一致（API Key 或网页会话）。

### 4.5 原生 TTS Web UI（index.html）

新增导航项「TTS」，页面结构：

1. **状态卡**
   - 图像侧：当前驻留模型（`/api/status`）；
   - TTS 侧：`未加载 / 加载中 / 就绪 / 生成中 / 排队`、GPU 放置、active/pending；
   - 按钮「加载 TTS」「释放 TTS」（手动让出 GPU）。
2. **三个子 tab**
   - 自定义音色：音色下拉（来自 `/api/voices`）、语言下拉、可选语气指令、文本框。
   - 声音设计：声音描述（必填）、语言、文本框。
   - 声音克隆：参考音频上传（拖拽/选择）+ 麦克风录音（HTTPS 可用）、参考文本（可选）、语言、文本框。
3. **生成与输出**
   - 生成按钮；排队/加载/合成状态；
   - 完成后出现播放器、下载按钮（`tts-<时间戳>.wav`）、耗时；
   - 只保留本次结果，刷新即丢；无历史列表。
4. **互斥提示**
   - TTS 就绪时点图像生成：提示“将停止 TTS 并释放三个模型”；
   - 图像生成中点 TTS：提示“等待当前图像任务完成”；
   - 排队中显示位置。

表单状态存 `localStorage`（每个子 tab 的文本、音色、语言、指令），不做音频历史。

录音：`MediaRecorder`（webm/opus）→ multipart 上传 → 服务端 ffmpeg 转 wav；不支持的浏览器隐藏录音按钮，只保留上传。

### 4.6 互斥与排队

资源规则（GPU0/1）：**同一时刻只有图像后端或 TTS 服务之一加载**。

- 图像请求（`run_model`、`_run_comfy`、设计接口）入口：
  - TTS `active>0` → 等待 TTS 空闲（FIFO）；
  - TTS `ready/loading` 且空闲 → 停止整个 TTS 服务，等端口关闭/显存释放；
  - 然后走现有 `acquire_model` / ComfyUI 流程。
- TTS 请求（`/api/tts/*`、`/v1/audio/speech`）入口：
  - 图像 `active>0`（直连或 ComfyUI）→ 等待图像空闲；
  - 图像模型驻留但空闲 → 释放直连服务 / ComfyUI；
  - 然后启动 TTS 服务（三个模型），就绪后生成。
- 双方都在等：按到达顺序 FIFO，用一个共享的切换队列/条件变量裁决。
- 状态展示：`/api/status` 增加 TTS 字段；TTS 页状态卡同时显示两侧。
- `_busy()` 计入 TTS `loading/active/pending`，`deploy-guard.sh` 不会在 TTS 生成时重启。
- BFF 重启：启动时检查 5570 `/health`，健康则接管；不健康时用 `_pid_on_port(5570)` 清理残留进程。

### 4.7 无历史

- TTS 请求不调用 `record_job()`，不写 `jobs` 表，不生成缩略图，不进图库。
- 音频只在响应中返回；服务端临时文件即删。
- 可选：`/var/log/imagehub_tts.log` 只记请求元数据（时间、变体、字符数、耗时、成功/失败），不含音频和全文。

### 4.8 端口与进程

| 组件 | 端口 | 说明 |
|---|---|---|
| ImageHub BFF | 5569 | 改 HTTPS |
| TTS 服务 | 127.0.0.1:5570 | 新增，仅本机 |
| ComfyUI / h3 | 5568 | 不变 |
| 直连服务 | 5564~5567 | 不变 |
| vLLM | 5563 | 不变，GPU2/3 |

TTS 服务由 BFF 以子进程方式启动/停止（沿用 `SERVICES` 模式），使用 `Qwen3TTS` 专用 Python；不新增 systemd 服务。

### 4.9 配置

新增环境变量：

| 变量 | 默认 | 说明 |
|---|---|---|
| `IMAGEHUB_TLS_CERT` | 空 | 证书路径，存在则 HTTPS |
| `IMAGEHUB_TLS_KEY` | 空 | 私钥路径 |
| `IMAGEHUB_TTS_PY` | `/opt/conda3/envs/qwen-tts/bin/python` | TTS 服务解释器 |
| `IMAGEHUB_TTS_PORT` | `5570` | 内部端口 |
| `IMAGEHUB_TTS_CUSTOM` | `/opt/models/Qwen3-TTS/CustomVoice` | CustomVoice 权重目录 |
| `IMAGEHUB_TTS_DESIGN` | `/opt/models/Qwen3-TTS/VoiceDesign` | VoiceDesign 权重目录 |
| `IMAGEHUB_TTS_CLONE` | `/opt/models/Qwen3-TTS/Base` | Base 权重目录 |
| `IMAGEHUB_TTS_MAX_UPLOAD` | `20971520` | 参考音频字节上限（BFF 与 TTS 服务共用） |
| `IMAGEHUB_TTS_MAX_SECONDS` | `60` | 参考音频时长上限（TTS 服务） |

## 5. 测试与验收

### 5.1 TTS 服务单元测试

- 假模型（`--fake`）下：`/health`、`/api/status`、`/api/voices`；
- 三个变体路由参数校验（缺 text/instruct/ref_audio 返回 400）；
- 队列：并发两个请求，第二个 `pending=1`，串行完成；
- 参考音频转码（ffmpeg）与超长/超大拒绝；
- `/v1/audio/speech` 映射；错误返回结构。

### 5.2 BFF 单元测试

- 互斥：TTS 就绪时图像请求会先停 TTS；图像驻留时空闲，TTS 请求会先释放图像；
- 双方 active 时排队，状态字段正确；
- `_busy()` 覆盖 TTS 状态；`deploy-guard.sh` 不误重启（HTTPS + `--cacert`）；
- `/api/tts/*`、`/v1/audio/speech` 鉴权与转发；
- TTS 请求不写 `jobs`（断言行数不变）。

### 5.3 前端验收（Playwright，mock API）

- TTS tab 三个子 tab 切换、表单状态保存；
- 音色下拉加载、生成后播放器/下载出现；
- 加载中/排队中/等待图像任务三种状态文案；
- 录音按钮在不支持 `mediaDevices` 时隐藏；
- 图像生成与 TTS 互斥提示；
- 不上传音频、不产生历史请求。

### 5.4 真实三模型 spike（部署前）

- 在 GPU0/1 空闲时加载三个 1.7B，记录每个模型的显存、总显存、加载耗时；
- 三个变体各生成一段中文/英文音频，记录耗时和音质；
- 确认 GPU0 两模型 + GPU1 一模型稳定；
- 验收：总显存 < 40GB（留余量），三路生成成功。若失败，按 §4.3 降级并回报。

### 5.5 HTTPS 验收

- `curl --cacert /etc/imagehub/tls/imagehub.crt https://192.168.0.70:5569/api/status` 200；
- 浏览器首次证书提示 → 继续 → 登录 SSO 成功；
- 麦克风权限可弹出并录音；
- `deploy-guard.sh`、`tools/full_stress.py` 改 HTTPS 后可用；
- 回滚：去掉 TLS 环境变量、恢复 `run.py`/`app.py`/`index.html`，重启即回 HTTP。

## 6. 风险

| 风险 | 缓解 |
|---|---|
| 三个 1.7B 超 48GB 或长文本 activation OOM | 先 spike；降级为两个常驻 + 一个按需；限制单次文本长度 |
| 三模型加载 30~120s，用户体验差 | UI 进度/状态；提示“图像任务后需重新加载” |
| TTS 与图像切换频繁，反复加载 | 队列合并；提示切换成本；保留手动「释放 TTS」 |
| 自签证书警告/API 客户端失败 | 文档说明；内部脚本用 `--cacert` 指定证书；外部客户端导入证书；可选内网 CA |
| yz-login 仍是 HTTP 登录页 | 不改 yz-login；记录为已知限制 |
| 独立 conda 环境依赖冲突 | 新环境锁定 `transformers==4.57.3`；不装 gradio |
| transformers/torch 版本与 3090 不兼容 | spike 先验证；必要时固定 torch cu12x 版本 |
| 参考音频格式杂 | 服务端 ffmpeg 统一转码 + 时长/大小限制 |
| TTS 录音在非 HTTPS 下不可用 | 已切 HTTPS；不支持的浏览器退化为上传 |

## 7. 实施顺序

1. **三模型 spike**：新 conda 环境、下载三模型 + tokenizer、单 FastAPI 加载三模型、测显存/耗时/三路生成。
2. **TTS 服务**：路由、队列、状态、参考音频转码、`--fake` 测试模式 + 单元测试。
3. **BFF 集成**：TTS 子进程起停、`/api/tts/*` 与 `/v1/audio/speech` 转发、互斥/队列、`/api/status` 字段、busy、测试。
4. **原生 TTS UI**：导航、状态卡、三子 tab、上传/录音、播放/下载、互斥提示 + Playwright。
5. **HTTPS**：自签证书、`run.py` TLS、systemd 环境变量、cookie、内部脚本改 https、SSO 验证。
6. **版本/文档/NOTICE**：1.6.0、README、NOTICE。
7. **生产部署**：备份、部署、重启窗口、验收、回滚预案。
