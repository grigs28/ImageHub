# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概览

ImageHub（v1.7.3）是一个单进程 FastAPI BFF + 单文件前端的统一文生图/图片编辑/TTS 平台：

- 对外提供 OpenAI 兼容接口（`POST /v1/images/generations`、`POST /v1/audio/speech`）与设计类接口（`/v1/design/*`）
- 内部统一调度 4 个基础图像模型（`h3` / `qwen` / `zimage` / `moss`）和图片编辑模型（`edit`）
- 两条执行后端：**直连 diffusers 服务**（`services/`，适合文生图/编辑）和 **ComfyUI**（`/opt/ComfyUI`，设计类 ControlNet 唯一路径）
- **TTS**：`tts_service/` 独立 FastAPI（127.0.0.1:5570），同时加载 Qwen3-TTS 1.7B 的 CustomVoice / VoiceDesign / Base 三个变体；原生页面，不使用 Gradio；与图像模型严格互斥、FIFO 排队
- 附带网页工作台、图库/历史、API Key、SSO、管理后台、GPU 功耗控制；5569 默认 HTTPS（自签证书）

代码与文档均为中文。改动功能后需同步更新 README/`docs/` 中对应说明。

## 运行与验证

### 环境

- 目标运行环境是 GPU 服务器（文档中的 `192.168.0.70:5569`）；本仓库当前签出位置可能没有 GPU/模型/ComfyUI，不要假设服务在本机可跑
- Python 解释器：`/opt/conda3/envs/imagegen/bin/python`（torch/diffusers/PIL 等重型依赖在此环境，`requirements.txt` 只列出 BFF 依赖）
- 关键环境：`/opt/models/`（权重）、`/opt/ComfyUI`（:5568）、vLLM 3.6（:5563）、SQLite `imagehub.db`

### 启动

```bash
pip install -r requirements.txt

# 手动启动（run.py 会调用 uvicorn 并注册 aiops 数据库）
python run.py

# 或直接启动 BFF
/opt/conda3/envs/imagegen/bin/python -m uvicorn app:app --host 0.0.0.0 --port 5569
```

systemd 部署：

```bash
systemctl restart imagehub          # BFF :5569
./deploy-guard.sh                   # 先等生成任务结束后再重启，避免打断任务
systemctl status comfyui-h3         # ComfyUI 由 BFF 在需要时 systemctl start
```

### 验证（仓库无自动化测试）

```bash
python -m py_compile app.py services/*.py tools/*.py   # 最低限度语法检查
curl -s --cacert /etc/imagehub/tls/imagehub.crt https://127.0.0.1:5569/api/status               # 驻留模型/并发/切换状态
curl -s --cacert /etc/imagehub/tls/imagehub.crt https://127.0.0.1:5569/api/models               # 模型就绪状态

# 真实生成（外部调用需 API Key）
curl -s --cacert /etc/imagehub/tls/imagehub.crt -X POST https://127.0.0.1:5569/v1/images/generations \
  -H 'Content-Type: application/json' \
  -d '{"apikey":"ih-...","model":"zimage","prompt":"测试","size":"1024x1024","response_format":"url"}'

python tools/full_stress.py ih-...                     # 全功能组合压测，日志写入 /var/log/full_stress.log
```

日志：`/var/log/imagehub_*.log`（每个直连服务）、`journalctl -u imagehub`、`journalctl -u comfyui-h3`。

## 架构

### 代码结构

- `app.py`：约 2400 行的单体 BFF，包含模型/服务注册表、调度、直连服务生命周期、ComfyUI 工作流、全部路由、鉴权、SQLite、设计接口、异步队列、压力测试
- `index.html`：无构建步骤的单页前端，由 `GET /` 直接读取并注入用户信息；页面分「编辑比较 / 设计 / 图库 / API / 管理」，状态存 `localStorage`
- `services/*.py`：每个模型的独立 OpenAI 兼容 FastAPI 服务，由 `app.py` 以子进程方式启动/停止
- `run.py`：uvicorn 包装器 + aiops 进程注册
- `tools/full_stress.py`：真实接口组合压测脚本

### 进程与端口

| 组件 | 端口 | 说明 |
|---|---|---|
| ImageHub BFF | 5569 | 唯一入口（默认 HTTPS，自签证书） |
| TTS 服务 | 127.0.0.1:5570 | Qwen3-TTS 1.7B × 3，仅本机，由 BFF 起停 |
| vLLM Qwen3.6 | 5563 | 对话服务，独立于图像调度 |
| ComfyUI | 5568 | 设计/ControlNet 后端 |
| `qwen` 直连 | 5566 | diffusers Qwen-Image-2.1-Turbo |
| `zimage` 直连 | 5565 | diffusers Z-Image-Turbo |
| `moss` 直连 | 5567 | diffusers MOSS-Image-SFT |
| `edit` 直连 | 5564 | Qwen-Image-Edit-2511 |
| `h3` 直连 | 5568 | sd.cpp `sd-server`，**与 ComfyUI 同端口，天然互斥** |

`SERVICES` 字典是直连服务的唯一事实来源（端口、cwd、启动命令、`CUDA_VISIBLE_DEVICES`、权重 `needs` 路径）；`MODELS` 是 API/前端可见的模型注册表。`flux2klein`（FLUX.2 Klein 4B）和 `nemotron`（Nemotron-30B-A3B）是 **ComfyUI-only**：`comfy_only: True`，`needs` 直接列权重路径，`WF_BUILDERS`/`MODEL_WORKFLOWS` 对应 `wf_flux2klein`/`wf_nemotron`；`_run_model_impl` 会强制 `backend=comfy`。

### 调度模型

- GPU 显存只能驻留一个直连服务。`acquire_model()` / `release_model()` 配合 `_state` + `_cond` 实现：同模型并发并行、跨模型排队、切换 = 停掉其它服务再启动目标服务
- 当前驻留模型持久化到 `state.json`，BFF 重启时由 `_adopt_running_services()` 接管已在跑的服务
- 启动后 `_preload_default()` 默认预热 `qwen`（`IMAGEHUB_PRELOAD=0` 可关闭），并等待 3.6 空闲
- 可选门控 `settings.json` 的 `gate`：切换前读 vLLM 指标，等 `vllm:num_requests_running` 归零，避免加载瞬间整机风险
- **模型加载/切换是已知最高风险窗口**（大块显存分配/释放），前端在需要切换时弹确认，API 不弹
- **图像 ↔ TTS 互斥**：`_arbiter_acquire/_arbiter_release` 用 FIFO 票号串行化切换。图像请求入口 `_tts_release_for_image()` 会先停整个 TTS 服务；TTS 请求入口 `_tts_begin_request()` 会先等图像空闲并停直连服务/ComfyUI。`run_model`、`api_edit`、`_design_generate` 都经 `_image_session()`；TTS 生成经 `_tts_generation()`。`/api/status` 的 `busy` 计入 TTS loading/active/pending

### 双后端与请求路径

`_req_backend()` 的优先级：请求参数 `backend`（旧参数 `com`）→ `settings.json` 的 `backend` → 默认 `direct`。

- **direct**：`run_model()` → `acquire_model()` → HTTP 调 `127.0.0.1:<port>/v1/images/generations`（`h3` 走 sd.cpp 的 `/sdcpp/v1/img_gen` + 轮询）→ 图片写入 `OUTPUT_DIR`
- **comfy**：`_run_comfy()` → 停掉全部直连服务 → `ensure_comfy()`（必要时 `systemctl start comfyui-h3`）→ `WF_BUILDERS[model]` 生成节点图 → `POST /prompt` → 轮询 `/history` → 返回 ComfyUI 输出文件名
- `_comfy_lock` 串行化所有 ComfyUI 请求；`_comfy_prepare()` 只在切换底座模型时 `/free` 卸载
- `MODEL_WORKFLOWS` 允许同一模型按功能选工作流（如 `qwen: t2i / controlnet`）
- **设计类接口一律走 ComfyUI**：`_design_generate()` 会先停直连服务、再确保 ComfyUI 就绪。diffusers 没有 Qwen-Image-2.1 ControlNet 实现，不要再规划直连 ControlNet

### 数据与文件

`db()` 在每次连接时建表并做增量迁移（`ALTER TABLE` 包在 try/except 中），同时把遗留 `running` 任务标记为 `interrupted`。表：`jobs`、`users`、`api_keys`、`audit`、`presets`、`projects`、`assets`，SQLite WAL 模式。

- 生成图：`OUTPUT_DIR`，默认 `/opt/vLLM/imagehub/output`（环境变量 `IMAGEHUB_OUTPUT`）；`/image` 还会回退查找 `/opt/ComfyUI/output`
- 缩略图：`thumbs/` 下按 `文件名_宽度.jpg` 缓存，`GET /thumb/{file}?w=N`
- 历史：`GET /api/history?limit=&offset=`，总数字段在响应头 `X-Total-Count`
- 单模型生成用 `record_job()` 落库；多模型对比用 `run_job()` 逐模型落盘（`_jobs` 内存态 + SQLite）
- **TTS 输出不落 `jobs`**：`/api/tts/*` 与 `/v1/audio/speech` 直接返回 WAV，同时写入 TTS 专用语音库（`tts_audio` 表 + `output/tts/`，`IMAGEHUB_TTS_OUT` 可覆盖），不生成缩略图、不进图库；语音记录 API：`GET/DELETE /api/tts/history`、`POST /api/tts/history/batch-delete`、`GET /api/tts/audio/{file}`。TTS 服务使用独立 `qwen-tts` conda 环境（`transformers==4.57.3`），参考音频经 ffmpeg 转 24kHz 单声道，限制 60s/20MB
- **新增 `jobs` 列时必须同时更新所有 `INSERT` 语句**（历史上曾因加列后 SQL 未同步导致对比任务不写历史）

### 鉴权与权限

- SSO：`/` 或 `/callback` 收到 `ticket` → 调 yz-login 验证 → 同步到本地 `users` 表；`is_admin` **以本地库为准**，新用户按 `IMAGEHUB_ADMINS`（默认 `grigs`）初始化
- 页面入口未登录会跳转 SSO；`/api/*`、`/image/*` 不做登录拦截
- `/v1/*` 生成/设计接口要求 API Key（请求体 `apikey` 或 `Authorization: Bearer`）或网页会话；Key 只存 SHA-256，`key_plain` 仅供管理员查看
- 管理接口（`/api/admin/*`、设置、功耗、压测）要求管理员：会话 `is_admin`、`role=admin` 的 Key，或 `X-Admin-Token` 匹配 `IMAGEHUB_ADMIN_TOKEN`
- 注意现状不一致：`/api/compare`、`/api/edit`、`/api/history` 目前未做 Key 校验，改动鉴权时不要只改 `/v1/*`

## 配置

`.env`（从 `.env.example` 复制，勿提交）：

| 变量 | 说明 |
|---|---|
| `YZ_LOGIN_URL` / `YZ_APP_ID` / `YZ_APP_SECRET` | yz-login SSO；`YZ_APP_SECRET` 非空才启用 SSO |
| `SESSION_SECRET` | 会话签名密钥 |

常用运行时环境变量：`IMAGEHUB_OUTPUT`、`IMAGEHUB_PY`、`IMAGEHUB_ADMINS`、`IMAGEHUB_ADMIN_TOKEN`、`IMAGEHUB_PRELOAD`、`VLLM_METRICS`、`VLLM_CHAT`、`VLLM_MODEL`、`IMAGEHUB_TLS_CERT`、`IMAGEHUB_TLS_KEY`、`IMAGEHUB_TTS_PY`、`IMAGEHUB_TTS_PORT`。

HTTPS：`tlsutil.build_uvicorn_cmd()` 只在证书和私钥都存在时追加 `--ssl-certfile/--ssl-keyfile`；`_tls_enabled()` 决定会话 cookie 的 `https_only`。自签证书放 `/etc/imagehub/tls/`，内部健康检查用 `--cacert`，不要关闭 TLS 校验。

`settings.json`（BFF 自动创建）：`backend`（`direct|comfy`）、`gate`、`clocks`。

## 修改时的注意事项

- **部署路径是硬编码的**：`run.py` 的 `BASE_DIR`、`SERVICES` 的 cwd、`imagehub.service` 的 `WorkingDirectory` 都指向 `/opt/vLLM/imagehub`（小写），部署时 `run.py` 会以 `0_imagehub.py` 名称放置。本仓库签出目录为 `/opt/vLLM/ImageHub`，直接运行前需调整或建立同名路径
- **默认后端以代码为准**：`_DEF_SETTINGS["backend"] = "direct"`。README 个别段落写「默认 comfyui」属于旧口径
- **GPU 归属**：3.6 vLLM 使用宿主机 GPU2/3；图像服务通过 `CUDA_VISIBLE_DEVICES=0,1` 只看到 GPU0/1。`/api/gpu` 只上报宿主机 GPU2/3，改布局时注意这里
- **3090（sm_86）不支持 convrot 量化**（int8_convrot/w4a8），会触发 CUDA illegal memory access；只能用 GGUF/bf16
- `qwen` 的 1536/2048 依赖 VAE tiling（服务默认 `QIMAGE_VAE_TILING=1`），否则 VAE 解码 OOM
- 所有模型单边上限 2048，超限由服务端显式返回 400
- 新增模型需要：`SERVICES` 增加服务定义 → `MODELS` 注册 →（ComfyUI 路径）写 `wf_*` 并加入 `WF_BUILDERS` / `MODEL_WORKFLOWS` → OpenAI 别名加入 `MODEL_ALIASES`；权重按 `needs` 路径准备
- ComfyUI 节点图（`wf_h3/wf_qwen/wf_zimage/wf_moss/wf_qwen_edit/wf_qwen_cn`）是手写扁平化节点，节点 id 互相引用，改动时先对照 `/opt/ComfyUI` 实际加载器参数
- 前端没有构建链，直接改 `index.html`；`GET /` 已设置 `no-store`，浏览器仍可能缓存，调试时强制刷新
- 仓库没有 lint/format/测试配置，不要臆造 `pytest`/`ruff` 命令

## 文档地图

- `README.md`：部署步骤、模型权重下载、API、调度行为、常见问题
- `docs/ImageHub调用说明.md`：全部对外接口与参数（最完整的 API 参考）
- `docs/ImageHub设计版需求分析.md` / `docs/ImageHub设计版成果报告.md`：设计版需求、双后端决策、实测数据与已知限制
- `docs/ImageHub全功能压测记录.md`：16 项功能组合压测与历史问题修复
- `docs/70服务器GPU归属调整.md`：3.6 与图像服务的 GPU 重新分配
- `docs/ImageHub文生图平台.md`：平台概览、模型对比与踩坑记录
