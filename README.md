# ImageHub

> 统一文生图 / 图片编辑平台：一个网页 + 一套 OpenAI 兼容 API，统一调度多个开源图像模型。
> 支持两条执行后端：**直连 diffusers 服务**（推荐）或 **ComfyUI**；ImageHub 负责后端切换、模型驻留调度、并发排队、历史记录与鉴权。

![version](https://img.shields.io/badge/version-1.4.0-blue)

## 1.5.0 主要更新（设计版）

- **页面拆分**：工作台 / 图库 / API / 管理（仅管理员），刷新保持当前页
- **权限体系**：yz-login 用户同步到本地库，`is_admin` 以本地为准，可提权/降权；接口按管理员保护
- **API Key**：可选（Bearer 或请求体 `apikey`），UI 可见可复制，只保留必要数量
- **ControlNet（设计类）**：Qwen-Image-2.1 Fun ControlNet Union
  - `POST /v1/design/render`（控制图 + 类型 + 强度，支持蒙版做局部重绘）
  - `POST /v1/design/clay2render`（白模 → 效果图）
  - `POST /v1/design/sketch2render`（线稿/SU 截图 → 效果图）
  - `POST /v1/design/batch`（同提示词多 seed，n ≤ 4）
  - `GET /v1/jobs/{id}`（任务查询）
- **多工作流注册表**：同一模型可按功能选不同工作流（qwen: t2i / controlnet）
- **项目与审计**：项目选择器、项目内任务查询、管理页审计日志
- **稳定性**：ComfyUI 串行锁加超时（600s），避免请求永久排队

### ControlNet 模型准备

```bash
# 模型放入 ComfyUI 的 model_patches（不是 controlnet 目录）
mkdir -p /opt/ComfyUI/models/model_patches
ln -sf /opt/models/Comfy-Org/Qwen-Image-2.1/model_patches/qwen_image_2.1_fun_controlnet_union_int8_convrot.safetensors \
       /opt/ComfyUI/models/model_patches/
systemctl restart comfyui-h3
```

> int8 convrot 版（3.5G）在 RTX 3090 上实测可用；bf16 版（7G）显存更紧张。
> ControlNet 目前仅支持 ComfyUI 后端；直连 diffusers 适配在路线图 P3。

## 1.4.0 主要更新

- **双后端可切换**：`直连 diffusers`（每模型一个独立 FastAPI 服务，`services/` 目录）与 `ComfyUI`，可在页面底部切换并保存到 `settings.json`
- **API 增加 `backend` 参数**：`comfyui | direct`，按请求选择后端；省略则用全局设置
- **模型卡「生成」按钮**：按当前提示词、尺寸、种子、后端直接出图，结果进入「对比结果」列表并立即写入历史
- **运行设置面板**：后端切换、vLLM 空闲门控、GPU 锁频（1410MHz）开关，一键保存
- **3.6 对话压测**：页面内开始/停止，实时显示 ok/err/耗时
- **历史记录元数据**：每条记录显示分辨率 · seed · 耗时
- **尺寸保护**：按模型校验最大分辨率（qwen 1024，其余 2048），超限直接返回明确错误
- **稳定性提示**：vLLM 满载时切换图像后端可能触发整机冻结（跨 GPU 显存分配 + P2P 同步问题），建议低峰切换或开启门控

---

## 特性

- **多模型统一入口**：H3-Image / Qwen-Image-2.1（原版 + 无限制）/ Z-Image-Turbo / MOSS-Image / Qwen-Image-Edit-2511
- **按需加载 + 常驻复用**：请求哪个模型自动加载，加载后不卸载；同模型并行、跨模型排队，不打断进行中的生成
- **网页对比**：同一提示词多模型并排出图，支持灯箱放大（滚轮缩放/拖动）、历史缩略图、复制提示词、批量删除
- **图片编辑**：上传新图或直接选用历史图，用自然语言指令修改
- **OpenAI 兼容 API**：`POST /v1/images/generations`，可直接用 OpenAI SDK
- **双卡分组件**：DiT 与文本编码器分置两张显卡，24GB 卡也能跑大模型
- **运维友好**：GPU 功耗上限控制（网页可调）、功耗/温度日志（重启溯源）、SSO 接入（yz-login）、systemd 服务化

---

## 架构

```
浏览器 / 程序 (HTTP)
        │
        ▼
  ImageHub BFF (FastAPI :5569)
  ├─ 模型驻留管理器（同模型并行 / 跨模型排队）
  ├─ 工作流模板（每个模型一套）
  ├─ 历史记录（SQLite）+ 图片代理
  ├─ OpenAI 兼容 API
  └─ SSO 回调（可选）
        │  ComfyUI HTTP API (/prompt /history /free)
        ▼
  ComfyUI (:5568)  ── MultiGPU 加载器 ──►  GPU0/1/2/3
        │
        └─ 模型权重（GGUF / safetensors）
```

---

## 支持模型

| key | 模型 | 参数 | 默认尺寸 | 单卡/双卡 | 说明 |
|-----|------|------|---------|----------|------|
| `h3` | MiniMax H3-Image（GGUF Q5） | 33B DiT（6 步） | 2048×2048 | 双卡 | 画质最强，中文文字最好 |
| `qwen` | Qwen-Image-2.1-Turbo | 7B DiT（8 步） | 1024×1024 | 双卡 | 中文排版强 |
| `qwen_uc` | Qwen-Image-2.1-Turbo（无限制） | 同上 + abliterated 编码器 | 1024×1024 | 双卡 | 同画质，限制更少 |
| `zimage` | Z-Image-Turbo | 6B DiT（8 步） | 1024×1024 | 双卡 | 速度最快 |
| `moss` | MOSS-Image-SFT | 5B DiT（25 步） | 1024×1024 | 双卡 | 布局/海报专精 |
| — | Qwen-Image-Edit-2511 | 20B DiT（40 步） | 1024×1024 | 双卡 | 图片编辑（`/api/edit`） |

> 显存参考（24GB×2，实测峰值）：H3 约 34G、Qwen 约 38G、MOSS 约 28G、Z-Image 约 19G。

---

## 环境要求

| 项目 | 要求 |
|------|------|
| 操作系统 | Linux（推荐 openEuler / Ubuntu） |
| Python | 3.10+ |
| GPU | 2× 24GB NVIDIA（3090/4090 等），CUDA 12.x/13.x |
| 内存 | ≥ 64GB（建议 128GB） |
| 磁盘 | ≥ 300GB（模型权重） |
| 引擎 | ComfyUI ≥ 0.46 + 自定义节点（GGUF / MultiGPU） |

> ⚠️ **3090（sm_86）不支持 convrot 系列量化**（int8_convrot / w4a8），加载会触发 `CUDA illegal memory access`。请使用 **GGUF** 或 **bf16** 权重。

---

## 部署步骤

### 1. 安装 ComfyUI 与自定义节点

```bash
cd /opt
git clone --depth 1 https://github.com/comfyanonymous/ComfyUI.git
cd ComfyUI
pip install -r requirements.txt

cd custom_nodes
git clone --depth 1 https://github.com/city96/ComfyUI-GGUF.git
git clone --depth 1 https://github.com/pollockjj/ComfyUI-MultiGPU.git
pip install gguf
```

### 2. 下载模型权重

ComfyUI 模型目录结构：

```
ComfyUI/models/
├── diffusion_models/
├── text_encoders/
└── vae/
```

推荐用 `hf`（HuggingFace）或 `modelscope` 下载，再用软链接放入 ComfyUI。

#### 2.1 H3-Image（GGUF）

```bash
# DiT（Q5_K_M，13.1G）
hf download Abiray/MiniMax-H3-Image-GGUF \
  --local-dir /opt/models/Abiray/MiniMax-H3-Image-GGUF \
  --include "MiniMax-H3-Image-Pruned-Q5_K_M.gguf"

# H3 专用文本编码器（Q4_K_M，13.6G）——必须用这个，通用 Qwen3-VL GGUF 不兼容
hf download Abiray/MiniMax-H3-GGUF \
  --local-dir /opt/models/Abiray/MiniMax-H3-GGUF \
  --include "text_encoders/qwen3vl_32b_minimax_h3-Q4_K_M.gguf"

# VAE（24 通道图像 VAE，2.2G）
modelscope download --model MiniMax/MiniMax-H3-Image \
  --local_dir /opt/models/MiniMax/MiniMax-H3-Image --include "vae/*"

ln -sf /opt/models/Abiray/MiniMax-H3-Image-GGUF/MiniMax-H3-Image-Pruned-Q5_K_M.gguf \
       ComfyUI/models/diffusion_models/
ln -sf /opt/models/Abiray/MiniMax-H3-GGUF/text_encoders/qwen3vl_32b_minimax_h3-Q4_K_M.gguf \
       ComfyUI/models/text_encoders/
ln -sf /opt/models/MiniMax/MiniMax-H3-Image/vae/diffusion_pytorch_model.safetensors \
       ComfyUI/models/vae/minimax_h3_vae.safetensors
```

#### 2.2 Qwen-Image-2.1-Turbo（+ 无限制编码器）

```bash
# DiT + 编码器 + VAE（bf16，共 30.2G）
modelscope download --model Comfy-Org/Qwen-Image-2.1 \
  --local_dir /opt/models/Comfy-Org/Qwen-Image-2.1 \
  --include "diffusion_models/qwen_image_2.1_turbo_bf16.safetensors" \
            "text_encoders/qwen3vl_8b_bf16.safetensors" \
            "vae/qwen_image_2.1_vae_bf16.safetensors"

# 无限制版编码器（abliterated，16.3G）
hf download abenzerps/Qwen-Image-2.1-Uncensored-GGUF \
  --local-dir /opt/models/Comfy-Org/Qwen-Image-2.1-UC \
  --include "text_encoders/qwen3vl_8b_bf16.safetensors"

ln -sf /opt/models/Comfy-Org/Qwen-Image-2.1/diffusion_models/qwen_image_2.1_turbo_bf16.safetensors \
       ComfyUI/models/diffusion_models/
ln -sf /opt/models/Comfy-Org/Qwen-Image-2.1/text_encoders/qwen3vl_8b_bf16.safetensors \
       ComfyUI/models/text_encoders/
ln -sf /opt/models/Comfy-Org/Qwen-Image-2.1/vae/qwen_image_2.1_vae_bf16.safetensors \
       ComfyUI/models/vae/
ln -sf /opt/models/Comfy-Org/Qwen-Image-2.1-UC/text_encoders/qwen3vl_8b_bf16.safetensors \
       ComfyUI/models/text_encoders/qwen3vl_8b_uncensored.safetensors
```

#### 2.3 Z-Image-Turbo

```bash
modelscope download --model Comfy-Org/Z-Image-Turbo \
  --local_dir /opt/models/Comfy-Org/Z-Image-Turbo \
  --include "split_files/diffusion_models/z_image_turbo_bf16.safetensors" \
            "split_files/text_encoders/qwen_3_4b.safetensors" \
            "split_files/vae/ae.safetensors"

ln -sf /opt/models/Comfy-Org/Z-Image-Turbo/split_files/diffusion_models/z_image_turbo_bf16.safetensors \
       ComfyUI/models/diffusion_models/
ln -sf /opt/models/Comfy-Org/Z-Image-Turbo/split_files/text_encoders/qwen_3_4b.safetensors \
       ComfyUI/models/text_encoders/
ln -sf /opt/models/Comfy-Org/Z-Image-Turbo/split_files/vae/ae.safetensors ComfyUI/models/vae/
```

#### 2.4 MOSS-Image-SFT

```bash
modelscope download --model Comfy-Org/MOSS-Image \
  --local_dir /opt/models/Comfy-Org/MOSS-Image \
  --include "diffusion_models/moss_image_sft_bf16.safetensors" \
            "text_encoders/qwen3.5_9b_bf16.safetensors" \
            "vae/flux2-vae.safetensors"

ln -sf /opt/models/Comfy-Org/MOSS-Image/diffusion_models/moss_image_sft_bf16.safetensors \
       ComfyUI/models/diffusion_models/
ln -sf /opt/models/Comfy-Org/MOSS-Image/text_encoders/qwen3.5_9b_bf16.safetensors \
       ComfyUI/models/text_encoders/
ln -sf /opt/models/Comfy-Org/MOSS-Image/vae/flux2-vae.safetensors ComfyUI/models/vae/
```

#### 2.5 Qwen-Image-Edit-2511（图片编辑）

```bash
# DiT（GGUF Q6_K，16.7G）——不要用 int8_convrot，3090 上会崩
hf download unsloth/Qwen-Image-Edit-2511-GGUF \
  --local-dir /opt/models/Comfy-Org/Qwen-Image-Edit-2511-gguf \
  --include "qwen-image-edit-2511-Q6_K.gguf"

# 文本编码器（Qwen2.5-VL-7B，8.7G）
modelscope download --model Comfy-Org/Qwen-Image_ComfyUI \
  --local_dir /opt/models/Comfy-Org/Qwen-Image_ComfyUI \
  --include "split_files/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors"

# VAE（Wan 2.1，243M）
modelscope download --model Comfy-Org/Wan_2.1_ComfyUI_repackaged \
  --local_dir /opt/models/Comfy-Org/Wan2.1 --include "split_files/vae/wan_2.1_vae.safetensors"

ln -sf /opt/models/Comfy-Org/Qwen-Image-Edit-2511-gguf/qwen-image-edit-2511-Q6_K.gguf \
       ComfyUI/models/diffusion_models/
ln -sf /opt/models/Comfy-Org/Qwen-Image_ComfyUI/split_files/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors \
       ComfyUI/models/text_encoders/
ln -sf /opt/models/Comfy-Org/Wan2.1/split_files/vae/wan_2.1_vae.safetensors ComfyUI/models/vae/
```

### 3. 部署 ImageHub

```bash
mkdir -p /opt/vLLM/imagehub
cp app.py index.html run.py /opt/vLLM/imagehub/
pip install -r requirements.txt

# 环境变量（可选，SSO 与数据库注册用；不配也能跑）
cp .env.example /opt/vLLM/imagehub/.env
chmod 600 /opt/vLLM/imagehub/.env
# 编辑 .env 填入 YZ_APP_ID / YZ_APP_SECRET / SESSION_SECRET
```

手动启动测试：

```bash
cd /opt/vLLM/imagehub && python run.py
# 浏览器打开 http://<服务器IP>:5569
```

### 4. systemd 服务

```bash
# ComfyUI 引擎（由 ImageHub 按需拉起，因此可不设开机自启）
cp comfyui-h3.service /etc/systemd/system/   # 按需修改路径
systemctl enable --now comfyui-h3

# ImageHub
cp imagehub.service /etc/systemd/system/
systemctl daemon-reload
systemctl enable --now imagehub
```

### 5. 功耗与运维（可选，推荐）

```bash
cp gpu-power-cap.sh gpu-power-logger.sh /usr/local/bin/ && chmod +x /usr/local/bin/gpu-power-*.sh
cp gpu-power-cap.service gpu-power-logger.service /etc/systemd/system/
systemctl daemon-reload
systemctl enable --now gpu-power-cap gpu-power-logger
```

- `gpu-power-cap`：开机限制 GPU 功耗（默认 300/300/250/250W，可用 `GPU0_LIMIT` 等环境变量覆盖）
- `gpu-power-logger`：每 5 秒记录功耗/温度到 `/var/log/gpu-power.log`，用于异常重启溯源
- `deploy-guard.sh`：安全重启（有生成任务时等待其完成）

---

## API

### 文生图（OpenAI 兼容）


```bash
curl -X POST http://<IP>:5569/v1/images/generations \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen","prompt":"赛博朋克城市夜景","size":"1024x1024","seed":42,"backend":"direct","response_format":"url"}'
```

| 参数 | 说明 |
|------|------|
| `model` | `h3` / `qwen` / `zimage` / `moss` / **`auto`** |
| `prompt` | 提示词 |
| `size` | `宽x高`，不传用模型默认 |
| `response_format` | `url`（相对路径）或 `b64_json` |
| `only_loaded` | `true` 时只用当前已加载模型，不触发加载（未加载返回 409/503） |
| `seed` | 随机种子，默认 42 |
| `backend` | `comfyui`（默认，或全局设置）/ `direct`（直连 diffusers） |

`model: "auto"`：优先用当前已加载的模型；没有则回落到 `qwen`。

**分辨率上限**（服务端校验，超限返回明确错误）：`h3`/`qwen`/`zimage`/`moss` 均为 2048（moss 2048 极慢 ~8 分钟，不建议）。

> 说明：`qwen` 的 1536/2048 依赖 **VAE tiling**（服务已默认开启 `QIMAGE_VAE_TILING=1`），否则 VAE 解码会 OOM。实测 1024 约 10s、1536 约 26s、2048 约 52s（双卡 balanced）。

Python（OpenAI SDK）：

```python
from openai import OpenAI
import base64

client = OpenAI(base_url="http://<IP>:5569/v1", api_key="none")
r = client.images.generate(model="qwen", prompt="水彩橘猫", size="1024x1024",
                           response_format="b64_json")
open("out.png","wb").write(base64.b64decode(r.data[0].b64_json))
```

### 其他接口

| 接口 | 说明 |
|------|------|
| `GET /v1/models` | 模型列表 |
| `POST /api/compare` | 多模型对比，返回 `job_id` |
| `GET /api/jobs/{id}` | 对比任务进度与结果 |
| `POST /api/edit` | 图片编辑（`prompt` + `image_b64`） |
| `POST /api/launch` | 预热加载 |
| `GET /api/status` | 当前驻留模型 / 并发数 / 版本 |
| `GET /api/power` · `POST /api/power` | GPU 功耗查询/设置 |
| `GET /api/history` · `DELETE /api/history/{id}` · `POST /api/history/batch-delete` | 历史记录 |
| `GET /api/version` | 版本号 |
| `GET /api/settings` · `POST /api/settings` | 后端/门控/锁频设置（持久化到 settings.json） |
| `POST /api/stress36/start` · `POST /api/stress36/stop` · `GET /api/stress36/status` | 3.6 对话压测 |

### 调度行为

| 场景 | 行为 |
|------|------|
| 同模型连续请求 | 复用已加载模型（只花生成时间） |
| 请求不同模型 | 等当前生成结束 → 卸载 → 加载新模型（+20~40s） |
| 同模型并发 | 并行排队 |
| 跨模型并发 | 排队，不打断进行中的任务 |
| 空闲 | 不自动卸载 |

---

## 目录结构

```
imagehub/
├── app.py                  # BFF 主程序（工作流、调度、API、SSO、历史）
├── index.html              # 前端单页
├── run.py                  # 启动包装（数据库注册 + uvicorn）
├── requirements.txt
├── .env.example            # 环境变量模板
├── imagehub.service        # systemd 单元（:5569）
├── deploy-guard.sh         # 安全重启
├── gpu-power-cap.sh/.service
└── gpu-power-logger.sh/.service
```

---

## 常见问题

| 问题 | 原因 / 处理 |
|------|------------|
| `/prompt` 返回 400 `required_input_missing: weight_dtype` | `UNETLoaderMultiGPU` 必须传 `weight_dtype: "default"` |
| 加载模型 `CUDA illegal memory access` | 3090（sm_86）不支持 convrot 量化；换 GGUF/bf16 权重 |
| 对比到一半停止 | 曾经是重启服务打断任务；现已支持逐模型落盘 + 中断标记，重启请用 `deploy-guard.sh` |
| 显存不足 | 模型互斥，确保切换前 `/free`；或降低 `max-model-len` / 使用量化权重 |
| 机器满载时重启 | 检查电源功率与 12V 分路；先 `nvidia-smi -pl 250` 降压，并查看 `/var/log/gpu-power.log` |
| 页面显示"准备中" | 对应模型权重未下载完成，接口会返回 503 |
| 浏览器看到旧页面 | 服务已加 `no-store`，强制刷新 Ctrl+Shift+R |

---

## License

MIT
