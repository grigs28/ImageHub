# ImageHub 设计版 · 成果报告

> 日期：2026-11-21　版本：**1.5.0（设计版）**
> 服务器：192.168.0.70　端口：5569　仓库：https://github.com/grigs28/ImageHub
> 相关文档：`ImageHub设计版需求分析.md`（需求）、`ImageHub调用说明.md`（API 调用）

---

## 1. 总体成果

把原来"文生图 + 对比 + 编辑 + 运维配置"挤在一页的 ImageHub，改造成**面向设计业务、权限分层、双后端可切换、可编程调用**的平台：

- 页面拆分为 **编辑比较 / 设计 / 图库 / API / 管理** 五个入口
- 新增 **ControlNet 设计工作流**（白模/线稿 → 效果图、局部重绘、多角度、扩图）
- 新增 **权限体系**（本地为准的提权/降权、受保护用户、API Key）
- 新增 **设计领域能力**：自动深度图、材质/风格预设、批量方案、异步队列 + webhook
- 稳定性结论修正：整机冻结根因是**模型加载**，不是与 3.6 并发

---

## 2. 架构

### 2.1 双后端

| 后端 | 执行方式 | 用途 |
|---|---|---|
| **直连 diffusers** | 每模型一个 FastAPI 服务（`services/`） | 文生图、编辑（速度快：qwen 1024 ≈ 10s） |
| **ComfyUI** | 节点工作流（`wf_*`）+ ControlNet | 设计类全部功能（唯一支持 Qwen-Image-2.1 ControlNet 的路径） |

- 工作流注册表 `MODEL_WORKFLOWS`：同一模型可按功能选不同工作流（qwen: `t2i` / `controlnet`）
- 后端切换写入 `settings.json`，重启保持
- 同一时间只驻留一套服务（GPU2/3 共 48G）

### 2.2 权限

- yz-login 登录 → 同步写入本地 `users` 表；`is_admin` **以本地为准**，管理页可改
- 管理员判定：会话 / 管理员 Key（`role=admin`）/ 引导 Token
- `grigs` 为受保护账号，不可改权限、不可删除
- 管理类接口（设置、功耗、压测、用户、Key、审计）要求管理员，否则 403

### 2.3 数据（SQLite，WAL）

`jobs`（任务/图片/参数）、`users`、`api_keys`、`audit`、`projects`、`presets`

---

## 3. 功能与实测数据

### 3.1 设计（ControlNet，ComfyUI）

模型：**Qwen-Image-2.1 Fun ControlNet Union（int8_convrot, 3.5G）**，3090 实测可用。

| 能力 | 接口 | 实测 |
|---|---|---|
| 控制图生成 | `POST /v1/design/render` | 1024 ≈ 25s、**1536 ≈ 39s、2048 ≈ 69s** |
| 白模 → 效果图 | `POST /v1/design/clay2render` | 41.5s（生成 24s） |
| 线稿 → 效果图 | `POST /v1/design/sketch2render` | 27s（生成 24s） |
| 局部重绘 | `render` + `mask_image` | 25.1s |
| 多角度 | `POST /v1/design/views` | 2 视角 51.9s |
| 批量方案 | `POST /v1/design/batch` | n=2 → 46.1s |
| 扩图 | `POST /v1/design/outpaint` | 1024→1536，56.4s |
| 异步 + webhook | `POST /v1/design/async` | 25s 完成，回调实测收到 |
| 深度/法线图 | `POST /v1/design/preprocess` | **CPU 2.9s**，不占显存 |
| 预设库 | `GET /v1/design/presets` | 材质/光照/风格/视角，支持管理页自定义 |

### 3.2 性能关键优化

| 优化 | 效果 |
|---|---|
| 同底座模型不卸载 | 设计连拍：第一张 25s → **第二张 16s** |
| VAE tiling（qwen 文生图） | 1536/2048 从 OOM 变为可用（52s） |
| qwen 2048 连续 5 张 | **5/5 成功，52±1s，显存恒定** |

### 3.3 权限与接口

- API Key：`Authorization: Bearer` 或请求体 `apikey`；**外部调用必须带 Key**，网页登录态可直接调用
- Key 在管理页可见、可复制；`role`：user（生成/设计）/ admin（额外管理接口）
- 尺寸上限：h3/qwen/zimage/moss 均 2048；超限返回 400

---

## 4. 稳定性结论（口径已修正）

**整机冻结根因 = 模型加载/卸载时的大块显存分配与释放，不是与 3.6 并发运行。**

支持证据：
- 常驻模型 + 3.6 满载：qwen 30/30 张、h3 连续 5 张、moss 12/12，均无异常
- 所有冻结都出现在加载瞬间（含 ControlNet 首次加载）
- 功率、散热已排除（4 卡纯 CUDA 1471W/89°C 稳定）

因此采用的策略：
- 模型常驻、同底座不卸载；加载集中在少数时刻
- UI 在**需要加载/切换模型**时弹窗确认；API 直接加载
- 设计前停止直连服务，避免两边抢显存

---

## 5. 已知限制与风险

| 项 | 说明 |
|---|---|
| ControlNet 仅 ComfyUI | diffusers 0.43 无 Qwen-Image-2.1 ControlNet pipeline；直连需自行移植（1~2 天，高风险） |
| ComfyUI 版本脆弱 | 2026 年多起"升级即坏"事故；**必须冻结版本**（ComfyUI commit、插件、模型、驱动） |
| 单点 | 设计 100% 依赖 ComfyUI；建议保留直连作为文生图兜底 |
| 僵死无自愈 | 工作流异常可能卡住实例但不退出，建议加健康检查自动重启 |
| 双通道约束未做 | 行业标准是深度 + 线稿叠加；我们目前单通道 |
| 风格一致性 | 无 IP-Adapter / LoRA，同质化风险 |
| 局部错位 | 8 步蒸馏模型在细密栏杆等复杂结构上可能错位 |

---

## 6. 部署要点

```bash
# ControlNet 模型（ComfyUI model_patches，不是 controlnet 目录）
ln -sf /opt/models/Comfy-Org/Qwen-Image-2.1/model_patches/qwen_image_2.1_fun_controlnet_union_int8_convrot.safetensors \
       /opt/ComfyUI/models/model_patches/
systemctl restart comfyui-h3

# 深度预处理模型
modelscope download --model depth-anything/Depth-Anything-V2-Small-hf \
  --local_dir /opt/models/depth-anything/Depth-Anything-V2-Small-hf
```

- 服务：`imagehub.service`（5569）、`comfyui-h3.service`（5568）、`tclf90_Qwen3.6...`（5563）
- 配置：`/opt/vLLM/imagehub/settings.json`（backend/gate/clocks）、`.env`（SSO、管理 Token）
- 日志：`/var/log/imagehub_*.log`、`journalctl -u imagehub/comfyui-h3`

---

## 7. 代码与提交

- 本地源码：`/opt/vLLM/docs/脚本/imagehub_app.py`、`imagehub_index.html`
- 独立项目：`/opt/vLLM/ImageHub/`（app.py、index.html、services/、README）
- 主要提交：`c8a2d41`（设计版）、`50c6663`（预设/深度/webhook）、`48ddcbf`（同模型不卸载）、`732c7a9`（按需弹窗）
- 版本号：`1.5.0`
