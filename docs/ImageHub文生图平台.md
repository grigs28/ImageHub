# ImageHub · 统一文生图对比平台

> 上线日期：2026-11-20
> 访问地址：**https://192.168.0.70:5569**

## 一、功能

- 4 个模型任选（可多选），同一提示词一次提交，并排出图对比
- 显示每个模型的耗时、尺寸、seed，图片可点开原图
- 历史记录（SQLite，**20 条/页分页**，服务端缩略图），点击可回看
- **双后端**：直连 diffusers（文生图/编辑，快）与 ComfyUI（ControlNet 设计）
- **设计能力**：白模/线稿→效果图、局部重绘、多角度、扩图、批量、异步+webhook
- **权限**：yz-login 同步本地，管理员可提权/降权；外部 API 必须带 Key
- 长操作右下角**浮窗进度**（逐步状态 + 计时）

> 2026-11-21 起 GPU 归属调整：3.6 固定 GPU2/3，图像服务固定 GPU0/1（详见 `70服务器GPU归属调整.md`）

## 二、架构

```
浏览器 (5569 前端)
   ↓
ImageHub BFF (FastAPI :5569)
   · 4 套 workflow 模板（参数化 prompt/尺寸/种子）
   · 顺序调度 + /free 卸载（显存互斥）
   · 结果聚合 / 历史 / 图片代理
   ↓ ComfyUI HTTP API (/prompt, /history, /free)
ComfyUI 0.46 (:5568, GPU2+3)
   ↓ MultiGPU 加载器
4 套权重（DiT↔cuda:0, 编码器↔cuda:1）
   ↓
GPU2 + GPU3
```

## 二·补充、交互功能（2026-11-20 迭代）

| 功能 | 说明 |
|------|------|
| **单独预热加载** | 每个模型卡片上的"预热加载"按钮，会先卸载其它模型再加载该模型（小尺寸 1 步跑通），状态点变绿显示"已加载" |
| **直达 ComfyUI** | 卡片上"打开 ComfyUI"链接，新标签页打开 `http://<host>:5568` |
| **同页图片查看器** | 点击结果图片在页内打开灯箱：**滚轮缩放（0.15~12 倍，以光标为中心）**、拖动平移、双击复位、Esc/点背景关闭 |
| 状态轮询 | 前端每 4 秒查询 `/api/status`，显示当前已加载模型 |

新增接口：
```bash
POST /api/launch  {"model":"zimage"}   # 预热加载
GET  /api/status                       # {"loaded":"zimage"|null}
```
预热尺寸：H3 1024²、Qwen/Z-Image/MOSS 640²（1 步，仅用于加载权重）。

## 二·补充二、常驻调度模型（2026-11-20 迭代）

ImageHub 已升级为**常驻模型服务**，可直接被程序调用：

- **按需加载**：未加载的模型首次请求自动加载（+20~40s）
- **常驻复用**：加载后不卸载，同模型后续请求只用生成时间
- **同模型并行**：同模型的并发请求并行排队执行
- **跨模型排队**：换模型时等待当前生成完成后再卸载+加载，不打断任务
- **空闲不卸载**（按配置）：显存一直占用，响应最快

接口：`POST /v1/images/generations`（OpenAI 兼容）、`GET /v1/models`、`/api/status`、`/api/launch`。
完整调用说明见 **`ImageHub调用说明.md`**。

实测：zimage 首次 21s（含加载）→ 第二次 4s（复用）；zimage→qwen 切换并生成 6s。

## 三、模型与实测（同 prompt：赛博朋克城市夜景，霓虹灯招牌写着中文杭州）

| 模型 | 权重 | 默认尺寸 | 步数/采样 | 耗时 | 显存 | 质量 |
|------|------|---------|----------|------|------|------|
| **MiniMax H3-Image** | Abiray GGUF Q5 + H3 专用编码器 Q4 | 2048² | 6 / euler-simple / CFG1 | 30s | 38G | ⭐⭐⭐⭐⭐ "杭州"完美、电影级写实 |
| **Qwen-Image-2.1-Turbo** | Comfy-Org bf16 | 1024² | 8 / 自定义 sigma / CFG1 | 24s | 30G | ⭐⭐⭐⭐ "杭州"清晰、其余文字部分失真 |
| **Z-Image-Turbo** | Comfy-Org bf16 | 1024² | 8 / res_multistep / CFG1 | 15s | 19G | ⭐⭐⭐⭐ 速度最快，文字可辨 |
| **MOSS-Image-SFT** | Comfy-Org bf16 | 1024² | 25 / er_sde / CFG2.5 | 48s | 28G | ⭐⭐⭐ 霓虹文字多且可辨，写实感稍弱 |

- 4 个模型全选总耗时约 **2 分钟**（含每次切换的模型加载）
- 每次切换前调用 ComfyUI `/free` 卸载上一模型，避免 OOM

## 四、API

```bash
# 提交对比
curl -X POST https://192.168.0.70:5569/api/compare \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"...","models":["h3","qwen","zimage","moss"],"seed":42}'
# → {"job_id":"..."}

# 查询进度
curl https://192.168.0.70:5569/api/jobs/{job_id}

# 模型列表 / 历史
curl https://192.168.0.70:5569/api/models
curl https://192.168.0.70:5569/api/history
```

可选参数：`width`、`height`（不给用模型默认）、`seed`。

## 五、文件位置

| 内容 | 路径 |
|------|------|
| BFF 主程序 | `/opt/vLLM/imagehub/app.py` |
| 前端页面 | `/opt/vLLM/imagehub/index.html` |
| 启动脚本（含数据库注册） | `/opt/vLLM/imagehub/0_imagehub.py` |
| systemd | `imagehub.service`（:5569） |
| ComfyUI | `/opt/ComfyUI`（`comfyui-h3.service`，:5568） |
| 权重 | `/opt/models/Comfy-Org/`、`/opt/models/Abiray/`、`/opt/models/Qwen/` |
| 生成图 | `/opt/ComfyUI/output/hub_*.png` |

## 六、扩展新模型

在 `app.py` 中新增：
1. 写一个 `wf_xxx(prompt, w, h, seed)` 返回 ComfyUI API 格式工作流（节点 id → class_type/inputs）
2. 在 `MODELS` 中登记（label、默认尺寸、note）
3. 模型文件软链到 `/opt/ComfyUI/models/{diffusion_models,text_encoders,vae}/`

## 七、踩坑记录

1. **UNETLoaderMultiGPU 必填 `weight_dtype`**（值为 `default`），否则 `/prompt` 返回 400 `required_input_missing`
2. **MultiGPU 加载器参数**：`unet_name/weight_dtype/device`、`clip_name/type/device`、`vae_name/device`
3. **CLIPLoader type 需匹配模型**：Qwen→`qwen_image`、Z-Image→`lumina2`、MOSS→`moss_image`、H3→`minimax`
4. **模型互斥**：H3/Qwen/MOSS/Z-Image 都要占用 GPU2+3 的部分或全部，切换前必须 `/free`，否则 OOM
5. **官方模板是子图格式**，不能直接走 API；需按子图内节点手工搭平铺工作流
6. H3 必须用专用编码器 `qwen3vl_32b_minimax_h3-Q4_K_M.gguf`（通用 Qwen3VL GGUF 加载失败）
7. 前端 `border-color:#3a4considerable` 这类笔误会让样式静默失效——写 CSS 后要肉眼核对
