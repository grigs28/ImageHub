# ImageHub 调用说明

> 服务地址：**https://192.168.0.70:5569**
> 证书：自签证书，浏览器首次访问需接受；命令行用 `--cacert /etc/imagehub/tls/imagehub.crt`（或在客户端导入该证书）。
> 下面的 curl 示例省略了 `--cacert`，使用自签证书时请补上；TTS 接口见第九节。
> 本文档可下载：`GET /api/docs`，或 API 页右上角「下载 API 说明」。
> 底层：**双后端**——直连 diffusers（GPU0/1，文生图/编辑）与 ComfyUI（GPU0/1，ControlNet 设计）
> 支持按需加载、常驻复用、同模型并行、跨模型排队；3.6（vLLM）独立运行在 GPU2/3

> **鉴权（2026-11-21 起）**：外部调用必须带 API Key（请求体 `apikey` 或 `Authorization: Bearer`），
> 无 Key 返回 401。网页登录状态可直接调用。Key 在「管理 → API Key」查看/复制。

## 一、总览

| 接口 | 方法 | 用途 |
|------|------|------|
| `/v1/images/generations` | POST | **OpenAI 兼容**：单模型文生图（支持 `backend`/`apikey`） |
| `/v1/design/render` | POST | ControlNet 结构化生成（控制图 + 强度 + 可选蒙版） |
| `/v1/design/clay2render` | POST | 白模 → 效果图 |
| `/v1/design/sketch2render` | POST | 线稿/SU 截图 → 效果图 |
| `/v1/design/views` | POST | 多角度批量（人视/鸟瞰/立面等，≤4） |
| `/v1/design/batch` | POST | 同提示词多 seed（≤4） |
| `/v1/design/outpaint` | POST | 四边扩图（+expand 像素） |
| `/v1/design/async` | POST | 异步任务（排队 + 可选 webhook 回调） |
| `/v1/design/preprocess` | POST | 自动深度图 / 法线图（CPU，约 3 秒） |
| `/v1/design/presets` | GET | 材质/光照/风格/视角预设 |
| `/v1/jobs/{id}` | GET | 任务查询（同步/异步通用） |
| `/v1/models` | GET | 模型列表（含 capabilities / max_size） |
| `/api/compare` | POST | 多模型对比（一次提交，依次出图） |
| `/api/edit` | POST | 图片编辑（Qwen-Image-Edit-2511，上传图片+指令） |
| `/api/launch` | POST | 预热加载某模型（不生成） |
| `/api/status` | GET | 当前驻留模型 / 并发数 / 是否在切换 |
| `/thumb/{filename}?w=N` | GET | 缩略图（服务端缩放并缓存） |
| `/image/{filename}` | GET | 访问生成结果原图 |
| `/api/history?limit=&offset=` | GET | 历史分页（响应头 `X-Total-Count`） |
| `/` | GET | 网页界面 |

**可用模型 key**：`h3`、`qwen`、`zimage`、`moss`、`flux2klein`、`nemotron`
（别名：`minimax-h3`、`qwen-image-2.1-turbo`、`z-image-turbo`、`moss-image`、`flux`/`flux-2-klein`、`nemotron-image`；`flux2klein`/`nemotron` 仅走 ComfyUI）

| key | 模型 | 默认尺寸 | 典型耗时（已加载） |
|-----|------|---------|------------------|
| `h3` | MiniMax H3-Image | 2048×2048 | 5~10s |
| `qwen` | Qwen-Image-2.1-Turbo | 1024×1024 | 10~20s |
| `zimage` | Z-Image-Turbo | 1024×1024 | 4~8s |
| `moss` | MOSS-Image-SFT | 1024×1024 | 30~48s |
| `flux2klein` | FLUX.2 Klein 4B | 1024×1024 | 约 10s（4 步，ComfyUI） |
| `nemotron` | Nemotron-30B-A3B | 1024×1024 | 约 30~60s（ComfyUI，首次加载更久） |

> 首次请求某模型会自动加载（+20~40 秒）；之后同模型请求直接复用。

## 二、OpenAI 兼容调用

### curl

```bash
curl -X POST https://192.168.0.70:5569/v1/images/generations \
  -H "Content-Type: application/json" \
  -d '{
    "model": "h3",
    "prompt": "赛博朋克城市夜景，霓虹灯招牌写着杭州",
    "size": "2048x2048",
    "response_format": "url"
  }'
```

返回：

```json
{
  "created": 1795111111,
  "data": [{"url": "/image/hub_h3_00001_.png", "revised_prompt": "..."}],
  "model": "h3",
  "duration_s": 7.3,
  "size": "2048x2048"
}
```

- `response_format`: `url`（返回相对路径，拼 `https://192.168.0.70:5569` 访问）或 `b64_json`（base64）
- 完整图片地址：`https://192.168.0.70:5569/image/hub_h3_00001_.png`

### Python（OpenAI SDK）

```python
from openai import OpenAI
import base64

client = OpenAI(base_url="https://192.168.0.70:5569/v1", api_key="none")

r = client.images.generate(
    model="qwen",                  # h3 / qwen / zimage / moss
    prompt="水彩风格的橘猫坐在窗台看雨",
    size="1024x1024",
    response_format="b64_json",
)
img = base64.b64decode(r.data[0].b64_json)
open("out.png", "wb").write(img)
```

### Python（requests，不依赖 SDK）

```python
import requests, base64
r = requests.post("https://192.168.0.70:5569/v1/images/generations", json={
    "model": "zimage",
    "prompt": "cyberpunk city at night",
    "size": "1024x1024",
    "response_format": "b64_json",
}, timeout=600)
r.raise_for_status()
d = r.json()
open("out.png", "wb").write(base64.b64decode(d["data"][0]["b64_json"]))
print("耗时", d.get("duration_s"), "秒")
```

### backend / apikey 参数

| 参数 | 取值 | 说明 |
|---|---|---|
| `apikey` | 管理页生成的 Key | **必填**（或 `Authorization: Bearer`） |
| `backend` | `comfyui` / `direct` | 可选；留空用全局设置 |
| `seed` | 整数 | 可选，默认 42 |
| `size` | `宽x高` | 单边 ≤2048（服务端校验） |

```bash
curl -X POST https://192.168.0.70:5569/v1/images/generations \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen","prompt":"赛博朋克城市夜景","size":"1024x1024",
       "seed":42,"backend":"direct","apikey":"<你的KEY>","response_format":"url"}'
```

### model 参数特殊值

| model 值 | 行为 |
|----------|------|
| `h3` / `qwen` / `qwen_uc` / `zimage` / `moss` / `flux2klein` / `nemotron` | 指定模型；未加载则自动加载；`flux2klein`/`nemotron` 仅 ComfyUI |
| **`auto`**（别名 `loaded` / `current`） | **优先用当前已加载模型**；没有已加载模型时**回落加载 h3** |

配合 `only_loaded` 参数（默认 false）：

| 组合 | 行为 |
|------|------|
| `{"model":"h3","only_loaded":true}` | 若当前已加载的不是 h3 → **返回 409**，不切换不加载 |
| `{"model":"auto","only_loaded":true}` | 若有已加载模型 → 用它；**没有则返回 503**（不加载） |
| `{"model":"auto"}` | 有已加载用已加载；没有则加载 h3 |

示例：

```bash
# 只用已加载的模型，绝不触发加载（适合低延迟场景）
curl -X POST https://192.168.0.70:5569/v1/images/generations \
  -H "Content-Type: application/json" \
  -d '{"model":"auto","only_loaded":true,"prompt":"a cat","response_format":"url"}'
```

错误返回（仅限 `only_loaded=true` 时）：

```json
{"detail":{"error":"当前已加载的是 zimage，未加载 h3",
           "hint":"去掉 only_loaded 会自动切换（约 20~40 秒）",
           "loaded":"zimage","requested":"h3"}}
```

### 查看模型列表

```bash
curl https://192.168.0.70:5569/v1/models
```

## 三、多模型对比

```bash
curl -X POST https://192.168.0.70:5569/api/compare \
  -H "Content-Type: application/json" \
  -d '{"prompt":"赛博朋克城市夜景","models":["h3","qwen","zimage","moss"],"seed":42}'
# → {"job_id":"xxx"}
# flux2klein / nemotron 也可以放进 models（仅 ComfyUI；首次加载较久）

curl https://192.168.0.70:5569/api/jobs/xxx
```

返回的 `results` 数组按模型顺序给出 `status/duration/image`。

可选参数：`width`、`height`（不给用模型默认）、`seed`（默认 42）。

## 四、图片编辑（Qwen-Image-Edit-2511）

```bash
curl -X POST https://192.168.0.70:5569/api/edit \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "把背景换成夜晚的赛博朋克街道",
    "image_b64": "data:image/png;base64,<...>",
    "steps": 40,
    "seed": 42
  }'
```

返回 `{"status":"ok","image":"hub_edit_xxx.png","duration":36.5}`。
网页端也有"图片编辑"面板，可直接上传图片。

## 五、预热与状态

```bash
# 预热（提前加载，避免首个请求等待）
curl -X POST https://192.168.0.70:5569/api/launch -d '{"model":"h3"}' \
  -H "Content-Type: application/json"

# 状态
curl https://192.168.0.70:5569/api/status
# {"loaded":"h3","active":0,"switching":false}
```

## 六、调度行为（重要）

| 场景 | 行为 | 耗时 |
|------|------|------|
| 同一模型连续请求 | 直接复用已加载模型 | 仅生成耗时 |
| 请求不同模型 | **等当前生成跑完** → 卸载 → 加载新模型 → 生成 | +20~40s |
| 同模型并发 | **并行排队执行**（ComfyUI 内部队列） | 顺序完成 |
| 跨模型并发 | **排队**，不打断任何正在跑的生成 | 按顺序 |
| 空闲时 | **不自动卸载**，模型一直驻留 | 0 |

- 同时只有一个重模型驻留（显存 48G 限制，H3/Qwen/MOSS 各需约 28~38G）
- 切换只在 `active == 0` 时进行，运行中的任务不会被中断
- `api/status` 的 `active` 是当前生成中的请求数，`switching` 表示正在换模型

## 七、注意事项

1. `n` 参数目前只支持 1（每次请求生成 1 张）
2. 尺寸建议：`h3` 用 2048×2048（模型有效区间 4~20MP），其余用 1024×1024
3. 超时设置：首次请求（含加载）建议客户端超时 ≥ 120 秒，MOSS 生成建议 ≥ 180 秒
4. 需要 API Key（见第一节）；请勿直接暴露到公网
5. 生成图存放在 `/opt/ComfyUI/output/`，可通过 `/image/{文件名}` 访问

## 八、错误码

| HTTP | 含义 | 处理 |
|------|------|------|
| 400 | 模型名、size 或参数错误 | 检查 `model`/`size`/必填字段 |
| 401 | 缺少/无效 API Key，或未登录 | 检查 `apikey` / Bearer / 会话 |
| 500 | 生成失败（显存/模型加载/CUDA） | 查看返回 `detail`；必要时重启 `comfyui-h3.service` |
| 404 | 图片/语音记录不存在 | 确认文件名或记录 ID |
| 503 | 模型或 TTS 加载中、排队超时 | 稍后重试；查看 `/api/status`、`/api/tts/status` |

## 九、语音合成（TTS）

TTS 服务内部端口 `127.0.0.1:5570`，对外统一走 `5569`。与图像模型**互斥**：图像请求会先停 TTS，TTS 请求会先停图像；双方忙则 FIFO 排队。三个 1.7B 模型（CustomVoice / VoiceDesign / Base）同时常驻，首次加载约 4 秒。

### 鉴权

与图像 API 一致：

- POST：请求体或表单 `apikey`
- GET：查询参数 `?apikey=`
- 也支持 `Authorization: Bearer ih-...` 或网页登录会话
- 无 Key/会话返回 401

### POST /v1/audio/speech（OpenAI 兼容，CustomVoice）

```bash
curl --cacert imagehub.crt -X POST https://192.168.0.70:5569/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"input":"你好，欢迎使用 ImageHub。","voice":"Vivian","language":"Chinese","response_format":"wav","apikey":"ih-..."}' \
  -o out.wav
```

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `input` | string | 是 | 文本，≤2000 字 |
| `voice` | string | 否 | 默认 `Vivian`；另有 Serena / Uncle_Fu / Dylan / Eric / Ryan / Aiden / Ono_Anna / Sohee |
| `language` | string | 否 | `Auto`（默认）/ Chinese / English / Japanese / Korean / German / French / Russian / Portuguese / Spanish / Italian |
| `response_format` | string | 否 | 只支持 `wav` |
| `apikey` | string | 是 | 也可用 Bearer 或网页会话 |

响应：`audio/wav`，24kHz 单声道。

### POST /api/tts/speak（自定义音色 / 声音设计）

```bash
# 自定义音色
{"variant":"custom","text":"你好","language":"Chinese","speaker":"Vivian","instruct":"温柔平静","apikey":"ih-..."}

# 声音设计（instruct 必填）
{"variant":"design","text":"你好","language":"Chinese","instruct":"一位年轻女性，温柔、清晰、语速稍慢","apikey":"ih-..."}
```

| 参数 | 必填 | 说明 |
|------|------|------|
| `variant` | 是 | `custom` / `design` |
| `text` | 是 | 文本，≤2000 字 |
| `language` | 否 | 默认 `Auto` |
| `speaker` | custom 必填 | 预设音色名 |
| `instruct` | design 必填 | 声音描述或语气指令 |

### POST /api/tts/clone（声音克隆，multipart）

```bash
curl --cacert imagehub.crt -X POST https://192.168.0.70:5569/api/tts/clone \
  -F "apikey=ih-..." \
  -F "text=你好，这是克隆测试" -F "language=Chinese" \
  -F "ref_text=参考音频里说的内容" -F "ref_audio=@ref.wav" \
  -o clone.wav
```

参考音频 ≤60 秒 / ≤20MB，支持 wav/mp3/m4a；不填 `ref_text` 需 `x_vector_only_mode=true`（克隆质量会下降）。

### 语音记录

```bash
# 列表
curl --cacert imagehub.crt \
  "https://192.168.0.70:5569/api/tts/history?limit=10&offset=0&apikey=ih-..."

# 播放/下载
curl --cacert imagehub.crt \
  "https://192.168.0.70:5569/api/tts/audio/tts_xxx.wav?apikey=ih-..." -o out.wav

# 删除单条 / 批量删除 / 打包下载
curl --cacert imagehub.crt -X DELETE \
  "https://192.168.0.70:5569/api/tts/history/记录ID?apikey=ih-..."
curl --cacert imagehub.crt -X POST -H "Content-Type: application/json" \
  -d '{"ids":["id1","id2"],"apikey":"ih-..."}' \
  https://192.168.0.70:5569/api/tts/history/batch-delete
curl --cacert imagehub.crt -X POST -H "Content-Type: application/json" \
  -d '{"ids":["id1","id2"],"apikey":"ih-..."}' \
  https://192.168.0.70:5569/api/tts/history/batch-download -o tts.zip
```

记录字段：`id`、`created`、`variant`、`text`、`language`、`speaker`、`instruct`、`file`、`duration`（音频时长）、`elapsed`（生成用时）、`size`。

### 状态与手动加载

- `GET /api/tts/status`：`loading/ready/error`、active/pending、三个模型状态、图像侧状态
- `GET /api/tts/voices`：预设音色和支持语言
- `POST /api/tts/load?apikey=...` / `POST /api/tts/unload?apikey=...`：手动加载/释放

### TTS 注意事项

- 文本 ≤2000 字；参考音频 ≤60 秒 / ≤20MB
- 生成记录写入 TTS 专用语音库（`tts_audio` 表 + `output/tts/`），不写图像 `jobs`、不进图库
- 图像与 TTS 不共存：图像请求会停 TTS 三个模型，TTS 请求会停图像服务
- 首次加载约 4 秒；状态和排队可通过 `/api/tts/status` 查看
