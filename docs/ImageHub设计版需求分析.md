# ImageHub 设计版 · 需求分析与规划

> 日期：2026-11-21　版本目标：2.0（设计工作台 + ControlNet + 管理后台）
> 约束前提：3.6（vLLM，GPU0/1）必须常驻可用；图像服务只占 GPU2/3；单模型最大 2048，超过需 VAE tiling。

---

## 1. 现状诊断

| 问题 | 具体表现 |
|---|---|
| 功能混杂 | 文生图、多模型对比、图片编辑、历史、API 文档、GPU 功耗、运行设置、压测全部挤在一个页面 |
| 角色不分 | 设计师用的功能与管理员运维功能（后端切换/锁频/压测）混排，普通用户可能误操作 |
| 能力缺失 | 没有 ControlNet，无法锁定白模/线稿结构；没有设计领域工作流（白模→效果图、材质替换、多角度、批量方案） |
| API 零散 | `/v1/images/generations`、`/api/edit`、`/api/compare` 三套语义，缺少统一的异步任务模型与能力声明 |
| 无鉴权分层 | `/image/*`、`/api/*` 全部放行；API 没有 key、没有配额 |
| 无项目概念 | 历史只是平铺列表，设计项目无法归档与复现 |

---

## 2. 角色与目标

| 角色 | 诉求 |
|---|---|
| **设计师**（主要用户） | 概念出图、白模/线稿转效果图、局部改图、批量方案、图板；操作简单、结果可复现 |
| **管理员** | 后端切换、模型启停、显存/锁频/功耗、压测、日志、API Key 与配额、审计 |
| **第三方系统** | 稳定的异步 API、能力声明、错误码规范、鉴权 |

设计原则：
1. **一页一职责**：工作台 / 对比 / 图库 / API / 管理 分开。
2. **配置与使用分离**：所有配置（后端、门控、锁频、功耗、压测、Key）只在管理区，仅管理员可见。
3. **能力显式化**：每个模型声明自己支持 t2i / edit / controlnet / inpaint / 最大分辨率。
4. **可复现**：每个任务记录 prompt、seed、尺寸、模型、ControlNet 参数、参考图。

---

## 3. 信息架构（新界面规划）

```
┌──────────────────────────────────────────────────────────────┐
│ ImageHub  设计工作台 | 对比实验 | 图库 | API        [项目▾]  │
│                                     [后端: 直连] [👤 管理员] │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  工作台（左侧工具，右侧画布）                                 │
│  ┌────────────┬──────────────────────────────────────────┐   │
│  │ ○ 文生图    │ 提示词 / 参考图 / 控制图（ControlNet）     │   │
│  │ ○ 白模→效果图│                                           │   │
│  │ ○ 线稿→效果图│            [生成]  [批量]  [对比]         │   │
│  │ ○ 图片编辑   │                                           │   │
│  │ ○ 局部重绘   │  结果区（缩略图网格，点击看大图）          │   │
│  │ ○ 图板生成   │                                           │   │
│  └────────────┴──────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────┘
```

| 路由 | 页面 | 可见性 |
|---|---|---|
| `/` | 设计工作台（默认） | 登录用户 |
| `/compare` | 多模型对比实验 | 登录用户 |
| `/library` | 图库（按项目/标签，素材复用） | 登录用户 |
| `/api-docs` | API 文档（含用户自己的 Key） | 登录用户 |
| `/admin` | 管理后台（配置/运维/Key/审计） | **仅 is_admin** |

工作台内的工具（子模式）：
1. 文生图（纯提示词）
2. **白模 → 效果图**（上传白模截图 + 材质/光照/风格）
3. **线稿/SU 截图 → 效果图**（Canny/线稿 ControlNet）
4. 图片编辑（多图参考，Edit-2511）
5. 局部重绘（蒙版 + 指令）
6. 图板生成（中文标注讲解板）

---

## 4. ControlNet 方案

### 4.1 选型

| 方案 | 模型 | 优点 | 落地路线 |
|---|---|---|---|
| **A. Fun-ControlNet-Union**（推荐） | Qwen-Image Fun ControlNet Union（~2.3G） | 一个模型支持 Canny / 线稿 / 深度 / 姿态；社区工作流成熟（ComfyUI 节点 + Lightning LoRA） | ComfyUI 路线先行 |
| B. InstantX ControlNet-Union | ~2.3G | 同样多条件 | diffusers `QwenImageControlNetPipeline` |
| C. 专用单任务 ControlNet | Canny / Depth 各一 | 精度略好 | 不推荐（模型多、切换成本高） |

### 4.2 处理链

```
参考图 → 预处理（Canny / DepthAnythingV2 / 线稿提取）
        → ControlNet（强度 control_strength）
        → Qwen DiT 去噪（8 步 Turbo 或 Lightning 4 步）
        → VAE 解码（高分辨率开 tiling）
```

### 4.3 参数与约束

| 参数 | 默认 | 说明 |
|---|---|---|
| `control_type` | `canny` | canny / lineart / depth / normal |
| `control_strength` | 0.8（线稿）/ 1.0（深度） | 越低越"自由" |
| `size` | 1024 | ControlNet 建议 ≤1536；2048 需实测显存 |
| 模型 | Qwen-Image-2.1-Turbo | 已有 |

### 4.4 资源与风险

- 新增显存：ControlNet ~2.3G + 预处理器（Depth Anything V2 small ~100M，可放 CPU）
- 当前 2048 时 GPU2/3 各约 17G（余 7G）→ ControlNet 下 2048 仍可行，但需实测
- 预处理器放 CPU，避免和 DiT 抢显存

---

## 5. 设计类专用功能（领域封装）

| 功能 | 输入 | 输出 | 依赖 |
|---|---|---|---|
| 白模 → 效果图 | 白模截图 + 材质/光照/风格 | 写实效果图 | Edit-2511 或 ControlNet(depth) |
| 线稿 → 效果图 | 线稿 / SU 截图 | 写实效果图 | ControlNet(canny/lineart) |
| 材质替换 | 原图 + 材质词 + 区域描述 | 替换材质 | Edit-2511 |
| 多角度 / 视角切换 | 原图 + 相机描述 | 同风格新视角 | Edit-2511 |
| 批量方案 | 提示词 + 种子列表（n=4~8） | 网格对比 | 任意模型 |
| 图板生成 | 提示词（含标注要求） | 中文标注讲解板 | h3 / qwen（中文文字强） |
| 项目归档 | 任务自动入库 | 项目内可复现 | 数据模型 |

---

## 6. API 规划（可编程调用）

### 6.1 统一任务模型

所有生成类请求 -> 返回 `job_id`；轮询或 webhook 获取结果。保留现有同步接口兼容旧调用。

```
POST /v1/design/render      # 通用：t2i / ref2img（含可选 control 图）
POST /v1/design/clay2render # 白模→效果图（预设提示词模板 + 材质/光照参数）
POST /v1/design/sketch2render # 线稿→效果图（Canny/Lineart）
POST /v1/design/edit        # 编辑（多图参考）
POST /v1/design/inpaint     # 局部重绘（image + mask + prompt）
POST /v1/design/batch       # 批量方案（n 个种子）
GET  /v1/jobs/{id}          # 状态/结果
POST /v1/jobs/{id}/cancel
GET  /v1/models              # 含 capabilities、default_size、max_size
```

### 6.2 能力声明示例

```json
{"id":"qwen","capabilities":["t2i","edit","controlnet","batch"],
 "default_size":[1024,1024],"max_size":2048,
 "control_types":["canny","lineart","depth"]}
```

### 6.3 鉴权与配额

- API Key：管理员在 `/admin` 生成，库中只存哈希；请求头 `Authorization: Bearer <key>`
- 每 Key：QPS、每日张数、可用模型、可用能力（如只读/只出图）
- 错误码：400 参数/能力不符、401 未鉴权、429 超配额、503 后端不可用

---

## 7. 管理后台（仅管理员）

| 区块 | 内容 |
|---|---|
| 后端 | 直连 / ComfyUI 切换；当前存活服务；启动预加载模型 |
| 模型 | 启用/禁用、最大分辨率、VAE tiling、显存上限 |
| 稳定性 | vLLM 空闲门控、GPU 锁频、功耗（保留但不建议启用） |
| 压测 | 3.6 对话压测（并发/时长/实时 ok-err） |
| 任务 | 队列长度、进行中任务、失败重试、取消 |
| API Key | 生成/禁用/配额/最近使用 |
| 审计 | 配置变更记录（谁、何时、改了什么） |
| 日志 | 各服务日志入口（qwen/moss/edit/h3/comfy） |

权限：复用 yz-login 的 `is_admin` 字段；`/admin` 路由与所有 `/api/admin/*` 双端校验。

---

## 8. 数据模型（SQLite 扩展）

```sql
projects(id, name, owner, created)
assets(id, project_id, kind, path, meta, created)          -- kind: image/mask/control/result
jobs(id, project_id, kind, model, prompt, params_json,
     seed, size, status, result_json, created, finished)
api_keys(id, name, key_hash, owner, quota_json, enabled, last_used)
audit(id, actor, action, detail, created)
settings(key, value)  -- 后端/门控/锁频/预加载等
```

---

## 9. 非功能需求

| 维度 | 要求 |
|---|---|
| 稳定性 | 3.6 有流量时禁止/延后模型切换（门控）；ControlNet 预处理放 CPU |
| 性能 | qwen 1024 ≤12s、1536 ≤30s、2048 ≤60s；edit 10 步 ≤90s；ControlNet 1024 ≤35s |
| 并发 | 单模型并行、跨模型排队；队列上限 32，超时 30 分钟 |
| 数据 | 任务全参数落库，可复现；素材按项目隔离 |
| 安全 | API Key 哈希存储、按 Key 限流；`/image` 与素材增加鉴权（可选白名单） |
| 可观测 | 结构化日志、任务耗时、GPU 峰值采样（沿用 5 秒采样器） |

---

## 10. 分期实施

| 阶段 | 内容 | 预估 |
|---|---|---|
| **P0** | 界面拆分（工作台/对比/图库/API/管理）、管理员路由与鉴权、配置迁到管理页、API Key 基础 | 1~2 天 |
| **P1** | ControlNet（ComfyUI 路线：Fun-ControlNet-Union）、白模→效果图、线稿→效果图、异步 job API | 3~5 天 |
| **P2** | 批量方案、项目/素材库、配额与审计、图板模板 | ~1 周 |
| **P3** | 扩图/局部重绘/风格材质库（已完成）；直连 ControlNet **不做**（统一走 ComfyUI） | 已完成 |

---

## 11. 风险与待确认

| 风险 | 应对 |
|---|---|
| ControlNet + 2048 显存不足 | 先实测；不足则 ControlNet 限 1536 或深度图降分辨率 |
| ComfyUI 与 3.6 并发冻结 | 已知问题：切换/加载时段门控；优先直连路线 |
| SSO `is_admin` 字段可用性 | 需确认 yz-login 返回字段；不可用则本地维护管理员表 |
| 旧接口兼容 | 保留 `/v1/images/generations` 与 `backend` 参数，新接口并存 |

**待你确认**：
1. 管理员判定用 yz-login 的 `is_admin`，还是本地白名单？
2. ControlNet 先走 ComfyUI（快，但稳定性风险）还是等直连 diffusers 适配（稳，但工作量大）？
3. 是否需要项目/素材库（P2），还是先做 P0+P1？
