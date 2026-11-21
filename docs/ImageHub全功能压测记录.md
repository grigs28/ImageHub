# ImageHub 全功能组合压测记录

> 日期：2026-11-21　环境：3.6 在 GPU2/3（四并发满载），图像服务在 GPU0/1
> 脚本：`/opt/vLLM/full_stress.py`（本地 `docs/脚本/full_stress.py`）
> 日志：`/var/log/full_stress.log`、`/var/log/full_stress_gpu.log`（每 2 秒采样）

## 1. 覆盖范围

同一轮压测内依次覆盖：4 个模型的 ComfyUI 文生图（每次都是模型切换）、直连文生图（触发后端切换）、
ControlNet、白模→效果图、线稿→效果图、多角度、批量、局部重绘、扩图、异步任务、图片编辑、多模型对比。

## 2. 结果（3.6 全程四并发满载）

| # | 步骤 | 耗时 | 结果 |
|---|---|---|---|
| 1 | comfy t2i zimage | 28.1s | ✅ |
| 2 | comfy t2i qwen | 23.1s | ✅ |
| 3 | comfy t2i h3 | 23.1s | ✅ |
| 4 | comfy t2i moss | 59.1s | ✅ |
| 5 | direct t2i zimage | 31.8s | ✅ |
| 6 | direct t2i qwen | 26.9s | ✅ |
| 7 | ControlNet render | 36.2s | ✅ |
| 8 | clay2render（白模） | 14.0s | ✅ |
| 9 | sketch2render（线稿） | 12.2s | ✅ |
| 10 | views ×2（多角度） | 24.2s | ✅ |
| 11 | batch n=2 | 20.1s | ✅ |
| 12 | inpaint（蒙版） | 12.2s | ✅ |
| 13 | outpaint（+256） | 31.0s | ✅ 1536² |
| 14 | async job | 16.0s | ✅ |
| 15 | edit（12 步） | 101.1s | ✅（修复后复测） |
| 16 | compare ×3（h3/qwen/zimage） | 75.0s | ✅ |

3.6 侧：全程 `running=4, waiting=4`，0 错误；机器连续运行 14 小时 40 分，**无重启**。

## 3. GPU 峰值（压测期间）

| GPU | 峰值显存 | 利用率 | 温度 | 功耗 |
|---|---|---|---|---|
| GPU0 | 19.2G | 100% | 81°C | 350W（卡上限） |
| GPU1 | 15.4G | 100% | 69°C | 331W |
| GPU2（3.6） | 20.3G | 98% | 59°C | 296W |
| GPU3（3.6） | 20.3G | 98% | 66°C | 310W |

## 4. 发现并修复的问题

**图片编辑（/api/edit）失败**：`提交失败: HTTP Error 400`。

- 原因：编辑工作流引用的 `qwen_2.5_vl_7b_bf16.safetensors` 不存在（ComfyUI 只有 fp8 版，3090 无法计算）
- 修复：把已下载的 bf16 文件链接到 ComfyUI：
  ```bash
  ln -sf /opt/models/Comfy-Org/Qwen-Image_ComfyUI/split_files/text_encoders/qwen_2.5_vl_7b.safetensors \
         /opt/ComfyUI/models/text_encoders/qwen_2.5_vl_7b_bf16.safetensors
  systemctl restart comfyui-h3
  ```
- 复测：12 步编辑 101 秒成功

## 5. 结论

- 新布局（3.6→GPU2/3、图像→GPU0/1）在「4 并发对话 + 全功能图像轮转 + 反复模型加载/后端互切」下**稳定，无崩溃**
- 不支持「图像代码绕过可见设备操作 0/1」的假设：现在图像就在 0/1，同样零异常
- 现有风险控制有效：模型常驻、同底座不卸载、加载前确认
