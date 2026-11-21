# 70 服务器 GPU 归属调整（3.6 与图像服务互换）

> 日期：2026-11-21
> 结论：**3.6 固定在 GPU2/3，图像服务（ComfyUI + 直连）固定在 GPU0/1**；服务名、端口、启动参数全部不变。

## 1. 调整内容

| 服务 | 调整前 | 调整后 |
|---|---|---|
| Qwen3.6（tclf90_Qwen3.6-35B-A3B-AWQ.service，5563） | GPU0/1 | **GPU2/3** |
| ComfyUI（comfyui-h3.service，5568） | GPU2/3 | **GPU0/1** |
| 直连 qwen / moss / edit | GPU2/3 | **GPU0/1** |
| 直连 zimage | GPU2（单卡，CPU 卸载） | **GPU0/1（balanced 分卡）** |
| h3（sd-server，5568） | GPU2/3 | GPU0/1 |

服务名、端口、模型路径、其余启动参数均未改动。

## 2. 依据

- 实测两对卡推理速度基本一致（并发 1/2/4/8 吞吐差异 ≤2%），换卡**不影响 3.6 性能**
- GPU0 为 PCIe **x8**（其余 x4），图像侧大模型加载走更宽链路
- 供电分布：3.6 独占电源 B（GPU2/3），图像 + CPU 在电源 A（GPU0/1），均在 1000W 内

## 3. 改动文件（回退即反向修改）

| 文件 | 改动 |
|---|---|
| `/opt/models/tclf90/.../0_Qwen3.6-35B-A3B-AWQ.py` | `env_vars['CUDA_VISIBLE_DEVICES'] = '2,3'`（**真正生效处**，会覆盖 systemd） |
| `/etc/systemd/system/tclf90_...service` | `Environment="CUDA_VISIBLE_DEVICES=2,3"` |
| `/opt/ComfyUI/0_ComfyUI-H3.py` | `GPU_DEVICE = "0,1"` |
| `/etc/systemd/system/comfyui-h3.service` | `Environment="CUDA_VISIBLE_DEVICES=0,1"` |
| `/opt/vLLM/imagehub/app.py` `SERVICES` | 各服务 env 改为 `0,1`；zimage 增加 `ZIMAGE_BALANCED=1` |

> 注意：**只改 systemd 不生效**，3.6 的启动脚本内部硬编码了设备号。

## 4. 回退步骤

```bash
# 1) 把上面四处的 0,1 / 2,3 对调回原值
# 2) 重载并重启
systemctl daemon-reload
systemctl restart tclf90_Qwen3.6-35B-A3B-AWQ.service
systemctl restart comfyui-h3 imagehub
```

## 5. zimage 改为双卡（balanced）

原来的单卡 + CPU 卸载模式速度一般，且生成分辨率偏低。现改为 **accelerate `device_map="balanced"`**：

```
balanced 分卡: {'transformer': 0, 'text_encoder': 1, 'vae': 1}
```

实测（1024 提示词，8 步）：

| 分辨率 | 耗时 | 备注 |
|---|---|---|
| 1024×1024 | **15.8s** | 经 ImageHub 28.5s（含停 ComfyUI + 加载） |
| 1536×1536 | **40.5s** | |
| 2048×2048 | **85.8s** | GPU0 峰值 23.8G，**贴近上限，慎用** |

> 手动 `.to("cuda:1")` 的 split 方式会触发 diffusers 的设备判定 bug（`self.device` 取第一个模块），
> 已放弃；`device_map="balanced"` 由 accelerate 统一管理搬运，稳定可用。

## 6. 验证记录（2026-11-21）

| 链路 | 结果 |
|---|---|
| 3.6（GPU2/3，5563） | ✅ 20.3G×2，服务正常 |
| ComfyUI 文生图（GPU0/1） | ✅ 34.4s（含加载） |
| zimage 直连（GPU0/1，balanced） | ✅ 1024 28.5s（含切换） |
| 设计 ControlNet（GPU0/1） | ✅ 32.3s（含切换），显存 18.3G/14.9G |

## 7. 已知现象

- 直连与 ComfyUI 互切时，若 ComfyUI 显存尚未完全释放就加载直连服务，偶发一次失败；重试即成功。后续可在 `_stop_comfy()` 里增加"等待显存真正释放"的检查。
