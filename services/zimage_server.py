#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Z-Image-Turbo 文生图服务（OpenAI 兼容 /v1/images/generations）
模型：Tongyi-MAI/Z-Image-Turbo（6B DiT + Qwen3-4B 文本编码器，Apache 2.0）
运行：uvicorn 由外层包装脚本启动
"""
import base64
import io
import os
import time
import uuid
import logging

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
from typing import Optional

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

MODEL_DIR = os.environ.get("ZIMAGE_MODEL_DIR", "/opt/models/Tongyi-MAI/Z-Image-Turbo")
OUTPUT_DIR = os.environ.get("ZIMAGE_OUTPUT_DIR", "/opt/models/Tongyi-MAI/Z-Image-Turbo/outputs")
SERVED_NAME = os.environ.get("ZIMAGE_SERVED_NAME", "z-image-turbo")
os.makedirs(OUTPUT_DIR, exist_ok=True)

app = FastAPI(title="Z-Image-Turbo Service")
_pipe = None


def get_pipe():
    global _pipe
    if _pipe is None:
        from diffusers import ZImagePipeline
        logger.info(f"加载模型: {MODEL_DIR}")
        t0 = time.time()
        if os.environ.get("ZIMAGE_BALANCED", "0") == "1":
            pipe = ZImagePipeline.from_pretrained(
                MODEL_DIR, torch_dtype=torch.bfloat16,
                device_map="balanced", max_memory={0: "22GiB", 1: "22GiB"},
            )
            logger.info(f"balanced 分卡: {getattr(pipe, 'hf_device_map', None)}")
        else:
            pipe = ZImagePipeline.from_pretrained(
                MODEL_DIR, torch_dtype=torch.bfloat16, low_cpu_mem_usage=False,
            )
        # 双卡：DiT+VAE 在 cuda:0，文本编码器在 cuda:1（更省时，且给高分辨率留余量）
        if os.environ.get("ZIMAGE_BALANCED", "0") == "1":
            pass
        elif os.environ.get("ZIMAGE_SPLIT", "0") == "1":
            pipe.transformer.to("cuda:0")
            pipe.vae.to("cuda:0")
            pipe.text_encoder.to("cuda:1")
            try:
                from diffusers import ZImagePipeline as _ZP
                _ZP._execution_device = property(lambda self: torch.device("cuda:0"))
                # 有些 diffusers pipeline 用 self.device 决定 latents 设备，这里一并固定到 DiT 卡
                _ZP.device = property(lambda self: torch.device("cuda:0"))
                _orig_enc = _ZP.encode_prompt

                def _enc(self, *a, **kw):
                    kw["device"] = self.text_encoder.device
                    out = _orig_enc(self, *a, **kw)
                    tgt = torch.device("cuda:0")
                    if isinstance(out, tuple):
                        return tuple(x.to(tgt) if torch.is_tensor(x) else x for x in out)
                    return out.to(tgt) if torch.is_tensor(out) else out
                _ZP.encode_prompt = _enc
                logger.info("模式: 双卡 split（DiT+VAE=cuda:0, TE=cuda:1）")
            except Exception as e:
                logger.warning(f"双卡补丁失败，回退单卡: {e}")
                pipe.text_encoder.to("cuda:0")
        else:
            # 24GB 卡 + 30.6GB 权重 → 分层 CPU 卸载（文本编码器用完即卸）
            pipe.enable_model_cpu_offload()
        try:
            pipe.transformer.set_attention_backend("flash")
        except Exception as e:
            logger.warning(f"flash attention 不可用，用默认 SDPA: {e}")
        try:
            pipe.vae.enable_slicing()
            if os.environ.get("ZIMAGE_SPLIT", "0") == "1" or os.environ.get("ZIMAGE_BALANCED", "0") == "1":
                pipe.vae.enable_tiling()
        except Exception:
            pass
        _pipe = pipe
        logger.info(f"模型加载完成，耗时 {time.time()-t0:.1f}s")
    return _pipe


class ImageRequest(BaseModel):
    prompt: str
    model: Optional[str] = None
    n: int = 1
    size: str = "1024x1024"
    response_format: str = "b64_json"   # b64_json | url
    num_inference_steps: int = 8        # Turbo 蒸馏模型 8 步（4~9 可用）
    seed: Optional[int] = None
    negative_prompt: Optional[str] = None


@app.on_event("startup")
def _startup():
    get_pipe()   # 启动即加载，避免首个请求超时


@app.get("/health")
def health():
    return {"status": "ok", "model": SERVED_NAME, "loaded": _pipe is not None}


@app.get("/v1/models")
def models():
    return {"object": "list", "data": [{"id": SERVED_NAME, "object": "model", "owned_by": "local"}]}


@app.post("/v1/images/generations")
def generate(req: ImageRequest):
    try:
        w, h = [int(x) for x in req.size.lower().split("x")]
    except Exception:
        raise HTTPException(400, "size 格式应为 1024x1024")
    pipe = get_pipe()
    data = []
    for i in range(max(1, min(req.n, 4))):
        gen = None
        if req.seed is not None:
            gen = torch.Generator(device="cpu").manual_seed(req.seed + i)
        t0 = time.time()
        kwargs = dict(prompt=req.prompt, width=w, height=h,
                      num_inference_steps=req.num_inference_steps, generator=gen)
        if req.negative_prompt:
            kwargs["negative_prompt"] = req.negative_prompt
        with torch.inference_mode():
            image = pipe(**kwargs).images[0]
        cost = time.time() - t0
        logger.info(f"生成完成 {w}x{h} steps={req.num_inference_steps} 耗时 {cost:.2f}s")
        if req.response_format == "url":
            name = f"{uuid.uuid4().hex}.png"
            path = os.path.join(OUTPUT_DIR, name)
            image.save(path)
            data.append({"url": f"/files/{name}", "revised_prompt": req.prompt})
        else:
            buf = io.BytesIO()
            image.save(buf, format="PNG")
            data.append({"b64_json": base64.b64encode(buf.getvalue()).decode(),
                         "revised_prompt": req.prompt})
    return JSONResponse({"created": int(time.time()), "data": data,
                         "usage": {"images": len(data)}})


@app.get("/files/{name}")
def files(name: str):
    path = os.path.join(OUTPUT_DIR, os.path.basename(name))
    if not os.path.isfile(path):
        raise HTTPException(404, "not found")
    return FileResponse(path, media_type="image/png")
