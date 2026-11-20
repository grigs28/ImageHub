#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Qwen-Image-2.1-Turbo 文生图服务（OpenAI 兼容 /v1/images/generations）
模型：Qwen/Qwen-Image-2.1-Turbo（7B DiT + Qwen3-VL-8B 编码器，8 步蒸馏）
24GB 卡策略：DiT+VAE 常驻 GPU，文本编码器 CPU 卸载
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

MODEL_DIR = os.environ.get("QIMAGE_MODEL_DIR", "/opt/models/Qwen/Qwen-Image-2.1-Turbo")
OUTPUT_DIR = os.environ.get("QIMAGE_OUTPUT_DIR", "/opt/models/Qwen/Qwen-Image-2.1-Turbo/outputs")
SERVED_NAME = os.environ.get("QIMAGE_SERVED_NAME", "qwen-image-2.1-turbo")
os.makedirs(OUTPUT_DIR, exist_ok=True)

app = FastAPI(title="Qwen-Image-2.1-Turbo Service")
_pipe = None


_mode = "unknown"


def get_pipe():
    global _pipe, _mode
    if _pipe is None:
        from diffusers import QwenImage21Pipeline
        logger.info(f"加载模型: {MODEL_DIR}")
        t0 = time.time()
        # 双卡分组件模式：编码器 cuda:0，DiT+VAE cuda:1（官方 device_map 方案）
        if os.environ.get("QIMAGE_SPLIT", "1") == "1":
            pipe = QwenImage21Pipeline.from_pretrained(
                MODEL_DIR, dtype=torch.bfloat16,
                device_map="balanced",
                max_memory={0: "23GiB", 1: "23GiB"},
            )
            logger.info(f"双卡分组件: {pipe.hf_device_map}")
            _mode = "split-2gpu"
            try:
                pipe.vae.enable_slicing()
            except Exception:
                pass
            _pipe = pipe
            logger.info(f"模型加载完成，耗时 {time.time()-t0:.1f}s，模式={_mode}")
            return _pipe

        quant = os.environ.get("QIMAGE_ENCODER_QUANT", "int4")
        pipe = None
        if quant != "none":
            try:
                from transformers import Qwen3VLForConditionalGeneration, BitsAndBytesConfig
                if quant == "int4":
                    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                             bnb_4bit_compute_dtype=torch.bfloat16)
                else:
                    bnb = BitsAndBytesConfig(load_in_8bit=True)
                te = Qwen3VLForConditionalGeneration.from_pretrained(
                    os.path.join(MODEL_DIR, "text_encoder"),
                    quantization_config=bnb, device_map="cpu", dtype=torch.bfloat16)
                pipe = QwenImage21Pipeline.from_pretrained(
                    MODEL_DIR, text_encoder=te, dtype=torch.bfloat16)
                pipe.transformer.to("cuda")
                pipe.vae.to("cuda")
                pipe.text_encoder.to("cuda")
                torch.cuda.empty_cache()
                free, total = torch.cuda.mem_get_info()
                logger.info(f"int8 编码器模式，显存剩余 {free/1024**3:.1f}/{total/1024**3:.1f} GiB")
                _mode = "int8-resident"
            except Exception as e:
                logger.warning(f"int8 编码器方案失败，回退 CPU 卸载: {e}")
                pipe = None
        if pipe is None:
            pipe = QwenImage21Pipeline.from_pretrained(MODEL_DIR, dtype=torch.bfloat16)
            pipe.enable_model_cpu_offload()
            _mode = "cpu-offload"
        try:
            pipe.vae.enable_slicing()
        except Exception:
            pass
        _pipe = pipe
        logger.info(f"模型加载完成，耗时 {time.time()-t0:.1f}s，模式={_mode}")
    return _pipe


class ImageRequest(BaseModel):
    prompt: str
    model: Optional[str] = None
    n: int = 1
    size: str = "1024x1024"
    response_format: str = "b64_json"
    num_inference_steps: Optional[int] = None   # None = 用 checkpoint 内置 8 步调度
    seed: Optional[int] = None
    negative_prompt: Optional[str] = None
    use_kv_cache: bool = True


@app.on_event("startup")
def _startup():
    get_pipe()


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
            gen = torch.Generator("cpu").manual_seed(req.seed + i)
        t0 = time.time()
        kwargs = dict(prompt=req.prompt, width=w, height=h,
                      use_kv_cache=req.use_kv_cache, generator=gen)
        if req.num_inference_steps:
            kwargs["num_inference_steps"] = req.num_inference_steps
        if req.negative_prompt:
            kwargs["negative_prompt"] = req.negative_prompt
        with torch.inference_mode():
            image = pipe(**kwargs).images[0]
        cost = time.time() - t0
        logger.info(f"生成完成 {w}x{h} 耗时 {cost:.2f}s")
        if req.response_format == "url":
            name = f"{uuid.uuid4().hex}.png"
            image.save(os.path.join(OUTPUT_DIR, name))
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
