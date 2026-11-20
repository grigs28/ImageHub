#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MOSS-Image-SFT 文生图服务（OpenAI 兼容 /v1/images/generations）
模型：OpenMOSS/MOSS-Image-SFT（5.1B DiT + Qwen3.5-9B 编码器，Apache 2.0）
双卡分组件：编码器 cuda:0，DiT+VAE cuda:1
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

MODEL_DIR = os.environ.get("MOSS_MODEL_DIR", "/opt/models/OpenMOSS/MOSS-Image-SFT")
OUTPUT_DIR = os.environ.get("MOSS_OUTPUT_DIR", "/opt/models/OpenMOSS/MOSS-Image-SFT/outputs")
SERVED_NAME = os.environ.get("MOSS_SERVED_NAME", "moss-image")
os.makedirs(OUTPUT_DIR, exist_ok=True)
app = FastAPI(title="MOSS-Image Service")
_pipe = None


def get_pipe():
    global _pipe
    if _pipe is None:
        import sys
        sys.path.insert(0, MODEL_DIR)
        from pipeline_moss_image import MOSSImagePipeline
        logger.info(f"加载模型: {MODEL_DIR}")
        t0 = time.time()
        pipe = MOSSImagePipeline.from_pretrained(
            MODEL_DIR, dtype=torch.bfloat16, trust_remote_code=True,
        )
        if os.environ.get("MOSS_SPLIT", "0") == "1":
            pipe.text_encoder.to("cuda:0")
            pipe.transformer.to("cuda:1")
            pipe.vae.to("cuda:1")
            type(pipe)._execution_device = property(lambda self: torch.device("cuda:1"))
        else:
            # 自定义管线不支持跨设备，单卡 + CPU 卸载
            pipe.enable_model_cpu_offload()
        torch.cuda.empty_cache()
        logger.info(f"分卡: {getattr(pipe, 'hf_device_map', None)}")
        try:
            pipe.vae.enable_slicing()
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
    response_format: str = "b64_json"
    num_inference_steps: int = 50
    guidance_scale: float = 3.0
    seed: Optional[int] = None
    negative_prompt: str = ""


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
        kwargs = dict(prompt=req.prompt, height=h, width=w,
                      num_inference_steps=req.num_inference_steps,
                      guidance_scale=req.guidance_scale, generator=gen)
        if req.negative_prompt:
            kwargs["negative_prompt"] = req.negative_prompt
        with torch.inference_mode():
            image = pipe(**kwargs).images[0]
        cost = time.time() - t0
        logger.info(f"生成完成 {w}x{h} steps={req.num_inference_steps} 耗时 {cost:.2f}s")
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
