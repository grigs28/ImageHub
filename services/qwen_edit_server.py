#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Qwen-Image-Edit-2511 直连服务（OpenAI 兼容 /v1/images/generations）
DiT 用本地 GGUF（Q6_K 约 17G），文本编码器/VAE/调度器来自 diffusers 仓库。
双卡：DiT+VAE → cuda:0，文本编码器 → cuda:1
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
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

MODEL_DIR = os.environ.get("EDIT_MODEL_DIR", "/opt/models/Qwen/Qwen-Image-Edit-2511")
GGUF_PATH = os.environ.get("EDIT_GGUF", "/opt/models/Comfy-Org/Qwen-Image-Edit-2511-gguf/qwen-image-edit-2511-Q6_K.gguf")
OUTPUT_DIR = os.environ.get("EDIT_OUTPUT_DIR", "/opt/models/Qwen/Qwen-Image-Edit-2511/outputs")
SERVED_NAME = os.environ.get("EDIT_SERVED_NAME", "qwen-image-edit-2511")
os.makedirs(OUTPUT_DIR, exist_ok=True)

app = FastAPI(title="Qwen-Image-Edit-2511 Service")
_pipe = None


def get_pipe():
    global _pipe
    if _pipe is None:
        from diffusers import (QwenImageEditPlusPipeline, QwenImageTransformer2DModel,
                               GGUFQuantizationConfig)
        t0 = time.time()
        logger.info(f"加载 GGUF DiT: {GGUF_PATH}")
        transformer = QwenImageTransformer2DModel.from_single_file(
            GGUF_PATH,
            config=os.path.join(MODEL_DIR, "transformer"),
            quantization_config=GGUFQuantizationConfig(compute_dtype=torch.bfloat16),
            torch_dtype=torch.bfloat16,
        )
        logger.info("加载 pipeline 组件（text_encoder/vae/tokenizer/scheduler）")
        pipe = QwenImageEditPlusPipeline.from_pretrained(
            MODEL_DIR, transformer=transformer, torch_dtype=torch.bfloat16)
        mode = os.environ.get("EDIT_MODE", "offload")
        if mode == "split":
            # 双卡：DiT+VAE → cuda:0，text_encoder → cuda:1
            # 关键：把 _execution_device 固定到 DiT 所在卡，否则输入会被送到 TE 的卡上
            pipe.transformer.to("cuda:0")
            pipe.vae.to("cuda:0")
            pipe.text_encoder.to("cuda:1")
            QwenImageEditPlusPipeline._execution_device = property(
                lambda self: torch.device("cuda:0"))
            # prompt 编码必须在 text_encoder 自己所在的卡上进行（diffusers #12066）
            _orig_enc = QwenImageEditPlusPipeline.encode_prompt

            def _enc_patched(self, *args, **kwargs):
                if getattr(self, "text_encoder", None) is not None:
                    kwargs["device"] = self.text_encoder.device
                out = _orig_enc(self, *args, **kwargs)
                # 编码结果搬回 DiT 所在卡
                tgt = torch.device("cuda:0")
                if isinstance(out, tuple):
                    return tuple(x.to(tgt) if torch.is_tensor(x) else x for x in out)
                return out.to(tgt) if torch.is_tensor(out) else out

            QwenImageEditPlusPipeline.encode_prompt = _enc_patched
            logger.info("模式: 双卡 split (DiT+VAE=cuda:0, TE=cuda:1)")
        elif os.environ.get("EDIT_OFFLOAD", "1") == "1":
            # 官方推荐：单卡 + 模块级 CPU offload（TE 与 DiT 不会同时在显存）
            pipe.enable_model_cpu_offload()
            logger.info("模式: 单卡 + CPU offload")
        else:
            pipe.transformer.to("cuda:0")
            pipe.vae.to("cuda:0")
            pipe.text_encoder.to("cuda:0")
            logger.info("模式: 单卡全驻留")
        try:
            pipe.vae.enable_slicing()
        except Exception:
            pass
        _pipe = pipe
        logger.info(f"加载完成，耗时 {time.time()-t0:.1f}s")
    return _pipe


class EditRequest(BaseModel):
    prompt: str
    image: Optional[str] = None          # base64（可带 data:image/... 前缀）
    image_path: Optional[str] = None     # 服务器本地路径
    model: Optional[str] = None
    size: str = "1024x1024"
    num_inference_steps: int = 20
    true_cfg_scale: float = 4.0
    guidance_scale: float = 1.0
    negative_prompt: str = " "
    seed: Optional[int] = None
    response_format: str = "b64_json"


def _load_image(req: EditRequest) -> Image.Image:
    if req.image:
        b64 = req.image.split(",", 1)[-1] if req.image.startswith("data:") else req.image
        return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
    if req.image_path:
        return Image.open(req.image_path).convert("RGB")
    raise HTTPException(400, "需要 image(base64) 或 image_path")


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
def generate(req: EditRequest):
    try:
        w, h = [int(x) for x in req.size.lower().split("x")]
    except Exception:
        raise HTTPException(400, "size 格式应为 1024x1024")
    pipe = get_pipe()
    src = _load_image(req)
    # 按目标长边缩放，保持比例
    scale = min(w / src.width, h / src.height)
    if scale < 1:
        src = src.resize((max(16, int(src.width * scale)), max(16, int(src.height * scale))), Image.LANCZOS)
    gen = None
    if req.seed is not None:
        gen = torch.Generator("cpu").manual_seed(req.seed)
    t0 = time.time()
    with torch.inference_mode():
        out = pipe(
            image=[src],
            prompt=req.prompt,
            negative_prompt=req.negative_prompt,
            num_inference_steps=req.num_inference_steps,
            true_cfg_scale=req.true_cfg_scale,
            guidance_scale=req.guidance_scale,
            generator=gen,
        ).images[0]
    cost = time.time() - t0
    logger.info(f"编辑完成 {src.width}x{src.height} 耗时 {cost:.2f}s")
    if req.response_format == "url":
        name = f"{uuid.uuid4().hex}.png"
        out.save(os.path.join(OUTPUT_DIR, name))
        return JSONResponse({"created": int(time.time()),
                             "data": [{"url": f"/files/{name}", "revised_prompt": req.prompt}]})
    buf = io.BytesIO()
    out.save(buf, format="PNG")
    return JSONResponse({"created": int(time.time()),
                         "data": [{"b64_json": base64.b64encode(buf.getvalue()).decode(),
                                   "revised_prompt": req.prompt}]})


@app.get("/files/{name}")
def files(name: str):
    path = os.path.join(OUTPUT_DIR, os.path.basename(name))
    if not os.path.isfile(path):
        raise HTTPException(404, "not found")
    return FileResponse(path, media_type="image/png")
