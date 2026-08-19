import io
import os
import time
import uuid
from typing import Optional

import numpy as np
import soundfile as sf
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from transformers import AutoProcessor, DiaForConditionalGeneration

MODEL_ID = os.getenv("MODEL_ID", "nari-labs/Dia-1.6B-0626")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32
SAMPLE_RATE = 44100

app = FastAPI(title="Dia TTS API", version="1.1.0")
processor = None
model = None
model_load_ms = None


class TTSRequest(BaseModel):
    text: str = Field(..., min_length=1)
    max_new_tokens: int = Field(default=1024, ge=64, le=4096)
    seed: Optional[int] = None


def sync_cuda():
    if DEVICE == "cuda":
        torch.cuda.synchronize()


@app.on_event("startup")
def load_model():
    global processor, model, model_load_ms

    print(f"[startup] Loading {MODEL_ID} on {DEVICE} with dtype={DTYPE}")
    started = time.perf_counter_ns()

    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = DiaForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=DTYPE,
        low_cpu_mem_usage=True,
    ).to(DEVICE)
    model.eval()
    sync_cuda()

    model_load_ms = (time.perf_counter_ns() - started) / 1_000_000
    print(f"[startup] Model loaded in {model_load_ms:.2f} ms")


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": MODEL_ID,
        "device": DEVICE,
        "cuda_available": torch.cuda.is_available(),
        "model_load_ms": model_load_ms,
    }


@app.post("/tts")
def tts(req: TTSRequest):
    if model is None or processor is None:
        raise HTTPException(status_code=503, detail="Model is not loaded yet")

    request_id = str(uuid.uuid4())
    server_start = time.perf_counter_ns()

    if req.seed is not None:
        torch.manual_seed(req.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(req.seed)

    try:
        preprocess_start = time.perf_counter_ns()
        inputs = processor(text=[req.text], padding=True, return_tensors="pt").to(DEVICE)
        sync_cuda()
        preprocess_ms = (time.perf_counter_ns() - preprocess_start) / 1_000_000

        inference_start = time.perf_counter_ns()
        with torch.inference_mode():
            outputs = model.generate(**inputs, max_new_tokens=req.max_new_tokens)
        sync_cuda()
        inference_ms = (time.perf_counter_ns() - inference_start) / 1_000_000

        decode_start = time.perf_counter_ns()
        decoded = processor.batch_decode(outputs)
        audio = decoded[0] if isinstance(decoded, (list, tuple)) else decoded
        if hasattr(audio, "detach"):
            audio = audio.detach().float().cpu().numpy()
        audio = np.asarray(audio)
        while audio.ndim > 1 and len(audio) == 1:
            audio = audio[0]
        audio = np.squeeze(audio).astype(np.float32, copy=False)
        decode_ms = (time.perf_counter_ns() - decode_start) / 1_000_000

        audio_duration_s = len(audio) / SAMPLE_RATE

        encode_start = time.perf_counter_ns()
        buffer = io.BytesIO()
        sf.write(buffer, audio, SAMPLE_RATE, format="WAV", subtype="PCM_16")
        buffer.seek(0)
        encode_ms = (time.perf_counter_ns() - encode_start) / 1_000_000

        server_total_ms = (time.perf_counter_ns() - server_start) / 1_000_000
        rtf = (inference_ms / 1000.0) / audio_duration_s if audio_duration_s > 0 else 0.0

        headers = {
            "X-Request-ID": request_id,
            "X-Preprocess-Time-MS": f"{preprocess_ms:.2f}",
            "X-Inference-Time-MS": f"{inference_ms:.2f}",
            "X-Decode-Time-MS": f"{decode_ms:.2f}",
            "X-Encoding-Time-MS": f"{encode_ms:.2f}",
            "X-Server-Total-MS": f"{server_total_ms:.2f}",
            "X-Audio-Duration-S": f"{audio_duration_s:.3f}",
            "X-RTF": f"{rtf:.4f}",
            "X-Sample-Rate": str(SAMPLE_RATE),
        }

        return StreamingResponse(buffer, media_type="audio/wav", headers=headers)

    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"TTS generation failed: {exc}") from exc
