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

    print("=" * 80)
    print("DIA TTS STARTUP")
    print("=" * 80)
    print(f"Model             : {MODEL_ID}")
    print(f"Device            : {DEVICE}")
    print(f"PyTorch           : {torch.__version__}")
    print(f"PyTorch CUDA      : {torch.version.cuda}")
    print(f"CUDA available    : {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"GPU               : {torch.cuda.get_device_name(0)}")

    print(f"DTYPE             : {DTYPE}")
    print("=" * 80)

    started = time.perf_counter_ns()

    processor = AutoProcessor.from_pretrained(MODEL_ID)

    model = DiaForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=DTYPE,
        low_cpu_mem_usage=True,
    ).to(DEVICE)

    model.eval()

    sync_cuda()

    model_load_ms = (
        time.perf_counter_ns() - started
    ) / 1_000_000

    print(f"[startup] Model loaded in {model_load_ms:.2f} ms")
    print(f"[startup] Model loaded in {model_load_ms / 1000:.2f} sec")

    if torch.cuda.is_available():
        print(
            f"[startup] GPU allocated: "
            f"{torch.cuda.memory_allocated() / 1024**2:.2f} MB"
        )

        print(
            f"[startup] GPU reserved : "
            f"{torch.cuda.memory_reserved() / 1024**2:.2f} MB"
        )


@app.get("/")
def root():
    return {
        "service": "Dia TTS API",
        "model": MODEL_ID,
        "device": DEVICE,
    }


@app.get("/health")
def health():
    result = {
        "status": "ok",
        "model": MODEL_ID,
        "device": DEVICE,
        "cuda_available": torch.cuda.is_available(),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "model_load_ms": model_load_ms,
    }

    if torch.cuda.is_available():
        result["gpu"] = torch.cuda.get_device_name(0)

    return result


@app.post("/tts")
def tts(req: TTSRequest):
    if model is None or processor is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded yet",
        )

    request_id = str(uuid.uuid4())
    server_start = time.perf_counter_ns()

    if req.seed is not None:
        torch.manual_seed(req.seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(req.seed)

    try:
        # ================================================================
        # Preprocessing
        # ================================================================

        preprocess_start = time.perf_counter_ns()

        inputs = processor(
            text=[req.text],
            padding=True,
            return_tensors="pt",
        )

        inputs = inputs.to(model.device)

        sync_cuda()

        preprocess_ms = (
            time.perf_counter_ns() - preprocess_start
        ) / 1_000_000

        # ================================================================
        # Inference
        # ================================================================

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        inference_start = time.perf_counter_ns()

        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=req.max_new_tokens,
                guidance_scale=3.0,
                temperature=1.8,
                top_p=0.90,
                top_k=45,
            )

        sync_cuda()

        inference_ms = (
            time.perf_counter_ns() - inference_start
        ) / 1_000_000

        # ================================================================
        # Decode
        # ================================================================

        decode_start = time.perf_counter_ns()

        decoded = processor.batch_decode(outputs)

        audio = (
            decoded[0]
            if isinstance(decoded, (list, tuple))
            else decoded
        )

        if torch.is_tensor(audio):
            audio = (
                audio
                .detach()
                .float()
                .cpu()
                .numpy()
            )

        audio = np.asarray(audio)
        audio = np.squeeze(audio)

        if audio.ndim != 1:
            raise RuntimeError(
                f"Unexpected audio shape: {audio.shape}"
            )

        audio = audio.astype(
            np.float32,
            copy=False,
        )

        if len(audio) == 0:
            raise RuntimeError("Generated audio is empty")

        decode_ms = (
            time.perf_counter_ns() - decode_start
        ) / 1_000_000

        # ================================================================
        # Audio duration
        # ================================================================

        audio_duration_s = (
            len(audio) / SAMPLE_RATE
        )

        # ================================================================
        # WAV encode
        # ================================================================

        encode_start = time.perf_counter_ns()

        buffer = io.BytesIO()

        sf.write(
            buffer,
            audio,
            SAMPLE_RATE,
            format="WAV",
            subtype="PCM_16",
        )

        buffer.seek(0)

        encode_ms = (
            time.perf_counter_ns() - encode_start
        ) / 1_000_000

        # ================================================================
        # Metrics
        # ================================================================

        server_total_ms = (
            time.perf_counter_ns() - server_start
        ) / 1_000_000

        generation_rtf = (
            (inference_ms / 1000.0)
            / audio_duration_s
            if audio_duration_s > 0
            else 0.0
        )

        total_rtf = (
            (server_total_ms / 1000.0)
            / audio_duration_s
            if audio_duration_s > 0
            else 0.0
        )

        gpu_allocated_mb = 0.0
        gpu_reserved_mb = 0.0
        gpu_peak_mb = 0.0

        if torch.cuda.is_available():
            gpu_allocated_mb = (
                torch.cuda.memory_allocated()
                / 1024**2
            )

            gpu_reserved_mb = (
                torch.cuda.memory_reserved()
                / 1024**2
            )

            gpu_peak_mb = (
                torch.cuda.max_memory_allocated()
                / 1024**2
            )

        print("")
        print("=" * 80)
        print("SERVER LATENCY")
        print("=" * 80)
        print(f"Request ID          : {request_id}")
        print(f"Preprocess          : {preprocess_ms:.2f} ms")
        print(f"Inference           : {inference_ms:.2f} ms")
        print(f"Decode              : {decode_ms:.2f} ms")
        print(f"WAV encoding        : {encode_ms:.2f} ms")
        print(f"SERVER TOTAL        : {server_total_ms:.2f} ms")
        print(f"Audio duration      : {audio_duration_s:.3f} sec")
        print(f"Generation RTF      : {generation_rtf:.4f}")
        print(f"Total RTF           : {total_rtf:.4f}")

        if torch.cuda.is_available():
            print(f"GPU allocated       : {gpu_allocated_mb:.2f} MB")
            print(f"GPU reserved        : {gpu_reserved_mb:.2f} MB")
            print(f"GPU peak            : {gpu_peak_mb:.2f} MB")

        print("=" * 80)

        headers = {
            "X-Request-ID": request_id,
            "X-Preprocess-Time-MS": f"{preprocess_ms:.2f}",
            "X-Inference-Time-MS": f"{inference_ms:.2f}",
            "X-Decode-Time-MS": f"{decode_ms:.2f}",
            "X-Encoding-Time-MS": f"{encode_ms:.2f}",
            "X-Server-Total-MS": f"{server_total_ms:.2f}",
            "X-Audio-Duration-S": f"{audio_duration_s:.3f}",
            "X-Generation-RTF": f"{generation_rtf:.4f}",
            "X-RTF": f"{total_rtf:.4f}",
            "X-Sample-Rate": str(SAMPLE_RATE),
            "X-GPU-Allocated-MB": f"{gpu_allocated_mb:.2f}",
            "X-GPU-Reserved-MB": f"{gpu_reserved_mb:.2f}",
            "X-GPU-Peak-MB": f"{gpu_peak_mb:.2f}",
        }

        return StreamingResponse(
            buffer,
            media_type="audio/wav",
            headers=headers,
        )

    except torch.cuda.OutOfMemoryError as exc:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        raise HTTPException(
            status_code=507,
            detail=f"CUDA out of memory: {exc}",
        ) from exc

    except Exception as exc:
        print(
            f"[error] Request {request_id}: "
            f"{type(exc).__name__}: {exc}"
        )

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        raise HTTPException(
            status_code=500,
            detail=(
                f"TTS generation failed: "
                f"{type(exc).__name__}: {exc}"
            ),
        ) from exc
