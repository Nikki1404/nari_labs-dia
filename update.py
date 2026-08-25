#server.py-
import io
import os
import random
import time
import uuid

import numpy as np
import soundfile as sf
import torch

from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field
from transformers import AutoProcessor, DiaForConditionalGeneration


# =============================================================================
# CONFIG
# =============================================================================

MODEL_ID = os.getenv(
    "MODEL_ID",
    "nari-labs/Dia-1.6B-0626",
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DTYPE = (
    torch.float16
    if DEVICE == "cuda"
    else torch.float32
)

SAMPLE_RATE = 44100


# For your long call:
#
# 86 tokens ~= 1 sec according to Dia documentation.
#
# 12288 ~= ~143 sec theoretical audio budget.
#
# You can override this from client.
DEFAULT_MAX_NEW_TOKENS = 12288

MAX_ALLOWED_NEW_TOKENS = 16384


# =============================================================================
# APP
# =============================================================================

app = FastAPI(
    title="Dia TTS API",
    version="3.0.0",
)

processor = None
model = None
model_load_ms = None


# =============================================================================
# REQUEST
# =============================================================================

class TTSRequest(BaseModel):

    text: str = Field(
        ...,
        min_length=1,
    )

    max_new_tokens: int = Field(
        default=DEFAULT_MAX_NEW_TOKENS,
        ge=256,
        le=MAX_ALLOWED_NEW_TOKENS,
    )

    seed: int = Field(
        default=1234,
        ge=0,
        le=2147483647,
    )


# =============================================================================
# HELPERS
# =============================================================================

def sync_cuda():

    if torch.cuda.is_available():
        torch.cuda.synchronize()


def set_seed(seed: int):
    """
    Best-effort reproducibility for Dia native voices.
    """

    random.seed(seed)

    np.random.seed(
        seed % (2**32 - 1)
    )

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Improve repeatability.
    if torch.backends.cudnn.is_available():

        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


# =============================================================================
# STARTUP
# =============================================================================

@app.on_event("startup")
def load_model():

    global processor
    global model
    global model_load_ms

    print("")
    print("=" * 80)
    print("DIA TTS STARTUP")
    print("=" * 80)

    print(
        f"Model             : {MODEL_ID}"
    )

    print(
        f"Device            : {DEVICE}"
    )

    print(
        f"PyTorch           : {torch.__version__}"
    )

    print(
        f"PyTorch CUDA      : {torch.version.cuda}"
    )

    print(
        f"CUDA available    : "
        f"{torch.cuda.is_available()}"
    )

    if torch.cuda.is_available():

        print(
            f"GPU               : "
            f"{torch.cuda.get_device_name(0)}"
        )

    print(
        f"DTYPE             : {DTYPE}"
    )

    print(
        "Speaker mapping   : "
        "[S1]=Agent, [S2]=Customer"
    )

    print(
        "Generation mode   : "
        "ONE generate() for complete transcript"
    )

    print("=" * 80)

    started = (
        time.perf_counter_ns()
    )

    processor = AutoProcessor.from_pretrained(
        MODEL_ID
    )

    model = DiaForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=DTYPE,
        low_cpu_mem_usage=True,
    ).to(
        DEVICE
    )

    model.eval()

    sync_cuda()

    model_load_ms = (
        time.perf_counter_ns()
        - started
    ) / 1_000_000

    print(
        f"[startup] Model loaded in "
        f"{model_load_ms:.2f} ms"
    )

    print(
        f"[startup] Model loaded in "
        f"{model_load_ms / 1000:.2f} sec"
    )

    if torch.cuda.is_available():

        print(
            f"[startup] GPU allocated : "
            f"{torch.cuda.memory_allocated() / 1024**2:.2f} MB"
        )

        print(
            f"[startup] GPU reserved  : "
            f"{torch.cuda.memory_reserved() / 1024**2:.2f} MB"
        )


# =============================================================================
# ROOT
# =============================================================================

@app.get("/")
def root():

    return {
        "service":
            "Dia TTS API",

        "model":
            MODEL_ID,

        "device":
            DEVICE,

        "generation_mode":
            "single_generation",

        "speaker_mapping": {
            "S1": "agent",
            "S2": "customer",
        },
    }


# =============================================================================
# HEALTH
# =============================================================================

@app.get("/health")
def health():

    result = {

        "status":
            "ok",

        "model":
            MODEL_ID,

        "device":
            DEVICE,

        "cuda_available":
            torch.cuda.is_available(),

        "torch_version":
            torch.__version__,

        "torch_cuda_version":
            torch.version.cuda,

        "model_load_ms":
            model_load_ms,

        "generation_mode":
            "single_generation",

        "default_max_new_tokens":
            DEFAULT_MAX_NEW_TOKENS,

        "max_allowed_new_tokens":
            MAX_ALLOWED_NEW_TOKENS,

        "speaker_mapping": {
            "S1": "agent",
            "S2": "customer",
        },
    }

    if torch.cuda.is_available():

        result["gpu"] = (
            torch.cuda.get_device_name(0)
        )

    return result


# =============================================================================
# TTS
# =============================================================================

@app.post("/tts")
def tts(req: TTSRequest):

    if (
        model is None
        or processor is None
    ):

        raise HTTPException(
            status_code=503,
            detail="Model is not loaded yet",
        )

    request_id = str(
        uuid.uuid4()
    )

    server_start = (
        time.perf_counter_ns()
    )

    try:

        print("")
        print("=" * 80)
        print("DIA TTS REQUEST")
        print("=" * 80)

        print(
            f"Request ID        : {request_id}"
        )

        print(
            f"Seed              : {req.seed}"
        )

        print(
            f"Max new tokens    : "
            f"{req.max_new_tokens}"
        )

        print(
            f"Characters        : "
            f"{len(req.text)}"
        )

        print(
            f"Words             : "
            f"{len(req.text.split())}"
        )

        print(
            "Generation mode   : "
            "ONE model.generate()"
        )

        print("=" * 80)

        # =====================================================================
        # Seed
        # =====================================================================

        set_seed(
            req.seed
        )

        # =====================================================================
        # Preprocess
        # =====================================================================

        preprocess_start = (
            time.perf_counter_ns()
        )

        inputs = processor(
            text=[
                req.text
            ],
            padding=True,
            return_tensors="pt",
        )

        inputs = inputs.to(
            model.device
        )

        sync_cuda()

        preprocess_ms = (
            time.perf_counter_ns()
            - preprocess_start
        ) / 1_000_000

        # =====================================================================
        # GPU memory stats
        # =====================================================================

        if torch.cuda.is_available():

            torch.cuda.reset_peak_memory_stats()

        # =====================================================================
        # ONE inference for COMPLETE transcript
        # =====================================================================

        inference_start = (
            time.perf_counter_ns()
        )

        with torch.inference_mode():

            outputs = model.generate(
                **inputs,

                max_new_tokens=
                    req.max_new_tokens,

                guidance_scale=3.0,

                temperature=1.8,

                top_p=0.90,

                top_k=45,
            )

        sync_cuda()

        inference_ms = (
            time.perf_counter_ns()
            - inference_start
        ) / 1_000_000

        # =====================================================================
        # Decode
        # =====================================================================

        decode_start = (
            time.perf_counter_ns()
        )

        decoded = processor.batch_decode(
            outputs
        )

        audio = (
            decoded[0]
            if isinstance(
                decoded,
                (list, tuple),
            )
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

        audio = np.asarray(
            audio
        )

        audio = np.squeeze(
            audio
        )

        if audio.ndim != 1:

            raise RuntimeError(
                f"Unexpected audio shape: "
                f"{audio.shape}"
            )

        audio = audio.astype(
            np.float32,
            copy=False,
        )

        if len(audio) == 0:

            raise RuntimeError(
                "Generated audio is empty"
            )

        if not np.all(
            np.isfinite(audio)
        ):

            raise RuntimeError(
                "Generated audio contains "
                "NaN or Inf values."
            )

        decode_ms = (
            time.perf_counter_ns()
            - decode_start
        ) / 1_000_000

        # =====================================================================
        # Audio duration
        # =====================================================================

        audio_duration_s = (
            len(audio)
            / SAMPLE_RATE
        )

        # =====================================================================
        # WAV
        # =====================================================================

        encode_start = (
            time.perf_counter_ns()
        )

        buffer = io.BytesIO()

        sf.write(
            buffer,
            audio,
            SAMPLE_RATE,
            format="WAV",
            subtype="PCM_16",
        )

        wav_bytes = (
            buffer.getvalue()
        )

        encode_ms = (
            time.perf_counter_ns()
            - encode_start
        ) / 1_000_000

        # =====================================================================
        # Metrics
        # =====================================================================

        server_total_ms = (
            time.perf_counter_ns()
            - server_start
        ) / 1_000_000

        generation_rtf = (
            (
                inference_ms
                / 1000
            )
            / audio_duration_s

            if audio_duration_s > 0

            else 0
        )

        total_rtf = (
            (
                server_total_ms
                / 1000
            )
            / audio_duration_s

            if audio_duration_s > 0

            else 0
        )

        gpu_allocated_mb = 0
        gpu_reserved_mb = 0
        gpu_peak_mb = 0

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

        # =====================================================================
        # Logs
        # =====================================================================

        print("")
        print("=" * 80)
        print("SERVER LATENCY")
        print("=" * 80)

        print(
            f"Request ID        : "
            f"{request_id}"
        )

        print(
            f"Seed              : "
            f"{req.seed}"
        )

        print(
            f"Max new tokens    : "
            f"{req.max_new_tokens}"
        )

        print(
            f"Preprocess        : "
            f"{preprocess_ms:.2f} ms"
        )

        print(
            f"Inference         : "
            f"{inference_ms:.2f} ms"
        )

        print(
            f"Decode            : "
            f"{decode_ms:.2f} ms"
        )

        print(
            f"WAV encoding      : "
            f"{encode_ms:.2f} ms"
        )

        print(
            f"SERVER TOTAL      : "
            f"{server_total_ms:.2f} ms"
        )

        print(
            f"Audio duration    : "
            f"{audio_duration_s:.3f} sec"
        )

        print(
            f"Generation RTF    : "
            f"{generation_rtf:.4f}"
        )

        print(
            f"Total RTF         : "
            f"{total_rtf:.4f}"
        )

        if torch.cuda.is_available():

            print(
                f"GPU allocated     : "
                f"{gpu_allocated_mb:.2f} MB"
            )

            print(
                f"GPU reserved      : "
                f"{gpu_reserved_mb:.2f} MB"
            )

            print(
                f"GPU peak          : "
                f"{gpu_peak_mb:.2f} MB"
            )

        print("=" * 80)

        # =====================================================================
        # Headers
        # =====================================================================

        headers = {

            "X-Request-ID":
                request_id,

            "X-Seed":
                str(req.seed),

            "X-Max-New-Tokens":
                str(req.max_new_tokens),

            "X-Preprocess-Time-MS":
                f"{preprocess_ms:.2f}",

            "X-Inference-Time-MS":
                f"{inference_ms:.2f}",

            "X-Decode-Time-MS":
                f"{decode_ms:.2f}",

            "X-Encoding-Time-MS":
                f"{encode_ms:.2f}",

            "X-Server-Total-MS":
                f"{server_total_ms:.2f}",

            "X-Audio-Duration-S":
                f"{audio_duration_s:.3f}",

            "X-Generation-RTF":
                f"{generation_rtf:.4f}",

            "X-RTF":
                f"{total_rtf:.4f}",

            "X-Sample-Rate":
                str(SAMPLE_RATE),

            "X-GPU-Allocated-MB":
                f"{gpu_allocated_mb:.2f}",

            "X-GPU-Reserved-MB":
                f"{gpu_reserved_mb:.2f}",

            "X-GPU-Peak-MB":
                f"{gpu_peak_mb:.2f}",
        }

        # =====================================================================
        # Complete WAV response
        # =====================================================================

        return Response(
            content=wav_bytes,
            media_type="audio/wav",
            headers=headers,
        )

    except torch.cuda.OutOfMemoryError as exc:

        if torch.cuda.is_available():

            torch.cuda.empty_cache()

        raise HTTPException(
            status_code=507,
            detail=(
                f"CUDA out of memory: "
                f"{exc}"
            ),
        ) from exc

    except Exception as exc:

        print(
            f"[error] Request "
            f"{request_id}: "
            f"{type(exc).__name__}: "
            f"{exc}"
        )

        if torch.cuda.is_available():

            torch.cuda.empty_cache()

        raise HTTPException(
            status_code=500,
            detail=(
                "TTS generation failed: "
                f"{type(exc).__name__}: "
                f"{exc}"
            ),
        ) from exc



#client.py-
#!/usr/bin/env python3
#!/usr/bin/env python3

import argparse
import io
import json
import queue
import struct
import sys
import threading
import time
from pathlib import Path

import numpy as np
import requests
import soundfile as sf


# =============================================================================
# HELPERS
# =============================================================================

def read_text_file(path_value):
    path = Path(path_value)

    if not path.exists():
        print(f"File not found: {path}")
        sys.exit(1)

    return path.read_text(
        encoding="utf-8"
    ).strip()


def safe_float(headers, key, default=0.0):
    try:
        return float(
            headers.get(key, default)
        )
    except Exception:
        return default


def receive_exact(raw, size):
    data = bytearray()

    while len(data) < size:
        part = raw.read(
            size - len(data)
        )

        if not part:
            raise EOFError(
                "Server stream ended unexpectedly."
            )

        data.extend(part)

    return bytes(data)


def read_frame(raw):
    metadata_size = struct.unpack(
        ">I",
        receive_exact(raw, 4),
    )[0]

    metadata = json.loads(
        receive_exact(
            raw,
            metadata_size,
        ).decode("utf-8")
    )

    pcm_size = struct.unpack(
        ">Q",
        receive_exact(raw, 8),
    )[0]

    pcm_bytes = b""

    if pcm_size > 0:
        pcm_bytes = receive_exact(
            raw,
            pcm_size,
        )

    return metadata, pcm_bytes


# =============================================================================
# NORMAL / SEED MODE
#
# Used for:
#
# python client.py \
#   --seed 1234 \
#   --text "[S1] ... [S2] ..." \
#   --output seed_1234.wav
#
# This expects the server's ordinary JSON TTS endpoint behavior.
# =============================================================================

def run_seed_mode(args):
    if not args.text and not args.text_file:
        print(
            "Seed mode requires --text or --text-file."
        )
        sys.exit(1)

    if args.text_file:
        text = read_text_file(
            args.text_file
        )
    else:
        text = args.text.strip()

    url = (
        args.server.rstrip("/")
        + "/tts"
    )

    payload = {
        "text": text,
        "seed": args.seed,
        "max_new_tokens": args.max_new_tokens,
    }

    print("")
    print("=" * 80)
    print("DIA SEED TEST MODE")
    print("=" * 80)

    print(f"Server            : {url}")
    print(f"Seed              : {args.seed}")
    print(f"Words             : {len(text.split())}")
    print(
        f"Max new tokens    : "
        f"{args.max_new_tokens}"
    )
    print(f"Output            : {args.output}")
    print(f"Play              : {args.play}")

    print("=" * 80)

    request_start = time.perf_counter()

    try:
        response = requests.post(
            url,
            json=payload,
            timeout=args.timeout,
        )

    except requests.RequestException as exc:
        print(
            f"Request failed: {exc}"
        )
        sys.exit(1)

    headers_received = (
        time.perf_counter()
    )

    if response.status_code != 200:
        print(
            f"Request failed "
            f"({response.status_code}): "
            f"{response.text}"
        )
        sys.exit(1)

    audio_bytes = response.content

    if not audio_bytes:
        print(
            "Server returned no audio."
        )
        sys.exit(1)

    first_audio_time = (
        time.perf_counter()
    )

    output_path = Path(
        args.output
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_bytes(
        audio_bytes
    )

    try:
        with sf.SoundFile(
            io.BytesIO(audio_bytes)
        ) as wav:
            sample_rate = wav.samplerate
            channels = wav.channels
            frames = len(wav)

            duration = (
                frames / sample_rate
            )

    except Exception as exc:
        print(
            f"Could not inspect WAV: {exc}"
        )

        sample_rate = 0
        channels = 0
        frames = 0
        duration = 0.0

    request_end = (
        time.perf_counter()
    )

    ttfb_ms = (
        headers_received
        - request_start
    ) * 1000

    ttfa_ms = (
        first_audio_time
        - request_start
    ) * 1000

    total_ms = (
        request_end
        - request_start
    ) * 1000

    print("")
    print("=" * 80)
    print("CLIENT LATENCY")
    print("=" * 80)

    print(
        f"TTFB              : "
        f"{ttfb_ms:.2f} ms"
    )

    print(
        f"TTFA / TTFT       : "
        f"{ttfa_ms:.2f} ms"
    )

    print(
        f"CLIENT TOTAL      : "
        f"{total_ms:.2f} ms"
    )

    print(
        f"Audio duration    : "
        f"{duration:.3f} sec"
    )

    print("")
    print("=" * 80)
    print("SERVER LATENCY")
    print("=" * 80)

    print(
        f"Preprocess        : "
        f"{safe_float(response.headers, 'X-Preprocess-Time-MS'):.2f} ms"
    )

    print(
        f"Inference         : "
        f"{safe_float(response.headers, 'X-Inference-Time-MS'):.2f} ms"
    )

    print(
        f"Decode            : "
        f"{safe_float(response.headers, 'X-Decode-Time-MS'):.2f} ms"
    )

    print(
        f"SERVER TOTAL      : "
        f"{safe_float(response.headers, 'X-Server-Total-MS'):.2f} ms"
    )

    print("")
    print("=" * 80)
    print("AUDIO")
    print("=" * 80)

    print(f"Saved             : {output_path}")
    print(f"Seed              : {args.seed}")
    print(f"Sample rate       : {sample_rate}")
    print(f"Channels          : {channels}")
    print(f"Duration          : {duration:.3f}s")

    print("=" * 80)

    if args.play:
        play_audio_file(
            output_path,
            args.speed,
        )


# =============================================================================
# REFERENCE CONDITIONED LONG MODE
#
# Used for:
#
# python client.py \
#   --text-file full_call.txt \
#   --reference-audio reference.wav \
#   --reference-text reference.txt \
#   --output full_call.wav \
#   --play
# =============================================================================

def run_reference_mode(args):
    if not args.text_file:
        print(
            "Reference mode requires --text-file."
        )
        sys.exit(1)

    if not args.reference_audio:
        print(
            "Reference mode requires "
            "--reference-audio."
        )
        sys.exit(1)

    if not args.reference_text:
        print(
            "Reference mode requires "
            "--reference-text."
        )
        sys.exit(1)

    transcript_path = Path(
        args.text_file
    )

    reference_audio_path = Path(
        args.reference_audio
    )

    reference_text_path = Path(
        args.reference_text
    )

    for path in [
        transcript_path,
        reference_audio_path,
        reference_text_path,
    ]:
        if not path.exists():
            print(
                f"File not found: {path}"
            )
            sys.exit(1)

    text = read_text_file(
        transcript_path
    )

    reference_text = read_text_file(
        reference_text_path
    )

    url = (
        args.server.rstrip("/")
        + "/tts"
    )

    print("")
    print("=" * 80)
    print("DIA REFERENCE LONG-TTS MODE")
    print("=" * 80)

    print(f"Server            : {url}")
    print(f"Transcript        : {transcript_path}")
    print(
        f"Reference audio   : "
        f"{reference_audio_path}"
    )
    print(
        f"Reference text    : "
        f"{reference_text_path}"
    )
    print(
        f"Max new tokens    : "
        f"{args.max_new_tokens}"
    )
    print(f"Output            : {args.output}")
    print(f"Play              : {args.play}")
    print(f"Speed             : {args.speed}")

    print("=" * 80)

    # =========================================================================
    # Sequential playback queue
    # =========================================================================

    playback_queue = queue.Queue()

    playback_enabled = False
    playback_thread = None

    if args.play:
        try:
            import sounddevice as sd

            playback_enabled = True

            def playback_worker():
                while True:
                    item = playback_queue.get()

                    if item is None:
                        playback_queue.task_done()
                        break

                    (
                        block_index,
                        block_count,
                        audio,
                        sample_rate,
                    ) = item

                    playback_audio = (
                        adjust_speed(
                            audio,
                            args.speed,
                        )
                    )

                    print("")
                    print(
                        f"[PLAY] Starting block "
                        f"{block_index}/"
                        f"{block_count}"
                    )

                    sd.play(
                        playback_audio,
                        sample_rate,
                    )

                    # Next block cannot begin
                    # until current block ends.
                    sd.wait()

                    print(
                        f"[PLAY] Finished block "
                        f"{block_index}/"
                        f"{block_count}"
                    )

                    playback_queue.task_done()

            playback_thread = (
                threading.Thread(
                    target=playback_worker,
                    daemon=True,
                )
            )

            playback_thread.start()

        except Exception as exc:
            print(
                f"Playback disabled: {exc}"
            )

            playback_enabled = False

    # =========================================================================
    # Multipart request
    # =========================================================================

    request_start = (
        time.perf_counter()
    )

    try:
        with open(
            reference_audio_path,
            "rb",
        ) as reference_handle:

            files = {
                "reference_audio": (
                    reference_audio_path.name,
                    reference_handle,
                    "audio/wav",
                )
            }

            data = {
                "text":
                    text,

                "reference_text":
                    reference_text,

                "max_new_tokens":
                    str(
                        args.max_new_tokens
                    ),
            }

            response = requests.post(
                url,
                data=data,
                files=files,
                stream=True,
                timeout=args.timeout,
            )

    except requests.RequestException as exc:
        print(
            f"Request failed: {exc}"
        )
        sys.exit(1)

    headers_received = (
        time.perf_counter()
    )

    if response.status_code != 200:
        print(
            f"Request failed "
            f"({response.status_code}): "
            f"{response.text}"
        )
        sys.exit(1)

    # =========================================================================
    # Output WAV
    # =========================================================================

    output_path = Path(
        args.output
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    wav_writer = sf.SoundFile(
        str(output_path),
        mode="w",
        samplerate=44100,
        channels=1,
        subtype="PCM_16",
        format="WAV",
    )

    complete_audio_parts = []

    first_audio_time = None
    block_counter = 0
    server_metrics = {}

    try:
        while True:
            metadata, pcm_bytes = (
                read_frame(
                    response.raw
                )
            )

            frame_type = (
                metadata.get("type")
            )

            if frame_type == "audio":
                block_counter += 1

                if first_audio_time is None:
                    first_audio_time = (
                        time.perf_counter()
                    )

                pcm = np.frombuffer(
                    pcm_bytes,
                    dtype="<i2",
                )

                audio = (
                    pcm.astype(
                        np.float32
                    )
                    / 32767.0
                )

                wav_writer.write(
                    audio
                )

                complete_audio_parts.append(
                    audio.copy()
                )

                print("")
                print(
                    f"[RECV] Block "
                    f"{metadata['block_index']}/"
                    f"{metadata['block_count']}"
                )

                print(
                    f"       Duration  : "
                    f"{metadata['audio_duration_s']:.2f}s"
                )

                print(
                    f"       Inference : "
                    f"{metadata['inference_ms']:.2f}ms"
                )

                if playback_enabled:
                    playback_queue.put(
                        (
                            metadata[
                                "block_index"
                            ],
                            metadata[
                                "block_count"
                            ],
                            audio.copy(),
                            metadata[
                                "sample_rate"
                            ],
                        )
                    )

            elif frame_type == "end":
                server_metrics = metadata
                break

    finally:
        wav_writer.close()

    receive_complete_time = (
        time.perf_counter()
    )

    # =========================================================================
    # Finish queued playback
    # =========================================================================

    if playback_enabled:
        playback_queue.join()

        playback_queue.put(None)

        playback_queue.join()

        if playback_thread:
            playback_thread.join(
                timeout=10,
            )

    playback_complete_time = (
        time.perf_counter()
    )

    # =========================================================================
    # Optional adjusted WAV
    # =========================================================================

    adjusted_path = None

    if (
        args.save_adjusted
        and
        args.speed != 1.0
        and
        complete_audio_parts
    ):
        try:
            complete_audio = (
                np.concatenate(
                    complete_audio_parts
                )
            )

            adjusted = adjust_speed(
                complete_audio,
                args.speed,
            )

            speed_string = (
                str(args.speed)
                .replace(".", "_")
            )

            adjusted_path = (
                output_path.with_name(
                    output_path.stem
                    + "_speed_"
                    + speed_string
                    + output_path.suffix
                )
            )

            sf.write(
                str(adjusted_path),
                adjusted,
                44100,
                subtype="PCM_16",
            )

        except Exception as exc:
            print(
                f"Adjusted WAV save failed: "
                f"{exc}"
            )

    # =========================================================================
    # Metrics
    # =========================================================================

    ttfb_ms = (
        headers_received
        - request_start
    ) * 1000

    ttfa_ms = (
        (
            first_audio_time
            - request_start
        ) * 1000
        if first_audio_time
        else 0.0
    )

    receive_total_ms = (
        receive_complete_time
        - request_start
    ) * 1000

    playback_total_ms = (
        playback_complete_time
        - request_start
    ) * 1000

    print("")
    print("=" * 80)
    print("CLIENT LATENCY")
    print("=" * 80)

    print(
        f"HTTP headers / TTFB : "
        f"{ttfb_ms:.2f} ms"
    )

    print(
        f"TTFA / TTFT         : "
        f"{ttfa_ms:.2f} ms"
    )

    print(
        f"Receive total       : "
        f"{receive_total_ms:.2f} ms"
    )

    if args.play:
        print(
            f"E2E incl playback   : "
            f"{playback_total_ms:.2f} ms"
        )

    print(
        f"Blocks received     : "
        f"{block_counter}"
    )

    print("")
    print("=" * 80)
    print("SERVER METRICS")
    print("=" * 80)

    print(
        f"Blocks              : "
        f"{server_metrics.get('block_count', 0)}"
    )

    print(
        f"Preprocess          : "
        f"{server_metrics.get('preprocess_ms', 0):.2f} ms"
    )

    print(
        f"Inference           : "
        f"{server_metrics.get('inference_ms', 0):.2f} ms"
    )

    print(
        f"Decode              : "
        f"{server_metrics.get('decode_ms', 0):.2f} ms"
    )

    print(
        f"SERVER TOTAL        : "
        f"{server_metrics.get('server_total_ms', 0):.2f} ms"
    )

    print(
        f"Audio duration      : "
        f"{server_metrics.get('audio_duration_s', 0):.2f}s"
    )

    print(
        f"Generation RTF      : "
        f"{server_metrics.get('generation_rtf', 0):.4f}"
    )

    print(
        f"Total RTF           : "
        f"{server_metrics.get('total_rtf', 0):.4f}"
    )

    print(
        f"GPU peak            : "
        f"{server_metrics.get('gpu_peak_mb', 0):.2f} MB"
    )

    print("")
    print("=" * 80)
    print("OUTPUT")
    print("=" * 80)

    print(
        f"Original WAV        : "
        f"{output_path}"
    )

    if adjusted_path:
        print(
            f"Adjusted WAV        : "
            f"{adjusted_path}"
        )

    print(
        f"Reference WAV       : "
        f"{reference_audio_path}"
    )

    print("=" * 80)


# =============================================================================
# AUDIO PLAYBACK HELPERS
# =============================================================================

def adjust_speed(
    audio,
    speed,
):
    if speed == 1.0:
        return audio

    try:
        import librosa

        return (
            librosa.effects.time_stretch(
                audio,
                rate=speed,
            )
        )

    except Exception as exc:
        print(
            f"[WARN] Speed adjustment "
            f"failed: {exc}"
        )

        return audio


def play_audio_file(
    path,
    speed,
):
    try:
        import sounddevice as sd

        audio, sr = sf.read(
            str(path),
            dtype="float32",
        )

        audio = adjust_speed(
            audio,
            speed,
        )

        print("")
        print(
            f"Playing audio at speed "
            f"{speed}..."
        )

        sd.play(
            audio,
            sr,
        )

        sd.wait()

        print(
            "Playback complete."
        )

    except Exception as exc:
        print(
            f"Playback failed: {exc}"
        )


# =============================================================================
# ENTRY
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Dia TTS client supporting "
            "seed testing and reference-conditioned long TTS"
        )
    )

    parser.add_argument(
        "--server",
        default="http://localhost:8000",
    )

    # -------------------------------------------------------------------------
    # Seed/native mode
    # -------------------------------------------------------------------------

    parser.add_argument(
        "--text",
        default=None,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help=(
            "If provided WITHOUT reference-audio/reference-text, "
            "run native seed-test mode."
        ),
    )

    # -------------------------------------------------------------------------
    # Shared text-file support
    # -------------------------------------------------------------------------

    parser.add_argument(
        "--text-file",
        default=None,
    )

    # -------------------------------------------------------------------------
    # Reference mode
    # -------------------------------------------------------------------------

    parser.add_argument(
        "--reference-audio",
        default=None,
    )

    parser.add_argument(
        "--reference-text",
        default=None,
    )

    # -------------------------------------------------------------------------
    # Common
    # -------------------------------------------------------------------------

    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=4096,
    )

    parser.add_argument(
        "--output",
        default="output.wav",
    )

    parser.add_argument(
        "--play",
        action="store_true",
    )

    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--save-adjusted",
        action="store_true",
    )

    parser.add_argument(
        "--timeout",
        type=float,
        default=1800.0,
    )

    args = parser.parse_args()

    # =========================================================================
    # MODE SELECTION
    # =========================================================================

    reference_mode = (
        args.reference_audio is not None
        or
        args.reference_text is not None
    )

    if reference_mode:
        run_reference_mode(
            args
        )

    else:
        if args.seed is None:
            print(
                "Native mode requires --seed."
            )
            sys.exit(1)

        run_seed_mode(
            args
        )


if __name__ == "__main__":
    main()


#requirements.txt-
fastapi
uvicorn[standard]

transformers
accelerate

numpy
scipy
soundfile
pydantic


#Dockerfile
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04


ENV http_proxy="http://163.116.128.80:8080"
ENV https_proxy="http://163.116.128.80:8080"


ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

ENV CC=/usr/bin/gcc
ENV CXX=/usr/bin/g++

ENV MODEL_ID="nari-labs/Dia-1.6B-0626"


RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
        python3-dev \
        gcc \
        g++ \
        build-essential \
        git \
        libsndfile1 \
        ffmpeg \
    && rm -rf /var/lib/apt/lists/*


RUN python3 -m pip install --upgrade \
    pip \
    setuptools \
    wheel


# PyTorch CUDA 12.4
RUN python3 -m pip install \
    torch==2.6.0 \
    --index-url https://download.pytorch.org/whl/cu124


WORKDIR /app


COPY requirements.txt \
    /app/requirements.txt


RUN python3 -m pip install \
    -r /app/requirements.txt


COPY server.py \
    /app/server.py


RUN python3 -c "\
import torch; \
print('Torch:', torch.__version__); \
print('Torch CUDA:', torch.version.cuda)"


EXPOSE 8000


CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000", "--log-level", "info"]




docker rm -f dia-tts 2>/dev/null || true
docker build --no-cache -t dia-tts .
docker run --rm --gpus all dia-tts python3 -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
docker run --rm --gpus all --ipc=host --shm-size=8g -p 8000:8000 dia-tts

python client.py --server http://localhost:8000 --text-file full_call.txt --seed 8472 --max-new-tokens 12288 --output full_call_seed_8472.wav --play
python client.py --server http://localhost:8000 --text-file full_call.txt --seed 8472 --max-new-tokens 16384 --output full_call_seed_8472.wav --play
python client.py --server http://localhost:8000 --text-file full_call.txt --seed 8472 --max-new-tokens 12288 --output full_call.wav --speed 0.85 --save-adjusted --play
$seeds=@(); while($seeds.Count -lt 25){$seeds+=Get-Random -Minimum 1000 -Maximum 45001;$seeds=@($seeds|Sort-Object -Unique)}; foreach($seed in $seeds){Write-Host "Generating seed $seed"; python client.py --server http://localhost:8000 --text-file full_call.txt --seed $seed --max-new-tokens 12288 --output "full_call_seed_$seed.wav"}
python client.py --server http://<SERVER-IP>:8000 --text-file full_call.txt --reference-audio reference.wav --reference-text reference.txt --max-new-tokens 3072 --output full_call.wav --play
python client.py --server http://localhost:8000 --text-file full_call.txt --reference-audio reference.wav --reference-text reference.txt --max-new-tokens 4096 --output full_call.wav --play
