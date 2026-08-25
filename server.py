#!/usr/bin/env python3

import io
import json
import os
import random
import re
import struct
import time
import uuid
from math import gcd

import numpy as np
import soundfile as sf
import torch

from fastapi import (
    FastAPI,
    File,
    Form,
    HTTPException,
    UploadFile,
)
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, Field
from scipy.signal import resample_poly
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


# =============================================================================
# LONG-TEXT SETTINGS
# =============================================================================

TARGET_WORDS_PER_BLOCK = 35
MAX_WORDS_PER_BLOCK = 55

DEFAULT_REFERENCE_MAX_NEW_TOKENS = 4096
DEFAULT_SEED_MAX_NEW_TOKENS = 1024


# =============================================================================
# APP
# =============================================================================

app = FastAPI(
    title="Dia TTS API",
    version="6.0.0",
)

processor = None
model = None
model_load_ms = None


# =============================================================================
# SEED REQUEST
# =============================================================================

class SeedTTSRequest(BaseModel):

    text: str = Field(
        ...,
        min_length=1,
    )

    seed: int = Field(
        default=1234,
        ge=0,
        le=2147483647,
    )

    max_new_tokens: int = Field(
        default=DEFAULT_SEED_MAX_NEW_TOKENS,
        ge=256,
        le=4096,
    )


# =============================================================================
# CUDA / RNG
# =============================================================================

def sync_cuda():

    if torch.cuda.is_available():
        torch.cuda.synchronize()


def set_seed(seed: int):

    random.seed(seed)

    np.random.seed(
        seed % (2**32 - 1)
    )

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


# =============================================================================
# TEXT HELPERS
# =============================================================================

def normalize_tags(text: str):

    text = re.sub(
        r"\[\s*s1\s*\]",
        "[S1]",
        text,
        flags=re.IGNORECASE,
    )

    text = re.sub(
        r"\[\s*s2\s*\]",
        "[S2]",
        text,
        flags=re.IGNORECASE,
    )

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


def parse_turns(text: str):

    text = normalize_tags(text)

    matches = list(
        re.finditer(
            r"\[(S1|S2)\]\s*",
            text,
            flags=re.IGNORECASE,
        )
    )

    if not matches:

        raise ValueError(
            "Transcript must contain [S1] and/or [S2] tags."
        )

    turns = []

    for index, match in enumerate(matches):

        speaker = match.group(1).upper()

        start = match.end()

        if index + 1 < len(matches):
            end = matches[index + 1].start()
        else:
            end = len(text)

        speech = text[start:end].strip()

        if speech:
            turns.append(
                (speaker, speech)
            )

    if not turns:

        raise ValueError(
            "No dialogue found after speaker tags."
        )

    return turns


def split_sentences(text: str):

    return [
        item.strip()
        for item in re.split(
            r"(?<=[.!?])\s+",
            text,
        )
        if item.strip()
    ]


def split_long_turn(
    speaker: str,
    speech: str,
):

    if (
        len(speech.split())
        <= MAX_WORDS_PER_BLOCK
    ):
        return [
            (speaker, speech)
        ]

    sentences = split_sentences(
        speech
    )

    pieces = []

    current = []
    current_words = 0

    for sentence in sentences:

        sentence_words = len(
            sentence.split()
        )

        if (
            sentence_words
            > MAX_WORDS_PER_BLOCK
        ):

            if current:

                pieces.append(
                    (
                        speaker,
                        " ".join(current),
                    )
                )

                current = []
                current_words = 0

            words = sentence.split()

            for index in range(
                0,
                len(words),
                MAX_WORDS_PER_BLOCK,
            ):

                pieces.append(
                    (
                        speaker,
                        " ".join(
                            words[
                                index:
                                index + MAX_WORDS_PER_BLOCK
                            ]
                        ),
                    )
                )

            continue

        if (
            current
            and
            current_words + sentence_words
            > MAX_WORDS_PER_BLOCK
        ):

            pieces.append(
                (
                    speaker,
                    " ".join(current),
                )
            )

            current = []
            current_words = 0

        current.append(sentence)
        current_words += sentence_words

    if current:

        pieces.append(
            (
                speaker,
                " ".join(current),
            )
        )

    return pieces


def build_dialogue_blocks(text: str):
    """
    Keep multiple S1/S2 turns together.

    Prefer block boundaries immediately before S1 so a new block
    naturally begins with S1.
    """

    turns = []

    for speaker, speech in parse_turns(text):

        turns.extend(
            split_long_turn(
                speaker,
                speech,
            )
        )

    blocks = []

    current = []
    current_words = 0

    for speaker, speech in turns:

        word_count = len(
            speech.split()
        )

        # Preferred boundary.
        if (
            current
            and
            speaker == "S1"
            and
            current_words
            >= TARGET_WORDS_PER_BLOCK
        ):

            blocks.append(current)

            current = []
            current_words = 0

        # Harder size boundary, still only before S1.
        elif (
            current
            and
            speaker == "S1"
            and
            current_words + word_count
            > MAX_WORDS_PER_BLOCK
        ):

            blocks.append(current)

            current = []
            current_words = 0

        current.append(
            (
                speaker,
                speech,
            )
        )

        current_words += word_count

    if current:
        blocks.append(current)

    return [
        " ".join(
            f"[{speaker}] {speech}"
            for speaker, speech
            in block
        )
        for block in blocks
    ]


# =============================================================================
# REFERENCE AUDIO
# =============================================================================

def load_reference_audio(
    wav_bytes: bytes,
):

    audio, sample_rate = sf.read(
        io.BytesIO(wav_bytes),
        dtype="float32",
    )

    if audio.ndim == 2:

        audio = np.mean(
            audio,
            axis=1,
        )

    audio = np.asarray(
        audio,
        dtype=np.float32,
    )

    if len(audio) == 0:

        raise ValueError(
            "Reference audio is empty."
        )

    if sample_rate != SAMPLE_RATE:

        divisor = gcd(
            sample_rate,
            SAMPLE_RATE,
        )

        audio = resample_poly(
            audio,
            SAMPLE_RATE // divisor,
            sample_rate // divisor,
        ).astype(
            np.float32
        )

    duration = (
        len(audio)
        / SAMPLE_RATE
    )

    return audio, duration


# =============================================================================
# AUDIO DECODING
# =============================================================================

def decode_plain_audio(outputs):

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

    audio = np.squeeze(
        np.asarray(
            audio,
            dtype=np.float32,
        )
    )

    if audio.ndim != 1:

        raise RuntimeError(
            f"Unexpected audio shape: "
            f"{audio.shape}"
        )

    if len(audio) == 0:

        raise RuntimeError(
            "Dia returned empty audio."
        )

    return audio


def decode_conditioned_audio(
    outputs,
    prompt_len,
):

    decoded = processor.batch_decode(
        outputs,
        audio_prompt_len=prompt_len,
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

    audio = np.squeeze(
        np.asarray(
            audio,
            dtype=np.float32,
        )
    )

    if audio.ndim != 1:

        raise RuntimeError(
            f"Unexpected conditioned "
            f"audio shape: {audio.shape}"
        )

    if len(audio) == 0:

        raise RuntimeError(
            "Dia returned empty conditioned audio."
        )

    if not np.all(
        np.isfinite(audio)
    ):

        raise RuntimeError(
            "Generated audio contains NaN/Inf."
        )

    return audio


# =============================================================================
# CUSTOM STREAM FRAME
# =============================================================================

def make_frame(
    metadata: dict,
    pcm_bytes: bytes = b"",
):

    metadata_bytes = json.dumps(
        metadata
    ).encode(
        "utf-8"
    )

    return (
        struct.pack(
            ">I",
            len(metadata_bytes),
        )
        +
        metadata_bytes
        +
        struct.pack(
            ">Q",
            len(pcm_bytes),
        )
        +
        pcm_bytes
    )


# =============================================================================
# MODEL STARTUP
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

    print("=" * 80)

    started = (
        time.perf_counter_ns()
    )

    processor = (
        AutoProcessor
        .from_pretrained(
            MODEL_ID
        )
    )

    model = (
        DiaForConditionalGeneration
        .from_pretrained(
            MODEL_ID,
            torch_dtype=DTYPE,
            low_cpu_mem_usage=True,
        )
        .to(
            DEVICE
        )
    )

    model.eval()

    sync_cuda()

    model_load_ms = (
        time.perf_counter_ns()
        - started
    ) / 1_000_000

    print(
        f"Model loaded      : "
        f"{model_load_ms:.2f} ms"
    )

    print(
        f"Model loaded      : "
        f"{model_load_ms / 1000:.2f} sec"
    )

    print("=" * 80)


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

        "endpoints": {
            "/tts_seed":
                "native seed auditioning",

            "/tts":
                "reference-conditioned long TTS",
        },
    }

    if torch.cuda.is_available():

        result["gpu"] = (
            torch.cuda.get_device_name(0)
        )

    return result


# =============================================================================
# /tts_seed
#
# JSON request:
#
# {
#   "text": "...",
#   "seed": 1234,
#   "max_new_tokens": 1024
# }
# =============================================================================

@app.post("/tts_seed")
def tts_seed(
    req: SeedTTSRequest,
):

    if (
        model is None
        or processor is None
    ):

        raise HTTPException(
            status_code=503,
            detail="Model is not loaded yet.",
        )

    request_id = str(
        uuid.uuid4()
    )

    server_start = (
        time.perf_counter_ns()
    )

    try:

        set_seed(
            req.seed
        )

        # =====================================================================
        # PREPROCESS
        # =====================================================================

        preprocess_start = (
            time.perf_counter_ns()
        )

        inputs = processor(
            text=[
                normalize_tags(
                    req.text
                )
            ],
            padding=True,
            return_tensors="pt",
        ).to(
            model.device
        )

        sync_cuda()

        preprocess_ms = (
            time.perf_counter_ns()
            - preprocess_start
        ) / 1_000_000

        # =====================================================================
        # INFERENCE
        # =====================================================================

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

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
        # DECODE
        # =====================================================================

        decode_start = (
            time.perf_counter_ns()
        )

        audio = decode_plain_audio(
            outputs
        )

        decode_ms = (
            time.perf_counter_ns()
            - decode_start
        ) / 1_000_000

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

        server_total_ms = (
            time.perf_counter_ns()
            - server_start
        ) / 1_000_000

        generation_rtf = (
            (
                inference_ms
                / 1000.0
            )
            / audio_duration_s

            if audio_duration_s > 0

            else 0.0
        )

        gpu_allocated = 0.0
        gpu_reserved = 0.0
        gpu_peak = 0.0

        if torch.cuda.is_available():

            gpu_allocated = (
                torch.cuda.memory_allocated()
                / 1024**2
            )

            gpu_reserved = (
                torch.cuda.memory_reserved()
                / 1024**2
            )

            gpu_peak = (
                torch.cuda.max_memory_allocated()
                / 1024**2
            )

        print("")
        print("=" * 80)
        print("SEED TTS")
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
            f"Inference         : "
            f"{inference_ms:.2f} ms"
        )

        print(
            f"Audio duration    : "
            f"{audio_duration_s:.2f} sec"
        )

        print("=" * 80)

        headers = {

            "X-Request-ID":
                request_id,

            "X-Seed":
                str(req.seed),

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

            "X-Sample-Rate":
                str(SAMPLE_RATE),

            "X-GPU-Allocated-MB":
                f"{gpu_allocated:.2f}",

            "X-GPU-Reserved-MB":
                f"{gpu_reserved:.2f}",

            "X-GPU-Peak-MB":
                f"{gpu_peak:.2f}",
        }

        return Response(
            content=wav_bytes,
            media_type="audio/wav",
            headers=headers,
        )

    except Exception as exc:

        if torch.cuda.is_available():

            torch.cuda.empty_cache()

        raise HTTPException(
            status_code=500,
            detail=(
                "Seed TTS failed: "
                f"{type(exc).__name__}: "
                f"{exc}"
            ),
        ) from exc


# =============================================================================
# REFERENCE-CONDITIONED GENERATOR
# =============================================================================

def generate_reference_blocks(
    full_text: str,
    reference_text: str,
    reference_audio: np.ndarray,
    max_new_tokens: int,
    request_id: str,
):

    server_start = (
        time.perf_counter_ns()
    )

    full_text = normalize_tags(
        full_text
    )

    reference_text = normalize_tags(
        reference_text
    )

    blocks = build_dialogue_blocks(
        full_text
    )

    total_preprocess_ms = 0.0
    total_inference_ms = 0.0
    total_decode_ms = 0.0
    total_audio_samples = 0

    if torch.cuda.is_available():

        torch.cuda.reset_peak_memory_stats()

    print("")
    print("=" * 80)
    print("REFERENCE-CONDITIONED TTS")
    print("=" * 80)

    print(
        f"Request ID        : "
        f"{request_id}"
    )

    print(
        f"Blocks            : "
        f"{len(blocks)}"
    )

    print(
        f"Reference duration: "
        f"{len(reference_audio) / SAMPLE_RATE:.2f}s"
    )

    print("=" * 80)

    for block_index, block_text in enumerate(
        blocks,
        start=1,
    ):

        print("")
        print("-" * 80)

        print(
            f"Block "
            f"{block_index}/"
            f"{len(blocks)}"
        )

        print(
            block_text
        )

        # Reference transcript must be before the new text.
        conditioned_text = (
            reference_text
            + " "
            + block_text
        )

        # =====================================================================
        # PREPROCESS
        # =====================================================================

        preprocess_start = (
            time.perf_counter_ns()
        )

        inputs = processor(
            text=[
                conditioned_text
            ],
            audio=reference_audio,
            padding=True,
            return_tensors="pt",
        ).to(
            model.device
        )

        prompt_len = (
            processor
            .get_audio_prompt_len(
                inputs[
                    "decoder_attention_mask"
                ]
            )
        )

        sync_cuda()

        preprocess_ms = (
            time.perf_counter_ns()
            - preprocess_start
        ) / 1_000_000

        total_preprocess_ms += (
            preprocess_ms
        )

        # =====================================================================
        # INFERENCE
        # =====================================================================

        inference_start = (
            time.perf_counter_ns()
        )

        with torch.inference_mode():

            outputs = model.generate(
                **inputs,
                max_new_tokens=
                    max_new_tokens,
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

        total_inference_ms += (
            inference_ms
        )

        # =====================================================================
        # DECODE NEW AUDIO ONLY
        # =====================================================================

        decode_start = (
            time.perf_counter_ns()
        )

        audio = (
            decode_conditioned_audio(
                outputs,
                prompt_len,
            )
        )

        decode_ms = (
            time.perf_counter_ns()
            - decode_start
        ) / 1_000_000

        total_decode_ms += (
            decode_ms
        )

        total_audio_samples += (
            len(audio)
        )

        audio_duration_s = (
            len(audio)
            / SAMPLE_RATE
        )

        # =====================================================================
        # PCM16
        # =====================================================================

        pcm = np.clip(
            audio,
            -1.0,
            1.0,
        )

        pcm = (
            pcm * 32767.0
        ).astype(
            "<i2"
        )

        pcm_bytes = (
            pcm.tobytes()
        )

        print(
            f"Inference         : "
            f"{inference_ms:.2f} ms"
        )

        print(
            f"Audio duration    : "
            f"{audio_duration_s:.2f}s"
        )

        # Send block immediately.
        yield make_frame(
            {
                "type":
                    "audio",

                "request_id":
                    request_id,

                "block_index":
                    block_index,

                "block_count":
                    len(blocks),

                "sample_rate":
                    SAMPLE_RATE,

                "channels":
                    1,

                "audio_duration_s":
                    audio_duration_s,

                "preprocess_ms":
                    preprocess_ms,

                "inference_ms":
                    inference_ms,

                "decode_ms":
                    decode_ms,
            },

            pcm_bytes,
        )

    # =========================================================================
    # FINAL
    # =========================================================================

    server_total_ms = (
        time.perf_counter_ns()
        - server_start
    ) / 1_000_000

    audio_duration_s = (
        total_audio_samples
        / SAMPLE_RATE
    )

    generation_rtf = (
        (
            total_inference_ms
            / 1000.0
        )
        / audio_duration_s

        if audio_duration_s > 0

        else 0.0
    )

    total_rtf = (
        (
            server_total_ms
            / 1000.0
        )
        / audio_duration_s

        if audio_duration_s > 0

        else 0.0
    )

    gpu_allocated = 0.0
    gpu_reserved = 0.0
    gpu_peak = 0.0

    if torch.cuda.is_available():

        gpu_allocated = (
            torch.cuda.memory_allocated()
            / 1024**2
        )

        gpu_reserved = (
            torch.cuda.memory_reserved()
            / 1024**2
        )

        gpu_peak = (
            torch.cuda.max_memory_allocated()
            / 1024**2
        )

    yield make_frame(
        {
            "type":
                "end",

            "request_id":
                request_id,

            "block_count":
                len(blocks),

            "preprocess_ms":
                total_preprocess_ms,

            "inference_ms":
                total_inference_ms,

            "decode_ms":
                total_decode_ms,

            "server_total_ms":
                server_total_ms,

            "audio_duration_s":
                audio_duration_s,

            "generation_rtf":
                generation_rtf,

            "total_rtf":
                total_rtf,

            "gpu_allocated_mb":
                gpu_allocated,

            "gpu_reserved_mb":
                gpu_reserved,

            "gpu_peak_mb":
                gpu_peak,
        }
    )


# =============================================================================
# /tts
#
# multipart/form-data:
#
# text
# reference_text
# max_new_tokens
# reference_audio
# =============================================================================

@app.post("/tts")
def tts_reference(
    text: str = Form(...),

    reference_text: str = Form(...),

    max_new_tokens: int = Form(
        DEFAULT_REFERENCE_MAX_NEW_TOKENS
    ),

    reference_audio: UploadFile = File(...),
):

    if (
        model is None
        or processor is None
    ):

        raise HTTPException(
            status_code=503,
            detail="Model is not loaded yet.",
        )

    if not (
        256
        <= max_new_tokens
        <= 4096
    ):

        raise HTTPException(
            status_code=400,
            detail=(
                "max_new_tokens must be "
                "between 256 and 4096."
            ),
        )

    request_id = str(
        uuid.uuid4()
    )

    try:

        reference_bytes = (
            reference_audio.file.read()
        )

        if not reference_bytes:

            raise ValueError(
                "Uploaded reference audio is empty."
            )

        (
            reference_array,
            reference_duration,
        ) = load_reference_audio(
            reference_bytes
        )

        print("")
        print(
            f"[reference] File     : "
            f"{reference_audio.filename}"
        )

        print(
            f"[reference] Duration : "
            f"{reference_duration:.2f}s"
        )

        return StreamingResponse(
            generate_reference_blocks(
                full_text=text,

                reference_text=
                    reference_text,

                reference_audio=
                    reference_array,

                max_new_tokens=
                    max_new_tokens,

                request_id=
                    request_id,
            ),

            media_type=(
                "application/"
                "x-dia-reference-stream"
            ),

            headers={
                "X-Stream-Mode":
                    "reference-pcm",

                "X-Request-ID":
                    request_id,

                "X-Sample-Rate":
                    str(SAMPLE_RATE),
            },
        )

    except Exception as exc:

        if torch.cuda.is_available():

            torch.cuda.empty_cache()

        print(
            f"[ERROR] "
            f"{type(exc).__name__}: "
            f"{exc}"
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "Reference TTS failed: "
                f"{type(exc).__name__}: "
                f"{exc}"
            ),
        ) from exc
