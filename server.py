import io
import json
import os
import re
import struct
import time
import uuid

import numpy as np
import soundfile as sf
import torch

from fastapi import FastAPI, HTTPException
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, Field
from transformers import AutoProcessor, DiaForConditionalGeneration


# =============================================================================
# CONFIGURATION
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
# LONG-TEXT CONFIG
# =============================================================================

# Anything below this stays on the OLD / fast single-generation path.
SHORT_TEXT_MAX_WORDS = 45

# Long conversations are split into approximately this many words/chunk.
MAX_WORDS_PER_CHUNK = 40

# Token budget PER generation.
DEFAULT_MAX_NEW_TOKENS = 2048


# =============================================================================
# FASTAPI
# =============================================================================

app = FastAPI(
    title="Dia TTS API",
    version="2.0.0",
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
        le=4096,
    )

    seed: int = Field(
        default=1234,
        ge=0,
        le=2147483647,
    )


# =============================================================================
# CUDA
# =============================================================================

def sync_cuda():

    if DEVICE == "cuda":
        torch.cuda.synchronize()


def set_seed(seed: int):

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


# =============================================================================
# TEXT HELPERS
# =============================================================================

def normalize_tags(text: str):

    # [s1], [ S1 ], etc -> [S1]

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

    # Normalize whitespace.
    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


def count_words_without_tags(text: str):

    clean = re.sub(
        r"\[(S1|S2)\]",
        "",
        text,
        flags=re.IGNORECASE,
    )

    return len(
        clean.split()
    )


def parse_turns(text: str):

    text = normalize_tags(text)

    pattern = r"\[(S1|S2)\]\s*"

    matches = list(
        re.finditer(
            pattern,
            text,
            flags=re.IGNORECASE,
        )
    )

    if not matches:

        raise ValueError(
            "Transcript must contain [S1] and/or [S2] tags."
        )

    turns = []

    for i, match in enumerate(matches):

        speaker = (
            match.group(1).upper()
        )

        start = match.end()

        if i + 1 < len(matches):

            end = (
                matches[i + 1].start()
            )

        else:

            end = len(text)

        speech = text[
            start:end
        ].strip()

        if speech:

            turns.append(
                (
                    speaker,
                    speech,
                )
            )

    return turns


def split_sentences(text: str):

    return [
        sentence.strip()
        for sentence in re.split(
            r"(?<=[.!?])\s+",
            text,
        )
        if sentence.strip()
    ]


def split_long_turn(
    speaker: str,
    speech: str,
):

    # Normal turn.
    if (
        len(speech.split())
        <= MAX_WORDS_PER_CHUNK
    ):

        return [
            (
                speaker,
                speech,
            )
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

        # -------------------------------------------------------------
        # A single sentence itself is huge.
        # -------------------------------------------------------------

        if (
            sentence_words
            > MAX_WORDS_PER_CHUNK
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

            for i in range(
                0,
                len(words),
                MAX_WORDS_PER_CHUNK,
            ):

                pieces.append(
                    (
                        speaker,
                        " ".join(
                            words[
                                i:
                                i + MAX_WORDS_PER_CHUNK
                            ]
                        ),
                    )
                )

            continue

        # -------------------------------------------------------------
        # Current chunk would become too large.
        # -------------------------------------------------------------

        if (
            current
            and
            current_words + sentence_words
            > MAX_WORDS_PER_CHUNK
        ):

            pieces.append(
                (
                    speaker,
                    " ".join(current),
                )
            )

            current = []
            current_words = 0

        current.append(
            sentence
        )

        current_words += (
            sentence_words
        )

    if current:

        pieces.append(
            (
                speaker,
                " ".join(current),
            )
        )

    return pieces


def build_dialogue_chunks(text: str):

    original_turns = parse_turns(
        text
    )

    expanded_turns = []

    for speaker, speech in original_turns:

        expanded_turns.extend(
            split_long_turn(
                speaker,
                speech,
            )
        )

    chunks = []

    current_turns = []
    current_words = 0

    for speaker, speech in expanded_turns:

        word_count = len(
            speech.split()
        )

        # -------------------------------------------------------------
        # Close current chunk before exceeding the target.
        # -------------------------------------------------------------

        if (
            current_turns
            and
            current_words + word_count
            > MAX_WORDS_PER_CHUNK
        ):

            chunks.append(
                current_turns
            )

            current_turns = []
            current_words = 0

        current_turns.append(
            (
                speaker,
                speech,
            )
        )

        current_words += (
            word_count
        )

        # -------------------------------------------------------------
        # Prefer boundary after customer response.
        # -------------------------------------------------------------

        if (
            speaker == "S2"
            and
            current_words >= 25
        ):

            chunks.append(
                current_turns
            )

            current_turns = []
            current_words = 0

    if current_turns:

        chunks.append(
            current_turns
        )

    final_chunks = []

    for chunk in chunks:

        chunk_text = " ".join(

            f"[{speaker}] {speech}"

            for speaker, speech
            in chunk
        )

        final_chunks.append(
            chunk_text
        )

    return final_chunks


# =============================================================================
# AUDIO HELPERS
# =============================================================================

def decode_audio(outputs):

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
            f"Unexpected audio shape: {audio.shape}"
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
            "Generated audio contains NaN/Inf."
        )

    return audio


# =============================================================================
# CORE DIA INFERENCE
# =============================================================================

def generate_audio(
    text: str,
    seed: int,
    max_new_tokens: int,
):

    # =================================================================
    # Seed
    # =================================================================

    set_seed(
        seed
    )

    # =================================================================
    # Preprocess
    # =================================================================

    preprocess_start = (
        time.perf_counter_ns()
    )

    inputs = processor(
        text=[
            text
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

    # =================================================================
    # Inference
    # =================================================================

    inference_start = (
        time.perf_counter_ns()
    )

    with torch.inference_mode():

        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
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

    # =================================================================
    # Decode
    # =================================================================

    decode_start = (
        time.perf_counter_ns()
    )

    audio = decode_audio(
        outputs
    )

    decode_ms = (
        time.perf_counter_ns()
        - decode_start
    ) / 1_000_000

    return {
        "audio":
            audio,

        "preprocess_ms":
            preprocess_ms,

        "inference_ms":
            inference_ms,

        "decode_ms":
            decode_ms,
    }


# =============================================================================
# STREAM FRAME
# =============================================================================

def make_frame(
    metadata: dict,
    pcm_bytes: bytes = b"",
):

    metadata_bytes = (
        json.dumps(
            metadata
        ).encode(
            "utf-8"
        )
    )

    # 4 bytes metadata length
    # metadata
    # 8 bytes audio length
    # PCM bytes

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

    print(
        "Speaker mapping   : "
        "[S1]=Agent, [S2]=Customer"
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
        f"[startup] Model loaded in "
        f"{model_load_ms:.2f} ms"
    )

    print(
        f"[startup] Model loaded in "
        f"{model_load_ms / 1000:.2f} sec"
    )

    if torch.cuda.is_available():

        print(
            f"[startup] GPU allocated: "
            f"{torch.cuda.memory_allocated() / 1024**2:.2f} MB"
        )

        print(
            f"[startup] GPU reserved : "
            f"{torch.cuda.memory_reserved() / 1024**2:.2f} MB"
        )


# =============================================================================
# HEALTH
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

        "speaker_mapping": {
            "S1": "agent",
            "S2": "customer",
        },
    }


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

        "short_text_max_words":
            SHORT_TEXT_MAX_WORDS,

        "long_chunk_max_words":
            MAX_WORDS_PER_CHUNK,

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
# SHORT TEXT
#
# Same approach as your old code.
# =============================================================================

def short_tts(
    req: TTSRequest,
    request_id: str,
):

    server_start = (
        time.perf_counter_ns()
    )

    if torch.cuda.is_available():

        torch.cuda.reset_peak_memory_stats()

    result = generate_audio(
        text=normalize_tags(
            req.text
        ),
        seed=req.seed,
        max_new_tokens=req.max_new_tokens,
    )

    audio = result[
        "audio"
    ]

    preprocess_ms = result[
        "preprocess_ms"
    ]

    inference_ms = result[
        "inference_ms"
    ]

    decode_ms = result[
        "decode_ms"
    ]

    audio_duration_s = (
        len(audio)
        / SAMPLE_RATE
    )

    # =================================================================
    # Encode WAV
    # =================================================================

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

    # =================================================================
    # Metrics
    # =================================================================

    server_total_ms = (
        time.perf_counter_ns()
        - server_start
    ) / 1_000_000

    generation_rtf = (
        (
            inference_ms / 1000
        )
        / audio_duration_s
        if audio_duration_s > 0
        else 0
    )

    total_rtf = (
        (
            server_total_ms / 1000
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

    print("")
    print("=" * 80)
    print("SERVER LATENCY - SHORT")
    print("=" * 80)

    print(
        f"Request ID        : {request_id}"
    )

    print(
        f"Seed              : {req.seed}"
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
        f"WAV encode        : "
        f"{encode_ms:.2f} ms"
    )

    print(
        f"SERVER TOTAL      : "
        f"{server_total_ms:.2f} ms"
    )

    print(
        f"Audio duration    : "
        f"{audio_duration_s:.3f}s"
    )

    print(
        f"Generation RTF    : "
        f"{generation_rtf:.4f}"
    )

    print("=" * 80)

    headers = {

        "X-Stream-Mode":
            "wav",

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

    return Response(
        content=wav_bytes,
        media_type="audio/wav",
        headers=headers,
    )


# =============================================================================
# LONG TEXT STREAM
# =============================================================================

def long_tts_generator(
    req: TTSRequest,
    request_id: str,
):

    server_start = (
        time.perf_counter_ns()
    )

    chunks = build_dialogue_chunks(
        req.text
    )

    print("")
    print("=" * 80)
    print("LONG DIA REQUEST")
    print("=" * 80)

    print(
        f"Request ID        : {request_id}"
    )

    print(
        f"Seed              : {req.seed}"
    )

    print(
        f"Chunks            : {len(chunks)}"
    )

    print("=" * 80)

    total_preprocess_ms = 0
    total_inference_ms = 0
    total_decode_ms = 0

    total_audio_samples = 0

    if torch.cuda.is_available():

        torch.cuda.reset_peak_memory_stats()

    for index, chunk_text in enumerate(
        chunks,
        start=1,
    ):

        print("")
        print("-" * 80)

        print(
            f"Generating chunk "
            f"{index}/{len(chunks)}"
        )

        print(
            f"Text: {chunk_text}"
        )

        # -------------------------------------------------------------
        # Same seed for every conversation chunk.
        # -------------------------------------------------------------

        result = generate_audio(
            text=chunk_text,
            seed=req.seed,
            max_new_tokens=req.max_new_tokens,
        )

        audio = result[
            "audio"
        ]

        total_preprocess_ms += (
            result["preprocess_ms"]
        )

        total_inference_ms += (
            result["inference_ms"]
        )

        total_decode_ms += (
            result["decode_ms"]
        )

        total_audio_samples += (
            len(audio)
        )

        chunk_duration_s = (
            len(audio)
            / SAMPLE_RATE
        )

        # =================================================================
        # Convert float32 -> PCM16
        # =================================================================

        pcm = np.clip(
            audio,
            -1.0,
            1.0,
        )

        pcm = (
            pcm
            * 32767.0
        ).astype(
            "<i2"
        )

        pcm_bytes = (
            pcm.tobytes()
        )

        print(
            f"Chunk {index} inference : "
            f"{result['inference_ms']:.2f} ms"
        )

        print(
            f"Chunk {index} audio     : "
            f"{chunk_duration_s:.2f}s"
        )

        # =================================================================
        # Send this chunk immediately.
        # =================================================================

        yield make_frame(

            {
                "type":
                    "audio",

                "request_id":
                    request_id,

                "chunk_index":
                    index,

                "chunk_count":
                    len(chunks),

                "sample_rate":
                    SAMPLE_RATE,

                "channels":
                    1,

                "sample_width":
                    2,

                "audio_duration_s":
                    chunk_duration_s,

                "preprocess_ms":
                    result["preprocess_ms"],

                "inference_ms":
                    result["inference_ms"],

                "decode_ms":
                    result["decode_ms"],
            },

            pcm_bytes,
        )

    # =============================================================================
    # Final metrics
    # =============================================================================

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

    gpu_allocated_mb = (
        torch.cuda.memory_allocated()
        / 1024**2
        if torch.cuda.is_available()
        else 0
    )

    gpu_reserved_mb = (
        torch.cuda.memory_reserved()
        / 1024**2
        if torch.cuda.is_available()
        else 0
    )

    gpu_peak_mb = (
        torch.cuda.max_memory_allocated()
        / 1024**2
        if torch.cuda.is_available()
        else 0
    )

    print("")
    print("=" * 80)
    print("SERVER LATENCY - LONG")
    print("=" * 80)

    print(
        f"Chunks            : {len(chunks)}"
    )

    print(
        f"Preprocess        : "
        f"{total_preprocess_ms:.2f} ms"
    )

    print(
        f"Inference         : "
        f"{total_inference_ms:.2f} ms"
    )

    print(
        f"Decode            : "
        f"{total_decode_ms:.2f} ms"
    )

    print(
        f"SERVER TOTAL      : "
        f"{server_total_ms:.2f} ms"
    )

    print(
        f"Audio duration    : "
        f"{audio_duration_s:.3f}s"
    )

    print(
        f"Generation RTF    : "
        f"{generation_rtf:.4f}"
    )

    print(
        f"Total RTF         : "
        f"{total_rtf:.4f}"
    )

    print("=" * 80)

    # =============================================================================
    # End frame
    # =============================================================================

    yield make_frame(
        {
            "type":
                "end",

            "request_id":
                request_id,

            "seed":
                req.seed,

            "chunk_count":
                len(chunks),

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
                gpu_allocated_mb,

            "gpu_reserved_mb":
                gpu_reserved_mb,

            "gpu_peak_mb":
                gpu_peak_mb,
        }
    )


# =============================================================================
# /tts
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

    try:

        normalized_text = normalize_tags(
            req.text
        )

        word_count = (
            count_words_without_tags(
                normalized_text
            )
        )

        print("")
        print(
            f"[request] Words={word_count}, "
            f"seed={req.seed}"
        )

        # =================================================================
        # SHORT TEXT = OLD FAST PATH
        # =================================================================

        if (
            word_count
            <= SHORT_TEXT_MAX_WORDS
        ):

            print(
                "[request] Using SHORT single-generation mode"
            )

            return short_tts(
                req,
                request_id,
            )

        # =================================================================
        # LONG TEXT = CHUNK STREAM
        # =================================================================

        print(
            "[request] Using LONG streaming mode"
        )

        return StreamingResponse(
            long_tts_generator(
                req,
                request_id,
            ),
            media_type="application/x-dia-pcm-stream",
            headers={
                "X-Stream-Mode":
                    "pcm",

                "X-Request-ID":
                    request_id,

                "X-Seed":
                    str(req.seed),

                "X-Sample-Rate":
                    str(SAMPLE_RATE),
            },
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
