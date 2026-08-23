import io
import json
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
from fastapi.responses import StreamingResponse
from scipy.signal import resample_poly
from transformers import (
    AutoProcessor,
    DiaForConditionalGeneration,
)


# =============================================================================
# CONFIG
# =============================================================================

MODEL_ID = "nari-labs/Dia-1.6B-0626"

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

DTYPE = (
    torch.float16
    if DEVICE == "cuda"
    else torch.float32
)

SAMPLE_RATE = 44100


# =============================================================================
# LONG-TEXT CONFIG
# =============================================================================

# Target moderate dialogue blocks.
#
# We do NOT split every sentence.
# We preserve multiple S1/S2 turns together.
TARGET_WORDS_PER_BLOCK = 35

# Harder upper target.
MAX_WORDS_PER_BLOCK = 55

# Per block, NOT entire transcript.
DEFAULT_MAX_NEW_TOKENS = 3072


# =============================================================================
# APP
# =============================================================================

app = FastAPI(
    title="Dia Reference Conditioned TTS",
    version="5.0.0",
)

processor = None
model = None
model_load_ms = None


# =============================================================================
# CUDA
# =============================================================================

def sync_cuda():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


# =============================================================================
# TEXT NORMALIZATION
# =============================================================================

def normalize_tags(text: str) -> str:

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
            "Transcript must contain [S1] and [S2] tags."
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


# =============================================================================
# LONG TURN SPLITTING
# =============================================================================

def split_sentences(text: str):

    return [
        value.strip()
        for value in re.split(
            r"(?<=[.!?])\s+",
            text,
        )
        if value.strip()
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

    output = []

    current = []
    current_words = 0

    for sentence in sentences:

        sentence_words = len(
            sentence.split()
        )

        # One sentence itself is extremely large.
        if (
            sentence_words
            > MAX_WORDS_PER_BLOCK
        ):

            if current:

                output.append(
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
                MAX_WORDS_PER_BLOCK,
            ):

                output.append(
                    (
                        speaker,
                        " ".join(
                            words[
                                i:
                                i + MAX_WORDS_PER_BLOCK
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

            output.append(
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

        output.append(
            (
                speaker,
                " ".join(current),
            )
        )

    return output


# =============================================================================
# DIALOGUE-AWARE BLOCKING
# =============================================================================

def build_dialogue_blocks(text: str):
    """
    Critical goals:

    1. Keep several S1/S2 turns together.
    2. Prefer starting a new block with S1.
    3. Avoid arbitrarily cutting a sentence.
    4. Avoid one enormous Dia generation.
    """

    original_turns = parse_turns(
        text
    )

    turns = []

    for speaker, speech in original_turns:

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

        # -------------------------------------------------------------
        # Best boundary:
        #
        # We already have sufficient text and next turn is S1.
        # This means the next block naturally starts with S1.
        # -------------------------------------------------------------

        if (
            current
            and
            speaker == "S1"
            and
            current_words
            >= TARGET_WORDS_PER_BLOCK
        ):

            blocks.append(
                current
            )

            current = []
            current_words = 0

        # -------------------------------------------------------------
        # Hard safety check.
        #
        # Still prefer breaking before S1.
        # -------------------------------------------------------------

        elif (
            current
            and
            speaker == "S1"
            and
            current_words + word_count
            > MAX_WORDS_PER_BLOCK
        ):

            blocks.append(
                current
            )

            current = []
            current_words = 0

        current.append(
            (speaker, speech)
        )

        current_words += word_count

    if current:

        blocks.append(
            current
        )

    formatted_blocks = []

    for block in blocks:

        formatted_blocks.append(
            " ".join(
                f"[{speaker}] {speech}"
                for speaker, speech
                in block
            )
        )

    return formatted_blocks


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

    # Stereo -> mono.
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

    # Dia uses 44.1kHz audio.
    if sample_rate != SAMPLE_RATE:

        divisor = gcd(
            sample_rate,
            SAMPLE_RATE,
        )

        audio = resample_poly(
            audio,
            SAMPLE_RATE // divisor,
            sample_rate // divisor,
        )

        audio = np.asarray(
            audio,
            dtype=np.float32,
        )

    duration = (
        len(audio)
        / SAMPLE_RATE
    )

    return audio, duration


# =============================================================================
# GENERATED AUDIO DECODING
# =============================================================================

def decode_generated_audio(
    outputs,
    prompt_len,
):

    decoded = processor.batch_decode(
        outputs,
        audio_prompt_len=prompt_len,
    )

    if isinstance(
        decoded,
        (list, tuple),
    ):

        audio = decoded[0]

    else:

        audio = decoded

    if torch.is_tensor(audio):

        audio = (
            audio
            .detach()
            .float()
            .cpu()
            .numpy()
        )

    audio = np.asarray(
        audio,
        dtype=np.float32,
    )

    audio = np.squeeze(audio)

    if audio.ndim != 1:

        raise RuntimeError(
            f"Unexpected decoded audio shape: "
            f"{audio.shape}"
        )

    if len(audio) == 0:

        raise RuntimeError(
            "Dia returned empty generated audio."
        )

    if not np.all(
        np.isfinite(audio)
    ):

        raise RuntimeError(
            "Generated audio contains NaN/Inf."
        )

    return audio


# =============================================================================
# CUSTOM STREAM FRAMING
# =============================================================================

def make_frame(
    metadata: dict,
    pcm_bytes: bytes = b"",
):
    """
    Frame:

    4 bytes : JSON metadata length
    N bytes : JSON
    8 bytes : PCM payload length
    M bytes : PCM16 audio
    """

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

    data = {

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

        "sample_rate":
            SAMPLE_RATE,

        "generation_mode":
            "uploaded-reference-conditioned",

        "speaker_mapping": {
            "S1": "agent",
            "S2": "customer",
        },
    }

    if torch.cuda.is_available():

        data["gpu"] = (
            torch.cuda.get_device_name(0)
        )

    return data


# =============================================================================
# GENERATOR
# =============================================================================

def generate_blocks(
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

    print("")
    print("=" * 80)
    print("REFERENCE-CONDITIONED REQUEST")
    print("=" * 80)

    print(
        f"Request ID        : {request_id}"
    )

    print(
        f"Blocks            : {len(blocks)}"
    )

    print(
        f"Reference duration: "
        f"{len(reference_audio) / SAMPLE_RATE:.2f}s"
    )

    print(
        f"Max new tokens    : "
        f"{max_new_tokens}"
    )

    print("=" * 80)

    total_preprocess_ms = 0.0
    total_inference_ms = 0.0
    total_decode_ms = 0.0

    total_samples = 0

    if torch.cuda.is_available():

        torch.cuda.reset_peak_memory_stats()

    # =========================================================================
    # GENERATE EACH MODERATE BLOCK
    # =========================================================================

    for block_index, block_text in enumerate(
        blocks,
        start=1,
    ):

        print("")
        print("-" * 80)

        print(
            f"Block "
            f"{block_index}/{len(blocks)}"
        )

        print(
            block_text
        )

        # ---------------------------------------------------------------------
        # Important:
        #
        # Reference transcript corresponds to reference_audio.
        # New dialogue is appended after it.
        # ---------------------------------------------------------------------

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
        )

        inputs = inputs.to(
            model.device
        )

        # Official Dia audio-conditioning mechanism.
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

        total_inference_ms += (
            inference_ms
        )

        # =====================================================================
        # DECODE NEW AUDIO ONLY
        # =====================================================================

        decode_start = (
            time.perf_counter_ns()
        )

        audio = decode_generated_audio(
            outputs,
            prompt_len,
        )

        decode_ms = (
            time.perf_counter_ns()
            - decode_start
        ) / 1_000_000

        total_decode_ms += (
            decode_ms
        )

        total_samples += len(audio)

        duration = (
            len(audio)
            / SAMPLE_RATE
        )

        # =====================================================================
        # FLOAT32 -> PCM16
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
            f"{duration:.2f} sec"
        )

        # =====================================================================
        # SEND BLOCK IMMEDIATELY
        # =====================================================================

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
                    duration,

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
    # FINAL METRICS
    # =========================================================================

    server_total_ms = (
        time.perf_counter_ns()
        - server_start
    ) / 1_000_000

    audio_duration_s = (
        total_samples
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

    print("")
    print("=" * 80)
    print("SERVER TOTAL")
    print("=" * 80)

    print(
        f"Blocks            : "
        f"{len(blocks)}"
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
        f"{audio_duration_s:.2f} sec"
    )

    print(
        f"Generation RTF    : "
        f"{generation_rtf:.4f}"
    )

    print("=" * 80)

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
# TTS API
# =============================================================================

@app.post("/tts")
def tts(
    text: str = Form(...),
    reference_text: str = Form(...),
    max_new_tokens: int = Form(
        DEFAULT_MAX_NEW_TOKENS
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

    if (
        max_new_tokens < 256
        or
        max_new_tokens > 4096
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

        # =====================================================================
        # LOAD UPLOADED REFERENCE WAV
        # =====================================================================

        reference_bytes = (
            reference_audio.file.read()
        )

        if not reference_bytes:

            raise ValueError(
                "Uploaded reference audio is empty."
            )

        reference_array, reference_duration = (
            load_reference_audio(
                reference_bytes
            )
        )

        print("")
        print(
            f"[request] Reference file: "
            f"{reference_audio.filename}"
        )

        print(
            f"[request] Reference duration: "
            f"{reference_duration:.2f}s"
        )

        return StreamingResponse(
            generate_blocks(
                full_text=text,
                reference_text=reference_text,
                reference_audio=reference_array,
                max_new_tokens=max_new_tokens,
                request_id=request_id,
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
                "TTS generation failed: "
                f"{type(exc).__name__}: "
                f"{exc}"
            ),
        ) from exc
