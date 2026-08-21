import io
import os
import re
import time
import uuid
import threading

import numpy as np
import soundfile as sf
import torch

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from transformers import AutoProcessor, DiaForConditionalGeneration


# =============================================================================
# CONFIGURATION
# =============================================================================

MODEL_ID = os.getenv(
    "MODEL_ID",
    "nari-labs/Dia-1.6B-0626",
)

SAMPLE_RATE = 44100

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32


# Public speaker convention:
#
#   [S1] = AGENT
#   [S2] = CUSTOMER
#
# Internally each turn is generated independently so that we can use
# a different random seed for the two roles.

DEFAULT_AGENT_SEED = 103
DEFAULT_CUSTOMER_SEED = 217

MIN_SEED = 0
MAX_SEED = 2_147_483_647


# Dia's authors recommend moderate input lengths.
# Individual long turns are split before generation.
MAX_WORDS_PER_GENERATION = 42


# Official Transformers example uses 3072.
# Because each turn is generated separately, this is ample room without
# requiring the entire conversation to fit into one generation.
DEFAULT_MAX_NEW_TOKENS = 3072


# Silence between dialogue turns.
TURN_SILENCE_MS = 180


# Protect global RNG + GPU model from concurrent requests.
generation_lock = threading.Lock()


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

    agent_seed: int = Field(
        default=DEFAULT_AGENT_SEED,
        ge=MIN_SEED,
        le=MAX_SEED,
    )

    customer_seed: int = Field(
        default=DEFAULT_CUSTOMER_SEED,
        ge=MIN_SEED,
        le=MAX_SEED,
    )

    max_new_tokens: int = Field(
        default=DEFAULT_MAX_NEW_TOKENS,
        ge=256,
        le=4096,
    )


# =============================================================================
# CUDA HELPERS
# =============================================================================

def sync_cuda():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def set_seed(seed: int):

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


# =============================================================================
# TEXT HELPERS
# =============================================================================

def normalize_tags(text: str) -> str:
    """
    Converts:

        [s1]
        [ S1 ]
        [s2]

    into:

        [S1]
        [S2]
    """

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

    return text.strip()


def parse_turns(text: str):
    """
    Example:

        [S1] Hello.
        [S2] Hi.
        [S1] How can I help?

    becomes:

        [
            ("S1", "Hello."),
            ("S2", "Hi."),
            ("S1", "How can I help?")
        ]
    """

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
            "No dialogue was found after the speaker tags."
        )

    return turns


def split_long_sentence(sentence: str, max_words: int):
    """
    Last-resort split for a single very long sentence.
    """

    words = sentence.split()

    return [
        " ".join(
            words[i:i + max_words]
        )
        for i in range(
            0,
            len(words),
            max_words,
        )
    ]


def split_long_turn(speech: str):
    """
    Split one speaker turn into Dia-safe pieces.

    First try sentence boundaries.
    If one sentence itself is too large, split by word count.
    """

    if len(speech.split()) <= MAX_WORDS_PER_GENERATION:
        return [speech]

    sentences = re.split(
        r"(?<=[.!?])\s+",
        speech.strip(),
    )

    chunks = []

    current_sentences = []
    current_word_count = 0

    for sentence in sentences:

        sentence = sentence.strip()

        if not sentence:
            continue

        sentence_words = len(
            sentence.split()
        )

        # One single sentence is already too long.
        if sentence_words > MAX_WORDS_PER_GENERATION:

            if current_sentences:

                chunks.append(
                    " ".join(
                        current_sentences
                    )
                )

                current_sentences = []
                current_word_count = 0

            chunks.extend(
                split_long_sentence(
                    sentence,
                    MAX_WORDS_PER_GENERATION,
                )
            )

            continue

        # Current chunk + sentence too long.
        if (
            current_sentences
            and
            current_word_count + sentence_words
            > MAX_WORDS_PER_GENERATION
        ):

            chunks.append(
                " ".join(
                    current_sentences
                )
            )

            current_sentences = []
            current_word_count = 0

        current_sentences.append(
            sentence
        )

        current_word_count += (
            sentence_words
        )

    if current_sentences:

        chunks.append(
            " ".join(
                current_sentences
            )
        )

    return chunks


def create_generation_units(text: str):
    """
    Converts the entire conversation into small units.

    External roles remain:

        S1 = agent
        S2 = customer

    Each unit carries the original role and its text.
    """

    turns = parse_turns(text)

    units = []

    original_turn_number = 0

    for speaker, speech in turns:

        original_turn_number += 1

        pieces = split_long_turn(
            speech
        )

        for piece_index, piece in enumerate(
            pieces,
            start=1,
        ):

            units.append(
                {
                    "speaker": speaker,
                    "speech": piece,
                    "turn": original_turn_number,
                    "piece": piece_index,
                    "pieces": len(pieces),
                }
            )

    return units


# =============================================================================
# AUDIO HELPERS
# =============================================================================

def decode_generated_audio(outputs):

    decoded = processor.batch_decode(
        outputs
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

    audio = np.squeeze(
        audio
    )

    if audio.ndim != 1:

        raise RuntimeError(
            f"Unexpected generated audio shape: {audio.shape}"
        )

    if len(audio) == 0:

        raise RuntimeError(
            "Dia generated empty audio."
        )

    if not np.all(
        np.isfinite(audio)
    ):

        raise RuntimeError(
            "Generated audio contains NaN or Inf."
        )

    return audio


def concatenate_audio(parts):

    if not parts:
        raise RuntimeError(
            "No generated audio to concatenate."
        )

    silence_samples = int(
        SAMPLE_RATE
        * TURN_SILENCE_MS
        / 1000
    )

    silence = np.zeros(
        silence_samples,
        dtype=np.float32,
    )

    result = []

    for index, part in enumerate(parts):

        result.append(
            np.asarray(
                part,
                dtype=np.float32,
            )
        )

        if index < len(parts) - 1:
            result.append(
                silence
            )

    return np.concatenate(
        result
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
        f"Model              : {MODEL_ID}"
    )

    print(
        f"Device             : {DEVICE}"
    )

    print(
        f"PyTorch            : {torch.__version__}"
    )

    print(
        f"PyTorch CUDA       : {torch.version.cuda}"
    )

    print(
        f"CUDA available     : {torch.cuda.is_available()}"
    )

    if not torch.cuda.is_available():

        raise RuntimeError(
            "CUDA is not available. "
            "Check NVIDIA driver, Docker GPU runtime "
            "and PyTorch CUDA compatibility."
        )

    print(
        f"GPU                : {torch.cuda.get_device_name(0)}"
    )

    print(
        f"DTYPE              : {DTYPE}"
    )

    print(
        "Speaker mapping    : "
        "S1=agent, S2=customer"
    )

    print(
        f"Default agent seed : {DEFAULT_AGENT_SEED}"
    )

    print(
        f"Default customer   : {DEFAULT_CUSTOMER_SEED}"
    )

    print("=" * 80)

    started = time.perf_counter_ns()

    processor = AutoProcessor.from_pretrained(
        MODEL_ID,
    )

    model = DiaForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=DTYPE,
        low_cpu_mem_usage=True,
    ).to(DEVICE)

    model.eval()

    sync_cuda()

    model_load_ms = (
        time.perf_counter_ns()
        - started
    ) / 1_000_000

    print("")
    print(
        f"[startup] Model loaded in "
        f"{model_load_ms:.2f} ms"
    )

    print(
        f"[startup] Model loaded in "
        f"{model_load_ms / 1000:.2f} sec"
    )

    print(
        f"[startup] GPU allocated: "
        f"{torch.cuda.memory_allocated() / 1024**2:.2f} MB"
    )

    print(
        f"[startup] GPU reserved : "
        f"{torch.cuda.memory_reserved() / 1024**2:.2f} MB"
    )

    print("=" * 80)


# =============================================================================
# ROOT / HEALTH
# =============================================================================

@app.get("/")
def root():

    return {
        "service": "Dia TTS API",
        "model": MODEL_ID,
        "device": DEVICE,
        "speaker_mapping": {
            "S1": "agent",
            "S2": "customer",
        },
    }


@app.get("/health")
def health():

    result = {
        "status": (
            "ok"
            if model is not None
            else "loading"
        ),
        "model": MODEL_ID,
        "device": DEVICE,
        "cuda_available": torch.cuda.is_available(),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "model_load_ms": model_load_ms,
        "default_agent_seed": DEFAULT_AGENT_SEED,
        "default_customer_seed": DEFAULT_CUSTOMER_SEED,
        "max_words_per_generation": MAX_WORDS_PER_GENERATION,
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

    if model is None or processor is None:

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

        # =====================================================================
        # Parse the ENTIRE transcript
        # =====================================================================

        units = create_generation_units(
            req.text
        )

        print("")
        print("=" * 80)
        print("DIA TTS REQUEST")
        print("=" * 80)

        print(
            f"Request ID          : {request_id}"
        )

        print(
            f"Agent seed / S1     : {req.agent_seed}"
        )

        print(
            f"Customer seed / S2  : {req.customer_seed}"
        )

        print(
            f"Dialogue units      : {len(units)}"
        )

        print(
            f"Max new tokens/unit : {req.max_new_tokens}"
        )

        print("=" * 80)

        generated_parts = []

        total_preprocess_ms = 0.0
        total_inference_ms = 0.0
        total_decode_ms = 0.0

        # =====================================================================
        # IMPORTANT
        #
        # Seeds modify global RNG state. We keep all generation for the request
        # inside the same lock.
        # =====================================================================

        with generation_lock:

            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()

            for index, unit in enumerate(
                units,
                start=1,
            ):

                original_speaker = unit[
                    "speaker"
                ]

                speech = unit[
                    "speech"
                ]

                if original_speaker == "S1":

                    role = "AGENT"
                    seed = req.agent_seed

                else:

                    role = "CUSTOMER"
                    seed = req.customer_seed

                # =============================================================
                # Reset same role-specific seed for every utterance.
                # =============================================================

                set_seed(
                    seed
                )

                # =============================================================
                # Why convert to [S1] internally?
                #
                # Dia's official guidance says generation should begin at [S1].
                # Since each role is now generated independently, we use a
                # one-speaker [S1] prompt internally.
                #
                # External role mapping is NOT changed:
                #
                #   original S1 = agent
                #   original S2 = customer
                # =============================================================

                generation_text = (
                    f"[S1] {speech}"
                )

                print("")
                print("-" * 80)

                print(
                    f"Unit {index}/{len(units)}"
                )

                print(
                    f"Original speaker    : "
                    f"{original_speaker} ({role})"
                )

                print(
                    f"Seed                : {seed}"
                )

                print(
                    f"Original turn       : "
                    f"{unit['turn']}"
                )

                print(
                    f"Piece               : "
                    f"{unit['piece']}/{unit['pieces']}"
                )

                print(
                    f"Text                : {speech}"
                )

                # =============================================================
                # Preprocess
                # =============================================================

                preprocess_start = (
                    time.perf_counter_ns()
                )

                inputs = processor(
                    text=[
                        generation_text
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

                total_preprocess_ms += (
                    preprocess_ms
                )

                # =============================================================
                # Inference
                # =============================================================

                inference_start = (
                    time.perf_counter_ns()
                )

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
                    time.perf_counter_ns()
                    - inference_start
                ) / 1_000_000

                total_inference_ms += (
                    inference_ms
                )

                # =============================================================
                # Decode
                # =============================================================

                decode_start = (
                    time.perf_counter_ns()
                )

                audio = decode_generated_audio(
                    outputs
                )

                decode_ms = (
                    time.perf_counter_ns()
                    - decode_start
                ) / 1_000_000

                total_decode_ms += (
                    decode_ms
                )

                generated_parts.append(
                    audio
                )

                unit_duration_s = (
                    len(audio)
                    / SAMPLE_RATE
                )

                print(
                    f"Preprocess          : "
                    f"{preprocess_ms:.2f} ms"
                )

                print(
                    f"Inference           : "
                    f"{inference_ms:.2f} ms"
                )

                print(
                    f"Decode              : "
                    f"{decode_ms:.2f} ms"
                )

                print(
                    f"Audio duration      : "
                    f"{unit_duration_s:.3f} sec"
                )

        # =====================================================================
        # Join everything into ONE WAV
        # =====================================================================

        concat_start = (
            time.perf_counter_ns()
        )

        final_audio = concatenate_audio(
            generated_parts
        )

        concat_ms = (
            time.perf_counter_ns()
            - concat_start
        ) / 1_000_000

        audio_duration_s = (
            len(final_audio)
            / SAMPLE_RATE
        )

        # =====================================================================
        # Encode WAV
        # =====================================================================

        encode_start = (
            time.perf_counter_ns()
        )

        buffer = io.BytesIO()

        sf.write(
            buffer,
            final_audio,
            SAMPLE_RATE,
            format="WAV",
            subtype="PCM_16",
        )

        buffer.seek(0)

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
        # Server logs
        # =====================================================================

        print("")
        print("=" * 80)
        print("SERVER LATENCY")
        print("=" * 80)

        print(
            f"Request ID           : {request_id}"
        )

        print(
            f"Generation units     : {len(units)}"
        )

        print(
            f"Agent seed           : {req.agent_seed}"
        )

        print(
            f"Customer seed        : {req.customer_seed}"
        )

        print(
            f"Preprocess total     : "
            f"{total_preprocess_ms:.2f} ms"
        )

        print(
            f"Inference total      : "
            f"{total_inference_ms:.2f} ms"
        )

        print(
            f"Decode total         : "
            f"{total_decode_ms:.2f} ms"
        )

        print(
            f"Concatenate          : "
            f"{concat_ms:.2f} ms"
        )

        print(
            f"WAV encoding         : "
            f"{encode_ms:.2f} ms"
        )

        print(
            f"SERVER TOTAL         : "
            f"{server_total_ms:.2f} ms"
        )

        print(
            f"Audio duration       : "
            f"{audio_duration_s:.3f} sec"
        )

        print(
            f"Generation RTF       : "
            f"{generation_rtf:.4f}"
        )

        print(
            f"Total RTF            : "
            f"{total_rtf:.4f}"
        )

        print(
            f"GPU allocated        : "
            f"{gpu_allocated_mb:.2f} MB"
        )

        print(
            f"GPU reserved         : "
            f"{gpu_reserved_mb:.2f} MB"
        )

        print(
            f"GPU peak             : "
            f"{gpu_peak_mb:.2f} MB"
        )

        print("=" * 80)

        # =====================================================================
        # Response headers
        # =====================================================================

        headers = {

            "X-Request-ID":
                request_id,

            "X-Generation-Units":
                str(len(units)),

            "X-Agent-Seed":
                str(req.agent_seed),

            "X-Customer-Seed":
                str(req.customer_seed),

            "X-Preprocess-Time-MS":
                f"{total_preprocess_ms:.2f}",

            "X-Inference-Time-MS":
                f"{total_inference_ms:.2f}",

            "X-Decode-Time-MS":
                f"{total_decode_ms:.2f}",

            "X-Concat-Time-MS":
                f"{concat_ms:.2f}",

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

        # ONE response containing the entire conversation.
        return StreamingResponse(
            buffer,
            media_type="audio/wav",
            headers=headers,
        )

    except torch.cuda.OutOfMemoryError as exc:

        torch.cuda.empty_cache()

        raise HTTPException(
            status_code=507,
            detail=(
                f"CUDA out of memory: {exc}"
            ),
        ) from exc

    except Exception as exc:

        print("")
        print(
            f"[error] Request {request_id}: "
            f"{type(exc).__name__}: {exc}"
        )

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        raise HTTPException(
            status_code=500,
            detail=(
                "TTS generation failed: "
                f"{type(exc).__name__}: {exc}"
            ),
        ) from exc
