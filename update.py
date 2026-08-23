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
import sys
import time
from pathlib import Path

import requests
import soundfile as sf


def safe_float(headers, key, default=0.0):
    try:
        return float(headers.get(key, default))
    except Exception:
        return default


def load_text(args):
    if args.text_file:
        path = Path(args.text_file)

        if not path.exists():
            print(f"Text file not found: {path}")
            sys.exit(1)

        return path.read_text(
            encoding="utf-8"
        ).strip()

    if args.text:
        return args.text.strip()

    print("Provide --text or --text-file")
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Dia TTS Client"
    )

    parser.add_argument(
        "--server",
        default="http://localhost:8000",
    )

    parser.add_argument(
        "--text",
        default=None,
    )

    parser.add_argument(
        "--text-file",
        default=None,
    )

    parser.add_argument(
        "--output",
        default="output.wav",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
    )

    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=12288,
    )

    parser.add_argument(
        "--play",
        action="store_true",
    )

    # =================================================================
    # CLIENT-SIDE SPEED CONTROL
    # =================================================================

    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help=(
            "Client-side playback speed. "
            "1.0=original, 0.9=10%% slower, "
            "0.8=20%% slower"
        ),
    )

    parser.add_argument(
        "--save-adjusted",
        action="store_true",
        help=(
            "Save speed-adjusted audio as a "
            "separate WAV file."
        ),
    )

    parser.add_argument(
        "--timeout",
        type=float,
        default=1800,
    )

    args = parser.parse_args()

    # =================================================================
    # Validate speed
    # =================================================================

    if args.speed <= 0:
        print("--speed must be greater than 0")
        sys.exit(1)

    # I recommend keeping this in a sensible range.
    if args.speed < 0.5 or args.speed > 2.0:
        print(
            "Recommended --speed range is "
            "0.5 to 2.0"
        )
        sys.exit(1)

    text = load_text(args)

    url = (
        args.server.rstrip("/")
        + "/tts"
    )

    # IMPORTANT:
    # speed is NOT sent to the server.
    #
    # The server remains completely unchanged.
    payload = {
        "text": text,
        "seed": args.seed,
        "max_new_tokens": args.max_new_tokens,
    }

    print("")
    print("=" * 80)
    print("DIA TTS CLIENT")
    print("=" * 80)

    print(
        f"Server            : {url}"
    )

    print(
        "Speaker mapping   : "
        "[S1]=Agent, [S2]=Customer"
    )

    print(
        f"Seed              : {args.seed}"
    )

    print(
        f"Max new tokens    : "
        f"{args.max_new_tokens}"
    )

    print(
        f"Words             : "
        f"{len(text.split())}"
    )

    print(
        f"Characters        : "
        f"{len(text)}"
    )

    print(
        f"Output            : "
        f"{args.output}"
    )

    print(
        f"Play              : "
        f"{args.play}"
    )

    print(
        f"Playback speed    : "
        f"{args.speed}"
    )

    print(
        f"Save adjusted     : "
        f"{args.save_adjusted}"
    )

    print("=" * 80)

    # =================================================================
    # Request
    # =================================================================

    request_start = time.perf_counter()

    try:
        response = requests.post(
            url,
            json=payload,
            timeout=args.timeout,
        )

    except requests.exceptions.Timeout:
        print(
            f"Request timed out after "
            f"{args.timeout:.1f} seconds."
        )
        sys.exit(1)

    except requests.RequestException as exc:
        print(
            f"Request failed: {exc}"
        )
        sys.exit(1)

    headers_received = time.perf_counter()

    if response.status_code != 200:
        print(
            f"Request failed "
            f"({response.status_code}): "
            f"{response.text}"
        )
        sys.exit(1)

    audio_bytes = response.content

    first_audio_time = time.perf_counter()
    request_end = first_audio_time

    if not audio_bytes:
        print("Server returned no audio.")
        sys.exit(1)

    # =================================================================
    # Save ORIGINAL server audio
    # =================================================================

    output_path = Path(args.output)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_bytes(
        audio_bytes
    )

    # =================================================================
    # Inspect WAV
    # =================================================================

    try:
        with sf.SoundFile(
            io.BytesIO(audio_bytes)
        ) as wav:

            sample_rate = wav.samplerate
            frames = len(wav)
            channels = wav.channels

            audio_duration_s = (
                frames / sample_rate
            )

    except Exception as exc:
        print(
            f"Could not inspect WAV: {exc}"
        )

        sample_rate = 0
        frames = 0
        channels = 0
        audio_duration_s = 0

    # =================================================================
    # Client metrics
    # =================================================================

    client_ttfb_ms = (
        headers_received
        - request_start
    ) * 1000

    client_ttfa_ms = (
        first_audio_time
        - request_start
    ) * 1000

    client_total_ms = (
        request_end
        - request_start
    ) * 1000

    client_rtf = (
        (client_total_ms / 1000)
        / audio_duration_s
        if audio_duration_s > 0
        else 0
    )

    # =================================================================
    # Server metrics
    # =================================================================

    preprocess_ms = safe_float(
        response.headers,
        "X-Preprocess-Time-MS",
    )

    inference_ms = safe_float(
        response.headers,
        "X-Inference-Time-MS",
    )

    decode_ms = safe_float(
        response.headers,
        "X-Decode-Time-MS",
    )

    encoding_ms = safe_float(
        response.headers,
        "X-Encoding-Time-MS",
    )

    server_total_ms = safe_float(
        response.headers,
        "X-Server-Total-MS",
    )

    server_audio_duration = safe_float(
        response.headers,
        "X-Audio-Duration-S",
    )

    generation_rtf = safe_float(
        response.headers,
        "X-Generation-RTF",
    )

    server_rtf = safe_float(
        response.headers,
        "X-RTF",
    )

    gpu_allocated = safe_float(
        response.headers,
        "X-GPU-Allocated-MB",
    )

    gpu_reserved = safe_float(
        response.headers,
        "X-GPU-Reserved-MB",
    )

    gpu_peak = safe_float(
        response.headers,
        "X-GPU-Peak-MB",
    )

    request_id = response.headers.get(
        "X-Request-ID",
        "N/A",
    )

    # =================================================================
    # Print latency
    # =================================================================

    print("")
    print("=" * 80)
    print("CLIENT LATENCY")
    print("=" * 80)

    print(
        f"HTTP request -> response : "
        f"{client_ttfb_ms:.2f} ms"
    )

    print(
        f"TTFT / TTFA             : "
        f"{client_ttfa_ms:.2f} ms"
    )

    print(
        f"CLIENT TOTAL            : "
        f"{client_total_ms:.2f} ms"
    )

    print(
        f"Original audio duration : "
        f"{audio_duration_s:.3f} sec"
    )

    print(
        f"Client E2E RTF          : "
        f"{client_rtf:.4f}"
    )

    print("")
    print("=" * 80)
    print("SERVER LATENCY")
    print("=" * 80)

    print(
        f"Preprocess              : "
        f"{preprocess_ms:.2f} ms"
    )

    print(
        f"Inference / generation  : "
        f"{inference_ms:.2f} ms"
    )

    print(
        f"Decode                  : "
        f"{decode_ms:.2f} ms"
    )

    print(
        f"WAV encoding            : "
        f"{encoding_ms:.2f} ms"
    )

    print(
        f"SERVER TOTAL            : "
        f"{server_total_ms:.2f} ms"
    )

    print(
        f"Audio duration          : "
        f"{server_audio_duration:.3f} sec"
    )

    print(
        f"Generation RTF          : "
        f"{generation_rtf:.4f}"
    )

    print(
        f"Server total RTF        : "
        f"{server_rtf:.4f}"
    )

    print(
        f"GPU allocated           : "
        f"{gpu_allocated:.2f} MB"
    )

    print(
        f"GPU reserved            : "
        f"{gpu_reserved:.2f} MB"
    )

    print(
        f"GPU peak                : "
        f"{gpu_peak:.2f} MB"
    )

    print("")
    print("=" * 80)
    print("AUDIO")
    print("=" * 80)

    print(
        f"Original saved          : "
        f"{output_path}"
    )

    print(
        f"Sample rate             : "
        f"{sample_rate} Hz"
    )

    print(
        f"Channels                : "
        f"{channels}"
    )

    print(
        f"Original duration       : "
        f"{audio_duration_s:.3f} sec"
    )

    print(
        f"Seed                    : "
        f"{args.seed}"
    )

    print(
        f"Playback speed          : "
        f"{args.speed}"
    )

    print(
        f"Request ID              : "
        f"{request_id}"
    )

    print("=" * 80)

    # =================================================================
    # SPEED ADJUSTMENT
    #
    # Happens entirely on CLIENT.
    # Server output remains unchanged.
    # =================================================================

    if (
        args.play
        or args.save_adjusted
    ):

        try:
            import numpy as np
            import librosa

            print("")
            print(
                "Loading audio for "
                "client-side processing..."
            )

            audio, sr = sf.read(
                str(output_path),
                dtype="float32",
            )

            adjusted_audio = audio

            # =========================================================
            # Time stretch
            #
            # 1.00 = unchanged
            # 0.95 = 5% slower
            # 0.90 = 10% slower
            # 0.85 = 15% slower
            # 0.80 = 20% slower
            #
            # Pitch remains approximately unchanged.
            # =========================================================

            if args.speed != 1.0:

                print(
                    f"Adjusting speed: "
                    f"1.00 -> {args.speed}"
                )

                # Handle mono/stereo properly.
                if audio.ndim == 1:

                    adjusted_audio = (
                        librosa.effects.time_stretch(
                            audio,
                            rate=args.speed,
                        )
                    )

                else:

                    channels_list = []

                    for channel in range(
                        audio.shape[1]
                    ):

                        stretched = (
                            librosa.effects.time_stretch(
                                audio[:, channel],
                                rate=args.speed,
                            )
                        )

                        channels_list.append(
                            stretched
                        )

                    min_length = min(
                        len(channel)
                        for channel
                        in channels_list
                    )

                    adjusted_audio = np.stack(
                        [
                            channel[:min_length]
                            for channel
                            in channels_list
                        ],
                        axis=1,
                    )

            adjusted_duration_s = (
                len(adjusted_audio)
                / sr
            )

            print(
                f"Original duration : "
                f"{audio_duration_s:.3f} sec"
            )

            print(
                f"Adjusted duration : "
                f"{adjusted_duration_s:.3f} sec"
            )

            # =========================================================
            # Optional save
            # =========================================================

            if args.save_adjusted:

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
                    adjusted_audio,
                    sr,
                    subtype="PCM_16",
                )

                print(
                    f"Adjusted saved   : "
                    f"{adjusted_path}"
                )

            # =========================================================
            # Playback
            # =========================================================

            if args.play:

                try:
                    import sounddevice as sd

                except ImportError:

                    print(
                        "sounddevice is not installed."
                    )

                    print(
                        "Install using:"
                    )

                    print(
                        "python -m pip install "
                        "sounddevice"
                    )

                    return

                print("")
                print(
                    f"Playing complete audio "
                    f"at speed={args.speed}..."
                )

                playback_start = (
                    time.perf_counter()
                )

                sd.play(
                    adjusted_audio,
                    sr,
                )

                sd.wait()

                playback_ms = (
                    time.perf_counter()
                    - playback_start
                ) * 1000

                print(
                    "Playback complete."
                )

                print(
                    f"Playback duration : "
                    f"{playback_ms / 1000:.3f} sec"
                )

        except ImportError as exc:

            print(
                f"Missing dependency: {exc}"
            )

            print("")
            print(
                "Install:"
            )

            print(
                "python -m pip install "
                "librosa soundfile sounddevice"
            )

        except Exception as exc:

            print(
                f"Speed adjustment/playback "
                f"failed: {exc}"
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
