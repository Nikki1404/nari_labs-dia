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

        print(
            f"File not found: {path}"
        )

        sys.exit(1)

    return path.read_text(
        encoding="utf-8"
    ).strip()


def safe_float(
    headers,
    key,
    default=0.0,
):

    try:

        return float(
            headers.get(
                key,
                default,
            )
        )

    except Exception:

        return default


def receive_exact(
    raw,
    size,
):

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
        receive_exact(
            raw,
            4,
        ),
    )[0]

    metadata = json.loads(
        receive_exact(
            raw,
            metadata_size,
        ).decode(
            "utf-8"
        )
    )

    pcm_size = struct.unpack(
        ">Q",
        receive_exact(
            raw,
            8,
        ),
    )[0]

    pcm_bytes = b""

    if pcm_size > 0:

        pcm_bytes = receive_exact(
            raw,
            pcm_size,
        )

    return (
        metadata,
        pcm_bytes,
    )


# =============================================================================
# SPEED
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
            librosa
            .effects
            .time_stretch(
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
            f"Playing at speed="
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
            f"Playback failed: "
            f"{exc}"
        )


# =============================================================================
# SEED MODE
# =============================================================================

def run_seed_mode(args):

    if args.text_file:

        text = read_text_file(
            args.text_file
        )

    elif args.text:

        text = args.text.strip()

    else:

        print(
            "Seed mode requires "
            "--text or --text-file."
        )

        sys.exit(1)

    # IMPORTANT:
    # Seed mode goes to /tts_seed.
    url = (
        args.server.rstrip("/")
        + "/tts_seed"
    )

    payload = {
        "text":
            text,

        "seed":
            args.seed,

        "max_new_tokens":
            args.max_new_tokens,
    }

    print("")
    print("=" * 80)
    print("DIA SEED TEST MODE")
    print("=" * 80)

    print(
        f"Server            : "
        f"{url}"
    )

    print(
        f"Seed              : "
        f"{args.seed}"
    )

    print(
        f"Words             : "
        f"{len(text.split())}"
    )

    print(
        f"Max new tokens    : "
        f"{args.max_new_tokens}"
    )

    print(
        f"Output            : "
        f"{args.output}"
    )

    print(
        f"Play              : "
        f"{args.play}"
    )

    print("=" * 80)

    request_start = (
        time.perf_counter()
    )

    try:

        response = requests.post(
            url,
            json=payload,
            timeout=args.timeout,
        )

    except requests.RequestException as exc:

        print(
            f"Request failed: "
            f"{exc}"
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

    audio_bytes = (
        response.content
    )

    first_audio_time = (
        time.perf_counter()
    )

    if not audio_bytes:

        print(
            "No audio returned."
        )

        sys.exit(1)

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

    with sf.SoundFile(
        io.BytesIO(
            audio_bytes
        )
    ) as wav:

        sample_rate = wav.samplerate
        channels = wav.channels
        frames = len(wav)

        duration = (
            frames
            / sample_rate
        )

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

    print(
        f"Audio duration    : "
        f"{duration:.3f} sec"
    )

    print("")
    print("=" * 80)
    print("OUTPUT")
    print("=" * 80)

    print(
        f"Saved             : "
        f"{output_path}"
    )

    print(
        f"Seed              : "
        f"{args.seed}"
    )

    print("=" * 80)

    if args.play:

        play_audio_file(
            output_path,
            args.speed,
        )


# =============================================================================
# REFERENCE MODE
# =============================================================================

def run_reference_mode(args):

    if not args.text_file:

        print(
            "Reference mode requires "
            "--text-file."
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
                f"File not found: "
                f"{path}"
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

    print(
        f"Server            : "
        f"{url}"
    )

    print(
        f"Transcript        : "
        f"{transcript_path}"
    )

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

    print(
        f"Output            : "
        f"{args.output}"
    )

    print(
        f"Play              : "
        f"{args.play}"
    )

    print(
        f"Speed             : "
        f"{args.speed}"
    )

    print("=" * 80)

    # =========================================================================
    # PLAYBACK QUEUE
    # =========================================================================

    playback_queue = (
        queue.Queue()
    )

    playback_enabled = False
    playback_thread = None

    if args.play:

        try:

            import sounddevice as sd

            playback_enabled = True

            def playback_worker():

                while True:

                    item = (
                        playback_queue.get()
                    )

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
                        f"[PLAY] Starting "
                        f"block "
                        f"{block_index}/"
                        f"{block_count}"
                    )

                    sd.play(
                        playback_audio,
                        sample_rate,
                    )

                    # Strict sequential playback.
                    sd.wait()

                    print(
                        f"[PLAY] Finished "
                        f"block "
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
                f"Playback disabled: "
                f"{exc}"
            )

            playback_enabled = False

    # =========================================================================
    # MULTIPART HTTP
    # =========================================================================

    request_start = (
        time.perf_counter()
    )

    try:

        with open(
            reference_audio_path,
            "rb",
        ) as handle:

            files = {
                "reference_audio": (
                    reference_audio_path.name,
                    handle,
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
            f"Request failed: "
            f"{exc}"
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
    # OUTPUT WRITER
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
                metadata.get(
                    "type"
                )
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

                server_metrics = (
                    metadata
                )

                break

    finally:

        wav_writer.close()

    receive_complete_time = (
        time.perf_counter()
    )

    # =========================================================================
    # WAIT FOR PLAYBACK
    # =========================================================================

    if playback_enabled:

        playback_queue.join()

        playback_queue.put(
            None
        )

        playback_queue.join()

        if playback_thread:

            playback_thread.join(
                timeout=10
            )

    playback_complete_time = (
        time.perf_counter()
    )

    # =========================================================================
    # OPTIONAL ADJUSTED FILE
    # =========================================================================

    adjusted_path = None

    if (
        args.save_adjusted
        and
        args.speed != 1.0
        and
        complete_audio_parts
    ):

        full_audio = np.concatenate(
            complete_audio_parts
        )

        adjusted_audio = (
            adjust_speed(
                full_audio,
                args.speed,
            )
        )

        speed_string = (
            str(args.speed)
            .replace(
                ".",
                "_",
            )
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
            44100,
            subtype="PCM_16",
        )

    # =========================================================================
    # METRICS
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

    print("=" * 80)


# =============================================================================
# MAIN
# =============================================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Dia TTS client: "
            "seed audition + "
            "reference-conditioned long TTS"
        )
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
        "--seed",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--reference-audio",
        default=None,
    )

    parser.add_argument(
        "--reference-text",
        default=None,
    )

    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=None,
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

    if args.speed <= 0:

        print(
            "--speed must be > 0"
        )

        sys.exit(1)

    # =========================================================================
    # REFERENCE MODE
    # =========================================================================

    if (
        args.reference_audio is not None
        or
        args.reference_text is not None
    ):

        if args.max_new_tokens is None:
            args.max_new_tokens = 4096

        run_reference_mode(
            args
        )

        return

    # =========================================================================
    # SEED MODE
    # =========================================================================

    if args.seed is not None:

        if args.max_new_tokens is None:
            args.max_new_tokens = 1024

        run_seed_mode(
            args
        )

        return

    print("")
    print(
        "Choose one mode:"
    )

    print("")
    print(
        "Seed mode:"
    )

    print(
        "  --seed <number> "
        "--text \"[S1] ... [S2] ...\""
    )

    print("")
    print(
        "Reference mode:"
    )

    print(
        "  --text-file full_call.txt "
        "--reference-audio reference.wav "
        "--reference-text reference.txt"
    )

    sys.exit(1)


if __name__ == "__main__":
    main()
