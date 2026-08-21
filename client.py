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


def load_text(args):

    if args.text_file:

        path = Path(
            args.text_file
        )

        if not path.exists():

            print(
                f"Text file not found: "
                f"{path}"
            )

            sys.exit(1)

        return path.read_text(
            encoding="utf-8"
        ).strip()

    if args.text:

        return args.text.strip()

    print(
        "Provide --text or --text-file"
    )

    sys.exit(1)


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
                "Server stream ended unexpectedly"
            )

        data.extend(
            part
        )

    return bytes(
        data
    )


def read_frame(raw):

    metadata_size = struct.unpack(
        ">I",
        receive_exact(
            raw,
            4,
        ),
    )[0]

    metadata_bytes = receive_exact(
        raw,
        metadata_size,
    )

    metadata = json.loads(
        metadata_bytes.decode(
            "utf-8"
        )
    )

    audio_size = struct.unpack(
        ">Q",
        receive_exact(
            raw,
            8,
        ),
    )[0]

    pcm_bytes = b""

    if audio_size > 0:

        pcm_bytes = receive_exact(
            raw,
            audio_size,
        )

    return (
        metadata,
        pcm_bytes,
    )


# =============================================================================
# SHORT WAV RESPONSE
# =============================================================================

def handle_short_wav(
    response,
    args,
    request_start,
    headers_received,
):

    audio_bytes = (
        response.content
    )

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

    request_end = (
        time.perf_counter()
    )

    with sf.SoundFile(
        io.BytesIO(
            audio_bytes
        )
    ) as wav:

        sample_rate = wav.samplerate
        frames = len(wav)
        channels = wav.channels

        duration = (
            frames
            / sample_rate
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
        f"TTFT / TTFA       : "
        f"{ttfa_ms:.2f} ms"
    )

    print(
        f"CLIENT TOTAL      : "
        f"{total_ms:.2f} ms"
    )

    print(
        f"Audio duration    : "
        f"{duration:.3f}s"
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
        f"Generation RTF    : "
        f"{safe_float(response.headers, 'X-Generation-RTF'):.4f}"
    )

    print("")
    print("=" * 80)
    print("AUDIO")
    print("=" * 80)

    print(
        f"Saved             : "
        f"{output_path}"
    )

    print(
        f"Sample rate       : "
        f"{sample_rate} Hz"
    )

    print(
        f"Channels          : "
        f"{channels}"
    )

    print(
        f"Seed              : "
        f"{args.seed}"
    )

    print("=" * 80)

    # =================================================================
    # Playback
    # =================================================================

    if args.play:

        try:

            import sounddevice as sd

            audio, sr = sf.read(
                str(output_path),
                dtype="float32",
            )

            print("")
            print(
                "Playing audio..."
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
# LONG STREAM
# =============================================================================

def handle_long_stream(
    response,
    args,
    request_start,
    headers_received,
):

    output_path = Path(
        args.output
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # =================================================================
    # Playback queue
    # =================================================================

    playback_queue = (
        queue.Queue()
    )

    playback_thread = None
    playback_enabled = False

    # =================================================================
    # Playback worker
    #
    # STRICT ORDER:
    #
    # Chunk 1 finishes completely before chunk 2 starts.
    # =================================================================

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
                        chunk_index,
                        chunk_count,
                        audio,
                        sample_rate,
                    ) = item

                    print("")
                    print(
                        f"[PLAY] Starting chunk "
                        f"{chunk_index}/"
                        f"{chunk_count}"
                    )

                    start = (
                        time.perf_counter()
                    )

                    sd.play(
                        audio,
                        sample_rate,
                    )

                    # -------------------------------------------------
                    # CRITICAL:
                    #
                    # Block until THIS chunk is fully played.
                    #
                    # Chunk 2 cannot start before this returns.
                    # -------------------------------------------------

                    sd.wait()

                    playback_ms = (
                        time.perf_counter()
                        - start
                    ) * 1000

                    print(
                        f"[PLAY] Finished chunk "
                        f"{chunk_index}/"
                        f"{chunk_count} "
                        f"({playback_ms:.2f} ms)"
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
                f"Playback unavailable: {exc}"
            )

            playback_enabled = False

    # =================================================================
    # Progressive WAV writer
    # =================================================================

    wav_writer = sf.SoundFile(
        str(
            output_path
        ),
        mode="w",
        samplerate=44100,
        channels=1,
        subtype="PCM_16",
        format="WAV",
    )

    first_audio_time = None
    last_audio_received = None

    chunk_counter = 0

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

            # =========================================================
            # AUDIO
            # =========================================================

            if frame_type == "audio":

                chunk_counter += 1

                now = (
                    time.perf_counter()
                )

                if first_audio_time is None:

                    first_audio_time = now

                last_audio_received = now

                # -----------------------------------------------------
                # PCM16 -> float32
                # -----------------------------------------------------

                pcm = np.frombuffer(
                    pcm_bytes,
                    dtype="<i2",
                )

                float_audio = (
                    pcm.astype(
                        np.float32
                    )
                    / 32767.0
                )

                # -----------------------------------------------------
                # Save immediately to final WAV.
                # -----------------------------------------------------

                wav_writer.write(
                    float_audio
                )

                print("")
                print(
                    f"[RECV] Chunk "
                    f"{metadata['chunk_index']}/"
                    f"{metadata['chunk_count']}"
                )

                print(
                    f"       Audio duration : "
                    f"{metadata['audio_duration_s']:.2f}s"
                )

                print(
                    f"       Inference      : "
                    f"{metadata['inference_ms']:.2f} ms"
                )

                # -----------------------------------------------------
                # QUEUE.
                #
                # Even if this chunk arrives while another chunk
                # is playing, it only waits here.
                # -----------------------------------------------------

                if playback_enabled:

                    playback_queue.put(
                        (
                            metadata[
                                "chunk_index"
                            ],
                            metadata[
                                "chunk_count"
                            ],
                            float_audio.copy(),
                            metadata[
                                "sample_rate"
                            ],
                        )
                    )

            # =========================================================
            # END
            # =========================================================

            elif frame_type == "end":

                server_metrics = (
                    metadata
                )

                break

    finally:

        wav_writer.close()

    network_done_time = (
        time.perf_counter()
    )

    # =================================================================
    # IMPORTANT:
    #
    # Server might already be finished generating all chunks,
    # but the client may still be playing queued audio.
    #
    # Wait for all playback to finish.
    # =================================================================

    if playback_enabled:

        playback_queue.join()

        playback_queue.put(
            None
        )

        playback_queue.join()

        if playback_thread:

            playback_thread.join(
                timeout=10,
            )

    playback_done_time = (
        time.perf_counter()
    )

    if first_audio_time is None:

        print(
            "No audio was received."
        )

        return

    # =================================================================
    # Metrics
    # =================================================================

    ttfb_ms = (
        headers_received
        - request_start
    ) * 1000

    ttfa_ms = (
        first_audio_time
        - request_start
    ) * 1000

    network_total_ms = (
        network_done_time
        - request_start
    ) * 1000

    complete_total_ms = (
        playback_done_time
        - request_start
    ) * 1000

    print("")
    print("=" * 80)
    print("CLIENT LATENCY")
    print("=" * 80)

    print(
        f"HTTP headers / TTFB  : "
        f"{ttfb_ms:.2f} ms"
    )

    print(
        f"TTFT / TTFA          : "
        f"{ttfa_ms:.2f} ms"
    )

    print(
        f"Generation/receive   : "
        f"{network_total_ms:.2f} ms"
    )

    if args.play:

        print(
            f"E2E incl playback    : "
            f"{complete_total_ms:.2f} ms"
        )

    print(
        f"Chunks received      : "
        f"{chunk_counter}"
    )

    print("")
    print("=" * 80)
    print("SERVER LATENCY")
    print("=" * 80)

    print(
        f"Preprocess           : "
        f"{server_metrics.get('preprocess_ms', 0):.2f} ms"
    )

    print(
        f"Inference            : "
        f"{server_metrics.get('inference_ms', 0):.2f} ms"
    )

    print(
        f"Decode               : "
        f"{server_metrics.get('decode_ms', 0):.2f} ms"
    )

    print(
        f"SERVER TOTAL         : "
        f"{server_metrics.get('server_total_ms', 0):.2f} ms"
    )

    print(
        f"Audio duration       : "
        f"{server_metrics.get('audio_duration_s', 0):.3f}s"
    )

    print(
        f"Generation RTF       : "
        f"{server_metrics.get('generation_rtf', 0):.4f}"
    )

    print(
        f"Total RTF            : "
        f"{server_metrics.get('total_rtf', 0):.4f}"
    )

    print(
        f"GPU allocated        : "
        f"{server_metrics.get('gpu_allocated_mb', 0):.2f} MB"
    )

    print(
        f"GPU reserved         : "
        f"{server_metrics.get('gpu_reserved_mb', 0):.2f} MB"
    )

    print(
        f"GPU peak             : "
        f"{server_metrics.get('gpu_peak_mb', 0):.2f} MB"
    )

    print("")
    print("=" * 80)
    print("AUDIO")
    print("=" * 80)

    print(
        f"Saved                : "
        f"{output_path}"
    )

    print(
        f"Seed                 : "
        f"{args.seed}"
    )

    print(
        f"Chunks               : "
        f"{chunk_counter}"
    )

    print("=" * 80)


# =============================================================================
# MAIN
# =============================================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Dia TTS Client"
        )
    )

    parser.add_argument(
        "--server",
        default=(
            "http://localhost:8000"
        ),
    )

    parser.add_argument(
        "--text",
        default=None,
    )

    parser.add_argument(
        "--text-file",
        default=None,
        help=(
            "Read full transcript "
            "from a text file."
        ),
    )

    parser.add_argument(
        "--output",
        default="output.wav",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help=(
            "Dia RNG seed controlling "
            "the generated S1/S2 voice pair."
        ),
    )

    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=2048,
    )

    parser.add_argument(
        "--play",
        action="store_true",
    )

    parser.add_argument(
        "--timeout",
        type=float,
        default=1800,
    )

    args = parser.parse_args()

    text = load_text(
        args
    )

    url = (
        args.server.rstrip("/")
        + "/tts"
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
        f"Seed              : "
        f"{args.seed}"
    )

    print(
        f"Words             : "
        f"{len(text.split())}"
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

    stream_mode = (
        response.headers.get(
            "X-Stream-Mode",
            "wav",
        )
    )

    print(
        f"Response mode     : "
        f"{stream_mode}"
    )

    # =================================================================
    # LONG
    # =================================================================

    if stream_mode == "pcm":

        handle_long_stream(
            response,
            args,
            request_start,
            headers_received,
        )

    # =================================================================
    # SHORT
    # =================================================================

    else:

        handle_short_wav(
            response,
            args,
            request_start,
            headers_received,
        )


if __name__ == "__main__":

    main()
