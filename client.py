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
        return float(
            headers.get(
                key,
                default,
            )
        )

    except Exception:
        return default


def read_text(args):

    if args.text_file:

        path = Path(
            args.text_file
        )

        if not path.exists():

            print(
                f"Text file does not exist: {path}"
            )

            sys.exit(1)

        return path.read_text(
            encoding="utf-8"
        ).strip()

    if args.text:
        return args.text.strip()

    print(
        "Provide either --text or --text-file."
    )

    sys.exit(1)


def main():

    parser = argparse.ArgumentParser(
        description="Client for Dia TTS API"
    )

    parser.add_argument(
        "--server",
        default="http://localhost:8000",
        help="Dia TTS server URL",
    )

    parser.add_argument(
        "--text",
        default=None,
        help=(
            'Dialogue text, e.g. '
            '"[S1] Hello. [S2] Hi."'
        ),
    )

    parser.add_argument(
        "--text-file",
        default=None,
        help=(
            "Read complete dialogue from a UTF-8 text file."
        ),
    )

    parser.add_argument(
        "--output",
        default="output.wav",
        help="Output WAV file",
    )

    parser.add_argument(
        "--agent-seed",
        type=int,
        default=103,
        help=(
            "Dia RNG seed for S1 / agent voice."
        ),
    )

    parser.add_argument(
        "--customer-seed",
        type=int,
        default=217,
        help=(
            "Dia RNG seed for S2 / customer voice."
        ),
    )

    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=3072,
        help=(
            "Maximum generated tokens for each internal utterance."
        ),
    )

    parser.add_argument(
        "--play",
        action="store_true",
        help=(
            "Play generated WAV after saving."
        ),
    )

    parser.add_argument(
        "--timeout",
        type=float,
        default=1800.0,
        help=(
            "HTTP timeout in seconds. Long dialogue may require several minutes."
        ),
    )

    args = parser.parse_args()

    if not (
        0
        <= args.agent_seed
        <= 2_147_483_647
    ):

        print(
            "agent-seed must be between "
            "0 and 2147483647"
        )

        sys.exit(1)

    if not (
        0
        <= args.customer_seed
        <= 2_147_483_647
    ):

        print(
            "customer-seed must be between "
            "0 and 2147483647"
        )

        sys.exit(1)

    text = read_text(
        args
    )

    url = (
        args.server.rstrip("/")
        + "/tts"
    )

    payload = {
        "text": text,
        "agent_seed": args.agent_seed,
        "customer_seed": args.customer_seed,
        "max_new_tokens": args.max_new_tokens,
    }

    print("")
    print("=" * 80)
    print("DIA TTS CLIENT")
    print("=" * 80)

    print(
        f"Server             : {url}"
    )

    print(
        "Speaker mapping    : "
        "S1=AGENT / S2=CUSTOMER"
    )

    print(
        f"Agent seed         : {args.agent_seed}"
    )

    print(
        f"Customer seed      : {args.customer_seed}"
    )

    print(
        f"Text characters    : {len(text)}"
    )

    print(
        f"Text words         : {len(text.split())}"
    )

    print(
        f"Max new tokens     : {args.max_new_tokens}"
    )

    print(
        f"Output             : {args.output}"
    )

    print(
        f"Play               : {args.play}"
    )

    print("=" * 80)

    request_start = (
        time.perf_counter()
    )

    first_byte_time = None
    audio_bytes = bytearray()

    try:

        with requests.post(
            url,
            json=payload,
            stream=True,
            timeout=args.timeout,
        ) as response:

            headers_received = (
                time.perf_counter()
            )

            if response.status_code != 200:

                try:
                    error_text = response.text

                except Exception:
                    error_text = (
                        "<unable to decode server response>"
                    )

                print(
                    f"Request failed "
                    f"({response.status_code}): "
                    f"{error_text}"
                )

                sys.exit(1)

            response_headers = dict(
                response.headers
            )

            for chunk in response.iter_content(
                chunk_size=8192
            ):

                if not chunk:
                    continue

                if first_byte_time is None:

                    first_byte_time = (
                        time.perf_counter()
                    )

                audio_bytes.extend(
                    chunk
                )

    except requests.exceptions.Timeout:

        print(
            f"Request timed out after "
            f"{args.timeout:.1f} seconds."
        )

        sys.exit(1)

    except requests.exceptions.ConnectionError as exc:

        print(
            f"Connection failed: {exc}"
        )

        sys.exit(1)

    except requests.RequestException as exc:

        print(
            f"Request failed: {exc}"
        )

        sys.exit(1)

    request_end = (
        time.perf_counter()
    )

    if not audio_bytes:

        print(
            "Server returned no audio."
        )

        sys.exit(1)

    if first_byte_time is None:

        first_byte_time = (
            request_end
        )

    # =============================================================================
    # SAVE WAV
    # =============================================================================

    output_path = Path(
        args.output
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        output_path,
        "wb",
    ) as file:

        file.write(
            audio_bytes
        )

    # =============================================================================
    # INSPECT WAV
    # =============================================================================

    try:

        with sf.SoundFile(
            io.BytesIO(
                audio_bytes
            )
        ) as wav:

            sample_rate = (
                wav.samplerate
            )

            frames = len(
                wav
            )

            channels = (
                wav.channels
            )

            audio_duration_s = (
                frames
                / sample_rate
            )

    except Exception as exc:

        print(
            f"Could not inspect WAV: {exc}"
        )

        sample_rate = 0
        frames = 0
        channels = 0
        audio_duration_s = 0.0

    # =============================================================================
    # CLIENT LATENCY
    # =============================================================================

    client_ttfb_ms = (
        headers_received
        - request_start
    ) * 1000

    client_ttfa_ms = (
        first_byte_time
        - request_start
    ) * 1000

    first_audio_to_done_ms = (
        request_end
        - first_byte_time
    ) * 1000

    client_total_ms = (
        request_end
        - request_start
    ) * 1000

    client_rtf = (
        (
            client_total_ms
            / 1000
        )
        / audio_duration_s
        if audio_duration_s > 0
        else 0.0
    )

    # =============================================================================
    # SERVER METRICS
    # =============================================================================

    server_preprocess_ms = safe_float(
        response_headers,
        "X-Preprocess-Time-MS",
    )

    server_inference_ms = safe_float(
        response_headers,
        "X-Inference-Time-MS",
    )

    server_decode_ms = safe_float(
        response_headers,
        "X-Decode-Time-MS",
    )

    concat_ms = safe_float(
        response_headers,
        "X-Concat-Time-MS",
    )

    server_encoding_ms = safe_float(
        response_headers,
        "X-Encoding-Time-MS",
    )

    server_total_ms = safe_float(
        response_headers,
        "X-Server-Total-MS",
    )

    server_audio_duration_s = safe_float(
        response_headers,
        "X-Audio-Duration-S",
    )

    generation_rtf = safe_float(
        response_headers,
        "X-Generation-RTF",
    )

    server_total_rtf = safe_float(
        response_headers,
        "X-RTF",
    )

    gpu_allocated_mb = safe_float(
        response_headers,
        "X-GPU-Allocated-MB",
    )

    gpu_reserved_mb = safe_float(
        response_headers,
        "X-GPU-Reserved-MB",
    )

    gpu_peak_mb = safe_float(
        response_headers,
        "X-GPU-Peak-MB",
    )

    generation_units = (
        response_headers.get(
            "X-Generation-Units",
            "N/A",
        )
    )

    request_id = (
        response_headers.get(
            "X-Request-ID",
            "N/A",
        )
    )

    server_sample_rate = (
        response_headers.get(
            "X-Sample-Rate",
            str(sample_rate),
        )
    )

    # =============================================================================
    # PRINT CLIENT METRICS
    # =============================================================================

    print("")
    print("=" * 80)
    print("CLIENT LATENCY")
    print("=" * 80)

    print(
        f"HTTP request -> headers/TTFB : "
        f"{client_ttfb_ms:.2f} ms"
    )

    print(
        f"TTFT / TTFA                  : "
        f"{client_ttfa_ms:.2f} ms"
    )

    print(
        f"First audio -> complete       : "
        f"{first_audio_to_done_ms:.2f} ms"
    )

    print(
        f"CLIENT TOTAL                  : "
        f"{client_total_ms:.2f} ms"
    )

    print(
        f"Audio duration                : "
        f"{audio_duration_s:.3f} sec"
    )

    print(
        f"Client E2E RTF                : "
        f"{client_rtf:.4f}"
    )

    print(
        f"Response size                 : "
        f"{len(audio_bytes) / 1024:.2f} KiB"
    )

    # =============================================================================
    # SERVER METRICS
    # =============================================================================

    print("")
    print("=" * 80)
    print("SERVER LATENCY")
    print("=" * 80)

    print(
        f"Generation units              : "
        f"{generation_units}"
    )

    print(
        f"Agent seed                    : "
        f"{args.agent_seed}"
    )

    print(
        f"Customer seed                 : "
        f"{args.customer_seed}"
    )

    print(
        f"Preprocess                    : "
        f"{server_preprocess_ms:.2f} ms"
    )

    print(
        f"Inference / generation        : "
        f"{server_inference_ms:.2f} ms"
    )

    print(
        f"Decode                        : "
        f"{server_decode_ms:.2f} ms"
    )

    print(
        f"Concatenate                   : "
        f"{concat_ms:.2f} ms"
    )

    print(
        f"WAV encoding                  : "
        f"{server_encoding_ms:.2f} ms"
    )

    print(
        f"SERVER TOTAL                  : "
        f"{server_total_ms:.2f} ms"
    )

    print(
        f"Server audio duration         : "
        f"{server_audio_duration_s:.3f} sec"
    )

    print(
        f"Generation RTF                : "
        f"{generation_rtf:.4f}"
    )

    print(
        f"Server total RTF              : "
        f"{server_total_rtf:.4f}"
    )

    print(
        f"GPU allocated                 : "
        f"{gpu_allocated_mb:.2f} MB"
    )

    print(
        f"GPU reserved                  : "
        f"{gpu_reserved_mb:.2f} MB"
    )

    print(
        f"GPU peak                      : "
        f"{gpu_peak_mb:.2f} MB"
    )

    # =============================================================================
    # AUDIO
    # =============================================================================

    print("")
    print("=" * 80)
    print("AUDIO")
    print("=" * 80)

    print(
        f"Saved                         : "
        f"{output_path}"
    )

    print(
        f"Sample rate                   : "
        f"{sample_rate} Hz"
    )

    print(
        f"Server sample rate            : "
        f"{server_sample_rate} Hz"
    )

    print(
        f"Channels                      : "
        f"{channels}"
    )

    print(
        f"Frames                        : "
        f"{frames}"
    )

    print(
        f"Duration                      : "
        f"{audio_duration_s:.3f} sec"
    )

    print(
        f"Request ID                    : "
        f"{request_id}"
    )

    print("=" * 80)

    # =============================================================================
    # PLAYBACK
    # =============================================================================

    if args.play:

        try:

            import sounddevice as sd

            print("")
            print(
                f"Playing {output_path} ..."
            )

            audio, sr = sf.read(
                str(
                    output_path
                ),
                dtype="float32",
            )

            sd.play(
                audio,
                sr,
            )

            sd.wait()

            print(
                "Playback complete."
            )

        except ImportError:

            print("")
            print(
                "Playback requires sounddevice."
            )

            print(
                "Install using:"
            )

            print(
                "python -m pip install sounddevice"
            )

        except Exception as exc:

            print("")
            print(
                f"Audio playback failed: {exc}"
            )


if __name__ == "__main__":
    main()
