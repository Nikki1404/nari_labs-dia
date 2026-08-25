#!/usr/bin/env python3

import argparse
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
# TEXT
# =============================================================================

def read_text_file(path_value):

    path = Path(
        path_value
    )

    if not path.exists():

        print(
            f"File not found: "
            f"{path}"
        )

        sys.exit(1)

    return path.read_text(
        encoding="utf-8"
    ).strip()


def get_input_text(args):

    # -------------------------------------------------------------------------
    # File mode
    # -------------------------------------------------------------------------

    if args.text_file:

        return read_text_file(
            args.text_file
        )

    # -------------------------------------------------------------------------
    # Direct CLI text
    # -------------------------------------------------------------------------

    if args.text:

        return args.text.strip()

    print(
        "Provide either "
        "--text or --text-file."
    )

    sys.exit(1)


# =============================================================================
# STREAM
# =============================================================================

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
                "Server stream "
                "ended unexpectedly."
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

    if pcm_size:

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
            librosa.effects
            .time_stretch(
                audio,
                rate=speed,
            )
        )

    except Exception as exc:

        print(
            f"[WARN] Speed change "
            f"failed: {exc}"
        )

        return audio


# =============================================================================
# MAIN
# =============================================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Dia reference-conditioned "
            "TTS client"
        )
    )

    parser.add_argument(
        "--server",
        default="http://localhost:8000",
    )

    # -------------------------------------------------------------------------
    # Input can be either direct text OR file.
    # -------------------------------------------------------------------------

    input_group = (
        parser.add_mutually_exclusive_group(
            required=True
        )
    )

    input_group.add_argument(
        "--text",
        default=None,
        help=(
            "Direct text such as "
            "'[S1] Hello. [S2] Hi.'"
        ),
    )

    input_group.add_argument(
        "--text-file",
        default=None,
        help=(
            "TXT file containing "
            "the complete transcript."
        ),
    )

    parser.add_argument(
        "--reference-audio",
        required=True,
    )

    parser.add_argument(
        "--reference-text",
        required=True,
    )

    parser.add_argument(
        "--output",
        default="output.wav",
    )

    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=3072,
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

    # =============================================================================
    # INPUT
    # =============================================================================

    text = get_input_text(
        args
    )

    reference_audio_path = Path(
        args.reference_audio
    )

    reference_text_path = Path(
        args.reference_text
    )

    if not reference_audio_path.exists():

        print(
            f"Reference audio "
            f"not found: "
            f"{reference_audio_path}"
        )

        sys.exit(1)

    if not reference_text_path.exists():

        print(
            f"Reference text "
            f"not found: "
            f"{reference_text_path}"
        )

        sys.exit(1)

    reference_text = (
        read_text_file(
            reference_text_path
        )
    )

    url = (
        args.server.rstrip("/")
        + "/tts"
    )

    print("")
    print("=" * 80)
    print("DIA REFERENCE TTS CLIENT")
    print("=" * 80)

    if args.text_file:

        print(
            f"Input source      : "
            f"{args.text_file}"
        )

    else:

        print(
            "Input source      : "
            "CLI --text"
        )

    print(
        f"Input words       : "
        f"{len(text.split())}"
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

    print("=" * 80)

    # =============================================================================
    # PLAYBACK THREAD
    # =============================================================================

    playback_queue = queue.Queue()

    playback_enabled = False

    playback_thread = None

    if args.play:

        try:

            import sounddevice as sd

            playback_enabled = True

            def playback_worker():

                while True:

                    item = (
                        playback_queue
                        .get()
                    )

                    if item is None:

                        playback_queue
                        .task_done()

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
                        f"[PLAY] "
                        f"{block_index}/"
                        f"{block_count}"
                    )

                    sd.play(
                        playback_audio,
                        sample_rate,
                    )

                    # Strictly wait for this block.
                    sd.wait()

                    print(
                        f"[PLAY] finished "
                        f"{block_index}/"
                        f"{block_count}"
                    )

                    playback_queue
                    .task_done()

            playback_thread = (
                threading.Thread(
                    target=
                        playback_worker,
                    daemon=True,
                )
            )

            playback_thread.start()

        except Exception as exc:

            print(
                f"Playback disabled: "
                f"{exc}"
            )

    # =============================================================================
    # REQUEST
    # =============================================================================

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

    # =============================================================================
    # WAV
    # =============================================================================

    output_path = Path(
        args.output
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

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

    complete_audio_parts = []

    first_audio_time = None

    server_metrics = {}

    block_counter = 0

    try:

        while True:

            (
                metadata,
                pcm_bytes,
            ) = read_frame(
                response.raw
            )

            frame_type = metadata.get(
                "type"
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

                # =====================================================
                # Append to ONE final WAV.
                # =====================================================

                wav_writer.write(
                    audio
                )

                complete_audio_parts.append(
                    audio.copy()
                )

                print("")
                print(
                    f"[RECV] "
                    f"{metadata['block_index']}/"
                    f"{metadata['block_count']}"
                )

                print(
                    f"       words      : "
                    f"{metadata.get('block_words')}"
                )

                print(
                    f"       text       : "
                    f"{metadata.get('block_text')}"
                )

                print(
                    f"       audio      : "
                    f"{metadata['audio_duration_s']:.2f}s"
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

    # =============================================================================
    # PLAYBACK FINISH
    # =============================================================================

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

    # =============================================================================
    # OPTIONAL SPEED-ADJUSTED OUTPUT
    # =============================================================================

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

        adjusted_audio = adjust_speed(
            full_audio,
            args.speed,
        )

        speed_name = str(
            args.speed
        ).replace(
            ".",
            "_",
        )

        adjusted_path = (
            output_path.with_name(
                output_path.stem
                + "_speed_"
                + speed_name
                + output_path.suffix
            )
        )

        sf.write(
            adjusted_path,
            adjusted_audio,
            44100,
            subtype="PCM_16",
        )

        print(
            f"Adjusted WAV      : "
            f"{adjusted_path}"
        )

    # =============================================================================
    # METRICS
    # =============================================================================

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

    total_ms = (
        receive_complete_time
        - request_start
    ) * 1000

    print("")
    print("=" * 80)
    print("CLIENT LATENCY")
    print("=" * 80)

    print(
        f"HTTP TTFB         : "
        f"{ttfb_ms:.2f} ms"
    )

    print(
        f"TTFA              : "
        f"{ttfa_ms:.2f} ms"
    )

    print(
        f"Receive total     : "
        f"{total_ms:.2f} ms"
    )

    print(
        f"Blocks            : "
        f"{block_counter}"
    )

    print("")
    print("=" * 80)
    print("SERVER")
    print("=" * 80)

    print(
        f"Server total      : "
        f"{server_metrics.get('server_total_ms', 0):.2f} ms"
    )

    print(
        f"Audio duration    : "
        f"{server_metrics.get('audio_duration_s', 0):.2f}s"
    )

    print(
        f"Generation RTF    : "
        f"{server_metrics.get('generation_rtf', 0):.4f}"
    )

    print("")
    print(
        f"Saved             : "
        f"{output_path}"
    )


if __name__ == "__main__":

    main()
