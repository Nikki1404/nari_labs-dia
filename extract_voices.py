#!/usr/bin/env python3

import argparse
import re
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly


# =============================================================================
# TEXT
# =============================================================================

def parse_reference_text(path: Path):

    text = path.read_text(
        encoding="utf-8"
    )

    text = re.sub(
        r"\s+",
        " ",
        text,
    ).strip()

    matches = list(
        re.finditer(
            r"\[(S1|S2)\]\s*",
            text,
            flags=re.IGNORECASE,
        )
    )

    turns = []

    for i, match in enumerate(matches):

        speaker = match.group(1).upper()

        start = match.end()

        if i + 1 < len(matches):

            end = matches[
                i + 1
            ].start()

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

    if len(turns) != 2:

        raise ValueError(
            "This extractor expects reference.txt "
            "to contain exactly TWO speaker turns, "
            "for example:\n"
            "[S1] Agent sentence. [S2] Customer sentence."
        )

    if turns[0][0] != "S1":
        raise ValueError(
            "First turn must be [S1]."
        )

    if turns[1][0] != "S2":
        raise ValueError(
            "Second turn must be [S2]."
        )

    return turns


# =============================================================================
# AUDIO
# =============================================================================

def load_mono(path: Path):

    audio, sr = sf.read(
        str(path),
        dtype="float32",
    )

    if audio.ndim == 2:

        audio = np.mean(
            audio,
            axis=1,
        )

    return (
        np.asarray(
            audio,
            dtype=np.float32,
        ),
        sr,
    )


# =============================================================================
# FRAME RMS
# =============================================================================

def calculate_rms(
    audio,
    frame_size,
    hop_size,
):

    values = []

    positions = []

    for start in range(
        0,
        max(
            1,
            len(audio)
            - frame_size,
        ),
        hop_size,
    ):

        frame = audio[
            start:
            start + frame_size
        ]

        if len(frame) == 0:
            continue

        rms = np.sqrt(
            np.mean(
                frame * frame
            )
            + 1e-12
        )

        values.append(
            rms
        )

        positions.append(
            start
            + len(frame) // 2
        )

    return (
        np.asarray(values),
        np.asarray(positions),
    )


# =============================================================================
# FIND S1 -> S2 BOUNDARY
# =============================================================================

def find_speaker_boundary(
    audio,
    sr,
    s1_text,
    s2_text,
    search_fraction=0.18,
):

    # -------------------------------------------------------------------------
    # Approximate where boundary should be based on amount of text.
    # -------------------------------------------------------------------------

    s1_words = max(
        1,
        len(
            s1_text.split()
        ),
    )

    s2_words = max(
        1,
        len(
            s2_text.split()
        ),
    )

    expected_ratio = (
        s1_words
        /
        (
            s1_words
            + s2_words
        )
    )

    expected_sample = int(
        len(audio)
        * expected_ratio
    )

    # -------------------------------------------------------------------------
    # RMS every 10ms using ~40ms windows.
    # -------------------------------------------------------------------------

    frame_size = int(
        sr * 0.040
    )

    hop_size = int(
        sr * 0.010
    )

    rms, positions = calculate_rms(
        audio,
        frame_size,
        hop_size,
    )

    if len(rms) == 0:

        return expected_sample

    # -------------------------------------------------------------------------
    # Only search around expected transcript boundary.
    # Prevents selecting an internal pause far away.
    # -------------------------------------------------------------------------

    radius = int(
        len(audio)
        * search_fraction
    )

    left = max(
        0,
        expected_sample - radius,
    )

    right = min(
        len(audio),
        expected_sample + radius,
    )

    mask = (
        (positions >= left)
        &
        (positions <= right)
    )

    if not np.any(mask):

        return expected_sample

    local_rms = rms[
        mask
    ]

    local_positions = positions[
        mask
    ]

    # -------------------------------------------------------------------------
    # Smooth RMS over ~120ms so we prefer a true gap rather than one tiny dip.
    # -------------------------------------------------------------------------

    smoothing_frames = max(
        1,
        int(
            0.120
            /
            (
                hop_size
                / sr
            )
        ),
    )

    kernel = np.ones(
        smoothing_frames,
        dtype=np.float32,
    ) / smoothing_frames

    smoothed = np.convolve(
        local_rms,
        kernel,
        mode="same",
    )

    index = int(
        np.argmin(
            smoothed
        )
    )

    return int(
        local_positions[
            index
        ]
    )


# =============================================================================
# TRIM OUTER SILENCE
# =============================================================================

def trim_outer_silence(
    audio,
    sr,
    threshold_ratio=0.025,
    padding_ms=40,
):

    if len(audio) == 0:
        return audio

    frame = max(
        1,
        int(
            sr * 0.020
        ),
    )

    hop = max(
        1,
        int(
            sr * 0.010
        ),
    )

    rms, positions = calculate_rms(
        audio,
        frame,
        hop,
    )

    if len(rms) == 0:

        return audio

    peak = float(
        np.max(rms)
    )

    if peak <= 0:

        return audio

    threshold = (
        peak
        * threshold_ratio
    )

    active = positions[
        rms >= threshold
    ]

    if len(active) == 0:

        return audio

    padding = int(
        sr
        * padding_ms
        / 1000
    )

    start = max(
        0,
        int(active[0])
        - padding,
    )

    end = min(
        len(audio),
        int(active[-1])
        + padding,
    )

    return audio[
        start:end
    ]


# =============================================================================
# SPLIT ONE FILE
# =============================================================================

def split_voice_file(
    wav_path,
    s1_text,
    s2_text,
):

    audio, sr = load_mono(
        wav_path
    )

    split = find_speaker_boundary(
        audio,
        sr,
        s1_text,
        s2_text,
    )

    # Leave a tiny gap around transition to reduce speaker contamination.
    gap = int(
        sr * 0.030
    )

    s1_end = max(
        0,
        split - gap,
    )

    s2_start = min(
        len(audio),
        split + gap,
    )

    s1 = audio[
        :s1_end
    ]

    s2 = audio[
        s2_start:
    ]

    s1 = trim_outer_silence(
        s1,
        sr,
    )

    s2 = trim_outer_silence(
        s2,
        sr,
    )

    return (
        s1,
        s2,
        sr,
        split / sr,
    )


# =============================================================================
# MAIN
# =============================================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Extract Dia S1/S2 voices "
            "from seed-generated WAV files."
        )
    )

    parser.add_argument(
        "--input-dir",
        required=True,
        help=(
            "Directory containing seed_*.wav"
        ),
    )

    parser.add_argument(
        "--reference-text",
        required=True,
        help=(
            "Exact two-turn text used "
            "to generate all seed WAVs."
        ),
    )

    parser.add_argument(
        "--agent-file",
        required=True,
        help=(
            "Seed WAV whose S1 voice "
            "should become agent.wav. "
            "Example: seed_3096.wav"
        ),
    )

    parser.add_argument(
        "--output-dir",
        default="extracted_voices",
    )

    parser.add_argument(
        "--pattern",
        default="seed_*.wav",
    )

    args = parser.parse_args()


    input_dir = Path(
        args.input_dir
    )

    reference_path = Path(
        args.reference_text
    )

    output_dir = Path(
        args.output_dir
    )

    customer_dir = (
        output_dir
        / "customers"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    customer_dir.mkdir(
        parents=True,
        exist_ok=True,
    )


    turns = parse_reference_text(
        reference_path
    )

    s1_text = turns[0][1]

    s2_text = turns[1][1]


    print("")
    print("=" * 80)
    print("REFERENCE")
    print("=" * 80)

    print(
        f"S1 / Agent text   : "
        f"{s1_text}"
    )

    print(
        f"S2 / Customer text: "
        f"{s2_text}"
    )

    print("=" * 80)


    wav_files = sorted(
        input_dir.glob(
            args.pattern
        )
    )

    if not wav_files:

        raise FileNotFoundError(
            f"No WAVs found using "
            f"{input_dir / args.pattern}"
        )


    # =========================================================================
    # Extract all S2 customer options
    # =========================================================================

    for wav_path in wav_files:

        try:

            s1_audio, s2_audio, sr, split_s = (
                split_voice_file(
                    wav_path,
                    s1_text,
                    s2_text,
                )
            )

            output_customer = (
                customer_dir
                /
                f"customer_{wav_path.stem}.wav"
            )

            sf.write(
                str(
                    output_customer
                ),
                s2_audio,
                sr,
                subtype="PCM_16",
            )

            print(
                f"[CUSTOMER] "
                f"{wav_path.name}"
                f" -> "
                f"{output_customer.name}"
                f" | boundary="
                f"{split_s:.2f}s"
            )

        except Exception as exc:

            print(
                f"[FAILED] "
                f"{wav_path.name}: "
                f"{exc}"
            )


    # =========================================================================
    # Extract selected S1 as agent.wav
    # =========================================================================

    agent_path = Path(
        args.agent_file
    )

    if not agent_path.is_absolute():

        agent_path = (
            input_dir
            / agent_path
        )

    if not agent_path.exists():

        raise FileNotFoundError(
            f"Agent WAV not found: "
            f"{agent_path}"
        )


    agent_audio, _, agent_sr, split_s = (
        split_voice_file(
            agent_path,
            s1_text,
            s2_text,
        )
    )

    final_agent = (
        output_dir
        / "agent.wav"
    )

    sf.write(
        str(
            final_agent
        ),
        agent_audio,
        agent_sr,
        subtype="PCM_16",
    )


    print("")
    print("=" * 80)
    print("DONE")
    print("=" * 80)

    print(
        f"Agent source      : "
        f"{agent_path.name}"
    )

    print(
        f"Agent output      : "
        f"{final_agent}"
    )

    print(
        f"Customer options  : "
        f"{customer_dir}"
    )

    print(
        f"Customers created : "
        f"{len(wav_files)}"
    )

    print("=" * 80)


if __name__ == "__main__":

    main()


#python extract_voices.py --input-dir shortlisted\random_wavs --reference-text reference.txt --agent-file seed_3096.wav --output-dir extracted_voices
