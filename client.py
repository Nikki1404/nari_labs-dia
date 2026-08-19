import argparse
import re
import time
from pathlib import Path

import requests
import soundfile as sf
from jiwer import cer, wer


def now_ns():
    return time.perf_counter_ns()


def elapsed_ms(start_ns, end_ns=None):
    if end_ns is None:
        end_ns = now_ns()
    return (end_ns - start_ns) / 1_000_000.0


def clean_reference(text: str) -> str:
    text = re.sub(r"\[S\d+\]", " ", text)
    text = re.sub(r"\([^)]*\)", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def transcribe_for_accuracy(audio_path: Path, model_size: str, device: str):
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise RuntimeError(
            "Accuracy requested but faster-whisper is not installed. "
            "Run: pip install faster-whisper jiwer"
        ) from exc

    compute_type = "float16" if device == "cuda" else "int8"
    model = WhisperModel(model_size, device=device, compute_type=compute_type)
    segments, _ = model.transcribe(str(audio_path), language="en", beam_size=5)
    return " ".join(segment.text.strip() for segment in segments).strip()


def main():
    parser = argparse.ArgumentParser(description="Dia TTS latency/accuracy client")
    parser.add_argument("--server", default="http://localhost:8000")
    parser.add_argument("--text", required=True)
    parser.add_argument("--output", default="output.wav")
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--accuracy", action="store_true", help="ASR-transcribe output and compute WER/CER")
    parser.add_argument("--asr-model", default="small.en", help="faster-whisper model used for accuracy")
    parser.add_argument("--asr-device", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args()

    payload = {"text": args.text, "max_new_tokens": args.max_new_tokens, "seed": args.seed}
    url = f"{args.server.rstrip('/')}/tts"

    session = requests.Session()
    request_start_ns = now_ns()

    response = session.post(url, json=payload, timeout=args.timeout, stream=True)
    headers_received_ns = now_ns()
    ttfb_ms = elapsed_ms(request_start_ns, headers_received_ns)

    if not response.ok:
        raise SystemExit(f"Request failed ({response.status_code}): {response.text}")

    output = Path(args.output)
    first_audio_ns = None
    total_bytes = 0

    with output.open("wb") as f:
        for chunk in response.iter_content(chunk_size=4096):
            if not chunk:
                continue
            if first_audio_ns is None:
                first_audio_ns = now_ns()
            f.write(chunk)
            total_bytes += len(chunk)

    request_done_ns = now_ns()
    ttfa_ms = elapsed_ms(request_start_ns, first_audio_ns) if first_audio_ns else float("nan")
    total_ms = elapsed_ms(request_start_ns, request_done_ns)
    download_after_first_audio_ms = elapsed_ms(first_audio_ns, request_done_ns) if first_audio_ns else 0.0

    info = sf.info(str(output))
    audio_duration_s = float(info.duration)
    client_rtf = (total_ms / 1000.0) / audio_duration_s if audio_duration_s > 0 else 0.0

    print("=" * 80)
    print("CLIENT LATENCY")
    print("=" * 80)
    print(f"HTTP request -> headers/TTFB : {ttfb_ms:.2f} ms")
    print(f"TTFT / TTFA                 : {ttfa_ms:.2f} ms")
    print(f"First audio -> complete      : {download_after_first_audio_ms:.2f} ms")
    print(f"CLIENT TOTAL                 : {total_ms:.2f} ms")
    print(f"Audio duration               : {audio_duration_s:.3f} s")
    print(f"Client E2E RTF               : {client_rtf:.4f}")
    print(f"Response size                : {total_bytes / 1024:.2f} KiB")

    print("\n" + "=" * 80)
    print("SERVER LATENCY")
    print("=" * 80)
    print(f"Preprocess                    : {response.headers.get('X-Preprocess-Time-MS', 'n/a')} ms")
    print(f"Inference / generation        : {response.headers.get('X-Inference-Time-MS', 'n/a')} ms")
    print(f"Decode                        : {response.headers.get('X-Decode-Time-MS', 'n/a')} ms")
    print(f"WAV encoding                  : {response.headers.get('X-Encoding-Time-MS', 'n/a')} ms")
    print(f"SERVER TOTAL                  : {response.headers.get('X-Server-Total-MS', 'n/a')} ms")
    print(f"Server RTF                    : {response.headers.get('X-RTF', 'n/a')}")
    print(f"Request ID                    : {response.headers.get('X-Request-ID', 'n/a')}")
    print(f"Sample rate                   : {response.headers.get('X-Sample-Rate', 'n/a')} Hz")

    if args.accuracy:
        accuracy_start_ns = now_ns()
        hypothesis = transcribe_for_accuracy(output, args.asr_model, args.asr_device)
        asr_ms = elapsed_ms(accuracy_start_ns)
        reference = clean_reference(args.text)
        word_error_rate = wer(reference, hypothesis)
        char_error_rate = cer(reference, hypothesis)
        word_accuracy = max(0.0, 1.0 - word_error_rate) * 100.0
        char_accuracy = max(0.0, 1.0 - char_error_rate) * 100.0

        print("\n" + "=" * 80)
        print("TTS ACCURACY (ASR-BASED)")
        print("=" * 80)
        print(f"Reference                     : {reference}")
        print(f"ASR transcript                : {hypothesis}")
        print(f"WER                           : {word_error_rate * 100:.2f}%")
        print(f"Word accuracy                 : {word_accuracy:.2f}%")
        print(f"CER                           : {char_error_rate * 100:.2f}%")
        print(f"Character accuracy            : {char_accuracy:.2f}%")
        print(f"Accuracy ASR evaluation time  : {asr_ms:.2f} ms")
        print("NOTE: this measures intelligibility/content fidelity through ASR; it is not a MOS or voice-quality score.")

    print("\nSaved audio:", output.resolve())


if __name__ == "__main__":
    main()
