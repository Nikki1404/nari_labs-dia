# Dia 1.6B Standalone TTS App

Minimal standalone GPU TTS API using `nari-labs/Dia-1.6B-0626`.

Dia generates English dialogue from transcripts using speaker tags such as `[S1]` and `[S2]`. It can also generate non-verbal expressions such as `(laughs)` and `(coughs)`.

## Project structure

```text
dia_standalone_app/
├── server.py
├── client.py
├── Dockerfile
├── requirements.txt
└── README.md
```

## Requirements

- NVIDIA GPU
- NVIDIA driver + NVIDIA Container Toolkit
- Docker
- Around 10 GB GPU VRAM or more is recommended for the full Dia model

## Build

```bash
docker build -t dia-tts:latest .
```

The Docker build downloads `nari-labs/Dia-1.6B-0626` into the image.

## Run

```bash
docker run --rm --gpus all -p 8000:8000 --name dia-tts dia-tts:latest
```

Check health:

```bash
curl http://localhost:8000/health
```

## Generate speech

Install the client dependency locally if needed:

```bash
pip install requests
```

Single speaker:

```bash
python client.py \
  --server http://localhost:8000 \
  --text "[S1] Hello. This is Dia running from my standalone API." \
  --output hello.wav
```

Dialogue:

```bash
python client.py \
  --server http://localhost:8000 \
  --text "[S1] Hi, how are you today? [S2] I'm doing great. (laughs) How about you?" \
  --output dialogue.wav
```

For more consistent generated voices, reuse a fixed seed:

```bash
python client.py \
  --server http://localhost:8000 \
  --text "[S1] This voice should be more repeatable." \
  --seed 42 \
  --output seeded.wav
```

## API

### `GET /health`

Returns service/model/GPU status.

### `POST /tts`

Example request body:

```json
{
  "text": "[S1] Hello from Dia.",
  "max_new_tokens": 1024,
  "seed": 42
}
```

The response body is a WAV file at 44.1 kHz.

Response headers include:

- `X-Request-ID`
- `X-Generation-Time-MS`
- `X-Sample-Rate`

## Notes

- Dia currently supports English generation.
- The upstream model card says the full model requires roughly 10 GB VRAM.
- The first model initialization can take longer because supporting audio-codec assets may also be initialized/downloaded.
- This starter version uses request/response HTTP rather than streaming WebSocket audio. That keeps the initial architecture simple. A WebSocket streaming layer can be added next without changing the model container concept.

## Latency and accuracy metrics

The updated client prints:

- HTTP request -> headers / TTFB
- TTFT / TTFA (time to first audio byte)
- first-audio-to-complete time
- total client latency
- audio duration
- client E2E RTF
- server preprocessing, inference, decode, WAV encoding and total time
- server RTF
- optional ASR-based WER, CER, word accuracy and character accuracy

Basic latency test:

```bash
python client.py --server http://localhost:8000 --text "[S1] Hello, this is a Dia latency test." --output latency_test.wav
```

Latency + content accuracy test:

```bash
python client.py --server http://localhost:8000 --text "[S1] Welcome to Inspira Financial. How can I help you today?" --output accuracy_test.wav --accuracy
```

Use GPU for the optional Whisper evaluator if available:

```bash
python client.py --server http://localhost:8000 --text "[S1] Welcome to Inspira Financial." --output accuracy_test.wav --accuracy --asr-device cuda --asr-model small.en
```

`WER`/`CER` are proxy measurements for intelligibility and content fidelity. They do not measure naturalness, emotion quality, speaker similarity, or human MOS.

Because the current Dia generation path produces the waveform before returning the HTTP response, TTFB and TTFA will normally be very close. True streaming TTFA requires an incremental/streaming generation implementation.
