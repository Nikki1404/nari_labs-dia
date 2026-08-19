FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MODEL_ID=nari-labs/Dia-1.6B-0626 \
    HF_HOME=/app/hf-cache

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-dev \
    libsndfile1 \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install --index-url https://download.pytorch.org/whl/cu126 torch torchvision torchaudio && \
    python3 -m pip install -r requirements.txt

# Download model weights during image build so runtime does not need to fetch them.
RUN python3 - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download("nari-labs/Dia-1.6B-0626")
PY

COPY server.py client.py README.md ./

EXPOSE 8000

CMD ["python3", "-m", "uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
