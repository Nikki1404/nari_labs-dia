FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV http_proxy="http://163.116.128.80:8080"
ENV https_proxy="http://163.116.128.80:8080"

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

ENV CC=/usr/bin/gcc
ENV CXX=/usr/bin/g++

ENV MODEL_ID="nari-labs/Dia-1.6B-0626"

ENV HF_HOME=/app/hf_cache
ENV TRANSFORMERS_CACHE=/app/hf_cache

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
        libportaudio2 \
        ffmpeg \
    && rm -rf /var/lib/apt/lists/*


RUN python3 -m pip install --upgrade \
    pip \
    setuptools \
    wheel


RUN python3 -m pip install \
    torch==2.6.0 \
    --index-url https://download.pytorch.org/whl/cu124

WORKDIR /app

COPY requirements.txt /app/requirements.txt


RUN python3 -m pip install \
    -r /app/requirements.txt



RUN python3 -c "\
from huggingface_hub import snapshot_download; \
snapshot_download( \
    repo_id='nari-labs/Dia-1.6B-0626', \
    allow_patterns=[ \
        '*.json', \
        '*.safetensors', \
        '*.txt', \
        '*.model' \
    ] \
); \
print('Dia model downloaded successfully')"


COPY server.py /app/server.py

RUN python3 -c "\
import torch; \
print('Torch version :', torch.__version__); \
print('Torch CUDA    :', torch.version.cuda)"

EXPOSE 8000

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000", "--log-level", "info"]
