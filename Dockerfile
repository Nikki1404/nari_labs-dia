FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV http_proxy="http://163.116.128.80:8080"
ENV https_proxy="http://163.116.128.80:8080"

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

# Triton needs a C compiler
ENV CC=/usr/bin/gcc
ENV CXX=/usr/bin/g++

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    gcc \
    g++ \
    build-essential \
    git \
    curl \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN python3 -m pip install --upgrade pip setuptools wheel

# IMPORTANT:
# Install a PyTorch version explicitly built for CUDA 12.4.
RUN pip3 install \
    torch==2.6.0 \
    torchvision==0.21.0 \
    torchaudio==2.6.0 \
    --index-url https://download.pytorch.org/whl/cu124

COPY requirements.txt /app/requirements.txt

# requirements.txt should NOT contain torch/torchvision/torchaudio
RUN pip3 install -r /app/requirements.txt

COPY server.py /app/server.py
COPY client.py /app/client.py

EXPOSE 8000

CMD ["python3", "server.py"]
