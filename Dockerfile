FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV http_proxy="http://163.116.128.80:8080"
ENV https_proxy="http://163.116.128.80:8080"

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

ENV CC=/usr/bin/gcc
ENV CXX=/usr/bin/g++


RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-dev \
    gcc \
    g++ \
    build-essential \
    git \
    libsndfile1 \
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


COPY server.py /app/server.py


RUN python3 -c "\
import torch; \
print('Torch version:', torch.__version__); \
print('Torch CUDA:', torch.version.cuda)"


EXPOSE 8000

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000", "--log-level", "info"]
