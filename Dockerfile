FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu22.04


ENV http_proxy="http://163.116.128.80:8080"
ENV https_proxy="http://163.116.128.80:8080"

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
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
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

RUN python3 -m pip install --upgrade pip setuptools wheel

RUN pip install --no-cache-dir -r requirements.txt

COPY server.py .
COPY client.py .

EXPOSE 8000

CMD ["python3", "server.py"]
