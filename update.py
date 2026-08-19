/usr/local/lib/python3.10/dist-packages/transformers/tokenization_utils_base.py:2355: UserWarning: `max_length` is ignored when `padding`=`True` and there is no truncation strategy. To pad to max length, use `padding='max_length'`.
  warnings.warn(
INFO:     172.17.0.1:46786 - "POST /tts HTTP/1.1" 500 Internal Server Error

(venv) PS C:\Users\re_nikitav\Documents\nari_labs-dia>  python client.py --server http://localhost:8000  --text "[S1] Hello. This is Dia running from my standalone API." --output hello.wav
Request failed (500): {"detail":"TTS generation failed: Failed to find C compiler. Please specify via CC environment variable or set triton.knobs.build.impl."}

(base) root@EC03-E01-AICOE1:/home/CORP/re_nikitav/nari_labs-dia# docker run --rm --gpus all -p 8000:8000 dia-tts

==========
== CUDA ==
==========

CUDA Version 12.6.3

Container image Copyright (c) 2016-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

This container image and its contents are governed by the NVIDIA Deep Learning Container License.
By pulling and using the container, you accept the terms and conditions of this license:
https://developer.nvidia.com/ngc/nvidia-deep-learning-container-license

A copy of this license is made available in this container at /NGC-DL-CONTAINER-LICENSE for your convenience.

/usr/local/lib/python3.10/dist-packages/torch/cuda/__init__.py:188: UserWarning: CUDA initialization: The NVIDIA driver on your system is too old (found version 12060). Please update your GPU driver by downloading and installing a new version from the URL: http://www.nvidia.com/Download/index.aspx Alternatively, go to: https://pytorch.org to install a PyTorch version that has been compiled with your version of the CUDA driver. (Triggered internally at /__w/pytorch/pytorch/c10/cuda/CUDAFunctions.cpp:119.)
  return torch._C._cuda_getDeviceCount() > 0
/app/server.py:37: DeprecationWarning:
        on_event is deprecated, use lifespan event handlers instead.

        Read more about it in the
        [FastAPI docs for Lifespan Events](https://fastapi.tiangolo.com/advanced/events/).

  @app.on_event("startup")


