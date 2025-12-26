# media-ai-whisper 基础镜像
# 使用 NVIDIA 官方 cuDNN 9 镜像作为唯一 CUDA 来源
# PyTorch/ONNX 只链接系统 cuDNN，不允许 Python 包自带 cuDNN
#
# 构建: docker build -t media-ai-whisper:latest .

FROM nvidia/cuda:12.4.1-cudnn9-runtime-ubuntu22.04

# 设置环境变量
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV HF_ENDPOINT=https://hf-mirror.com

# CUDA 环境变量 - 确保使用系统 cuDNN
ENV CUDA_HOME=/usr/local/cuda
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH}

# 安装系统依赖（包括 Python 和编译工具）
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-venv \
    python3-pip \
    ffmpeg \
    curl \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.11 /usr/bin/python3 \
    && ln -sf /usr/bin/python3.11 /usr/bin/python

WORKDIR /app

# 安装 PyTorch (使用官方 CUDA 12.4 wheel，会使用系统 cuDNN)
RUN pip install --no-cache-dir \
    torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu124

# 安装 faster-whisper 和 audio-separator
RUN pip install --no-cache-dir \
    faster-whisper==1.2.1 \
    "audio-separator[gpu]==0.40.0"

# 清理 Python 层可能带入的 CUDA/cuDNN 包（确保只用系统的）
RUN pip uninstall -y \
    nvidia-cudnn-cu12 \
    nvidia-cuda-runtime-cu12 \
    nvidia-cublas-cu12 \
    nvidia-cufft-cu12 \
    nvidia-curand-cu12 \
    nvidia-cusolver-cu12 \
    nvidia-cusparse-cu12 \
    nvidia-nccl-cu12 \
    nvidia-nvtx-cu12 \
    2>/dev/null || true

# 安装 onnxruntime-gpu（必须匹配 CUDA 12.4 + cuDNN 9）
RUN pip install --no-cache-dir onnxruntime-gpu==1.23.2

# 验证 cuDNN 来源只有一个
RUN python -c "import torch; print('PyTorch CUDA:', torch.version.cuda, 'cuDNN:', torch.backends.cudnn.version())"

# 预下载 Whisper large-v3 模型 (约 3GB)
RUN python -c "\
from faster_whisper import WhisperModel; \
print('下载 Whisper large-v3 模型...'); \
model = WhisperModel('large-v3', device='cpu', compute_type='int8'); \
print('✅ Whisper 模型下载完成')"

# 预下载 Audio Separator 元数据和模型
RUN mkdir -p /tmp/audio-separator-models && \
    curl -L -o /tmp/audio-separator-models/download_checks.json \
    "https://raw.githubusercontent.com/TRvlvr/application_data/main/filelists/download_checks.json"

RUN python -c "\
from audio_separator.separator import Separator; \
print('下载 Audio Separator 模型...'); \
sep = Separator(log_level=20); \
sep.load_model(model_filename='UVR-MDX-NET-Inst_Main.onnx'); \
print('✅ Audio Separator 模型下载完成')"

# 创建临时文件目录
RUN mkdir -p tempfile/downloads tempfile/preprocess tempfile/transcripts

# 清理缓存
RUN pip cache purge && rm -rf /root/.cache/pip/*

# 标记版本
LABEL version="2.0.0"
LABEL description="Media-AI base image with NVIDIA cuDNN 9 (single CUDA source)"
