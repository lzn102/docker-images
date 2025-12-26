# media-ai-whisper 基础镜像 (多阶段构建)
# 使用 NVIDIA 官方 cuDNN 9 镜像作为唯一 CUDA 来源
# 第一阶段用 devel 编译，第二阶段用 runtime 减小体积
#
# 构建: docker build -t media-ai-whisper:latest .

#====================================================================
# 第一阶段: 编译环境
#====================================================================
FROM nvidia/cuda:12.4.1-cudnn9-devel-ubuntu22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV HF_ENDPOINT=https://hf-mirror.com

# 安装 Python 和编译工具
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-dev \
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

# 安装 PyTorch (使用官方 CUDA 12.4 wheel)
RUN pip install --no-cache-dir \
    torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu124

# 安装 faster-whisper 和 audio-separator
RUN pip install --no-cache-dir \
    faster-whisper==1.2.1 \
    "audio-separator[gpu]==0.40.0"

# 清理 Python 层的 CUDA/cuDNN 包（确保只用系统的）
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

# 安装 onnxruntime-gpu
RUN pip install --no-cache-dir onnxruntime-gpu==1.23.2

# 验证 cuDNN
RUN python -c "import torch; print('PyTorch CUDA:', torch.version.cuda, 'cuDNN:', torch.backends.cudnn.version())"

# 预下载 Whisper large-v3 模型
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

# 清理缓存
RUN pip cache purge && rm -rf /root/.cache/pip/*

#====================================================================
# 第二阶段: 运行时环境 (体积更小)
#====================================================================
FROM nvidia/cuda:12.4.1-cudnn9-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV HF_ENDPOINT=https://hf-mirror.com
ENV CUDA_HOME=/usr/local/cuda
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH}

# 只安装运行时依赖（不需要编译工具）
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3-pip \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.11 /usr/bin/python3 \
    && ln -sf /usr/bin/python3.11 /usr/bin/python

WORKDIR /app

# 从编译阶段复制 Python 包
COPY --from=builder /usr/local/lib/python3.11/dist-packages /usr/local/lib/python3.11/dist-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# 从编译阶段复制预下载的模型
COPY --from=builder /root/.cache/huggingface /root/.cache/huggingface
COPY --from=builder /tmp/audio-separator-models /tmp/audio-separator-models

# 创建临时文件目录
RUN mkdir -p tempfile/downloads tempfile/preprocess tempfile/transcripts

# 验证
RUN python -c "import torch; print('PyTorch CUDA:', torch.version.cuda, 'cuDNN:', torch.backends.cudnn.version())"
RUN python -c "from faster_whisper import WhisperModel; print('✅ faster-whisper 可用')"

# 标记版本
LABEL version="2.0.0"
LABEL description="Media-AI base image with NVIDIA cuDNN 9 (multi-stage build)"
