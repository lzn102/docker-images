# media-ai 基础镜像
# 使用 PyTorch 官方镜像，自带完整的 CUDA + cuDNN + PyTorch 环境
#
# 优化策略:
# 1. 基础镜像 & 系统依赖 (极少变动)
# 2. 模型文件下载 (大文件，极少变动)
# 3. 项目依赖同步 (pyproject.toml 变动时触发)
#
# 构建: docker build -t media-ai-base:latest .

# PyTorch 官方镜像: CUDA 12.1 + cuDNN 8 + PyTorch 2.1.0
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

# 设置环境变量
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# 1. 安装系统依赖 & uv
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/* \
    && pip install uv

# 设置工作目录
WORKDIR /app

# 2. 创建 venv 并安装下载工具 (用于预下载模型)
RUN uv venv .venv && \
    . .venv/bin/activate && \
    uv pip install "faster-whisper>=1.0.0" "audio-separator[gpu]>=0.18.0"

# 3. 预下载 Whisper large-v3 模型
RUN . .venv/bin/activate && python -c "\
from faster_whisper import WhisperModel; \
print('下载 Whisper large-v3...'); \
WhisperModel('large-v3', device='cpu', compute_type='int8'); \
print('完成')"

# 4. 预下载 Audio Separator 元数据和模型
RUN mkdir -p /tmp/audio-separator-models && \
    curl -L -o /tmp/audio-separator-models/download_checks.json \
    "https://raw.githubusercontent.com/TRvlvr/application_data/main/filelists/download_checks.json"

# 预下载 Audio Separator 模型
RUN . .venv/bin/activate && python -c "\
from audio_separator.separator import Separator; \
print('下载 Audio Separator 模型...'); \
sep = Separator(log_level=20); \
sep.load_model(model_filename='UVR-MDX-NET-Inst_Main.onnx'); \
sep.load_model(model_filename='MDX23C-8KFFT-InstVoc_HQ.ckpt'); \
sep.load_model(model_filename='model_bs_roformer_ep_317_sdr_12.9755.ckpt'); \
print('完成')"

# 5. 复制依赖配置文件
COPY pyproject.toml README.md ./

# 6. 同步完整项目依赖
RUN uv sync

# 7. 创建临时文件目录 & 清理
RUN mkdir -p tempfile/downloads tempfile/preprocess tempfile/transcripts && \
    rm -rf /root/.cache/pip/*

# 标记版本
LABEL version="2.0.0"
LABEL description="Media-AI base image with PyTorch official CUDA runtime"
