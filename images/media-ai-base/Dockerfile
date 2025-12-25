# media-ai 基础镜像
# 包含 CUDA、Python、Whisper 模型和音频处理依赖
#
# 构建: docker build -t media-ai-base:latest .
# 拉取: docker pull ghcr.io/YOUR_USERNAME/media-ai-base:latest

FROM nvidia/cuda:12.4.0-cudnn-runtime-ubuntu22.04

# 设置环境变量
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# 安装系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-venv \
    python3-pip \
    ffmpeg \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.10 /usr/bin/python3 \
    && ln -sf /usr/bin/python3.10 /usr/bin/python

# 安装 uv
RUN pip3 install uv

# 设置工作目录
WORKDIR /app

# 复制依赖配置文件
COPY pyproject.toml README.md ./

# 安装 Python 依赖
RUN uv sync

# 预下载 Whisper large-v3 模型
RUN uv run python3 -c "from faster_whisper import WhisperModel; print('下载 Whisper large-v3...'); WhisperModel('large-v3', device='cpu', compute_type='int8'); print('完成')"

# 预下载 Audio Separator 模型
RUN uv run python3 -c "\
from audio_separator.separator import Separator; \
print('下载 Audio Separator 模型...'); \
sep = Separator(log_level=20); \
sep.load_model(model_filename='UVR-MDX-NET-Inst_Main.onnx'); \
sep.load_model(model_filename='MDX23C-8KFFT-InstVoc_HQ.ckpt'); \
sep.load_model(model_filename='model_bs_roformer_ep_317_sdr_12.9755.ckpt'); \
print('完成')"

# 创建临时文件目录
RUN mkdir -p tempfile/downloads tempfile/preprocess tempfile/transcripts

# 清理缓存
RUN rm -rf /root/.cache/pip/* /tmp/*

# 标记版本
LABEL version="1.0.0"
LABEL description="Media-AI base image with Whisper and Audio Separator"
