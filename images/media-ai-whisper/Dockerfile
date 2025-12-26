# media-ai-whisper 基础镜像
# 只包含大型文件和模型，不安装应用依赖
#
# 构建: docker build -t media-ai-whisper:latest .

FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime

# 设置环境变量
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV HF_ENDPOINT=https://hf-mirror.com

# 安装系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 只安装用于下载模型的最小依赖（版本与 pyproject.toml 保持一致）
RUN pip install --no-cache-dir faster-whisper==1.2.1 "audio-separator[gpu]==0.40.0"

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

# 清理不需要保留的包 (模型已下载到缓存目录)
# 保留 PyTorch，应用依赖由 deploy/Dockerfile 安装

# 标记版本
LABEL version="1.0.0"
LABEL description="Media-AI base image with pre-downloaded models"
