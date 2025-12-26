# media-ai 基础镜像
# 包含 CUDA、Python、Whisper 模型和音频处理依赖
#
# 优化策略:
# 1. 基础镜像 & 系统依赖 (极少变动)
# 2. 只有模型下载工具的 venv (极少变动)
# 3. 模型文件下载 (大文件，极少变动)
# 4. 项目依赖同步 (pyproject.toml 变动时触发)
#
# 构建: docker build -t media-ai-base:latest .
# 拉取: docker pull ghcr.io/YOUR_USERNAME/media-ai-base:latest

FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

# 设置环境变量
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# 1. 安装系统依赖 & uv
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-venv \
    python3-pip \
    ffmpeg \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.10 /usr/bin/python3 \
    && ln -sf /usr/bin/python3.10 /usr/bin/python \
    && pip3 install uv

# 设置工作目录
WORKDIR /app

# 2. 创建临时 venv 并安装下载工具
# 这一步手动安装 faster-whisper 和 audio-separator 用于下载模型
# 这样即使 pyproject.toml 变动，只要这一行不改，模型下载层就能被缓存
RUN uv venv .venv && \
    . .venv/bin/activate && \
    uv pip install "faster-whisper>=1.0.0" "audio-separator[gpu]>=0.18.0"

# 3. 预下载 Whisper large-v3 模型
# 模型会被下载到 /root/.cache/huggingface 等目录，与 .venv 无关
RUN . .venv/bin/activate && python3 -c "from faster_whisper import WhisperModel; print('下载 Whisper large-v3...'); WhisperModel('large-v3', device='cpu', compute_type='int8'); print('完成')"

# 4. 预下载 Audio Separator 元数据和模型
# 先下载元数据文件，避免运行时访问 GitHub
RUN mkdir -p /tmp/audio-separator-models && \
    curl -L -o /tmp/audio-separator-models/download_checks.json \
    "https://raw.githubusercontent.com/TRvlvr/application_data/main/filelists/download_checks.json"

# 预下载 Audio Separator 模型
RUN . .venv/bin/activate && python3 -c "from audio_separator.separator import Separator; print('下载 Audio Separator 模型...'); sep = Separator(log_level=20); sep.load_model(model_filename='UVR-MDX-NET-Inst_Main.onnx'); sep.load_model(model_filename='MDX23C-8KFFT-InstVoc_HQ.ckpt'); sep.load_model(model_filename='model_bs_roformer_ep_317_sdr_12.9755.ckpt'); print('完成')"

# 5. 复制依赖配置文件 (变动频繁部分开始)
COPY pyproject.toml README.md ./

# 6. 同步完整项目依赖
# 这会更新 .venv 中的包版本以匹配 pyproject.toml
RUN uv sync

# 7. 创建临时文件目录 & 清理
RUN mkdir -p tempfile/downloads tempfile/preprocess tempfile/transcripts && \
    rm -rf /root/.cache/pip/* /tmp/*

# 标记版本
LABEL version="1.0.1"
LABEL description="Media-AI base image with optimized layering"
