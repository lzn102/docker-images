# media-ai-whisper 基础镜像
# 使用 NVIDIA 官方 cuDNN 镜像 (runtime 版本减小体积)
# 临时安装编译工具，用完后清理
#
# 构建: docker build -t media-ai-whisper:latest .

FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV HF_ENDPOINT=https://hf-mirror.com
ENV CUDA_HOME=/usr/local/cuda
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH}

# 安装运行时依赖 + 临时编译工具
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-dev \
    python3-pip \
    ffmpeg \
    curl \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.11 /usr/bin/python3 \
    && ln -sf /usr/bin/python3.11 /usr/bin/python

WORKDIR /app

# 安装 PyTorch 2.4.1 (匹配 cuDNN 9.1，稳定版本)
# 使用 cu121 版本，兼容性好
RUN pip install --no-cache-dir \
    torch==2.4.1 torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu121

# 安装核心 AI 库
# 关键：显式安装 onnxruntime-gpu，防止后续依赖拉取 CPU 版本
RUN pip install --no-cache-dir \
    onnxruntime-gpu \
    faster-whisper==1.2.1 \
    "audio-separator[gpu]==0.40.0"

# 注入 LD_LIBRARY_PATH (关键优化)
# 让所有基于此镜像的应用都能自动找到 PyTorch 自带的 NVIDIA 运行时库
# 这样不仅解决了 Docker 内的库加载问题，也与我们在 Host 上的 run.sh 逻辑保持一致
ENV LD_LIBRARY_PATH=/usr/local/lib/python3.11/site-packages/torch/lib:/usr/local/lib/python3.11/site-packages/nvidia/cudnn/lib:/usr/local/lib/python3.11/site-packages/nvidia/cublas/lib:${LD_LIBRARY_PATH}

# 清理编译工具（减小体积）
RUN apt-get purge -y build-essential python3.11-dev && \
    apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/*

# 验证
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

# 创建临时文件目录
RUN mkdir -p tempfile/downloads tempfile/preprocess tempfile/transcripts

# 清理缓存
RUN pip cache purge && rm -rf /root/.cache/pip/*

# 标记版本
LABEL version="2.0.0"
LABEL description="Media-AI base image with NVIDIA cuDNN (runtime, smaller size)"
