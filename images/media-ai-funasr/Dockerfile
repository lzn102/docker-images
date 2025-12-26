# media-ai-funasr 基础镜像
# 使用 NVIDIA 官方 cuDNN 镜像
# 预下载 Fun-ASR-Nano-2512 模型
#
# 构建: docker build -t media-ai-funasr:latest .

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

# 安装 PyTorch (使用官方 CUDA 12.4 wheel)
RUN pip install --no-cache-dir \
    torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu124

# 安装 Fun-ASR 依赖
RUN pip install --no-cache-dir \
    "funasr>=1.2.7" \
    transformers \
    modelscope

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

# 清理编译工具
RUN apt-get purge -y build-essential python3.11-dev && \
    apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/*

# 验证 PyTorch
RUN python -c "import torch; print('PyTorch CUDA:', torch.version.cuda, 'cuDNN:', torch.backends.cudnn.version())"

# 克隆 Fun-ASR 项目（获取 model.py）
RUN git clone --depth 1 https://github.com/FunAudioLLM/Fun-ASR.git /app/Fun-ASR

# 预下载 Fun-ASR-Nano-2512 模型
WORKDIR /app/Fun-ASR
RUN python -c "\
from funasr import AutoModel; \
print('下载 Fun-ASR-Nano-2512 模型...'); \
model = AutoModel( \
    model='FunAudioLLM/Fun-ASR-Nano-2512', \
    trust_remote_code=True, \
    remote_code='./model.py', \
    device='cpu' \
); \
print('✅ Fun-ASR 模型下载完成')"

WORKDIR /app

# 创建临时文件目录
RUN mkdir -p tempfile/downloads tempfile/preprocess tempfile/transcripts

# 清理缓存
RUN pip cache purge && rm -rf /root/.cache/pip/*

# 标记版本
LABEL version="1.0.0"
LABEL description="Media-AI Fun-ASR base image with pre-downloaded Fun-ASR-Nano-2512 model"
