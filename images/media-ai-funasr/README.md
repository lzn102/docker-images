# media-ai-funasr

基于 NVIDIA CUDA 官方镜像的 Fun-ASR 语音识别服务基础镜像。

## 特点

- **基础镜像**: `nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04`
- **预装模型**: Fun-ASR-Nano-2512 (800M 参数)
- **支持语言**: 31 种语言，含 7 大方言 + 26 口音

## 预装依赖

- funasr (Fun-ASR 框架)
- transformers (Hugging Face)
- PyTorch (CUDA 12.4)
- ffmpeg (音视频处理)

## 构建

```bash
docker build -t media-ai-funasr:latest .
```

## 使用

```dockerfile
FROM media-ai-funasr:latest

WORKDIR /app
COPY model.py .  # 从 Fun-ASR 项目复制
COPY main.py .

CMD ["python", "main.py"]
```

## 模型特性

- 支持中文方言：吴语、粤语、闽语、客家话、赣语、湘语、晋语
- 支持 26 个地区口音
- 歌词识别、说唱识别
- 远场高噪声识别（93% 准确率）
