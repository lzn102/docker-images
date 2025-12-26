# media-ai-whisper

基于 PyTorch 官方镜像的 Whisper 语音识别服务基础镜像。

## 特点

- **基础镜像**: `pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime`
- **无虚拟环境**: 直接使用 pip 安装，避免多层环境冲突
- **预装模型**: Whisper large-v3 + Audio Separator

## 预装依赖

- faster-whisper (语音识别)
- audio-separator (人声分离)
- fastapi + uvicorn (Web 服务)
- ffmpeg (音视频处理)

## 构建

```bash
docker build -t media-ai-whisper:latest .
```

## 使用

```dockerfile
FROM media-ai-whisper:latest

WORKDIR /app
COPY . .

CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "5555"]
```
