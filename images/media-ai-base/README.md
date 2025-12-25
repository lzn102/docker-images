# Media-AI Base Image

基于 NVIDIA CUDA 的 AI 语音识别基础镜像。

## 包含内容

- CUDA 12.4 Runtime
- Python 3.10
- FFmpeg
- faster-whisper (large-v3 模型已预下载)
- audio-separator (3 个模型已预下载)

## 使用

```bash
docker pull ghcr.io/YOUR_USERNAME/media-ai-base:latest
```
