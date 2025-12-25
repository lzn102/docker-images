# Docker Images

自用 Docker 镜像仓库，通过 GitHub Actions 自动构建并推送到 GitHub Container Registry。

## 镜像列表

| 镜像            | 说明                                   | 大小  |
| --------------- | -------------------------------------- | ----- |
| `media-ai-base` | AI 语音识别基础镜像（含 Whisper 模型） | ~10GB |

## 使用方式

### 拉取镜像

```bash
# 公开仓库
docker pull ghcr.io/YOUR_USERNAME/media-ai-base:latest

# 私有仓库
echo $GITHUB_TOKEN | docker login ghcr.io -u YOUR_USERNAME --password-stdin
docker pull ghcr.io/YOUR_USERNAME/media-ai-base:latest
```

### 手动构建

在 GitHub Actions 页面点击 "Run workflow"，选择要构建的镜像。

## 目录结构

```
docker-images/
├── .github/workflows/build.yml
├── images/
│   └── media-ai-base/
│       ├── Dockerfile
│       ├── pyproject.toml
│       └── README.md
└── README.md
```
