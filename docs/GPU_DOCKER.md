GPU Docker image
================

This repository includes a GPU-enabled Dockerfile at `Dockerfile.gpu`.

Build (requires NVIDIA Container Toolkit / Docker with GPU support):

```bash
# Build the image (from project root)
docker build -f Dockerfile.gpu -t piper:gpu .
```

Run (example):

```bash
# Run with nvidia runtime (Docker >= 19.03 uses --gpus)
docker run --gpus all -p 8001:8001 --rm piper:gpu
```

Notes:
- The GPU image uses CUDA 12.1 cuDNN runtime and installs PyTorch wheels for `cu121`.
- If you need a different CUDA version, update `Dockerfile.gpu`'s base image and the
  PyTorch index URL (`--index-url https://download.pytorch.org/whl/cuXXX`).
- The `http-service/requirements.txt` contains a CPU-specific torch index. The GPU
  Dockerfile installs other requirements and then installs the appropriate CUDA torch
  wheel separately.
