FROM python:3.10-slim AS builder

RUN apt-get update && \
    apt-get install --yes --no-install-recommends \
      build-essential cmake ninja-build git

WORKDIR /app

COPY pyproject.toml setup.py CMakeLists.txt MANIFEST.in README.md ./
COPY src/piper/ ./src/piper/
COPY script/setup script/dev_build script/package ./script/
RUN script/setup --dev
RUN script/dev_build
RUN script/package

# -----------------------------------------------------------------------------

FROM python:3.10-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_BREAK_SYSTEM_PACKAGES=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    unzip \
    && rm -rf /var/lib/apt/lists/*

COPY http-service/requirements.txt .

RUN pip3 install --no-cache-dir --upgrade pip setuptools wheel && \
    pip3 install --no-cache-dir numpy==1.26.0 && \
    pip3 install --no-cache-dir --no-build-isolation -r requirements.txt

COPY http-service/ ./http-service/
RUN mv http-service/prepare_data.sh . && chmod +x ./prepare_data.sh

RUN ./prepare_data.sh

RUN apt-get purge -y --auto-remove unzip

COPY --from=builder /app/dist/piper_tts-*linux*.whl ./dist/
RUN pip3 install ./dist/piper_tts-*linux*.whl

RUN useradd -m -u 1000 ttsuser && \
    chown -R ttsuser:ttsuser /app

USER ttsuser

ENV TTS_MODEL_DIR=./data/piper/models
ENV TTS_VOICES_FILE=./data/piper/voices.json
ENV TTS_PORT=8001

ENV EZAFE_MODEL_PATH=./ezafe_model_quantized
ENV HOMOGRAPH_MODEL_PATH=./onnx-homo-ge2pe

EXPOSE 8001

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8001/api/tts/health || exit 1

CMD ["python", "http-service/main.py"]
