FROM python:3.10-slim AS builder

RUN apt-get update && \
    apt-get install --yes --no-install-recommends \
      build-essential cmake ninja-build git

WORKDIR /app

RUN pip install --upgrade pip && \
    pip install scikit-build setuptools wheel cmake ninja

COPY pyproject.toml setup.py CMakeLists.txt MANIFEST.in README.md ./
COPY script/ ./script/

RUN mkdir -p src/piper
COPY src/piper/espeakbridge.c src/piper/espeakbridge.pyi ./src/piper/

RUN python setup.py build_ext --inplace

COPY src/piper/ ./src/piper/
RUN python setup.py bdist_wheel --dist-dir dist

# -----------------------------------------------------------------------------

FROM python:3.10-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_BREAK_SYSTEM_PACKAGES=1

WORKDIR /app

COPY http-service/requirements.txt .

RUN pip install gdown
COPY http-service/prepare_data.sh ./prepare_data.sh
RUN apt-get update && apt-get install -y --no-install-recommends curl unzip && \
    rm -rf /var/lib/apt/lists/* && \
    chmod +x ./prepare_data.sh && \
    ./prepare_data.sh && \
    apt-get purge -y --auto-remove unzip

RUN pip3 install --no-cache-dir --upgrade pip setuptools wheel && \
    pip3 install --no-cache-dir numpy==1.26.0 && \
    pip3 install --no-cache-dir --no-build-isolation -r requirements.txt

COPY http-service/ ./http-service/

COPY --from=builder /app/dist/piper_tts-*.whl ./dist/
RUN pip3 install ./dist/piper_tts-*.whl

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
