#!/bin/bash
set -e

echo "--- Starting data and model preparation ---"

DATA_DIR="data/piper"
MODELS_DIR="${DATA_DIR}/models"
MANA_DIR="${MODELS_DIR}/fa/fa_IR/mana/medium"
EZAFE_ZIP="ezafe_model_quantized.zip"
EZAFE_DIR="ezafe_model_quantized"
HOMO_DIR="onnx-homo-ge2pe"
HOMO_ZIP="onnx-model.zip"
DICT_PATH="${DATA_DIR}/homograph_dictionary.parquet"

mkdir -p "${MANA_DIR}"

if [ ! -f "${DATA_DIR}/voices.json" ]; then
  echo "Downloading voices.json..."
  gdown 1PVauqxaGyCsCDthWMVRus0uDsus6D4U1 -O "${DATA_DIR}/voices.json"
else
  echo "voices.json already exists, skipping."
fi

if [ -z "$(ls -A ${MODELS_DIR} 2>/dev/null)" ]; then
  echo "Downloading piper voices..."
  gdown 13pxf4H0-2phQVUe_Cyvdvg5SKKaSL2tc -O piper_voices.zip
  unzip -q piper_voices.zip -d "${MODELS_DIR}/"
  rm piper_voices.zip
else
  echo "Piper models directory not empty, skipping."
fi

if [ ! -f "${MANA_DIR}/fa_IR-mana-medium.onnx" ]; then
  echo "Downloading and processing custom mana voice..."
  gdown 1t8pU3cE0qUWo0iRFfjsk4fWJCxcY0gwn -O mana.zip
  unzip -q mana.zip -d temp_mana_dir
  mv temp_mana_dir/checkpoint*/model.onnx "${MANA_DIR}/fa_IR-mana-medium.onnx"
  mv temp_mana_dir/checkpoint*/model.onnx.json "${MANA_DIR}/fa_IR-mana-medium.onnx.json"
  rm mana.zip
  rm -rf temp_mana_dir
else
  echo "Custom mana voice already exists, skipping."
fi

if [ ! -d "${EZAFE_DIR}" ]; then
  echo "Downloading and unzipping ezafe model..."
  gdown 1vdInn73cHsMCszktCqOT1zOtz0ukBwUp -O "${EZAFE_ZIP}"
  unzip -q "${EZAFE_ZIP}" -d ./
  mv ./"${EZAFE_DIR}"/model_quantized.onnx ./"${EZAFE_DIR}"/model.onnx
  rm "${EZAFE_ZIP}"
else
  echo "Ezafe model directory already exists, skipping."
fi

if [ ! -d "${HOMO_DIR}" ]; then
  echo "Downloading and unzipping homograph model..."
  gdown 1nKvWjm8Z6qSHKmFqLX1me0g5j-LjlhB4 -O "${HOMO_ZIP}"
  unzip -q "${HOMO_ZIP}" -d ./
  mv content/onnx-homo-ge2pe ./"${HOMO_DIR}"
  rm "${HOMO_ZIP}"
  rm -rf content
else
  echo "Homograph model directory already exists, skipping."
fi

if [ ! -f "${DICT_PATH}" ]; then
  echo "Downloading Homograph Dictionary Parquet..."
  curl -L "https://huggingface.co/datasets/MahtaFetrat/HomoRich-G2P-Persian/resolve/main/data/train-01.parquet" -o "${DICT_PATH}"
else
  echo "Homograph Dictionary already exists, skipping."
fi

echo "--- Data preparation complete ---"