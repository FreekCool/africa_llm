#!/bin/sh

# SBATCH resources – mirror your fine-tuning jobs, adjust as needed
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --partition=gpu_h100
#SBATCH --time=48:00:00

# Paths (edit these if you move things)
DATA_PATH="/home/fcool/africa_llm/inference/results/test_results/inference_data.csv"
ADAPTER_DIR="/projects/prjs1308/africa_llm_data/results/inference_models/20260308_205449_simple_gemma3/simple_gemma3_4b_lr0.0001_seed42_20260308_205449"
OUTPUT_DIR="/home/fcool/africa_llm/inference/results/test_results"
CACHE_DIR="/projects/prjs1308/huggingface/"

# Range selection:
# - To use an explicit index range, set RANGE="start:end", e.g. RANGE="0:1000"
# - To use a quarter of the dataframe, leave RANGE empty and set QUARTER=1/2/3/4
RANGE="${RANGE:-}"      # can be overridden via sbatch --export=RANGE=0:1000 or env
QUARTER="${QUARTER:-1}" # default to first quarter if RANGE is empty

echo "DATA_PATH   = ${DATA_PATH}"
echo "ADAPTER_DIR = ${ADAPTER_DIR}"
echo "OUTPUT_DIR  = ${OUTPUT_DIR}"
echo "CACHE_DIR   = ${CACHE_DIR}"
echo "RANGE       = ${RANGE}"
echo "QUARTER     = ${QUARTER}"

cd /home/fcool/africa_llm || exit 1

if [ -n "${RANGE}" ]; then
  echo "Running inference_africa.py on explicit range ${RANGE}"
  python3.11 inference/jobs/inference_africa.py \
    --data-path "${DATA_PATH}" \
    --adapter-dir "${ADAPTER_DIR}" \
    --cache-dir "${CACHE_DIR}" \
    --range "${RANGE}" \
    --output-dir "${OUTPUT_DIR}"
else
  echo "Running inference_africa.py on quarter ${QUARTER}"
  python3.11 inference/jobs/inference_africa.py \
    --data-path "${DATA_PATH}" \
    --adapter-dir "${ADAPTER_DIR}" \
    --cache-dir "${CACHE_DIR}" \
    --quarter "${QUARTER}" \
    --output-dir "${OUTPUT_DIR}"
fi

