#!/bin/sh

# SBATCH resources – mirror your fine-tuning jobs, adjust as needed
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --partition=gpu_h100
#SBATCH --time=48:00:00

# Static paths (edit these if you move things)
DATA_PATH="/home/fcool/africa_llm/inference/data/inference_data_filtered.csv"
ADAPTER_DIR="/projects/prjs1308/africa_llm_data/results/inference_models/20260608_121714_simple_gemma3/simple_gemma3_4b_lr0.0001_seed42_20260608_121714"
OUTPUT_DIR="/projects/prjs1308/africa_llm_data/inference_results_junev2"
CACHE_DIR="/projects/prjs1308/huggingface/"

# Quarter (1-4) comes from the first sbatch argument, e.g.
#   sbatch run_inference_africa.sbatch 2
# Defaults to 1 if no argument is given.
QUARTER="${1:-1}"

echo "DATA_PATH   = ${DATA_PATH}"
echo "ADAPTER_DIR = ${ADAPTER_DIR}"
echo "OUTPUT_DIR  = ${OUTPUT_DIR}"
echo "CACHE_DIR   = ${CACHE_DIR}"
echo "QUARTER     = ${QUARTER}"

cd /home/fcool/africa_llm || exit 1

# Isolated transformers overlay. This adapter (trained 2026-03-08) needs the
# transformers >=4.52 module layout (model.language_model.layers), but the shared
# ~/.local regressed to 4.50.3 — which leaves the LoRA unbound (all keys "missing").
# We pin 4.53.3 in $HOME/tf_infer and run with -S so the overlay wins the import
# path while torch/peft/bnb still load from ~/.local — without modifying the
# shared env. Build once: pip install --target=$HOME/tf_infer --no-deps transformers==4.53.3
PYTHONPATH="$HOME/tf_infer:$HOME/.local/lib/python3.11/site-packages" python3.11 -S inference/jobs/inference_africa.py \
  --data-path "${DATA_PATH}" \
  --adapter-dir "${ADAPTER_DIR}" \
  --cache-dir "${CACHE_DIR}" \
  --quarter "${QUARTER}" \
  --output-dir "${OUTPUT_DIR}"

