#!/bin/sh

# SBATCH resources – mirror your fine-tuning / inference jobs
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --partition=gpu_h100
#SBATCH --time=48:00:00

# Diagnostic: run the SAME adapter the production inference job uses over the FULL
# training set, with training-faithful preprocessing (500-token transcript cap),
# and score against gold. One job over all ~2300 rows (metrics are accumulated
# globally, so there is no partitioning here).

# Static paths (edit these if you move things)
CSV_PATH="/projects/prjs1308/africa_llm_data/AFRICA-TRAIN-DB-3jun2026.csv"
ADAPTER_DIR="/projects/prjs1308/africa_llm_data/results/inference_models/20260608_121714_simple_gemma3/simple_gemma3_4b_lr0.0001_seed42_20260608_121714"
OUTPUT_DIR="/projects/prjs1308/africa_llm_data/train_eval_results"
CACHE_DIR="/projects/prjs1308/huggingface/"

# Optional first argument = smoke-slice size, forwarded as --limit N. With no
# argument the full training set is scored, e.g.
#   sbatch run_inference_train_eval.sbatch 20     # smoke: first 20 rows
#   sbatch run_inference_train_eval.sbatch        # full run
if [ -n "${1}" ]; then
  LIMIT_ARGS="--limit ${1}"
else
  LIMIT_ARGS=""
fi

echo "CSV_PATH    = ${CSV_PATH}"
echo "ADAPTER_DIR = ${ADAPTER_DIR}"
echo "OUTPUT_DIR  = ${OUTPUT_DIR}"
echo "CACHE_DIR   = ${CACHE_DIR}"
echo "LIMIT       = ${LIMIT_ARGS}"

cd /home/fcool/africa_llm || exit 1

# Isolated transformers overlay. The adapter needs the transformers >=4.52 module
# layout (model.language_model.layers), but the shared ~/.local regressed to
# 4.50.3 — which leaves the LoRA unbound (all keys "missing"). We pin 4.53.3 in
# $HOME/tf_infer and run with -S so the overlay wins the import path while
# torch/peft/bnb still load from ~/.local — without modifying the shared env.
# Build once: pip install --target=$HOME/tf_infer --no-deps transformers==4.53.3
PYTHONPATH="$HOME/tf_infer:$HOME/.local/lib/python3.11/site-packages" python3.11 -S inference/jobs/inference_train_eval.py \
  --csv-path "${CSV_PATH}" \
  --adapter-dir "${ADAPTER_DIR}" \
  --cache-dir "${CACHE_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  ${LIMIT_ARGS}
