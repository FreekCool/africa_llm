#!/bin/sh

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1
#SBATCH --partition=gpu_h100
#SBATCH --time=00:45:00

# Cheap smoke test of the simple_gemma3 inference fix (tiny subset, 1 epoch).
# Writes to results/smoketest. Override sizes with AFRICA_SMOKE_N_TRAIN / AFRICA_SMOKE_N_TEST.
export AFRICA_SMOKE_TEST=1
python3.11 /home/fcool/africa_llm/jobs/gemma3_finetune.py
