# 27b fine-tune on the current pipeline (2026-09-13)

**Goal:** first `google/gemma-3-27b-it` run on the live `simple_gemma3` path
(2004 label-name codebook, 3jun CSV, completion-only masking). Same recipe as the
4b baseline `20260610_174324` (lr 1e-4, seed 42, r=64/α=128, 1840 train / 460 test,
val_size 0.2); only model size and epoch count differ. Decides how many epochs a
27b fulltrain (inference adapter) should use.

## Why 27b

The only earlier 27b runs (March, `slurm-20561603` vs 4b `slurm-20561539`) were a
clean single-variable comparison but on the OLD pipeline (numeric codebook, no loss
masking, n≈120 test). 27b won 14/22 targets on test acc and macro-F1 (n-weighted F1
0.706 vs 0.642) and led at every epoch, but both failed the free-string targets
(`resource_distribution_for_whom_region1`, `subgroup_unity_text`: in_label 0–18%).
Not comparable to June numbers — hence this run.

## Decision: 3 epochs, full val+test generation every epoch

Budget, 4b measured (10 Jun) × March-measured 27b/4b ratios (3.6× train,
2.2× per-prompt generation):

| phase / epoch | 4b measured | 27b projected |
|---|---|---|
| training | 9,320 s | 33,700 s |
| val gen (368) | 10,065 s | 22,100 s |
| test gen (460) | 12,332 s | 27,100 s |
| **per epoch** | **8.8 h** | **23.0 h** |

5 epochs ≈ 115 h against the 120 h wall — rejected. 3 epochs ≈ 69 h. Chosen over
"5 epochs, test only at the end" to keep the per-epoch test curve comparable to the
4b baseline. March 27b's best test F1 was at epoch 3.

`early_stopping = 2` is a no-op at 3 epochs; left unchanged. Smoketest skipped by
decision — first-epoch memory is the watch item (sequences ~6.9k tokens vs 4096 in
March; gradient checkpointing is on via `prepare_model_for_kbit_training`).

## Status

- [x] `jobs/gemma3_finetune.py`: `gemma_model="27b"`, `epochs=3`
- [ ] Launch `jobs/run_gemma3_finetuned.sbatch` on Snellius
- [ ] Compare per-epoch val/test metrics vs `20260610_174324` (acc_applicable, macro-F1)
- [ ] Pick epoch count for the 27b fulltrain (`gemma3_finetune_fulltrain.py`)
