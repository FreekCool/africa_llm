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
- [x] Launch `jobs/run_gemma3_finetuned.sbatch` on Snellius — run_id `20260913_165248`
  (trainable params 466,010,112; `[verify-mask]` 4.51%, same as 4b)
- [x] Epoch 0 in. Measured: training 49,469 s (13.7 h — 47% over projection),
  val gen 18,563 s (50.4 s/prompt), test gen 22,917 s (49.8 s/prompt), parse 100%.
  25.3 h/epoch → ~76 h for 3 epochs. Test mean over 27 targets: acc_app 0.809,
  macro-F1 0.571 (4b ep0: 0.739–0.746 / 0.481–0.494; 4b ep1: 0.796–0.815 / 0.576–0.589).
- [x] **Run `20260913_165248` (job 26642070) TIMEOUT at 48:00:00** on 2026-09-15 16:52,
  ~3.9 h into epoch-1 test generation. Root cause: uncommitted Snellius-side edit
  `--time=120:00:00` → `48:00:00` in `jobs/run_gemma3_finetuned.sbatch` (repo value
  never changed). Salvaged: epoch-0 val+test CSVs, epoch-1 val CSV; only the epoch-0
  adapter is saved (no resume path). Guard added: slurm logs gitignored, handoff
  requires empty `git status` on Snellius + `scontrol show job | grep TimeLimit`.
- [x] Relaunched 2026-09-16 as job 26793414 (gcn147) after cleaning the Snellius tree;
  verified `TimeLimit=5-00:00:00`, partition MaxTime 5-00:00:00. Expect ~76 h.
  run_id: TBD (`grep "All outputs" slurm-26793414.out`)
- [ ] Compare per-epoch val/test metrics vs `20260610_174324` (acc_applicable, macro-F1)
- [ ] Pick epoch count for the 27b fulltrain (`gemma3_finetune_fulltrain.py`)
