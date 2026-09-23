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
- [x] Run `20260916_153320` finished all 3 epochs (job 26793414, 75.7 h, parse rate
  99.6–100%). Per-epoch TEST mean acc_app 0.809 / **0.841** / 0.823, macro-F1
  0.582 / 0.598 / 0.595; VAL acc_app 0.802 / **0.840** / 0.826. Both splits pick
  epoch 1.
- [x] Compared vs 4b `20260610_174324`. Best-vs-best (both epoch 1): test acc_app
  0.841 vs 0.815, val 0.840 vs 0.797. At equal epoch 0 the gap is larger (test
  acc_app 0.809 vs 0.746). Rule of thumb: 27b after 1 epoch ≈ 4b after 2, then
  +0.026 on top, for 5.3× train / 1.9× inference cost. Caveats: the 4b baseline
  never ran epoch 2 and was still improving; `national_unity_narrow`'s +0.223 is
  n=52 vs 10 — the real finding there is field coverage (27b emits the field
  15→52→72 rows/epoch, 4b never does: 9→10). March free-string failure is gone
  (in_label 87–93% vs 0–18%).
- [x] **Epoch count for the 27b fulltrain: 2** (D4). Epoch 2 of the dev run
  regressed six gated fields on BOTH splits (`subgroup_unity_text` −0.153,
  `climate_change` −0.097, `resource_distribution_{by_whom1,for_whom1}` −0.089,
  `_gender` −0.081, `_for_whom_region1` −0.077) while their plain accuracy moved
  +0.002 — N/A drift, and those fields are where 27b's advantage over 4b lives.
  Decided per-example, not per-epoch: the best dev checkpoint had seen each row
  twice (1472 × 2), so 2 fulltrain epochs (2300 × 2) matches that repetition count
  while giving 56% more steps from unique data. Counter-argument on record: on the
  7 targets where macro-F1 is measurable (rarest class ≥5 expected test rows) epoch
  2 wins (test 0.687 vs 0.648, val 0.794 vs 0.666) by starting to predict `unclear`
  on `religion`/`african_unity`/`subgroup_unity`/`national_unity` — a trade against
  conditional-field accuracy, not a free gain.
- Note: the fulltrain run folder under `results/testing/` will contain ONLY
  `experiment_config.json` — no val/test generation runs, so no metrics CSVs are
  written. That is expected, not a failed run; the only in-run quality signal is
  the `[verify-mask]` fraction (~4.5%). Adapter → `results/inference_models/`,
  overwritten each epoch, so only the epoch-1 (final) weights survive.
- [ ] Launch the 27b fulltrain (`jobs/run_gemma3_finetune_fulltrain.sbatch`,
  `gemma_model="27b"`, `epochs=2`, ~43 h) → inference adapter
- [ ] Point `inference/jobs/run_inference_africa.sbatch` ADAPTER_DIR at the new
  27b adapter and re-verify with a `0:4` slice (transformers overlay layout check)
