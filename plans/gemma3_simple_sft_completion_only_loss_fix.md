# Fix Gemma-3 simple SFT: train loss only on the answer (completion-only masking)

## Context

The `simple_gemma3` fine-tune (`jobs/run_gemma3_finetuned.sbatch` →
`jobs/gemma3_finetune.py` → `train_validate(mtype="simple_gemma3")` →
`run_simple_gemma3`) produced near-zero accuracy on most of the 27 targets.
`results_logs/` shows ~4% JSON parse rate and garbage / empty / instruction-echo
generations, **despite** a healthy-looking `eval_loss=0.077` and
`eval_mean_token_accuracy=0.983`.

**Root cause (confirmed, not hypothesis).** The trainer is built with
`SFTConfig(dataset_text_field="text", packing=False)` + `SFTTrainer(...)` with
**no loss masking** (no `data_collator`, no `completion_only_loss`, no
`assistant_only_loss`) — see `agent_utils/gemma3_finetune_simple.py:1453-1487`.
The single `text` field (built in `build_simple_sft_dataset:442-500` via
`_build_chat_text_simple:73-103`) is the *entire* chat sequence: a **static
~7k-token codebook** in the system role + capped transcript + a ~150-300 token
answer JSON. So the LM loss is averaged over ~7-9k tokens that are ~97% the
static codebook the model already predicts trivially. The answer JSON — the only
thing we care about — is ~3% of tokens, so its gradient is diluted ~30:1 and is
never effectively learned. `eval_mean_token_accuracy=0.983` is the tell: the model
nails the static prefix and fails the answer. Result: low loss, garbage generation.

**Ruled out** (so we don't chase them): slot tokens (this is the plain-JSON
`simple_gemma3` path, no slots); "only 1 epoch" (`train_loss` drops 0.64→0.28
across the manual epoch loop — it does train multiple epochs); "dataset too small"
(val=368, test=460 ⇒ ~1,472 train examples); the KV-prefix-cache inference path
(it correctly reconstructs the training sequence with correct `position_ids` and
`add_generation_prompt=True`).

**Intended outcome.** Compute the SFT loss only on the assistant answer tokens, so
the model actually learns to emit the JSON. Expected: JSON parse rate jumps well
above 4%, per-target accuracy rises from ~0, and `eval_loss` becomes a *higher but
meaningful* answer-only number that decreases as the model learns.

## Scope

Masking fix **+** LoRA scaling fix (`α=128`). Both land in one run.
LR-schedule sawtooth is documented and **deferred** (not part of this change).

## Changes — single file: `agent_utils/gemma3_finetune_simple.py`

### 1. Completion-only loss masking (primary, root-cause fix)

Mask every label up to and including the Gemma assistant-turn template
`<start_of_turn>model\n`, so loss is computed only on the answer JSON + its
trailing `<end_of_turn>`. This operates on the **existing** `text` field — it
preserves the already-correct chat construction and `<end_of_turn>` termination
and is purely additive. `packing` must stay `False` (it already is).

- **Line 35** — extend the import:
  `from trl import SFTTrainer, SFTConfig` → add `, DataCollatorForCompletionOnlyLM`.
- **After the `SFTConfig(...)` block (after line 1478), before `trainer = SFTTrainer(`** —
  build the collator using **token IDs** (not the raw string, to avoid Gemma's
  standalone-vs-in-context tokenization mismatch):
  ```python
  response_template_ids = tokenizer.encode("<start_of_turn>model\n", add_special_tokens=False)
  completion_collator = DataCollatorForCompletionOnlyLM(
      response_template=response_template_ids, tokenizer=tokenizer)
  ```
- **`SFTTrainer(...)` (lines 1481-1487)** — add `data_collator=completion_collator`.

**Fallback** (only if `DataCollatorForCompletionOnlyLM` is missing/misbehaves in
this TRL 0.18 build): a tiny module-level collator subclassing
`transformers.DataCollatorForLanguageModeling(mlm=False)` that finds the
`response_template_ids` subsequence in each `input_ids` and sets
`labels[:end] = -100` (and `labels[:] = -100` if the template isn't found). Same
semantics, zero TRL-internal dependency. This is a correct fix, not a workaround.

### 2. LoRA scaling (secondary, research-backed)

`LoraConfig` at **lines 1292-1302** currently `r=64, lora_alpha=16` ⇒ scaling
`α/r = 0.25`, which structurally under-drives the adapter (every update divided by
4). Change **line 1293**: `lora_alpha=16` → `lora_alpha=128` ⇒ scaling `2.0`
(`α = 2r`, the standard QLoRA setting). Keep `r=64`, `lora_dropout=0.1`. If the
now-meaningful train loss diverges, fall back to `lora_alpha=64` (scaling 1.0).

### 3. LR-schedule sawtooth — DEFERRED (documented only)

`for ep in range(epochs): trainer.train()` with `num_train_epochs=1` (line 1455)
+ `lr_scheduler_type="linear"` restarts warmup+decay each epoch (sawtooth). It is
functionally correct, just suboptimal. Smoothing it properly requires moving the
per-epoch `run_simple_val_inference`/metrics/early-stop block into a
`TrainerCallback.on_epoch_end` and calling `trainer.train()` once — a real refactor
of lines 1494-1608. **Out of scope for this bugfix.**

## Verification (run on Snellius — cannot run locally, no torch/trl on the mac)

**A. Prove the mask before training.** One-shot debug block right after the
collator is built: take 1-2 examples from `dataset`, tokenize as the trainer will
(`add_special_tokens=False`, `text` already carries `<bos>`), run them through
`completion_collator`, and print the supervised-label fraction + decoded supervised
tokens.
- PASS: supervised fraction ≈ **3-5%** (only the answer, not ~100%); decoded
  supervised text is exactly the answer JSON + `<end_of_turn>` — no codebook, no
  transcript, no `<start_of_turn>model` prefix.
- FAIL signatures: ~100% ⇒ template IDs not matched; 0% ⇒ template not found
  (check `response_template_ids` and that `packing=False`).

**B. Watch the next full training run** (old metrics were the bug, so they should
*change*):
- `eval_loss` should be **higher** than 0.077 and actually move (now answer-only).
- `eval_mean_token_accuracy` drops from the misleading 0.983 to a real number,
  then climbs.
- JSON parse rate (`parse_stats` in `run_simple_val_inference`) jumps well above 4%.
- Per-target accuracy across the 27 targets rises from ~0.
- Generation still terminates (the `<end_of_turn>` stays in the supervised labels).

## Notes
- All edits confined to `agent_utils/gemma3_finetune_simple.py`.
- Commit after the change (per workflow); run the job on Snellius via the existing
  `jobs/run_gemma3_finetuned.sbatch` under GNU Screen.
