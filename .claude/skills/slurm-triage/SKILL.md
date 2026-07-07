---
name: slurm-triage
description: Diagnose a failed or suspicious africa_llm Snellius run from its slurm-*.out log or results folder. Use when the user pastes job output, reports a crash or bad metrics, or asks why a run failed. Match against the known failure signatures BEFORE proposing any code change.
---

# Slurm run triage

Work through this in order. Most "new" failures here are re-occurrences of a known
signature — check the table first, and only debug fresh if nothing matches.

## 1. Match against known signatures

| Symptom in log/results | Root cause | Fix |
|---|---|---|
| PEFT warning "Found missing adapter keys" (huge list), outputs look like base Gemma-3, metrics near zero-shot | transformers module layout ≠ adapter's training layout; LoRA loaded as all-zeros | Run via the `$HOME/tf_infer` overlay + `python3.11 -S` (see snellius-handoff skill). Never "fix" by installing into shared `~/.local`. |
| Crash `mha_graph.execute ... false` (cuDNN) on H100 | SDPA + Gemma-3 head_dim=256 cuDNN bug | `attn_implementation="eager"` in the loader (already in `inference/jobs/inference_africa.py`; keep it in any new loader) |
| `eval_loss` tiny (~0.08) and `eval_mean_token_accuracy` ~0.98, but generations are garbage / JSON parse rate ~4% | LM loss averaged over the ~7k-token static codebook — no completion-only masking; answer gradient diluted ~30:1 | Completion-only collator masking everything up to `<start_of_turn>model\n` (token-ID template, `packing=False`). See `plans/gemma3_simple_sft_completion_only_loss_fix.md`. |
| `FileNotFoundError` for `africa_prompt_2004.txt` | Intentional guard — codebook not deployed | Deploy the codebook to `prompts/`; do NOT re-add a fallback to the numeric `africa_prompt_2602*` codebook |
| `TypeError` around SFTTrainer kwargs | TRL 0.18 API: needs `SFTConfig` + `processing_class` (not `tokenizer=`) | Match the pattern already in `agent_utils/gemma3_finetune_simple.py` |
| dtype crash during val/test generation | Fixed in commit 23349ac | Compare against that commit before re-deriving |
| Empty transcripts in training text / answers right-truncated | Codebook (~7k tok) + too-small `max_tokens` starved the transcript budget | Never truncate the codebook, only the transcript (`_truncate_transcript`, `max_text_tokens`); jobs set `max_tokens=12288`; the code raises if codebook+answer don't fit |

## 2. Extract the training curve

```bash
grep -aoE "\{'loss': [0-9.]+, 'grad_norm': [0-9.eE+-]+, 'learning_rate': [0-9.eE+-]+, 'epoch': [0-9.]+\}" slurm-<jobid>.out
```

Note: the LR sawtooths by design (per-epoch `trainer.train()` loop restarts the
linear schedule each epoch) — that is known and deferred, not a bug to fix.

## 3. Metrics sanity (results_logs/<timestamp>_<mtype>/)

- **JSON parse rate** (`parse_stats`) is the first health check — low parse rate
  means the run is broken regardless of loss.
- Judge per-target quality by `accuracy_applicable` (`acc_app`, gold ≠
  "not applicable"), not plain accuracy — gated fields are mostly N/A and plain
  accuracy is inflated by trivially predicting N/A.
- Several fields are severely class-imbalanced: prefer macro-F1 / per-class recall
  over accuracy when comparing runs.
- `multi_value` fields (`language`, `topic`, `resource_distribution_by_whom1`,
  `resource_distribution_for_whom1`) are scored as order-insensitive sets;
  `Other(<lang>)` counts as in-label for `language`.

## 4. Only then debug fresh

Follow the global debugging discipline: quote the decisive log line, check package
versions in the env that ran (the job logs `PYTHONPATH`/paths at the top), one
hypothesis at a time. Report findings to the user before changing code — fixes run
on Snellius, so a wrong guess costs a full job round-trip.
