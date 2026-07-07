# CLAUDE.md — africa_llm

Research codebase: multi-target annotation of African political Facebook posts
(presidents/MPs) with fine-tuned Gemma-3. One post → one JSON object with **27
codebook fields** (label-name string values). QLoRA SFT + batched inference on the
**Snellius** HPC (SLURM, user `fcool`). The **live path is `mtype="simple_gemma3"`**
— slot-tokens, iLoRA, multimodal, and llama3 modules in `agent_utils/` are
legacy/alternative machinery the current jobs do not use.

## Operating rules (non-negotiable)

1. **The mac is edit-only.** No torch/trl locally; `import agent_utils` always fails
   (its `__init__` pulls torch). All training/inference/eval runs on Snellius. Never
   claim model/tokenizer/trainer code is "verified" from a local run.
2. **Never SSH to Snellius.** Local auth fails and the user drives all remote steps.
   Deliver exact copy-paste command blocks instead — use the `snellius-handoff`
   skill.
3. **Never modify the shared `~/.local` on Snellius.** Version pins are isolated
   overlays (`pip install --target=$HOME/tf_infer --no-deps ...` + `python3.11 -S`).
4. **Source-of-truth dataset:** `AFRICA-TRAIN-DB-3jun2026.csv` (2300 records —
   `wc -l` shows ~24k because quoted text fields contain newlines; trust pandas).
   Snellius: `/projects/prjs1308/africa_llm_data/`, local sample: `data_examples/`.
5. **Label-name convention everywhere.** The numeric codebook
   `codebooks/africa_prompt_2602*.txt` must never be read; jobs raise
   `FileNotFoundError` if `africa_prompt_2004.txt` is missing — never re-add a
   fallback.
6. **Never truncate the codebook — only the transcript** (`_truncate_transcript`,
   `max_text_tokens=500`, jobs set `max_tokens=12288`). Code raises if
   codebook+answer don't fit; keep it that way.
7. **Loud failure over fallback.** `assert_codebook_conformance` raises on
   out-of-codebook values → fix the data at source, never patch in code.
8. **Read `plans/` before working** — plans record confirmed root causes and
   resolved decisions (D1/D2/D3, B1–B3); don't re-litigate them.

## Stale artifacts — do not trust

- **`agent-train-validation.ipynb` is STALE**: numeric TARGETS, old column names
  (`topic01`, `resource_distribution_for_gender`), `african_videos.json` source,
  debug slices `[:30]/[:10]`, `max_tokens=4096`. Useful only to see the
  `train_validate` call shape. **`jobs/gemma3_finetune.py` is authoritative.**
- **`prepare_data.ipynb` is STALE** — superseded by
  `agent_utils/africa_dataprep.py:load_3jun_training_df()`.
- `reports/codebook_2004_migration_checklist.md` predates the 3jun switch; the plan
  in `plans/2004_codebook_new_csv_migration.md` supersedes it.
- `inference/check_inference_coverage.py` scans `.../inference_results` but the
  current sbatch writes to `.../inference_results_junev2` — update before trusting.

## The live pipeline

```
TRAIN  jobs/run_gemma3_finetuned.sbatch            (H100, 120h)
       jobs/run_gemma3_finetune_fulltrain.sbatch   (no val/test → inference adapter)
       jobs/run_gemma3_smoketest.sbatch            (45min, AFRICA_SMOKE_TEST=1)
         → jobs/gemma3_finetune.py | gemma3_finetune_fulltrain.py
         → africa_dataprep.load_3jun_training_df(CSV_PATH) → (df, TARGETS)
         → train_validate(mtype="simple_gemma3", ...)        [agent_utils/utils.py]
         → run_simple_gemma3(...)                  [agent_utils/gemma3_finetune_simple.py]

INFER  inference/jobs/run_inference_africa.sbatch <part 1..6 | start:end>
         → PYTHONPATH="$HOME/tf_infer:$HOME/.local/lib/python3.11/site-packages" \
           python3.11 -S inference/jobs/inference_africa.py ...
```

Current config (both training jobs): `gemma_model="4b"` (`google/gemma-3-4b-it`),
`lr=1e-4`, `batch_size=1`, `grad_accum=4`, `epochs=2`, `seed=42`, 4-bit nf4 QLoRA
`r=64, alpha=128` (scaling 2.0), TRL 0.18 `SFTConfig` + `processing_class`,
completion-only collator, manual per-epoch `trainer.train()` loop (LR sawtooth is
known and deferred — not a bug).

Snellius paths: repo `/home/fcool/africa_llm`, data+prompts+results under
`/projects/prjs1308/africa_llm_data/`, HF cache `/projects/prjs1308/huggingface/`.
System prompt read from `prompts/africa_prompt_2004.txt` with `encoding="utf-8-sig"`
(file has a BOM). Adapter dirs carry a `run_config.json` (model_id, prompts,
targets_spec, lengths) that inference reads back — keep it complete when changing
what training saves.

Output conventions: results folder `{run_id}_{mtype}_{4b|27b}/` with
`experiment_config.json` + per-epoch CSVs
`simple_gemma3_{model}_{val|test}_metrics_lr{lr}_seed{seed}_epoch{N}_{run_id}.csv`
(special rows `_timing`, `_json_parse`). Inference writes append-mode
`inference_predictions_{range}_{run_id}.csv` and resumes by skipping already-seen
ids (6-way partition happens before the skip, so parts stay disjoint).

## Repo map

```
agent_utils/
  utils.py                  train_validate router; insert_text_once; build_sft_dataset;
                            slot-token machinery (legacy path)
  africa_dataprep.py        LIVE data prep: load_3jun_training_df, build_targets_spec
                            (27-field TARGETS), assert_codebook_conformance
  gemma3_finetune_simple.py LIVE trainer: run_simple_gemma3, _build_chat_text_simple,
                            _truncate_transcript, run_simple_val_inference
  eval_utils.py, slot_trainer.py, ilora_utils.py, gemma3_finetune.py,
  gemma3_mm_finetune.py, gemma3_ilora_finetune.py, gemma3_zeroshot*.py,
  llama3_*.py, inference_vllm_gemma3.py        ← legacy/alternative, not on live path
jobs/                       training entry scripts + sbatch (authoritative config)
inference/jobs/             batch inference script + sbatch (tf_infer overlay)
codebooks/                  africa_prompt_2004.txt (LIVE) | africa_prompt_2602-2.txt (numeric, FORBIDDEN)
data_examples/              local inspection samples (3jun CSV etc.)
data_import/, results_logs/, inference/data/   git-ignored local data/outputs
plans/                      design docs with confirmed root causes — read first
```

mtype dispatch (`train_validate`): `llama3|gemma3` zero-shot,
`fine_tuned_llama3|fine_tuned_gemma3` slot-token LoRA, `ilora_*` continual
learning, `simple_gemma3` plain-JSON SFT (LIVE). `prompt` must contain exactly one
`{}`; text goes in via `insert_text_once`, **never `str.format`** (prompts contain
literal JSON braces).

## Known failure modes (already hit once — don't rediscover)

| Symptom | Cause | Fix |
|---|---|---|
| PEFT "Found missing adapter keys", outputs = base model | transformers layout ≠ adapter's training layout (≥4.52: `model.language_model.layers.*`) | `$HOME/tf_infer` overlay + `python3.11 -S` (`-S` is essential: sitecustomize force-prepends `~/.local`, beating PYTHONPATH) |
| cuDNN crash `mha_graph.execute` on H100 | SDPA + Gemma-3 head_dim=256 | `attn_implementation="eager"` in every loader |
| eval_loss ~0.08, token acc ~0.98, garbage output, ~4% parse rate | loss averaged over ~7k-token static codebook (no completion masking) | `DataCollatorForCompletionOnlyLM` on token IDs of `<start_of_turn>model\n`, `packing=False`; verify supervised fraction ≈3–5% |
| dtype crash in generation | 4-bit + bf16 KV cache vs fp32 query | generate under `torch.autocast(dtype=compute_dtype)`; never `.to(device)` a bnb model |
| `TypeError` building SFTTrainer | TRL 0.18 API | `SFTConfig(dataset_text_field=..., max_seq_length=..., packing=...)` + `SFTTrainer(processing_class=tokenizer)` |
| Empty transcripts / truncated answers in training text | codebook starved the token budget | rule 6 above (B1 fix) |

Full triage procedure + log-grep patterns: `slurm-triage` skill. Big slurm logs and
run folders → delegate to the `results-analyst` agent instead of reading them in
the main conversation. Diffs touching `agent_utils/`, `jobs/`, or `inference/` →
run the `africa-reviewer` agent before committing.

## Data & targets

- 2300 records, `id` + `text` + 27 label columns, values are label-name strings,
  `;`-joined for multi-value. E4: `national_unity_narrow == "NOT CODED"` → null
  (1589 rows). 4 duplicate ids and 8 blank-topic rows are kept by decision D3.
- `TARGETS` spec lives in `africa_dataprep.build_targets_spec()`: per target
  `{"type": "multiclass"|"string", "allowed": [...], "multi_value"?, "allow_other_paren"?, "eval"?}`.
  String targets (`resource_distribution_for_whom_ethnic1`, `..._region1`,
  `subgroup_unity_text`) get `allowed` filled from data at runtime (8/351/385).
- Scoring: `multi_value` fields (`language`, `topic`,
  `resource_distribution_by_whom1`, `resource_distribution_for_whom1`) are
  order-insensitive sets; `Other(<x>)` in-label for `language`; judge quality by
  `accuracy_applicable` (gold ≠ "not applicable"), not plain accuracy; several
  fields are class-imbalanced → prefer macro-F1. Absent-but-allowed labels stay in
  `allowed` (the model may legitimately predict them).
- Train caps transcripts at 500 tokens; inference passes them uncapped —
  intentional asymmetry.

## Verification (before any Snellius handoff)

- **Data-prep change** → local dry-run gates (`dataprep-dryrun` skill): 2300 rows /
  0 dropped, 27 keys, 0 numeric, 0 `NOT CODED`, 1589 nulls, conformance passes.
- **Training-path change** → smoketest sbatch first (`AFRICA_SMOKE_TEST=1`, 45min);
  check the `[verify-mask]` supervised fraction (~3–5%) in the log.
- **Inference/adapter change** → `--range 0:4` slice first; success = no
  missing-adapter-keys warning + valid JSON rows.
- New adapter → compare its `adapter_model.safetensors` key layout to the env's
  transformers before launching six 48h jobs.

## Environment & conventions

- Snellius runtime: `python3.11` + shared `~/.local` site-packages (+ `tf_infer`
  overlay for inference). **No requirements manifest exists** — versions live in
  comments (TRL 0.18, transformers 4.53.3 overlay); check installed versions before
  blaming code. Local: conda env `vve_nxt` (pandas-level work only).
- Debug env vars: `AFRICA_DEBUG` (default on), `AFRICA_DEBUG_IDX`,
  `AFRICA_DEBUG_CHARS`; smoketest: `AFRICA_SMOKE_TEST`, `AFRICA_SMOKE_N_TRAIN`,
  `AFRICA_SMOKE_N_TEST`.
- Notebook outputs are stripped on commit (`nbstripout` via `.gitattributes`).
- Commit after each completed task (never push unless asked); user commits directly
  to `main`.
