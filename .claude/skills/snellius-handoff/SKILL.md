---
name: snellius-handoff
description: Produce the exact copy-paste command block for running africa_llm work on Snellius (git pull, sbatch, screen, monitoring). Use whenever code is ready to run on Snellius, the user asks to launch/deploy/run a job, or a change needs remote verification. Claude must NEVER ssh to Snellius itself — this skill produces commands the USER runs.
---

# Snellius handoff

Claude cannot and must not SSH to Snellius (`fcool@snellius.surf.nl`) — local auth
fails and the user explicitly drives all remote steps. Your job ends at: clean local
commit + an exact copy-paste command block. The user runs it (in their terminal or
via the `! <cmd>` prefix in this session).

## Procedure

1. **Local state first.** `git status` must be clean and the work committed. Do not
   push unless the user asked; instead make "push" the first line of the handoff
   block so the user does it.
2. **Check deploy prerequisites** for the job being launched (see table below). If a
   required remote file might be missing, include the `ls` check in the block so the
   job doesn't fail 2 hours in.
3. **Smoke test before full runs.** For any new or changed inference/training path,
   the block must first submit a tiny slice and only then the real job:
   `sbatch inference/jobs/run_inference_africa.sbatch 0:4` (range syntax = index
   slice). Success = no PEFT "missing adapter keys" warning in the log + valid JSON
   output.
4. **Emit the block** in one fenced code section, annotated per line.

## Canonical block (adapt, don't reinvent)

```bash
# local mac
git push

# on Snellius (user runs)
cd /home/fcool/africa_llm && git pull

# training
sbatch jobs/run_gemma3_finetuned.sbatch          # simple_gemma3 fine-tune, H100, 120h
sbatch jobs/run_gemma3_finetune_fulltrain.sbatch # full-train variant
sbatch jobs/run_gemma3_smoketest.sbatch          # small smoke test

# inference (6-way split of the filtered CSV; or an explicit start:end range)
for i in 1 2 3 4 5 6; do sbatch inference/jobs/run_inference_africa.sbatch $i; done
sbatch inference/jobs/run_inference_africa.sbatch 0:4    # smoke slice first!

# monitor
squeue -u fcool
tail -f slurm-<jobid>.out
```

Long interactive work (not sbatch) goes in GNU Screen: `screen -S <name>`, detach
`Ctrl-A d`, resume `screen -r <name>`.

## Deploy prerequisites

| Job needs | Remote path |
|---|---|
| Training CSV | `/projects/prjs1308/africa_llm_data/AFRICA-TRAIN-DB-3jun2026.csv` |
| Label-name codebook | `/projects/prjs1308/africa_llm_data/prompts/africa_prompt_2004.txt` (jobs read this filename directly; missing → intentional `FileNotFoundError`) |
| Inference data | `/home/fcool/africa_llm/inference/data/inference_data_filtered.csv` |
| Adapter | `ADAPTER_DIR` in `inference/jobs/run_inference_africa.sbatch` (edit the static path there when switching adapters) |
| HF cache | `/projects/prjs1308/huggingface/` |
| transformers overlay | `$HOME/tf_infer` — build once: `pip install --target=$HOME/tf_infer --no-deps transformers==4.53.3`. The inference sbatch runs `PYTHONPATH="$HOME/tf_infer:$HOME/.local/lib/python3.11/site-packages" python3.11 -S ...`; the `-S` is essential (this HPC python force-prepends `~/.local` via sitecustomize, beating plain PYTHONPATH). NEVER install into the shared `~/.local`. |

## When switching to a new adapter

The overlay pin (4.53.3) matches the current adapter's training-time transformers.
For a NEW adapter: compare its `adapter_model.safetensors` key layout against the
env's transformers Gemma-3 module layout. `...model.language_model.layers.*` =
transformers ≥4.52; `...language_model.model.layers.*` (+ spurious `vision_tower`) =
older. Mismatch → the LoRA silently loads as zeros and you get base Gemma-3. Build a
matching overlay; verify with the 0:4 smoke slice.
