import sys
print(sys.version)

import os
from pathlib import Path
from sklearn.model_selection import train_test_split

# --- make sure Python can find the repo root ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)  # one level up from jobs/
sys.path.insert(0, PROJECT_ROOT)

import agent_utils
from agent_utils.utils import train_validate
from agent_utils.africa_dataprep import load_3jun_training_df

# - test whether utils loaded
agent_utils.test_function()

CACHE_DIR = '/projects/prjs1308/huggingface/'

# Labels + transcripts come from the 3jun CSV (label-name format, embedded `text`).
# load_3jun_training_df does the id cast, drops no-text rows, E4 NOT CODED→null,
# builds targets_json, fills string-target allowed, and runs the codebook-conformance
# guard — returning the prepared df + the 2004 label-name TARGETS spec. Shared with
# gemma3_finetune_fulltrain.py so the two jobs' data-prep can't drift.
CSV_PATH = Path("/projects/prjs1308/africa_llm_data/AFRICA-TRAIN-DB-3jun2026.csv")
df, TARGETS = load_3jun_training_df(CSV_PATH)

# split
train_df, test_df = train_test_split(df, test_size=0.2, random_state=1, shuffle=True)

print(train_df[["text", "targets_json"]].head(2).to_string(index=False))
print(test_df[["text", "targets_json"]].head(2).to_string(index=False))

# System prompt = codebook (long, static → KV-cached at inference). User prompt = short template from file.
# utf-8-sig strips a leading BOM (U+FEFF) so it doesn't become an extra token and affect model behavior.
PROMPTS_DIR = Path("/projects/prjs1308/africa_llm_data/prompts")

system_prompt_path = PROMPTS_DIR / "africa_prompt_2004.txt"
inference_prompt_path = PROMPTS_DIR / "inference_prompt.txt"
if system_prompt_path.exists():
    system_prompt = system_prompt_path.read_text(encoding="utf-8-sig").strip()
    if inference_prompt_path.exists():
        prompt = inference_prompt_path.read_text(encoding="utf-8-sig").strip()
    else:
        prompt = "Social media post text:\n\n{}\n\nAnnotate this post according to the codebook and return a single JSON object only."
    print("Using system + user prompt split (KV prefix caching enabled).")
    print("System prompt length:", len(system_prompt))
    print("User prompt (from inference_prompt.txt):", prompt[:80], "...")
else:
    # The label-name 2004 codebook is required for a correct experiment. Fail loudly
    # rather than silently falling back to the numeric africa_prompt_2602.txt, which
    # would train a numeric codebook against label-name gold and revert the migration.
    raise FileNotFoundError(
        f"Required system prompt not found: {system_prompt_path}\n"
        "Deploy the 2004 label-name codebook there, e.g.\n"
        f"  cp codebooks/africa_prompt_2004.txt {system_prompt_path}"
    )

seeds = [42]
results_dir = '/projects/prjs1308/africa_llm_data/results/testing'
model_dir = '/projects/prjs1308/africa_llm_data/results/job_models'
batch_size = 1
# Must hold the FULL codebook (never truncated) + capped transcript + answer. The
# codebook system prompt is ~5–8k tokens; 12288 leaves headroom on an H100. The job
# raises if the codebook+answer ever wouldn't fit (it is never silently truncated).
max_tokens = 12288
max_text_tokens = 500   # cap the TRANSCRIPT only at this many tokens (codebook is never capped)
early_stopping = 2
epochs = 2
gemma_model = "4b"  # "4b" | "27b" for simple_gemma3

# ── Smoke test (set AFRICA_SMOKE_TEST=1) ──────────────────────────────
# Cheap end-to-end check of the inference fix: a tiny train/test subset and
# a single epoch, written to a separate results/smoketest folder so it can't
# be confused with a real run. Watch the val/test inference log for a high
# JSON parse rate and clean JSON output (no repetition, hit_max≈0). Sizes are
# env-overridable (AFRICA_SMOKE_N_TRAIN / AFRICA_SMOKE_N_TEST). Leave the flag
# unset for a normal full run.
if os.environ.get("AFRICA_SMOKE_TEST", "0") == "1":
    n_train_smoke = int(os.environ.get("AFRICA_SMOKE_N_TRAIN", "64"))
    n_test_smoke = int(os.environ.get("AFRICA_SMOKE_N_TEST", "16"))
    train_df = train_df.iloc[:n_train_smoke]
    test_df = test_df.iloc[:n_test_smoke]
    epochs = 1
    early_stopping = 1
    results_dir = '/projects/prjs1308/africa_llm_data/results/smoketest'
    print(f"[SMOKE TEST] train={len(train_df)} test={len(test_df)} epochs={epochs} "
          f"results_dir={results_dir}")

train_validate(
    mtype="simple_gemma3",      # <── key change
    train_df=train_df,
    test_df=test_df,
    text_col="text",            # column with transcript/post
    target_col="targets_json",  # not used by simple runner, but fine to keep
    prompt=prompt,              # must contain "{}" once for transcript insertion
    answer_col="targets_json",  # column with JSON answers
    train_val_seeds=seeds,
    val_size=0.2,
    results_folder=results_dir,
    model_dir=model_dir,
    batch_size=batch_size,
    max_tokens=max_tokens,
    max_new_tokens=500,
    cache_dir=CACHE_DIR,
    local_model=None,
    text_only_res=None,
    early_stopping_patience=early_stopping,
    epochs=epochs,
    learning_rates=[0.0001],
    gemma_model=gemma_model,    # "4b" | "27b"
    targets_spec=TARGETS,      # for inference: in-label / set scoring of semicolon-separated multiclass (e.g. topic)
    system_prompt=system_prompt,  # codebook in system role → KV prefix caching at inference
    max_text_tokens=max_text_tokens,  # transcript cap (codebook never truncated)
)