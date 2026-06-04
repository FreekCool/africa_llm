"""
Full-train variant: finetune on the entire dataset with no validation or test split.
Use this to produce a single saved model trained on all data for the given number of epochs.
"""
import sys
print(sys.version)

import os
from pathlib import Path

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

# Same 3jun data-prep + 2004 TARGETS as gemma3_finetune.py (shared helper). Full-train
# uses the entire dataset as train with no validation/test split.
CSV_PATH = Path("/projects/prjs1308/africa_llm_data/AFRICA-TRAIN-DB-3jun2026.csv")
df, TARGETS = load_3jun_training_df(CSV_PATH)

train_df = df.copy()
test_df = df.iloc[0:0].copy()   # empty test set (same columns, zero rows)
print("Train rows:", len(train_df))
print("Test rows:", len(test_df))

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
model_dir = '/projects/prjs1308/africa_llm_data/results/inference_models'
batch_size = 1
# Must hold the FULL codebook (never truncated) + capped transcript + answer. The
# codebook system prompt is ~5–8k tokens; 12288 leaves headroom on an H100. The job
# raises if the codebook+answer ever wouldn't fit (it is never silently truncated).
max_tokens = 12288
max_text_tokens = 500   # cap the TRANSCRIPT only at this many tokens (codebook is never capped)
early_stopping = 5
epochs = 5
gemma_model = "4b"

# val_size=0 → no validation split (train on full data); test_df is already empty
train_validate(
    mtype="simple_gemma3",
    train_df=train_df,
    test_df=test_df,
    text_col="text",
    target_col="targets_json",
    prompt=prompt,
    answer_col="targets_json",
    train_val_seeds=seeds,
    val_size=0,   # no validation set — full dataset used for training
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
    gemma_model=gemma_model,
    targets_spec=TARGETS,
    system_prompt=system_prompt,
    max_text_tokens=max_text_tokens,  # transcript cap (codebook never truncated)
)
