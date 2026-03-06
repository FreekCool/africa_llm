import sys
print(sys.version)

import os
import json
import torch
import random
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from sklearn.model_selection import train_test_split

# --- make sure Python can find the repo root ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)  # one level up from jobs/
sys.path.insert(0, PROJECT_ROOT)

import agent_utils
from agent_utils.utils import train_validate, build_multi_task_splits, rebalance_binary_to_fixed_n

# - test whether utils loaded
agent_utils.test_function()

CACHE_DIR = '/projects/prjs1308/huggingface/'

TOPIC01_ALLOWED = [
    "NO TOPIC",
    "ECONOMY",
    "CIVIL RIGHTS",
    "HEALTH",
    "AGRICULTURE",
    "LABOR",
    "EDUCATION",
    "ENVIRONMENT",
    "ENERGY",
    "IMMIGRATION",
    "TRANSPORTATION",
    "LAW AND CRIME",
    "SOCIAL WELFARE",
    "HOUSING",
    "DOMESTIC COMMERCE",
    "DEFENSE",
    "TECHNOLOGY",
    "FOREIGN TRADE",
    "INTERNATIONAL AFFAIRS",
    "GOVERNMENT OPERATIONS",
    "PUBLIC LANDS",
    "CULTURE",
    "ETHNICITY"
]

# Mapping from old topic numbers to new topic numbers (as strings)
topic_mapping = {
    0: 'NO TOPIC',
    1: 'ECONOMY',
    2: 'CIVIL RIGHTS',
    3: 'HEALTH',
    4: 'AGRICULTURE',
    5: 'LABOR',
    6: 'EDUCATION',
    7: 'ENVIRONMENT',
    8: 'ENERGY',
    9: 'IMMIGRATION',
    10: 'TRANSPORTATION',
    12: 'LAW AND CRIME',  # Law and Crime
    13: 'SOCIAL WELFARE',  # Social Welfare
    14: 'HOUSING',  # Housing
    15: 'DOMESTIC COMMERCE',  # Domestic Commerce
    16: 'DEFENSE',  # Defense
    17: 'TECHNOLOGY',  # Technology
    18: 'FOREIGN TRADE',  # Foreign Trade
    19: 'INTERNATIONAL AFFAIRS',  # International Affairs
    20: 'GOVERNMENT OPERATIONS',  # Government Operations
    21: 'PUBLIC LANDS',  # Public Lands
    23: 'CULTURE',
    24: 'ETHNICITY'   # Gun control
} # TOPIC MAPPING TO BE APPLIED TO DATAFRAME


json_path = Path("/projects/prjs1308/africa_llm_data/africa_jsons/african_videos.json")

with json_path.open("r", encoding="utf-8") as f:
    records = json.load(f)

print("N records:", len(records))
print(records[0])

df = pd.json_normalize(records)

# (keep your list/dict -> json-string cleanup if you want)
for col in df.columns:
    if df[col].apply(lambda x: isinstance(x, (list, dict))).any():
        df[col] = df[col].apply(lambda x: json.dumps(x, ensure_ascii=False) if isinstance(x, (list, dict)) else x)

# 1) define target cols: everything right of 'text'
text_idx = df.columns.get_loc("text")
target_cols = list(df.columns[text_idx + 1:])

# 2) drop rows where ALL targets are NaN (before split)
mask_all_nan = df[target_cols].isna().all(axis=1)
print("Dropping rows with all targets NaN:", int(mask_all_nan.sum()), "/", len(df))
df = df.loc[~mask_all_nan].reset_index(drop=True)

df["topic01"] = pd.to_numeric(df["topic01"], errors="coerce").astype("Int64").map(topic_mapping)

# 3) split
train_df, test_df = train_test_split(df, test_size=0.2, random_state=1, shuffle=True)

# 4) add targets_json to both (same target_cols list)
def add_targets_json(df_):
    df_["targets_json"] = df_.apply(
        lambda row: json.dumps(
            {c: (None if pd.isna(row[c]) else row[c]) for c in target_cols},
            ensure_ascii=False,
        ),
        axis=1,
    )

add_targets_json(train_df)
add_targets_json(test_df)

print(train_df[["text", "targets_json"]].head(2).to_string(index=False))
print(test_df[["text", "targets_json"]].head(2).to_string(index=False))

print(train_df.head())

# System prompt = codebook (long, static → KV-cached at inference). User prompt = short template from file.
# utf-8-sig strips a leading BOM (U+FEFF) so it doesn't become an extra token and affect model behavior.
PROMPTS_DIR = Path("/projects/prjs1308/africa_llm_data/prompts")

system_prompt_path = PROMPTS_DIR / "africa_prompt_system.txt"
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
    system_prompt = None
    # Fallback: single full prompt (no prefix caching)
    prompt_path = PROMPTS_DIR / "africa_prompt_2602.txt"
    prompt = prompt_path.read_text(encoding="utf-8-sig").strip()
    print("Using single prompt (no system_prompt file found).")
    print("Length:", len(prompt))

TARGETS = {
    # multiclass
    "language": {"type": "multiclass", "allowed": [1,2,3,4,5,6,7,8,99]},
    "resource_distribution_by_whom1": {"type": "multiclass", "allowed": [3,2,1,0,-1,99]},
    "resource_distribution_for_whom1": {"type": "multiclass", "allowed": [1,0,-1,99]},
    "climate_change": {"type": "multiclass", "allowed": [0,1,2,99]},
    "topic01": {"type": "multiclass", "allowed": TOPIC01_ALLOWED},
    "pro_us": {"type": "multiclass", "allowed": [1,2,3,99]},
    "pro_russia": {"type": "multiclass", "allowed": [1,2,3,99]},
    "pro_china": {"type": "multiclass", "allowed": [1,2,3,99]},
    "pro_un": {"type": "multiclass", "allowed": [1,2,3,99]},
    "pro_imf": {"type": "multiclass", "allowed": [1,2,3,99]},  # (or pro_mf if that's your column)
    "pro_democracy": {"type": "multiclass", "allowed": [1,2,3,99]},

    # binary (optionally allow 99 if your data uses it)
    "politics": {"type": "binary", "allowed": [0,1,99]},
    "domestic_politics": {"type": "binary", "allowed": [0,1,99]},
    "foreign_politics": {"type": "binary", "allowed": [0,1,99]},
    "resource_distribution": {"type": "binary", "allowed": [0,1,99]},
    "resource_distribution_for_gender": {"type": "binary", "allowed": [0,1,99]},
    "anti_western": {"type": "binary", "allowed": [0,1,99]},
    "national_unity": {"type": "binary", "allowed": [0,1,99]},
    "subgroup_unity": {"type": "binary", "allowed": [0,1,99]},
    "african_unity": {"type": "binary", "allowed": [0,1,99]},
    "political_opponents": {"type": "binary", "allowed": [0,1,99]},
    "religion": {"type": "binary", "allowed": [0,1,99]},

    # string fields (allowed filled from train+test+val in next cell)
    "resource_distribution_for_whom_ethnic1": {
        "type": "string",
        "allowed": [],
        "eval": {
            "metric": "exact",              # exact match after normalization
            "normalize": ["strip", "lower", "collapse_ws"],
            "empty_allowed": True,          # because it's only required when a condition holds
            "track_unique_incorrect": True,
            "max_unique_incorrect": 200,
        },
    },
    "resource_distribution_for_whom_region1": {
        "type": "string",
        "allowed": [],
        "eval": {
            "metric": "exact",
            "normalize": ["strip", "lower", "collapse_ws"],
            "empty_allowed": True,
            "track_unique_incorrect": True,
            "max_unique_incorrect": 200,
        },
    },
    "subgroup_unity_text": {
        "type": "string",
        "allowed": [],
        "eval": {
            "metric": "exact",
            "normalize": ["strip", "lower", "collapse_ws"],
            "empty_allowed": True,
            "track_unique_incorrect": True,
            "max_unique_incorrect": 500,
        },
    },
}

# Set string targets' allowed list from all annotated values in train + test (val is a subset of train)
STRING_TARGETS = [
    "resource_distribution_for_whom_ethnic1",
    "resource_distribution_for_whom_region1",
    "subgroup_unity_text",
]
combined = pd.concat([train_df, test_df], ignore_index=True)
for t in STRING_TARGETS:
    if t not in TARGETS or TARGETS[t].get("type") != "string":
        continue
    if t not in combined.columns:
        continue
    vals = combined[t].dropna().astype(str).str.strip()
    vals = vals[vals != ""]
    unique_vals = sorted(vals.unique().tolist())
    TARGETS[t]["allowed"] = unique_vals
    print(f"{t}: allowed = {len(unique_vals)} values")

seeds = [42]
results_dir = '/projects/prjs1308/africa_llm_data/results/testing'
model_dir = '/projects/prjs1308/africa_llm_data/results/job_models'
batch_size = 1
max_tokens = 4096
early_stopping = 5
epochs = 5
gemma_model = "4b"  # "4b" | "27b" for simple_gemma3

train_validate(
    mtype="simple_gemma3",      # <── key change
    train_df=train_df[:30],
    test_df=test_df[:10],
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
    targets_spec=TARGETS,      # for inference: normalize semicolon-separated multiclass (e.g. topic01)
    system_prompt=system_prompt,  # codebook in system role → KV prefix caching at inference
)