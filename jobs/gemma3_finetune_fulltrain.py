"""
Full-train variant: finetune on the entire dataset with no validation or test split.
Use this to produce a single saved model trained on all data for the given number of epochs.
"""
import sys
print(sys.version)

import os
import json
import torch
import random
import pandas as pd
from tqdm import tqdm
from pathlib import Path

# --- make sure Python can find the repo root ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)  # one level up from jobs/
sys.path.insert(0, PROJECT_ROOT)

import agent_utils
from agent_utils.utils import train_validate

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
    12: 'LAW AND CRIME',
    13: 'SOCIAL WELFARE',
    14: 'HOUSING',
    15: 'DOMESTIC COMMERCE',
    16: 'DEFENSE',
    17: 'TECHNOLOGY',
    18: 'FOREIGN TRADE',
    19: 'INTERNATIONAL AFFAIRS',
    20: 'GOVERNMENT OPERATIONS',
    21: 'PUBLIC LANDS',
    23: 'CULTURE',
    24: 'ETHNICITY'
}

json_path = Path("/projects/prjs1308/africa_llm_data/africa_jsons/african_videos.json")

with json_path.open("r", encoding="utf-8") as f:
    records = json.load(f)

print("N records:", len(records))

df = pd.json_normalize(records)

for col in df.columns:
    if df[col].apply(lambda x: isinstance(x, (list, dict))).any():
        df[col] = df[col].apply(lambda x: json.dumps(x, ensure_ascii=False) if isinstance(x, (list, dict)) else x)

text_idx = df.columns.get_loc("text")
target_cols = list(df.columns[text_idx + 1:])

mask_all_nan = df[target_cols].isna().all(axis=1)
print("Dropping rows with all targets NaN:", int(mask_all_nan.sum()), "/", len(df))
df = df.loc[~mask_all_nan].reset_index(drop=True)

df["topic01"] = pd.to_numeric(df["topic01"], errors="coerce").astype("Int64").map(topic_mapping)

# Full train: no train/test split — use entire df as train
train_df = df.copy()

def add_targets_json(df_):
    df_["targets_json"] = df_.apply(
        lambda row: json.dumps(
            {c: (None if pd.isna(row[c]) else row[c]) for c in target_cols},
            ensure_ascii=False,
        ),
        axis=1,
    )

add_targets_json(train_df)

# Empty test set (same columns, zero rows)
test_df = train_df.iloc[0:0].copy()

print("Train rows:", len(train_df))
print("Test rows:", len(test_df))

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
    prompt_path = PROMPTS_DIR / "africa_prompt_2602.txt"
    prompt = prompt_path.read_text(encoding="utf-8-sig").strip()
    print("Using single prompt (no system_prompt file found).")
    print("Length:", len(prompt))

TARGETS = {
    "language": {"type": "multiclass", "allowed": [1,2,3,4,5,6,7,8,99]},
    "resource_distribution_by_whom1": {"type": "multiclass", "allowed": [3,2,1,0,-1,99]},
    "resource_distribution_for_whom1": {"type": "multiclass", "allowed": [1,0,-1,99]},
    "climate_change": {"type": "multiclass", "allowed": [0,1,2,99]},
    "topic01": {"type": "multiclass", "allowed": TOPIC01_ALLOWED},
    "pro_us": {"type": "multiclass", "allowed": [1,2,3,99]},
    "pro_russia": {"type": "multiclass", "allowed": [1,2,3,99]},
    "pro_china": {"type": "multiclass", "allowed": [1,2,3,99]},
    "pro_un": {"type": "multiclass", "allowed": [1,2,3,99]},
    "pro_imf": {"type": "multiclass", "allowed": [1,2,3,99]},
    "pro_democracy": {"type": "multiclass", "allowed": [1,2,3,99]},
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
    "resource_distribution_for_whom_ethnic1": {
        "type": "string",
        "allowed": [],
        "eval": {"metric": "exact", "normalize": ["strip", "lower", "collapse_ws"], "empty_allowed": True, "track_unique_incorrect": True, "max_unique_incorrect": 200},
    },
    "resource_distribution_for_whom_region1": {
        "type": "string",
        "allowed": [],
        "eval": {"metric": "exact", "normalize": ["strip", "lower", "collapse_ws"], "empty_allowed": True, "track_unique_incorrect": True, "max_unique_incorrect": 200},
    },
    "subgroup_unity_text": {
        "type": "string",
        "allowed": [],
        "eval": {"metric": "exact", "normalize": ["strip", "lower", "collapse_ws"], "empty_allowed": True, "track_unique_incorrect": True, "max_unique_incorrect": 500},
    },
}

STRING_TARGETS = [
    "resource_distribution_for_whom_ethnic1",
    "resource_distribution_for_whom_region1",
    "subgroup_unity_text",
]
for t in STRING_TARGETS:
    if t not in TARGETS or TARGETS[t].get("type") != "string":
        continue
    if t not in train_df.columns:
        continue
    vals = train_df[t].dropna().astype(str).str.strip()
    vals = vals[vals != ""]
    unique_vals = sorted(vals.unique().tolist())
    TARGETS[t]["allowed"] = unique_vals
    print(f"{t}: allowed = {len(unique_vals)} values")

seeds = [42]
results_dir = '/projects/prjs1308/africa_llm_data/results/testing'
model_dir = '/projects/prjs1308/africa_llm_data/results/inference_models'
batch_size = 1
max_tokens = 4096
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
)
