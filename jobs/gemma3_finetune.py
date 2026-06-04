import sys
print(sys.version)

import os
import re
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

# Codebook §12 topic labels (up to 3, ;-joined). "not applicable" is retained
# even though §12 doesn't spell it out — gated rows can carry it (S4).
TOPIC_ALLOWED = [
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
    "ETHNICITY",
    "not applicable",
]

# Labels + transcripts come from the 3jun CSV. It is the 1jun label set with the
# three E3 string fixes applied and a `text` column merged in, so transcripts are
# read straight from the CSV (verified: text matches african_videos.json on every
# overlapping id). No json join, so the E1 id-cast-join failure can't occur.
# CSV `id` is float64 (e.g. 1712829139429512.0) -> cast to int64 for a clean key.
CSV_PATH = Path("/projects/prjs1308/africa_llm_data/AFRICA-TRAIN-DB-3jun2026.csv")

df = pd.read_csv(CSV_PATH)
df["id"] = pd.to_numeric(df["id"], errors="coerce").astype("int64")

# Label columns = every column except the id key and the transcript text.
label_cols = [c for c in df.columns if c not in ("id", "text")]

# Keep only rows with a usable transcript; log how many are dropped.
has_text = df["text"].notna() & (df["text"].astype(str).str.strip() != "")
n_total = len(df)
n_with_text = int(has_text.sum())
print(f"[data] rows={n_total} with_text={n_with_text} dropped={n_total - n_with_text}")
df = df.loc[has_text].reset_index(drop=True)

# E4: "NOT CODED" in national_unity_narrow is the annotators' "never coded this field"
# sentinel (~1589 rows), not a real label. Blank it to null so the field is excluded
# from the training target and from metrics; the row itself is kept. (Resolved Q6.)
not_coded = df["national_unity_narrow"].astype(str).str.strip() == "NOT CODED"
print(f"[data] national_unity_narrow NOT CODED -> null: {int(not_coded.sum())} rows")
df.loc[not_coded, "national_unity_narrow"] = None

# Build targets_json directly from the label columns. Values are already the
# codebook label-name strings (e.g. topic = "TRANSPORTATION"), so no topic_mapping
# or numeric recode is applied.
df["targets_json"] = df[label_cols].apply(
    lambda row: json.dumps(
        {c: (None if pd.isna(row[c]) else row[c]) for c in label_cols},
        ensure_ascii=False,
    ),
    axis=1,
)

# split
train_df, test_df = train_test_split(df, test_size=0.2, random_state=1, shuffle=True)

print(train_df[["text", "targets_json"]].head(2).to_string(index=False))
print(test_df[["text", "targets_json"]].head(2).to_string(index=False))

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

# Codebook label-name target spec (2004 codebook), ordered §1–§27. Every `allowed`
# set is the codebook's label-name list, data-verified against the 3jun CSV:
# 26/27 categorical fields match exactly; the only out-of-codebook values are
# language's free-form Other(<lang>), accepted via `allow_other_paren` (D2).
# `multi_value` fields are `;`-joined and scored as order-insensitive sets (D1).
# Conditional fields are multiclass (they carry "not applicable"/"unclear"), not binary.
TARGETS = {
    "language": {"type": "multiclass", "multi_value": True, "allow_other_paren": True, "allowed": [
        "English", "French", "Arabic", "Portuguese", "Swahili", "Hausa", "Yoruba",
        "Other", "Unclear"]},
    "politics": {"type": "multiclass", "allowed": [
        "politics", "not political", "unclear"]},
    "domestic_politics": {"type": "multiclass", "allowed": [
        "domestic politics", "not domestic politics", "unclear", "not applicable"]},
    "foreign_politics": {"type": "multiclass", "allowed": [
        "foreign politics", "not foreign politics", "unclear", "not applicable"]},
    "resource_distribution": {"type": "multiclass", "allowed": [
        "resource distribution", "not resource distribution", "unclear", "not applicable"]},
    "resource_distribution_by_whom1": {"type": "multiclass", "multi_value": True, "allowed": [
        "other state", "international organisation", "national government", "other",
        "not specified", "unclear", "not applicable"]},
    "resource_distribution_for_whom1": {"type": "multiclass", "multi_value": True, "allowed": [
        "specific locality or group", "country-wide", "not specified", "unclear",
        "not applicable"]},
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
    "resource_distribution_gender": {"type": "multiclass", "allowed": [
        "resources for women", "resources not specifically for women", "unclear",
        "not applicable"]},
    "climate_change": {"type": "multiclass", "allowed": [
        "mentions climate change", "mentions sustainability", "unclear", "not applicable"]},
    "topic": {"type": "multiclass", "multi_value": True, "allowed": TOPIC_ALLOWED},
    "pro_us": {"type": "multiclass", "allowed": [
        "positive towards the US", "neutral towards the US", "negative towards the US",
        "unclear", "no mention of the US", "not applicable"]},
    "pro_russia": {"type": "multiclass", "allowed": [
        "positive towards Russia", "neutral towards Russia", "negative towards Russia",
        "unclear", "no mention of Russia", "not applicable"]},
    "pro_china": {"type": "multiclass", "allowed": [
        "positive towards China", "neutral towards China", "negative towards China",
        "unclear", "no mention of China", "not applicable"]},
    "pro_un": {"type": "multiclass", "allowed": [
        "positive towards the UN", "neutral towards the UN", "negative towards the UN",
        "unclear", "no mention of the UN", "not applicable"]},
    "pro_imf": {"type": "multiclass", "allowed": [
        "positive towards the IMF", "neutral towards the IMF", "negative towards the IMF",
        "unclear", "no mention of the IMF", "not applicable"]},
    "pro_democracy": {"type": "multiclass", "allowed": [
        "positive towards democracy", "neutral towards democracy", "negative towards democracy",
        "unclear", "no mention of democracy", "not applicable"]},
    "anti_western": {"type": "multiclass", "allowed": [
        "anti-western", "not anti-western", "unclear", "not applicable"]},
    "national_unity": {"type": "multiclass", "allowed": [
        "national unity", "no mention of the nation or national unity", "unclear"]},
    "national_unity_narrow": {"type": "multiclass", "allowed": [
        "patriotism", "not specifically patriotic", "unclear"]},  # NOT CODED → null (E4)
    "subgroup_unity": {"type": "multiclass", "allowed": [
        "subgroup unity", "no mention of specific subgroup", "unclear"]},
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
    "african_unity": {"type": "multiclass", "allowed": [
        "african unity", "no mention of africa", "unclear"]},
    "political_opponents": {"type": "multiclass", "allowed": [
        "mentions political opponents", "no mention of political opponents", "unclear",
        "not applicable"]},
    "political_opponents_viol": {"type": "multiclass", "allowed": [
        "mentions violent group", "no mention of violent group", "unclear", "not applicable"]},
    "religion": {"type": "multiclass", "allowed": [
        "religious", "no mention of religion", "unclear"]},
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

# Codebook-conformance guard (T4): TARGETS keys must match the CSV label columns
# 1-for-1, and every non-string field's `;`-split atomic values must be a subset of
# its `allowed` set. The only exemption is language's free-form Other(<lang>)
# (allow_other_paren / D2). national_unity_narrow's "NOT CODED" is already nulled by
# E4, so it never reaches here. If this fails, fix the data at source — do not patch
# in code (hard rule: no workarounds). This is the loud-failure guard for a CSV swap.
_OTHER_PAREN_RE = re.compile(r"^Other\(.*\)$")

target_keys, csv_keys = set(TARGETS.keys()), set(label_cols)
if target_keys != csv_keys:
    raise ValueError(
        "TARGETS keys != CSV label columns.\n"
        f"  only in TARGETS: {sorted(target_keys - csv_keys)}\n"
        f"  only in CSV:     {sorted(csv_keys - target_keys)}"
    )

violations = {}
for col, spec in TARGETS.items():
    if spec.get("type") == "string":
        continue
    allowed_set = {str(a).strip() for a in spec["allowed"]}
    allow_other = spec.get("allow_other_paren", False)
    bad = {}
    for cell in df[col].dropna().astype(str):
        for atom in cell.split(";"):
            atom = atom.strip()
            if not atom or atom in allowed_set:
                continue
            if allow_other and _OTHER_PAREN_RE.match(atom):
                continue
            bad[atom] = bad.get(atom, 0) + 1
    if bad:
        violations[col] = dict(sorted(bad.items()))
if violations:
    raise ValueError(
        "Out-of-codebook values found (fix the data at source, not in code):\n"
        + "\n".join(f"  {c}: {b}" for c, b in violations.items())
    )
n_checked = sum(1 for s in TARGETS.values() if s.get("type") != "string")
print(f"[conformance] OK — {n_checked} categorical fields ⊆ codebook label sets; "
      f"TARGETS keys == CSV columns ({len(target_keys)})")

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
    targets_spec=TARGETS,      # for inference: in-label / set scoring of semicolon-separated multiclass (e.g. topic)
    system_prompt=system_prompt,  # codebook in system role → KV prefix caching at inference
)