import os
import sys
import json
import argparse
from datetime import datetime
from types import SimpleNamespace

import torch
import pandas as pd

# Reuse the exact adapter loader from the production inference job (same 4-bit nf4
# QLoRA base + PeftModel + eager attention + run_config.json readback), and the
# training data prep / prompt build / scorer so this diagnostic is byte-identical
# to what training saw and how per-epoch validation was scored.
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_SCRIPT_DIR))
for _p in (_SCRIPT_DIR, _PROJECT_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from inference_africa import load_model_and_tokenizer  # noqa: E402
from agent_utils.africa_dataprep import load_3jun_training_df  # noqa: E402
from agent_utils.gemma3_finetune_simple import (  # noqa: E402
    build_simple_val_prompts,
    run_simple_val_inference,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the fine-tuned adapter over the FULL training set and score "
        "against gold, using training-faithful preprocessing (500-token cap)."
    )
    parser.add_argument(
        "--csv-path",
        type=str,
        default="/projects/prjs1308/africa_llm_data/AFRICA-TRAIN-DB-3jun2026.csv",
        help="Path to the 3jun training CSV (loaded via load_3jun_training_df).",
    )
    parser.add_argument(
        "--adapter-dir",
        type=str,
        required=True,
        help="Fine-tuned adapter dir (containing adapter_model.safetensors and run_config.json).",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="/projects/prjs1308/huggingface/",
        help="HF cache directory for the base Gemma model (reused from training).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory where the metrics CSV and per-record predictions CSV are written.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=None,
        help="Override max_new_tokens; defaults to the value from run_config.json.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N examples (smoke slice); default all rows.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Full training set with gold; TARGETS has string 'allowed' filled from data.
    df, targets = load_3jun_training_df(args.csv_path)

    (
        model,
        tokenizer,
        device_str,
        system_prompt,
        prompt_template,
        max_tokens,
        default_max_new_tokens,
        compute_dtype,
    ) = load_model_and_tokenizer(args.adapter_dir, args.cache_dir)

    # run_simple_val_inference branches on device.type, so pass a torch.device.
    device = torch.device(device_str)

    # gemma_model is only used to name the metrics CSV; read it back from the adapter.
    with open(os.path.join(args.adapter_dir, "run_config.json"), "r", encoding="utf-8") as f:
        gemma_model = json.load(f).get("gemma_model")

    # Training-faithful prompts: same 500-token transcript cap as build_simple_sft_dataset.
    prompts, gold = build_simple_val_prompts(
        df,
        tokenizer,
        prompt_template,
        text_col="text",
        answer_col="targets_json",
        max_seq_length=max_tokens,
        system_prompt=system_prompt,
        max_text_tokens=500,
    )

    # Ids aligned to prompts using the identical skip filter build_simple_val_prompts uses.
    ids = []
    for _, row in df.iterrows():
        if pd.isna(row["text"]) or pd.isna(row["targets_json"]):
            continue
        ids.append(int(row["id"]))
    assert len(ids) == len(prompts) == len(gold), (
        f"id/prompt/gold misalignment: {len(ids)} ids, {len(prompts)} prompts, {len(gold)} gold"
    )

    if args.limit is not None:
        prompts = prompts[: args.limit]
        gold = gold[: args.limit]
        ids = ids[: args.limit]

    print(f"[train-eval] scoring {len(prompts)} examples from {args.csv_path}")

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(args.output_dir, exist_ok=True)
    predictions_out_path = os.path.join(
        args.output_dir, f"train_eval_predictions_{run_id}.csv"
    )

    run_simple_val_inference(
        SimpleNamespace(model=model),
        tokenizer,
        device,
        prompts,
        gold,
        max_new_tokens=(args.max_new_tokens or default_max_new_tokens),
        max_examples=10,
        results_folder=args.output_dir,
        split_name="train",
        targets_spec=targets,
        gemma_model=gemma_model,
        run_id=run_id,
        epoch=0,
        seed=42,
        compute_dtype=compute_dtype,
        val_ids=ids,
        predictions_out_path=predictions_out_path,
    )

    print(f"[train-eval] per-record gold vs generated labels: {predictions_out_path}")


if __name__ == "__main__":
    main()
