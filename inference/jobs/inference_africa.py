import os
import sys
import json
import math
import time
import argparse
from datetime import datetime

import torch
import pandas as pd
from transformers import AutoProcessor, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Gemma-3 Africa LLM inference over a CSV of posts."
    )
    parser.add_argument(
        "--data-path",
        type=str,
        required=True,
        help="Path to input CSV with at least 'id' and 'text' columns.",
    )
    parser.add_argument(
        "--adapter-dir",
        type=str,
        required=True,
        help="Path to the fine-tuned adapter directory (the one containing adapter_model.safetensors and run_config.json).",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="/projects/prjs1308/huggingface/",
        help="HF cache directory for the base Gemma model (reused from training).",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--range",
        type=str,
        help="Explicit index range as 'start:end' (0-based, end exclusive).",
    )
    group.add_argument(
        "--quarter",
        type=int,
        choices=[1, 2, 3, 4],
        help="Process the 1st, 2nd, 3rd, or 4th quarter of the dataframe.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/home/fcool/africa_llm/inference/results/test_results",
        help="Directory where per-range inference CSVs will be written.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=None,
        help="Override max_new_tokens used at inference; defaults to value from run_config.json.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="If set, only print which rows would be processed and exit.",
    )
    return parser.parse_args()


def load_model_and_tokenizer(adapter_dir: str, cache_dir: str):
    # Load run-time config saved during fine-tuning
    run_cfg_path = os.path.join(adapter_dir, "run_config.json")
    with open(run_cfg_path, "r", encoding="utf-8") as f:
        run_cfg = json.load(f)

    base_model_id = run_cfg["model_id"]
    system_prompt = run_cfg.get("system_prompt")
    prompt_template = run_cfg["prompt_template"]
    max_tokens = run_cfg.get("max_tokens", 4096)
    default_max_new_tokens = run_cfg.get("max_new_tokens", 500)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_dtype = (
        torch.bfloat16
        if device == "cuda" and torch.cuda.is_bf16_supported()
        else torch.float16
    )

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=False,
    )

    processor = AutoProcessor.from_pretrained(adapter_dir, cache_dir=cache_dir)
    tokenizer = getattr(processor, "tokenizer", processor)

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        quantization_config=quant_config,
        device_map={"": 0} if device == "cuda" else None,
        torch_dtype=compute_dtype if device == "cuda" else None,
        cache_dir=cache_dir,
    )

    model = PeftModel.from_pretrained(base_model, adapter_dir)
    model.eval()
    model.to(device)

    print("Loaded base model:", base_model_id)
    print("Using adapter from:", adapter_dir)
    print("Device:", device)

    return model, tokenizer, device, system_prompt, prompt_template, max_tokens, default_max_new_tokens


def setup_agent_utils_import(project_root: str):
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    from agent_utils.gemma3_finetune_simple import _extract_pred_json  # type: ignore

    return _extract_pred_json


def build_chat_input(tokenizer, system_prompt: str | None, prompt_template: str, text: str) -> str:
    instruction = prompt_template.format(text)
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": instruction})
    chat = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    return chat


def generate_annotation(
    model,
    tokenizer,
    device: str,
    system_prompt: str | None,
    prompt_template: str,
    _extract_pred_json,
    text: str,
    default_max_new_tokens: int,
    max_new_tokens_override: int | None = None,
):
    """Generate JSON annotation for a single post.

    Returns (raw_text, parsed_json, elapsed_sec).
    Uses _extract_pred_json first; if that fails, falls back to a
    loose key/value regex extractor so we still recover partial
    annotations when the JSON is slightly malformed/truncated.
    """
    import re

    max_new_tokens = max_new_tokens_override or default_max_new_tokens
    start = time.time()

    chat = build_chat_input(tokenizer, system_prompt, prompt_template, text)
    inputs = tokenizer(chat, return_tensors="pt").to(device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    gen_ids = output_ids[0, inputs["input_ids"].shape[1]:]
    completion = tokenizer.decode(gen_ids, skip_special_tokens=True)

    parsed = _extract_pred_json(completion)

    if parsed is None:
        pattern = r'"([A-Za-z0-9_]+)"\s*:\s*(null|true|false|-?\d+\.\d+|-?\d+|"(?:[^"\\]|\\.)*")'
        results = {}
        for m in re.finditer(pattern, completion):
            key = m.group(1)
            raw_val = m.group(2)
            if raw_val == "null":
                val = None
            elif raw_val in ("true", "false"):
                val = 1 if raw_val == "true" else 0
            elif raw_val.startswith('"') and raw_val.endswith('"'):
                try:
                    val = json.loads(raw_val)
                except Exception:
                    val = raw_val.strip('"')
            else:
                try:
                    val = float(raw_val) if "." in raw_val else int(raw_val)
                except Exception:
                    val = raw_val
            results[key] = val
        parsed = results or None

    elapsed = time.time() - start
    return completion, parsed, elapsed


def main():
    args = parse_args()

    if not os.path.exists(args.data_path):
        print(f"Input CSV not found at {args.data_path}. Nothing to do.")
        return

    df = pd.read_csv(args.data_path)
    if "id" not in df.columns or "text" not in df.columns:
        raise ValueError("Input CSV must contain at least 'id' and 'text' columns.")

    print(f"Loaded {len(df)} rows from {args.data_path}")

    # Determine row range
    if args.quarter is not None:
        N = len(df)
        chunk_size = math.ceil(N / 4)
        start_idx = (args.quarter - 1) * chunk_size
        end_idx = min(args.quarter * chunk_size, N)
        range_name = f"quarter_{args.quarter}"
    else:
        try:
            start_s, end_s = args.range.split(":")
            start_idx = int(start_s)
            end_idx = int(end_s)
        except Exception as e:
            raise ValueError(f"Invalid --range format '{args.range}', expected 'start:end'") from e
        range_name = f"indices_{start_idx}_{end_idx}"

    subset = df.iloc[start_idx:end_idx].copy()
    print(
        f"Running inference on rows [{start_idx}, {end_idx}) "
        f"→ {len(subset)} examples (range={range_name})"
    )

    if args.dry_run:
        print("Dry run requested; exiting before loading model.")
        return

    # Load model/tokenizer and helper from repo
    model, tokenizer, device, system_prompt, prompt_template, max_tokens, default_max_new_tokens = load_model_and_tokenizer(
        args.adapter_dir,
        args.cache_dir,
    )
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    _extract_pred_json = setup_agent_utils_import(project_root)

    inference_times: list[float] = []
    n_seen = 0
    n_ok = 0

    # Stable output columns based on training targets_spec in run_config
    with open(os.path.join(args.adapter_dir, "run_config.json"), "r", encoding="utf-8") as f:
        run_cfg = json.load(f)
    targets_spec = run_cfg.get("targets_spec") or {}
    target_cols = sorted(targets_spec.keys())
    pred_columns = ["id", "json_ok", "range_name"] + target_cols

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(
        args.output_dir,
        f"inference_predictions_{range_name}_{run_id}.csv",
    )

    print(f"Writing per-example results to {out_path}")

    for idx, (_, row) in enumerate(subset.iterrows(), start=start_idx):
        text = row["text"]
        row_id = row["id"]

        raw_out, parsed_out, t_sec = generate_annotation(
            model=model,
            tokenizer=tokenizer,
            device=device,
            system_prompt=system_prompt,
            prompt_template=prompt_template,
            _extract_pred_json=_extract_pred_json,
            text=text,
            default_max_new_tokens=default_max_new_tokens,
            max_new_tokens_override=args.max_new_tokens,
        )
        inference_times.append(t_sec)
        avg_time = sum(inference_times) / len(inference_times)

        n_seen += 1
        json_ok = isinstance(parsed_out, dict)
        if json_ok:
            n_ok += 1
        success_pct = (n_ok / n_seen) * 100.0

        rec = {"id": row_id, "json_ok": json_ok, "range_name": range_name}
        if isinstance(parsed_out, dict):
            for k in target_cols:
                if k in parsed_out:
                    rec[k] = parsed_out[k]

        print("\n" + "=" * 80)
        print(f"ROW {idx} (id={row_id})")
        print("TEXT:")
        print(text)
        print("\nRAW COMPLETION:")
        print(raw_out)
        print("\nPARSED JSON:")
        print(parsed_out)
        print(f"\nInference time: {t_sec:.2f}s  |  Running avg: {avg_time:.2f}s")
        print(f"JSON extracted: {json_ok}  |  Running success: {success_pct:.1f}%")

        row_df = pd.DataFrame([[rec.get(col, None) for col in pred_columns]], columns=pred_columns)
        write_header = not os.path.exists(out_path)
        row_df.to_csv(out_path, mode="a", index=False, header=write_header)

    print(
        f"\nSaved predictions for {len(inference_times)} rows (appended) to {out_path}"
    )


if __name__ == "__main__":
    main()

