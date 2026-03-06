# agent_utils/gemma3_finetune_simple.py
"""
Simple multi-target JSON fine-tuning for Gemma-3.

This is a deliberately minimal version, closely modelled on
``llama3_ilora_finetune.py``.  No slot tokens, no custom loss,
no custom collator — just plain SFT on the full JSON answer string
using TRL's SFTTrainer.

The model learns to generate the complete JSON answer
(with plain values, not slot tokens) given the user prompt + transcript.
"""

import os
import gc
import time
import json
import datetime

import torch
import pandas as pd
from collections import defaultdict
from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from trl import SFTTrainer
from transformers import (
    AutoTokenizer,
    AutoProcessor,
    AutoModelForCausalLM,
    TrainingArguments,
    BitsAndBytesConfig,
)
from peft import (
    PeftModel,
    LoraConfig,
    prepare_model_for_kbit_training,
    get_peft_model,
)

from .utils import (
    setup_seed,
    create_result_filename,
    create_model_dirname,
    print_gpu_memory,
    insert_text_once,
    _extract_last_json,  # robust JSON substring extractor used in main pipeline
)


# ── helpers ───────────────────────────────────────────────────────────

def _build_chat_text_simple(tokenizer, instruction: str, answer: str = None) -> str:
    """Build a chat-formatted string using the tokenizer's chat template."""
    messages = [{"role": "user", "content": instruction}]
    if answer is not None:
        messages.append({"role": "assistant", "content": answer})

    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=(answer is None),
        )
    # Fallback for tokenizers without chat templates
    text = f"<|user|>\n{instruction}"
    if answer is not None:
        text += f"\n<|assistant|>\n{answer}"
    return text


def _token_len(text: str, tokenizer) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))


def build_simple_sft_dataset(
    df,
    tokenizer,
    prompt_template: str,
    text_col: str,
    answer_col: str,
    max_seq_length: int = 4096,
) -> Dataset:
    """
    Build a HuggingFace Dataset with a single ``text`` column containing
    chat-formatted SFT strings: user prompt (with transcript) + assistant
    answer (the raw JSON string).

    Transcript is truncated dynamically so the full string fits in
    ``max_seq_length``.  No slot tokens, no special processing.
    """
    texts = []
    safety_margin = 48

    for _, row in df.iterrows():
        raw_text = row[text_col]
        if pd.isna(raw_text):
            continue
        answer = row[answer_col]
        if pd.isna(answer):
            continue
        answer_str = str(answer)

        # Measure fixed overhead (prompt + chat wrappers + answer, no transcript)
        instruction_empty = insert_text_once(prompt_template, "")
        full_empty = _build_chat_text_simple(tokenizer, instruction_empty, answer_str)
        overhead = _token_len(full_empty, tokenizer)

        transcript_budget = max(max_seq_length - overhead - safety_margin, 0)

        # Truncate transcript to budget
        all_tokens = tokenizer.tokenize(str(raw_text))
        trunc_tokens = all_tokens[:transcript_budget]
        trunc_text = tokenizer.convert_tokens_to_string(trunc_tokens)

        # Build final string
        instruction = insert_text_once(prompt_template, trunc_text)
        full_text = _build_chat_text_simple(tokenizer, instruction, answer_str)

        # Safety check
        final_len = _token_len(full_text, tokenizer)
        if final_len > max_seq_length and len(trunc_tokens) > 0:
            overshoot = final_len - max_seq_length + 32
            trunc_tokens = trunc_tokens[:-overshoot] if overshoot < len(trunc_tokens) else []
            trunc_text = tokenizer.convert_tokens_to_string(trunc_tokens)
            instruction = insert_text_once(prompt_template, trunc_text)
            full_text = _build_chat_text_simple(tokenizer, instruction, answer_str)

        texts.append(full_text)

    print(f"[simple-sft] Built {len(texts)} training examples "
          f"(max_seq_length={max_seq_length})")
    if texts:
        sample_len = _token_len(texts[0], tokenizer)
        print(f"  example 0: {sample_len} tokens, {len(texts[0])} chars")
        print(f"  tail: ...{texts[0][-400:]}")

    return Dataset.from_dict({"text": texts})


def build_simple_val_prompts(
    df,
    tokenizer,
    prompt_template: str,
    text_col: str,
    answer_col: str,
    max_seq_length: int = 4096,
):
    """
    Build validation prompts (user turn only, with transcript) and gold answers.
    Same truncation logic as build_simple_sft_dataset so prompts match training.
    Returns (val_prompts, val_gold_raw) for inference.
    """
    prompts = []
    gold_raw = []
    safety_margin = 48

    for _, row in df.iterrows():
        raw_text = row[text_col]
        if pd.isna(raw_text):
            continue
        answer = row[answer_col]
        if pd.isna(answer):
            continue
        answer_str = str(answer)

        instruction_empty = insert_text_once(prompt_template, "")
        full_empty = _build_chat_text_simple(tokenizer, instruction_empty, answer_str)
        overhead = _token_len(full_empty, tokenizer)
        transcript_budget = max(max_seq_length - overhead - safety_margin, 0)

        all_tokens = tokenizer.tokenize(str(raw_text))
        trunc_tokens = all_tokens[:transcript_budget]
        trunc_text = tokenizer.convert_tokens_to_string(trunc_tokens)
        instruction = insert_text_once(prompt_template, trunc_text)

        # Prompt only (user turn + generation prompt), no assistant answer
        prompt_only = _build_chat_text_simple(tokenizer, instruction, answer=None)
        prompts.append(prompt_only)
        gold_raw.append(answer_str)

    return prompts, gold_raw


def _extract_pred_json(raw_completion: str):
    """
    Best-effort extraction of a JSON object from the model's completion.

    Handles common patterns like Markdown fences:
      ```json
      { ... }
      ```
    and returns a dict or None.
    """
    if not raw_completion:
        return None

    s = raw_completion.strip()

    # Robustly strip leading/trailing markdown fences like:
    # ```json\n{...}\n```
    if "```" in s:
        # keep only the part between the first and last fence
        first_fence = s.find("```")
        last_fence = s.rfind("```")
        if first_fence != -1 and last_fence != -1 and last_fence > first_fence:
            inner = s[first_fence + 3 : last_fence]
            s = inner.strip()

    # Use the robust helper from utils.py to get the last {...} block
    candidate = _extract_last_json(s)

    try:
        return json.loads(candidate)
    except Exception:
        return None


def run_simple_val_inference(
    trainer,
    tokenizer,
    device,
    val_prompts,
    val_gold_raw,
    max_new_tokens: int = 400,
    max_examples: int = 5,
    results_folder: str | None = None,
    mtype: str = "simple_gemma3",
    learning_rate: float | None = None,
    epoch: int | None = None,
    seed: int | None = None,
    split_name: str = "val",
):
    """
    Run generation on validation prompts, print a few examples, and compute
    simple per-target metrics (accuracy, precision, recall, F1).
    """
    N = min(len(val_prompts), len(val_gold_raw))
    if N == 0:
        print("[val-inference] No validation examples to run.")
        return

    n_print = min(max_examples, N)

    # storage for metrics
    per_target_true = defaultdict(list)
    per_target_pred = defaultdict(list)

    print("\n" + "=" * 80)
    print(f"{split_name.upper()} INFERENCE (first {n_print} examples, max_new_tokens={max_new_tokens})")
    print("=" * 80)

    pad_token_id = getattr(tokenizer, "pad_token_id", None) or getattr(
        tokenizer, "eos_token_id", None
    )
    trainer.model.eval()

    for i in range(N):
        prompt_text = val_prompts[i]
        gold = val_gold_raw[i]

        enc = tokenizer(
            prompt_text,
            return_tensors="pt",
            padding=False,
            truncation=False,
        )
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        with torch.no_grad():
            generated = trainer.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_return_sequences=1,
                pad_token_id=pad_token_id,
            )

        input_len = input_ids.shape[-1]
        new_tokens = generated[0, input_len:].detach().cpu()
        raw_completion = tokenizer.decode(new_tokens, skip_special_tokens=True)

        # Pretty-print only for the first n_print examples
        if i < n_print:
            print(f"\n--- Example {i + 1}/{N} ---")
            print(f"PROMPT (last 350 chars): ...{prompt_text[-350:]}")
            print(f"GOLD: {gold[:500]}{'...' if len(gold) > 500 else ''}")
            print(f"GENERATED: {raw_completion[:800]}{'...' if len(raw_completion) > 800 else ''}")

        parsed = _extract_pred_json(raw_completion)

        # parse gold JSON
        try:
            gold_dict = json.loads(gold)
        except Exception:
            gold_dict = {}

        # accumulate labels per target
        if gold_dict:
            for t, g_val in gold_dict.items():
                if g_val is None:
                    continue  # skip missing gold

                # Normalize numeric labels (e.g. 1.0 -> 1)
                if isinstance(g_val, float) and g_val.is_integer():
                    g_val_norm = int(g_val)
                else:
                    g_val_norm = g_val

                if parsed is not None and t in parsed:
                    p_val = parsed[t]
                    if isinstance(p_val, float) and p_val.is_integer():
                        p_val_norm = int(p_val)
                    else:
                        p_val_norm = p_val
                else:
                    p_val_norm = "MISSING"

                per_target_true[t].append(g_val_norm)
                per_target_pred[t].append(p_val_norm)

        if i < n_print:
            if parsed is not None:
                pred_str = json.dumps(parsed, ensure_ascii=False)
                print(f"PREDICTED_JSON: {pred_str[:800]}{'...' if len(pred_str) > 800 else ''}")
            else:
                snippet = raw_completion.strip()
                print(
                    "PREDICTED_JSON: <failed to parse JSON> | "
                    f"snippet={snippet[:200].replace(chr(10), ' ')}"
                    f"{'...' if len(snippet) > 200 else ''}"
                )

        # Light progress indicator every 5 examples
        if (i + 1) % 5 == 0 or i == N - 1:
            print(f"[val-inference] processed {i + 1}/{N} validation examples")

    # ---- per-target metrics ----
    print("\n" + "-" * 80)
    print(f"{split_name.upper()} METRICS PER TARGET (N={N} examples with prompts)")
    print("-" * 80)
    header = (
        f"{'target':25s} {'n':>5s} "
        f"{'acc':>8s} {'prec':>8s} {'rec':>8s} {'f1':>8s} "
        f"{'answered%':>10s} {'in_label%':>10s}"
    )
    print(header)
    print("-" * len(header))

    rows = []

    for t in sorted(per_target_true.keys()):
        all_true = per_target_true[t]
        all_pred = per_target_pred[t]
        if not all_true:
            continue

        # Filter to positions where we have a concrete prediction
        pairs = [(g, p) for g, p in zip(all_true, all_pred) if p != "MISSING"]
        n_gold = len(all_true)
        n_answered = len(pairs)

        answered_frac = n_answered / n_gold if n_gold > 0 else 0.0

        # "Expected labels" = label set observed in gold for this target
        gold_label_set = set(all_true)
        n_in_label = sum(1 for _, p in pairs if p in gold_label_set)
        in_label_frac = n_in_label / n_gold if n_gold > 0 else 0.0

        if n_answered > 0:
            y_true = [g for g, _ in pairs]
            y_pred = [p for _, p in pairs]

            try:
                acc = accuracy_score(y_true, y_pred)
                prec, rec, f1, _ = precision_recall_fscore_support(
                    y_true, y_pred, average="macro", zero_division=0
                )
            except Exception:
                acc = prec = rec = f1 = 0.0
        else:
            acc = prec = rec = f1 = 0.0

        print(
            f"{t:25s} {n_gold:5d} "
            f"{acc:8.3f} {prec:8.3f} {rec:8.3f} {f1:8.3f} "
            f"{answered_frac*100:10.1f} {in_label_frac*100:10.1f}"
        )

        # Collect example-level answer statistics per target for CSV export
        answers_in_label = sorted({str(p) for _, p in pairs if p in gold_label_set})
        answers_out_label = sorted({str(p) for _, p in pairs if p not in gold_label_set})

        rows.append(
            {
                "target": t,
                "n_gold": n_gold,
                "n_answered": n_answered,
                "n_in_label": n_in_label,
                "accuracy": acc,
                "precision_macro": prec,
                "recall_macro": rec,
                "f1_macro": f1,
                "answered_frac": answered_frac,
                "in_label_frac": in_label_frac,
                "answers_in_label": ";".join(answers_in_label),
                "answers_out_of_label": ";".join(answers_out_label),
            }
        )

    print("=" * 80 + "\n")

    # Optionally save metrics to CSV
    if results_folder is not None and rows:
        os.makedirs(results_folder, exist_ok=True)
        lr_str = f"{learning_rate}" if learning_rate is not None else "na"
        ep_str = f"{epoch}" if epoch is not None else "na"
        seed_str = f"{seed}" if seed is not None else "na"
        csv_name = f"{mtype}_{split_name}_metrics_lr{lr_str}_seed{seed_str}_epoch{ep_str}.csv"
        csv_path = os.path.join(results_folder, csv_name)
        pd.DataFrame(rows).to_csv(csv_path, index=False)
        print(f"[{split_name}-metrics] Saved per-target metrics to {csv_path}")


# ── main entry point ──────────────────────────────────────────────────

def run_simple_gemma3(
    train_df,
    test_df,
    text_col: str,
    answer_col: str,
    prompt: str,
    train_val_seeds=(42,),
    val_size=0.2,
    results_folder=None,
    model_dir=None,
    max_tokens=4096,
    batch_size=1,
    max_new_tokens=300,
    cache_dir=None,
    local_model=None,
    early_stopping_patience=3,
    epochs=5,
    learning_rates=(1e-4,),
    grad_accum_steps=4,
    model_id="google/gemma-3-27b-it",
    max_val_infer=5,
):
    """
    Simple multi-target JSON fine-tuning for Gemma-3.

    Closely mirrors ``run_fine_tuned_llama3_ilora`` but:
      - Uses Gemma-3 12B-IT (configurable via ``model_id``)
      - Trains with plain SFTTrainer (no slot tokens, no custom loss)
      - The assistant answer is the raw JSON string from ``answer_col``
    """
    mtype = "simple_gemma3"

    # ── GPU / device ──────────────────────────────────────────────────
    gpu_avail = torch.cuda.is_available()
    device = torch.device("cuda" if gpu_avail else "cpu")
    print(f"Device: {device}")

    pynvml_mod = None
    handle = None
    if gpu_avail:
        try:
            import pynvml as _pynvml
            _pynvml.nvmlInit()
            handle = _pynvml.nvmlDeviceGetHandleByIndex(0)
            pynvml_mod = _pynvml
        except Exception:
            pass

    # ── Tokenizer ─────────────────────────────────────────────────────
    original_model = model_id if local_model is None else model_id
    print(f"Loading tokenizer from {model_id}")

    processor = AutoProcessor.from_pretrained(model_id, cache_dir=cache_dir)
    tokenizer = getattr(processor, "tokenizer", processor)

    if getattr(tokenizer, "pad_token", None) is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"

    # ── Quantisation + LoRA config ────────────────────────────────────
    compute_dtype = torch.bfloat16 if (gpu_avail and torch.cuda.is_bf16_supported()) else torch.float16

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=False,
    )

    peft_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.1,
        r=64,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )

    if pynvml_mod and handle:
        print("Initial GPU usage")
        print_gpu_memory(handle, pynvml_mod)

    cv_performances = pd.DataFrame()

    # ── Pre-build TEST prompts for inference ─────────────────────────-
    test_prompts, test_gold_raw = build_simple_val_prompts(
        df=test_df,
        tokenizer=tokenizer,
        prompt_template=prompt,
        text_col=text_col,
        answer_col=answer_col,
        max_seq_length=max_tokens,
    )
    print(f"[simple-sft] Built {len(test_prompts)} TEST prompts for inference")

    # ── LR / seed loop ────────────────────────────────────────────────
    for learning_rate in learning_rates:
        fold_counter = 0

        for train_val_seed in train_val_seeds:
            setup_seed(train_val_seed)
            print(f"\n=== LR {learning_rate} | seed {train_val_seed} | fold {fold_counter} ===")

            # ── Train / val split ─────────────────────────────────────
            train_rows, val_rows = train_test_split(
                train_df, test_size=val_size, random_state=train_val_seed,
            )
            print(f"Train: {len(train_rows)} rows  |  Val: {len(val_rows)} rows")

            # ── 1) Build datasets ─────────────────────────────────────
            dataset = build_simple_sft_dataset(
                df=train_rows,
                tokenizer=tokenizer,
                prompt_template=prompt,
                text_col=text_col,
                answer_col=answer_col,
                max_seq_length=max_tokens,
            )

            val_dataset = build_simple_sft_dataset(
                df=val_rows,
                tokenizer=tokenizer,
                prompt_template=prompt,
                text_col=text_col,
                answer_col=answer_col,
                max_seq_length=max_tokens,
            )

            # Validation prompts for inference (same truncation as train)
            val_prompts, val_gold_raw = build_simple_val_prompts(
                df=val_rows,
                tokenizer=tokenizer,
                prompt_template=prompt,
                text_col=text_col,
                answer_col=answer_col,
                max_seq_length=max_tokens,
            )
            print(f"[simple-sft] Built {len(val_prompts)} validation prompts for inference")

            # ── Print 1–2 full training examples before training ───────
            n_print = min(2, len(dataset))
            print("\n" + "=" * 80)
            print(f"TRAINING EXAMPLES (first {n_print} full strings)")
            print("=" * 80)
            for i in range(n_print):
                full_text = dataset["text"][i]
                n_tok = _token_len(full_text, tokenizer)
                print(f"\n--- Example {i} ({len(full_text)} chars, {n_tok} tokens) ---\n")
                print(full_text)
                print()
            print("=" * 80 + "\n")

            # ── 2) Load model ─────────────────────────────────────────
            print(f"Loading model: {original_model}")
            model = AutoModelForCausalLM.from_pretrained(
                original_model,
                quantization_config=quant_config,
                device_map={"": 0} if gpu_avail else None,
                cache_dir=cache_dir,
                torch_dtype=compute_dtype if gpu_avail else None,
            )
            model.config.use_cache = False
            model.config.pad_token_id = tokenizer.pad_token_id

            if gpu_avail and quant_config is not None:
                model = prepare_model_for_kbit_training(model)

            if local_model is not None:
                model = PeftModel.from_pretrained(model, local_model, is_trainable=True)
            else:
                model = get_peft_model(model, peft_config)

            print(f"Model loaded for fold {fold_counter}")
            model.print_trainable_parameters()

            # ── 3) Training arguments ─────────────────────────────────
            training_args = TrainingArguments(
                output_dir="./results_simple",
                num_train_epochs=1,
                per_device_train_batch_size=batch_size,
                per_device_eval_batch_size=batch_size,
                gradient_accumulation_steps=grad_accum_steps,
                optim="paged_adamw_32bit",
                do_eval=True,
                eval_strategy="epoch",
                save_steps=3000,
                logging_steps=25,
                learning_rate=learning_rate,
                weight_decay=0.001,
                fp16=False,
                bf16=False,
                max_grad_norm=0.3,
                max_steps=-1,
                warmup_ratio=0.03,
                group_by_length=True,
                lr_scheduler_type="linear",
                report_to="tensorboard",
                seed=train_val_seed,
            )

            # ── 4) Trainer ────────────────────────────────────────────
            trainer = SFTTrainer(
                model=model,
                train_dataset=dataset,
                eval_dataset=val_dataset,
                dataset_text_field="text",
                max_seq_length=max_tokens,
                tokenizer=tokenizer,
                args=training_args,
                packing=False,
            )

            print(f"Trainer ready  |  train bs={trainer.args.per_device_train_batch_size}  "
                  f"grad_accum={trainer.args.gradient_accumulation_steps}")

            # Directory where we'll save the adapter/tokenizer for this LR/seed.
            # We overwrite the same folder each epoch, keeping only the latest.
            if model_dir is not None:
                save_subdir = f"{mtype}_lr{learning_rate}_seed{train_val_seed}"
                model_save_dir = os.path.join(model_dir, save_subdir)
                os.makedirs(model_save_dir, exist_ok=True)
            else:
                model_save_dir = None

            # ── 5) Epoch loop ─────────────────────────────────────────
            best_eval_loss = float("inf")
            no_improvement_counter = 0

            for ep in range(epochs):
                start_time = time.time()
                print(f"\nEpoch {ep} | LR {learning_rate}")

                trainer.train()

                # Extract losses from log history
                train_loss = trainer.state.log_history[-1].get("train_loss", None)
                eval_loss = trainer.state.log_history[-2].get("eval_loss", None)
                elapsed = time.time() - start_time

                print(f"  train_loss={train_loss}  eval_loss={eval_loss}  "
                      f"time={elapsed:.0f}s")

                if pynvml_mod and handle:
                    print_gpu_memory(handle, pynvml_mod)

                # Validation inference: generate on val prompts and print + metrics
                run_simple_val_inference(
                    trainer=trainer,
                    tokenizer=tokenizer,
                    device=device,
                    val_prompts=val_prompts,
                    val_gold_raw=val_gold_raw,
                    max_new_tokens=max_new_tokens,
                    max_examples=max_val_infer,
                    results_folder=results_folder,
                    mtype=mtype,
                    learning_rate=learning_rate,
                    epoch=ep,
                    seed=train_val_seed,
                    split_name="val",
                )

                # Test inference: same procedure on held-out test set
                run_simple_val_inference(
                    trainer=trainer,
                    tokenizer=tokenizer,
                    device=device,
                    val_prompts=test_prompts,
                    val_gold_raw=test_gold_raw,
                    max_new_tokens=max_new_tokens,
                    max_examples=max_val_infer,
                    results_folder=results_folder,
                    mtype=mtype,
                    learning_rate=learning_rate,
                    epoch=ep,
                    seed=train_val_seed,
                    split_name="test",
                )

                # Save model + tokenizer after each epoch (overwrites previous epoch)
                if model_save_dir is not None:
                    print(f"[save] Saving adapter/tokenizer to {model_save_dir}")
                    model.save_pretrained(model_save_dir)
                    tokenizer.save_pretrained(model_save_dir)

                # Early stopping
                if eval_loss is not None:
                    if eval_loss < best_eval_loss:
                        best_eval_loss = eval_loss
                        no_improvement_counter = 0
                    else:
                        no_improvement_counter += 1

                    if no_improvement_counter >= early_stopping_patience:
                        print(f"Early stopping after {ep+1} epochs.")
                        break

            # ── Cleanup ───────────────────────────────────────────────
            print("Training done, cleaning up")
            del model
            gc.collect()
            if gpu_avail:
                torch.cuda.empty_cache()
            fold_counter += 1

    return cv_performances
