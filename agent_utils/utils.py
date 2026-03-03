# agent_utils/utils.py

import sys
import os
import re
import gc
import ast
import json
import time
import math
import torch
import random
import datetime
from collections import defaultdict, Counter

import torch
import numpy as np
import pandas as pd

from PIL import Image

from peft import LoraConfig
from datasets import Dataset
from transformers.image_utils import load_image
from transformers import BitsAndBytesConfig, TrainingArguments

from sklearn.metrics import precision_recall_fscore_support

# ==============================================================================
# DEBUG CONFIG
# ==============================================================================
DEBUG = bool(int(os.environ.get("AFRICA_DEBUG", "1")))          # AFRICA_DEBUG=1
DEBUG_EXAMPLE_IDX = int(os.environ.get("AFRICA_DEBUG_IDX", "1")) # AFRICA_DEBUG_IDX=0
DEBUG_MAX_CHARS = int(os.environ.get("AFRICA_DEBUG_CHARS", "4000"))
DEBUG_ONCE = {"printed": False}  # global guard


# ==============================================================================
# ROUTER
# ==============================================================================
def train_validate(
    mtype,
    train_df,
    test_df,
    test_unseen_df=None,
    text_col=None,
    target_col=None,
    prompt=None,              # STRING template, e.g. "Text:\n{}\nAnswer:"
    answer_col=None,          # df column containing supervised answer text (e.g. "targets_json")
    train_val_seeds=None,
    val_size=0.2,
    results_folder=None,
    model_dir=None,
    image_col=None,
    id_col=None,
    max_words=None,
    max_tokens=None,
    batch_size=None,
    image_folder=None,
    class_labels=None,
    max_new_tokens=None,
    cache_dir=None,
    local_model=None,
    text_only_res=None,
    image_only_res=None,
    mm_res=None,
    save_val_file=False,
    use_random_half=False,
    early_stopping_patience=3,
    epochs=10,
    vlm_images_to_include=10,
    learning_rates=(0.0002,),
    buffer_size=500,
    ema_alpha=0.25,
    reg_weight=1.0,
    targets_spec=None,
):
    SUPPORTED_MODELS = [
        "llama3",
        "gemma3",
        "fine_tuned_llama3",
        "fine_tuned_gemma3",
        "ilora_llama3",
        "ilora_gemma3",
    ]

    if mtype not in SUPPORTED_MODELS:
        raise ValueError(f"{mtype} is not supported. Supported models: {SUPPORTED_MODELS}")

    if prompt is None or not isinstance(prompt, str) or "{}" not in prompt:
        raise ValueError("You must pass `prompt` as a string template containing '{}' (for the transcript).")

    if text_col is None:
        raise ValueError("You must pass `text_col` (e.g. 'transcript').")

    if answer_col is None:
        raise ValueError("You must pass `answer_col` (e.g. 'targets_json').")

    if mtype == "llama3":
        from .llama3_zeroshot import run_llama3_base
        return run_llama3_base(
            train_df=train_df,
            test_df=test_df,
            test_unseen_df=test_unseen_df,
            text_col=text_col,
            target_col=target_col,
            prompt=prompt,
            answer_col=answer_col,
            train_val_seeds=train_val_seeds,
            val_size=val_size,
            results_folder=results_folder,
            max_tokens=max_tokens,
            class_labels=class_labels,
            max_new_tokens=max_new_tokens,
            cache_dir=cache_dir,
            local_model=local_model,
            text_only_res=text_only_res,
        )

    if mtype == "gemma3":
        from .gemma3_zeroshot import run_gemma3_base
        return run_gemma3_base(
            train_df=train_df,
            test_df=test_df,
            test_unseen_df=test_unseen_df,
            text_col=text_col,
            target_col=target_col,
            prompt=prompt,
            answer_col=answer_col,
            train_val_seeds=train_val_seeds,
            val_size=val_size,
            results_folder=results_folder,
            max_tokens=max_tokens,
            class_labels=class_labels,
            max_new_tokens=max_new_tokens,
            cache_dir=cache_dir,
            local_model=local_model,
            text_only_res=text_only_res,
        )

    if mtype == "fine_tuned_llama3":
        from .llama3_finetune import run_fine_tuned_llama3
        return run_fine_tuned_llama3(
            train_df=train_df,
            test_df=test_df,
            test_unseen_df=test_unseen_df,
            text_col=text_col,
            target_col=target_col,
            prompt=prompt,
            answer_col=answer_col,
            train_val_seeds=train_val_seeds,
            val_size=val_size,
            results_folder=results_folder,
            model_dir=model_dir,
            max_tokens=max_tokens,
            batch_size=batch_size,
            class_labels=class_labels,
            max_new_tokens=max_new_tokens,
            cache_dir=cache_dir,
            local_model=local_model,
            text_only_res=text_only_res,
            early_stopping_patience=early_stopping_patience,
            epochs=epochs,
            learning_rates=learning_rates,
        )

    if mtype == "fine_tuned_gemma3":
        from .gemma3_finetune import run_fine_tuned_gemma3
        return run_fine_tuned_gemma3(
            train_df=train_df,
            test_df=test_df,
            test_unseen_df=test_unseen_df,
            text_col=text_col,
            target_col=target_col,
            prompt=prompt,
            answer_col=answer_col,
            id_col=id_col,
            train_val_seeds=train_val_seeds,
            val_size=val_size,
            results_folder=results_folder,
            model_dir=model_dir,
            max_tokens=max_tokens,
            batch_size=batch_size,
            class_labels=class_labels,
            max_new_tokens=max_new_tokens,
            cache_dir=cache_dir,
            local_model=local_model,
            text_only_res=text_only_res,
            image_folder=image_folder,
            vlm_images_to_include=vlm_images_to_include,
            early_stopping_patience=early_stopping_patience,
            epochs=epochs,
            learning_rates=learning_rates,
            targets_spec=targets_spec,
        )

    if mtype == "ilora_llama3":
        from .llama3_ilora_finetune import run_fine_tuned_llama3_ilora
        return run_fine_tuned_llama3_ilora(
            train_df=train_df,
            test_df=test_df,
            test_unseen_df=test_unseen_df,
            text_col=text_col,
            target_col=target_col,
            prompt=prompt,
            answer_col=answer_col,
            train_val_seeds=train_val_seeds,
            val_size=val_size,
            results_folder=results_folder,
            model_dir=model_dir,
            max_tokens=max_tokens,
            batch_size=batch_size,
            class_labels=class_labels,
            max_new_tokens=max_new_tokens,
            cache_dir=cache_dir,
            local_model=local_model,
            text_only_res=text_only_res,
            early_stopping_patience=early_stopping_patience,
            epochs=epochs,
            learning_rates=learning_rates,
            buffer_size=buffer_size,
            ema_alpha=ema_alpha,
            reg_weight=reg_weight,
        )

    if mtype == "ilora_gemma3":
        from .gemma3_ilora_finetune import run_fine_tuned_gemma3_ilora
        return run_fine_tuned_gemma3_ilora(
            train_df=train_df,
            test_df=test_df,
            test_unseen_df=test_unseen_df,
            text_col=text_col,
            target_col=target_col,
            prompt=prompt,
            answer_col=answer_col,
            train_val_seeds=train_val_seeds,
            val_size=val_size,
            results_folder=results_folder,
            model_dir=model_dir,
            max_tokens=max_tokens,
            batch_size=batch_size,
            class_labels=class_labels,
            max_new_tokens=max_new_tokens,
            cache_dir=cache_dir,
            local_model=local_model,
            text_only_res=text_only_res,
            early_stopping_patience=early_stopping_patience,
            epochs=epochs,
            learning_rates=learning_rates,
            buffer_size=buffer_size,
            ema_alpha=ema_alpha,
            reg_weight=reg_weight,
        )

    raise RuntimeError("Unreachable: model dispatch fell through.")


# ==============================================================================
# IDEFICS2 DATA COLLATOR (multimodal SFT)
# ==============================================================================
class MyDataCollator:
    """
    Multimodal SFT collator.

    Expects dataset rows like:
      {"image": "/path/to.jpg", "query": "...", "answers": ["..."]}

    Supports:
      - Idefics2 / Idefics3 style processors (often have "<image>" token)
      - Gemma3 vision style (image token id lives in model.config.image_token_index)

    Notes:
      - By default masks PAD tokens and IMAGE tokens in labels with -100.
      - If you need legacy behavior (pad -> image_token_id), set legacy_pad_to_image_token=True.
    """

    def __init__(
        self,
        processor,
        model=None,
        image_token_id=None,
        legacy_pad_to_image_token=False,
        strict_image_token=False,
    ):
        self.processor = processor
        self.model = model
        self.legacy_pad_to_image_token = legacy_pad_to_image_token
        self.strict_image_token = strict_image_token

        self.tokenizer = getattr(processor, "tokenizer", processor)

        # Resolve image token id robustly (works across model families)
        self.image_token_id = image_token_id
        if self.image_token_id is None:
            self.image_token_id = self._infer_image_token_id()

        if self.strict_image_token and self.image_token_id is None:
            raise ValueError(
                "Could not infer image_token_id. Pass model=... or image_token_id=... explicitly."
            )

    def _infer_image_token_id(self):
        tok = self.tokenizer

        # 1) Gemma3 vision: image token id is typically here
        if self.model is not None:
            cfg = getattr(self.model, "config", None)
            if cfg is not None:
                v = getattr(cfg, "image_token_index", None)
                if isinstance(v, int):
                    return v

        # 2) Some processors/tokenizers expose this directly
        v = getattr(tok, "image_token_id", None)
        if isinstance(v, int):
            return v

        # 3) Some store a string token name in special_tokens_map(_extended)
        for attr in ("special_tokens_map_extended", "special_tokens_map"):
            m = getattr(tok, attr, None)
            if isinstance(m, dict):
                # try common keys
                for key in ("image_token", "img_token", "image"):
                    if key in m:
                        token_str = m[key]
                        try:
                            tid = tok.convert_tokens_to_ids(token_str)
                            if isinstance(tid, int) and tid != tok.unk_token_id:
                                return tid
                        except Exception:
                            pass

        # 4) Idefics2/3 commonly uses "<image>" in additional_special_tokens
        add = getattr(tok, "additional_special_tokens", None)
        if isinstance(add, (list, tuple)) and "<image>" in add:
            try:
                tid = tok.convert_tokens_to_ids("<image>")
                if isinstance(tid, int) and tid != tok.unk_token_id:
                    return tid
            except Exception:
                pass

        # 5) last resort: try convert directly
        try:
            tid = tok.convert_tokens_to_ids("<image>")
            if isinstance(tid, int) and tid != tok.unk_token_id:
                return tid
        except Exception:
            pass

        return None

    def __call__(self, examples):
        texts = []
        images = []
    
        for ex in examples:
            image_path = ex["image"]
            if not os.path.exists(image_path):
                continue
    
            try:
                img = Image.open(image_path).convert("RGB")
            except Exception:
                continue
    
            question = ex["query"]
            answer = random.choice(ex["answers"])
    
            messages = [
                {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question}]},
                {"role": "assistant", "content": [{"type": "text", "text": answer}]},
            ]
    
            text = self.processor.apply_chat_template(messages, add_generation_prompt=False)
            texts.append(text.strip())
            images.append([img])  # keep this; we'll flatten only if needed
    
        if not texts:
            raise ValueError("No valid examples to collate (bad/missing images?).")
    
        # ✅ CHANGED PART STARTS HERE
        try:
            # Idefics-style expects List[List[Image]]
            batch = self.processor(text=texts, images=images, return_tensors="pt", padding=True)
        except Exception as e1:
            try:
                # Gemma-style often expects List[Image]
                flat_images = [img_group[0] for img_group in images]
                batch = self.processor(text=texts, images=flat_images, return_tensors="pt", padding=True)
            except Exception as e2:
                print("❌ Processor failed with nested and flat images.")
                print(f"Nested error: {repr(e1)}")
                print(f"Flat error:   {repr(e2)}")
                print("Example text[:200]:", texts[0][:200] if texts else "N/A")
                raise
        # ✅ CHANGED PART ENDS HERE
    
        # ----- labels -----
        labels = batch["input_ids"].clone()
    
        pad_id = getattr(self.tokenizer, "pad_token_id", None)
        if pad_id is not None:
            if self.legacy_pad_to_image_token and self.image_token_id is not None:
                labels[labels == pad_id] = self.image_token_id
            else:
                labels[labels == pad_id] = -100
    
        if self.image_token_id is not None:
            labels[labels == self.image_token_id] = -100
    
        batch["labels"] = labels
        return batch


# ==============================================================================
# MISC SMALL HELPERS
# ==============================================================================
def test_function():
    print("This is a test. utils.py file loaded!")

def create_result_filename(target_col, mtype, learning_rate, seed):
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"cv_performances_{mtype}_{target_col}_{learning_rate}_seed{seed}_{timestamp}.csv"


def create_model_dirname(target_col, mtype, learning_rate, val_seed, timestamp):
    # timestamp is accepted to keep your signature; not used in name unless you want it
    return f"model_{mtype}_{target_col}_{learning_rate}_seed{val_seed}"


def create_timing_filename(target_col, mtype):
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"timing_{mtype}_{target_col}_{timestamp}.txt"


def log_start_time(filename):
    start_time = datetime.datetime.now()
    with open(filename, "w") as f:
        f.write(f"Start Time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    return start_time


def log_end_time(filename, start_time):
    end_time = datetime.datetime.now()
    duration = end_time - start_time
    with open(filename, "a") as f:
        f.write(f"End Time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total Duration: {duration}\n")


def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


# ==============================================================================
# QUANT + PEFT CONFIGS
# ==============================================================================

def set_quant_and_peft_config_llama(compute_dtype):
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=False,
    )

    peft_args = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.1,
        r=64,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "k_proj", "q_proj", "v_proj", "o_proj",
            "gate_proj", "down_proj", "up_proj",
        ],
    )

    return quant_config, peft_args


# You referenced these in gemma3_finetune.py.
# Implementations depend on your exact choice of target_modules.
def set_quant_and_peft_config_gemma(compute_dtype):
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=False,
    )

    # NOTE: target_modules for Gemma can differ. These are common for decoder LMs.
    peft_args = LoraConfig(
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

    return quant_config, peft_args


def set_training_params_llama(seed, batch_size, learning_rate):
    return TrainingArguments(
        output_dir="./results",
        num_train_epochs=1,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=1,
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
        seed=seed,
    )


def set_training_params_gemma(
    seed,
    batch_size,
    learning_rate,
    group_by_length=True,
    grad_accum_steps=4,
):
    """
    Gemma-specific TrainingArguments.

    Notes:
      - We explicitly set gradient_accumulation_steps so you can keep
        per-device batch_size small (e.g., 1) and still get an effective
        batch size >1 without OOM.
    """
    args = set_training_params_llama(seed, batch_size, learning_rate)
    args.group_by_length = group_by_length
    args.gradient_accumulation_steps = grad_accum_steps
    return args


# ==============================================================================
# DATASET BUILDING (TEXT SFT)
# ==============================================================================
def build_split_data(
    df,
    tokenizer,
    max_tokens,
    text_col,
    first_prompt_col,
    second_prompt_col,
    target_col,
    task_col,
    id_col=None,
):
    prompts = []
    texts = []
    task_names = []
    class_labels_list = []
    labels = []
    followup_prompts = []
    video_ids = []
    instructions_plain = []

    for _, row in df.iterrows():
        text = row[text_col]
        if pd.isna(text):
            continue

        prompt_template = row[first_prompt_col]
        followup_prompt = row[second_prompt_col]
        task_name = row[task_col]

        raw_labels = row["class_labels"]
        cls = json.loads(raw_labels) if isinstance(raw_labels, str) else raw_labels

        tokens = tokenizer.tokenize(text)[:max_tokens]
        shortened_text = tokenizer.convert_tokens_to_string(tokens)

        instruction = prompt_template.format(shortened_text)

        instructions_plain.append(instruction)

        if id_col is not None:
            video_ids.append(row[id_col])
        else:
            video_ids.append(None)

        if hasattr(tokenizer, "apply_chat_template"):
            messages = [{"role": "user", "content": instruction}]
            formatted_prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            formatted_prompt = f"<|user|>{instruction}<|assistant|>"

        label = row[target_col]
        if len(cls) > 2:
            label = str(label).lower()

        texts.append(shortened_text)
        prompts.append(formatted_prompt)
        task_names.append(task_name)
        class_labels_list.append(cls)
        followup_prompts.append(followup_prompt)
        labels.append(label)

    return (
        prompts,
        texts,
        task_names,
        class_labels_list,
        labels,
        followup_prompts,
        video_ids,
        instructions_plain,
    )

def build_split_data_json(
    df,
    tokenizer,
    max_tokens,
    text_col,
    prompt,          # string template containing '{}' once
    answer_col,      # e.g. 'targets_json'
    id_col=None,
):
    """
    New split builder for the *bundled multi-target JSON* task.

    Each row becomes:
      - instruction: prompt with transcript inserted
      - formatted_prompt: chat-templated prompt with generation prompt added
      - label: the gold JSON string (answer_col)

    Returns the same 8-tuple shape as old build_split_data, but:
      - task_names is a constant placeholder (e.g. 'ALL_TARGETS')
      - class_labels_list is placeholder (None)
      - followup_prompts is placeholder (None)
    """

    prompts = []
    texts = []
    task_names = []
    class_labels_list = []
    labels = []
    followup_prompts = []
    video_ids = []
    instructions_plain = []

    for _, row in df.iterrows():
        raw_text = row[text_col]
        if pd.isna(raw_text):
            continue

        # supervised answer is already a JSON string
        answer = row[answer_col]
        if pd.isna(answer):
            continue

        # truncate transcript
        tokens = tokenizer.tokenize(str(raw_text))[:max_tokens]
        truncated_text = tokenizer.convert_tokens_to_string(tokens)

        # IMPORTANT: do NOT use .format() because prompt contains JSON braces
        instruction = insert_text_once(prompt, truncated_text)
        instructions_plain.append(instruction)

        # id/video id
        video_ids.append(row[id_col] if id_col is not None else None)

        # prompt for generation
        if hasattr(tokenizer, "apply_chat_template"):
            messages = [{"role": "user", "content": instruction}]
            formatted_prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            formatted_prompt = f"<|user|>{instruction}<|assistant|>"

        prompts.append(formatted_prompt)
        texts.append(truncated_text)

        # placeholders to keep old 8-tuple shape
        task_names.append("ALL_TARGETS")
        class_labels_list.append(None)
        followup_prompts.append(None)

        # gold label = json string
        labels.append(str(answer))

    return (
        prompts,
        texts,
        task_names,
        class_labels_list,
        labels,
        followup_prompts,
        video_ids,
        instructions_plain,
    )

def insert_text_once(prompt: str, text: str) -> str:
    """
    Safely insert `text` into `prompt` at the first '{}' occurrence,
    WITHOUT triggering Python .format() parsing (so JSON braces in the prompt are safe).
    """
    if "{}" not in prompt:
        raise ValueError("Prompt template must contain '{}' exactly once to insert the text.")
    pre, post = prompt.split("{}", 1)
    return pre + text + post

def _token_len(text, tokenizer):
    """Token count of a raw string (no special-token additions)."""
    return len(tokenizer(text, add_special_tokens=False)["input_ids"])


def _build_chat_text(tokenizer, instruction, answer=None):
    """Build chat-formatted string.  answer=None → user turn only."""
    if hasattr(tokenizer, "apply_chat_template"):
        if answer is not None:
            messages = [
                {"role": "user", "content": instruction},
                {"role": "assistant", "content": answer},
            ]
        else:
            messages = [{"role": "user", "content": instruction}]
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False,
        )
    else:
        if answer is not None:
            return f"<|user|>{instruction}<|assistant|>{answer}<|end_of_text|>"
        return f"<|user|>{instruction}"


def build_sft_dataset(
    df,
    tokenizer,
    max_tokens,
    text_col,
    prompt,
    answer_col,
    id_col=None,
    image_folder=None,
    vlm_images_to_include=10,
    extract_number_fn=None,
    validate_images=True,
    skip_if_no_images=True,
    allowed_exts=(".jpg", ".jpeg", ".png", ".webp"),
    targets_spec=None,
):
    if prompt is None or not isinstance(prompt, str) or "{}" not in prompt:
        raise ValueError("`prompt` must be a string template containing '{}'.")

    multimodal = image_folder is not None

    if multimodal:
        if id_col is None:
            raise ValueError("In multimodal mode you must provide id_col (video id column).")
        if extract_number_fn is None:
            extract_number_fn = extract_number

    # -------------------------
    # TEXT-ONLY PATH
    # -------------------------
    if not multimodal:
        input_prompts = []
        _n_debug_printed = 0
        max_seq = max_tokens if max_tokens is not None else 4096

        for row_i, (_, row) in enumerate(df.iterrows()):
            input_text = row[text_col]
            if pd.isna(input_text):
                continue

            answer = row[answer_col]
            if pd.isna(answer):
                continue

            # --- 1) Convert answer FIRST (needed for budget calculation) ---
            if targets_spec is not None:
                try:
                    answer_for_sft = targets_json_to_slot_json(str(answer), targets_spec)
                except Exception as e:
                    print(f"[WARN] targets_json_to_slot_json failed: {e}. Falling back to raw answer.")
                    answer_for_sft = str(answer)
            else:
                answer_for_sft = str(answer)

            # --- 2) Dynamic transcript budgeting ---
            # Build a full chat with EMPTY transcript to measure the fixed
            # overhead (prompt template + chat wrappers + assistant answer).
            # The remaining token budget goes to the transcript.
            instruction_empty = insert_text_once(prompt, "")
            full_text_empty = _build_chat_text(tokenizer, instruction_empty, answer_for_sft)
            len_empty = _token_len(full_text_empty, tokenizer)

            safety_margin = 48
            transcript_budget = max_seq - len_empty - safety_margin
            if transcript_budget < 0:
                transcript_budget = 0

            # --- 3) Truncate transcript to budget ---
            all_transcript_tokens = tokenizer.tokenize(str(input_text))
            truncated_tokens = all_transcript_tokens[:transcript_budget]
            truncated_text = tokenizer.convert_tokens_to_string(truncated_tokens)

            # --- 4) Build final full_text ---
            instruction = insert_text_once(prompt, truncated_text)
            full_text = _build_chat_text(tokenizer, instruction, answer_for_sft)

            # --- 5) Verify length; shrink transcript if still over budget ---
            final_len = _token_len(full_text, tokenizer)
            shrink_iters = 0
            while final_len > max_seq and len(truncated_tokens) > 0 and shrink_iters < 5:
                overshoot = final_len - max_seq
                cut = max(overshoot + 16, 64)
                if cut >= len(truncated_tokens):
                    truncated_tokens = []
                else:
                    truncated_tokens = truncated_tokens[:-cut]
                truncated_text = tokenizer.convert_tokens_to_string(truncated_tokens)
                instruction = insert_text_once(prompt, truncated_text)
                full_text = _build_chat_text(tokenizer, instruction, answer_for_sft)
                final_len = _token_len(full_text, tokenizer)
                shrink_iters += 1

            # --- Debug prints (first 2 examples only) ---
            if DEBUG and _n_debug_printed < 2:
                if _n_debug_printed == 0:
                    print("\n" + "=" * 100)
                    print("TOKEN BUDGET DEBUG (build_sft_dataset, text-only)")
                    print("=" * 100)
                    print(f"max_seq_length: {max_seq}")
                    user_only_text = _build_chat_text(tokenizer, instruction_empty)
                    print(f"instruction_template_tokens (no transcript, user-only): "
                          f"{_token_len(user_only_text, tokenizer)}")
                    print(f"full_text_empty_tokens (no transcript + answer): {len_empty}")
                    print(f"transcript_budget (first row): {transcript_budget}")

                print(f"\n--- row_i={row_i} ---")
                print(f"  raw_transcript_tokens: {len(all_transcript_tokens)}")
                print(f"  truncated_transcript_tokens: {len(truncated_tokens)}")
                print(f"  answer_for_sft_tokens: {_token_len(answer_for_sft, tokenizer)}")
                print(f"  final_full_text_tokens: {final_len}  (max_seq={max_seq})")
                print(f"  fits_in_max_seq: {final_len <= max_seq}")
                tail = full_text[-800:]
                print(f"  full_text tail (last 800 chars):")
                print(f"  {tail}")
                print(f"  slot_tokens_in_tail: {'<@' in tail}")

                _n_debug_printed += 1
                if _n_debug_printed >= 2:
                    print("=" * 100 + "\n")

            input_prompts.append(full_text)

        return Dataset.from_dict({"text": input_prompts})

    # -------------------------
    # MULTIMODAL PATH
    # -------------------------
    mm_rows = []
    missing_img_dirs = 0
    no_valid_frames = 0
    skipped_bad_images = 0

    for i, (_, row) in enumerate(df.iterrows()):
        if i % 200 == 0:
            print(f"[build_sft_dataset] processed rows: {i}")

        video_id = row[id_col]
        input_text = row[text_col]
        if pd.isna(input_text):
            continue

        answer = row[answer_col]

        # Same reasoning as in the text-only branch: keep some budget for the
        # assistant answer so it survives SFT truncation.
        if max_tokens is not None and max_tokens > 512:
            text_budget = max_tokens - 512
        else:
            text_budget = max_tokens

        tokens = tokenizer.tokenize(str(input_text))[:text_budget]
        truncated_text = tokenizer.convert_tokens_to_string(tokens)

        query = insert_text_once(prompt, truncated_text)

        video_image_path = os.path.join(image_folder, str(video_id))
        if not os.path.exists(video_image_path):
            missing_img_dirs += 1
            if not skip_if_no_images:
                pass
            continue

        files = [f for f in os.listdir(video_image_path) if f.lower().endswith(allowed_exts)]

        numbered = []
        for f in files:
            n = extract_number_fn(f)
            if n is None:
                continue
            if 0 <= n < vlm_images_to_include:
                numbered.append((n, f))

        numbered.sort(key=lambda x: x[0])
        filtered_files = [f for _, f in numbered]

        if not filtered_files:
            no_valid_frames += 1
            continue

        for f in filtered_files:
            image_path = os.path.join(video_image_path, f)

            if validate_images:
                try:
                    with Image.open(image_path) as img:
                        img = img.convert("RGB")
                        arr = np.array(img)
                        if arr.ndim != 3 or arr.shape[2] != 3:
                            skipped_bad_images += 1
                            continue
                except Exception:
                    skipped_bad_images += 1
                    continue

            mm_rows.append(
                {"id": str(video_id), "image": image_path, "query": query, "answers": [str(answer)]}
            )

    print(
        f"[build_sft_dataset] multimodal examples: {len(mm_rows)} | "
        f"missing_img_dirs: {missing_img_dirs} | "
        f"no_valid_frames: {no_valid_frames} | "
        f"skipped_bad_images: {skipped_bad_images}"
    )

    if not mm_rows:
        raise ValueError(
            "No multimodal training examples were created. "
            "Check image_folder/id_col, directory layout, and frame filenames."
        )

    return Dataset.from_list(mm_rows)

def preprocess_function(example, tokenizer, max_tokens):
    # single canonical preprocess function (no global tokenizer)
    return tokenizer(
        example["text"],
        truncation=True,
        padding="max_length",
        max_length=max_tokens,
    )


# ==============================================================================
# FOLLOW-UP PROMPT FOR LLAMA STYLE (kept for backward-compat)
# ==============================================================================
def handle_follow_up_prompt_llama_standard(
    text,
    llm_follow_up_prompt,
    max_new_tokens,
    class_labels,
    mtype,
    pipeline=None,
    tokenizer=None,
    trainer=None,
    target_col=None,
):
    """
    This function is retained mainly for older inference code paths.
    Your new eval_utils.py does follow-up more cleanly; prefer that.
    """
    follow_up_prompt = llm_follow_up_prompt.format(text)

    if pipeline is not None and (mtype == "llama2" or mtype == "llama3"):
        messages = [{"role": "user", "content": follow_up_prompt}]
        final_prompts = pipeline.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        terminators = [
            pipeline.tokenizer.eos_token_id,
            pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>"),
        ]

        outputs = pipeline(
            final_prompts,
            max_length=1300,
            max_new_tokens=max_new_tokens,
            eos_token_id=terminators,
            do_sample=False,
            temperature=0.6,
            top_p=0.9,
        )
        result = outputs[0]["generated_text"]

    else:
        if tokenizer is None or trainer is None:
            raise ValueError("tokenizer and trainer must be provided when pipeline is None.")

        if mtype == "fine_tuned_llama2":
            prompt = f"<s>[INST]{follow_up_prompt}[/INST]"
        else:
            prompt = f"<|user|>{follow_up_prompt}<|assistant|>"

        input_ids = tokenizer.encode(prompt, return_tensors="pt")
        input_ids = input_ids.to("cuda" if torch.cuda.is_available() else "cpu")

        trainer.model.eval()
        with torch.no_grad():
            generated_ids = trainer.model.generate(
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                num_return_sequences=1,
                do_sample=False,
            )

        generated_ids = generated_ids.detach().cpu()
        result = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

    # Parse / clean
    if mtype in ("llama2", "fine_tuned_llama2"):
        parts = result.split("[/INST]")
        first = parts[1] if len(parts) > 1 else result
    else:
        if "<|assistant|>" in result:
            first = result.split("<|assistant|>")[-1]
        else:
            lines = [ln for ln in result.splitlines() if ln.strip()]
            first = lines[-1] if lines else ""

    answer_follow = first.rstrip(".,").lower().strip().strip("\"'").replace("'", "").replace('"', "")

    if target_col in ("typology", "ideology"):
        for class_lab in class_labels:
            if class_lab in answer_follow:
                answer_follow = class_lab
                break

    return answer_follow if answer_follow in class_labels else "na"


# ==============================================================================
# MULTIMODAL FOLLOW-UP (IDEFICS2)
# ==============================================================================
def handle_follow_up_prompt(
    processor,
    model,
    messages,
    video_image_path,
    current_limit,
    text,
    device,
    llm_follow_up_prompt,
    images=None,
    is_text_only=False,
    is_images_only=False,
):
    """
    Return a short follow-up answer for Idefics2-type models.
    """
    if not is_text_only and not is_images_only:
        reply_messages = [{"role": "user", "content": []}]
        reply_images = []

        k = 0
        for file in os.listdir(video_image_path):
            if k == current_limit:
                break
            image_path = f"{video_image_path}/{file}"
            img = load_image(image_path)
            reply_images.append(img)
            reply_messages[0]["content"].append({"type": "image"})
            k += 1

        reply_format_prompt = llm_follow_up_prompt.format(text)
        reply_messages[0]["content"].append({"type": "text", "text": reply_format_prompt})

        idefics_prompt = processor.apply_chat_template(reply_messages, add_generation_prompt=True)
        inputs = processor(text=idefics_prompt, images=reply_images, return_tensors="pt")

    elif (not is_text_only) and is_images_only:
        reply_messages = [{"role": "user", "content": []}]
        reply_images = []

        for file in os.listdir(video_image_path):
            if len(reply_images) == current_limit:
                break
            image_path = f"{video_image_path}/{file}"
            img = load_image(image_path)
            reply_images.append(img)
            reply_messages[0]["content"].append({"type": "image"})

        reply_messages[0]["content"].append({"type": "text", "text": llm_follow_up_prompt})
        idefics_prompt = processor.apply_chat_template(reply_messages, add_generation_prompt=True)
        inputs = processor(text=idefics_prompt, images=reply_images, return_tensors="pt")

    else:
        reply_messages = [{"role": "user", "content": []}]
        reply_format_prompt = llm_follow_up_prompt.format(text)
        reply_messages[0]["content"].append({"type": "text", "text": reply_format_prompt})

        idefics_prompt = processor.apply_chat_template(reply_messages, add_generation_prompt=True)
        inputs = processor(text=idefics_prompt, return_tensors="pt")

    inputs = {k: v.to(device) for k, v in inputs.items()}

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    generated_ids = model.generate(**inputs, max_new_tokens=3, do_sample=False)
    generated_ids = generated_ids.detach().cpu()
    generated_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)

    parts = generated_texts[0].split("Assistant:")
    reply_answer = parts[-1].rstrip(".,").strip().lower()
    return reply_answer


# ==============================================================================
# TEXT CLEANING
# ==============================================================================
def cleanse_text(input_text: str) -> str:
    removed_chars = []
    cleansed = []

    for ch in str(input_text):
        if ord(ch) < 128:
            cleansed.append(ch)
        else:
            removed_chars.append(ch)

    if removed_chars:
        print(f"Removed non-ASCII characters: {set(removed_chars)}")

    return "".join(cleansed)


# ==============================================================================
# VLM TRAIN DATA PREP
# ==============================================================================
def process_train_data(
    train_rows: pd.DataFrame,
    id_col: str,
    text_col: str,
    llm_answer_col: str,
    target_col: str,
    class_labels: list,
    max_tokens: int,
    prompt: str,
    image_folder: str,
    vlm_images_to_include: int,
    tokenizer,
    extract_number,
) -> list:
    train_data = []

    for i, (_, row) in enumerate(train_rows.iterrows()):
        if i % 100 == 0:
            print(f"Preprocessing video {i}")

        video_id = row[id_col]
        text = row[text_col]
        if pd.isna(text):
            continue

        llm_answer = row[llm_answer_col]

        # label currently unused here, but keep if you later add it to item
        if len(class_labels) == 2:
            _ = int(row[target_col])
        else:
            _ = row[target_col]

        tokenized = tokenizer(text, truncation=True, max_length=max_tokens, return_tensors="pt")
        truncated_text = tokenizer.decode(tokenized["input_ids"][0], skip_special_tokens=True)

        query = prompt.format(truncated_text)

        video_image_path = os.path.join(image_folder, str(video_id))
        if not os.path.exists(video_image_path):
            continue

        files = os.listdir(video_image_path)
        filtered_files = [f for f in files if extract_number(f) in range(0, vlm_images_to_include)]
        filtered_files.sort(key=extract_number)

        for file in filtered_files:
            image_path = os.path.join(video_image_path, file)
            train_data.append(
                {
                    "id": video_id,
                    "image": image_path,
                    "answers": [llm_answer],
                    "query": query,
                }
            )

    print(f"Amount of training data: {len(train_data)}")
    return train_data

def rebalance_binary_to_fixed_n(
    df: pd.DataFrame,
    label_col: str = "target",
    total_n: int = 1000,
    p_minority: float = 0.5,
    random_state: int = 42,
    allow_majority_oversample: bool = False,
) -> pd.DataFrame:
    """
    Rebalance a *binary* labeled dataframe to exactly `total_n` rows, with
    `p_minority` fraction of the minority class.

    - Minority class is inferred from value_counts (smallest count).
    - Minority is sampled WITH replacement if needed.
    - Majority is sampled WITHOUT replacement by default.
    - If majority is too small (rare, but possible after filtering),
      set allow_majority_oversample=True to sample it with replacement too.
    """
    if df.empty:
        return df

    vc = df[label_col].value_counts(dropna=False)
    if len(vc) != 2:
        # If this happens for hateful/misinformation, something is off.
        # For multiclass tasks, don't use this function.
        raise ValueError(f"Expected binary labels in {label_col}, got counts:\n{vc}")

    minority_label = vc.idxmin()
    majority_label = vc.idxmax()

    df_min = df[df[label_col] == minority_label]
    df_maj = df[df[label_col] == majority_label]

    n_min = int(round(total_n * p_minority))
    n_maj = total_n - n_min

    # sample minority (with replacement if needed)
    df_min_s = df_min.sample(
        n=n_min,
        replace=(len(df_min) < n_min),
        random_state=random_state,
    )

    # sample majority (without replacement by default)
    replace_maj = False
    if len(df_maj) < n_maj:
        if not allow_majority_oversample:
            # fallback: take all majority, and top up with more minority
            # (keeps total_n, but p_minority may drift upward)
            df_maj_s = df_maj.sample(n=len(df_maj), replace=False, random_state=random_state)
            missing = n_maj - len(df_maj_s)
            df_min_extra = df_min.sample(
                n=missing,
                replace=True,
                random_state=random_state + 1,
            )
            out = pd.concat([df_min_s, df_maj_s, df_min_extra], ignore_index=True)
            return out.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
        else:
            replace_maj = True

    df_maj_s = df_maj.sample(
        n=n_maj,
        replace=replace_maj,
        random_state=random_state,
    )

    out = pd.concat([df_min_s, df_maj_s], ignore_index=True)
    out = out.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    return out


# ==============================================================================
# CLIP UTILS (optional / only if you actually use it)
# ==============================================================================
def preprocess_pairs_clip(image, text, model, preprocess, device):
    """
    Requires `import clip` if you use it. Kept as-is.
    """
    image = preprocess(image).unsqueeze(0).to(device)

    # If you actually use this, you must `import clip` in this file or pass tokenizer
    import clip  # local import to avoid hard dependency at import time

    text_tokens = clip.tokenize([text], truncate=True).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text_tokens)

    return torch.cat((image_features, text_features), dim=1)


# ==============================================================================
# VIDEO-LEVEL AGGREGATION
# ==============================================================================
def get_video_level_predictions(video_ids, predictions, threshold):
    unique_video_ids = []
    for vid in video_ids:
        if vid not in unique_video_ids:
            unique_video_ids.append(vid)

    prediction_dict = {}
    for video_id, pred in zip(video_ids, predictions):
        prediction_dict.setdefault(video_id, []).append(pred)

    final_predictions = []
    for video_id in unique_video_ids:
        preds = prediction_dict[video_id]
        count_ones = preds.count(1)
        total = len(preds)
        final_predictions.append(1 if count_ones > (total * threshold) else 0)

    return final_predictions, unique_video_ids


def get_video_level_predictions_multiclass(video_ids, predictions):
    unique_video_ids = []
    for vid in video_ids:
        if vid not in unique_video_ids:
            unique_video_ids.append(vid)

    prediction_dict = {}
    for video_id, pred in zip(video_ids, predictions):
        prediction_dict.setdefault(video_id, []).append(pred)

    final_predictions = []
    for video_id in unique_video_ids:
        preds = prediction_dict[video_id]
        most_common = Counter(preds).most_common(1)[0][0]
        final_predictions.append(most_common)

    return final_predictions, unique_video_ids


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, image_paths, texts, transform=None):
        self.image_paths = image_paths
        self.texts = texts
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        text = self.texts[idx]
        return image, text


# ==============================================================================
# MULTI-TASK SPLITS
# ==============================================================================
def build_multi_task_splits(train_sets, test_sets, n_tasks=None, task_order=None):
    all_tasks = list(train_sets.keys())

    if task_order is not None:
        ordered_tasks = [t for t in task_order if t in train_sets]
        for t in all_tasks:
            if t not in ordered_tasks:
                ordered_tasks.append(t)
    else:
        ordered_tasks = all_tasks

    if n_tasks is None or n_tasks >= len(ordered_tasks):
        seen_tasks = ordered_tasks
    else:
        seen_tasks = ordered_tasks[:n_tasks]

    unseen_tasks = [t for t in all_tasks if t not in seen_tasks]

    train_seen = pd.concat([train_sets[t] for t in seen_tasks], ignore_index=True)
    test_seen = pd.concat([test_sets[t] for t in seen_tasks], ignore_index=True)

    if unseen_tasks:
        test_unseen = pd.concat([test_sets[t] for t in unseen_tasks], ignore_index=True)
    else:
        test_unseen = pd.DataFrame()

    return {
        "train_seen": train_seen,
        "test_seen": test_seen,
        "test_unseen": test_unseen,
        "seen_tasks": seen_tasks,
        "unseen_tasks": unseen_tasks,
    }


# ==============================================================================
# FILENAME NUMBER EXTRACTION + DICT PARSING
# ==============================================================================
def extract_number(filename):
    match = re.search(r"-(\d+)\.jpg$", filename)
    return int(match.group(1)) if match else None


def preprocess_dictionary_string(dict_string):
    dict_string = dict_string.strip()
    dict_string = re.sub(r"(\w+):", r'"\1":', dict_string)
    dict_string = re.sub(r": (Yes|No)([,\}])", r': "\1"\2', dict_string)
    return dict_string


def extract_dictionary_from_string(s):
    start = s.find("{")
    end = s.rfind("}") + 1
    dict_str = s[start:end]

    try:
        return ast.literal_eval(dict_str)
    except (SyntaxError, ValueError) as e:
        print("Error parsing dictionary string:", e)
        return {}


# ==============================================================================
# GPU MEMORY PRINT (CUDA / NVML only)
# ==============================================================================
def print_gpu_memory(handle, pynvml):
    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)

    print(
        f"GPU Memory Used: {mem_info.used / 1024**2:.2f} MB / "
        f"{mem_info.total / 1024**2:.2f} MB"
    )
    print(f"GPU Utilization: {utilization.gpu}%")



def debug_print_multitask_sft_bundle(
    *,
    is_mm: bool,
    train_rows,
    val_rows,
    dataset,
    val_dataset,
    text_col: str,
    answer_col: str,
    id_col: str | None = None,
    # from build_split_data_json:
    val_prompts=None,
    val_labels=None,
    val_video_ids=None,
    val_instructions_plain=None,
    # printing controls
    n_df_rows: int = 2,
    n_ds_rows: int = 2,
    max_text_chars: int = 400,
    max_answer_chars: int = 400,
    max_sft_chars: int = 1200,
    max_prompt_chars: int = 1200,
    max_label_chars: int = 800,
    max_instr_chars: int = 600,
):
    """
    Debug/inspection printing for your multi-target SFT setup (JSON supervision).

    - Shows a couple raw dataframe rows (text + answer JSON)
    - Shows a couple dataset items (what SFTTrainer actually consumes)
    - Shows a couple val inference prompts + gold labels (for later eval)

    Safe for text-only and multimodal datasets.
    """

    def _short(s: str, n: int) -> str:
        return str(s)[:n].replace("\n", "\\n")

    print("\n" + "=" * 80)
    print("DATA INSPECTION")
    print("=" * 80)

    print(f"is_mm: {is_mm}")
    print(f"train_rows: {len(train_rows)} | val_rows: {len(val_rows)}")
    print(f"train_dataset size: {len(dataset)} | val_dataset size: {len(val_dataset)}")
    print("\nTrain dataset columns:", getattr(dataset, "column_names", "N/A"))
    print("Val   dataset columns:", getattr(val_dataset, "column_names", "N/A"))

    # ---- raw df preview ----
    def _preview_df_rows(df_, name, n=2):
        print("\n" + "-" * 80)
        print(f"{name}: raw df preview ({n} rows)")
        print("-" * 80)

        for i, row in df_.head(n).iterrows():
            vid = None
            if id_col is not None and id_col in df_.columns:
                vid = row[id_col]

            txt = row[text_col] if text_col in df_.columns else None
            ans = row[answer_col] if answer_col in df_.columns else None

            print(f"[idx={i}] id={vid}")
            print(f" {text_col}[:{max_text_chars}]   =", _short(txt, max_text_chars))
            print(f" {answer_col}[:{max_answer_chars}] =", _short(ans, max_answer_chars))

    _preview_df_rows(train_rows, "TRAIN_ROWS", n=n_df_rows)
    _preview_df_rows(val_rows,   "VAL_ROWS",   n=n_df_rows)

    # ---- dataset preview ----
    def _preview_sft_samples(ds, name, n=2):
        print("\n" + "-" * 80)
        print(f"{name}: SFT dataset sample preview ({n} items)")
        print("-" * 80)

        for j in range(min(n, len(ds))):
            ex = ds[j]
            if isinstance(ex, dict) and "text" in ex:
                print(f"[{name} ds idx={j}] text[:{max_sft_chars}] = {_short(ex['text'], max_sft_chars)}")
            else:
                # multimodal
                keys = list(ex.keys()) if isinstance(ex, dict) else type(ex)
                print(f"[{name} ds idx={j}] keys = {keys}")
                if isinstance(ex, dict):
                    print(" image =", ex.get("image"))
                    print(" query[:400]  =", _short(ex.get("query", ""), 400))
                    print(" answers[:200]=", _short(ex.get("answers", ""), 200))

    _preview_sft_samples(dataset,     "TRAIN_DATASET", n=n_ds_rows)
    _preview_sft_samples(val_dataset, "VAL_DATASET",   n=n_ds_rows)

    # ---- val inference inputs preview ----
    if val_prompts is not None and val_labels is not None:
        print("\n" + "-" * 80)
        print("VAL INFERENCE INPUTS (from build_split_data_json)")
        print("-" * 80)
        print(f"val_prompts: {len(val_prompts)} | val_labels: {len(val_labels)}")

        for k in range(min(2, len(val_prompts))):
            vid = val_video_ids[k] if (val_video_ids is not None and k < len(val_video_ids)) else None
            instr = val_instructions_plain[k] if (val_instructions_plain is not None and k < len(val_instructions_plain)) else None

            print(f"[k={k}] id={vid}")
            if instr is not None:
                print(f" instruction_plain[:{max_instr_chars}] =", _short(instr, max_instr_chars))
            print(f" prompt_for_generation[:{max_prompt_chars}] =", _short(val_prompts[k], max_prompt_chars))
            print(f" gold_label_json[:{max_label_chars}] =", _short(val_labels[k], max_label_chars))

    print("=" * 80)
    print("END DATA INSPECTION")
    print("=" * 80 + "\n")


def build_slot_token_map(targets_spec: dict) -> dict:
    """
    Returns:
      token_map[target][value_str] = token_str
    Only for targets of type binary/multiclass.
    """
    token_map = {}
    for t, spec in (targets_spec or {}).items():
        if spec.get("type") not in ("binary", "multiclass"):
            continue
        allowed = spec.get("allowed", [])
        m = {}
        for v in allowed:
            v_str = str(v)
            m[v_str] = f"<@{t}={v_str}>"
        token_map[t] = m
    return token_map



def _short(s, n=400):
    return str(s)[:n].replace("\n", "\\n")

def all_slot_tokens(targets_spec: dict) -> list[str]:
    token_map = build_slot_token_map(targets_spec)
    toks = []
    for t, m in token_map.items():
        toks.extend(m.values())
    # deterministic order
    return sorted(set(toks))

def _normalize_value_for_slots(v):
    # turn numpy scalars into python scalars
    if hasattr(v, "item"):
        try:
            v = v.item()
        except Exception:
            pass

    # float 1.0 -> int 1
    if isinstance(v, float) and v.is_integer():
        return int(v)

    return v

def targets_json_to_slot_json(answer_json_str: str, targets_spec: dict) -> str:
    """
    Converts {"politics":1,"topic01":"HEALTH",...}
    into {"politics":"<@politics=1>","topic01":"<@topic01=HEALTH>",...}
    for targets of type binary/multiclass.
    Leaves other fields unchanged.
    """
    d = json.loads(answer_json_str) if isinstance(answer_json_str, str) else dict(answer_json_str)
    token_map = build_slot_token_map(targets_spec)

    out = {}
    for k, v in d.items():
        spec = (targets_spec or {}).get(k, None)

        if spec is None:
            out[k] = v
            continue

        ttype = spec.get("type")
        if ttype in ("binary", "multiclass"):
            v = _normalize_value_for_slots(v)
            if v is None:
                allowed_strs = {str(a) for a in spec.get("allowed", [])}
                if "99" in allowed_strs:
                    v = 99
                elif "-1" in allowed_strs:
                    v = -1
            v_str = "None" if v is None else str(v)
            tok = token_map.get(k, {}).get(v_str, None)
            out[k] = tok if tok is not None else None
        else:
            out[k] = v

    return json.dumps(out, ensure_ascii=False)

class SlotLossCollator:
    def __init__(self, tokenizer, targets_spec: dict):
        self.tokenizer = tokenizer
        self.targets_spec = targets_spec or {}
        self.token_map = build_slot_token_map(self.targets_spec)

        self.target_token_ids = {}
        self.target_num_classes = {}
        for t, spec in self.targets_spec.items():
            if spec.get("type") not in ("binary", "multiclass"):
                continue
            allowed = spec.get("allowed", [])
            ids = []
            for v in allowed:
                tok = self.token_map[t][str(v)]
                tid = tokenizer.convert_tokens_to_ids(tok)
                ids.append(tid)
            self.target_token_ids[t] = set(ids)
            self.target_num_classes[t] = (2 if spec.get("type") == "binary" else len(allowed))

    def __call__(self, features):
        # TRL-prepared dataset gives input_ids/attention_mask only
        input_ids = torch.tensor([f["input_ids"] for f in features], dtype=torch.long)
        attention_mask = torch.tensor([f["attention_mask"] for f in features], dtype=torch.long)

        # Create labels ourselves
        labels = input_ids.clone()

        pad_id = getattr(self.tokenizer, "pad_token_id", None)
        if pad_id is not None:
            labels[labels == pad_id] = -100

        labels_by_target = {}
        for t, ids_set in self.target_token_ids.items():
            lab_t = torch.full_like(labels, -100)
            mask = torch.zeros_like(labels, dtype=torch.bool)
            for tid in ids_set:
                mask |= (labels == tid)
            lab_t[mask] = labels[mask]
            labels_by_target[t] = lab_t

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,  # kept for convenience
            "labels_by_target": labels_by_target,
            "target_num_classes": self.target_num_classes,
        }

def debug_print_slot_token_setup(tokenizer, targets_spec, max_targets=999):
    print("\n" + "=" * 100)
    print("SLOT TOKEN SETUP DEBUG")
    print("=" * 100)

    toks = all_slot_tokens(targets_spec)
    print(f"Total slot tokens: {len(toks)}")
    if len(toks) <= 50:
        print("Slot tokens:", toks[:50])
    else:
        print("Slot tokens sample:", toks[:10], "...", toks[-10:])

    # verify token ids are not unk
    unk_id = getattr(tokenizer, "unk_token_id", None)
    bad = []
    for tok in toks:
        tid = tokenizer.convert_tokens_to_ids(tok)
        if unk_id is not None and tid == unk_id:
            bad.append(tok)

    print("Tokenizer unk_token_id:", unk_id)
    print("Bad (mapped to UNK) slot tokens:", len(bad))
    if bad:
        print("Example bad tokens:", bad[:20])

    # per target diagnostics
    token_map = build_slot_token_map(targets_spec)
    shown = 0
    for t, spec in targets_spec.items():
        if spec.get("type") not in ("binary", "multiclass"):
            continue
        shown += 1
        if shown > max_targets:
            break

        allowed = spec.get("allowed", [])
        ids = []
        for v in allowed:
            tok = token_map[t][str(v)]
            tid = tokenizer.convert_tokens_to_ids(tok)
            ids.append(tid)

        print(f"\nTarget: {t} | type={spec.get('type')} | #allowed={len(allowed)}")
        print(" allowed[:10]:", allowed[:10])
        print(" slot_tokens[:5]:", [token_map[t][str(v)] for v in allowed[:5]])
        print(" token_ids[:10]:", ids[:10])

    print("=" * 100 + "\n")

def debug_print_one_sft_string_example(
    df,
    idx,
    tokenizer,
    max_tokens,
    text_col,
    prompt,
    answer_col,
    targets_spec=None,
):
    row = df.iloc[idx]
    raw_text = row[text_col]
    raw_answer = row[answer_col]

    # truncate transcript exactly as build_sft_dataset does
    tokens = tokenizer.tokenize(str(raw_text))[:max_tokens]
    truncated_text = tokenizer.convert_tokens_to_string(tokens)

    instruction = insert_text_once(prompt, truncated_text)

    if targets_spec is not None:
        try:
            answer_slot = targets_json_to_slot_json(str(raw_answer), targets_spec)
        except Exception as e:
            answer_slot = str(raw_answer)
            print("[WARN] slot conversion failed:", repr(e))
    else:
        answer_slot = str(raw_answer)

    # build chat text like build_sft_dataset
    if hasattr(tokenizer, "apply_chat_template"):
        messages = [
            {"role": "user", "content": instruction},
            {"role": "assistant", "content": answer_slot},
        ]
        full_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    else:
        full_text = f"<|user|>{instruction}<|assistant|>{answer_slot}<|end_of_text|>"

    print("\n" + "=" * 100)
    print(f"RAW SFT STRING EXAMPLE | df_idx={idx}")
    print("=" * 100)
    print("raw_text[:500]:", _short(raw_text, 500))
    print("raw_answer_json[:800]:", _short(raw_answer, 800))
    print("answer_slot_json[:800]:", _short(answer_slot, 800))
    print("-" * 100)
    print("FULL_TEXT (first chars):")
    print(_short(full_text, DEBUG_MAX_CHARS))
    print("=" * 100 + "\n")

    return full_text

def debug_tokenize_and_locate_slot_tokens(full_text, tokenizer, targets_spec):
    enc = tokenizer(full_text, return_tensors=None, add_special_tokens=False)
    input_ids = enc["input_ids"]

    inv = {v: k for k, v in tokenizer.get_vocab().items()}  # may be large, but ok once
    toks = all_slot_tokens(targets_spec)
    tok_ids = {tok: tokenizer.convert_tokens_to_ids(tok) for tok in toks}

    print("\n" + "=" * 100)
    print("TOKENIZATION DEBUG")
    print("=" * 100)
    print("len(input_ids):", len(input_ids))

    # Count occurrences of each slot token id
    counts = {}
    for tok, tid in tok_ids.items():
        if tid is None:
            continue
        c = sum(1 for x in input_ids if x == tid)
        if c > 0:
            counts[tok] = c

    print("Slot tokens found in tokenized sequence:", len(counts))
    # print some
    for tok, c in list(counts.items())[:30]:
        print(f"  {tok} -> count {c}")

    # Show a small window around first occurrence of any slot token
    if counts:
        first_tok = next(iter(counts.keys()))
        tid = tok_ids[first_tok]
        pos = next(i for i, x in enumerate(input_ids) if x == tid)

        lo = max(0, pos - 25)
        hi = min(len(input_ids), pos + 25)
        window_ids = input_ids[lo:hi]
        window_text = tokenizer.decode(window_ids, skip_special_tokens=False)

        print("\nFirst slot token:", first_tok, "at pos", pos)
        print("Decoded window around it:")
        print(window_text)

    print("=" * 100 + "\n")

class DebugBatchCollator:
    """
    Wrap any collator and print one batch once.
    """
    def __init__(self, collator, tokenizer=None):
        self.collator = collator
        self.tokenizer = tokenizer

    def __call__(self, features):
        batch = self.collator(features)

        if DEBUG and not DEBUG_ONCE["printed"]:
            DEBUG_ONCE["printed"] = True
            print("\n" + "=" * 100)
            print("FIRST BATCH DEBUG (collator output)")
            print("=" * 100)
            print("batch keys:", list(batch.keys()))
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    print(f"  {k}: shape={tuple(v.shape)} dtype={v.dtype}")
                elif isinstance(v, dict):
                    print(f"  {k}: dict with keys={list(v.keys())[:10]} (len={len(v)})")
                else:
                    print(f"  {k}: type={type(v)}")

            if "labels" in batch and isinstance(batch["labels"], torch.Tensor):
                lab = batch["labels"]
                n = int((lab != -100).sum().item())
                print("labelled tokens (labels != -100):", n)

            # print a decoded example if possible
            if self.tokenizer is not None and "input_ids" in batch:
                ex0 = batch["input_ids"][0].tolist()
                print("\nDecoded input_ids[0] head:")
                print(_short(self.tokenizer.decode(ex0, skip_special_tokens=False), 2000))

            print("=" * 100 + "\n")

        return batch