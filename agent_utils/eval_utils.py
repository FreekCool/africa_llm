# agent_utils/eval_utils.py

import re
import os
import torch
import pandas as pd
from PIL import Image
from collections import defaultdict
from sklearn.metrics import precision_recall_fscore_support

# Matches chat tags like <|assistant|>, <|eot_id|>, etc.
_TAG_RE = re.compile(r"<\|.*?\|>")


def _normalize_text(s: str) -> str:
    """Normalize decoded model output into something comparable to labels."""
    if s is None:
        return ""
    s = str(s)

    # Remove special tags explicitly
    s = _TAG_RE.sub(" ", s)

    # Normalize whitespace (incl NBSP)
    s = s.replace("\u00A0", " ")
    s = re.sub(r"\s+", " ", s).strip()

    # Strip quotes
    s = re.sub(r"[\"'“”]", "", s)

    # Lowercase
    s = s.lower().strip()

    # Remove surrounding punctuation
    s = s.strip(" \t\r\n.,:;!?)(")

    return s


def _extract_number_dash_ext(filename: str):
    """
    Matches your training convention: something like "...-3.jpg" or "...-12.png".
    Returns int or None.
    """
    m = re.search(r"-(\d+)\.(jpg|jpeg|png|webp)$", filename.lower())
    return int(m.group(1)) if m else None


def _load_video_images(image_folder: str, video_id, max_images: int = 10):
    """
    IMPORTANT: aligned with training.

    Training included ONLY files where extract_number(...) is in range(0, vlm_images_to_include),
    and then sorted by that number.

    We do the same:
      - only numbered frames matching -<nr>.<ext>
      - only keep nr in [0, max_images-1]
      - sort by nr
    """
    video_dir = os.path.join(image_folder, str(video_id))
    if not os.path.isdir(video_dir):
        return []

    files = [f for f in os.listdir(video_dir) if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))]

    numbered = []
    for f in files:
        n = _extract_number_dash_ext(f)
        if n is None:
            continue
        if 0 <= n < max_images:
            numbered.append((n, f))

    numbered.sort(key=lambda x: x[0])

    imgs = []
    for _, f in numbered[:max_images]:
        p = os.path.join(video_dir, f)
        try:
            imgs.append(Image.open(p).convert("RGB"))
        except Exception:
            continue

    return imgs


def _is_yesno_task(class_labels) -> bool:
    if not class_labels or len(class_labels) != 2:
        return False
    norm = sorted([_normalize_text(x) for x in class_labels])
    return norm == ["no", "yes"]


def _map_to_label(raw_completion: str, class_labels):
    """
    Map raw decoded completion to a label in class_labels (normalized) or 'na'.
    Returns (mapped_label, cleaned_text).
    """
    cleaned = _normalize_text(raw_completion)

    # YES/NO tasks: find first occurrence of "yes" or "no"
    if _is_yesno_task(class_labels):
        m = re.search(r"\b(yes|no)\b", cleaned)
        if m:
            return m.group(1), cleaned
        return "na", cleaned

    # Multiclass tasks
    norm_labels = [_normalize_text(lbl) for lbl in class_labels]
    norm_labels = [lbl for lbl in norm_labels if lbl]  # drop empty

    # Exact match
    if cleaned in norm_labels:
        return cleaned, cleaned

    # Substring match (prefer longest)
    candidates = [lbl for lbl in norm_labels if lbl in cleaned]
    if candidates:
        best = max(candidates, key=len)
        return best, cleaned

    # Reverse containment (cleaned is part of label)
    candidates = [lbl for lbl in norm_labels if cleaned and cleaned in lbl]
    if candidates:
        best = max(candidates, key=len)
        return best, cleaned

    return "na", cleaned


def _generate_completion(trainer, tokenizer, device, prompt_text, max_new_tokens):
    """
    Text-only: generate and decode ONLY the completion (new tokens), not the whole prompt.
    """
    enc = tokenizer(
        prompt_text,
        return_tensors="pt",
        padding=False,
        truncation=False,
    )

    input_ids = enc["input_ids"].to(device)
    attention_mask = enc.get("attention_mask", None)
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if pad_token_id is None:
        pad_token_id = getattr(tokenizer, "eos_token_id", None)

    trainer.model.eval()
    with torch.no_grad():
        generated = trainer.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_return_sequences=1,
            pad_token_id=pad_token_id,
        )

    generated = generated.detach().cpu()

    input_len = input_ids.shape[-1]
    new_tokens = generated[0, input_len:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


def _try_format_prompt(template_or_prompt: str, text: str) -> str:
    """
    If followup_prompt is a template containing `{}`, format it with the text.
    Otherwise return it as-is.
    """
    if not isinstance(template_or_prompt, str):
        return str(template_or_prompt)

    try:
        if "{" in template_or_prompt and "}" in template_or_prompt:
            return template_or_prompt.format(text)
    except Exception:
        pass

    return template_or_prompt


def generate_completion(
    trainer,
    tokenizer=None,
    device="cuda",
    prompt_text=None,
    max_new_tokens=3,
    processor=None,
    messages=None,
    images=None,
):
    """
    Generate ONLY the completion tokens.

    Two modes:
      1) Text-only: provide tokenizer + prompt_text
      2) Multimodal: provide processor + messages (+ images optional)
    """

    # -----------------------
    # MULTIMODAL MODE
    # -----------------------
    if processor is not None and messages is not None:
        tok = getattr(processor, "tokenizer", None)
        if tok is None:
            raise ValueError("processor has no tokenizer; pass tokenizer explicitly or use a processor with tokenizer.")

        prompt = processor.apply_chat_template(messages, add_generation_prompt=True)

        # Processor variance: some accept List[List[Image]], others List[Image]
        if images is None:
            enc = processor(text=prompt, return_tensors="pt", padding=False)
        else:
            try:
                enc = processor(text=prompt, images=[images], return_tensors="pt", padding=False)
            except Exception:
                enc = processor(text=prompt, images=images, return_tensors="pt", padding=False)

        # Move all tensors to device
        enc = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in enc.items()}

        input_ids = enc["input_ids"]
        pad_token_id = getattr(tok, "pad_token_id", None) or getattr(tok, "eos_token_id", None)

        trainer.model.eval()
        with torch.no_grad():
            generated = trainer.model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_return_sequences=1,
                pad_token_id=pad_token_id,
            )

        generated = generated.detach().cpu()
        input_len = input_ids.shape[-1]
        new_tokens = generated[0, input_len:]
        return tok.decode(new_tokens, skip_special_tokens=True)

    # -----------------------
    # TEXT-ONLY MODE
    # -----------------------
    if tokenizer is None or prompt_text is None:
        raise ValueError("Text-only mode requires tokenizer + prompt_text.")

    return _generate_completion(
        trainer=trainer,
        tokenizer=tokenizer,
        device=device,
        prompt_text=prompt_text,
        max_new_tokens=max_new_tokens,
    )


def run_taskwise_inference(
    prompts,
    texts,
    task_names,
    class_labels_list,
    labels,
    followup_prompts,
    trainer,
    tokenizer,
    device,
    max_new_tokens,
    mtype,
    epoch,
    fold,
    fine_tune_type,
    split_name,
    train_loss="na",
    val_loss="na",
    elapsed_time="na",
    seed="na",
    # Optional multimodal args
    processor=None,
    image_folder=None,
    video_ids=None,
    instructions_plain=None,
    vlm_images_to_include=10,
):
    """
    Run generation + follow-up + per-task evaluation for one split.
    Prints per example (like your old workflow).
    """
    task_answers = defaultdict(list)
    task_labels_dict = defaultdict(list)
    task_na_counts = defaultdict(int)
    task_totals = defaultdict(int)
    task_class_labels = {}

    task_offlabel_first = defaultdict(list)

    # Decide once per call if we can do multimodal inference
    use_mm_global = (
        processor is not None
        and image_folder is not None
        and video_ids is not None
        and instructions_plain is not None
    )

    for j in range(len(prompts)):
        prompt_j = prompts[j]
        task_name = task_names[j]
        class_labels_j = class_labels_list[j]
        followup_prompt_j = followup_prompts[j]

        if task_name not in task_class_labels:
            task_class_labels[task_name] = class_labels_j

        # ---------- FIRST PASS ----------
        if use_mm_global:
            imgs = _load_video_images(
                image_folder=image_folder,
                video_id=video_ids[j],
                max_images=vlm_images_to_include,
            )

            mm_content = [{"type": "image"} for _ in imgs]
            mm_content.append({"type": "text", "text": instructions_plain[j]})
            messages = [{"role": "user", "content": mm_content}]

            raw_completion = generate_completion(
                trainer=trainer,
                device=device,
                processor=processor,
                messages=messages,
                images=imgs if imgs else None,
                max_new_tokens=max_new_tokens,
            )
        else:
            raw_completion = _generate_completion(
                trainer=trainer,
                tokenizer=tokenizer,
                device=device,
                prompt_text=prompt_j,
                max_new_tokens=max_new_tokens,
            )

        mapped, cleaned = _map_to_label(raw_completion, class_labels_j)

        print("\n==============================")
        print(f"[{mtype} | {split_name}] example {j+1}/{len(prompts)}")
        print(f"task: {task_name}")
        print(f"class_labels: {class_labels_j}")
        print(f"gold: {labels[j]}")
        print(f"FIRST PASS RAW: {raw_completion!r}")
        print(f"FIRST PASS CLEAN: {cleaned!r}")
        print(f"FIRST PASS MAPPED: {mapped!r}")

        # ---------- FOLLOW UP ----------
        if mapped == "na":
            print(f"ANSWER NOT IN CLASS LABELS: {class_labels_j}")
            task_offlabel_first[task_name].append(cleaned)

            followup_text = _try_format_prompt(followup_prompt_j, texts[j])

            if use_mm_global:
                imgs = _load_video_images(
                    image_folder=image_folder,
                    video_id=video_ids[j],
                    max_images=vlm_images_to_include,
                )

                mm_content = [{"type": "image"} for _ in imgs]
                mm_content.append({"type": "text", "text": followup_text})
                messages = [{"role": "user", "content": mm_content}]

                raw_completion_2 = generate_completion(
                    trainer=trainer,
                    device=device,
                    processor=processor,
                    messages=messages,
                    images=imgs if imgs else None,
                    max_new_tokens=max_new_tokens,
                )
            else:
                raw_completion_2 = _generate_completion(
                    trainer=trainer,
                    tokenizer=tokenizer,
                    device=device,
                    prompt_text=followup_text,
                    max_new_tokens=max_new_tokens,
                )

            mapped_2, cleaned_2 = _map_to_label(raw_completion_2, class_labels_j)

            print("----- FOLLOWUP -----")
            print(f"FOLLOWUP RAW: {raw_completion_2!r}")
            print(f"FOLLOWUP CLEAN: {cleaned_2!r}")
            print(f"FOLLOWUP MAPPED: {mapped_2!r}")
            print("--------------------")

            mapped = mapped_2

        print("==============================\n")

        if mapped != "na":
            task_answers[task_name].append(mapped)
        else:
            task_answers[task_name].append("na")
            task_na_counts[task_name] += 1

        task_labels_dict[task_name].append(labels[j])
        task_totals[task_name] += 1

    # ---- compute metrics per task ----
    all_rows = []

    for task_name, answers_task in task_answers.items():
        labels_task = task_labels_dict[task_name]
        class_labels_task = task_class_labels[task_name]
        total_prompts = task_totals[task_name]
        na_count = task_na_counts[task_name]

        percentage_na = (na_count / total_prompts) * 100.0 if total_prompts > 0 else 0.0

        if len(class_labels_task) == 2 and _is_yesno_task(class_labels_task):
            df_task = evaluate_predictions_binary(
                answers_task,
                labels_task,
                mtype,
                train_loss=train_loss,
                val_loss=val_loss,
                epoch=epoch,
                fold=fold,
                fine_tune_type=fine_tune_type,
                percentage_na=percentage_na,
                elapsed_time=elapsed_time,
            )
        else:
            df_task = evaluate_predictions_multiclass(
                answers_task,
                labels_task,
                mtype,
                train_loss=train_loss,
                val_loss=val_loss,
                epoch=epoch,
                fold=fold,
                fine_tune_type=fine_tune_type,
                percentage_na=percentage_na,
                elapsed_time=elapsed_time,
            )

        offlabel = task_offlabel_first.get(task_name, [])
        offlabel_unique = list(set(offlabel))
        num_offlabel_first = len(offlabel)
        offlabel_ratio = (num_offlabel_first / total_prompts) if total_prompts > 0 else 0.0

        df_task["task_name"] = task_name
        df_task["split"] = split_name
        df_task["n_examples"] = total_prompts
        df_task["num_offlabel_first"] = num_offlabel_first
        df_task["offlabel_first_unique"] = str(offlabel_unique)
        df_task["offlabel_first_ratio"] = offlabel_ratio
        df_task["seed"] = seed

        all_rows.append(df_task)

    return pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()


def evaluate_predictions_binary(
    answers,
    labels,
    mtype,
    image_limit="na",
    train_loss="na",
    val_loss="na",
    epoch="na",
    fold="na",
    fine_tune_type="na",
    percentage_na="na",
    elapsed_time="na",
):
    total_prompts = len(answers)

    converted_responses = [
        1 if answer == "yes" else 0 if answer == "no" else -1
        for answer in answers
    ]

    correct_predictions = sum(
        1 for pred, true in zip(converted_responses, labels) if pred == true
    )
    accuracy = correct_predictions / total_prompts if total_prompts > 0 else 0.0

    adjusted_responses = [r if r != -1 else 0 for r in converted_responses]
    adjusted_labels = list(labels)

    prec_bin, rec_bin, f_bin, _ = precision_recall_fscore_support(
        adjusted_labels,
        adjusted_responses,
        average="binary",
        zero_division=0,
    )

    num_unique_elements = len(set(converted_responses))
    unique_elements = list(set(converted_responses))

    return pd.DataFrame(
        {
            "model": [mtype],
            "images_included": [image_limit],
            "accuracy": [accuracy],
            "precision (macro)": [prec_bin],
            "recall (macro)": [rec_bin],
            "f1score (macro)": [f_bin],
            "precision_micro": [prec_bin],
            "recall_micro": [rec_bin],
            "f1score_micro": [f_bin],
            "train_loss": [train_loss],
            "val_loss": [val_loss],
            "epoch": [epoch],
            "fold": [fold],
            "type": [fine_tune_type],
            "amount_of_prompts": [total_prompts],
            "percentage_nan": [percentage_na],
            "num_diff_answers": [num_unique_elements],
            "unique_answers": [str(unique_elements)],
            "elapsed_time": [str(elapsed_time)],
        }
    )


def evaluate_predictions_multiclass(
    answers,
    labels,
    mtype,
    image_limit="na",
    train_loss="na",
    val_loss="na",
    epoch="na",
    fold="na",
    fine_tune_type="na",
    percentage_na="na",
    elapsed_time="na",
):
    total_prompts = len(answers)

    converted_responses = [
        response if response != "na" and pd.notna(response) else -1
        for response in answers
    ]
    converted_labels = [label if pd.notna(label) else -1 for label in labels]

    precision_macro, recall_macro, fscore_macro, _ = precision_recall_fscore_support(
        converted_labels,
        converted_responses,
        average="macro",
        zero_division=0,
    )

    precision_micro, recall_micro, fscore_micro, _ = precision_recall_fscore_support(
        converted_labels,
        converted_responses,
        average="micro",
        zero_division=0,
    )

    correct_predictions = sum(
        1 for pred, true in zip(converted_responses, converted_labels) if pred == true
    )
    accuracy = correct_predictions / total_prompts if total_prompts > 0 else 0.0

    num_unique_elements = len(set(converted_responses))
    unique_elements = list(set(converted_responses))

    return pd.DataFrame(
        {
            "model": [mtype],
            "images_included": [image_limit],
            "accuracy": [accuracy],
            "precision (macro)": [precision_macro],
            "recall (macro)": [recall_macro],
            "f1score (macro)": [fscore_macro],
            "precision_micro": [precision_micro],
            "recall_micro": [recall_micro],
            "f1score_micro": [fscore_micro],
            "train_loss": [train_loss],
            "val_loss": [val_loss],
            "epoch": [epoch],
            "fold": [fold],
            "type": [fine_tune_type],
            "amount_of_prompts": [total_prompts],
            "percentage_nan": [percentage_na],
            "num_diff_answers": [num_unique_elements],
            "unique_answers": [str(unique_elements)],
            "elapsed_time": [str(elapsed_time)],
        }
    )