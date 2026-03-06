# agent_utils/gemma3_finetune_simple.py
"""
Simple multi-target JSON fine-tuning for Gemma-3.

This is a deliberately minimal version, closely modelled on
``llama3_ilora_finetune.py``.  No slot tokens, no custom loss,
no custom collator — just plain SFT on the full JSON answer string
using TRL's SFTTrainer.

The model learns to generate the complete JSON answer
(with plain values, not slot tokens) given the user prompt + transcript.

Inference speedups (no quality loss):
  - System prompt is KV-cache reused (suffix-only sent to model).
  - Early stop when output parses as complete JSON (fewer decode steps).
  - Batched inference: set inference_batch_size=4 or 8 in train_validate().
For maximum throughput (e.g. 10k+ posts), use ``inference_vllm_gemma3``
with a merged checkpoint and vLLM (pip install vllm).
"""

import os
import gc
import time
import json
import datetime
import tempfile

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

def _strip_bom(s: str) -> str:
    """Remove leading UTF-8 BOM so it doesn't become an extra token."""
    if s and s[0] == "\ufeff":
        return s[1:]
    return s


def _build_chat_text_simple(tokenizer, instruction: str, answer: str = None,
                            system_prompt: str = None) -> str:
    """Build a chat-formatted string using the tokenizer's official chat template.

    All special tokens (e.g. Gemma 3's <bos>, <start_of_turn>user, <end_of_turn>,
    <start_of_turn>model) come from the tokenizer's apply_chat_template; we do
    not create or hardcode any sequence tokens ourselves.
    When *system_prompt* is provided it is passed as a ``system`` role message;
    Gemma 3's template folds it into the first user turn automatically.
    """
    instruction = _strip_bom(instruction)
    if system_prompt:
        system_prompt = _strip_bom(system_prompt)
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": instruction})
    if answer is not None:
        messages.append({"role": "assistant", "content": answer})

    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=(answer is None),
        )
    # Fallback only for tokenizers without a chat template (not used for Gemma 3).
    raise ValueError(
        "This runner requires a tokenizer with apply_chat_template (e.g. Gemma 3). "
        "Do not use a tokenizer that lacks an official chat template."
    )


def _token_len(text: str, tokenizer) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))


def _precompute_prefix_kv(model, tokenizer, system_prompt, device):
    """Pre-compute KV cache for the static system-prompt prefix.

    Returns ``(past_key_values, prefix_token_count)`` or ``(None, 0)``
    when *system_prompt* is falsy or the prefix can't be cached safely.

    The common prefix is found by building two full prompts with different
    dummy instructions and comparing their token sequences.  This is
    template-agnostic and works regardless of how the tokenizer folds the
    system role into the conversation.
    """
    if not system_prompt:
        return None, 0

    # Build two prompts that differ only in the instruction part
    text_a = _build_chat_text_simple(tokenizer, "PLACEHOLDER_AAA", system_prompt=system_prompt)
    text_b = _build_chat_text_simple(tokenizer, "PLACEHOLDER_BBB", system_prompt=system_prompt)

    ids_a = tokenizer(text_a, return_tensors="pt")["input_ids"][0]
    ids_b = tokenizer(text_b, return_tensors="pt")["input_ids"][0]

    # Walk both sequences to find where they first diverge
    prefix_len = 0
    for i in range(min(len(ids_a), len(ids_b))):
        if ids_a[i] != ids_b[i]:
            break
        prefix_len = i + 1

    if prefix_len < 10:
        print(
            f"[prefix-cache] Common prefix too short ({prefix_len} tokens) "
            "— disabling KV prefix caching"
        )
        return None, 0

    prefix_ids = ids_a[:prefix_len].unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        out = model(input_ids=prefix_ids, use_cache=True)

    print(f"[prefix-cache] Cached {prefix_len} system-prompt tokens for KV reuse")
    return out.past_key_values, prefix_len


def _has_complete_json(text: str) -> bool:
    """True if text contains at least one parseable JSON object (for early stopping)."""
    if not text or "{" not in text or "}" not in text:
        return False
    candidate = _extract_last_json(text.strip())
    try:
        json.loads(candidate)
        return True
    except Exception:
        return False


def _generate_with_prefix_cache(
    model,
    tokenizer,
    suffix_ids: torch.Tensor,
    prefix_kv,
    prefix_len: int,
    max_new_tokens: int,
    pad_token_id: int | None,
    device: torch.device,
    stop_on_complete_json: bool = True,
):
    """
    Autoregressive generation using a precomputed prefix KV cache.
    Only the suffix tokens are run through the model; then we decode one token at a time.
    If stop_on_complete_json is True, stops as soon as the decoded output parses as JSON
    (saves decode steps; no quality change).
    Returns (new_token_ids, prefill_sec, decode_sec) where new_token_ids is a 1D tensor on CPU.
    """
    eos_token_id = getattr(tokenizer, "eos_token_id", None) or pad_token_id
    batch_size = suffix_ids.shape[0]
    suffix_len = suffix_ids.shape[1]

    # Prefill: forward the suffix with the prefix cache (only these tokens hit the model)
    position_ids = torch.arange(
        prefix_len, prefix_len + suffix_len, dtype=torch.long, device=device
    ).unsqueeze(0).expand(batch_size, -1)
    t0 = time.perf_counter()
    with torch.no_grad():
        outputs = model(
            input_ids=suffix_ids,
            position_ids=position_ids,
            past_key_values=prefix_kv,
            use_cache=True,
        )
    if device.type == "cuda":
        torch.cuda.synchronize()
    prefill_sec = time.perf_counter() - t0
    logits = outputs.logits  # (batch, suffix_len, vocab_size)
    past_key_values = outputs.past_key_values

    next_token_logits = logits[:, -1, :]
    next_tokens = next_token_logits.argmax(dim=-1, keepdim=True)
    generated = [next_tokens]

    cur_len = suffix_len + 1
    t_decode = time.perf_counter()
    for step in range(max_new_tokens - 1):
        next_token = next_tokens
        if eos_token_id is not None and (next_token == eos_token_id).all().item():
            break
        position_ids = torch.full(
            (batch_size, 1), prefix_len + cur_len, dtype=torch.long, device=device
        )
        with torch.no_grad():
            outputs = model(
                input_ids=next_token,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=True,
            )
        logits = outputs.logits
        past_key_values = outputs.past_key_values
        next_tokens = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated.append(next_tokens)
        cur_len += 1
        # Early stop when output looks like complete JSON (every 16 steps to limit overhead)
        if stop_on_complete_json and batch_size == 1 and step % 16 == 15:
            so_far = torch.cat(generated, dim=1)[0].detach().cpu()
            decoded = tokenizer.decode(so_far, skip_special_tokens=True)
            if _has_complete_json(decoded):
                break
    if device.type == "cuda":
        torch.cuda.synchronize()
    decode_sec = time.perf_counter() - t_decode

    out = torch.cat(generated, dim=1)[0].detach().cpu()
    return out, prefill_sec, decode_sec


def _generate_with_prefix_cache_batched(
    model,
    tokenizer,
    suffix_ids_list: list,
    prefix_kv,
    prefix_len: int,
    max_new_tokens: int,
    pad_token_id: int | None,
    device: torch.device,
    stop_on_complete_json: bool = True,
):
    """
    Batched generation with shared prefix cache. suffix_ids_list is a list of 1D tensors
    (suffix token ids per prompt). Pads to max length, runs one prefill for the batch,
    then decode until all sequences hit EOS or max_new_tokens.
    Returns list of (new_token_ids_1d, prefill_sec, decode_sec); prefill/decode_sec are
    shared (total batch time).
    """
    eos_token_id = getattr(tokenizer, "eos_token_id", None) or pad_token_id
    B = len(suffix_ids_list)
    if B == 0:
        return []
    pad_id = pad_token_id if pad_token_id is not None else 0
    max_suffix = max(s.shape[0] for s in suffix_ids_list)
    suffix_lens = [s.shape[0] for s in suffix_ids_list]
    # Pad to (B, max_suffix)
    padded = torch.full((B, max_suffix), pad_id, dtype=torch.long, device=device)
    attn_mask = torch.zeros(B, max_suffix, dtype=torch.long, device=device)
    for i, s in enumerate(suffix_ids_list):
        s = s.to(device)
        padded[i, : s.shape[0]] = s
        attn_mask[i, : s.shape[0]] = 1
    position_ids = torch.zeros(B, max_suffix, dtype=torch.long, device=device)
    for i in range(B):
        position_ids[i, : suffix_lens[i]] = torch.arange(
            prefix_len, prefix_len + suffix_lens[i], dtype=torch.long, device=device
        )
    prefix_batched = _expand_past_kv_to_batch(prefix_kv, B)
    t0 = time.perf_counter()
    with torch.no_grad():
        outputs = model(
            input_ids=padded,
            attention_mask=attn_mask,
            position_ids=position_ids,
            past_key_values=prefix_batched,
            use_cache=True,
        )
    if device.type == "cuda":
        torch.cuda.synchronize()
    prefill_sec = time.perf_counter() - t0
    logits = outputs.logits  # (B, max_suffix, V)
    past_key_values = outputs.past_key_values
    last_pos = torch.tensor([suffix_lens[j] - 1 for j in range(B)], device=device)
    next_tokens = logits[torch.arange(B, device=device), last_pos, :].argmax(dim=-1, keepdim=True)  # (B, 1)
    generated = [next_tokens]
    finished = torch.zeros(B, dtype=torch.bool, device=device)
    cur_len = max_suffix + 1
    t_decode = time.perf_counter()
    for step in range(max_new_tokens - 1):
        next_token = next_tokens.clone()
        next_token[finished] = pad_id
        for j in range(B):
            if next_tokens[j].item() == eos_token_id:
                finished[j] = True
        if finished.all():
            break
        position_ids_step = torch.full(
            (B, 1), prefix_len + cur_len, dtype=torch.long, device=device
        )
        with torch.no_grad():
            outputs = model(
                input_ids=next_token,
                position_ids=position_ids_step,
                past_key_values=past_key_values,
                use_cache=True,
            )
        logits = outputs.logits
        past_key_values = outputs.past_key_values
        next_tokens = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated.append(next_tokens)
        cur_len += 1
    if device.type == "cuda":
        torch.cuda.synchronize()
    decode_sec = time.perf_counter() - t_decode
    # Split into per-sequence token lists (truncate at EOS)
    all_tokens = torch.cat(generated, dim=1)  # (B, num_gen)
    results = []
    for j in range(B):
        row = all_tokens[j].detach().cpu()
        if eos_token_id is not None:
            idx = (row == eos_token_id).nonzero(as_tuple=True)[0]
            if len(idx) > 0:
                row = row[: idx[0].item() + 1]
        results.append((row, prefill_sec, decode_sec))
    return results


def _expand_past_kv_to_batch(past_kv, batch_size: int):
    """Expand a batch-1 KV cache to batch_size (repeat the same prefix for each)."""
    if past_kv is None or batch_size <= 1:
        return past_kv
    try:
        from transformers.cache_utils import DynamicCache
        if isinstance(past_kv, DynamicCache):
            key_list = getattr(past_kv, "key_cache", None) or getattr(past_kv, "_key_states", None)
            val_list = getattr(past_kv, "value_cache", None) or getattr(past_kv, "_value_states", None)
            if key_list is not None and val_list is not None:
                expanded = DynamicCache()
                for i in range(len(key_list)):
                    k = key_list[i]   # (1, H, L, D)
                    v = val_list[i]
                    k_b = k.repeat(batch_size, 1, 1, 1)
                    v_b = v.repeat(batch_size, 1, 1, 1)
                    expanded.update(k_b, v_b, i)
                return expanded
    except Exception:
        pass
    if isinstance(past_kv, (list, tuple)):
        out = []
        for layer in past_kv:
            if isinstance(layer, (list, tuple)) and len(layer) >= 2:
                k, v = layer[0], layer[1]
                k_b = k.repeat(batch_size, *([1] * (k.dim() - 1)))
                v_b = v.repeat(batch_size, *([1] * (v.dim() - 1)))
                out.append((k_b, v_b))
            else:
                out.append(layer)
        return tuple(out)
    return past_kv


def _clone_past_kv(past_kv):
    """Clone a KV cache so ``generate()`` can mutate the copy freely."""
    if past_kv is None:
        return None
    import copy
    try:
        from transformers.cache_utils import DynamicCache
        if isinstance(past_kv, DynamicCache):
            # API varies by transformers version: key_cache/value_cache (old) vs _key_states/_value_states (new)
            key_list = getattr(past_kv, "key_cache", None) or getattr(past_kv, "_key_states", None)
            val_list = getattr(past_kv, "value_cache", None) or getattr(past_kv, "_value_states", None)
            if key_list is not None and val_list is not None and len(key_list) == len(val_list):
                clone = DynamicCache()
                for i in range(len(key_list)):
                    clone.update(key_list[i].clone(), val_list[i].clone(), i)
                return clone
    except Exception:
        pass
    try:
        from transformers.cache_utils import DynamicCache
        if isinstance(past_kv, DynamicCache):
            return copy.deepcopy(past_kv)
    except Exception:
        pass
    if isinstance(past_kv, (list, tuple)):
        return type(past_kv)(
            tuple(t.clone() for t in layer) if isinstance(layer, tuple)
            else (layer.clone() if hasattr(layer, "clone") else layer)
            for layer in past_kv
        )
    return copy.deepcopy(past_kv)


def build_simple_sft_dataset(
    df,
    tokenizer,
    prompt_template: str,
    text_col: str,
    answer_col: str,
    max_seq_length: int = 4096,
    system_prompt: str = None,
) -> Dataset:
    """
    Build a HuggingFace Dataset with a single ``text`` column containing
    chat-formatted SFT strings: user prompt (with transcript) + assistant
    answer (the raw JSON string).

    Transcript is truncated dynamically so the full string fits in
    ``max_seq_length``.  No slot tokens, no special processing.
    """
    texts = []
    instructions = []
    answers = []
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
        full_empty = _build_chat_text_simple(tokenizer, instruction_empty, answer_str, system_prompt=system_prompt)
        overhead = _token_len(full_empty, tokenizer)

        transcript_budget = max(max_seq_length - overhead - safety_margin, 0)

        # Truncate transcript to budget
        all_tokens = tokenizer.tokenize(str(raw_text))
        trunc_tokens = all_tokens[:transcript_budget]
        trunc_text = tokenizer.convert_tokens_to_string(trunc_tokens)

        # Build final string
        instruction = insert_text_once(prompt_template, trunc_text)
        full_text = _build_chat_text_simple(tokenizer, instruction, answer_str, system_prompt=system_prompt)

        # Safety check
        final_len = _token_len(full_text, tokenizer)
        if final_len > max_seq_length and len(trunc_tokens) > 0:
            overshoot = final_len - max_seq_length + 32
            trunc_tokens = trunc_tokens[:-overshoot] if overshoot < len(trunc_tokens) else []
            trunc_text = tokenizer.convert_tokens_to_string(trunc_tokens)
            instruction = insert_text_once(prompt_template, trunc_text)
            full_text = _build_chat_text_simple(tokenizer, instruction, answer_str, system_prompt=system_prompt)

        texts.append(full_text)
        instructions.append(instruction)
        answers.append(answer_str)

    print(f"[simple-sft] Built {len(texts)} training examples "
          f"(max_seq_length={max_seq_length})")
    if texts:
        sample_len = _token_len(texts[0], tokenizer)
        print(f"  example 0: {sample_len} tokens, {len(texts[0])} chars")
        # With system prompt, Gemma folds it into the first user turn: codebook should appear first in content
        if system_prompt and system_prompt.strip():
            # Strip BOM and leading/trailing whitespace for comparison
            codebook_start = system_prompt.lstrip("\ufeff").strip()[:60]
            if codebook_start not in texts[0]:
                print(
                    f"  [simple-sft] WARNING: codebook start {codebook_start!r} not found in example — "
                    "check system vs user split"
                )
        print(f"  tail: ...{texts[0][-400:]}")

    return Dataset.from_dict({"text": texts, "instruction": instructions, "answer": answers})


def build_simple_val_prompts(
    df,
    tokenizer,
    prompt_template: str,
    text_col: str,
    answer_col: str,
    max_seq_length: int = 4096,
    system_prompt: str = None,
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
        full_empty = _build_chat_text_simple(tokenizer, instruction_empty, answer_str, system_prompt=system_prompt)
        overhead = _token_len(full_empty, tokenizer)
        transcript_budget = max(max_seq_length - overhead - safety_margin, 0)

        all_tokens = tokenizer.tokenize(str(raw_text))
        trunc_tokens = all_tokens[:transcript_budget]
        trunc_text = tokenizer.convert_tokens_to_string(trunc_tokens)
        instruction = insert_text_once(prompt_template, trunc_text)

        # Prompt only (user turn + generation prompt), no assistant answer
        prompt_only = _build_chat_text_simple(tokenizer, instruction, answer=None, system_prompt=system_prompt)
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

    # First, try a direct JSON parse
    try:
        return json.loads(candidate)
    except Exception:
        pass

    # Many smaller Gemma variants emit “smart quotes” or other Unicode
    # punctuation that makes otherwise-valid JSON fail to parse.  Normalise
    # the most common offenders and try again.
    translation_table = {
        ord("“"): ord('"'),
        ord("”"): ord('"'),
        ord("„"): ord('"'),
        ord("‟"): ord('"'),
        ord("’"): ord("'"),
        ord("‘"): ord("'"),
        ord("\u00a0"): ord(" "),  # non‑breaking space
    }
    normalised = candidate.translate(translation_table)

    try:
        return json.loads(normalised)
    except Exception:
        return None


def _normalize_text_for_partial(s) -> set:
    """Normalize for partial match: strip, lower, collapse whitespace, split into words."""
    if s is None:
        return set()
    t = str(s).strip().lower()
    t = " ".join(t.split())  # collapse whitespace
    return set(w for w in t.split() if w)


def _string_exact_match(gold, pred) -> bool:
    """Case-insensitive, whitespace-normalized equality."""
    return _normalize_text_for_partial(gold) == _normalize_text_for_partial(pred)


def _string_partial_match(gold, pred) -> bool:
    """True if gold and pred share at least one word (after normalization)."""
    g_words = _normalize_text_for_partial(gold)
    p_words = _normalize_text_for_partial(pred)
    if not g_words or not p_words:
        return False
    return bool(g_words & p_words)


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
    training_time_sec: float | None = None,
    targets_spec: dict | None = None,
    prefix_kv=None,
    prefix_len: int = 0,
    inference_batch_size: int = 1,
    stop_on_complete_json: bool = True,
):
    """
    Run generation on validation prompts, print a few examples, and compute
    simple per-target metrics (accuracy, precision, recall, F1).
    Times inference and saves total/avg inference time (and optionally
    training_time_sec for val) in the same metrics CSV.
    If targets_spec is provided, "in label" / "answers_in_label" use the
    target's allowed list (in-scope) instead of the gold set, so e.g. topic01
    predictions like AGRICULTURE are in_label even when no gold had that value.
    """
    N = min(len(val_prompts), len(val_gold_raw))
    if N == 0:
        print("[val-inference] No validation examples to run.")
        return None

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
    use_prefix_cache_global = (
        prefix_kv is not None and prefix_len > 0 and inference_batch_size >= 1
    )
    inference_start = time.perf_counter()

    def process_one_result(i, prompt_text, gold, raw_completion, suffix_text_or_none):
        """Shared logic: parse, accumulate per-target, print if i < n_print."""
        parsed = _extract_pred_json(raw_completion)
        try:
            gold_dict = json.loads(gold)
        except Exception:
            gold_dict = {}
        if gold_dict:
            for t, g_val in gold_dict.items():
                if g_val is None:
                    continue
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
            print(f"\n--- Example {i + 1}/{N} ---")
            if use_prefix_cache_global and suffix_text_or_none is not None:
                print("SENT TO MODEL (suffix only; prefix in KV cache):")
                st = suffix_text_or_none[:600] + ("..." if len(suffix_text_or_none) > 600 else "")
                print(st)
                print("(Full prompt for reference, last 200 chars): ..." + prompt_text[-200:])
            else:
                print(f"PROMPT (last 350 chars): ...{prompt_text[-350:]}")
            print(f"GOLD: {gold[:500]}{'...' if len(gold) > 500 else ''}")
            print(f"GENERATED: {raw_completion[:800]}{'...' if len(raw_completion) > 800 else ''}")
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
        if (i + 1) % 5 == 0 or i == N - 1:
            print(f"[val-inference] processed {i + 1}/{N} validation examples")

    if use_prefix_cache_global and inference_batch_size > 1:
        # Batched path: shared prefix, multiple suffixes per forward
        kv_clone = _clone_past_kv(prefix_kv)
        batch_timing_printed = False
        for start in range(0, N, inference_batch_size):
            end = min(start + inference_batch_size, N)
            suffix_ids_list = []
            batch_indices = []
            for j in range(end - start):
                i = start + j
                enc = tokenizer(
                    val_prompts[i],
                    return_tensors="pt",
                    padding=False,
                    truncation=False,
                )
                full_ids = enc["input_ids"][0].to(device)
                if full_ids.shape[0] <= prefix_len:
                    continue
                suffix_ids_list.append(full_ids[prefix_len:])
                batch_indices.append(i)
            if not suffix_ids_list:
                continue
            results = _generate_with_prefix_cache_batched(
                model=trainer.model,
                tokenizer=tokenizer,
                suffix_ids_list=suffix_ids_list,
                prefix_kv=kv_clone,
                prefix_len=prefix_len,
                max_new_tokens=max_new_tokens,
                pad_token_id=pad_token_id,
                device=device,
                stop_on_complete_json=stop_on_complete_json,
            )
            if not batch_timing_printed and results:
                prefill_sec, decode_sec = results[0][1], results[0][2]
                print(
                    f"[prefix-cache] Batched inference (batch_size={len(suffix_ids_list)}): "
                    f"Prefill: {prefill_sec:.2f}s | Decode: {decode_sec:.2f}s"
                )
                batch_timing_printed = True
            for b in range(len(suffix_ids_list)):
                i = batch_indices[b]
                prompt_text = val_prompts[i]
                gold = val_gold_raw[i]
                raw_completion = tokenizer.decode(results[b][0], skip_special_tokens=True)
                suffix_text = tokenizer.decode(suffix_ids_list[b], skip_special_tokens=True)
                process_one_result(i, prompt_text, gold, raw_completion, suffix_text)
            for i in range(start, end):
                if i not in batch_indices:
                    enc = tokenizer(val_prompts[i], return_tensors="pt", padding=False, truncation=False)
                    full_ids = enc["input_ids"].to(device)
                    if full_ids.shape[-1] <= prefix_len:
                        continue
                    kv_single = _clone_past_kv(prefix_kv)
                    suf = full_ids[:, prefix_len:].to(device)
                    new_tok, _, _ = _generate_with_prefix_cache(
                        trainer.model, tokenizer, suf, kv_single, prefix_len,
                        max_new_tokens, pad_token_id, device, stop_on_complete_json=stop_on_complete_json,
                    )
                    raw_completion = tokenizer.decode(new_tok, skip_special_tokens=True)
                    suffix_text = tokenizer.decode(suf[0], skip_special_tokens=True)
                    process_one_result(i, val_prompts[i], val_gold_raw[i], raw_completion, suffix_text)
    else:
        # Per-example path
        for i in range(N):
            prompt_text = val_prompts[i]
            gold = val_gold_raw[i]
            enc = tokenizer(
                prompt_text,
                return_tensors="pt",
                padding=False,
                truncation=False,
            )
            full_ids = enc["input_ids"].to(device)
            use_prefix_cache = (
                prefix_kv is not None
                and prefix_len > 0
                and full_ids.shape[-1] > prefix_len
            )
            if use_prefix_cache:
                kv_clone = _clone_past_kv(prefix_kv)
                suffix_len = full_ids.shape[-1] - prefix_len
                suffix_ids = full_ids[:, prefix_len:].to(device)
                new_tokens, prefill_sec, decode_sec = _generate_with_prefix_cache(
                    model=trainer.model,
                    tokenizer=tokenizer,
                    suffix_ids=suffix_ids,
                    prefix_kv=kv_clone,
                    prefix_len=prefix_len,
                    max_new_tokens=max_new_tokens,
                    pad_token_id=pad_token_id,
                    device=device,
                    stop_on_complete_json=stop_on_complete_json,
                )
                raw_completion = tokenizer.decode(new_tokens, skip_special_tokens=True)
                suffix_text = tokenizer.decode(suffix_ids[0], skip_special_tokens=True)
                if i == 0:
                    print(
                        f"[prefix-cache] Only {suffix_len} tokens sent to model "
                        f"(prefix {prefix_len} tokens from cache). "
                        f"Prefill: {prefill_sec:.2f}s | Decode: {decode_sec:.2f}s"
                    )
            else:
                attention_mask = enc.get("attention_mask")
                if attention_mask is not None:
                    attention_mask = attention_mask.to(device)
                with torch.no_grad():
                    generated = trainer.model.generate(
                        input_ids=full_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                        num_return_sequences=1,
                        pad_token_id=pad_token_id,
                    )
                decode_offset = full_ids.shape[-1]
                new_tokens = generated[0, decode_offset:].detach().cpu()
                raw_completion = tokenizer.decode(new_tokens, skip_special_tokens=True)
                suffix_text = None
            process_one_result(
                i, prompt_text, gold, raw_completion,
                suffix_text if use_prefix_cache else None,
            )

    total_inference_sec = time.perf_counter() - inference_start
    avg_sec_per_prompt = total_inference_sec / N if N else 0.0
    print(f"[{split_name}-inference] total time: {total_inference_sec:.2f}s  |  n_prompts: {N}  |  avg: {avg_sec_per_prompt:.3f}s per prompt")
    if training_time_sec is not None and split_name == "val":
        print(f"[{split_name}-inference] training time (this epoch): {training_time_sec:.2f}s")

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

        # "In label" = in allowed set for all targets (binary, multiclass, string).
        # Allowed comes from TARGETS in the notebook; string targets get allowed from the dataset.
        gold_label_set = set(all_true)
        spec = (targets_spec or {}).get(t) if targets_spec else None
        allowed = spec.get("allowed") if spec and isinstance(spec, dict) else None
        if allowed is not None:
            allowed_set = {str(a).strip() for a in allowed}
            def _in_scope(p):
                return str(p).strip() in allowed_set
        else:
            def _in_scope(p):
                return p in gold_label_set
        n_in_label = sum(1 for _, p in pairs if _in_scope(p))
        in_label_frac = n_in_label / n_gold if n_gold > 0 else 0.0

        is_string_target = spec and spec.get("type") == "string"
        answers_partially_correct = []

        if n_answered > 0:
            y_true = [g for g, _ in pairs]
            if is_string_target:
                # For string targets: count as correct if exact match OR partial match (word overlap)
                y_pred = []
                partial_only_preds = set()
                for g, p in pairs:
                    exact = _string_exact_match(g, p)
                    partial = _string_partial_match(g, p)
                    if exact:
                        y_pred.append(g)
                    elif partial:
                        y_pred.append(g)  # count as correct
                        partial_only_preds.add(str(p).strip())
                    else:
                        y_pred.append(p)
                answers_partially_correct = sorted(partial_only_preds)
            else:
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
        answers_in_label = sorted({str(p) for _, p in pairs if _in_scope(p)})
        answers_out_label = sorted({str(p) for _, p in pairs if not _in_scope(p)})

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
                "answers_partially_correct": ";".join(answers_partially_correct) if answers_partially_correct else None,
                "total_inference_sec": None,
                "n_prompts": None,
                "avg_sec_per_prompt": None,
                "training_time_sec": None,
            }
        )

    # Append timing row (same CSV as per-target metrics)
    timing_row = {
        "target": "_timing",
        "n_gold": N,
        "n_answered": None,
        "n_in_label": None,
        "accuracy": None,
        "precision_macro": None,
        "recall_macro": None,
        "f1_macro": None,
        "answered_frac": None,
        "in_label_frac": None,
        "answers_in_label": None,
        "answers_out_of_label": None,
        "answers_partially_correct": None,
        "total_inference_sec": total_inference_sec,
        "n_prompts": N,
        "avg_sec_per_prompt": avg_sec_per_prompt,
        "training_time_sec": training_time_sec if split_name == "val" else None,
    }
    rows.append(timing_row)

    print("=" * 80 + "\n")

    # Optionally save metrics to CSV (includes timing row)
    if results_folder is not None and rows:
        os.makedirs(results_folder, exist_ok=True)
        lr_str = f"{learning_rate}" if learning_rate is not None else "na"
        ep_str = f"{epoch}" if epoch is not None else "na"
        seed_str = f"{seed}" if seed is not None else "na"
        csv_name = f"{mtype}_{split_name}_metrics_lr{lr_str}_seed{seed_str}_epoch{ep_str}.csv"
        csv_path = os.path.join(results_folder, csv_name)
        pd.DataFrame(rows).to_csv(csv_path, index=False)
        print(f"[{split_name}-metrics] Saved per-target metrics + timing to {csv_path}")

    return {
        "total_inference_sec": total_inference_sec,
        "n_prompts": N,
        "avg_sec_per_prompt": avg_sec_per_prompt,
    }


# ── main entry point ──────────────────────────────────────────────────

GEMMA_MODEL_IDS = {
    "4b": "google/gemma-3-4b-it",
    "27b": "google/gemma-3-27b-it",
}


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
    gemma_model="27b",
    max_val_infer=5,
    targets_spec=None,
    system_prompt: str = None,
    inference_batch_size: int = 1,
    stop_on_complete_json: bool = True,
):
    """
    Simple multi-target JSON fine-tuning for Gemma-3.

    Closely mirrors ``run_fine_tuned_llama3_ilora`` but:
      - Uses Gemma-3 4B-IT or 27B-IT (``gemma_model``: "4b" | "27b") or full ``model_id``
      - Trains with plain SFTTrainer (no slot tokens, no custom loss)
      - The assistant answer is the raw JSON string from ``answer_col``

    If ``system_prompt`` is provided the codebook/instructions are placed in
    the system role.  During inference the system-prompt KV cache is computed
    once per epoch and reused for every example (big speedup).
    """
    mtype = "simple_gemma3"
    run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    if model_dir is not None:
        base_run_dir = os.path.join(model_dir, f"{run_id}_{mtype}")
        os.makedirs(base_run_dir, exist_ok=True)
        print(f"[save] All outputs (trainer + adapter) under: {base_run_dir}")
    else:
        base_run_dir = None

    if gemma_model in GEMMA_MODEL_IDS:
        model_id = GEMMA_MODEL_IDS[gemma_model]
    else:
        model_id = gemma_model  # full HuggingFace model id
    print(f"Gemma model: {gemma_model} -> {model_id}")

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
        system_prompt=system_prompt,
    )
    print(f"[simple-sft] Built {len(test_prompts)} TEST prompts for inference")

    # Log how system vs user prompt are used (so user can verify the split)
    if system_prompt is not None:
        user_preview = (prompt[:80] + "...") if len(prompt) > 80 else prompt
        print(
            f"[simple-sft] Prompt split: system={len(system_prompt)} chars (codebook), "
            f"user template={len(prompt)} chars → instruction = template with '{{}}' filled by post"
        )
        print(f"[simple-sft] User template preview: {user_preview}")
    else:
        print("[simple-sft] Single prompt (no system_prompt); no KV prefix caching.")

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
                system_prompt=system_prompt,
            )

            val_dataset = build_simple_sft_dataset(
                df=val_rows,
                tokenizer=tokenizer,
                prompt_template=prompt,
                text_col=text_col,
                answer_col=answer_col,
                max_seq_length=max_tokens,
                system_prompt=system_prompt,
            )

            # Validation prompts for inference (same truncation as train)
            val_prompts, val_gold_raw = build_simple_val_prompts(
                df=val_rows,
                tokenizer=tokenizer,
                prompt_template=prompt,
                text_col=text_col,
                answer_col=answer_col,
                max_seq_length=max_tokens,
                system_prompt=system_prompt,
            )
            print(f"[simple-sft] Built {len(val_prompts)} validation prompts for inference")

            # ── Print 1–2 training examples with clear [SYSTEM] / [USER] / [ASSISTANT] ───────
            n_print = min(2, len(dataset))
            print("\n" + "=" * 80)
            print(f"TRAINING EXAMPLES (first {n_print})")
            print("=" * 80)
            for i in range(n_print):
                full_text = dataset["text"][i]
                n_tok = _token_len(full_text, tokenizer)
                instruction = dataset["instruction"][i]
                answer = dataset["answer"][i]
                print(f"\n--- Example {i} (full SFT: {len(full_text)} chars, {n_tok} tokens) ---\n")
                if system_prompt:
                    sys_preview = system_prompt.strip()[:500] + "..." if len(system_prompt) > 500 else system_prompt.strip()
                    print("[SYSTEM]")
                    print(sys_preview)
                    print()
                print("[USER]")
                print(instruction)
                print()
                print("[ASSISTANT]")
                print(answer[:1200] + ("..." if len(answer) > 1200 else ""))
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
            if base_run_dir is not None:
                trainer_output_dir = os.path.join(
                    base_run_dir, f"trainer_lr{learning_rate}_seed{train_val_seed}"
                )
                model_save_dir = os.path.join(
                    base_run_dir, f"{mtype}_lr{learning_rate}_seed{train_val_seed}"
                )
                os.makedirs(trainer_output_dir, exist_ok=True)
                os.makedirs(model_save_dir, exist_ok=True)
            else:
                trainer_output_dir = tempfile.mkdtemp(prefix="africa_llm_simple_")
                model_save_dir = None
            training_args = TrainingArguments(
                output_dir=trainer_output_dir,
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

            # model_save_dir already set above (under base_run_dir); we overwrite each epoch.

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

                # Pre-compute system-prompt KV cache (weights changed this epoch)
                prefix_kv, prefix_len = _precompute_prefix_kv(
                    trainer.model, tokenizer, system_prompt, device,
                )

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
                    training_time_sec=elapsed,
                    targets_spec=targets_spec,
                    prefix_kv=prefix_kv,
                    prefix_len=prefix_len,
                    inference_batch_size=inference_batch_size,
                    stop_on_complete_json=stop_on_complete_json,
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
                    targets_spec=targets_spec,
                    prefix_kv=prefix_kv,
                    prefix_len=prefix_len,
                    inference_batch_size=inference_batch_size,
                    stop_on_complete_json=stop_on_complete_json,
                )

                # Free KV cache for this epoch
                del prefix_kv
                if gpu_avail:
                    torch.cuda.empty_cache()

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
