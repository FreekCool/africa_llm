# agent_utils/gemma3_finetune_simple.py
"""
Simple multi-target JSON fine-tuning for Gemma-3.

This is a deliberately minimal version, closely modelled on
``llama3_ilora_finetune.py``.  No slot tokens, no custom loss,
no custom collator — just plain SFT on the full JSON answer string
using TRL's SFTTrainer.

The model learns to generate the complete JSON answer
(with plain values, not slot tokens) given the user prompt + transcript.

Inference uses the standard ``model.generate()`` on the full prompt
(system prompt + transcript), which stops at the model's end-of-turn token.
For maximum throughput (e.g. 10k+ posts), use ``inference_vllm_gemma3``
with a merged checkpoint and vLLM (pip install vllm).
"""

import os
import gc
import re
import time
import json
import datetime
import tempfile
import contextlib

import torch
import pandas as pd
from collections import defaultdict
from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from trl import SFTTrainer, SFTConfig, DataCollatorForCompletionOnlyLM
from transformers import (
    AutoTokenizer,
    AutoProcessor,
    AutoModelForCausalLM,
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
# Key aliases: model may output different key names than gold (e.g. resource_distribution_gender vs resource_distribution_for_gender)
try:
    from .utils import _EVAL_KEY_ALIASES
except ImportError:
    _EVAL_KEY_ALIASES = {}


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


def _truncate_transcript(
    tokenizer,
    prompt_template: str,
    raw_text,
    answer_str: str,
    system_prompt: str,
    max_seq_length: int,
    max_text_tokens: int,
    safety_margin: int = 48,
) -> str:
    """Cap the TRANSCRIPT only — the codebook system prompt is never truncated.

    Overhead = system prompt (codebook) + user template + answer + chat wrappers with an
    empty transcript. If that alone doesn't fit ``max_seq_length`` the codebook would have
    to be cut, so we raise instead (increase ``max_tokens``). Otherwise the transcript is
    capped at ``max_text_tokens`` (and never beyond the room left after codebook+answer),
    so the same cap is applied identically at train and inference time.
    """
    instruction_empty = insert_text_once(prompt_template, "")
    full_empty = _build_chat_text_simple(tokenizer, instruction_empty, answer_str, system_prompt=system_prompt)
    overhead = _token_len(full_empty, tokenizer)
    if overhead + safety_margin > max_seq_length:
        raise ValueError(
            f"max_seq_length={max_seq_length} too small: codebook+answer overhead is "
            f"{overhead} tokens (+{safety_margin} margin) with zero transcript. Increase "
            "max_tokens — the codebook must never be truncated; only the transcript is capped."
        )
    budget = min(max_text_tokens, max_seq_length - overhead - safety_margin)
    tokens = tokenizer.tokenize(str(raw_text))[:budget]
    return tokenizer.convert_tokens_to_string(tokens)


def build_simple_sft_dataset(
    df,
    tokenizer,
    prompt_template: str,
    text_col: str,
    answer_col: str,
    max_seq_length: int = 4096,
    system_prompt: str = None,
    max_text_tokens: int = 500,
) -> Dataset:
    """
    Build a HuggingFace Dataset with a single ``text`` column containing
    chat-formatted SFT strings: user prompt (with transcript) + assistant
    answer (the raw JSON string).

    Only the transcript is capped (at ``max_text_tokens``); the codebook system
    prompt is never truncated.  No slot tokens, no special processing.
    """
    texts = []
    instructions = []
    answers = []

    for _, row in df.iterrows():
        raw_text = row[text_col]
        if pd.isna(raw_text):
            continue
        answer = row[answer_col]
        if pd.isna(answer):
            continue
        answer_str = str(answer)

        trunc_text = _truncate_transcript(
            tokenizer, prompt_template, raw_text, answer_str,
            system_prompt, max_seq_length, max_text_tokens,
        )
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
        # Length distributions to catch an undersized max_tokens or odd answers.
        _al = pd.Series([_token_len(a, tokenizer) for a in answers])
        print(f"  answer tokens: mean={_al.mean():.0f} median={_al.median():.0f} "
              f"min={int(_al.min())} max={int(_al.max())} p95={_al.quantile(0.95):.0f}")
        print(f"  kept {len(texts)}/{len(df)} rows ({len(df) - len(texts)} skipped: NaN text/answer)")
        step = max(1, len(texts) // 50)
        _sl = pd.Series([_token_len(t, tokenizer) for t in texts[::step][:50]])
        print(f"  full-seq tokens (sample n={len(_sl)}): mean={_sl.mean():.0f} "
              f"median={_sl.median():.0f} min={int(_sl.min())} max={int(_sl.max())} "
              f"(max_seq_length={max_seq_length})")
        if int(_sl.max()) >= max_seq_length:
            print(f"  *** WARNING: sample reaches max_seq_length={max_seq_length} — "
                  "answer may be truncated; raise max_tokens. ***")

    return Dataset.from_dict({"text": texts, "instruction": instructions, "answer": answers})


def build_simple_val_prompts(
    df,
    tokenizer,
    prompt_template: str,
    text_col: str,
    answer_col: str,
    max_seq_length: int = 4096,
    system_prompt: str = None,
    max_text_tokens: int = 500,
):
    """
    Build validation prompts (user turn only, with transcript) and gold answers.
    Same transcript cap as build_simple_sft_dataset so prompts match training.
    Returns (val_prompts, val_gold_raw) for inference.
    """
    prompts = []
    gold_raw = []

    for _, row in df.iterrows():
        raw_text = row[text_col]
        if pd.isna(raw_text):
            continue
        answer = row[answer_col]
        if pd.isna(answer):
            continue
        answer_str = str(answer)

        trunc_text = _truncate_transcript(
            tokenizer, prompt_template, raw_text, answer_str,
            system_prompt, max_seq_length, max_text_tokens,
        )
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

    # First, try a direct JSON parse. json.loads can return any JSON type
    # (int, str, list, ...); the contract here is "a dict or None", so a
    # successfully-parsed non-dict means the model didn't emit an object.
    try:
        obj = json.loads(candidate)
        return obj if isinstance(obj, dict) else None
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
        obj = json.loads(normalised)
        return obj if isinstance(obj, dict) else None
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


# Multi-value (`;`-joined) scoring (D1) + language Other(<lang>) acceptance (D2).
# A target marked ``multi_value`` in targets_spec is scored as an order-insensitive
# set; ``allow_other_paren`` accepts free-form ``Other(...)`` atoms as in-label.
_OTHER_PAREN_RE = re.compile(r"^Other\(.*\)$")


def _split_atoms(v) -> list:
    """Split a (possibly ;-joined) label value into stripped, non-empty atoms."""
    return [a.strip() for a in str(v).split(";") if a.strip()]


def _canon_multi(v) -> str:
    """Canonical (sorted) form of a ;-joined label so ordering can't mis-score."""
    return ";".join(sorted(_split_atoms(v)))


def _atom_in_scope(atom: str, allowed_set: set, allow_other: bool) -> bool:
    return atom in allowed_set or (allow_other and _OTHER_PAREN_RE.match(atom) is not None)


# Sentinel for model predictions that are null (JSON "null"). Used so sklearn
# metrics get no None values (which raise) and null preds count as wrong.
_PRED_NULL_SENTINEL = "__NULL_PRED__"


def _is_applicable_gold(g) -> bool:
    """False when gold is the conditional-field "not applicable" sentinel.

    Gated fields (e.g. resource_distribution_by_whom1) are mostly N/A, so plain
    accuracy is inflated by trivially predicting N/A. accuracy_applicable scores
    only the rows where a real label was expected.
    """
    return not (isinstance(g, str) and g.strip().lower() == "not applicable")


def _resolve_pred_value(parsed: dict | None, gold_key: str) -> object:
    """Get prediction for gold key, using canonical key or _EVAL_KEY_ALIASES."""
    if not parsed:
        return None
    p = parsed.get(gold_key)
    if p is not None:
        return p
    for alias in _EVAL_KEY_ALIASES.get(gold_key, []):
        if alias in parsed:
            return parsed[alias]
    return None


def _normalize_pred_for_metric(p_val: object, g_val: object) -> object:
    """
    Normalize predicted value so it is comparable to gold for accuracy.
    E.g. gold=3 (int), pred="3" (str) -> return 3 so equality works.
    """
    if p_val is None:
        return None
    if g_val is None:
        return p_val
    if type(g_val) == type(p_val):
        if isinstance(g_val, float) and g_val.is_integer() and isinstance(p_val, float) and p_val.is_integer():
            return int(p_val)
        return p_val
    if isinstance(g_val, int) and not isinstance(g_val, bool):
        if isinstance(p_val, float) and p_val.is_integer():
            return int(p_val)
        if isinstance(p_val, str):
            try:
                return int(float(p_val))
            except (ValueError, TypeError):
                pass
    if isinstance(g_val, float):
        if isinstance(p_val, (int, float)):
            return float(p_val) if isinstance(p_val, int) else p_val
        if isinstance(p_val, str):
            try:
                return float(p_val)
            except (ValueError, TypeError):
                pass
    if isinstance(g_val, str):
        return str(p_val).strip()
    return p_val


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
    gemma_model: str | None = None,
    run_id: str | None = None,
    compute_dtype=torch.bfloat16,
    val_ids=None,
    predictions_out_path: str | None = None,
):
    """
    Run generation on validation prompts, print a few examples, and compute
    simple per-target metrics (accuracy, precision, recall, F1).
    Times inference and saves total/avg inference time (and optionally
    training_time_sec for val) in the same metrics CSV.
    If targets_spec is provided, "in label" / "answers_in_label" use the
    target's allowed list (in-scope) instead of the gold set, so e.g. topic01
    predictions like AGRICULTURE are in_label even when no gold had that value.
    If predictions_out_path is set, one row per example (id, json_ok, gold,
    generated, predicted_json) is appended there for later error analysis;
    val_ids (aligned to val_prompts) supplies the id column. Both default to
    None, leaving per-epoch training behaviour unchanged.
    """
    N = min(len(val_prompts), len(val_gold_raw))
    if N == 0:
        print("[val-inference] No validation examples to run.")
        return None

    n_print = min(max_examples, N)

    # storage for metrics
    per_target_true = defaultdict(list)
    per_target_pred = defaultdict(list)

    # Track how many completions we could parse a JSON object from. Items we
    # cannot parse are excluded from per-target metrics (no predictions to
    # score), so we report the parse rate separately and print each failure
    # rather than silently dropping it.
    parse_stats = {"ok": 0, "fail": 0}
    parse_failures = []  # (index, raw_completion snippet) for unparseable items

    # Generation diagnostics: token counts per example, plus how often the model
    # produced nothing or ran to max_new_tokens (a strong sign of never-closed JSON).
    gen_lengths = []
    gen_stats = {"empty": 0, "hit_max": 0}
    # Why JSON parsing failed, so we can tell truncation from malformed from empty.
    parse_fail_cats = {"empty": 0, "no_brace": 0, "unbalanced_brace": 0, "invalid_json": 0}

    print("\n" + "=" * 80)
    print(f"{split_name.upper()} INFERENCE (first {n_print} examples, max_new_tokens={max_new_tokens})")
    print("=" * 80)

    pad_token_id = getattr(tokenizer, "pad_token_id", None) or getattr(
        tokenizer, "eos_token_id", None
    )
    trainer.model.eval()
    inference_start = time.perf_counter()

    def process_one_result(i, prompt_text, gold, raw_completion, n_gen_tokens=None):
        """Shared logic: parse, accumulate per-target, print if i < n_print."""
        if n_gen_tokens is not None:
            gen_lengths.append(n_gen_tokens)
            if n_gen_tokens >= max_new_tokens:
                gen_stats["hit_max"] += 1
        if not (raw_completion or "").strip():
            gen_stats["empty"] += 1
        parsed = _extract_pred_json(raw_completion)
        if parsed is None:
            parse_stats["fail"] += 1
            snippet = (raw_completion or "").strip()
            parse_failures.append((i, snippet))
            if not snippet:
                parse_fail_cats["empty"] += 1
            elif "{" not in snippet:
                parse_fail_cats["no_brace"] += 1
            elif snippet.count("{") != snippet.count("}"):
                parse_fail_cats["unbalanced_brace"] += 1
            else:
                parse_fail_cats["invalid_json"] += 1
            # Always surface unparseable completions (not just the first
            # n_print) so we can see what the model produced instead of a
            # JSON object, rather than silently dropping the example.
            print(
                f"\n[parse-fail] Example {i + 1}/{N}: could not parse a JSON object "
                f"from the completion\n"
                f"  GOLD (first 300 chars): {gold[:300]}{'...' if len(gold) > 300 else ''}\n"
                f"  GENERATED (first 600 chars): "
                f"{snippet[:600].replace(chr(10), ' ')}{'...' if len(snippet) > 600 else ''}"
            )
        else:
            parse_stats["ok"] += 1
        try:
            gold_dict = json.loads(gold)
        except Exception:
            gold_dict = {}
        if gold_dict:
            for t, g_val in gold_dict.items():
                if g_val is None:
                    continue
                # "NOT CODED" marks a field the annotators never coded (currently
                # only national_unity_narrow). Treat it like a missing gold label so
                # it is excluded from metrics; training is unaffected.
                if isinstance(g_val, str) and g_val.strip() == "NOT CODED":
                    continue
                if isinstance(g_val, float) and g_val.is_integer():
                    g_val_norm = int(g_val)
                else:
                    g_val_norm = g_val
                p_val = _resolve_pred_value(parsed, t)
                if p_val is not None:
                    p_val_norm = _normalize_pred_for_metric(p_val, g_val_norm)
                else:
                    p_val_norm = "MISSING"
                per_target_true[t].append(g_val_norm)
                per_target_pred[t].append(p_val_norm)
        if i < n_print:
            print(f"\n--- Example {i + 1}/{N} ---")
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
        if predictions_out_path is not None:
            rec_id = val_ids[i] if val_ids is not None and i < len(val_ids) else None
            pred_json_str = json.dumps(parsed, ensure_ascii=False) if parsed is not None else None
            pred_row = pd.DataFrame(
                [{
                    "id": rec_id,
                    "json_ok": parsed is not None,
                    "gold": gold,
                    "generated": raw_completion,
                    "predicted_json": pred_json_str,
                }]
            )
            write_header = not os.path.exists(predictions_out_path)
            pred_row.to_csv(predictions_out_path, mode="a", index=False, header=write_header)

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
        attention_mask = enc.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        # Run generation under autocast so attention sees a single dtype.
        # The model is 4-bit + LoRA with bf16 compute; during generation
        # (use_cache=True) the bf16 KV cache holds k/v as bf16 while the fresh
        # query is upcast to fp32 by the rotary/norm/LoRA paths, and SDPA
        # rejects the mismatch. autocast casts q/k/v to a uniform dtype, the
        # same precision context bf16 training uses internally.
        amp_ctx = (
            torch.autocast(device_type="cuda", dtype=compute_dtype)
            if device.type == "cuda"
            else contextlib.nullcontext()
        )
        with torch.no_grad(), amp_ctx:
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
        process_one_result(
            i, prompt_text, gold, raw_completion,
            n_gen_tokens=int(new_tokens.shape[0]),
        )

    total_inference_sec = time.perf_counter() - inference_start
    avg_sec_per_prompt = total_inference_sec / N if N else 0.0
    print(f"[{split_name}-inference] total time: {total_inference_sec:.2f}s  |  n_prompts: {N}  |  avg: {avg_sec_per_prompt:.3f}s per prompt")
    if training_time_sec is not None and split_name == "val":
        print(f"[{split_name}-inference] training time (this epoch): {training_time_sec:.2f}s")

    # ---- JSON parse rate ----
    n_processed = parse_stats["ok"] + parse_stats["fail"]
    json_parse_frac = parse_stats["ok"] / n_processed if n_processed else 0.0
    print(
        f"[{split_name}-inference] JSON parse rate: {parse_stats['ok']}/{n_processed} "
        f"({json_parse_frac * 100:.1f}%) — {parse_stats['fail']} unparseable"
    )
    if parse_failures:
        fail_positions = ", ".join(str(idx + 1) for idx, _ in parse_failures)
        print(f"[{split_name}-inference] unparseable example positions: {fail_positions}")

    # ---- generation diagnostics ----
    if gen_lengths:
        _gl = pd.Series(gen_lengths)
        print(
            f"[{split_name}-inference] generated tokens: mean={_gl.mean():.0f} "
            f"median={_gl.median():.0f} min={int(_gl.min())} max={int(_gl.max())}  "
            f"| empty={gen_stats['empty']}  | hit_max(={max_new_tokens})={gen_stats['hit_max']}"
        )
    if parse_stats["fail"]:
        print(
            f"[{split_name}-inference] parse-fail breakdown: "
            f"empty={parse_fail_cats['empty']}  no_brace={parse_fail_cats['no_brace']}  "
            f"unbalanced_brace(truncated?)={parse_fail_cats['unbalanced_brace']}  "
            f"invalid_json={parse_fail_cats['invalid_json']}"
        )

    # ---- per-target metrics ----
    print("\n" + "-" * 80)
    print(f"{split_name.upper()} METRICS PER TARGET (N={N} examples with prompts)")
    print("-" * 80)
    header = (
        f"{'target':25s} {'n':>5s} "
        f"{'acc':>8s} {'prec':>8s} {'rec':>8s} {'f1':>8s} "
        f"{'prec_mi':>8s} {'rec_mi':>8s} {'f1_mi':>8s} "
        f"{'answered%':>10s} {'in_label%':>10s} "
        f"{'n_app':>6s} {'acc_app':>8s}"
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
        n_applicable = sum(1 for g in all_true if _is_applicable_gold(g))

        answered_frac = n_answered / n_gold if n_gold > 0 else 0.0

        # "In label" = in allowed set for all targets (binary, multiclass, string).
        # Allowed comes from TARGETS in the notebook; string targets get allowed from the dataset.
        gold_label_set = set(all_true)
        spec = (targets_spec or {}).get(t) if targets_spec else None
        allowed = spec.get("allowed") if spec and isinstance(spec, dict) else None
        is_multi_value = bool(spec.get("multi_value")) if spec else False
        allow_other = bool(spec.get("allow_other_paren")) if spec else False
        if allowed is not None:
            allowed_set = {str(a).strip() for a in allowed}
            def _in_scope(p):
                # Multi-value: every ;-split atom must be a valid codebook label.
                atoms = _split_atoms(p) if is_multi_value else [str(p).strip()]
                return bool(atoms) and all(
                    _atom_in_scope(a, allowed_set, allow_other) for a in atoms
                )
        else:
            def _in_scope(p):
                return p in gold_label_set
        n_in_label = sum(1 for _, p in pairs if _in_scope(p))
        in_label_frac = n_in_label / n_gold if n_gold > 0 else 0.0

        is_string_target = spec and spec.get("type") == "string"
        answers_partially_correct = []
        acc_applicable = None

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

            # Replace None (model output "null") with a sentinel so sklearn doesn't
            # raise on mixed-type/None arrays; null preds count as wrong.
            y_pred_clean = [
                _PRED_NULL_SENTINEL if p is None else p for p in y_pred
            ]
            # Order-insensitive comparison for ;-joined multi-value targets (D1):
            # canonicalize gold and pred to sorted atoms so "a;b" == "b;a".
            if is_multi_value:
                y_true_cmp = [_canon_multi(g) for g in y_true]
                y_pred_cmp = [
                    p if p == _PRED_NULL_SENTINEL else _canon_multi(p)
                    for p in y_pred_clean
                ]
            else:
                y_true_cmp = y_true
                y_pred_cmp = y_pred_clean
            try:
                acc = accuracy_score(y_true_cmp, y_pred_cmp)
                prec, rec, f1, _ = precision_recall_fscore_support(
                    y_true_cmp, y_pred_cmp, average="macro", zero_division=0
                )
                prec_micro, rec_micro, f1_micro, _ = precision_recall_fscore_support(
                    y_true_cmp, y_pred_cmp, average="micro", zero_division=0
                )
            except Exception:
                acc = prec = rec = f1 = 0.0
                prec_micro = rec_micro = f1_micro = 0.0

            # Accuracy on the applicable subset only (gold != "not applicable").
            app_keep = [_is_applicable_gold(g) for g in y_true]
            if any(app_keep):
                yt_app = [g for g, k in zip(y_true_cmp, app_keep) if k]
                yp_app = [p for p, k in zip(y_pred_cmp, app_keep) if k]
                try:
                    acc_applicable = accuracy_score(yt_app, yp_app)
                except Exception:
                    acc_applicable = None
        else:
            acc = prec = rec = f1 = 0.0
            prec_micro = rec_micro = f1_micro = 0.0

        acc_app_str = f"{acc_applicable:8.3f}" if acc_applicable is not None else f"{'NA':>8s}"
        print(
            f"{t:25s} {n_gold:5d} "
            f"{acc:8.3f} {prec:8.3f} {rec:8.3f} {f1:8.3f} "
            f"{prec_micro:8.3f} {rec_micro:8.3f} {f1_micro:8.3f} "
            f"{answered_frac*100:10.1f} {in_label_frac*100:10.1f} "
            f"{n_applicable:6d} {acc_app_str}"
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
                "n_applicable": n_applicable,
                "accuracy": acc,
                "accuracy_applicable": acc_applicable,
                "precision_macro": prec,
                "recall_macro": rec,
                "f1_macro": f1,
                "precision_micro": prec_micro,
                "recall_micro": rec_micro,
                "f1_micro": f1_micro,
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
        "n_applicable": None,
        "accuracy": None,
        "accuracy_applicable": None,
        "precision_macro": None,
        "recall_macro": None,
        "f1_macro": None,
        "precision_micro": None,
        "recall_micro": None,
        "f1_micro": None,
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

    # JSON parse-rate row: how many completions we could parse a JSON object
    # from (n_answered / n_gold = parse rate). Unparseable items are excluded
    # from per-target metrics, so this records the coverage they reflect.
    parse_row = dict(timing_row)
    parse_row.update(
        {
            "target": "_json_parse",
            "n_gold": n_processed,
            "n_answered": parse_stats["ok"],
            "answered_frac": json_parse_frac,
            "total_inference_sec": None,
            "n_prompts": None,
            "avg_sec_per_prompt": None,
            "training_time_sec": None,
        }
    )
    rows.append(parse_row)

    print("=" * 80 + "\n")

    # Optionally save metrics to CSV (includes timing row)
    if results_folder is not None and rows:
        os.makedirs(results_folder, exist_ok=True)
        lr_str = f"{learning_rate}" if learning_rate is not None else "na"
        ep_str = f"{epoch}" if epoch is not None else "na"
        seed_str = f"{seed}" if seed is not None else "na"
        model_str = gemma_model if gemma_model is not None else "na"
        ts_str = run_id if run_id is not None else datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_name = f"{mtype}_{model_str}_{split_name}_metrics_lr{lr_str}_seed{seed_str}_epoch{ep_str}_{ts_str}.csv"
        csv_path = os.path.join(results_folder, csv_name)
        pd.DataFrame(rows).to_csv(csv_path, index=False)
        print(f"[{split_name}-metrics] Saved per-target metrics + timing to {csv_path}")

    return {
        "total_inference_sec": total_inference_sec,
        "n_prompts": N,
        "avg_sec_per_prompt": avg_sec_per_prompt,
        "json_parse_ok": parse_stats["ok"],
        "json_parse_total": n_processed,
        "json_parse_frac": json_parse_frac,
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
    max_text_tokens: int = 500,
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
        lora_alpha=128,
        lora_dropout=0.1,
        r=64,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )

    # Per-experiment results folder: group this run's metrics CSVs under a timestamped
    # subdir (run_id) so repeated jobs never overwrite, and persist the exact
    # LoRA/quant/training settings alongside the metrics.
    if results_folder is not None:
        results_run_dir = os.path.join(results_folder, f"{run_id}_{mtype}_{gemma_model}")
        os.makedirs(results_run_dir, exist_ok=True)
        experiment_config = {
            "run_id": run_id,
            "mtype": mtype,
            "model_id": model_id,
            "gemma_model": gemma_model,
            "lora": {
                "r": peft_config.r,
                "alpha": peft_config.lora_alpha,
                "dropout": peft_config.lora_dropout,
                "scaling": peft_config.lora_alpha / peft_config.r if peft_config.r else None,
                "bias": peft_config.bias,
                "task_type": str(peft_config.task_type),
                "target_modules": list(peft_config.target_modules),
            },
            "quant": {
                "load_in_4bit": True,
                "quant_type": "nf4",
                "compute_dtype": str(compute_dtype),
                "double_quant": False,
            },
            "training": {
                "learning_rates": list(learning_rates),
                "epochs": epochs,
                "batch_size": batch_size,
                "grad_accum_steps": grad_accum_steps,
                "effective_batch_size": batch_size * grad_accum_steps,
                "early_stopping_patience": early_stopping_patience,
                "train_val_seeds": list(train_val_seeds),
                "val_size": val_size,
            },
            "lengths": {
                "max_tokens": max_tokens,
                "max_new_tokens": max_new_tokens,
                "max_text_tokens": max_text_tokens,
            },
            "data": {
                "n_train_total": len(train_df),
                "n_test": len(test_df),
            },
        }
        with open(os.path.join(results_run_dir, "experiment_config.json"), "w") as _f:
            json.dump(experiment_config, _f, indent=2, ensure_ascii=False)
        print(f"[results] Per-run metrics + config under: {results_run_dir}")
        results_folder = results_run_dir

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
        max_text_tokens=max_text_tokens,
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
            if val_size is not None and val_size <= 0:
                # Full-train mode: no validation split (e.g. gemma3_finetune_fulltrain.py)
                train_rows = train_df
                val_rows = train_df.iloc[0:0]
                print(f"Train: {len(train_rows)} rows  |  Val: 0 (full-train, no validation split)")
            else:
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
                max_text_tokens=max_text_tokens,
            )

            val_dataset = build_simple_sft_dataset(
                df=val_rows,
                tokenizer=tokenizer,
                prompt_template=prompt,
                text_col=text_col,
                answer_col=answer_col,
                max_seq_length=max_tokens,
                system_prompt=system_prompt,
                max_text_tokens=max_text_tokens,
            )

            # Validation prompts for inference (same transcript cap as train)
            val_prompts, val_gold_raw = build_simple_val_prompts(
                df=val_rows,
                tokenizer=tokenizer,
                prompt_template=prompt,
                text_col=text_col,
                answer_col=answer_col,
                max_seq_length=max_tokens,
                system_prompt=system_prompt,
                max_text_tokens=max_text_tokens,
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

            # ── Run config summary (one greppable block with the full setup) ──
            eff_batch = batch_size * grad_accum_steps
            steps_per_epoch = (len(dataset) + eff_batch - 1) // eff_batch if eff_batch else 0
            lora_scaling = peft_config.lora_alpha / peft_config.r if peft_config.r else 0.0
            print("\n" + "=" * 80)
            print("RUN CONFIG SUMMARY")
            print("=" * 80)
            print(f"  model         : {original_model}  (gemma_model={gemma_model})")
            print(f"  compute_dtype : {compute_dtype}")
            print(f"  LoRA          : r={peft_config.r}  alpha={peft_config.lora_alpha}  "
                  f"scaling={lora_scaling:.3f}  dropout={peft_config.lora_dropout}")
            print(f"  optim         : lr={learning_rate}  epochs={epochs}  seed={train_val_seed}")
            print(f"  batch         : per_device={batch_size}  grad_accum={grad_accum_steps}  "
                  f"effective={eff_batch}  steps/epoch≈{steps_per_epoch}")
            print(f"  lengths       : max_tokens={max_tokens}  max_new_tokens={max_new_tokens}  "
                  f"max_text_tokens={max_text_tokens}")
            print(f"  data          : train={len(dataset)}  val={len(val_rows)}  test={len(test_prompts)}")
            print("=" * 80 + "\n")

            # ── 3) Training arguments ─────────────────────────────────
            if base_run_dir is not None:
                trainer_output_dir = os.path.join(
                    base_run_dir, f"trainer_lr{learning_rate}_seed{train_val_seed}"
                )
                model_save_dir = os.path.join(
                    base_run_dir, f"{mtype}_{gemma_model}_lr{learning_rate}_seed{train_val_seed}_{run_id}"
                )
                os.makedirs(trainer_output_dir, exist_ok=True)
                os.makedirs(model_save_dir, exist_ok=True)
            else:
                trainer_output_dir = tempfile.mkdtemp(prefix="africa_llm_simple_")
                model_save_dir = None
            have_eval = len(val_dataset) > 0
            # TRL 0.18: SFT-specific fields (dataset_text_field / max_seq_length / packing)
            # live on SFTConfig, not on SFTTrainer; SFTConfig subclasses TrainingArguments.
            training_args = SFTConfig(
                output_dir=trainer_output_dir,
                num_train_epochs=1,
                per_device_train_batch_size=batch_size,
                per_device_eval_batch_size=batch_size,
                gradient_accumulation_steps=grad_accum_steps,
                optim="paged_adamw_32bit",
                do_eval=have_eval,
                eval_strategy="epoch" if have_eval else "no",
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
                dataset_text_field="text",
                max_seq_length=max_tokens,
                packing=False,
            )

            # ── 3b) Completion-only loss: mask prompt, train only on the answer ──
            # Gemma-3 opens the assistant turn with "<start_of_turn>model\n". Mask
            # every label up to and INCLUDING that template, so the loss is computed
            # only on the answer JSON + its trailing <end_of_turn>. Pass token IDs
            # (not the raw string) to avoid the standalone-vs-in-context tokenization
            # mismatch. packing must stay False (it is, above).
            response_template_ids = tokenizer.encode(
                "<start_of_turn>model\n", add_special_tokens=False
            )
            completion_collator = DataCollatorForCompletionOnlyLM(
                response_template=response_template_ids,
                tokenizer=tokenizer,
            )

            # One-shot mask check: supervised fraction should be ~3-5% (answer only),
            # and the supervised text should be exactly the answer JSON + <end_of_turn>.
            # Tokenize like the real trainer (add_special_tokens=True) and measure the
            # fraction over real (non-pad) tokens so the printed % is faithful.
            _n_check = min(8, len(dataset))
            _verify_feats = [
                {"input_ids": tokenizer(t, add_special_tokens=True)["input_ids"]}
                for t in dataset["text"][:_n_check]
            ]
            _verify_batch = completion_collator(_verify_feats)
            _attn = _verify_batch.get("attention_mask")
            _fracs = []
            for _i in range(_verify_batch["labels"].size(0)):
                _lab = _verify_batch["labels"][_i]
                _keep = _lab != -100
                _n_sup = int(_keep.sum())
                _real = int(_attn[_i].sum()) if _attn is not None else int(_lab.numel())
                _frac = _n_sup / _real if _real else 0.0
                _fracs.append(_frac)
                if _i < 2:
                    _kept_ids = _lab[_keep].tolist()
                    print(f"[verify-mask] ex{_i}: {_n_sup}/{_real} non-pad "
                          f"tokens supervised ({_frac:.1%})")
                    print(f"[verify-mask] ex{_i} supervised text: "
                          f"{tokenizer.decode(_kept_ids)!r}")
            _mean_frac = sum(_fracs) / len(_fracs) if _fracs else 0.0
            print(f"[verify-mask] mean supervised fraction over {len(_fracs)} examples: {_mean_frac:.2%}")
            if _mean_frac <= 0.0:
                print("[verify-mask] *** WARNING: 0% supervised — response template NOT found; "
                      "loss would train on nothing. Check response_template_ids / packing=False. ***")
            elif _mean_frac > 0.5:
                print("[verify-mask] *** WARNING: >50% supervised — prompt likely NOT masked; "
                      "loss may include the codebook. Check the response template. ***")

            # ── 4) Trainer ────────────────────────────────────────────
            trainer = SFTTrainer(
                model=model,
                train_dataset=dataset,
                eval_dataset=val_dataset if have_eval else None,
                processing_class=tokenizer,
                args=training_args,
                data_collator=completion_collator,
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

                # Extract losses from log history (eval_loss only when do_eval=True)
                train_loss = trainer.state.log_history[-1].get("train_loss", None)
                eval_loss = None
                if have_eval and len(trainer.state.log_history) >= 2:
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
                    training_time_sec=elapsed,
                    targets_spec=targets_spec,
                    gemma_model=gemma_model,
                    run_id=run_id,
                    compute_dtype=compute_dtype,
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
                    gemma_model=gemma_model,
                    run_id=run_id,
                    compute_dtype=compute_dtype,
                )

                if gpu_avail:
                    torch.cuda.empty_cache()

                # Early stopping + save best model
                def _save_checkpoint(reason: str):
                    if model_save_dir is None:
                        return
                    print(f"[save] {reason} — saving adapter/processor to {model_save_dir}")
                    model.save_pretrained(model_save_dir)
                    processor.save_pretrained(model_save_dir)
                    run_cfg = {
                        "model_id": model_id,
                        "system_prompt": system_prompt,
                        "prompt_template": prompt,
                        "max_tokens": max_tokens,
                        "max_new_tokens": max_new_tokens,
                        "targets_spec": targets_spec,
                        "gemma_model": gemma_model,
                    }
                    with open(os.path.join(model_save_dir, "run_config.json"), "w") as _f:
                        json.dump(run_cfg, _f, indent=2, ensure_ascii=False)

                if eval_loss is not None:
                    if eval_loss < best_eval_loss:
                        best_eval_loss = eval_loss
                        no_improvement_counter = 0
                        _save_checkpoint(f"New best eval_loss={eval_loss:.4f}")
                    else:
                        no_improvement_counter += 1

                    if no_improvement_counter >= early_stopping_patience:
                        print(f"Early stopping after {ep+1} epochs.")
                        break
                else:
                    # No eval loss available — save every epoch as fallback
                    _save_checkpoint(f"No eval_loss (epoch {ep})")

            # ── Cleanup ───────────────────────────────────────────────
            print("Training done, cleaning up")
            del model
            gc.collect()
            if gpu_avail:
                torch.cuda.empty_cache()
            fold_counter += 1

    return cv_performances
