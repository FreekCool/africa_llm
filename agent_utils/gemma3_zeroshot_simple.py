# agent_utils/gemma3_zeroshot_simple.py
"""
Zero-shot evaluation using the off-the-shelf Gemma-3 4B-IT or 27B-IT model.

Mirrors the prompt format and inference pipeline of ``gemma3_finetune_simple.py``
(JSON answer, optional system prompt, KV prefix caching) but does NOT fine-tune —
the base instruction-tuned model is loaded and evaluated directly on the test set.

Usage from train_validate:
    mtype = "zeroshot_simple_gemma3"
"""

import os
import gc
import datetime

import torch
import pandas as pd
from transformers import (
    AutoProcessor,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)

from .utils import (
    setup_seed,
    print_gpu_memory,
)

from .gemma3_finetune_simple import (
    GEMMA_MODEL_IDS,
    build_simple_val_prompts,
    run_simple_val_inference,
    _precompute_prefix_kv,
)


def run_zeroshot_simple_gemma3(
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
    use_4bit: bool = False,
):
    """
    Zero-shot evaluation of the off-the-shelf Gemma-3 IT model.

    Accepts the same signature as ``run_simple_gemma3`` so it can be swapped
    in via ``mtype`` in ``train_validate`` without changing any other arguments.
    Training-specific args (epochs, learning_rates, …) are accepted but ignored.

    Parameters
    ----------
    gemma_model : "4b" | "27b" | full HuggingFace model ID
    use_4bit    : Load model in 4-bit NF4 (useful for 27B on a single GPU).
    """
    mtype = "zeroshot_simple_gemma3"
    run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    if gemma_model in GEMMA_MODEL_IDS:
        model_id = GEMMA_MODEL_IDS[gemma_model]
    else:
        model_id = gemma_model  # full HuggingFace model id
    if local_model is not None:
        model_id = local_model
    print(f"Gemma model: {gemma_model} -> {model_id}")

    # ── GPU / device ──────────────────────────────────────────────────
    gpu_avail = torch.cuda.is_available()
    mps_avail = torch.backends.mps.is_available()

    if gpu_avail:
        device = torch.device("cuda")
        compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    elif mps_avail:
        device = torch.device("mps")
        compute_dtype = torch.float16
    else:
        device = torch.device("cpu")
        compute_dtype = torch.float32

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
    print(f"Loading tokenizer from {model_id}")
    processor = AutoProcessor.from_pretrained(model_id, cache_dir=cache_dir)
    tokenizer = getattr(processor, "tokenizer", processor)

    if getattr(tokenizer, "pad_token", None) is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"

    if pynvml_mod and handle:
        print("Initial GPU usage (before model load)")
        print_gpu_memory(handle, pynvml_mod)

    # ── Load base model (no LoRA, no fine-tuning) ─────────────────────
    print(f"Loading base Gemma-3 model ({model_id}) — zero-shot, no fine-tuning")
    load_kwargs = dict(
        torch_dtype=compute_dtype,
        cache_dir=cache_dir,
    )
    if use_4bit and gpu_avail:
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=False,
        )

    if gpu_avail:
        load_kwargs["device_map"] = "auto"
        model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs).to(device)

    model.eval()

    if pynvml_mod and handle:
        print("GPU usage after model load")
        print_gpu_memory(handle, pynvml_mod)

    # Simple wrapper so run_simple_val_inference (expects trainer.model) works
    class DummyTrainer:
        def __init__(self, m):
            self.model = m

    trainer = DummyTrainer(model)

    # ── Build TEST prompts ─────────────────────────────────────────────
    seed = train_val_seeds[0] if train_val_seeds else 42
    setup_seed(seed)

    test_prompts, test_gold_raw = build_simple_val_prompts(
        df=test_df,
        tokenizer=tokenizer,
        prompt_template=prompt,
        text_col=text_col,
        answer_col=answer_col,
        max_seq_length=max_tokens,
        system_prompt=system_prompt,
    )
    print(f"[zeroshot-simple] Built {len(test_prompts)} test prompts")

    os.makedirs(results_folder, exist_ok=True)

    # ── KV prefix cache for system prompt ─────────────────────────────
    prefix_kv, prefix_len = _precompute_prefix_kv(model, tokenizer, system_prompt, device)

    # ── Run inference on test set ──────────────────────────────────────
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
        learning_rate=None,
        epoch=0,
        seed=seed,
        split_name="test",
        training_time_sec=None,
        targets_spec=targets_spec,
        prefix_kv=prefix_kv,
        prefix_len=prefix_len,
        inference_batch_size=inference_batch_size,
        stop_on_complete_json=stop_on_complete_json,
        gemma_model=gemma_model,
        run_id=run_id,
    )

    # ── Cleanup ────────────────────────────────────────────────────────
    del prefix_kv
    del model
    gc.collect()
    if gpu_avail:
        torch.cuda.empty_cache()

    # Return empty DataFrame for compatibility with train_validate's cv_performances concat
    return pd.DataFrame()
