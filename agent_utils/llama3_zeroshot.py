# agent_utils/llama3_zeroshot.py

import os
import time

import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM

from .utils import (
    setup_seed,
    set_quant_and_peft_config_llama,
    build_split_data,
    create_result_filename,
    print_gpu_memory,
)

from .eval_utils import run_taskwise_inference


def run_llama3_base(
    train_df,
    test_df,
    test_unseen_df,
    text_col,
    target_col,
    first_prompt_col,
    second_prompt_col,
    train_val_seeds,
    val_size,
    results_folder,
    max_tokens,
    class_labels=None,
    max_new_tokens=3,
    cache_dir=None,
    local_model=None,
    text_only_res=None,
):
    """
    Zero-shot / non-finetuned LLaMA-3 evaluation.

    - Loads the pretrained LLaMA-3 model (optionally local).
    - Ignores training/validation (no fine-tuning).
    - Builds prompts with build_split_data.
    - Runs test / test_unseen via run_taskwise_inference.
    - NO training, NO LoRA.
    """

    mtype_label = "llama3"  # how it will appear in the CSV

    # ---------------- GPU / device setup ----------------
    gpu_avail = torch.cuda.is_available()
    mps_avail = torch.backends.mps.is_available()

    if gpu_avail:
        print("GPU available")
        device = torch.device("cuda")
    elif mps_avail:
        print("MPS available")
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # ---------------- NVML (CUDA-only) ----------------
    pynvml = None
    handle = None
    if gpu_avail:
        try:
            import pynvml as _pynvml

            _pynvml.nvmlInit()
            handle = _pynvml.nvmlDeviceGetHandleByIndex(0)
            pynvml = _pynvml
        except Exception as e:
            print(f"[WARN] NVML unavailable ({e}). Continuing without GPU mem prints.")

    # Column that identifies the task
    task_col = "target_name"

    # ---------------- Seed ----------------
    seed = train_val_seeds[0] if train_val_seeds else 42
    setup_seed(seed)

    # ---------------- Model & tokenizer -----------------
    print("Running zero-shot LLaMA-3 (no fine-tuning)")

    model_name = local_model or "meta-llama/Meta-Llama-3-8B-Instruct"
    if local_model:
        print(f"Loading local model: {model_name}")

    # dtype selection
    if gpu_avail:
        compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    elif mps_avail:
        compute_dtype = torch.float16
    else:
        compute_dtype = torch.float32

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        trust_remote_code=False,  # not needed for LLaMA-3; safer default
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"

    # Quantization: only safe on CUDA (bitsandbytes). On MPS/CPU, skip.
    quant_config = None
    if gpu_avail:
        try:
            quant_config, _ = set_quant_and_peft_config_llama(
                torch.float16 if compute_dtype == torch.float16 else torch.bfloat16
            )
        except Exception as e:
            print(f"[WARN] Quant config failed ({e}). Loading without quantization.")
            quant_config = None

    if pynvml and handle:
        print("Initial GPU usage (before loading base model)")
        print_gpu_memory(handle, pynvml)

    print("Loading base LLaMA-3 model (no LoRA, no fine-tuning)...")

    # Loading behavior:
    # - CUDA: device_map="auto" (or single GPU auto placement). Optional 4-bit quant.
    # - MPS/CPU: load normally then .to(device)
    if gpu_avail:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quant_config,
            torch_dtype=compute_dtype,
            device_map="auto",
            cache_dir=cache_dir,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=compute_dtype,
            cache_dir=cache_dir,
        ).to(device)

    model.eval()
    model.config.use_cache = True
    model.config.pad_token_id = tokenizer.pad_token_id

    # simple wrapper so we can reuse run_taskwise_inference (expects trainer.model)
    class DummyTrainer:
        def __init__(self, model):
            self.model = model

    trainer = DummyTrainer(model)
    cv_performances = pd.DataFrame()

    print("Label counts (for info only):")
    print("\nTrain label counts:")
    print(train_df[target_col].value_counts())
    print("\nTest label counts:")
    print(test_df[target_col].value_counts())
    if test_unseen_df is not None and not test_unseen_df.empty:
        print("\nTest-unseen label counts:")
        print(test_unseen_df[target_col].value_counts())

    # -------- build test/test_unseen splits --------
    (
        test_prompts,
        test_texts,
        test_task_names,
        test_class_labels_list,
        test_labels,
        test_followup_prompts,
    ) = build_split_data(
        df=test_df,
        tokenizer=tokenizer,
        max_tokens=max_tokens,
        text_col=text_col,
        first_prompt_col=first_prompt_col,
        second_prompt_col=second_prompt_col,
        target_col=target_col,
        task_col=task_col,
        mtype="llama3",  # ✅ important if you added mtype switching in utils.py
    )

    if test_unseen_df is not None and not test_unseen_df.empty:
        (
            unseen_prompts,
            unseen_texts,
            unseen_task_names,
            unseen_class_labels_list,
            unseen_labels,
            unseen_followup_prompts,
        ) = build_split_data(
            df=test_unseen_df,
            tokenizer=tokenizer,
            max_tokens=max_tokens,
            text_col=text_col,
            first_prompt_col=first_prompt_col,
            second_prompt_col=second_prompt_col,
            target_col=target_col,
            task_col=task_col,
            mtype="llama3",
        )
    else:
        unseen_prompts = []
        unseen_texts = []
        unseen_task_names = []
        unseen_class_labels_list = []
        unseen_labels = []
        unseen_followup_prompts = []

    # -------- filename for this run --------
    os.makedirs(results_folder, exist_ok=True)
    filename = create_result_filename(target_col, mtype_label, "base", seed)
    full_file_path = os.path.join(results_folder, filename)

    # ============================
    # TEST (seen tasks)
    # ============================
    start_test = time.time()
    test_cv = run_taskwise_inference(
        prompts=test_prompts,
        texts=test_texts,
        task_names=test_task_names,
        class_labels_list=test_class_labels_list,
        labels=test_labels,
        followup_prompts=test_followup_prompts,
        trainer=trainer,
        tokenizer=tokenizer,
        device=device,
        max_new_tokens=max_new_tokens,
        mtype="llama3",
        epoch=0,
        fold=0,
        fine_tune_type="test",
        split_name="test",
        train_loss="na",
        val_loss="na",
        elapsed_time=time.time() - start_test,
    )
    cv_performances = pd.concat([cv_performances, test_cv], ignore_index=True)

    # ============================
    # TEST (unseen tasks)
    # ============================
    if len(unseen_prompts) > 0:
        start_unseen = time.time()
        unseen_cv = run_taskwise_inference(
            prompts=unseen_prompts,
            texts=unseen_texts,
            task_names=unseen_task_names,
            class_labels_list=unseen_class_labels_list,
            labels=unseen_labels,
            followup_prompts=unseen_followup_prompts,
            trainer=trainer,
            tokenizer=tokenizer,
            device=device,
            max_new_tokens=max_new_tokens,
            mtype="llama3",
            epoch=0,
            fold=0,
            fine_tune_type="test_unseen",
            split_name="test_unseen",
            train_loss="na",
            val_loss="na",
            elapsed_time=time.time() - start_unseen,
        )
        cv_performances = pd.concat([cv_performances, unseen_cv], ignore_index=True)

    if "model" in cv_performances.columns:
        cv_performances["model"] = mtype_label

    cv_performances.to_csv(full_file_path, index=False)
    return cv_performances