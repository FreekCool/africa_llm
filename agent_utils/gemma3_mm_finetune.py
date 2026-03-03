# agent_utils/gemma3_finetune.py

import os
import gc
import re
import math
import time
import json
import datetime

import torch
import pandas as pd
from trl import SFTTrainer
from datasets import Dataset
from collections import defaultdict
from sklearn.model_selection import train_test_split

from transformers import AutoProcessor, Gemma3ForConditionalGeneration
from peft import PeftModel, prepare_model_for_kbit_training, get_peft_model

from collections import defaultdict, Counter
from PIL import Image
from sklearn.metrics import precision_recall_fscore_support

from .utils import (
    setup_seed,
    set_quant_and_peft_config_gemma,     # ✅ add in utils.py (below)
    set_training_params_gemma,           # ✅ add in utils.py (below)
    create_result_filename,
    create_model_dirname,
    handle_follow_up_prompt_llama_standard,  # not used here, but kept 1:1 like llama file imports
    print_gpu_memory,
    preprocess_function,                    # not used here, but kept 1:1 like llama file imports
    build_sft_dataset,
    build_split_data,
)

from .eval_utils import (
    run_taskwise_inference,
    evaluate_predictions_binary,
    evaluate_predictions_multiclass,
)


def run_fine_tuned_gemma3(
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
    model_dir,
    llm_answer_col,
    max_tokens,
    batch_size,
    class_labels=None,
    max_new_tokens=3,
    cache_dir=None,
    local_model=None,
    text_only_res=None,
    image_folder=None,
    early_stopping_patience=3,
    epochs=10,
    learning_rates=(0.0002,),
):

    mtype = "fine_tuned_gemma3"

    # ---------------- GPU / device setup ----------------
    gpu_avail = torch.cuda.is_available()
    mps_avail = torch.backends.mps.is_available()

    if gpu_avail:
        print("GPU available")
        device = torch.device("cuda")
    elif mps_avail:
        print("MPS GPU available")
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # NVML is CUDA-only; guard it so MPS/CPU runs don't crash
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

    # Row which signifies the task
    task_col = "target_name"

    # ---------------- Model & tokenizer -----------------
    print("Starting to fine-tune Gemma-3")

    if local_model is None:
        model_name = "google/gemma-3-4b-it"
        loaded_locally = False
        processor = AutoProcessor.from_pretrained(model_name, cache_dir=cache_dir)
        original_model = model_name
    else:
        model_name = local_model
        print("Loading local model")
        loaded_locally = True
        processor = AutoProcessor.from_pretrained(
            "google/gemma-3-4b-it", cache_dir=cache_dir
        )
        original_model = "google/gemma-3-4b-it"

    # dtype consistent with llama file
    compute_dtype = getattr(torch, "float16")
    if gpu_avail and torch.cuda.is_bf16_supported():
        # optional but safe: bf16 on supported GPUs
        compute_dtype = getattr(torch, "bfloat16")

    tokenizer = getattr(processor, "tokenizer", processor)

    # Ensure padding tokens are set (Gemma sometimes needs this)
    if getattr(tokenizer, "pad_token", None) is None and getattr(tokenizer, "eos_token", None) is not None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"

    quant_config, peft_args = set_quant_and_peft_config_gemma(compute_dtype)

    if pynvml and handle:
        print("Initial GPU usage")
        print_gpu_memory(handle, pynvml)

    cv_performances = pd.DataFrame()

    # -------------- Main LR loop --------------
    for learning_rate in learning_rates:
        fold_counter = 0

        for train_val_seed in train_val_seeds:
            setup_seed(train_val_seed)
            print(f"=== LR {learning_rate} | seed {train_val_seed} | fold {fold_counter} ===")

            # -------- multi-task train / val split ----------
            train_rows, val_rows = train_test_split(
                train_df,
                test_size=val_size,
                random_state=train_val_seed
            )

            # Optional sanity checks: label distribution
            print("Train label counts:")
            print(train_rows[target_col].value_counts())

            print("\nVal label counts:")
            print(val_rows[target_col].value_counts())

            print("\nTest label counts:")
            print(test_df[target_col].value_counts())

            # ======================
            # 1) BUILD TRAIN DATASET
            # ======================
            dataset = build_sft_dataset(
                df=train_rows,
                tokenizer=tokenizer,
                max_tokens=max_tokens,
                text_col=text_col,
                first_prompt_col=first_prompt_col,
                llm_answer_col=llm_answer_col,
                image_path = 
            )
            print(f"Dataset loaded for fold {fold_counter}")

            # =====================
            # 2) BUILD VAL PROMPTS + VAL DATASET
            # =====================
            (
                val_prompts,
                val_texts,
                val_task_names,
                val_class_labels_list,
                val_labels,
                val_followup_prompts,
            ) = build_split_data(
                df=val_rows,
                tokenizer=tokenizer,
                max_tokens=max_tokens,
                text_col=text_col,
                first_prompt_col=first_prompt_col,
                second_prompt_col=second_prompt_col,
                target_col=target_col,
                task_col=task_col,
            )

            val_dataset = build_sft_dataset(
                df=val_rows,
                tokenizer=tokenizer,
                max_tokens=max_tokens,
                text_col=text_col,
                first_prompt_col=first_prompt_col,
                llm_answer_col=llm_answer_col,
            )

            # ====================
            # 3) MODEL + TRAINER
            # ====================
            if loaded_locally:
                cv_performances = pd.read_csv(text_only_res)
                print("Reloaded previous results")
            else:
                cv_performances = pd.DataFrame()

            filename = create_result_filename(
                target_col, mtype, learning_rate, train_val_seed
            )
            full_file_path = os.path.join(results_folder, filename)

            print(f"Loading model for LR {learning_rate}")
            model = Gemma3ForConditionalGeneration.from_pretrained(
                original_model,
                quantization_config=quant_config,
                device_map={"": 0} if gpu_avail else None,
                cache_dir=cache_dir,
                torch_dtype=compute_dtype if gpu_avail else None,
            )

            model.config.use_cache = False
            model.config.pad_token_id = tokenizer.pad_token_id

            # prepare for k-bit training (same as llama)
            if gpu_avail and quant_config is not None:
                model = prepare_model_for_kbit_training(model)

            if local_model is not None:
                model = PeftModel.from_pretrained(
                    model,
                    local_model,
                    is_trainable=True
                )
            else:
                model = get_peft_model(model, peft_args)

            print(f"Model loaded for fold {fold_counter}")

            training_params = set_training_params_gemma(
                train_val_seed, batch_size, learning_rate
            )

            trainer = SFTTrainer(
                model=model,
                train_dataset=dataset,
                eval_dataset=val_dataset,
                peft_config=peft_args,
                dataset_text_field="text",
                max_seq_length=max_tokens,
                tokenizer=tokenizer,
                args=training_params,
            )
            print("Columns in trainer.train_dataset:", trainer.train_dataset.column_names)
            print("Trainer setup complete")

            if loaded_locally:
                offset = cv_performances["epoch"].max() + 1
                max_epochs = epochs - offset
            else:
                offset = 0
                max_epochs = epochs

            best_eval_loss = float("inf")
            no_improvement_counter = 0

            # ====================
            # 4) EPOCH LOOP
            # ====================
            for i in range(max_epochs):
                start_time = time.time()
                current_epoch = offset + i
                print(f"Epoch {current_epoch} | LR {learning_rate}")

                # ----- TRAIN -----
                trainer.train()

                if pynvml and handle:
                    print("GPU memory after training:")
                    print_gpu_memory(handle, pynvml)

                print(trainer.state.log_history)
                train_loss = trainer.state.log_history[-1]["train_loss"]
                eval_loss = trainer.state.log_history[-2]["eval_loss"]

                if pynvml and handle:
                    print("GPU memory after getting the loss:")
                    print_gpu_memory(handle, pynvml)

                # Early stopping bookkeeping
                if eval_loss < best_eval_loss:
                    best_eval_loss = eval_loss
                    no_improvement_counter = 0
                else:
                    no_improvement_counter += 1

                if no_improvement_counter >= early_stopping_patience:
                    print(
                        f"Early stopping triggered after {i+1} epochs with no improvement."
                    )
                    break

                # ============================
                # 5) VALIDATION INFERENCE (multi-task aware, order preserved)
                # ============================
                val_cv = run_taskwise_inference(
                    prompts=val_prompts,
                    texts=val_texts,
                    task_names=val_task_names,
                    class_labels_list=val_class_labels_list,
                    labels=val_labels,
                    followup_prompts=val_followup_prompts,
                    trainer=trainer,
                    tokenizer=tokenizer,
                    device=device,
                    max_new_tokens=max_new_tokens,
                    mtype=mtype,
                    epoch=current_epoch,
                    fold=fold_counter,
                    fine_tune_type="validation",
                    split_name="val",
                    train_loss=train_loss,
                    val_loss=eval_loss,
                    elapsed_time="na",
                    seed=train_val_seed,
                )

                cv_performances = pd.concat(
                    [cv_performances, val_cv], ignore_index=True
                )
                cv_performances.to_csv(full_file_path, index=False)

                # =======================
                # 6) TEST INFERENCE (multi-task, seen tasks)
                # =======================
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
                )

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
                    mtype=mtype,
                    epoch=current_epoch,
                    fold=fold_counter,
                    fine_tune_type="test",
                    split_name="test",
                    train_loss="na",
                    val_loss="na",
                    elapsed_time="na",
                    seed=train_val_seed,
                )

                cv_performances = pd.concat(
                    [cv_performances, test_cv], ignore_index=True
                )
                cv_performances.to_csv(full_file_path, index=False)

                # ==========================
                # 7) TEST INFERENCE (UNSEEN TASKS)
                # ==========================
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
                    )

                    end_time = time.time()
                    elapsed_time = end_time - start_time

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
                        mtype=mtype,
                        epoch=current_epoch,
                        fold=fold_counter,
                        fine_tune_type="test_unseen",
                        split_name="test_unseen",
                        train_loss="na",
                        val_loss="na",
                        elapsed_time=elapsed_time,
                        seed=train_val_seed,
                    )

                    cv_performances = pd.concat([cv_performances, unseen_cv], ignore_index=True)
                    cv_performances.to_csv(full_file_path, index=False)
                else:
                    print("No unseen tasks provided; skipping unseen evaluation.")

                # -------- SAVE MODEL --------
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                model_dirname = create_model_dirname(
                    target_col, mtype, learning_rate, train_val_seed, timestamp
                )
                if not os.path.exists(model_dirname):
                    os.makedirs(model_dirname)

                full_save_path = os.path.join(model_dir, model_dirname)
                print(f"Saving model to: {full_save_path}")
                model.save_pretrained(full_save_path)
                tokenizer.save_pretrained(full_save_path)

            print("Model saved, cleaning up")
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            fold_counter += 1

    return cv_performances