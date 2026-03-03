# agent_utils/llama3_ilora_finetune.py
"""
ILoRA-enhanced fine-tuning for LLaMA-3.

Integrates Incremental LoRA (ILoRA) with:
  - EMA (Exponential Moving Average) adapter for stability
  - Experience replay buffer for rehearsal
  - Consistency regularization between plastic and stable hidden states

This mitigates catastrophic forgetting on unseen annotation tasks during
multi-task fine-tuning.

Reference:
  Ren et al., "Analyzing and Reducing Catastrophic Forgetting in
  Parameter Efficient Tuning" (arXiv:2402.18865)
"""

import os
import gc
import time
import datetime

import torch
import pynvml
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import (
    AutoTokenizer,
    AutoProcessor,
    AutoModelForCausalLM,
)
from peft import (
    PeftModel,
    prepare_model_for_kbit_training,
    get_peft_model,
)

from .utils import (
    setup_seed,
    set_quant_and_peft_config_llama,
    set_training_params_llama,
    create_result_filename,
    create_model_dirname,
    print_gpu_memory,
    build_sft_dataset,
    build_split_data,
)

from .eval_utils import run_taskwise_inference
from .ilora_utils import ILoRASFTTrainer


# ==============================================================================
# MAIN ENTRY POINT
# ==============================================================================
def run_fine_tuned_llama3_ilora(
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
    early_stopping_patience=3,
    epochs=10,
    learning_rates=(0.0002,),
    # --- ILoRA-specific hyperparameters ---
    buffer_size=500,
    ema_alpha=0.25,
    reg_weight=1.0,
):
    """
    Fine-tune LLaMA-3 with ILoRA consistency regularisation.

    The evaluation structure is **identical** to ``run_fine_tuned_llama3``:
      - Per-epoch validation, test (seen tasks), and test_unseen inference
      - Early stopping on eval_loss
      - Model checkpointing

    Extra ILoRA parameters
    ----------------------
    buffer_size : int
        Number of examples kept in the replay buffer (default 500).
    ema_alpha : float
        EMA smoothing coefficient (default 0.25; higher → slower update).
    reg_weight : float
        Weight of the consistency regularisation loss (default 1.0).
    """

    mtype = "ilora_llama3"

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

    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)

    # Row which signifies the task
    task_col = "target_name"

    # ---------------- Model & tokenizer -----------------
    print(
        f"Starting ILoRA fine-tune of LLaMA-3  "
        f"(buffer={buffer_size}, ema_alpha={ema_alpha}, reg_weight={reg_weight})"
    )

    if local_model is None:
        model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
        loaded_locally = False
        processor = AutoProcessor.from_pretrained(
            model_name, do_image_splitting=False
        )
        original_model = model_name
    else:
        model_name = local_model
        print("Loading local model")
        loaded_locally = True
        processor = AutoProcessor.from_pretrained(
            "meta-llama/Meta-Llama-3-8B-Instruct",
            do_image_splitting=False,
        )
        original_model = "meta-llama/Meta-Llama-3-8B-Instruct"

    compute_dtype = getattr(torch, "float16")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"

    quant_config, peft_args = set_quant_and_peft_config_llama(compute_dtype)

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
                random_state=train_val_seed,
            )

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
            model = AutoModelForCausalLM.from_pretrained(
                original_model,
                quantization_config=quant_config,
                device_map={"": 0},
                cache_dir=cache_dir,
            )
            model.config.use_cache = False
            model.config.pretraining_tp = 1
            model.config.pad_token_id = tokenizer.pad_token_id

            model = prepare_model_for_kbit_training(model)

            if local_model is not None:
                model = PeftModel.from_pretrained(
                    model,
                    local_model,
                    is_trainable=True,
                )
            else:
                model = get_peft_model(model, peft_args)

            print(f"Model loaded for fold {fold_counter}")

            training_params = set_training_params_llama(
                train_val_seed, batch_size, learning_rate
            )

            # ---- ILoRA trainer (replaces plain SFTTrainer) ---------
            trainer = ILoRASFTTrainer(
                buffer_size=buffer_size,
                ema_alpha=ema_alpha,
                reg_weight=reg_weight,
                model=model,
                train_dataset=dataset,
                eval_dataset=val_dataset,
                dataset_text_field="text",
                max_seq_length=max_tokens,
                tokenizer=tokenizer,
                args=training_params,
            )
            print("Columns in trainer.train_dataset:", trainer.train_dataset.column_names)
            print("Trainer setup complete (ILoRA)")

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

                print("GPU memory after training:")
                print_gpu_memory(handle, pynvml)

                print(trainer.state.log_history)
                train_loss = trainer.state.log_history[-1]["train_loss"]
                eval_loss = trainer.state.log_history[-2]["eval_loss"]

                # Log ILoRA-specific metrics
                ilm = trainer._ilora_last_metrics
                if ilm:
                    print(
                        f"[ILoRA] last step — task_loss={ilm.get('task_loss', 0):.4f}  "
                        f"buffer_loss={ilm.get('buffer_loss', 0):.4f}  "
                        f"consistency_loss={ilm.get('consistency_loss', 0):.4f}  "
                        f"total_loss={ilm.get('total_loss', 0):.4f}  "
                        f"buffer_filled={ilm.get('buffer_filled', 0)}"
                    )

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
                        f"Early stopping triggered after {i+1} epochs "
                        f"with no improvement."
                    )
                    break

                # ============================
                # 5) VALIDATION INFERENCE
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
                # 6) TEST INFERENCE (seen tasks)
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
                # 7) TEST INFERENCE (unseen tasks)
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

                    cv_performances = pd.concat(
                        [cv_performances, unseen_cv], ignore_index=True
                    )
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
                # Save only the default (plastic) adapter — the one we trained
                model.set_adapter("default")
                model.save_pretrained(full_save_path)
                tokenizer.save_pretrained(full_save_path)

            print("Model saved, cleaning up")
            del model
            gc.collect()
            torch.cuda.empty_cache()
            fold_counter += 1

    return cv_performances
