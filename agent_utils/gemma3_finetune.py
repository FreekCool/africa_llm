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
from datasets import Dataset
from collections import defaultdict
from trl import SFTTrainer, SFTConfig
from sklearn.model_selection import train_test_split

from transformers import AutoProcessor, Gemma3ForConditionalGeneration
from peft import PeftModel, prepare_model_for_kbit_training, get_peft_model

from collections import defaultdict, Counter
from PIL import Image
from sklearn.metrics import precision_recall_fscore_support

from .slot_trainer import MultiTargetSlotSFTTrainer
from .utils import (
    DEBUG, DEBUG_EXAMPLE_IDX, DEBUG_MAX_CHARS,   # ✅ add these
    setup_seed,
    set_quant_and_peft_config_gemma,
    set_training_params_gemma,
    create_result_filename,
    create_model_dirname,
    handle_follow_up_prompt_llama_standard,
    print_gpu_memory,
    preprocess_function,
    build_sft_dataset,
    build_split_data,
    build_split_data_json,
    extract_number,
    MyDataCollator,
    debug_print_multitask_sft_bundle,
    build_slot_token_map,
    all_slot_tokens,
    SlotLossCollator,
    debug_print_slot_token_setup,                # ✅ you call this, so import it too
    debug_print_one_sft_string_example,          # ✅ you call this, so import it too
)

from .eval_utils import (
    run_taskwise_inference,
    evaluate_predictions_binary,
    evaluate_predictions_multiclass,
)


def run_fine_tuned_gemma3(
    train_df,
    test_df,
    test_unseen_df=None,
    text_col=None,
    target_col=None,
    prompt=None,              # NEW: prompt template string with "{}"
    answer_col=None,          # NEW: column with supervised answer text (e.g. "targets_json")
    train_val_seeds=None,
    val_size=0.2,
    results_folder=None,
    model_dir=None,
    max_tokens=None,
    batch_size=None,
    id_col=None,
    class_labels=None,
    max_new_tokens=3,
    cache_dir=None,
    local_model=None,
    text_only_res=None,
    image_folder=None,
    early_stopping_patience=3,
    vlm_images_to_include=10,
    epochs=10,
    learning_rates=(0.0002,),
    targets_spec=None,
):
    mtype = "fine_tuned_gemma3"

    # --- basic validation (fail fast) ---
    if prompt is None or not isinstance(prompt, str) or "{}" not in prompt:
        raise ValueError("run_fine_tuned_gemma3: `prompt` must be a string template containing '{}'.")

    if answer_col is None:
        raise ValueError("run_fine_tuned_gemma3: `answer_col` is required (e.g. 'targets_json').")

    if text_col is None:
        raise ValueError("run_fine_tuned_gemma3: `text_col` is required (e.g. 'transcript').")

    # optional: warn if answer_col missing
    if answer_col not in train_df.columns:
        raise ValueError(
            f"run_fine_tuned_gemma3: answer_col='{answer_col}' not in train_df columns. "
            f"Available columns: {list(train_df.columns)[:20]}..."
        )

    print("Finetuning Gemma3 (new API):")
    print(" - text_col:", text_col)
    print(" - answer_col:", answer_col)
    print(" - multimodal:", bool(image_folder))

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

    if image_folder is not None:
        print(f'Image folder received!!')

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

    if targets_spec:
        slot_tokens = all_slot_tokens(targets_spec)
        if slot_tokens:
            tokenizer.add_special_tokens({"additional_special_tokens": slot_tokens})

            if DEBUG and targets_spec:
                # sanity: slot tokens must map to unique non-unk ids
                unk = tokenizer.unk_token_id
                bad = []
                for tok in all_slot_tokens(targets_spec):
                    tid = tokenizer.convert_tokens_to_ids(tok)
                    if tid == unk:
                        bad.append(tok)
                print("[DEBUG] slot_tokens:", len(all_slot_tokens(targets_spec)))
                print("[DEBUG] bad slot tokens mapped to UNK:", len(bad))
                if bad:
                    print("[DEBUG] example bad:", bad[:20])

    # Ensure padding tokens are set (Gemma sometimes needs this)
    if getattr(tokenizer, "pad_token", None) is None and getattr(tokenizer, "eos_token", None) is not None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"

    if targets_spec is not None and DEBUG:
        debug_print_slot_token_setup(tokenizer, targets_spec)

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

            if DEBUG:
                debug_print_one_sft_string_example(
                    df=train_rows.reset_index(drop=True),
                    idx=min(DEBUG_EXAMPLE_IDX, len(train_rows)-1),
                    tokenizer=tokenizer,
                    max_tokens=max_tokens,
                    text_col=text_col,
                    prompt=prompt,
                    answer_col=answer_col,
                    targets_spec=targets_spec,
                )

            # Optional sanity checks: label distribution
            print("Train label counts:")
            print(train_rows[answer_col].value_counts())

            print("\nVal label counts:")
            print(val_rows[answer_col].value_counts())

            print("\nTest label counts:")
            print(test_df[answer_col].value_counts())

            # ======================
            # 1) BUILD TRAIN DATASET
            # ======================
            is_mm = image_folder is not None

            dataset = build_sft_dataset(
                df=train_rows,
                tokenizer=tokenizer,
                max_tokens=max_tokens,
                text_col=text_col,
                prompt=prompt,
                answer_col=answer_col,
                id_col=id_col,
                image_folder=image_folder if is_mm else None,
                vlm_images_to_include=vlm_images_to_include,
                targets_spec=targets_spec,          # ✅ ADD THIS
            )
            print(f"Dataset loaded for fold {fold_counter}")
            print("Train dataset columns:", dataset.column_names)

            if DEBUG:
                ex_idx = min(DEBUG_EXAMPLE_IDX, len(dataset) - 1)
                print("\n" + "=" * 100)
                print(f"DATASET TEXT SAMPLE | idx={ex_idx} | chars={len(dataset[ex_idx]['text'])}")
                print("=" * 100)
                print(dataset[ex_idx]["text"][:8000])  # full-ish
                print("=" * 100 + "\n")

            # ---- inspect one full raw training example (pre-tokenization) ----
            # ex_idx = 0  # pick any index
            # raw_text = dataset[ex_idx]["text"]
            
            # print("\n" + "="*100)
            # print(f"RAW SFT EXAMPLE (dataset idx={ex_idx}) | chars={len(raw_text)}")
            # print("="*100)
            # print(raw_text)          # full prompt + assistant answer
            # print("="*100 + "\n")
            
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
                val_video_ids,
                val_instructions_plain,
            ) = build_split_data_json(
                df=val_rows,
                tokenizer=tokenizer,
                max_tokens=max_tokens,
                text_col=text_col,
                prompt=prompt,
                answer_col=answer_col,
                id_col=id_col,
            )
            
            # IMPORTANT: val_dataset must match train mode (mm vs text)
            val_dataset = build_sft_dataset(
                df=val_rows,
                tokenizer=tokenizer,
                max_tokens=max_tokens,
                text_col=text_col,
                prompt=prompt,
                answer_col=answer_col,
                id_col=id_col,
                image_folder=image_folder if is_mm else None,
                vlm_images_to_include=vlm_images_to_include,
                targets_spec=targets_spec,          # ✅ ADD THIS
            )
            print("Val dataset columns:", val_dataset.column_names)

            # debug_print_multitask_sft_bundle(
            #     is_mm=is_mm,
            #     train_rows=train_rows,
            #     val_rows=val_rows,
            #     dataset=dataset,
            #     val_dataset=val_dataset,
            #     text_col=text_col,
            #     answer_col=answer_col,
            #     id_col=id_col,
            #     val_prompts=val_prompts,
            #     val_labels=val_labels,
            #     val_video_ids=val_video_ids,
            #     val_instructions_plain=val_instructions_plain,
            #     n_df_rows=2,
            #     n_ds_rows=2,
            # )
            
            # ====================
            # 3) MODEL + TRAINER
            # ====================
            if loaded_locally:
                cv_performances = pd.read_csv(text_only_res)
                print("Reloaded previous results")
            else:
                cv_performances = pd.DataFrame()
            
            filename = create_result_filename(target_col, mtype, learning_rate, train_val_seed)
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
            model.resize_token_embeddings(len(tokenizer))
            
            # prepare for k-bit training
            if gpu_avail and quant_config is not None:
                model = prepare_model_for_kbit_training(model)
            
            if local_model is not None:
                model = PeftModel.from_pretrained(model, local_model, is_trainable=True)
            else:
                model = get_peft_model(model, peft_args)
            
            print(f"Model loaded for fold {fold_counter}")
            
            training_params = set_training_params_gemma(
                    train_val_seed, batch_size, learning_rate,
                    group_by_length=(not is_mm),   # ✅ text-only True, multimodal False
                )


            if is_mm:
                training_params.remove_unused_columns = False

            print("Sample row keys:", dataset[0].keys())
            
            use_slot_loss = targets_spec is not None

            if use_slot_loss:
                # Use a single max_seq_length for this TRL version, which still
                # reads it directly from the constructor kwargs (default 1024 if
                # not provided).
                sft_max_seq = max_tokens if max_tokens is not None else 4096

                print(f"[SLOT-LOSS] Using max_seq_length={sft_max_seq}")

                collator = SlotLossCollator(tokenizer, targets_spec)

                trainer = MultiTargetSlotSFTTrainer(
                    model=model,
                    train_dataset=dataset,
                    eval_dataset=val_dataset,
                    tokenizer=tokenizer,
                    args=training_params,
                    data_collator=collator,
                    targets_spec=targets_spec,
                    max_seq_length=sft_max_seq,
                    dataset_text_field="text",
                    packing=False,
                )
            else:
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

            print("train bs:", trainer.args.per_device_train_batch_size)
            print("eval  bs:", trainer.args.per_device_eval_batch_size)

            print("remove_unused_columns:", trainer.args.remove_unused_columns)
            print("Columns in trainer.train_dataset:", trainer.train_dataset.column_names)
            print("Trainer setup complete")

            if DEBUG and use_slot_loss:
                print("\n" + "=" * 100)
                print("SLOT-LOSS PIPELINE SANITY CHECK (ONE BATCH)")
                print("=" * 100)
            
                dl = trainer.get_train_dataloader()
                batch = next(iter(dl))
            
                print("batch keys:", list(batch.keys()))
                print("input_ids shape:", tuple(batch["input_ids"].shape))
                print("labels shape:", tuple(batch["labels"].shape))
            
                # 1) do we even have labels_by_target?
                lbt = batch.get("labels_by_target", None)
                if lbt is None:
                    print("❌ labels_by_target is missing -> SlotLossCollator is not being used.")
                else:
                    print("✅ labels_by_target present. #targets:", len(lbt))
            
                    # 2) for each target, do we have at least 1 supervised position?
                    nonzero = {}
                    for t, lab in lbt.items():
                        nonzero[t] = int((lab != -100).sum().item())
            
                    # print first ~15 targets counts
                    for t in list(nonzero.keys())[:15]:
                        print(f"  {t}: supervised_positions={nonzero[t]}")
            
                    # 3) if ALL are zero, then slot tokens are not in labels
                    total_supervised = sum(nonzero.values())
                    print("TOTAL supervised positions across all targets:", total_supervised)
            
                    if total_supervised == 0:
                        print("❌ No supervised positions found. This means slot tokens are NOT in labels.")
                        print("   Likely cause: your dataset text does not contain <@target=value> tokens.")
                    else:
                        print("✅ Supervised positions found -> slot tokens are in labels.")
            
                    # 4) Show decoded batch[0] and confirm presence of a slot token string
                    text0 = tokenizer.decode(batch["input_ids"][0], skip_special_tokens=False)
                    sample_tok = all_slot_tokens(targets_spec)[0]
                    print("\nLooking for slot token in decoded batch[0]:", sample_tok)
                    print("FOUND?" , sample_tok in text0)
                    if sample_tok in text0:
                        pos = text0.find(sample_tok)
                        print("Context around first slot token occurrence:")
                        print(text0[max(0, pos-200):pos+200])
            
                print("=" * 100 + "\n")
            
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

    #             if pynvml and handle:
    #                 print("GPU memory after training:")
    #                 print_gpu_memory(handle, pynvml)

    #             print(trainer.state.log_history)
    #             train_loss = trainer.state.log_history[-1]["train_loss"]
    #             eval_loss = trainer.state.log_history[-2]["eval_loss"]

    #             if pynvml and handle:
    #                 print("GPU memory after getting the loss:")
    #                 print_gpu_memory(handle, pynvml)

    #             # Early stopping bookkeeping
    #             if eval_loss < best_eval_loss:
    #                 best_eval_loss = eval_loss
    #                 no_improvement_counter = 0
    #             else:
    #                 no_improvement_counter += 1

    #             if no_improvement_counter >= early_stopping_patience:
    #                 print(
    #                     f"Early stopping triggered after {i+1} epochs with no improvement."
    #                 )
    #                 break

    #             # ============================
    #             # 5) VALIDATION INFERENCE (multi-task aware, order preserved)
    #             # ============================
    #             val_cv = run_taskwise_inference(
    #                 prompts=val_prompts,
    #                 texts=val_texts,
    #                 task_names=val_task_names,
    #                 class_labels_list=val_class_labels_list,
    #                 labels=val_labels,
    #                 followup_prompts=val_followup_prompts,
    #                 trainer=trainer,
    #                 tokenizer=tokenizer,
    #                 device=device,
    #                 max_new_tokens=max_new_tokens,
    #                 mtype=mtype,
    #                 epoch=current_epoch,
    #                 fold=fold_counter,
    #                 fine_tune_type="validation",
    #                 split_name="val",
    #                 train_loss=train_loss,
    #                 val_loss=eval_loss,
    #                 elapsed_time="na",
    #                 seed=train_val_seed,
                
    #                 processor=processor if is_mm else None,
    #                 image_folder=image_folder if is_mm else None,
    #                 video_ids=val_video_ids,
    #                 instructions_plain=val_instructions_plain,
    #                 vlm_images_to_include=vlm_images_to_include,  # you said: always include 10 right away
    #             )

    #             cv_performances = pd.concat(
    #                 [cv_performances, val_cv], ignore_index=True
    #             )
    #             cv_performances.to_csv(full_file_path, index=False)

    #             # =======================
    #             # 6) TEST INFERENCE (multi-task, seen tasks)
    #             # =======================
    #             (
    #                 test_prompts,
    #                 test_texts,
    #                 test_task_names,
    #                 test_class_labels_list,
    #                 test_labels,
    #                 test_followup_prompts,
    #                 test_video_ids,
    #                 test_instructions_plain,
    #             ) = build_split_data(
    #                 df=test_df,
    #                 tokenizer=tokenizer,
    #                 max_tokens=max_tokens,
    #                 text_col=text_col,
    #                 first_prompt_col=first_prompt_col,
    #                 second_prompt_col=second_prompt_col,
    #                 target_col=target_col,
    #                 task_col=task_col,
    #                 id_col=id_col,   
    #             )
    #             test_cv = run_taskwise_inference(
    #                 prompts=test_prompts,
    #                 texts=test_texts,
    #                 task_names=test_task_names,
    #                 class_labels_list=test_class_labels_list,
    #                 labels=test_labels,
    #                 followup_prompts=test_followup_prompts,
    #                 trainer=trainer,
    #                 tokenizer=tokenizer,
    #                 device=device,
    #                 max_new_tokens=max_new_tokens,
    #                 mtype=mtype,
    #                 epoch=current_epoch,
    #                 fold=fold_counter,
    #                 fine_tune_type="test",
    #                 split_name="test",
    #                 train_loss="na",
    #                 val_loss="na",
    #                 elapsed_time="na",
    #                 seed=train_val_seed,
                
    #                 processor=processor if is_mm else None,
    #                 image_folder=image_folder if is_mm else None,
    #                 video_ids=test_video_ids,
    #                 instructions_plain=test_instructions_plain,
    #                 vlm_images_to_include=vlm_images_to_include,
    #             )

    #             cv_performances = pd.concat(
    #                 [cv_performances, test_cv], ignore_index=True
    #             )
    #             cv_performances.to_csv(full_file_path, index=False)

    #             # ==========================
    #             # 7) TEST INFERENCE (UNSEEN TASKS)
    #             # ==========================
    #             if test_unseen_df is not None and not test_unseen_df.empty:
    #                 (
    #                     unseen_prompts,
    #                     unseen_texts,
    #                     unseen_task_names,
    #                     unseen_class_labels_list,
    #                     unseen_labels,
    #                     unseen_followup_prompts,
    #                     unseen_video_ids,
    #                     unseen_instructions_plain,
    #                 ) = build_split_data(
    #                     df=test_unseen_df,
    #                     tokenizer=tokenizer,
    #                     max_tokens=max_tokens,
    #                     text_col=text_col,
    #                     first_prompt_col=first_prompt_col,
    #                     second_prompt_col=second_prompt_col,
    #                     target_col=target_col,
    #                     task_col=task_col,
    #                     id_col=id_col,   # ✅ add this
    #                 )

    #                 end_time = time.time()
    #                 elapsed_time = end_time - start_time

    #                 unseen_cv = run_taskwise_inference(
    #                     prompts=unseen_prompts,
    #                     texts=unseen_texts,
    #                     task_names=unseen_task_names,
    #                     class_labels_list=unseen_class_labels_list,
    #                     labels=unseen_labels,
    #                     followup_prompts=unseen_followup_prompts,
    #                     trainer=trainer,
    #                     tokenizer=tokenizer,
    #                     device=device,
    #                     max_new_tokens=max_new_tokens,
    #                     mtype=mtype,
    #                     epoch=current_epoch,
    #                     fold=fold_counter,
    #                     fine_tune_type="test_unseen",
    #                     split_name="test_unseen",
    #                     train_loss="na",
    #                     val_loss="na",
    #                     elapsed_time=elapsed_time,
    #                     seed=train_val_seed,
                        
    #                     processor=processor if is_mm else None,
    #                     image_folder=image_folder if is_mm else None,
    #                     video_ids=unseen_video_ids,
    #                     instructions_plain=unseen_instructions_plain,
    #                     vlm_images_to_include=vlm_images_to_include,   
    #                 )

    #                 cv_performances = pd.concat([cv_performances, unseen_cv], ignore_index=True)
    #                 cv_performances.to_csv(full_file_path, index=False)
    #             else:
    #                 print("No unseen tasks provided; skipping unseen evaluation.")

    #             # -------- SAVE MODEL --------
    #             timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    #             model_dirname = create_model_dirname(
    #                 target_col, mtype, learning_rate, train_val_seed, timestamp
    #             )
    #             if not os.path.exists(model_dirname):
    #                 os.makedirs(model_dirname)

    #             full_save_path = os.path.join(model_dir, model_dirname)
    #             print(f"Saving model to: {full_save_path}")
    #             model.save_pretrained(full_save_path)
    #             tokenizer.save_pretrained(full_save_path)

    #         print("Model saved, cleaning up")
    #         del model
    #         gc.collect()
    #         if torch.cuda.is_available():
    #             torch.cuda.empty_cache()
    #         fold_counter += 1

    # return cv_performances