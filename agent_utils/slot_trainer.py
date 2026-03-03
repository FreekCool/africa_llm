# agent_utils/slot_trainer.py
# Improved MultiTargetSlotSFTTrainer:
# - Supports sft_config forwarding (so max_seq_length actually applies)
# - Preserves labels_by_target / target_num_classes by setting remove_unused_columns=False automatically
# - Robust compute_loss with:
#   * per-target CE over slot-token positions only
#   * optional class-normalization by log(K)
#   * safe fallbacks + clear debug logging
# - Avoids breaking TRL/Trainer signatures (accepts *args/**kwargs)

import math
import torch
import torch.nn.functional as F
from trl import SFTTrainer


class MultiTargetSlotSFTTrainer(SFTTrainer):
    """
    Multi-target slot-token SFT trainer.

    Expected batch keys (from your SlotLossCollator):
      - input_ids: LongTensor [B,T]
      - attention_mask: LongTensor [B,T]
      - labels: LongTensor [B,T]              (optional fallback; usually = input_ids with PAD masked)
      - labels_by_target: dict[str, LongTensor [B,T]]  (slot positions kept, others -100)
      - target_num_classes: dict[str, int]             (#classes per target)

    Key features:
      - For each target, computes token-level CE only where that target's slot tokens appear.
      - Normalizes each target loss by log(K) to align binary vs multiclass scales.
      - Final loss = mean over targets that had at least 1 supervised token in the batch.
      - Forwards sft_config to TRL SFTTrainer so max_seq_length is respected.
    """

    def __init__(
        self,
        *args,
        targets_spec=None,
        sft_config=None,               # ✅ IMPORTANT: forward into SFTTrainer
        normalize_by_logK=True,        # optional: normalize each task loss by log(K)
        debug_once=False,              # optional: print one-time debug info about batch coverage
        **kwargs,
    ):
        # Ensure labels_by_target isn't dropped by HF Trainer column pruning
        # If caller did not set it, force it (safe for this custom setup).
        if "args" in kwargs and getattr(kwargs["args"], "remove_unused_columns", None) is True:
            kwargs["args"].remove_unused_columns = False

        super().__init__(*args, sft_config=sft_config, **kwargs)

        self.targets_spec = targets_spec or {}
        self.normalize_by_logK = normalize_by_logK
        self._debug_once = bool(debug_once)
        self._did_debug = False

    def _infer_K(self, target_name: str, target_num_classes: dict) -> int:
        # Prefer K from collator, else infer from targets_spec
        K = target_num_classes.get(target_name, None)
        if isinstance(K, int) and K >= 2:
            return K

        spec = self.targets_spec.get(target_name, {})
        ttype = spec.get("type")
        allowed = spec.get("allowed", None)

        if ttype == "binary":
            # if you include 99 as "unclear", CE sees 3 tokens, but "classes" conceptually 2.
            # For normalization, 2 is typically what you want.
            return 2
        if ttype == "multiclass" and isinstance(allowed, (list, tuple)) and len(allowed) >= 2:
            return len(allowed)

        # last resort
        return 2

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # Do NOT mutate caller inputs in a way that breaks upstream
        labels_by_target = inputs.pop("labels_by_target", None)
        target_num_classes = inputs.pop("target_num_classes", {}) or {}

        # If your collator didn't attach labels_by_target, fallback to normal SFT loss
        if labels_by_target is None or not isinstance(labels_by_target, dict) or len(labels_by_target) == 0:
            # ensure labels exist for fallback
            if "labels" not in inputs:
                # safest: make labels from input_ids (mask PAD)
                input_ids = inputs["input_ids"]
                labels = input_ids.clone()
                pad_id = getattr(self.tokenizer, "pad_token_id", None)
                if pad_id is not None:
                    labels[labels == pad_id] = -100
                inputs["labels"] = labels
            return super().compute_loss(model, inputs, return_outputs=return_outputs, **kwargs)

        # Forward pass
        outputs = model(**inputs)
        logits = outputs.logits  # [B, T, V]

        # Shift for causal LM
        shift_logits = logits[:, :-1, :].contiguous()  # [B, T-1, V]

        total = shift_logits.new_zeros(())
        n_used = 0

        # Optional one-time debug: how many slot tokens supervised per target in this batch?
        if self._debug_once and not self._did_debug:
            try:
                counts = {t: int((lab[:, 1:] != -100).sum().item()) for t, lab in labels_by_target.items()}
                self.log({f"debug/supervised_tokens/{t}": float(c) for t, c in list(counts.items())[:50]})
                self._did_debug = True
            except Exception:
                pass

        # Per-target losses (kept for logging)
        per_target_logs = {}

        for t, lab in labels_by_target.items():
            if not torch.is_tensor(lab):
                continue

            # Shift labels
            shift_labels = lab[:, 1:].contiguous()  # [B, T-1]

            # Valid supervised positions
            valid = (shift_labels != -100)
            n_tok = int(valid.sum().item())
            if n_tok == 0:
                continue

            # Compute CE over full vocab, but ignore -100
            loss_sum = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
                reduction="sum",
            )
            loss_avg = loss_sum / max(n_tok, 1)

            if self.normalize_by_logK:
                K = self._infer_K(t, target_num_classes)
                denom = math.log(max(K, 2))
                loss_norm = loss_avg / denom
            else:
                loss_norm = loss_avg

            total = total + loss_norm
            n_used += 1

            # log detached value
            per_target_logs[f"loss/{t}"] = float(loss_norm.detach().cpu().item())
            per_target_logs[f"ntok/{t}"] = float(n_tok)

        # Final loss
        if n_used == 0:
            # Fallback: normal LM loss
            if "labels" not in inputs:
                input_ids = inputs["input_ids"]
                labels = input_ids.clone()
                pad_id = getattr(self.tokenizer, "pad_token_id", None)
                if pad_id is not None:
                    labels[labels == pad_id] = -100
                inputs["labels"] = labels

            # recompute for fallback (simple + robust)
            fb_outputs = model(**inputs)
            fb_logits = fb_outputs.logits[:, :-1, :].contiguous()
            fb_labels = inputs["labels"][:, 1:].contiguous()

            loss = F.cross_entropy(
                fb_logits.view(-1, fb_logits.size(-1)),
                fb_labels.view(-1),
                ignore_index=-100,
                reduction="mean",
            )
            outputs = fb_outputs  # return the fallback outputs if requested
        else:
            loss = total / n_used

        # Emit logs (Trainer will include them at logging_steps)
        if per_target_logs:
            try:
                # include #targets used for signal
                per_target_logs["debug/targets_used"] = float(n_used)
                self.log(per_target_logs)
            except Exception:
                pass

        return (loss, outputs) if return_outputs else loss