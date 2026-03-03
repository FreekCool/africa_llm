# agent_utils/slot_trainer.py

import os
import math
import torch
import torch.nn.functional as F
from trl import SFTTrainer
from transformers import TrainerCallback

_DEBUG = bool(int(os.environ.get("AFRICA_DEBUG", "1")))


# ──────────────────────────────────────────────────────────────────────
# Callback: injects per-target metrics stored by the trainer into the
# standard HF Trainer log dict, so they appear at logging_steps in
# stdout / TensorBoard / WandB without relying on self.log() timing.
# ──────────────────────────────────────────────────────────────────────

class SlotMetricsCallback(TrainerCallback):
    """Merges the trainer's per-step slot metrics into the Trainer log."""

    def on_log(self, args, state, control, logs=None, model=None, **kwargs):
        trainer = kwargs.get("trainer", None)
        if trainer is None or logs is None:
            return
        last = getattr(trainer, "_last_per_target_logs", None)
        if last:
            logs.update(last)


# ──────────────────────────────────────────────────────────────────────
# Trainer
# ──────────────────────────────────────────────────────────────────────

class MultiTargetSlotSFTTrainer(SFTTrainer):
    """
    Multi-target slot-token SFT trainer.

    K policy (consistent everywhere):
        K = len(allowed)  for BOTH binary and multiclass.
        A "binary" target with allowed=[0,1,99] has K=3 because the
        model's CE loss sees 3 possible slot tokens.
        This is set in SlotLossCollator.target_num_classes and read
        back here via _infer_K.

    Expected batch keys (from SlotLossCollator):
      input_ids, attention_mask, labels,
      labels_by_target   dict[str, LongTensor [B,T]]
      target_num_classes  dict[str, int]
    """

    def __init__(
        self,
        *args,
        targets_spec=None,
        normalize_by_logK=True,
        **kwargs,
    ):
        if "args" in kwargs and getattr(kwargs["args"], "remove_unused_columns", None) is True:
            kwargs["args"].remove_unused_columns = False

        super().__init__(*args, **kwargs)

        self.targets_spec = targets_spec or {}
        self.normalize_by_logK = normalize_by_logK

        # Persistent store for the last step's per-target metrics.
        # The SlotMetricsCallback reads this on on_log.
        self._last_per_target_logs: dict = {}
        self._coverage_batches_logged = 0

        self.add_callback(SlotMetricsCallback())

        if _DEBUG:
            self._print_K_table_once()

    # ── K table (printed once at init) ────────────────────────────────

    def _print_K_table_once(self):
        print("\n" + "=" * 80)
        print("SLOT-LOSS K TABLE  (K = len(allowed) for every target)")
        print("=" * 80)
        for t, spec in self.targets_spec.items():
            if spec.get("type") not in ("binary", "multiclass"):
                continue
            allowed = spec.get("allowed", [])
            K = len(allowed)
            logK = math.log(max(K, 2))
            print(f"  {t:40s}  type={spec['type']:12s}  K={K:3d}  log(K)={logK:.4f}")
        print("=" * 80 + "\n")

    # ── K inference ───────────────────────────────────────────────────

    def _infer_K(self, target_name: str, target_num_classes: dict) -> int:
        K = target_num_classes.get(target_name, None)
        if isinstance(K, int) and K >= 2:
            return K
        spec = self.targets_spec.get(target_name, {})
        allowed = spec.get("allowed", None)
        if isinstance(allowed, (list, tuple)) and len(allowed) >= 2:
            return len(allowed)
        return 2

    # ── Loss ──────────────────────────────────────────────────────────

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels_by_target = inputs.pop("labels_by_target", None)
        target_num_classes = inputs.pop("target_num_classes", {}) or {}

        if not labels_by_target or not isinstance(labels_by_target, dict):
            if "labels" not in inputs:
                ids = inputs["input_ids"]
                lab = ids.clone()
                pad = getattr(self.tokenizer, "pad_token_id", None)
                if pad is not None:
                    lab[lab == pad] = -100
                inputs["labels"] = lab
            return super().compute_loss(model, inputs, return_outputs=return_outputs, **kwargs)

        outputs = model(**inputs)
        logits = outputs.logits
        shift_logits = logits[:, :-1, :].contiguous()
        B, T_minus1, V = shift_logits.shape

        total_loss = shift_logits.new_zeros(())
        n_used = 0
        total_ntok = 0

        step_logs: dict = {}

        for t, lab in labels_by_target.items():
            if not torch.is_tensor(lab):
                continue
            shift_labels = lab[:, 1:].contiguous()
            valid = (shift_labels != -100)
            n_tok = int(valid.sum().item())

            step_logs[f"ntok/{t}"] = float(n_tok)
            step_logs[f"coverage_frac/{t}"] = float(n_tok) / float(B * T_minus1) if T_minus1 > 0 else 0.0

            if n_tok == 0:
                continue
            total_ntok += n_tok

            loss_sum = F.cross_entropy(
                shift_logits.view(-1, V),
                shift_labels.view(-1),
                ignore_index=-100,
                reduction="sum",
            )
            loss_avg = loss_sum / n_tok

            if self.normalize_by_logK:
                K = self._infer_K(t, target_num_classes)
                loss_norm = loss_avg / math.log(max(K, 2))
            else:
                loss_norm = loss_avg

            total_loss = total_loss + loss_norm
            n_used += 1

            step_logs[f"loss/{t}"] = float(loss_norm.detach().cpu().item())

        # Aggregated stats
        step_logs["slot/targets_used"] = float(n_used)
        step_logs["slot/ntok_total"] = float(total_ntok)
        if n_used > 0:
            step_logs["slot/ntok_per_target_mean"] = float(total_ntok / n_used)
        if B > 0:
            step_logs["slot/ntok_per_seq_mean"] = float(total_ntok / B)

        # Store for callback
        self._last_per_target_logs = step_logs

        # First-N-batches coverage summary (DEBUG only)
        if _DEBUG and self._coverage_batches_logged < 3:
            self._coverage_batches_logged += 1
            ranked = sorted(
                [(t, step_logs.get(f"ntok/{t}", 0)) for t in labels_by_target],
                key=lambda x: -x[1],
            )
            print(f"\n[COVERAGE batch #{self._coverage_batches_logged}] "
                  f"targets_used={n_used}/{len(labels_by_target)}  ntok_total={total_ntok}")
            top5 = ranked[:5]
            bot5 = ranked[-5:]
            print(f"  top-5 by ntok: {[(t, int(n)) for t, n in top5]}")
            print(f"  bot-5 by ntok: {[(t, int(n)) for t, n in bot5]}")

        # Final loss
        if n_used == 0:
            if "labels" not in inputs:
                ids = inputs["input_ids"]
                lab = ids.clone()
                pad = getattr(self.tokenizer, "pad_token_id", None)
                if pad is not None:
                    lab[lab == pad] = -100
                inputs["labels"] = lab
            fb_out = model(**inputs)
            fb_logits = fb_out.logits[:, :-1, :].contiguous()
            fb_labels = inputs["labels"][:, 1:].contiguous()
            loss = F.cross_entropy(
                fb_logits.view(-1, fb_logits.size(-1)),
                fb_labels.view(-1),
                ignore_index=-100,
                reduction="mean",
            )
            outputs = fb_out
        else:
            loss = total_loss / n_used

        return (loss, outputs) if return_outputs else loss
