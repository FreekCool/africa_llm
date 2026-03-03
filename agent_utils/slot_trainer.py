# agent_utils/slot_trainer.py
#
# Multi-target slot-token SFT trainer matching professor requirements:
#   • Per-target CE loss (binary vs multiclass aware via restricted logits)
#   • Standardised / comparable losses via log(K) normalisation
#   • One combined scalar loss for backprop (configurable aggregation)
#   • Full-text LM loss regulariser so the model learns to *generate*
#     the slot-token JSON format (not just predict the right class at
#     masked positions)
#   • Per-target metrics logged to TensorBoard via callback
#   • Coverage diagnostics for first N batches

import os
import math
import torch
import torch.nn.functional as F
from trl import SFTTrainer
from transformers import TrainerCallback

_DEBUG = bool(int(os.environ.get("AFRICA_DEBUG", "1")))


# ──────────────────────────────────────────────────────────────────────
# Callback: merges trainer._last_per_target_logs into the HF Trainer
# log dict at every logging_steps so metrics appear in TensorBoard.
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

    Key design choices (matching professor's requirements):
        1. Per-target CE – computed over *only* the K slot-token logits
           for that target (restrict_to_target_token_ids=True) making it
           a proper K-class classification loss.
        2. Normalised by log(K) so binary (K=3 with 99) and multiclass
           (K=23 for topic01) losses are comparable.
        3. Aggregated into one scalar:
             "mean"           – average over used targets (default)
             "token_weighted" – weight each target by its ntok
             "custom"         – supply target_weights dict
        4. Full-text LM loss regulariser (``full_text_loss_weight``):
           The slot-only loss teaches which class is correct at each
           position but does NOT train the model to generate the
           surrounding JSON / slot-token syntax.  Adding a weighted
           standard LM loss on the complete assistant response teaches
           the model the full generation pattern so it actually outputs
           ``<@politics=1>`` at inference time, not plain integers.
        5. All metrics are logged under both short keys (loss/{t}) and
           namespaced keys (train/loss/{t}) for TensorBoard grouping.

    K policy: K = len(allowed) everywhere.
    """

    def __init__(
        self,
        *args,
        targets_spec=None,
        normalize_by_logK=True,
        restrict_to_target_token_ids=True,
        aggregate="mean",
        target_weights=None,
        full_text_loss_weight=0.1,
        **kwargs,
    ):
        if "args" in kwargs and getattr(kwargs["args"], "remove_unused_columns", None) is True:
            kwargs["args"].remove_unused_columns = False

        super().__init__(*args, **kwargs)

        self.targets_spec = targets_spec or {}
        self.normalize_by_logK = normalize_by_logK
        self.restrict_to_target_token_ids = restrict_to_target_token_ids
        self.aggregate = aggregate
        self.target_weights = target_weights or {}
        self.full_text_loss_weight = full_text_loss_weight

        self._last_per_target_logs: dict = {}
        self._coverage_batches_logged = 0

        # Pre-build ordered token-id lists and id→class-index maps per
        # target so compute_loss doesn't rebuild every step.
        self._target_allowed_ids: dict = {}      # t → LongTensor [K]
        self._target_id_to_class: dict = {}      # t → dict {token_id: class_idx}
        self._build_target_id_maps()

        self.add_callback(SlotMetricsCallback())

        if _DEBUG:
            self._print_K_table_once()

    # ── Pre-compute token id maps ─────────────────────────────────────

    def _build_target_id_maps(self):
        from .utils import build_slot_token_map
        token_map = build_slot_token_map(self.targets_spec)
        tokenizer = self.tokenizer

        for t, spec in self.targets_spec.items():
            if spec.get("type") not in ("binary", "multiclass"):
                continue
            allowed = spec.get("allowed", [])
            ordered_ids = []
            id_to_cls = {}
            for idx, v in enumerate(allowed):
                tok_str = token_map[t][str(v)]
                tid = tokenizer.convert_tokens_to_ids(tok_str)
                ordered_ids.append(tid)
                id_to_cls[tid] = idx
            self._target_allowed_ids[t] = torch.tensor(ordered_ids, dtype=torch.long)
            self._target_id_to_class[t] = id_to_cls

    # ── K table (debug, printed once) ─────────────────────────────────

    def _print_K_table_once(self):
        print("\n" + "=" * 80)
        print("SLOT-LOSS K TABLE  (K = len(allowed) for every target)")
        print(f"  restrict_to_target_token_ids = {self.restrict_to_target_token_ids}")
        print(f"  aggregate = {self.aggregate}")
        print(f"  full_text_loss_weight = {self.full_text_loss_weight}")
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

    # ── Loss (per-target, standardised, aggregated) ───────────────────

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels_by_target = inputs.pop("labels_by_target", None)
        target_num_classes = inputs.pop("target_num_classes", {}) or {}

        # Fallback: no slot labels → standard SFT loss
        if not labels_by_target or not isinstance(labels_by_target, dict):
            if "labels" not in inputs:
                ids = inputs["input_ids"]
                lab = ids.clone()
                pad = getattr(self.tokenizer, "pad_token_id", None)
                if pad is not None:
                    lab[lab == pad] = -100
                inputs["labels"] = lab
            return super().compute_loss(model, inputs, return_outputs=return_outputs, **kwargs)

        # Forward pass
        outputs = model(**inputs)
        logits = outputs.logits
        shift_logits = logits[:, :-1, :].contiguous()   # [B, T-1, V]
        B, T1, V = shift_logits.shape

        # Accumulators
        per_target_loss = {}   # t → normalised loss (detached float for logs)
        per_target_loss_t = {} # t → tensor (for backprop)
        per_target_ntok = {}   # t → int
        step_logs: dict = {}

        for t, lab in labels_by_target.items():
            if not torch.is_tensor(lab):
                continue
            shift_labels = lab[:, 1:].contiguous()  # [B, T-1]
            valid = (shift_labels != -100)
            n_tok = int(valid.sum().item())

            step_logs[f"ntok/{t}"] = float(n_tok)
            step_logs[f"coverage_frac/{t}"] = float(n_tok) / float(B * T1) if T1 > 0 else 0.0

            if n_tok == 0:
                continue

            # ── Per-target CE ────────────────────────────────────
            if self.restrict_to_target_token_ids and t in self._target_allowed_ids:
                # Classification-style: slice logits to the K allowed
                # token columns, remap labels to 0..K-1 class indices.
                allowed_ids = self._target_allowed_ids[t].to(shift_logits.device)
                id_to_cls = self._target_id_to_class[t]

                logits_t = shift_logits[..., allowed_ids]  # [B, T-1, K]

                labels_cls = torch.full_like(shift_labels, -100)
                for tid, cidx in id_to_cls.items():
                    labels_cls[shift_labels == tid] = cidx

                loss_sum = F.cross_entropy(
                    logits_t.reshape(-1, logits_t.size(-1)),
                    labels_cls.reshape(-1),
                    ignore_index=-100,
                    reduction="sum",
                )
            else:
                # Full-vocab fallback
                loss_sum = F.cross_entropy(
                    shift_logits.view(-1, V),
                    shift_labels.view(-1),
                    ignore_index=-100,
                    reduction="sum",
                )

            loss_avg = loss_sum / n_tok

            # Standardise by log(K) so losses are comparable
            if self.normalize_by_logK:
                K = self._infer_K(t, target_num_classes)
                loss_norm = loss_avg / math.log(max(K, 2))
            else:
                loss_norm = loss_avg

            per_target_loss_t[t] = loss_norm
            per_target_loss[t] = float(loss_norm.detach().cpu().item())
            per_target_ntok[t] = n_tok

            step_logs[f"loss/{t}"] = per_target_loss[t]

        n_used = len(per_target_loss_t)
        total_ntok = sum(per_target_ntok.values())

        # ── Full-text LM loss (teaches the model to generate the
        #    slot-token JSON format, not just predict at masked positions)
        lm_loss = outputs.loss  # already computed by the model from `labels`
        lm_loss_val = float(lm_loss.detach().item()) if lm_loss is not None else 0.0

        # ── Aggregation into one combined loss ────────────────────
        if n_used > 0:
            if self.aggregate == "token_weighted":
                slot_loss = sum(
                    per_target_loss_t[t] * per_target_ntok[t]
                    for t in per_target_loss_t
                ) / max(total_ntok, 1)
            elif self.aggregate == "custom" and self.target_weights:
                wsum = sum(
                    self.target_weights.get(t, 1.0) * per_target_loss_t[t]
                    for t in per_target_loss_t
                )
                wdenom = sum(
                    self.target_weights.get(t, 1.0)
                    for t in per_target_loss_t
                )
                slot_loss = wsum / max(wdenom, 1e-8)
            else:
                slot_loss = sum(per_target_loss_t.values()) / n_used

            loss = slot_loss
            if self.full_text_loss_weight > 0 and lm_loss is not None:
                loss = slot_loss + self.full_text_loss_weight * lm_loss
        else:
            # Fallback: no supervised targets → pure LM loss
            if lm_loss is not None:
                loss = lm_loss
            else:
                if "labels" not in inputs:
                    ids = inputs["input_ids"]
                    lab = ids.clone()
                    pad = getattr(self.tokenizer, "pad_token_id", None)
                    if pad is not None:
                        lab[lab == pad] = -100
                    inputs["labels"] = lab
                fb_out = model(**inputs)
                loss = fb_out.loss
                outputs = fb_out

        # ── Logging ───────────────────────────────────────────────
        step_logs["slot/targets_used"] = float(n_used)
        step_logs["slot/ntok_total"] = float(total_ntok)
        if n_used > 0:
            step_logs["slot/ntok_per_target_mean"] = float(total_ntok / n_used)
        if B > 0:
            step_logs["slot/ntok_per_seq_mean"] = float(total_ntok / B)
        step_logs["slot/loss_combined"] = float(loss.detach().cpu().item()) if n_used > 0 else 0.0
        step_logs["slot/lm_loss"] = lm_loss_val
        step_logs["slot/full_text_loss_weight"] = self.full_text_loss_weight

        # Add train/-namespaced aliases for TensorBoard grouping
        aliased: dict = {}
        for k, v in step_logs.items():
            aliased[k] = v
            aliased[f"train/{k}"] = v
        self._last_per_target_logs = aliased

        # First-N-batches coverage summary (DEBUG)
        if _DEBUG and self._coverage_batches_logged < 3:
            self._coverage_batches_logged += 1
            ranked = sorted(
                [(t, step_logs.get(f"ntok/{t}", 0)) for t in labels_by_target],
                key=lambda x: -x[1],
            )
            print(f"\n[COVERAGE batch #{self._coverage_batches_logged}] "
                  f"targets_used={n_used}/{len(labels_by_target)}  "
                  f"ntok_total={total_ntok}  loss={loss.item():.4f}")
            top5 = ranked[:5]
            bot5 = ranked[-5:]
            print(f"  top-5 by ntok: {[(t, int(n)) for t, n in top5]}")
            print(f"  bot-5 by ntok: {[(t, int(n)) for t, n in bot5]}")

        return (loss, outputs) if return_outputs else loss
