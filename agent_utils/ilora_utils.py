# agent_utils/ilora_utils.py
"""
Shared ILoRA components for both LLaMA-3 and Gemma-3 fine-tuning.

This module contains the core ILoRA machinery that is model-agnostic:
  - ReplayBuffer: Experience replay with reservoir sampling
  - EMAUpdateCallback: Updates the EMA adapter after each step
  - ILoRASFTTrainer: SFTTrainer subclass with consistency regularization

These components work with any PEFT model (LoRA adapters) regardless of
the base model architecture.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Optional, Tuple
from trl import SFTTrainer
from transformers import TrainerCallback
from peft import (
    get_peft_model_state_dict,
    set_peft_model_state_dict,
)


# ==============================================================================
# REPLAY BUFFER  (reservoir sampling)
# ==============================================================================
class ReplayBuffer:
    """
    Fixed-size experience replay buffer with reservoir sampling.
    Stores (input_ids, labels) pairs from the training stream so that
    the consistency regularisation always has "old" examples to compare
    the plastic and stable adapters on.
    """

    def __init__(self, buffer_size: int, device: str = "cuda"):
        self.buffer_size = buffer_size
        self.device = device
        self.num_seen = 0
        self.input_ids = [None] * buffer_size
        self.labels = [None] * buffer_size

    # ---- reservoir sampling ------------------------------------------------
    def _reservoir_idx(self) -> int:
        """Return slot index to overwrite, or -1 to skip this example."""
        if self.num_seen < self.buffer_size:
            return self.num_seen
        rand = np.random.randint(0, self.num_seen + 1)
        return rand if rand < self.buffer_size else -1

    def add(self, input_ids: torch.Tensor, labels: torch.Tensor):
        """Add a batch of examples using reservoir sampling."""
        batch_size = input_ids.shape[0]
        for i in range(batch_size):
            idx = self._reservoir_idx()
            self.num_seen += 1
            if idx >= 0:
                self.input_ids[idx] = input_ids[i].detach().to(self.device)
                self.labels[idx] = labels[i].detach().to(self.device)

    # ---- sampling ----------------------------------------------------------
    def sample(self, size: int) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Sample `size` examples (or fewer if the buffer is not full enough).
        Returns left-padded (input_ids, labels) tensors, or (None, None)
        if the buffer is empty.  Labels are padded with -100 (ignore_index).
        """
        filled = min(self.num_seen, self.buffer_size)
        if filled == 0:
            return None, None

        size = min(size, filled)
        choice = np.random.choice(filled, size=size, replace=False)

        max_len = max(self.input_ids[c].shape[0] for c in choice)

        # left-pad input_ids with 0, labels with -100 (CrossEntropy ignore)
        padded_ids = torch.stack([
            torch.cat([
                torch.zeros(max_len - self.input_ids[c].shape[0],
                            dtype=torch.long, device=self.device),
                self.input_ids[c],
            ])
            for c in choice
        ])
        padded_labels = torch.stack([
            torch.cat([
                torch.full((max_len - self.labels[c].shape[0],), -100,
                           dtype=torch.long, device=self.device),
                self.labels[c],
            ])
            for c in choice
        ])
        return padded_ids, padded_labels

    def is_empty(self) -> bool:
        return self.num_seen == 0


# ==============================================================================
# EMA UPDATE CALLBACK
# ==============================================================================
class EMAUpdateCallback(TrainerCallback):
    """
    After each training step, update the 'ema' adapter weights as an
    exponential moving average of the 'default' (plastic) adapter weights.

        ema = alpha * ema + (1 - alpha) * default

    where alpha ramps up from 0 to `ema_alpha` over the first few steps.
    """

    def __init__(self, peft_model, ema_alpha: float = 0.25):
        self.peft_model = peft_model
        self.ema_alpha = ema_alpha
        self._step = 0  # absolute counter (persists across train() calls)

    def on_step_end(self, args, state, control, **kwargs):
        self._step += 1
        alpha = min(1 - 1 / (self._step + 1), self.ema_alpha)

        # Snapshot the current (plastic / default) adapter parameters
        self.peft_model.set_adapter("default")
        default_params = {
            n: p.detach().clone()
            for n, p in self.peft_model.named_parameters()
            if p.requires_grad
        }

        # Update EMA: ema ← alpha·ema + (1-alpha)·default
        self.peft_model.set_adapter("ema")
        for name, param in self.peft_model.named_parameters():
            if name in default_params:
                param.data.mul_(alpha).add_(
                    default_params[name].data, alpha=1 - alpha
                )

        # Switch back to default for the next training step
        self.peft_model.set_adapter("default")


# ==============================================================================
# ILORA-ENHANCED SFT TRAINER
# ==============================================================================
class ILoRASFTTrainer(SFTTrainer):
    """
    SFTTrainer with ILoRA consistency regularisation.

    Overrides ``compute_loss`` to:
      1. Store the current batch in the replay buffer.
      2. Sample past examples from the buffer.
      3. Forward buffer data through both the *plastic* (default) and
         *stable* (EMA) adapters; compute selective consistency loss
         (only penalises hidden-state shrinkage, preserving knowledge).
      4. Combine:
           total = (task_loss + buffer_loss) / 2 + reg_weight × consistency

    During evaluation the override is bypassed → standard cross-entropy.
    """

    def __init__(
        self,
        buffer_size: int = 500,
        ema_alpha: float = 0.25,
        reg_weight: float = 1.0,
        **kwargs,
    ):
        # Store ILoRA hyper-params before the parent __init__ runs
        self._ilora_buffer_size = buffer_size
        self._ilora_ema_alpha = ema_alpha
        self._ilora_reg_weight = reg_weight

        super().__init__(**kwargs)
        self._setup_ilora()

    # ------------------------------------------------------------------ init
    def _setup_ilora(self):
        """Initialise ILoRA components: EMA adapter, replay buffer, callback."""
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # 1. Add EMA adapter (same LoRA config as default)
        peft_cfg = self.model.peft_config["default"]
        self.model.add_adapter("ema", peft_cfg)

        # 2. Copy default adapter weights → ema adapter (identical start)
        default_state = get_peft_model_state_dict(
            self.model, adapter_name="default"
        )
        set_peft_model_state_dict(
            self.model, default_state, adapter_name="ema"
        )

        # 3. Freeze ema parameters (they are updated manually by the callback)
        for name, param in self.model.named_parameters():
            if "ema" in name:
                param.requires_grad = False

        # 4. Ensure default adapter is active
        self.model.set_adapter("default")

        # 5. Replay buffer
        self.replay_buffer = ReplayBuffer(
            buffer_size=self._ilora_buffer_size,
            device=device,
        )

        # 6. Consistency loss (unreduced MSE for selective masking)
        self.consistency_loss_fn = nn.MSELoss(reduction="none")

        # 7. EMA update callback
        self.add_callback(
            EMAUpdateCallback(self.model, ema_alpha=self._ilora_ema_alpha)
        )

        # 8. Tracking metrics (accessible after each step for logging)
        self._ilora_last_metrics = {}

        print(
            f"[ILoRA] Initialised: buffer_size={self._ilora_buffer_size}, "
            f"ema_alpha={self._ilora_ema_alpha}, reg_weight={self._ilora_reg_weight}"
        )

    # ------------------------------------------------------------------ loss
    @staticmethod
    def _unwrap_peft_model(model):
        """Reach through any DDP / accelerate wrapper to the PeftModel."""
        m = model
        while hasattr(m, "module"):
            m = m.module
        return m

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """
        ILoRA-augmented loss computation.

        During evaluation (model.training is False) this falls back to the
        standard forward + cross-entropy so that early-stopping / eval_loss
        remain comparable to the vanilla fine-tuning baseline.
        """
        # ---- Evaluation: standard loss only --------------------------------
        if not model.training:
            outputs = model(**inputs)
            loss = outputs.loss
            return (loss, outputs) if return_outputs else loss

        # ---- Training: ILoRA regularisation --------------------------------
        input_ids = inputs["input_ids"]
        labels = inputs.get("labels", None)

        # 1) Store current batch in the replay buffer
        if labels is not None:
            self.replay_buffer.add(input_ids.detach(), labels.detach())

        # 2) Sample past examples from the buffer
        buf_ids, buf_labels = self.replay_buffer.sample(input_ids.shape[0])

        consistency_loss = torch.tensor(0.0, device=input_ids.device)
        buffer_task_loss = torch.tensor(0.0, device=input_ids.device)
        has_buffer = buf_ids is not None

        peft_model = self._unwrap_peft_model(model)

        # 3) Consistency regularisation on buffer data
        if has_buffer:
            # 3a) Plastic (default adapter) forward on buffer
            peft_model.set_adapter("default")
            plastic_out = model(
                input_ids=buf_ids,
                labels=buf_labels,
                output_hidden_states=True,
                return_dict=True,
            )
            buffer_task_loss = plastic_out.loss

            # 3b) Stable (EMA adapter) forward on buffer — no gradients
            peft_model.set_adapter("ema")
            with torch.no_grad():
                stable_out = model(
                    input_ids=buf_ids,
                    labels=buf_labels,
                    output_hidden_states=True,
                    return_dict=True,
                )

            # 3c) Selective consistency:
            #     reg_hidden = max(plastic, stable) element-wise
            #     loss = MSE(plastic, reg_hidden)
            #
            #     Where plastic >= stable the MSE is zero (no penalty).
            #     Where plastic < stable the model is penalised for
            #     "forgetting" (hidden activations shrinking).
            consistency_loss = torch.mean(
                torch.cat(
                    [
                        self.consistency_loss_fn(
                            p_h,
                            torch.where(p_h > s_h, p_h, s_h),
                        )
                        for p_h, s_h in zip(
                            plastic_out.hidden_states,
                            stable_out.hidden_states,
                        )
                    ],
                    dim=0,
                )
            )

            # Switch back to default for the main forward pass
            peft_model.set_adapter("default")

        # 4) Main forward on the current training batch
        outputs = model(**inputs)
        task_loss = outputs.loss

        # 5) Combine losses  (matches ILORA.py formulation)
        if has_buffer:
            total_loss = (
                (task_loss + buffer_task_loss) / 2.0
                + self._ilora_reg_weight * consistency_loss
            )
        else:
            total_loss = task_loss

        # 6) Track for logging
        self._ilora_last_metrics = {
            "task_loss": task_loss.item(),
            "buffer_loss": buffer_task_loss.item() if has_buffer else 0.0,
            "consistency_loss": consistency_loss.item() if has_buffer else 0.0,
            "total_loss": total_loss.item(),
            "buffer_filled": min(
                self.replay_buffer.num_seen, self.replay_buffer.buffer_size
            ),
        }

        return (total_loss, outputs) if return_outputs else total_loss
