"""scalar head on a frozen-ish lm — learns what 'better' means from pairwise labels."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer, PreTrainedModel

logger = logging.getLogger(__name__)


@dataclass
class PreferenceBatch:
    chosen_input_ids: torch.Tensor
    chosen_attention_mask: torch.Tensor
    rejected_input_ids: torch.Tensor
    rejected_attention_mask: torch.Tensor


def _pool_last_hidden(
    hidden: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """pool last hidden states based on attention mask"""
    mask = attention_mask.unsqueeze(-1)
    summed = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)
    return summed


class RewardModel(nn.Module):
    """bradley-terry on pooled encoder states, checkpointing keeps 7B-ish models from oom-ing"""

    def __init__(
        self,
        backbone_name: str,
        hidden_dropout: float = 0.1,
        freeze_backbone_layers: int = 0,
        use_gradient_checkpointing: bool = True,
        torch_dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(backbone_name, use_fast=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.backbone: PreTrainedModel = AutoModel.from_pretrained(
            backbone_name,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
        )
        hidden = int(getattr(self.backbone.config, "hidden_size", 0) or getattr(self.backbone.config, "d_model", 0))
        if hidden <= 0:
            raise ValueError(f"cannot infer hidden size from {type(self.backbone.config)}")

        self.score_head = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Dropout(hidden_dropout),
            nn.Linear(hidden, 1),
        )
        self.score_head = self.score_head.to(dtype=torch.float32)

        if freeze_backbone_layers > 0:
            layers = getattr(self.backbone, "encoder", None)
            layer_list = getattr(layers, "layer", None) if layers is not None else None
            if layer_list is not None:
                for layer in layer_list[:freeze_backbone_layers]:
                    for p in layer.parameters():
                        p.requires_grad = False
                logger.info("froze first %s transformer blocks", freeze_backbone_layers)

        if use_gradient_checkpointing:
            self.backbone.gradient_checkpointing_enable()

    def encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """encode inputs to get rewards"""
        out = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        h = out.last_hidden_state
        pooled = _pool_last_hidden(h, attention_mask)
        return self.score_head(pooled.float()).squeeze(-1)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """forward pass through the model"""
        return self.encode(input_ids, attention_mask)

    def bradley_terry_loss(self, batch: PreferenceBatch) -> torch.Tensor:
        """calculate the bradley-terry loss"""
        r_chosen = self.forward(batch.chosen_input_ids, batch.chosen_attention_mask)
        r_rejected = self.forward(batch.rejected_input_ids, batch.rejected_attention_mask)
        logits = r_chosen - r_rejected
        return F.softplus(-logits).mean()

    @torch.inference_mode()
    def predict_reward(
        self,
        texts: List[str],
        max_length: int = 512,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """predict rewards for a list of texts"""
        device = device or next(self.backbone.parameters()).device
        enc = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        return self.forward(enc["input_ids"], enc["attention_mask"])

    def training_step(self, batch: PreferenceBatch) -> dict[str, float]:
        """perform a training step and return metrics"""
        self.train()
        r_chosen = self.forward(batch.chosen_input_ids, batch.chosen_attention_mask)
        r_rejected = self.forward(batch.rejected_input_ids, batch.rejected_attention_mask)
        logits = r_chosen - r_rejected
        loss = F.softplus(-logits).mean()
        if not math.isfinite(loss.detach().item()):
            logger.error("NaN/Inf in BT loss — check data or LR")
            raise FloatingPointError("non-finite reward loss")
        return {
            "loss": float(loss.detach().item()),
            "reward_margin_mean": float(logits.mean().detach().item()),
        }