"""DPO when you can't afford online rollouts — reference model bookkeeping included."""

from __future__ import annotations

import copy
import logging
import math
from dataclasses import dataclass
from typing import Any, Optional

import torch
import torch.nn.functional as F
import wandb
from transformers import PreTrainedModel

logger = logging.getLogger(__name__)


@dataclass
class DPOConfig:
    beta: float = 0.1
    beta_end: Optional[float] = None
    beta_schedule_steps: int = 2000
    label_smoothing: float = 0.0
    reference_sync_every: int = 0


class DPOPipeline:
    """Reference policy frozen; policy trains with implicit preference signal."""

    def __init__(
        self,
        policy: PreTrainedModel,
        reference: PreTrainedModel,
        config: DPOConfig,
    ) -> None:
        self.policy = policy
        self.reference = reference.eval()
        for p in self.reference.parameters():
            p.requires_grad = False
        self.config = config
        self._step = 0
        self._implicit_rewards: list[float] = []

    def _beta(self) -> float:
        if self.config.beta_end is None:
            return self.config.beta
        t = min(1.0, self._step / max(1, self.config.beta_schedule_steps))
        return self.config.beta + t * (self.config.beta_end - self.config.beta)

    def dpo_loss(
        self,
        policy_chosen_logps: torch.Tensor,
        policy_rejected_logps: torch.Tensor,
        ref_chosen_logps: torch.Tensor,
        ref_rejected_logps: torch.Tensor,
    ) -> torch.Tensor:
        beta = self._beta()
        pi_logratios = policy_chosen_logps - policy_rejected_logps
        ref_logratios = ref_chosen_logps - ref_rejected_logps
        logits = beta * (pi_logratios - ref_logratios)
        if self.config.label_smoothing > 0:
            loss = (
                (1 - self.config.label_smoothing) * F.logsigmoid(logits)
                + self.config.label_smoothing * F.logsigmoid(-logits)
            ).mean()
            return -loss
        return -F.logsigmoid(logits).mean()

    def implicit_reward_stats(self, chosen_logps: torch.Tensor, ref_chosen: torch.Tensor) -> float:
        """Rough DPO implicit reward — mostly for dashboards."""
        return float((chosen_logps - ref_chosen).mean().detach().item())

    def maybe_refresh_reference(self) -> None:
        """If you want periodic ref sync (unusual for vanilla DPO), wire steps here."""
        if self.config.reference_sync_every <= 0:
            return
        if self._step > 0 and self._step % self.config.reference_sync_every == 0:
            logger.warning("reference_sync_every is set — copying policy -> ref (you probably don't want this)")
            self.reference.load_state_dict(self.policy.state_dict())

    def step(
        self,
        batch: dict[str, torch.Tensor],
        forward_policy: Any,
        forward_ref: Any,
    ) -> dict[str, float]:
        self._step += 1
        with torch.no_grad():
            ref_out = forward_ref(batch)
        pol_out = forward_policy(batch)
        loss = self.dpo_loss(
            pol_out["chosen_logps"],
            pol_out["rejected_logps"],
            ref_out["chosen_logps"],
            ref_out["rejected_logps"],
        )
        if not math.isfinite(loss.detach().item()):
            raise FloatingPointError("DPO loss blew up — lower beta or check logprobs")

        ir = self.implicit_reward_stats(pol_out["chosen_logps"], ref_out["chosen_logps"])
        self._implicit_rewards.append(ir)

        metrics = {
            "dpo/loss": float(loss.detach().item()),
            "dpo/beta": float(self._beta()),
            "dpo/chosen_logp_gap": float(
                (pol_out["chosen_logps"] - ref_out["chosen_logps"]).mean().detach().item()
            ),
            "dpo/implicit_reward_mean_recent": float(
                sum(self._implicit_rewards[-128:]) / max(len(self._implicit_rewards[-128:]), 1)
            ),
        }
        self.maybe_refresh_reference()
        wandb.log(metrics, step=self._step)
        return metrics

    @staticmethod
    def clone_reference(policy: PreTrainedModel) -> PreTrainedModel:
        ref = copy.deepcopy(policy)
        ref.eval()
        return ref

    def export_reference_state_dict(self) -> dict[str, torch.Tensor]:
        return {k: v.cpu().clone() for k, v in self.reference.state_dict().items()}

    def compare_to_ppo_baseline(self, dpo_reward_proxy: float, ppo_reward_baseline: float) -> dict[str, float]:
        delta = dpo_reward_proxy - ppo_reward_baseline
        return {
            "align/dpo_minus_ppo_proxy": float(delta),
            "align/rel_improvement": float(delta / (abs(ppo_reward_baseline) + 1e-6)),
        }
