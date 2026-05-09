"""thin trl wrapper — kl schedules and reward shaping live here so the notebook stays readable."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, Callable, Optional

import numpy as np
import torch
import wandb

logger = logging.getLogger(__name__)


@dataclass
class PPOConfig:
    kl_coef: float = 0.1
    kl_target: float = 0.01
    kl_horizon: int = 10000
    reward_clip: float = 5.0
    subtract_baseline: bool = True
    gamma: float = 1.0
    rollout_minibatch: int = 8


class PPOTrainerWrapper:
    """we don't reimplement ppo — we babysit trl's trainer and log the drama."""

    def __init__(
        self,
        inner_trainer: Any,
        config: PPOConfig,
        kl_schedule: Optional[Callable[[int], float]] = None,
    ) -> None:
        self.inner = inner_trainer
        self.config = config
        self._step = 0
        self.kl_schedule = kl_schedule or self._default_kl_schedule
        self._reward_hist: list[float] = []
        self._kl_hist: list[float] = []

    def _default_kl_schedule(self, step: int) -> float:
        progress = min(1.0, step / max(1, self.config.kl_horizon))
        return self.config.kl_coef * (0.5 * (1 + math.cos(math.pi * progress))) + self.config.kl_target

    def reset_reward_baseline(self) -> None:
        """call between curriculum stages so old rollouts don't poison the baseline."""
        self._reward_hist.clear()

    def shape_rewards(self, rewards: torch.Tensor | np.ndarray) -> torch.Tensor:
        if isinstance(rewards, np.ndarray):
            r = torch.from_numpy(rewards).float()
        else:
            r = rewards.float()
        r = torch.clamp(r, -self.config.reward_clip, self.config.reward_clip)
        if self.config.subtract_baseline and self._reward_hist:
            base = float(np.mean(self._reward_hist[-256:]))
            r = r - base
        self._reward_hist.extend(r.detach().cpu().tolist())
        return r

    def _apply_kl_coef(self, new_kl: float) -> None:
        if hasattr(self.inner, "kl_ctl") and hasattr(self.inner.kl_ctl, "value"):
            try:
                self.inner.kl_ctl.value = new_kl
                return
            except Exception as e:
                logger.debug(f"could not set kl_ctl.value — {e}")
        if hasattr(self.inner, "config") and hasattr(self.inner.config, "kl_coef"):
            try:
                self.inner.config.kl_coef = new_kl
            except Exception as e:
                logger.debug(f"could not set inner config kl_coef — {e}")

    def step_postprocess(self, stats: dict[str, Any]) -> dict[str, float]:
        self._step += 1
        new_kl = self.kl_schedule(self._step)
        self._apply_kl_coef(new_kl)

        out: dict[str, float] = {}
        for k in ("objective/kl", "kl", "kl_mean"):
            if k in stats:
                kl_v = float(stats[k])
                out["ppo/kl"] = kl_v
                self._kl_hist.append(kl_v)
                break
        if "ppo/mean_reward" in stats:
            out["ppo/reward_mean"] = float(stats["ppo/mean_reward"])
        if "ppo/std_reward" in stats:
            out["ppo/reward_std"] = float(stats["ppo/std_reward"])
        ent = stats.get("objective/entropy") or stats.get("entropy")
        if ent is not None:
            out["ppo/entropy"] = float(ent)
        cf = stats.get("ppo/clip_fraction") or stats.get("clip_fraction")
        if cf is not None:
            out["ppo/clip_fraction"] = float(cf)
        out["ppo/kl_coef_effective"] = float(new_kl)
        if len(self._kl_hist) >= 10:
            out["ppo/kl_rolling_mean"] = float(np.mean(self._kl_hist[-10:]))
        wandb.log(out, step=self._step)
        return out

    def aggregate_rollout_metrics(self, rewards: np.ndarray, advantages: Optional[np.ndarray] = None) -> dict[str, float]:
        """optional hook if you bypass trl stats dict and log raw numpy from env."""
        m: dict[str, float] = {
            "ppo/raw_reward_mean": float(np.mean(rewards)),
            "ppo/raw_reward_std": float(np.std(rewards)),
        }
        if advantages is not None:
            m["ppo/advantage_mean"] = float(np.mean(advantages))
        return m

    def rollout_batching_hint(self, total_prompts: int, world_size: int) -> int:
        base = max(1, total_prompts // world_size)
        aligned = base - (base % self.config.rollout_minibatch) if base >= self.config.rollout_minibatch else base
        return max(1, aligned)

    def attach_trl_callbacks(self) -> None:
        """if your trl version supports callbacks, register them here — placeholder for glue code."""
        cb = getattr(self.inner, "callbacks", None)
        if cb is None:
            logger.info("inner trainer has no callbacks list — skipping attach_trl_callbacks")
            return
        logger.debug("callbacks container present (%s entries)", len(cb) if hasattr(cb, "__len__") else "?")