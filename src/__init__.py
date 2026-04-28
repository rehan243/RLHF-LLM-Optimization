"""RLHF-ish training utilities — PPO, DPO, and a reward model that judges your life choices."""

from src.reward_model import RewardModel, PreferenceBatch
from src.ppo_trainer import PPOTrainerWrapper, PPOConfig
from src.dpo_trainer import DPOPipeline, DPOConfig

__all__ = [
    "RewardModel",
    "PreferenceBatch",
    "PPOTrainerWrapper",
    "PPOConfig",
    "DPOPipeline",
    "DPOConfig",
]
