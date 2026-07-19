"""rlhf-ish training utilities — ppo, dpo, and a reward model that judges your life choices."""

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

def check_imports() -> None:
    """check if required modules are imported successfully"""
    try:
        assert RewardModel is not None
        assert PreferenceBatch is not None
        assert PPOTrainerWrapper is not None
        assert PPOConfig is not None
        assert DPOPipeline is not None
        assert DPOConfig is not None
    except AssertionError as e:
        print(f"import error: {e}")
        # TODO: handle logging or raise a more descriptive error

check_imports()