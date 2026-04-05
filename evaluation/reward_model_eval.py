"""Reward Model Evaluation Framework - Rehan Malik"""

import numpy as np
from dataclasses import dataclass


@dataclass
class RewardEvalResult:
    accuracy: float
    kendall_tau: float
    agreement_rate: float
    calibration_error: float


def pairwise_accuracy(reward_model_scores: list[tuple], human_preferences: list[int]) -> float:
    """Measure how often reward model agrees with human preferences."""
    correct = 0
    for (score_a, score_b), pref in zip(reward_model_scores, human_preferences):
        model_pref = 0 if score_a > score_b else 1
        if model_pref == pref:
            correct += 1
    return correct / len(human_preferences) if human_preferences else 0.0


def calibration_error(predicted_probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    """Expected calibration error for reward model confidence."""
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (predicted_probs >= lo) & (predicted_probs < hi)
        if mask.sum() > 0:
            avg_conf = predicted_probs[mask].mean()
            avg_acc = labels[mask].mean()
            ece += mask.sum() * abs(avg_conf - avg_acc)
    return float(ece / len(labels))


if __name__ == "__main__":
    np.random.seed(42)
    scores = [(0.8, 0.3), (0.6, 0.7), (0.9, 0.1), (0.5, 0.5)]
    prefs = [0, 1, 0, 0]
    print(f"Pairwise accuracy: {pairwise_accuracy(scores, prefs):.2%}")
    probs = np.random.uniform(0, 1, 100)
    labels = (probs + np.random.normal(0, 0.2, 100) > 0.5).astype(float)
    print(f"Calibration error: {calibration_error(probs, labels):.4f}")
