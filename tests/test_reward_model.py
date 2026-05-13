import pytest
from reward_model import calculate_reward, normalize_scores  # assuming these exist

def test_calculate_reward():
    # testing with some example inputs
    preferences = [1, 0, 1, 1, 0]
    rewards = calculate_reward(preferences)
    
    # check if the output is as expected
    assert len(rewards) == len(preferences)
    assert all(r in [0, 1] for r in rewards)  # rewards should be binary

def test_normalize_scores():
    # testing normalization of scores
    scores = [1.0, 2.0, 3.0, 4.0]
    normalized = normalize_scores(scores)
    
    # check if scores are between 0 and 1
    assert all(0 <= n <= 1 for n in normalized)
    
    # check if sum of normalized scores is 1
    assert abs(sum(normalized) - 1.0) < 1e-6  # floating point precision

def test_empty_preferences():
    # testing with empty preferences
    preferences = []
    rewards = calculate_reward(preferences)
    
    assert rewards == []  # should return an empty list

# TODO: add more tests for edge cases and larger inputs if needed