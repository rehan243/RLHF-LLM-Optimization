import pytest
from reward_model import RewardModel

# test that the reward model gives expected outputs for known inputs
def test_reward_model_prediction():
    model = RewardModel()

    # let's assume we have some known inputs and expected outputs
    known_inputs = [
        "I really enjoyed this movie",
        "This was the worst experience ever",
        "The plot was okay, could be better"
    ]

    expected_outputs = [
        0.9,  # expected reward for positive sentiment
        0.1,  # expected reward for negative sentiment
        0.5   # expected reward for neutral sentiment
    ]

    for input_text, expected in zip(known_inputs, expected_outputs):
        prediction = model.predict(input_text)
        assert abs(prediction - expected) < 0.05, f"failed for input: {input_text}"

# test the model's behavior with unexpected input
def test_reward_model_unexpected_input():
    model = RewardModel()
    unexpected_input = None

    with pytest.raises(TypeError):
        model.predict(unexpected_input)

# test if the model handles empty string input
def test_reward_model_empty_input():
    model = RewardModel()
    prediction = model.predict("")
    assert prediction == 0.5, "expected output should be 0.5 for empty input"

# TODO: add more tests for edge cases and different input types