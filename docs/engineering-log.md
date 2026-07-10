# Engineering Log

Running notes on design decisions and lessons learned.


### 2026-07-08

During training with PPO for preference optimization, I noticed that overly aggressive clipping (high epsilon) tends to destabilize policy updates, leading to oscillations in reward signals; reducing epsilon slightly improved stability without sacrificing convergence speed. When incorporating reward modeling, it's crucial to balance the penalty for divergence in the learned reward function, over-penalizing can cause the policy to ignore nuanced preferences.

### 2026-07-10

- **DPO/PPO**: I've been experimenting with DPO and PPO for preference optimization. DPO was surprisingly effective in reducing hallucinations, but it took a lot of tuning to get the right balance between diversity and coherence. PPO, on the other hand, was faster to converge but required careful handling of the reward signal to avoid overfitting.

- **Reward Modeling**: I've been using a simple reward model based on task completion and accuracy. However, I noticed that the model tends to prioritize task completion over accuracy, which can lead to suboptimal responses. I've been experimenting with adding a penalty for incorrect responses to better balance the two.

- **Tradeoff**: The main tradeoff I've noticed is between diversity and coherence. DPO helps maintain diversity but can sometimes lead to less coherent responses, while PPO tends to produce more coherent responses but can be less
