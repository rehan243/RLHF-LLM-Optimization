# Engineering Log

Running notes on design decisions and lessons learned.


### 2026-07-08

During training with PPO for preference optimization, I noticed that overly aggressive clipping (high epsilon) tends to destabilize policy updates, leading to oscillations in reward signals; reducing epsilon slightly improved stability without sacrificing convergence speed. When incorporating reward modeling, it's crucial to balance the penalty for divergence in the learned reward function, over-penalizing can cause the policy to ignore nuanced preferences.
