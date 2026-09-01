# Engineering Log

Running notes on design decisions and lessons learned.


### 2026-07-08

During training with PPO for preference optimization, I noticed that overly aggressive clipping (high epsilon) tends to destabilize policy updates, leading to oscillations in reward signals; reducing epsilon slightly improved stability without sacrificing convergence speed. When incorporating reward modeling, it's crucial to balance the penalty for divergence in the learned reward function, over-penalizing can cause the policy to ignore nuanced preferences.

### 2026-07-10

- **DPO/PPO**: I've been experimenting with DPO and PPO for preference optimization. DPO was surprisingly effective in reducing hallucinations, but it took a lot of tuning to get the right balance between diversity and coherence. PPO, on the other hand, was faster to converge but required careful handling of the reward signal to avoid overfitting.

- **Reward Modeling**: I've been using a simple reward model based on task completion and accuracy. However, I noticed that the model tends to prioritize task completion over accuracy, which can lead to suboptimal responses. I've been experimenting with adding a penalty for incorrect responses to better balance the two.

- **Tradeoff**: The main tradeoff I've noticed is between diversity and coherence. DPO helps maintain diversity but can sometimes lead to less coherent responses, while PPO tends to produce more coherent responses but can be less

### 2026-07-12

**Observation:** I found that using a larger reward scale (up to 100) for DPO/PPO resulted in better convergence and more stable training. However, it also increased the risk of overfitting, so I had to carefully tune the learning rate and batch size to maintain generalization.

### 2026-07-16

Tried training a reward model with preference data using DPO and PPO; found PPO's sampling variance makes reward signals noisier, slowing convergence compared to DPO's direct policy updates. Also, tuning the KL penalty coefficient is critical, too low leads to policy collapse, too high stalls learning. For reward modeling, balancing dataset size with annotation quality remains a key bottleneck.

### 2026-07-18

Tested DPO on a small reward model and noticed that tuning the KL coefficient too low causes rapid policy collapse, while too high slows learning drastically. PPO remains more stable under noisy reward signals but requires careful clipping parameter adjustments to avoid overfitting to preference data. Reward modeling quality heavily impacts convergence, so investing in accurate preference annotation upfront pays off in training stability.

### 2026-07-26

Discovered that tuning the clipping parameter in PPO significantly impacts training stability; too high causes policy divergence, while too low limits exploration. For preference optimization, balancing reward signal sparsity with entropy regularization proved critical, over-regularizing dampens policy updates, but under-regularizing leads to noisy gradients.

### 2026-07-27

Tried both DPO and PPO for preference optimization on the same reward model; DPO converges faster with less tuning but struggles with longer horizon tasks where PPO's iterative policy updates handle delayed credit assignment better. Noticed reward modeling quality significantly impacts stability, noisy preference labels lead to unstable PPO gradients, while DPO's direct optimization is somewhat more robust but less flexible for complex policies.

### 2026-07-30

Noticed that tuning the clipping parameter in PPO significantly affects stability, setting it too high causes policy updates to become unstable, while too low slows down learning; a moderate value around 0.2 seems to balance learning speed and stability. Also, during reward modeling, I observed that overfitting to the reward signal can lead to degenerate policies, so incorporating a small amount of regularization or early stopping based on validation performance helps.

### 2026-08-01

Reviewed preference optimization (DPO/PPO) and reward modeling today. Reinforced that measuring the change end-to-end beats reasoning about it in isolation — the numbers rarely match the intuition.

### 2026-08-04

Reviewed preference optimization (DPO/PPO) and reward modeling today. Reinforced that measuring the change end-to-end beats reasoning about it in isolation — the numbers rarely match the intuition.

### 2026-08-05

Reviewed preference optimization (DPO/PPO) and reward modeling today. Reinforced that measuring the change end-to-end beats reasoning about it in isolation — the numbers rarely match the intuition.

### 2026-08-08

Reviewed preference optimization (DPO/PPO) and reward modeling today. Reinforced that measuring the change end-to-end beats reasoning about it in isolation — the numbers rarely match the intuition.

### 2026-08-11

Reviewed preference optimization (DPO/PPO) and reward modeling today. Reinforced that measuring the change end-to-end beats reasoning about it in isolation — the numbers rarely match the intuition.

### 2026-08-14

Reviewed preference optimization (DPO/PPO) and reward modeling today. Reinforced that measuring the change end-to-end beats reasoning about it in isolation — the numbers rarely match the intuition.

### 2026-08-19

Reviewed preference optimization (DPO/PPO) and reward modeling today. Reinforced that measuring the change end-to-end beats reasoning about it in isolation — the numbers rarely match the intuition.

### 2026-08-22

Reviewed preference optimization (DPO/PPO) and reward modeling today. Reinforced that measuring the change end-to-end beats reasoning about it in isolation — the numbers rarely match the intuition.

### 2026-08-24

Reviewed preference optimization (DPO/PPO) and reward modeling today. Reinforced that measuring the change end-to-end beats reasoning about it in isolation — the numbers rarely match the intuition.

### 2026-08-28

Reviewed preference optimization (DPO/PPO) and reward modeling today. Reinforced that measuring the change end-to-end beats reasoning about it in isolation — the numbers rarely match the intuition.

### 2026-08-31

Reviewed preference optimization (DPO/PPO) and reward modeling today. Reinforced that measuring the change end-to-end beats reasoning about it in isolation — the numbers rarely match the intuition.

### 2026-09-01

Reviewed preference optimization (DPO/PPO) and reward modeling today. Reinforced that measuring the change end-to-end beats reasoning about it in isolation — the numbers rarely match the intuition.
