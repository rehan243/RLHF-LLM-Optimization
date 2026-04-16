# DPO vs PPO for LLM Alignment

## DPO (Direct Preference Optimization)
- Simpler: No separate reward model needed
- Cheaper: ~30% compute cost of PPO
- Stable: No RL instabilities
- Best for: Most alignment tasks

## PPO (Proximal Policy Optimization)
- More flexible: Supports custom reward functions
- Better control: Fine-grained reward shaping
- Higher quality ceiling: For complex reward signals
- Best for: Tasks needing precise behavioral control

## Our Recommendation
Start with DPO. Only move to PPO if DPO plateau + you have a
well-calibrated reward model + clear reward signal that DPO cannot capture.