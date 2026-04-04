# RLHF-LLM-Optimization

Reinforcement Learning from Human Feedback (RLHF) framework using Proximal Policy Optimization (PPO) for enhancing LLM performance and reliability in production.

[![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-FFD21E?style=flat-square&logo=huggingface&logoColor=black)](https://huggingface.co)

---

## Overview

End-to-end RLHF pipeline for aligning Large Language Models with human preferences. This framework trains reward models from human feedback data, then uses PPO to optimize LLM outputs for quality, safety, and domain-specific requirements.

Developed at **Reallytics.ai** for production GenAI products requiring reliable, aligned LLM outputs.

## RLHF Pipeline

```
┌─────────────────────────────────────────────────────┐
│                Phase 1: Supervised Fine-Tuning        │
│                                                       │
│  Base Model (LLaMA/Mistral)                          │
│       │                                               │
│       ▼                                               │
│  SFT on curated instruction dataset                  │
│       │                                               │
│       ▼                                               │
│  SFT Model (policy_init)                             │
└───────────────────────┬─────────────────────────────┘
                        │
┌───────────────────────▼─────────────────────────────┐
│              Phase 2: Reward Model Training           │
│                                                       │
│  Human Preference Data                               │
│  (chosen vs rejected response pairs)                 │
│       │                                               │
│       ▼                                               │
│  Reward Model Training                               │
│  - Bradley-Terry preference model                    │
│  - Pairwise ranking loss                             │
│       │                                               │
│       ▼                                               │
│  Reward Model (scores any response)                  │
└───────────────────────┬─────────────────────────────┘
                        │
┌───────────────────────▼─────────────────────────────┐
│           Phase 3: PPO Optimization                   │
│                                                       │
│  ┌─────────┐  ┌──────────┐  ┌───────────────────┐   │
│  │ Policy  │─▶│ Generate │─▶│  Reward Model     │   │
│  │ (LLM)   │  │ Response │  │  Score Response   │   │
│  └────▲────┘  └──────────┘  └────────┬──────────┘   │
│       │                              │               │
│       │     ┌────────────────────────▼──────────┐   │
│       │     │       PPO Update                   │   │
│       │     │  - Advantage estimation            │   │
│       │     │  - Clipped policy gradient          │   │
│       │     │  - KL divergence penalty            │   │
│       └─────│  - Value function update            │   │
│             └────────────────────────────────────┘   │
└─────────────────────────────────────────────────────┘
```

## Key Features

- **Full RLHF Pipeline**: SFT, Reward Modeling, and PPO in a unified framework
- **Reward Model Training**: Bradley-Terry preference modeling from human comparison data
- **PPO Optimization**: Proximal Policy Optimization with KL divergence constraints
- **Reference Model**: KL penalty against SFT baseline prevents reward hacking
- **Multi-GPU Training**: Distributed training with DeepSpeed ZeRO Stage 2/3
- **Evaluation Suite**: Automated quality, safety, and helpfulness scoring
- **Production Integration**: Export aligned models for VLLM/TGI serving
- **HuggingFace TRL**: Built on the TRL library for stable training

## Tech Stack

| Category | Technologies |
|---|---|
| **Core** | Python, PyTorch |
| **RLHF** | TRL (PPO Trainer), DeepSpeed |
| **Models** | LLaMA-2, Mistral, GPT-NeoX |
| **Training** | HuggingFace Transformers, Accelerate |
| **Evaluation** | Custom metrics, MT-Bench, AlpacaEval |
| **Infrastructure** | CUDA, Multi-GPU, Docker |
| **Experiment Tracking** | Weights & Biases, TensorBoard |

## Project Structure

```
rlhf-llm-optimization/
├── configs/
│   ├── sft_config.yaml
│   ├── reward_model_config.yaml
│   └── ppo_config.yaml
├── data/
│   ├── prepare_sft_data.py
│   ├── prepare_preference_data.py
│   └── human_feedback_schema.py
├── sft/
│   ├── train_sft.py
│   └── sft_utils.py
├── reward_model/
│   ├── train_reward_model.py
│   ├── reward_model.py
│   └── preference_dataset.py
├── ppo/
│   ├── train_ppo.py
│   ├── ppo_trainer_config.py
│   └── kl_controller.py
├── evaluation/
│   ├── evaluate_alignment.py
│   ├── safety_benchmark.py
│   └── quality_metrics.py
├── serving/
│   ├── export_model.py
│   └── vllm_deploy.py
├── infrastructure/
│   ├── Dockerfile
│   ├── deepspeed_config.json
│   └── docker-compose.yml
├── notebooks/
│   ├── 01_sft_training.ipynb
│   ├── 02_reward_modeling.ipynb
│   └── 03_ppo_training.ipynb
├── tests/
├── requirements.txt
└── README.md
```

## Training Results

| Stage | Metric | Value |
|---|---|---|
| SFT | Validation Loss | 1.42 |
| Reward Model | Accuracy (preference prediction) | 78% |
| PPO | Win rate vs SFT baseline | 68% |
| PPO | KL divergence (vs reference) | < 0.5 |
| Final | Helpfulness (human eval) | +35% vs base |
| Final | Safety compliance | 96% |

## Quick Start

```bash
git clone https://github.com/rehan243/RLHF-LLM-Optimization.git
cd RLHF-LLM-Optimization

pip install -r requirements.txt

# Phase 1: Supervised Fine-Tuning
python sft/train_sft.py --config configs/sft_config.yaml

# Phase 2: Train Reward Model
python reward_model/train_reward_model.py --config configs/reward_model_config.yaml

# Phase 3: PPO Optimization
python ppo/train_ppo.py --config configs/ppo_config.yaml

# Export aligned model
python serving/export_model.py --checkpoint ./output/ppo_final
```

## Author

**Rehan Malik** - CTO @ Reallytics.ai

- [LinkedIn](https://linkedin.com/in/rehan-malik-cto)
- [GitHub](https://github.com/rehan243)
- [Email](mailto:rehanmalil99@gmail.com)

---

