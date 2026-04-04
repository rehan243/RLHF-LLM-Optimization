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

## Training Results

| Stage | Metric | Value |
|---|---|---|
| SFT | Validation Loss | 1.42 |
| Reward Model | Accuracy (preference prediction) | 78% |
| PPO | Win rate vs SFT baseline | 68% |
| PPO | KL divergence (vs reference) | < 0.5 |
| Final | Helpfulness (human eval) | +35% vs base |
| Final | Safety compliance | 96% |

---

> **Source Code**: The production source code for this project is maintained in a private repository due to proprietary and client confidentiality requirements. This repository documents the architecture, design decisions, and technical approach. For code-level discussions or collaboration inquiries, feel free to reach out.


## Author

**Rehan Malik** - CTO @ Reallytics.ai

- [LinkedIn](https://linkedin.com/in/rehan-malik-cto)
- [GitHub](https://github.com/rehan243)
- [Email](mailto:rehanmalil99@gmail.com)

---