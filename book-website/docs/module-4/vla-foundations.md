---
title: "VLA Foundations"
sidebar_position: 2
description: "Theoretical foundations of Vision-Language-Action models: architectures, training, and capabilities."
---

# VLA Foundations

Vision-Language-Action models combine insights from computer vision, natural language processing, and robot learning into unified systems capable of understanding commands and generating physical actions. This chapter provides the theoretical foundation for these powerful architectures.

## Overview

In this section, you will:

- Understand the transformer architecture underlying VLA models
- Learn how multimodal learning combines vision and language
- Explore the grounding problem in embodied AI
- Study landmark VLA systems and their innovations
- Understand training paradigms for VLA models

## Prerequisites

- Understanding of neural networks (layers, activations, backpropagation)
- Basic familiarity with transformers and attention
- Experience with PyTorch or similar framework
- Linear algebra fundamentals

---

## The Transformer Foundation

### Attention Is All You Need

VLA models build on the transformer architecture, which uses self-attention to process sequences:

```
┌─────────────────────────────────────────────────────────────────┐
│                    Transformer Architecture                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   Input: [token_1, token_2, ..., token_n]                       │
│                                                                  │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │                Multi-Head Self-Attention                 │   │
│   │                                                          │   │
│   │   Q = XW_q    K = XW_k    V = XW_v                      │   │
│   │                                                          │   │
│   │   Attention(Q,K,V) = softmax(QK^T / √d_k) V             │   │
│   │                                                          │   │
│   └─────────────────────────────────────────────────────────┘   │
│                           │                                      │
│                     Add & Norm                                   │
│                           │                                      │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │              Feed-Forward Network                        │   │
│   │         FFN(x) = max(0, xW_1 + b_1)W_2 + b_2           │   │
│   └─────────────────────────────────────────────────────────┘   │
│                           │                                      │
│                     Add & Norm                                   │
│                           │                                      │
│   Output: [out_1, out_2, ..., out_n]                            │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Key Properties for Robotics

| Property | Benefit for VLA |
|----------|-----------------|
| **Parallelization** | Process full sequences simultaneously |
| **Long-range dependencies** | Relate distant tokens (e.g., instruction to action) |
| **Flexibility** | Handle variable-length inputs/outputs |
| **Scalability** | Performance improves with scale |

### Attention Mechanism

```python title="attention.py"
"""Self-attention implementation."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SelfAttention(nn.Module):
    """Single-head self-attention."""

    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim

        # Linear projections for Q, K, V
        self.W_q = nn.Linear(embed_dim, embed_dim)
        self.W_k = nn.Linear(embed_dim, embed_dim)
        self.W_v = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor [batch, seq_len, embed_dim]

        Returns:
            Output tensor [batch, seq_len, embed_dim]
        """
        # Project to Q, K, V
        Q = self.W_q(x)  # [batch, seq_len, embed_dim]
        K = self.W_k(x)
        V = self.W_v(x)

        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1))
        scores = scores / math.sqrt(self.embed_dim)

        # Softmax to get attention weights
        attn_weights = F.softmax(scores, dim=-1)

        # Apply attention to values
        output = torch.matmul(attn_weights, V)

        return output, attn_weights


# Example usage
batch_size, seq_len, embed_dim = 4, 10, 256
x = torch.randn(batch_size, seq_len, embed_dim)

attention = SelfAttention(embed_dim)
output, weights = attention(x)

print(f"Input shape: {x.shape}")
print(f"Output shape: {output.shape}")
print(f"Attention weights shape: {weights.shape}")
```

---

## Multimodal Learning

### From Unimodal to Multimodal

VLA models must process multiple modalities:

```
┌─────────────────────────────────────────────────────────────────┐
│                    Multimodal Processing                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   Modality 1: Vision                                            │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │  Image → Patches → Patch Embedding → Vision Tokens      │   │
│   │  [224x224] → [16x16 patches] → [196 tokens x 768]      │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                  │
│   Modality 2: Language                                          │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │  Text → Tokenize → Token Embedding → Language Tokens    │   │
│   │  "Pick up" → [512, 3894, ...] → [n tokens x 768]       │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                  │
│   Modality 3: Robot State                                       │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │  State → Normalize → MLP → State Tokens                │   │
│   │  [q1...q7, gripper] → [state token x 768]              │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                  │
│   Combined Sequence:                                             │
│   [CLS] [vision_1...vision_196] [SEP] [lang_1...lang_n] [state] │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Cross-Modal Attention

```python title="cross_attention.py"
"""Cross-modal attention for VLA."""
import torch
import torch.nn as nn

class CrossModalAttention(nn.Module):
    """Attend from one modality to another."""

    def __init__(self, query_dim: int, key_dim: int, num_heads: int = 8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = query_dim // num_heads

        self.q_proj = nn.Linear(query_dim, query_dim)
        self.k_proj = nn.Linear(key_dim, query_dim)
        self.v_proj = nn.Linear(key_dim, query_dim)
        self.out_proj = nn.Linear(query_dim, query_dim)

    def forward(self, query: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """
        Args:
            query: Query modality [batch, query_len, query_dim]
            context: Context modality [batch, ctx_len, key_dim]

        Returns:
            Attended output [batch, query_len, query_dim]
        """
        batch_size = query.size(0)

        # Project
        Q = self.q_proj(query)
        K = self.k_proj(context)
        V = self.v_proj(context)

        # Reshape for multi-head attention
        Q = Q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # Attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, V)

        # Reshape and project
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.head_dim)
        return self.out_proj(out)


# Example: Language attending to vision
vision_tokens = torch.randn(4, 196, 768)  # 14x14 patches
language_tokens = torch.randn(4, 20, 768)  # 20 text tokens

cross_attn = CrossModalAttention(768, 768)
attended = cross_attn(language_tokens, vision_tokens)
print(f"Output shape: {attended.shape}")  # [4, 20, 768]
```

---

## The Grounding Problem

### Language to Physical World

A fundamental challenge in VLA is **grounding**—connecting abstract language to physical entities and actions:

```
┌─────────────────────────────────────────────────────────────────┐
│                    The Grounding Problem                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   Language: "Pick up the red cup"                               │
│                                                                  │
│   Grounding Requirements:                                        │
│                                                                  │
│   1. Object Grounding                                           │
│      "red cup" → specific object in scene                       │
│      Which pixels? Which 3D location?                           │
│                                                                  │
│   2. Action Grounding                                           │
│      "pick up" → specific motion sequence                       │
│      Approach trajectory? Grasp pose? Lift height?              │
│                                                                  │
│   3. Context Grounding                                          │
│      Implicit constraints from scene                            │
│      Avoid obstacles, respect physics, safety                   │
│                                                                  │
│   4. Success Grounding                                          │
│      What does "done" look like?                                │
│      Cup lifted? At specific height? In hand?                   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Approaches to Grounding

| Approach | Description | Examples |
|----------|-------------|----------|
| **End-to-End** | Learn grounding implicitly | RT-2, OpenVLA |
| **Modular** | Explicit grounding modules | SayCan, Code as Policies |
| **Hybrid** | Foundation model + specialized heads | PaLM-E |

---

## Landmark VLA Systems

### RT-2: Robotics Transformer 2

```
┌─────────────────────────────────────────────────────────────────┐
│                        RT-2 Architecture                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   Base: PaLI-X (Vision-Language Model, 55B parameters)          │
│                                                                  │
│   Input:                                                         │
│   ┌────────────────┐  ┌───────────────────────────────┐         │
│   │    Images      │  │  "What action should the      │         │
│   │    (history)   │  │   robot take to pick up       │         │
│   │                │  │   the can?"                   │         │
│   └────────┬───────┘  └───────────────┬───────────────┘         │
│            │                          │                          │
│            └──────────────┬───────────┘                          │
│                           │                                      │
│                    ┌──────▼──────┐                              │
│                    │   PaLI-X    │                              │
│                    │  Backbone   │                              │
│                    └──────┬──────┘                              │
│                           │                                      │
│   Output:         ┌───────▼───────────────────────────┐         │
│                   │  "1 128 91 241 5 101 127 100"     │         │
│                   │  (tokenized action)               │         │
│                   └───────────────────────────────────┘         │
│                                                                  │
│   Key Innovation: Actions as text tokens in LLM vocabulary      │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### PaLM-E: Embodied Language Model

```python title="palm_e_concept.py"
"""Conceptual PaLM-E architecture."""

class PaLME:
    """
    PaLM-E: An Embodied Multimodal Language Model

    Key ideas:
    1. Inject continuous sensor observations into language model
    2. Use ViT to encode images into "visual tokens"
    3. Interleave visual tokens with text tokens
    4. Output can be text (reasoning) or actions
    """

    def __init__(self):
        self.vision_encoder = ViT()  # Vision Transformer
        self.language_model = PaLM()  # 540B parameter LLM

    def forward(self, images, text):
        # Encode images to visual tokens
        visual_tokens = self.vision_encoder(images)

        # Interleave with text
        # "I see <img> on the table. Pick up the <img>."
        # becomes: [I, see, [vis_tokens], on, the, table, ...]

        combined = self.interleave(text, visual_tokens)

        # Generate output (text or actions)
        output = self.language_model(combined)

        return output
```

### Octo: Open-Source Generalist Policy

```
┌─────────────────────────────────────────────────────────────────┐
│                        Octo Architecture                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   Key Features:                                                  │
│   • Open-source and fine-tunable                                │
│   • Trained on 800K robot trajectories (Open X-Embodiment)      │
│   • Supports multiple robots and tasks                          │
│                                                                  │
│   Architecture:                                                  │
│   ┌─────────────┐  ┌─────────────┐  ┌─────────────┐            │
│   │   Images    │  │  Language   │  │   State     │            │
│   └──────┬──────┘  └──────┬──────┘  └──────┬──────┘            │
│          │                │                │                     │
│   ┌──────▼──────┐  ┌──────▼──────┐  ┌──────▼──────┐            │
│   │    ViT     │  │  T5 Encoder │  │    MLP     │             │
│   └──────┬──────┘  └──────┬──────┘  └──────┬──────┘            │
│          │                │                │                     │
│          └────────────────┼────────────────┘                    │
│                           │                                      │
│                   ┌───────▼───────┐                             │
│                   │  Transformer  │                             │
│                   │   Backbone    │                             │
│                   └───────┬───────┘                             │
│                           │                                      │
│                   ┌───────▼───────┐                             │
│                   │ Action Head   │                             │
│                   │ (Diffusion)   │                             │
│                   └───────────────┘                             │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Training Paradigms

### Behavioral Cloning

Learn directly from demonstrations:

```python title="behavioral_cloning.py"
"""Behavioral cloning training loop."""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

def train_bc(model, dataloader, optimizer, epochs=100):
    """
    Train policy via behavioral cloning.

    Dataset contains (observation, action) pairs from expert demonstrations.
    """
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        total_loss = 0

        for batch in dataloader:
            obs = batch['observation']  # Images + language + state
            expert_action = batch['action']  # Expert action

            optimizer.zero_grad()

            # Predict action
            predicted_action = model(obs)

            # MSE loss against expert
            loss = criterion(predicted_action, expert_action)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch}: Loss = {total_loss / len(dataloader):.4f}")
```

### Co-Training with Internet Data

Leverage internet-scale data alongside robot data:

```
┌─────────────────────────────────────────────────────────────────┐
│                    Co-Training Strategy                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   Internet Data (billions of examples)                          │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │  Image-text pairs, video captions, QA datasets          │   │
│   │  → Learn visual understanding, language, world knowledge│   │
│   └─────────────────────────────────────────────────────────┘   │
│                              +                                   │
│   Robot Data (millions of examples)                             │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │  Robot trajectories with language annotations           │   │
│   │  → Learn action generation, physical grounding          │   │
│   └─────────────────────────────────────────────────────────┘   │
│                              =                                   │
│   VLA Model                                                      │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │  • Generalizes to new objects (from internet data)      │   │
│   │  • Generates valid robot actions (from robot data)      │   │
│   │  • Follows complex instructions (from both)             │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Exercise 1: Implement Self-Attention

:::tip Exercise 1: Build Attention from Scratch
**Objective**: Implement multi-head self-attention.

**Steps**:

1. Implement single-head attention
2. Extend to multi-head attention
3. Add position encodings
4. Test on a sequence modeling task

**Starter Code**:
```python
class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        # TODO: Implement
        pass

    def forward(self, x):
        # TODO: Implement
        pass
```

**Verification**:
- Output shape matches input shape
- Attention weights sum to 1 along key dimension
- Different heads learn different patterns

**Time Estimate**: 45 minutes
:::

---

## Exercise 2: Multimodal Tokenization

:::tip Exercise 2: Tokenize Vision and Language
**Objective**: Create a unified token sequence from images and text.

**Steps**:

1. Patch an image into 16x16 patches
2. Project patches to embedding dimension
3. Tokenize text with a pretrained tokenizer
4. Concatenate into single sequence with special tokens

**Expected Output**:
```
[CLS] [IMG_1] ... [IMG_196] [SEP] [TEXT_1] ... [TEXT_N] [EOS]
```

**Time Estimate**: 30 minutes
:::

---

## Exercise 3: Analyze VLA Attention

:::tip Exercise 3: Visualize Cross-Modal Attention
**Objective**: Understand what VLA models attend to.

**Steps**:

1. Load a pre-trained VLA model (Octo or OpenVLA)
2. Run inference and extract attention weights
3. Visualize which image patches get attention for different instructions
4. Compare attention for different tasks

**Analysis Questions**:
- Does "pick up the red object" attend to red pixels?
- How does attention change over the action sequence?

**Time Estimate**: 60 minutes
:::

---

## Summary

In this chapter, you learned:

- **Transformers**: The architectural foundation of VLA models
- **Multimodal Learning**: Combining vision and language representations
- **Grounding**: The challenge of connecting language to physical action
- **Landmark Systems**: RT-2, PaLM-E, Octo and their innovations
- **Training**: Behavioral cloning and co-training strategies

VLA models represent a paradigm shift from task-specific robots to general-purpose systems that understand language. The transformer architecture enables this by providing a flexible framework for multimodal learning.

Next, explore [Vision Models for Robotics](/docs/module-4/vision-models) to understand how visual perception feeds into VLA systems.

## Further Reading

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Original transformer paper
- [RT-2](https://arxiv.org/abs/2307.15818) - Vision-Language-Action Models
- [PaLM-E](https://arxiv.org/abs/2303.03378) - Embodied Multimodal Language Model
- [Octo](https://octo-models.github.io/) - Open-Source Generalist Policy
- [A Survey on Vision-Language-Action Models](https://arxiv.org/abs/2402.00046)
