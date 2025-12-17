---
title: "Action Generation"
sidebar_position: 5
description: "From perception to action: policy architectures, action spaces, and motion generation techniques."
---

# Action Generation

The final step in any VLA pipeline is generating actions that achieve the desired goal. This chapter covers the various approaches to action generation, from discrete skill selection to continuous trajectory prediction. You'll learn about different action representations, policy architectures, and techniques for ensuring smooth, safe robot motion.

## Overview

In this section, you will:

- Understand different action space representations
- Implement transformer-based action prediction
- Build diffusion policies for smooth trajectories
- Design action tokenization schemes
- Create motion generation pipelines
- Handle action safety and constraints

## Prerequisites

- Understanding of neural network architectures
- Familiarity with PyTorch
- Knowledge of robot kinematics basics
- Completed [Vision Models](/docs/module-4/vision-models) and [Language Integration](/docs/module-4/language-integration)

---

## Action Space Design

### Types of Action Representations

```
┌─────────────────────────────────────────────────────────────────┐
│                    Action Space Taxonomy                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   Discrete Actions                                               │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │  • Skill selection: pick, place, push, pull             │   │
│   │  • Direction: up, down, left, right                     │   │
│   │  • Object choice: object_1, object_2, ...               │   │
│   │  + Simple to learn, interpretable                       │   │
│   │  - Limited precision, cannot express continuous motion  │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                  │
│   Continuous Actions                                             │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │  • Joint velocities: [dq1, dq2, ..., dqn]               │   │
│   │  • End-effector delta: [dx, dy, dz, droll, dpitch, dyaw]│   │
│   │  • Joint positions: [q1, q2, ..., qn]                   │   │
│   │  + Precise control, smooth motion                       │   │
│   │  - Higher dimensional, harder to learn                  │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                  │
│   Hybrid Actions                                                 │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │  • Skill + parameters: pick(x, y, z)                    │   │
│   │  • Waypoints: [(x1,y1,z1), (x2,y2,z2), ...]            │   │
│   │  • Tokenized continuous: discretized into bins          │   │
│   │  + Balance of expressiveness and learnability           │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Action Space Comparison

| Representation | Dimensionality | Precision | Learning Difficulty | Use Case |
|----------------|---------------|-----------|---------------------|----------|
| **Discrete skills** | Low (10-50) | Low | Easy | High-level planning |
| **Joint velocities** | Medium (6-12) | High | Medium | Velocity control |
| **EE delta pose** | Fixed (6-7) | High | Medium | Cartesian control |
| **Joint positions** | Medium (6-12) | High | Medium | Position control |
| **Tokenized** | High (100-1000) | Medium | Hard | VLA models |
| **Trajectories** | Very High | Very High | Hard | Smooth motion |

### Implementing Action Spaces

```python title="action_spaces.py"
"""Action space definitions for robot control."""
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional
from enum import Enum

class ActionType(Enum):
    DISCRETE = "discrete"
    CONTINUOUS = "continuous"
    HYBRID = "hybrid"


@dataclass
class ActionSpace:
    """Base action space definition."""
    action_type: ActionType
    dimension: int
    bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None

    def normalize(self, action: np.ndarray) -> np.ndarray:
        """Normalize action to [-1, 1]."""
        if self.bounds is None:
            return action
        low, high = self.bounds
        return 2.0 * (action - low) / (high - low) - 1.0

    def denormalize(self, action: np.ndarray) -> np.ndarray:
        """Denormalize action from [-1, 1]."""
        if self.bounds is None:
            return action
        low, high = self.bounds
        return low + (action + 1.0) * (high - low) / 2.0


class EndEffectorDeltaSpace(ActionSpace):
    """6-DOF end-effector delta action space."""

    def __init__(self, max_delta_pos: float = 0.05, max_delta_rot: float = 0.1):
        bounds = (
            np.array([-max_delta_pos] * 3 + [-max_delta_rot] * 3),
            np.array([max_delta_pos] * 3 + [max_delta_rot] * 3)
        )
        super().__init__(
            action_type=ActionType.CONTINUOUS,
            dimension=6,
            bounds=bounds
        )

    def to_transform(self, action: np.ndarray) -> np.ndarray:
        """Convert action to 4x4 transformation matrix."""
        dx, dy, dz, droll, dpitch, dyaw = action

        # Rotation matrix from Euler angles
        from scipy.spatial.transform import Rotation
        R = Rotation.from_euler('xyz', [droll, dpitch, dyaw]).as_matrix()

        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = [dx, dy, dz]
        return T


class JointPositionSpace(ActionSpace):
    """Joint position action space."""

    def __init__(self, joint_limits: List[Tuple[float, float]]):
        low = np.array([lim[0] for lim in joint_limits])
        high = np.array([lim[1] for lim in joint_limits])
        super().__init__(
            action_type=ActionType.CONTINUOUS,
            dimension=len(joint_limits),
            bounds=(low, high)
        )


class TokenizedActionSpace(ActionSpace):
    """Discretized continuous action space using bins."""

    def __init__(self, continuous_dim: int, num_bins: int = 256):
        self.continuous_dim = continuous_dim
        self.num_bins = num_bins

        super().__init__(
            action_type=ActionType.HYBRID,
            dimension=continuous_dim,
            bounds=(np.zeros(continuous_dim), np.ones(continuous_dim) * (num_bins - 1))
        )

    def tokenize(self, continuous_action: np.ndarray) -> np.ndarray:
        """Convert continuous action to tokens."""
        # Assume input is normalized to [0, 1]
        tokens = np.clip(
            (continuous_action * self.num_bins).astype(int),
            0, self.num_bins - 1
        )
        return tokens

    def detokenize(self, tokens: np.ndarray) -> np.ndarray:
        """Convert tokens back to continuous action."""
        return (tokens + 0.5) / self.num_bins


# Example usage
ee_space = EndEffectorDeltaSpace(max_delta_pos=0.02, max_delta_rot=0.05)
action = np.array([0.01, 0.0, -0.005, 0.0, 0.0, 0.02])
normalized = ee_space.normalize(action)
print(f"Normalized action: {normalized}")

tokenized_space = TokenizedActionSpace(continuous_dim=7, num_bins=256)
tokens = tokenized_space.tokenize(np.array([0.5, 0.3, 0.7, 0.1, 0.9, 0.5, 0.5]))
print(f"Tokens: {tokens}")
```

---

## Transformer Action Heads

### Action Prediction Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                Transformer Action Head                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   Vision Features     Language Features     Proprioception      │
│   [v1, v2, ..., vn]   [l1, l2, ..., lm]    [p1, p2, ..., pk]   │
│         │                    │                    │              │
│         └────────────────────┼────────────────────┘              │
│                              │                                   │
│                    ┌─────────▼─────────┐                        │
│                    │  Cross-Attention  │                        │
│                    │    Transformer    │                        │
│                    └─────────┬─────────┘                        │
│                              │                                   │
│              ┌───────────────┼───────────────┐                  │
│              │               │               │                   │
│       ┌──────▼──────┐ ┌──────▼──────┐ ┌─────▼──────┐           │
│       │  Position   │ │  Rotation   │ │  Gripper   │           │
│       │    Head     │ │    Head     │ │   Head     │           │
│       │  (x,y,z)    │ │ (qx,qy,qz,w)│ │  (open)    │           │
│       └─────────────┘ └─────────────┘ └────────────┘           │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Implementation

```python title="action_transformer.py"
"""Transformer-based action generation."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class ActionTransformer(nn.Module):
    """Transformer for action prediction from multimodal inputs."""

    def __init__(
        self,
        vision_dim: int = 768,
        language_dim: int = 768,
        proprio_dim: int = 12,
        hidden_dim: int = 512,
        num_layers: int = 4,
        num_heads: int = 8,
        action_dim: int = 7,
        action_horizon: int = 1,
        dropout: float = 0.1
    ):
        super().__init__()

        self.action_horizon = action_horizon

        # Input projections
        self.vision_proj = nn.Linear(vision_dim, hidden_dim)
        self.language_proj = nn.Linear(language_dim, hidden_dim)
        self.proprio_proj = nn.Linear(proprio_dim, hidden_dim)

        # Learnable action queries
        self.action_queries = nn.Parameter(
            torch.randn(action_horizon, hidden_dim)
        )

        # Transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers)

        # Action heads
        self.position_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3)
        )

        self.rotation_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 4)  # quaternion
        )

        self.gripper_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(
        self,
        vision_features: torch.Tensor,
        language_features: torch.Tensor,
        proprio: torch.Tensor
    ) -> dict:
        """
        Generate actions from multimodal inputs.

        Args:
            vision_features: [batch, num_patches, vision_dim]
            language_features: [batch, seq_len, language_dim]
            proprio: [batch, proprio_dim]

        Returns:
            dict with position, rotation, gripper actions
        """
        batch_size = vision_features.size(0)

        # Project inputs
        v = self.vision_proj(vision_features)  # [B, P, H]
        l = self.language_proj(language_features)  # [B, S, H]
        p = self.proprio_proj(proprio).unsqueeze(1)  # [B, 1, H]

        # Concatenate context
        context = torch.cat([v, l, p], dim=1)  # [B, P+S+1, H]

        # Expand action queries for batch
        queries = self.action_queries.unsqueeze(0).expand(batch_size, -1, -1)

        # Decode actions
        action_features = self.decoder(queries, context)  # [B, H, hidden]

        # Predict action components
        position = self.position_head(action_features)  # [B, H, 3]
        rotation = self.rotation_head(action_features)  # [B, H, 4]
        rotation = F.normalize(rotation, dim=-1)  # Normalize quaternion
        gripper = self.gripper_head(action_features)  # [B, H, 1]

        return {
            'position': position,
            'rotation': rotation,
            'gripper': gripper.squeeze(-1),
            'features': action_features
        }


class ActionTokenPredictor(nn.Module):
    """Predict discretized action tokens autoregressively."""

    def __init__(
        self,
        vocab_size: int = 256,
        action_dim: int = 7,
        hidden_dim: int = 512,
        num_layers: int = 6,
        num_heads: int = 8
    ):
        super().__init__()

        self.action_dim = action_dim
        self.vocab_size = vocab_size

        # Token embedding
        self.token_embed = nn.Embedding(vocab_size, hidden_dim)
        self.pos_embed = nn.Embedding(action_dim, hidden_dim)

        # Transformer
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers)

        # Output projection
        self.output_proj = nn.Linear(hidden_dim, vocab_size)

    def forward(
        self,
        context: torch.Tensor,
        action_tokens: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Predict action tokens.

        Args:
            context: [batch, context_len, hidden_dim] - encoded observation
            action_tokens: [batch, seq_len] - previous tokens (for training)

        Returns:
            logits: [batch, seq_len, vocab_size]
        """
        batch_size = context.size(0)
        device = context.device

        if action_tokens is None:
            # Generate from scratch
            return self._generate(context)

        # Training mode - teacher forcing
        seq_len = action_tokens.size(1)
        positions = torch.arange(seq_len, device=device)

        # Embed tokens and positions
        token_emb = self.token_embed(action_tokens)
        pos_emb = self.pos_embed(positions)
        query = token_emb + pos_emb

        # Causal mask
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=device),
            diagonal=1
        ).bool()

        # Decode
        hidden = self.decoder(
            query, context,
            tgt_mask=causal_mask
        )

        return self.output_proj(hidden)

    @torch.no_grad()
    def _generate(self, context: torch.Tensor) -> torch.Tensor:
        """Generate action tokens autoregressively."""
        batch_size = context.size(0)
        device = context.device

        tokens = []
        for i in range(self.action_dim):
            if i == 0:
                # First token - no previous context
                pos_emb = self.pos_embed(torch.tensor([0], device=device))
                query = pos_emb.expand(batch_size, 1, -1)
            else:
                # Use previous tokens
                prev_tokens = torch.stack(tokens, dim=1)
                positions = torch.arange(i, device=device)
                token_emb = self.token_embed(prev_tokens)
                pos_emb = self.pos_embed(positions)
                query = token_emb + pos_emb

            hidden = self.decoder(query, context)
            logits = self.output_proj(hidden[:, -1])  # Last position
            next_token = logits.argmax(dim=-1)
            tokens.append(next_token)

        return torch.stack(tokens, dim=1)
```

---

## Diffusion Policies

### Diffusion for Trajectory Generation

Diffusion models generate smooth, multi-modal action distributions:

```
┌─────────────────────────────────────────────────────────────────┐
│                    Diffusion Policy                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   Forward Process (Training):                                   │
│   ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐ │
│   │ Action   │───▶│ + Noise  │───▶│ + Noise  │───▶│  Pure    │ │
│   │ τ₀       │    │ τ₁       │    │ τ₂       │    │  Noise   │ │
│   └──────────┘    └──────────┘    └──────────┘    └──────────┘ │
│                                                                  │
│   Reverse Process (Inference):                                  │
│   ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐ │
│   │  Pure    │───▶│ Denoise  │───▶│ Denoise  │───▶│ Action   │ │
│   │  Noise   │    │ (UNet)   │    │ (UNet)   │    │ τ₀       │ │
│   └──────────┘    └──────────┘    └──────────┘    └──────────┘ │
│                        ▲               ▲               ▲        │
│                        │               │               │        │
│                   Observation     Observation     Observation   │
│                   Conditioning   Conditioning    Conditioning   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Implementation

```python title="diffusion_policy.py"
"""Diffusion policy for action generation."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class SinusoidalPosEmb(nn.Module):
    """Sinusoidal positional embedding for diffusion timestep."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        device = t.device
        half_dim = self.dim // 2
        emb = np.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t[:, None] * emb[None, :]
        return torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)


class ConditionalUNet1D(nn.Module):
    """1D U-Net for denoising action trajectories."""

    def __init__(
        self,
        action_dim: int = 7,
        action_horizon: int = 16,
        obs_dim: int = 512,
        hidden_dim: int = 256,
        num_layers: int = 4
    ):
        super().__init__()

        self.action_horizon = action_horizon
        self.action_dim = action_dim

        # Time embedding
        self.time_embed = nn.Sequential(
            SinusoidalPosEmb(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.Mish(),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )

        # Observation conditioning
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Downsampling path
        self.down_blocks = nn.ModuleList()
        self.down_convs = nn.ModuleList()

        in_channels = action_dim
        channels = [hidden_dim, hidden_dim * 2, hidden_dim * 4]

        for out_channels in channels:
            self.down_blocks.append(
                ResidualBlock1D(in_channels, out_channels, hidden_dim)
            )
            self.down_convs.append(
                nn.Conv1d(out_channels, out_channels, 3, stride=2, padding=1)
            )
            in_channels = out_channels

        # Middle block
        self.mid_block = ResidualBlock1D(channels[-1], channels[-1], hidden_dim)

        # Upsampling path
        self.up_blocks = nn.ModuleList()
        self.up_convs = nn.ModuleList()

        for i, out_channels in enumerate(reversed(channels[:-1])):
            in_ch = channels[-(i+1)]
            skip_ch = channels[-(i+2)]
            self.up_convs.append(
                nn.ConvTranspose1d(in_ch, in_ch, 4, stride=2, padding=1)
            )
            self.up_blocks.append(
                ResidualBlock1D(in_ch + skip_ch, out_channels, hidden_dim)
            )

        # Output projection
        self.out_conv = nn.Conv1d(channels[0], action_dim, 1)

    def forward(
        self,
        noisy_action: torch.Tensor,
        timestep: torch.Tensor,
        obs: torch.Tensor
    ) -> torch.Tensor:
        """
        Predict noise in action trajectory.

        Args:
            noisy_action: [batch, horizon, action_dim]
            timestep: [batch]
            obs: [batch, obs_dim]

        Returns:
            predicted_noise: [batch, horizon, action_dim]
        """
        # Embed time and observation
        t_emb = self.time_embed(timestep)  # [B, H]
        o_emb = self.obs_encoder(obs)  # [B, H]
        cond = t_emb + o_emb  # [B, H]

        # Reshape for 1D conv: [B, C, T]
        x = noisy_action.transpose(1, 2)

        # Downsampling
        skip_connections = []
        for block, conv in zip(self.down_blocks, self.down_convs):
            x = block(x, cond)
            skip_connections.append(x)
            x = conv(x)

        # Middle
        x = self.mid_block(x, cond)

        # Upsampling
        for conv, block, skip in zip(
            self.up_convs, self.up_blocks, reversed(skip_connections[:-1])
        ):
            x = conv(x)
            x = torch.cat([x, skip], dim=1)
            x = block(x, cond)

        # Output
        x = self.out_conv(x)

        return x.transpose(1, 2)  # [B, T, C]


class ResidualBlock1D(nn.Module):
    """Residual block with conditional normalization."""

    def __init__(self, in_channels: int, out_channels: int, cond_dim: int):
        super().__init__()

        self.conv1 = nn.Conv1d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv1d(out_channels, out_channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_channels)
        self.norm2 = nn.GroupNorm(8, out_channels)

        self.cond_proj = nn.Linear(cond_dim, out_channels * 2)

        if in_channels != out_channels:
            self.skip = nn.Conv1d(in_channels, out_channels, 1)
        else:
            self.skip = nn.Identity()

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = self.conv1(x)
        h = self.norm1(h)

        # Condition via scale and shift
        scale, shift = self.cond_proj(cond).chunk(2, dim=-1)
        h = h * (1 + scale.unsqueeze(-1)) + shift.unsqueeze(-1)

        h = F.mish(h)
        h = self.conv2(h)
        h = self.norm2(h)
        h = F.mish(h)

        return h + self.skip(x)


class DiffusionPolicy:
    """Complete diffusion policy for robot control."""

    def __init__(
        self,
        model: ConditionalUNet1D,
        action_dim: int = 7,
        action_horizon: int = 16,
        num_diffusion_steps: int = 100,
        beta_start: float = 0.0001,
        beta_end: float = 0.02
    ):
        self.model = model
        self.action_dim = action_dim
        self.action_horizon = action_horizon
        self.num_steps = num_diffusion_steps

        # Noise schedule
        self.betas = torch.linspace(beta_start, beta_end, num_diffusion_steps)
        self.alphas = 1 - self.betas
        self.alpha_cumprod = torch.cumprod(self.alphas, dim=0)

    def add_noise(
        self,
        action: torch.Tensor,
        t: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Add noise to action at timestep t."""
        device = action.device
        alpha_cumprod_t = self.alpha_cumprod[t].view(-1, 1, 1).to(device)

        noise = torch.randn_like(action)
        noisy_action = (
            torch.sqrt(alpha_cumprod_t) * action +
            torch.sqrt(1 - alpha_cumprod_t) * noise
        )

        return noisy_action, noise

    def training_loss(
        self,
        action: torch.Tensor,
        obs: torch.Tensor
    ) -> torch.Tensor:
        """Compute training loss."""
        batch_size = action.size(0)
        device = action.device

        # Sample random timesteps
        t = torch.randint(0, self.num_steps, (batch_size,), device=device)

        # Add noise
        noisy_action, noise = self.add_noise(action, t)

        # Predict noise
        predicted_noise = self.model(noisy_action, t.float(), obs)

        # MSE loss
        return F.mse_loss(predicted_noise, noise)

    @torch.no_grad()
    def sample(self, obs: torch.Tensor) -> torch.Tensor:
        """Generate action trajectory via reverse diffusion."""
        batch_size = obs.size(0)
        device = obs.device

        # Start from pure noise
        x = torch.randn(batch_size, self.action_horizon, self.action_dim, device=device)

        # Reverse diffusion
        for t in reversed(range(self.num_steps)):
            t_batch = torch.full((batch_size,), t, device=device, dtype=torch.float)

            # Predict noise
            predicted_noise = self.model(x, t_batch, obs)

            # Compute denoised action
            alpha = self.alphas[t]
            alpha_cumprod = self.alpha_cumprod[t]
            beta = self.betas[t]

            if t > 0:
                noise = torch.randn_like(x)
            else:
                noise = 0

            x = (
                1 / torch.sqrt(alpha) *
                (x - beta / torch.sqrt(1 - alpha_cumprod) * predicted_noise) +
                torch.sqrt(beta) * noise
            )

        return x


# Example usage
model = ConditionalUNet1D(action_dim=7, action_horizon=16, obs_dim=512)
policy = DiffusionPolicy(model)

# Training
obs = torch.randn(32, 512)  # Batch of observations
action = torch.randn(32, 16, 7)  # Ground truth trajectories
loss = policy.training_loss(action, obs)

# Inference
obs = torch.randn(1, 512)
trajectory = policy.sample(obs)
print(f"Generated trajectory: {trajectory.shape}")  # [1, 16, 7]
```

---

## Flow Matching Policies

### Flow Matching for Actions

Flow matching offers a simpler alternative to diffusion:

```python title="flow_matching_policy.py"
"""Flow matching policy for action generation."""
import torch
import torch.nn as nn
import torch.nn.functional as F

class FlowMatchingPolicy(nn.Module):
    """Flow matching for trajectory generation."""

    def __init__(
        self,
        action_dim: int = 7,
        action_horizon: int = 16,
        obs_dim: int = 512,
        hidden_dim: int = 256
    ):
        super().__init__()

        self.action_dim = action_dim
        self.action_horizon = action_horizon

        # Velocity network
        self.velocity_net = nn.Sequential(
            nn.Linear(action_dim * action_horizon + obs_dim + 1, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, action_dim * action_horizon)
        )

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        obs: torch.Tensor
    ) -> torch.Tensor:
        """
        Predict velocity field.

        Args:
            x: [batch, horizon * action_dim] - current position
            t: [batch, 1] - time
            obs: [batch, obs_dim] - observation

        Returns:
            velocity: [batch, horizon * action_dim]
        """
        inp = torch.cat([x, t, obs], dim=-1)
        return self.velocity_net(inp)

    def training_loss(
        self,
        action: torch.Tensor,
        obs: torch.Tensor
    ) -> torch.Tensor:
        """Compute flow matching loss."""
        batch_size = action.size(0)
        device = action.device

        # Flatten action
        action_flat = action.view(batch_size, -1)

        # Sample time uniformly
        t = torch.rand(batch_size, 1, device=device)

        # Sample noise (source distribution)
        noise = torch.randn_like(action_flat)

        # Interpolate between noise and action
        x_t = (1 - t) * noise + t * action_flat

        # Target velocity (optimal transport)
        target_velocity = action_flat - noise

        # Predict velocity
        predicted_velocity = self(x_t, t, obs)

        return F.mse_loss(predicted_velocity, target_velocity)

    @torch.no_grad()
    def sample(
        self,
        obs: torch.Tensor,
        num_steps: int = 10
    ) -> torch.Tensor:
        """Generate action via ODE integration."""
        batch_size = obs.size(0)
        device = obs.device

        # Start from noise
        x = torch.randn(
            batch_size,
            self.action_horizon * self.action_dim,
            device=device
        )

        # Euler integration
        dt = 1.0 / num_steps
        for i in range(num_steps):
            t = torch.full((batch_size, 1), i * dt, device=device)
            velocity = self(x, t, obs)
            x = x + velocity * dt

        return x.view(batch_size, self.action_horizon, self.action_dim)
```

---

## Action Safety and Constraints

### Constraint Enforcement

```python title="action_constraints.py"
"""Enforce safety constraints on generated actions."""
import torch
import torch.nn as nn
import numpy as np

class ActionConstraintEnforcer:
    """Enforce physical and safety constraints on actions."""

    def __init__(
        self,
        joint_limits: list,
        velocity_limits: list,
        workspace_bounds: dict,
        collision_checker=None
    ):
        self.joint_limits = torch.tensor(joint_limits)
        self.velocity_limits = torch.tensor(velocity_limits)
        self.workspace = workspace_bounds
        self.collision_checker = collision_checker

    def enforce_joint_limits(self, action: torch.Tensor) -> torch.Tensor:
        """Clamp actions to joint limits."""
        low = self.joint_limits[:, 0]
        high = self.joint_limits[:, 1]
        return torch.clamp(action, low, high)

    def enforce_velocity_limits(
        self,
        action: torch.Tensor,
        current_position: torch.Tensor,
        dt: float = 0.1
    ) -> torch.Tensor:
        """Limit action velocity."""
        delta = action - current_position
        max_delta = self.velocity_limits * dt

        # Clamp delta to velocity limits
        delta = torch.clamp(delta, -max_delta, max_delta)

        return current_position + delta

    def enforce_workspace(
        self,
        ee_position: torch.Tensor
    ) -> tuple[torch.Tensor, bool]:
        """Enforce workspace bounds on end-effector."""
        clamped = torch.zeros_like(ee_position)
        clamped[0] = torch.clamp(
            ee_position[0],
            self.workspace['x_min'],
            self.workspace['x_max']
        )
        clamped[1] = torch.clamp(
            ee_position[1],
            self.workspace['y_min'],
            self.workspace['y_max']
        )
        clamped[2] = torch.clamp(
            ee_position[2],
            self.workspace['z_min'],
            self.workspace['z_max']
        )

        modified = not torch.allclose(clamped, ee_position)
        return clamped, modified

    def smooth_trajectory(
        self,
        trajectory: torch.Tensor,
        smoothing_factor: float = 0.5
    ) -> torch.Tensor:
        """Apply smoothing to trajectory."""
        # Simple exponential moving average
        smoothed = trajectory.clone()
        for i in range(1, trajectory.size(0)):
            smoothed[i] = (
                smoothing_factor * smoothed[i-1] +
                (1 - smoothing_factor) * trajectory[i]
            )
        return smoothed

    def check_collision(self, action: torch.Tensor) -> bool:
        """Check if action would cause collision."""
        if self.collision_checker is None:
            return False
        return self.collision_checker(action.numpy())

    def apply_all_constraints(
        self,
        action: torch.Tensor,
        current_state: torch.Tensor
    ) -> tuple[torch.Tensor, dict]:
        """Apply all constraints and return info."""
        info = {
            'original': action.clone(),
            'modifications': []
        }

        # Joint limits
        action = self.enforce_joint_limits(action)
        if not torch.allclose(action, info['original']):
            info['modifications'].append('joint_limits')

        # Velocity limits
        action = self.enforce_velocity_limits(action, current_state)
        info['modifications'].append('velocity_limits')

        # Collision check
        if self.check_collision(action):
            info['modifications'].append('collision_stopped')
            action = current_state  # Stay in place

        info['final'] = action
        return action, info


class SafeActionWrapper(nn.Module):
    """Wrapper that enforces safety on any action generator."""

    def __init__(self, policy: nn.Module, constraint_enforcer: ActionConstraintEnforcer):
        super().__init__()
        self.policy = policy
        self.enforcer = constraint_enforcer

    def forward(self, *args, current_state: torch.Tensor = None, **kwargs):
        """Generate action and enforce constraints."""
        # Get raw action from policy
        raw_action = self.policy(*args, **kwargs)

        # Handle dict output
        if isinstance(raw_action, dict):
            action = raw_action.get('action', raw_action.get('position'))
        else:
            action = raw_action

        # Enforce constraints
        if current_state is not None:
            action, info = self.enforcer.apply_all_constraints(action, current_state)
        else:
            action = self.enforcer.enforce_joint_limits(action)
            info = {}

        if isinstance(raw_action, dict):
            raw_action['action'] = action
            raw_action['constraint_info'] = info
            return raw_action
        return action
```

---

## Exercise 1: Implement Action Transformer

:::tip Exercise 1: Action Prediction Head
**Objective**: Build a transformer-based action prediction module.

**Steps**:

1. Create input projections for vision, language, proprioception
2. Implement cross-attention between action queries and context
3. Add separate heads for position, rotation, gripper
4. Train on simulated pick-and-place trajectories
5. Evaluate action prediction accuracy

**Verification**:
```python
model = ActionTransformer()
action = model(vision_feat, lang_feat, proprio)
assert action['position'].shape == (batch, horizon, 3)
assert action['rotation'].shape == (batch, horizon, 4)
```

**Time Estimate**: 60 minutes
:::

---

## Exercise 2: Train Diffusion Policy

:::tip Exercise 2: Diffusion for Manipulation
**Objective**: Train a diffusion policy on manipulation demonstrations.

**Steps**:

1. Implement the noise schedule and forward process
2. Build the conditional U-Net denoiser
3. Implement the training loop with MSE loss
4. Add DDPM sampling for inference
5. Visualize generated trajectories

**Training Tips**:
- Use learning rate 1e-4
- Train for 100k steps minimum
- Start with action_horizon=8

**Time Estimate**: 90 minutes
:::

---

## Exercise 3: Safe Action Execution

:::tip Exercise 3: Constraint-Aware Actions
**Objective**: Implement safety constraints for action execution.

**Steps**:

1. Define joint limits and velocity bounds
2. Implement workspace enforcement
3. Add trajectory smoothing
4. Create collision checking interface
5. Test with intentionally unsafe predictions

**Safety Requirements**:
- Never exceed joint velocity by more than 10%
- Stop immediately if collision predicted
- Log all constraint violations

**Time Estimate**: 45 minutes
:::

---

## Summary

In this chapter, you learned:

- **Action Spaces**: Discrete, continuous, and hybrid representations
- **Transformer Actions**: Cross-attention for action prediction from context
- **Diffusion Policies**: Generating smooth, multi-modal trajectories
- **Flow Matching**: Simpler alternative to diffusion for action generation
- **Safety Constraints**: Enforcing limits, workspace bounds, and collision avoidance

Action generation is where VLA models produce their output - the robot motions that accomplish tasks. The choice of action representation, policy architecture, and constraint enforcement all significantly impact what behaviors the robot can express and how safely it operates.

Next, explore [End-to-End VLA Systems](/docs/module-4/end-to-end-vla) to see how all components come together into complete, deployable systems.

## Further Reading

- [Diffusion Policy Paper](https://arxiv.org/abs/2303.04137) - Chi et al.
- [Action Chunking with Transformers](https://arxiv.org/abs/2304.13705) - ACT Policy
- [Flow Matching for Robotics](https://arxiv.org/abs/2310.07842)
- [RT-1: Robotics Transformer](https://arxiv.org/abs/2212.06817)
- [Implicit Behavioral Cloning](https://arxiv.org/abs/2109.00137)
