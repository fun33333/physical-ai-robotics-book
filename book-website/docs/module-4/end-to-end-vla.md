---
title: "End-to-End VLA Systems"
sidebar_position: 6
description: "Building complete VLA systems: architecture patterns, training strategies, and real-world deployment."
---

# End-to-End VLA Systems

With an understanding of all the components, this chapter brings everything together into complete, deployable VLA systems. You'll learn architecture patterns that combine vision, language, and action generation into coherent wholes, along with training strategies and deployment considerations for real-world robotics.

## Overview

In this section, you will:

- Design complete VLA system architectures
- Implement end-to-end training pipelines
- Collect and manage demonstration data
- Fine-tune from foundation models
- Deploy VLA models on robot hardware
- Evaluate and iterate on system performance

## Prerequisites

- Completed all previous Module 4 chapters
- PyTorch and Hugging Face Transformers experience
- Access to robot simulation environment
- Understanding of imitation learning concepts

---

## VLA Architecture Patterns

### System Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    End-to-End VLA System                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   ┌──────────────────────────────────────────────────────────┐  │
│   │                      Inputs                               │  │
│   │   ┌───────────┐  ┌───────────┐  ┌───────────────────┐   │  │
│   │   │  Camera   │  │ Language  │  │  Proprioception   │   │  │
│   │   │  Images   │  │ Command   │  │  (Joint States)   │   │  │
│   │   └─────┬─────┘  └─────┬─────┘  └─────────┬─────────┘   │  │
│   └─────────┼──────────────┼──────────────────┼─────────────┘  │
│             │              │                  │                 │
│   ┌─────────▼──────────────▼──────────────────▼─────────────┐  │
│   │                    Encoders                              │  │
│   │   ┌───────────┐  ┌───────────┐  ┌───────────────────┐   │  │
│   │   │   Vision  │  │  Language │  │   State Encoder   │   │  │
│   │   │   (ViT)   │  │  (BERT)   │  │      (MLP)        │   │  │
│   │   └─────┬─────┘  └─────┬─────┘  └─────────┬─────────┘   │  │
│   └─────────┼──────────────┼──────────────────┼─────────────┘  │
│             │              │                  │                 │
│   ┌─────────▼──────────────▼──────────────────▼─────────────┐  │
│   │                  Fusion Module                           │  │
│   │          Cross-Attention Transformer Layers             │  │
│   └────────────────────────┬────────────────────────────────┘  │
│                            │                                    │
│   ┌────────────────────────▼────────────────────────────────┐  │
│   │                  Action Decoder                          │  │
│   │   ┌──────────────────────────────────────────────────┐  │  │
│   │   │  Diffusion / Transformer / Flow Matching Policy  │  │  │
│   │   └──────────────────────────────────────────────────┘  │  │
│   └────────────────────────┬────────────────────────────────┘  │
│                            │                                    │
│   ┌────────────────────────▼────────────────────────────────┐  │
│   │                     Outputs                              │  │
│   │      Action Trajectory: [a₁, a₂, ..., aₜ]               │  │
│   └─────────────────────────────────────────────────────────┘  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Architecture Variants

| Architecture | Vision | Language | Fusion | Action | Example |
|--------------|--------|----------|--------|--------|---------|
| **RT-1** | EfficientNet | USE | FiLM | Discrete tokens | Google |
| **RT-2** | ViT (PaLI) | LLM | Joint embedding | Discrete tokens | Google |
| **OpenVLA** | SigLIP | Llama 2 | Projector | Discrete tokens | Berkeley |
| **Octo** | ViT | T5 | Cross-attention | Diffusion | Berkeley |
| **ACT** | ResNet | - | Transformer | Continuous | Stanford |

---

## Complete VLA Implementation

### Full Model Architecture

```python title="vla_model.py"
"""Complete Vision-Language-Action model implementation."""
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from typing import Dict, Optional, Tuple

class VLAModel(nn.Module):
    """End-to-end VLA model for robot control."""

    def __init__(
        self,
        vision_encoder: str = "google/siglip-base-patch16-224",
        language_encoder: str = "bert-base-uncased",
        hidden_dim: int = 512,
        action_dim: int = 7,
        action_horizon: int = 16,
        num_transformer_layers: int = 6,
        num_heads: int = 8,
        freeze_encoders: bool = True
    ):
        super().__init__()

        self.action_dim = action_dim
        self.action_horizon = action_horizon

        # Vision encoder
        self.vision_encoder = AutoModel.from_pretrained(vision_encoder)
        self.vision_dim = self.vision_encoder.config.hidden_size

        # Language encoder
        self.language_encoder = AutoModel.from_pretrained(language_encoder)
        self.tokenizer = AutoTokenizer.from_pretrained(language_encoder)
        self.language_dim = self.language_encoder.config.hidden_size

        # Optionally freeze encoders
        if freeze_encoders:
            for param in self.vision_encoder.parameters():
                param.requires_grad = False
            for param in self.language_encoder.parameters():
                param.requires_grad = False

        # Projection layers
        self.vision_proj = nn.Linear(self.vision_dim, hidden_dim)
        self.language_proj = nn.Linear(self.language_dim, hidden_dim)
        self.proprio_proj = nn.Linear(action_dim, hidden_dim)

        # Fusion transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            batch_first=True
        )
        self.fusion_transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_transformer_layers
        )

        # Action decoder (using diffusion)
        from .action_generation import ConditionalUNet1D, DiffusionPolicy
        self.action_unet = ConditionalUNet1D(
            action_dim=action_dim,
            action_horizon=action_horizon,
            obs_dim=hidden_dim
        )
        self.diffusion = DiffusionPolicy(
            model=self.action_unet,
            action_dim=action_dim,
            action_horizon=action_horizon
        )

        # Learnable tokens
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim))
        self.action_token = nn.Parameter(torch.randn(1, 1, hidden_dim))

    def encode_observation(
        self,
        image: torch.Tensor,
        instruction: str,
        proprio: torch.Tensor
    ) -> torch.Tensor:
        """
        Encode multimodal observation.

        Args:
            image: [batch, 3, H, W] camera image
            instruction: text instruction (will be tokenized)
            proprio: [batch, action_dim] proprioceptive state

        Returns:
            fused_features: [batch, hidden_dim]
        """
        batch_size = image.size(0)
        device = image.device

        # Encode vision
        with torch.no_grad() if not self.vision_encoder.training else torch.enable_grad():
            vision_out = self.vision_encoder(pixel_values=image)
            vision_features = vision_out.last_hidden_state  # [B, num_patches, vision_dim]

        # Encode language
        tokens = self.tokenizer(
            instruction,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=77
        ).to(device)

        with torch.no_grad() if not self.language_encoder.training else torch.enable_grad():
            language_out = self.language_encoder(**tokens)
            language_features = language_out.last_hidden_state  # [B, seq_len, lang_dim]

        # Project to common dimension
        v = self.vision_proj(vision_features)  # [B, P, H]
        l = self.language_proj(language_features)  # [B, S, H]
        p = self.proprio_proj(proprio).unsqueeze(1)  # [B, 1, H]

        # Add special tokens
        cls = self.cls_token.expand(batch_size, -1, -1)
        action_tok = self.action_token.expand(batch_size, -1, -1)

        # Concatenate all tokens
        combined = torch.cat([cls, v, l, p, action_tok], dim=1)

        # Fuse with transformer
        fused = self.fusion_transformer(combined)

        # Extract action conditioning (last token)
        action_condition = fused[:, -1]  # [B, H]

        return action_condition

    def forward(
        self,
        image: torch.Tensor,
        instruction: str,
        proprio: torch.Tensor,
        action: torch.Tensor = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for training or inference.

        Args:
            image: [batch, 3, H, W]
            instruction: text string
            proprio: [batch, action_dim]
            action: [batch, horizon, action_dim] for training

        Returns:
            dict with loss (training) or action (inference)
        """
        # Encode observation
        obs_encoding = self.encode_observation(image, instruction, proprio)

        if action is not None:
            # Training: compute diffusion loss
            loss = self.diffusion.training_loss(action, obs_encoding)
            return {"loss": loss}
        else:
            # Inference: sample actions
            actions = self.diffusion.sample(obs_encoding)
            return {"actions": actions}

    @torch.no_grad()
    def predict_action(
        self,
        image: torch.Tensor,
        instruction: str,
        proprio: torch.Tensor
    ) -> torch.Tensor:
        """Predict action trajectory for deployment."""
        self.eval()
        output = self.forward(image, instruction, proprio)
        return output["actions"]


class VLATokenizer:
    """Unified tokenization for VLA inputs."""

    def __init__(
        self,
        text_tokenizer: str = "bert-base-uncased",
        action_vocab_size: int = 256,
        action_dim: int = 7
    ):
        self.text_tokenizer = AutoTokenizer.from_pretrained(text_tokenizer)
        self.action_vocab_size = action_vocab_size
        self.action_dim = action_dim

        # Special tokens for action
        self.action_start_token = action_vocab_size
        self.action_end_token = action_vocab_size + 1

    def tokenize_action(self, action: torch.Tensor) -> torch.Tensor:
        """Convert continuous action to tokens."""
        # Normalize to [0, 1]
        action_normalized = (action + 1) / 2  # Assume action in [-1, 1]
        action_normalized = torch.clamp(action_normalized, 0, 1)

        # Discretize
        tokens = (action_normalized * (self.action_vocab_size - 1)).long()
        return tokens

    def detokenize_action(self, tokens: torch.Tensor) -> torch.Tensor:
        """Convert tokens back to continuous action."""
        action_normalized = tokens.float() / (self.action_vocab_size - 1)
        action = action_normalized * 2 - 1  # Map back to [-1, 1]
        return action
```

### Training Pipeline

```python title="train_vla.py"
"""Training pipeline for VLA models."""
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import wandb
from tqdm import tqdm

class VLATrainer:
    """Trainer for VLA models."""

    def __init__(
        self,
        model: VLAModel,
        train_dataset,
        val_dataset,
        batch_size: int = 32,
        learning_rate: float = 1e-4,
        num_epochs: int = 100,
        gradient_accumulation: int = 1,
        device: str = "cuda"
    ):
        self.model = model.to(device)
        self.device = device
        self.num_epochs = num_epochs
        self.gradient_accumulation = gradient_accumulation

        # Data loaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4
        )

        # Optimizer
        self.optimizer = AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=0.01
        )

        # Scheduler
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=num_epochs * len(self.train_loader)
        )

    def train_epoch(self, epoch: int) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        num_batches = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        self.optimizer.zero_grad()

        for i, batch in enumerate(pbar):
            # Move to device
            images = batch["image"].to(self.device)
            instructions = batch["instruction"]
            proprio = batch["proprio"].to(self.device)
            actions = batch["action"].to(self.device)

            # Forward pass
            output = self.model(images, instructions, proprio, actions)
            loss = output["loss"] / self.gradient_accumulation

            # Backward pass
            loss.backward()

            if (i + 1) % self.gradient_accumulation == 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()

            total_loss += loss.item() * self.gradient_accumulation
            num_batches += 1

            pbar.set_postfix({"loss": loss.item() * self.gradient_accumulation})

        return total_loss / num_batches

    @torch.no_grad()
    def validate(self) -> dict:
        """Run validation."""
        self.model.eval()
        total_loss = 0
        total_mse = 0
        num_batches = 0

        for batch in self.val_loader:
            images = batch["image"].to(self.device)
            instructions = batch["instruction"]
            proprio = batch["proprio"].to(self.device)
            actions = batch["action"].to(self.device)

            # Compute loss
            output = self.model(images, instructions, proprio, actions)
            total_loss += output["loss"].item()

            # Compute action prediction error
            pred_actions = self.model.predict_action(images, instructions, proprio)
            mse = torch.nn.functional.mse_loss(pred_actions, actions)
            total_mse += mse.item()

            num_batches += 1

        return {
            "val_loss": total_loss / num_batches,
            "val_mse": total_mse / num_batches
        }

    def train(self):
        """Full training loop."""
        wandb.init(project="vla-training")

        best_val_loss = float("inf")

        for epoch in range(self.num_epochs):
            train_loss = self.train_epoch(epoch)
            val_metrics = self.validate()

            wandb.log({
                "train_loss": train_loss,
                **val_metrics,
                "epoch": epoch,
                "lr": self.scheduler.get_last_lr()[0]
            })

            print(f"Epoch {epoch}: train_loss={train_loss:.4f}, "
                  f"val_loss={val_metrics['val_loss']:.4f}")

            # Save best model
            if val_metrics["val_loss"] < best_val_loss:
                best_val_loss = val_metrics["val_loss"]
                torch.save(
                    self.model.state_dict(),
                    "best_vla_model.pt"
                )

        wandb.finish()
```

---

## Data Collection

### Demonstration Dataset

```python title="demonstration_dataset.py"
"""Dataset for robot demonstrations."""
import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
import h5py
from PIL import Image
from torchvision import transforms

class DemonstrationDataset(Dataset):
    """Dataset of robot demonstrations for VLA training."""

    def __init__(
        self,
        data_dir: str,
        action_horizon: int = 16,
        image_size: int = 224,
        augment: bool = True
    ):
        self.data_dir = Path(data_dir)
        self.action_horizon = action_horizon

        # Image transforms
        if augment:
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ColorJitter(0.1, 0.1, 0.1),
                transforms.RandomAffine(degrees=5, translate=(0.05, 0.05)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])

        # Load episode index
        self.episodes = self._load_episodes()
        self.samples = self._build_sample_index()

    def _load_episodes(self) -> list:
        """Load all episode files."""
        episodes = []
        for h5_file in sorted(self.data_dir.glob("*.h5")):
            with h5py.File(h5_file, "r") as f:
                episodes.append({
                    "path": h5_file,
                    "length": f["actions"].shape[0],
                    "instruction": f.attrs["instruction"]
                })
        return episodes

    def _build_sample_index(self) -> list:
        """Build index of valid samples."""
        samples = []
        for ep_idx, ep in enumerate(self.episodes):
            # Each timestep that has enough future actions
            for t in range(ep["length"] - self.action_horizon):
                samples.append((ep_idx, t))
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        ep_idx, t = self.samples[idx]
        episode = self.episodes[ep_idx]

        with h5py.File(episode["path"], "r") as f:
            # Load image
            image = Image.fromarray(f["images"][t])
            image = self.transform(image)

            # Load proprioception
            proprio = torch.tensor(f["proprio"][t], dtype=torch.float32)

            # Load action trajectory
            action = torch.tensor(
                f["actions"][t:t + self.action_horizon],
                dtype=torch.float32
            )

        return {
            "image": image,
            "instruction": episode["instruction"],
            "proprio": proprio,
            "action": action
        }


class DataCollector:
    """Collect demonstrations for VLA training."""

    def __init__(
        self,
        robot_interface,
        save_dir: str,
        camera_topic: str = "/camera/color/image_raw"
    ):
        self.robot = robot_interface
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.camera_topic = camera_topic

        self.episode_count = 0

    def collect_episode(self, instruction: str) -> bool:
        """Collect single demonstration episode."""
        print(f"Collecting episode for: {instruction}")
        print("Press Enter to start, 'q' to quit...")

        if input().lower() == 'q':
            return False

        # Storage for episode data
        images = []
        proprios = []
        actions = []

        print("Recording... Press Enter when done.")

        try:
            while True:
                # Get current observation
                image = self.robot.get_camera_image()
                proprio = self.robot.get_joint_positions()

                # Get action (from teleoperation or controller)
                action = self.robot.get_commanded_action()

                # Store
                images.append(image)
                proprios.append(proprio)
                actions.append(action)

                # Check for completion
                import sys
                import select
                if select.select([sys.stdin], [], [], 0.01)[0]:
                    if sys.stdin.readline().strip() == '':
                        break

        except KeyboardInterrupt:
            pass

        # Save episode
        if len(images) > 10:  # Minimum episode length
            self._save_episode(images, proprios, actions, instruction)
            self.episode_count += 1
            print(f"Saved episode {self.episode_count} ({len(images)} frames)")
            return True
        else:
            print("Episode too short, discarding.")
            return True

    def _save_episode(
        self,
        images: list,
        proprios: list,
        actions: list,
        instruction: str
    ):
        """Save episode to HDF5 file."""
        filename = self.save_dir / f"episode_{self.episode_count:05d}.h5"

        with h5py.File(filename, "w") as f:
            f.create_dataset("images", data=np.array(images))
            f.create_dataset("proprio", data=np.array(proprios))
            f.create_dataset("actions", data=np.array(actions))
            f.attrs["instruction"] = instruction
            f.attrs["length"] = len(images)
```

---

## Fine-Tuning from Foundation Models

### Loading Pre-trained Weights

```python title="finetune_vla.py"
"""Fine-tuning VLA from foundation models."""
import torch
from transformers import AutoModel

class VLAFineTuner:
    """Fine-tune VLA from pre-trained foundation models."""

    def __init__(
        self,
        base_model: VLAModel,
        pretrained_vla: str = None,
        freeze_vision: bool = True,
        freeze_language: bool = True,
        lora_rank: int = 16
    ):
        self.model = base_model

        # Load pre-trained VLA weights if available
        if pretrained_vla:
            self._load_pretrained(pretrained_vla)

        # Configure what to train
        self._configure_training(freeze_vision, freeze_language, lora_rank)

    def _load_pretrained(self, checkpoint_path: str):
        """Load pre-trained VLA weights."""
        state_dict = torch.load(checkpoint_path, map_location="cpu")

        # Handle partial loading
        model_dict = self.model.state_dict()
        pretrained_dict = {
            k: v for k, v in state_dict.items()
            if k in model_dict and v.shape == model_dict[k].shape
        }

        model_dict.update(pretrained_dict)
        self.model.load_state_dict(model_dict)

        print(f"Loaded {len(pretrained_dict)}/{len(model_dict)} parameters")

    def _configure_training(
        self,
        freeze_vision: bool,
        freeze_language: bool,
        lora_rank: int
    ):
        """Configure which parameters to train."""
        # Freeze encoders
        if freeze_vision:
            for param in self.model.vision_encoder.parameters():
                param.requires_grad = False

        if freeze_language:
            for param in self.model.language_encoder.parameters():
                param.requires_grad = False

        # Add LoRA to projection layers
        if lora_rank > 0:
            self._add_lora(lora_rank)

        # Print trainable parameters
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        print(f"Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    def _add_lora(self, rank: int):
        """Add LoRA adapters to linear layers."""
        from peft import LoraConfig, get_peft_model

        lora_config = LoraConfig(
            r=rank,
            lora_alpha=32,
            target_modules=["vision_proj", "language_proj", "proprio_proj"],
            lora_dropout=0.1
        )

        # Apply LoRA (simplified - would need proper implementation)
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Linear) and any(
                t in name for t in lora_config.target_modules
            ):
                # Add LoRA parameters
                in_features = module.in_features
                out_features = module.out_features

                module.lora_A = torch.nn.Parameter(
                    torch.randn(rank, in_features) * 0.01
                )
                module.lora_B = torch.nn.Parameter(
                    torch.zeros(out_features, rank)
                )


def create_finetuning_optimizer(model, learning_rate: float = 1e-4):
    """Create optimizer with different LR for different components."""
    param_groups = [
        # Frozen encoder params (if unfrozen for fine-tuning)
        {
            "params": [p for n, p in model.named_parameters()
                      if "encoder" in n and p.requires_grad],
            "lr": learning_rate * 0.1
        },
        # Projection layers
        {
            "params": [p for n, p in model.named_parameters()
                      if "proj" in n and p.requires_grad],
            "lr": learning_rate
        },
        # Action decoder
        {
            "params": [p for n, p in model.named_parameters()
                      if "action" in n or "diffusion" in n and p.requires_grad],
            "lr": learning_rate * 2
        },
        # LoRA parameters
        {
            "params": [p for n, p in model.named_parameters()
                      if "lora" in n and p.requires_grad],
            "lr": learning_rate * 5
        }
    ]

    return torch.optim.AdamW(param_groups, weight_decay=0.01)
```

---

## Deployment

### ROS 2 Deployment Node

```python title="vla_ros_node.py"
"""ROS 2 node for VLA model deployment."""
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import String
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import torch
import numpy as np
from threading import Lock

class VLAControlNode(Node):
    """ROS 2 node for VLA-based robot control."""

    def __init__(self):
        super().__init__('vla_controller')

        # Parameters
        self.declare_parameter('model_path', 'best_vla_model.pt')
        self.declare_parameter('control_rate', 10.0)
        self.declare_parameter('action_horizon', 16)
        self.declare_parameter('replan_interval', 4)

        # Load model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self._load_model()
        self.model.eval()

        # CV bridge
        self.bridge = CvBridge()

        # State
        self.current_image = None
        self.current_joints = None
        self.current_instruction = "do nothing"
        self.action_buffer = None
        self.action_index = 0
        self.state_lock = Lock()

        # Subscribers
        self.image_sub = self.create_subscription(
            Image, '/camera/color/image_raw',
            self.image_callback, 10
        )
        self.joint_sub = self.create_subscription(
            JointState, '/joint_states',
            self.joint_callback, 10
        )
        self.instruction_sub = self.create_subscription(
            String, '/vla/instruction',
            self.instruction_callback, 10
        )

        # Publisher
        self.action_pub = self.create_publisher(
            JointState, '/joint_commands', 10
        )

        # Control timer
        control_rate = self.get_parameter('control_rate').value
        self.control_timer = self.create_timer(
            1.0 / control_rate, self.control_loop
        )

        # Replan timer
        replan_interval = self.get_parameter('replan_interval').value
        self.replan_timer = self.create_timer(
            replan_interval / control_rate, self.replan
        )

        self.get_logger().info('VLA controller initialized')

    def _load_model(self) -> VLAModel:
        """Load VLA model."""
        model_path = self.get_parameter('model_path').value
        model = VLAModel()  # Configure as needed
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.to(self.device)
        return model

    def image_callback(self, msg):
        """Update current image."""
        with self.state_lock:
            self.current_image = self.bridge.imgmsg_to_cv2(msg, 'rgb8')

    def joint_callback(self, msg):
        """Update current joint state."""
        with self.state_lock:
            self.current_joints = np.array(msg.position)

    def instruction_callback(self, msg):
        """Update current instruction."""
        with self.state_lock:
            self.current_instruction = msg.data
            self.action_buffer = None  # Force replan
            self.action_index = 0
        self.get_logger().info(f'New instruction: {msg.data}')

    def replan(self):
        """Generate new action trajectory."""
        with self.state_lock:
            if self.current_image is None or self.current_joints is None:
                return

            image = self.current_image.copy()
            joints = self.current_joints.copy()
            instruction = self.current_instruction

        # Preprocess
        image_tensor = self._preprocess_image(image)
        proprio_tensor = torch.tensor(joints, dtype=torch.float32).unsqueeze(0)

        # Predict
        with torch.no_grad():
            actions = self.model.predict_action(
                image_tensor.to(self.device),
                instruction,
                proprio_tensor.to(self.device)
            )

        with self.state_lock:
            self.action_buffer = actions.cpu().numpy()[0]  # [horizon, action_dim]
            self.action_index = 0

        self.get_logger().debug('Replanned action trajectory')

    def _preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """Preprocess image for model."""
        from torchvision import transforms

        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        return transform(image).unsqueeze(0)

    def control_loop(self):
        """Execute action from buffer."""
        with self.state_lock:
            if self.action_buffer is None:
                return

            if self.action_index >= len(self.action_buffer):
                return

            action = self.action_buffer[self.action_index]
            self.action_index += 1

        # Publish action
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.position = action.tolist()
        self.action_pub.publish(msg)


def main():
    rclpy.init()
    node = VLAControlNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
```

### Inference Optimization

```python title="optimized_inference.py"
"""Optimize VLA model for real-time inference."""
import torch
import torch.nn as nn
import onnxruntime as ort

class OptimizedVLAInference:
    """Optimized VLA inference for deployment."""

    def __init__(
        self,
        model: VLAModel,
        use_fp16: bool = True,
        compile_model: bool = True
    ):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Move to device
        model = model.to(self.device)
        model.eval()

        # Convert to FP16
        if use_fp16 and self.device.type == 'cuda':
            model = model.half()

        # Compile model (PyTorch 2.0+)
        if compile_model:
            model = torch.compile(model, mode='reduce-overhead')

        self.model = model

    @torch.no_grad()
    def predict(
        self,
        image: torch.Tensor,
        instruction: str,
        proprio: torch.Tensor
    ) -> torch.Tensor:
        """Optimized prediction."""
        # Ensure correct dtype
        if self.model.vision_proj.weight.dtype == torch.float16:
            image = image.half()
            proprio = proprio.half()

        # Move to device
        image = image.to(self.device)
        proprio = proprio.to(self.device)

        # Predict
        return self.model.predict_action(image, instruction, proprio)

    def benchmark(self, num_iterations: int = 100) -> dict:
        """Benchmark inference speed."""
        import time

        # Dummy inputs
        image = torch.randn(1, 3, 224, 224)
        instruction = "pick up the red block"
        proprio = torch.randn(1, 7)

        # Warmup
        for _ in range(10):
            self.predict(image, instruction, proprio)

        # Synchronize CUDA
        if self.device.type == 'cuda':
            torch.cuda.synchronize()

        # Benchmark
        start = time.perf_counter()
        for _ in range(num_iterations):
            self.predict(image, instruction, proprio)

        if self.device.type == 'cuda':
            torch.cuda.synchronize()

        elapsed = time.perf_counter() - start
        avg_time = elapsed / num_iterations * 1000  # ms

        return {
            "avg_latency_ms": avg_time,
            "throughput_fps": 1000 / avg_time
        }


def export_to_onnx(model: VLAModel, output_path: str):
    """Export VLA model to ONNX."""
    model.eval()

    # Dummy inputs
    image = torch.randn(1, 3, 224, 224)
    proprio = torch.randn(1, 7)

    # Export encoder only (action decoder uses diffusion)
    class EncoderWrapper(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, image, proprio):
            return self.model.encode_observation(image, "example", proprio)

    wrapper = EncoderWrapper(model)

    torch.onnx.export(
        wrapper,
        (image, proprio),
        output_path,
        export_params=True,
        opset_version=14,
        input_names=['image', 'proprio'],
        output_names=['encoding'],
        dynamic_axes={
            'image': {0: 'batch'},
            'proprio': {0: 'batch'},
            'encoding': {0: 'batch'}
        }
    )

    print(f"Exported encoder to {output_path}")
```

---

## Evaluation

### Evaluation Metrics

```python title="evaluation.py"
"""Evaluation metrics for VLA systems."""
import numpy as np
from typing import Dict, List

class VLAEvaluator:
    """Evaluate VLA model performance."""

    def __init__(self, env, model):
        self.env = env
        self.model = model

    def evaluate_task(
        self,
        instruction: str,
        num_trials: int = 10,
        max_steps: int = 200
    ) -> Dict[str, float]:
        """Evaluate on specific task."""
        successes = []
        lengths = []
        rewards = []

        for trial in range(num_trials):
            obs = self.env.reset()
            total_reward = 0
            success = False

            for step in range(max_steps):
                # Get action from model
                action = self.model.predict_action(
                    obs['image'],
                    instruction,
                    obs['proprio']
                )

                # Execute first action from trajectory
                obs, reward, done, info = self.env.step(action[0])
                total_reward += reward

                if info.get('success', False):
                    success = True
                    break

                if done:
                    break

            successes.append(success)
            lengths.append(step + 1)
            rewards.append(total_reward)

        return {
            "success_rate": np.mean(successes),
            "avg_length": np.mean(lengths),
            "avg_reward": np.mean(rewards),
            "std_reward": np.std(rewards)
        }

    def evaluate_suite(
        self,
        tasks: List[str],
        num_trials_per_task: int = 10
    ) -> Dict[str, Dict[str, float]]:
        """Evaluate on suite of tasks."""
        results = {}
        for task in tasks:
            print(f"Evaluating: {task}")
            results[task] = self.evaluate_task(task, num_trials_per_task)
        return results

    def compute_generalization_gap(
        self,
        train_tasks: List[str],
        test_tasks: List[str]
    ) -> float:
        """Compute gap between train and test performance."""
        train_results = self.evaluate_suite(train_tasks)
        test_results = self.evaluate_suite(test_tasks)

        train_success = np.mean([r["success_rate"] for r in train_results.values()])
        test_success = np.mean([r["success_rate"] for r in test_results.values()])

        return train_success - test_success
```

---

## Exercise 1: Build Complete VLA

:::tip Exercise 1: End-to-End VLA System
**Objective**: Implement a complete VLA system from scratch.

**Steps**:

1. Implement vision encoder with pre-trained ViT
2. Add language encoder with BERT
3. Build fusion transformer
4. Integrate diffusion action decoder
5. Train on simulated demonstrations

**Verification**:
```python
model = VLAModel()
output = model(image, "pick up red cube", proprio)
assert "actions" in output or "loss" in output
```

**Time Estimate**: 2-3 hours
:::

---

## Exercise 2: Collect Demonstrations

:::tip Exercise 2: Data Collection Pipeline
**Objective**: Build demonstration collection system.

**Steps**:

1. Set up teleoperation interface
2. Implement episode recording
3. Create HDF5 storage format
4. Collect 50 demonstrations for one task
5. Build PyTorch Dataset class

**Data Requirements**:
- 30+ frames per episode
- Images, proprio, actions stored
- Instruction annotated per episode

**Time Estimate**: 90 minutes
:::

---

## Exercise 3: Deploy and Evaluate

:::tip Exercise 3: Real-World Deployment
**Objective**: Deploy VLA model on robot hardware.

**Steps**:

1. Export model with optimizations
2. Create ROS 2 control node
3. Implement action execution loop
4. Test on 3 different tasks
5. Measure success rate and latency

**Targets**:
- Inference < 100ms
- Success rate > 70% on trained tasks
- Smooth action execution

**Time Estimate**: 2 hours
:::

---

## Summary

In this chapter, you learned:

- **Architecture Patterns**: How RT-1, RT-2, OpenVLA, and Octo structure VLA models
- **Complete Implementation**: End-to-end VLA with vision, language, and action components
- **Training Pipelines**: Efficient training with demonstration data
- **Data Collection**: Building demonstration datasets for imitation learning
- **Fine-Tuning**: Adapting foundation models for robot tasks
- **Deployment**: ROS 2 integration and inference optimization
- **Evaluation**: Metrics and benchmarks for VLA performance

VLA systems represent the frontier of robot learning - combining the semantic understanding of foundation models with the precision needed for physical manipulation. Success requires careful attention to each component, from data collection through deployment.

This completes Module 4 on Vision-Language-Action models. You now have the foundation to build, train, and deploy modern VLA systems for robot control.

## Further Reading

- [RT-1 Paper](https://arxiv.org/abs/2212.06817) - Robotics Transformer
- [RT-2 Paper](https://arxiv.org/abs/2307.15818) - Vision-Language-Action Models
- [OpenVLA](https://openvla.github.io/) - Open-source VLA
- [Octo](https://octo-models.github.io/) - Generalist Robot Policy
- [RoboAgent](https://robopen.github.io/) - Large-Scale Robot Learning
- [Bridge Data](https://rail-berkeley.github.io/bridgedata/) - Robot Manipulation Dataset
