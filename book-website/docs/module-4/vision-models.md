---
title: "Vision Models for Robotics"
sidebar_position: 3
description: "Adapting vision foundation models for robotics: feature extraction, fine-tuning, and deployment."
---

# Vision Models for Robotics

Modern vision models trained on internet-scale data provide powerful feature representations that can be adapted for robotic perception. This chapter covers techniques for leveraging these models in robotics applications, from feature extraction to efficient deployment.

## Overview

In this section, you will:

- Understand Vision Transformer (ViT) architecture
- Learn about CLIP and contrastive vision-language models
- Implement feature extraction for robotics
- Apply efficient fine-tuning techniques
- Optimize models for real-time inference

## Prerequisites

- Understanding of convolutional neural networks
- Basic knowledge of transformers
- PyTorch experience
- Familiarity with image processing concepts

---

## Vision Transformer (ViT)

### Patch-Based Image Processing

ViT treats images as sequences of patches, enabling transformer processing:

```
┌─────────────────────────────────────────────────────────────────┐
│                    Vision Transformer (ViT)                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   Input Image: 224 x 224 x 3                                    │
│                                                                  │
│   Step 1: Split into patches                                    │
│   ┌───┬───┬───┬───┐                                             │
│   │ 1 │ 2 │ 3 │...│  14x14 grid = 196 patches                  │
│   ├───┼───┼───┼───┤  Each patch: 16x16x3 = 768 values          │
│   │ 15│ 16│ 17│...│                                             │
│   ├───┼───┼───┼───┤                                             │
│   │...│...│...│...│                                             │
│   └───┴───┴───┴───┘                                             │
│                                                                  │
│   Step 2: Linear projection (patch embedding)                   │
│   Each patch → 768-dim vector                                   │
│                                                                  │
│   Step 3: Add position embeddings                               │
│   [CLS] [patch_1 + pos_1] [patch_2 + pos_2] ... [patch_196]    │
│                                                                  │
│   Step 4: Transformer encoder                                   │
│   12-24 layers of self-attention                                │
│                                                                  │
│   Output: [CLS] token → global image representation             │
│           Patch tokens → spatial features                        │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### ViT Implementation

```python title="vit_encoder.py"
"""Vision Transformer encoder for robotics."""
import torch
import torch.nn as nn
from einops import rearrange

class PatchEmbedding(nn.Module):
    """Convert image to patch embeddings."""

    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2

        # Linear projection of flattened patches
        self.proj = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x):
        # x: [batch, channels, height, width]
        x = self.proj(x)  # [batch, embed_dim, h/patch, w/patch]
        x = rearrange(x, 'b e h w -> b (h w) e')  # [batch, num_patches, embed_dim]
        return x


class ViTEncoder(nn.Module):
    """Vision Transformer for feature extraction."""

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_channels=3,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0
    ):
        super().__init__()

        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches

        # Learnable CLS token and position embeddings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))

        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        batch_size = x.size(0)

        # Patch embedding
        x = self.patch_embed(x)  # [batch, num_patches, embed_dim]

        # Add CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # [batch, num_patches+1, embed_dim]

        # Add position embeddings
        x = x + self.pos_embed

        # Transformer encoding
        x = self.encoder(x)
        x = self.norm(x)

        return x  # [batch, num_patches+1, embed_dim]

    def get_cls_token(self, x):
        """Get global image representation."""
        return self.forward(x)[:, 0]  # [batch, embed_dim]

    def get_patch_features(self, x):
        """Get spatial patch features."""
        return self.forward(x)[:, 1:]  # [batch, num_patches, embed_dim]


# Example usage
model = ViTEncoder()
image = torch.randn(4, 3, 224, 224)

features = model(image)
print(f"Full output: {features.shape}")  # [4, 197, 768]

cls_feature = model.get_cls_token(image)
print(f"CLS token: {cls_feature.shape}")  # [4, 768]

patch_features = model.get_patch_features(image)
print(f"Patch features: {patch_features.shape}")  # [4, 196, 768]
```

---

## CLIP: Vision-Language Alignment

### Contrastive Learning

CLIP learns aligned vision and language representations:

```
┌─────────────────────────────────────────────────────────────────┐
│                    CLIP Architecture                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   Image                          Text                            │
│   ┌─────────┐                   ┌─────────────────────┐         │
│   │ [IMG]   │                   │ "A robot arm"       │         │
│   └────┬────┘                   └──────────┬──────────┘         │
│        │                                   │                     │
│   ┌────▼────┐                   ┌──────────▼──────────┐         │
│   │  Image  │                   │       Text          │         │
│   │ Encoder │                   │      Encoder        │         │
│   │  (ViT)  │                   │   (Transformer)     │         │
│   └────┬────┘                   └──────────┬──────────┘         │
│        │                                   │                     │
│   ┌────▼────┐                   ┌──────────▼──────────┐         │
│   │  Image  │                   │       Text          │         │
│   │Embedding│                   │     Embedding       │         │
│   │ [512]   │                   │       [512]         │         │
│   └────┬────┘                   └──────────┬──────────┘         │
│        │                                   │                     │
│        └───────────────┬───────────────────┘                    │
│                        │                                         │
│              Contrastive Loss                                    │
│        (matching pairs should be similar)                        │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Using CLIP for Robotics

```python title="clip_robotics.py"
"""Using CLIP for robotic perception."""
import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image

class CLIPRobotPerception:
    """CLIP-based object understanding for robotics."""

    def __init__(self, model_name="openai/clip-vit-base-patch32"):
        self.model = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model.eval()

    @torch.no_grad()
    def get_image_features(self, image):
        """Extract visual features from image."""
        inputs = self.processor(images=image, return_tensors="pt")
        features = self.model.get_image_features(**inputs)
        return features / features.norm(dim=-1, keepdim=True)

    @torch.no_grad()
    def get_text_features(self, texts):
        """Extract text features."""
        inputs = self.processor(text=texts, return_tensors="pt", padding=True)
        features = self.model.get_text_features(**inputs)
        return features / features.norm(dim=-1, keepdim=True)

    @torch.no_grad()
    def classify_object(self, image, candidate_labels):
        """Zero-shot object classification."""
        # Get image and text features
        image_features = self.get_image_features(image)
        text_features = self.get_text_features(candidate_labels)

        # Compute similarity
        similarity = (image_features @ text_features.T).softmax(dim=-1)

        # Return label with highest similarity
        best_idx = similarity.argmax().item()
        return candidate_labels[best_idx], similarity[0, best_idx].item()

    @torch.no_grad()
    def find_object(self, image, object_description):
        """Find if described object is in image."""
        image_features = self.get_image_features(image)
        text_features = self.get_text_features([object_description])

        similarity = (image_features @ text_features.T).item()
        return similarity > 0.25  # Threshold for presence


# Example usage
perception = CLIPRobotPerception()

# Load image (from camera or file)
image = Image.open("scene.jpg")

# Zero-shot classification
labels = ["red cup", "blue bottle", "green box", "white plate"]
detected, confidence = perception.classify_object(image, labels)
print(f"Detected: {detected} (confidence: {confidence:.2f})")

# Object presence check
has_cup = perception.find_object(image, "a red cup on the table")
print(f"Cup present: {has_cup}")
```

---

## Feature Extraction for VLA

### Extracting Spatial Features

VLA models need spatial features, not just global representations:

```python title="spatial_features.py"
"""Extract spatial features for VLA models."""
import torch
import torch.nn as nn
from torchvision import transforms
from transformers import ViTModel, ViTImageProcessor

class SpatialFeatureExtractor(nn.Module):
    """Extract spatial features from images for VLA."""

    def __init__(self, model_name="google/vit-base-patch16-224", freeze=True):
        super().__init__()

        self.processor = ViTImageProcessor.from_pretrained(model_name)
        self.model = ViTModel.from_pretrained(model_name)

        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False

        self.hidden_size = self.model.config.hidden_size
        self.num_patches = (self.model.config.image_size // self.model.config.patch_size) ** 2

    def forward(self, images):
        """
        Extract features from images.

        Args:
            images: PIL images or tensor [batch, 3, H, W]

        Returns:
            features: [batch, num_patches, hidden_size]
        """
        # Process images
        if not isinstance(images, torch.Tensor):
            inputs = self.processor(images, return_tensors="pt")
            pixel_values = inputs.pixel_values
        else:
            pixel_values = images

        # Extract features
        outputs = self.model(pixel_values, output_hidden_states=True)

        # Get patch tokens (exclude CLS token)
        patch_features = outputs.last_hidden_state[:, 1:]

        return patch_features

    def get_feature_map(self, images):
        """Get 2D feature map."""
        features = self.forward(images)
        batch_size = features.size(0)
        h = w = int(self.num_patches ** 0.5)

        # Reshape to spatial grid
        feature_map = features.view(batch_size, h, w, self.hidden_size)
        feature_map = feature_map.permute(0, 3, 1, 2)  # [batch, hidden, h, w]

        return feature_map


# Example for VLA input
extractor = SpatialFeatureExtractor()

# Process robot camera image
camera_image = torch.randn(1, 3, 224, 224)  # Simulated camera input
features = extractor(camera_image)

print(f"Spatial features: {features.shape}")  # [1, 196, 768]
# These 196 tokens (14x14 grid) feed into VLA transformer
```

### Multi-Scale Features

```python title="multiscale_features.py"
"""Multi-scale feature extraction for manipulation."""
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiScaleEncoder(nn.Module):
    """Extract features at multiple scales."""

    def __init__(self, base_encoder, scales=[1.0, 0.5, 0.25]):
        super().__init__()
        self.encoder = base_encoder
        self.scales = scales

    def forward(self, x):
        """Extract multi-scale features."""
        features = []
        original_size = x.shape[-2:]

        for scale in self.scales:
            if scale != 1.0:
                scaled_size = (int(original_size[0] * scale), int(original_size[1] * scale))
                scaled_x = F.interpolate(x, size=scaled_size, mode='bilinear', align_corners=False)
            else:
                scaled_x = x

            feat = self.encoder(scaled_x)
            features.append(feat)

        return features


# Usage for detailed manipulation understanding
base_encoder = SpatialFeatureExtractor()
multi_scale = MultiScaleEncoder(base_encoder, scales=[1.0, 0.5])

image = torch.randn(1, 3, 224, 224)
scale_features = multi_scale(image)

for i, feat in enumerate(scale_features):
    print(f"Scale {multi_scale.scales[i]}: {feat.shape}")
```

---

## Efficient Fine-Tuning

### LoRA: Low-Rank Adaptation

Fine-tune vision models efficiently without full retraining:

```python title="lora_finetuning.py"
"""LoRA fine-tuning for vision models."""
import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model
from transformers import ViTForImageClassification

class LoRAVisionModel:
    """Fine-tune ViT with LoRA for robotics tasks."""

    def __init__(self, base_model_name="google/vit-base-patch16-224"):
        # Load base model
        self.model = ViTForImageClassification.from_pretrained(
            base_model_name,
            num_labels=10,  # Customize for your task
            ignore_mismatched_sizes=True
        )

        # Configure LoRA
        lora_config = LoraConfig(
            r=16,  # Rank of adaptation matrices
            lora_alpha=32,  # Scaling factor
            target_modules=["query", "value"],  # Attention layers to adapt
            lora_dropout=0.1,
            bias="none"
        )

        # Apply LoRA
        self.model = get_peft_model(self.model, lora_config)

        # Print trainable parameters
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        print(f"Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    def train_step(self, images, labels, optimizer):
        """Single training step."""
        self.model.train()
        optimizer.zero_grad()

        outputs = self.model(images, labels=labels)
        loss = outputs.loss

        loss.backward()
        optimizer.step()

        return loss.item()


# Example: Fine-tune for robot object classification
lora_model = LoRAVisionModel()
# Trainable: ~300K / ~86M (0.35%) - 300x fewer parameters!
```

### Adapter Layers

```python title="adapter_layers.py"
"""Adapter layers for efficient transfer learning."""
import torch
import torch.nn as nn

class Adapter(nn.Module):
    """Bottleneck adapter for efficient fine-tuning."""

    def __init__(self, hidden_size, bottleneck_size=64):
        super().__init__()
        self.down_proj = nn.Linear(hidden_size, bottleneck_size)
        self.up_proj = nn.Linear(bottleneck_size, hidden_size)
        self.act = nn.GELU()

    def forward(self, x):
        residual = x
        x = self.down_proj(x)
        x = self.act(x)
        x = self.up_proj(x)
        return x + residual


class AdaptedViT(nn.Module):
    """ViT with adapter layers for robotics."""

    def __init__(self, base_model, bottleneck_size=64):
        super().__init__()
        self.base = base_model

        # Freeze base model
        for param in self.base.parameters():
            param.requires_grad = False

        # Add adapters after each transformer block
        hidden_size = self.base.config.hidden_size
        self.adapters = nn.ModuleList([
            Adapter(hidden_size, bottleneck_size)
            for _ in range(self.base.config.num_hidden_layers)
        ])

    def forward(self, x):
        # Embed patches
        x = self.base.embeddings(x)

        # Pass through transformer with adapters
        for i, layer in enumerate(self.base.encoder.layer):
            x = layer(x)[0]
            x = self.adapters[i](x)

        return x
```

---

## Real-Time Deployment

### Model Optimization

```python title="optimize_inference.py"
"""Optimize vision models for real-time robotics."""
import torch
from torch.quantization import quantize_dynamic
import onnx
import onnxruntime as ort

def optimize_for_inference(model, example_input):
    """Optimize model for deployment."""

    # 1. TorchScript tracing
    model.eval()
    traced_model = torch.jit.trace(model, example_input)

    # 2. Dynamic quantization (INT8)
    quantized_model = quantize_dynamic(
        model,
        {torch.nn.Linear},
        dtype=torch.qint8
    )

    return traced_model, quantized_model


def export_to_onnx(model, example_input, output_path):
    """Export to ONNX for cross-platform deployment."""
    model.eval()

    torch.onnx.export(
        model,
        example_input,
        output_path,
        export_params=True,
        opset_version=14,
        input_names=['image'],
        output_names=['features'],
        dynamic_axes={
            'image': {0: 'batch'},
            'features': {0: 'batch'}
        }
    )

    # Verify ONNX model
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)

    return output_path


class ONNXInference:
    """ONNX runtime inference for production."""

    def __init__(self, model_path):
        # Use GPU if available
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        self.session = ort.InferenceSession(model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name

    def __call__(self, image):
        """Run inference."""
        if isinstance(image, torch.Tensor):
            image = image.numpy()

        outputs = self.session.run(None, {self.input_name: image})
        return outputs[0]


# Example deployment pipeline
model = SpatialFeatureExtractor()
example = torch.randn(1, 3, 224, 224)

# Export
export_to_onnx(model, example, "vision_encoder.onnx")

# Deploy
inference = ONNXInference("vision_encoder.onnx")
features = inference(example)
print(f"Inference output: {features.shape}")
```

### Benchmark Results

| Optimization | Latency (ms) | Memory (MB) | Accuracy |
|--------------|--------------|-------------|----------|
| PyTorch FP32 | 15.2 | 350 | 100% |
| TorchScript | 12.8 | 350 | 100% |
| INT8 Quantized | 8.4 | 95 | 99.5% |
| ONNX + TensorRT | 4.2 | 280 | 100% |

---

## Exercise 1: Implement ViT Feature Extractor

:::tip Exercise 1: Build Feature Extractor
**Objective**: Create a ViT-based feature extractor for robot perception.

**Steps**:

1. Implement patch embedding from scratch
2. Add position encodings
3. Build transformer encoder
4. Extract both CLS and patch features
5. Test on robot camera images

**Verification**:
```python
# Your extractor should pass:
assert features.shape == (batch_size, 196, 768)
assert cls_token.shape == (batch_size, 768)
```

**Time Estimate**: 60 minutes
:::

---

## Exercise 2: Zero-Shot Object Detection

:::tip Exercise 2: CLIP for Robot Tasks
**Objective**: Use CLIP for zero-shot object understanding.

**Steps**:

1. Load pre-trained CLIP model
2. Create object vocabulary for robot workspace
3. Implement object presence detection
4. Add object localization using attention
5. Test on tabletop manipulation scene

**Expected Output**:
- Identify objects in scene without training
- Localize objects using attention maps

**Time Estimate**: 45 minutes
:::

---

## Exercise 3: Deploy Optimized Model

:::tip Exercise 3: Real-Time Inference
**Objective**: Optimize vision model for robot deployment.

**Steps**:

1. Export model to ONNX
2. Apply INT8 quantization
3. Benchmark inference speed
4. Test accuracy on held-out data
5. Compare latency across optimizations

**Target**: < 10ms inference on RTX 3060

**Time Estimate**: 45 minutes
:::

---

## Summary

In this chapter, you learned:

- **Vision Transformers**: Patch-based processing for image understanding
- **CLIP**: Aligned vision-language representations for zero-shot recognition
- **Feature Extraction**: Spatial features for VLA models
- **Fine-Tuning**: LoRA and adapters for efficient adaptation
- **Deployment**: Optimization techniques for real-time inference

Vision encoders are the eyes of VLA systems—they transform raw pixels into semantic representations that enable language grounding and action generation.

Next, explore [Language Model Integration](/docs/module-4/language-integration) to understand how language models connect to robot control.

## Further Reading

- [ViT Paper](https://arxiv.org/abs/2010.11929) - An Image is Worth 16x16 Words
- [CLIP Paper](https://arxiv.org/abs/2103.00020) - Learning Transferable Visual Models
- [DINOv2](https://arxiv.org/abs/2304.07193) - Self-Supervised Vision Features
- [SigLIP](https://arxiv.org/abs/2303.15343) - Sigmoid Loss for Vision-Language
- [LoRA Paper](https://arxiv.org/abs/2106.09685) - Low-Rank Adaptation
