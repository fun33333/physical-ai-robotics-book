---
title: "Isaac Lab (Reinforcement Learning)"
sidebar_position: 5
description: "Train robot policies with Isaac Lab: massively parallel simulation for reinforcement learning."
---

# Isaac Lab (Reinforcement Learning)

Isaac Lab (formerly known as Isaac Orbit) enables training robot policies through reinforcement learning at unprecedented scale. By running thousands of parallel simulations on GPU, you can train policies in hours that would take weeks with traditional approaches. This chapter shows you how to leverage Isaac Lab for learning-based robot control.

## Overview

In this section, you will:

- Understand reinforcement learning fundamentals for robotics
- Install and configure Isaac Lab
- Train policies using existing environments
- Design reward functions for desired behaviors
- Create custom training environments
- Implement domain randomization for sim-to-real transfer
- Deploy trained policies to real robots

## Prerequisites

- Completed [Isaac Sim](/docs/module-3/isaac-sim) chapter
- NVIDIA GPU with 12GB+ VRAM (recommended for training)
- Basic understanding of neural networks
- Familiarity with PyTorch
- Python proficiency

---

## Reinforcement Learning Fundamentals

### The RL Framework

```
┌─────────────────────────────────────────────────────────────────┐
│              Reinforcement Learning Loop                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   ┌─────────┐  action aₜ   ┌─────────────────────────────────┐  │
│   │         │─────────────▶│                                 │  │
│   │  Agent  │              │        Environment              │  │
│   │ (Policy)│◀─────────────│   (Isaac Lab Simulation)        │  │
│   │         │  state sₜ₊₁  │                                 │  │
│   └─────────┘  reward rₜ   └─────────────────────────────────┘  │
│                                                                  │
│   Goal: Learn policy π(s) → a that maximizes Σ γᵗ rₜ           │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Key Concepts

| Concept | Description | Example in Robotics |
|---------|-------------|---------------------|
| **State** | Observation of environment | Joint positions, velocities, sensor data |
| **Action** | Control output | Joint torques, velocity commands |
| **Reward** | Scalar feedback signal | Distance to goal, energy consumption |
| **Policy** | Mapping from state to action | Neural network controller |
| **Episode** | Single trial from start to termination | Robot attempt at task |

### Why GPU-Parallel RL?

```
┌─────────────────────────────────────────────────────────────────┐
│              Training Time Comparison                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   Task: Quadruped locomotion (10M timesteps)                    │
│                                                                  │
│   Single Environment (CPU):                                      │
│   ████████████████████████████████████████  ~10 hours           │
│                                                                  │
│   256 Parallel Envs (Isaac Lab):                                │
│   ██                                        ~5 minutes          │
│                                                                  │
│   4096 Parallel Envs (Isaac Lab):                               │
│   █                                         ~30 seconds         │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Installation

### Setup Isaac Lab

```bash title="Install Isaac Lab"
# 1. Ensure Isaac Sim is installed via Omniverse Launcher

# 2. Clone Isaac Lab
cd ~
git clone https://github.com/isaac-sim/IsaacLab.git
cd IsaacLab

# 3. Create conda environment
conda create -n isaaclab python=3.10 -y
conda activate isaaclab

# 4. Install Isaac Lab
./isaaclab.sh --install

# 5. Verify installation
python -c "import omni.isaac.lab; print('Isaac Lab ready!')"

# 6. Run sample environment
python source/standalone/tutorials/00_sim/create_empty.py
```

### Project Structure

```
IsaacLab/
├── source/
│   ├── extensions/
│   │   ├── omni.isaac.lab/          # Core library
│   │   ├── omni.isaac.lab_assets/   # Robot/object assets
│   │   └── omni.isaac.lab_tasks/    # Pre-built environments
│   └── standalone/
│       ├── workflows/               # Training scripts
│       │   ├── rl_games/
│       │   ├── rsl_rl/
│       │   └── skrl/
│       └── tutorials/               # Learning examples
├── docs/                            # Documentation
└── isaaclab.sh                      # Management script
```

---

## Training Your First Policy

### Available Environments

Isaac Lab includes ready-to-use environments:

| Environment | Task | Observation Dim | Action Dim |
|-------------|------|-----------------|------------|
| `Isaac-Cartpole-v0` | Balance pole | 4 | 1 |
| `Isaac-Ant-v0` | Quadruped locomotion | 60 | 8 |
| `Isaac-Humanoid-v0` | Bipedal walking | 87 | 21 |
| `Isaac-Franka-Reach-v0` | End-effector positioning | 18 | 7 |
| `Isaac-Franka-Lift-v0` | Object manipulation | 23 | 7 |
| `Isaac-Anymal-C-Flat-v0` | Legged robot walking | 48 | 12 |

### Training with RL Games

```bash title="Train Cartpole"
# Navigate to Isaac Lab
cd ~/IsaacLab

# Train with default config
python source/standalone/workflows/rl_games/train.py \
  --task Isaac-Cartpole-v0 \
  --num_envs 1024 \
  --max_iterations 500

# Training output:
# [INFO] Policy saved to: logs/rl_games/cartpole/...
# [INFO] Final reward: 195.2 (solved!)
```

### Training with RSL-RL

```bash title="Train quadruped locomotion"
# Train Anymal walking
python source/standalone/workflows/rsl_rl/train.py \
  --task Isaac-Anymal-C-Flat-v0 \
  --num_envs 4096 \
  --max_iterations 1000

# Monitor with TensorBoard
tensorboard --logdir logs/rsl_rl/anymal_c_flat
```

### Visualizing Trained Policies

```bash title="Play trained policy"
# Run trained policy with visualization
python source/standalone/workflows/rl_games/play.py \
  --task Isaac-Cartpole-v0 \
  --checkpoint logs/rl_games/cartpole/nn/cartpole.pth \
  --num_envs 32
```

---

## Understanding Environments

### Environment Structure

```python title="Environment anatomy"
"""Isaac Lab environment components."""
from omni.isaac.lab.envs import ManagerBasedRLEnv

class MyEnv(ManagerBasedRLEnv):
    """Custom environment structure."""

    # Key components:
    # 1. Scene - robots, objects, sensors
    # 2. Observations - what agent sees
    # 3. Actions - how agent controls
    # 4. Rewards - feedback signal
    # 5. Terminations - when to reset
    # 6. Curriculum - progressive difficulty
```

### Configuration System

```python title="env_config.py"
"""Environment configuration example."""
from omni.isaac.lab.envs import ManagerBasedRLEnvCfg
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.managers import (
    ObservationGroupCfg,
    RewardTermCfg,
    TerminationTermCfg,
)

@configclass
class CartpoleEnvCfg(ManagerBasedRLEnvCfg):
    """Cartpole environment configuration."""

    # Scene configuration
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=1024,
        env_spacing=4.0,
    )

    # Observation configuration
    observations: ObservationsCfg = ObservationsCfg()

    # Action configuration
    actions: ActionsCfg = ActionsCfg()

    # Reward configuration
    rewards: RewardsCfg = RewardsCfg()

    # Termination configuration
    terminations: TerminationsCfg = TerminationsCfg()

    # Episode length
    episode_length_s: float = 5.0
```

---

## Reward Engineering

### Reward Function Design

```python title="reward_functions.py"
"""Reward function examples for robot learning."""
from omni.isaac.lab.managers import RewardTermCfg
import omni.isaac.lab.utils.math as math_utils

def reward_alive(env) -> torch.Tensor:
    """Reward for staying alive (not falling)."""
    return torch.ones(env.num_envs, device=env.device)

def reward_forward_velocity(env, target_vel: float = 1.0) -> torch.Tensor:
    """Reward for moving forward at target velocity."""
    # Get base linear velocity
    base_vel = env.scene["robot"].data.root_lin_vel_b
    forward_vel = base_vel[:, 0]  # x-velocity

    # Reward being close to target
    vel_error = torch.abs(forward_vel - target_vel)
    return torch.exp(-vel_error / 0.25)

def reward_action_smoothness(env) -> torch.Tensor:
    """Penalize jerky actions."""
    action_diff = env.actions - env.previous_actions
    return -torch.sum(action_diff ** 2, dim=-1)

def reward_energy_efficiency(env) -> torch.Tensor:
    """Penalize high torques."""
    torques = env.scene["robot"].data.applied_torque
    return -torch.sum(torch.abs(torques), dim=-1) * 0.001
```

### Reward Configuration

```python title="rewards_cfg.py"
"""Reward configuration for locomotion."""
from omni.isaac.lab.managers import RewardTermCfg as RewTerm

@configclass
class RewardsCfg:
    """Reward terms for locomotion task."""

    # Positive rewards
    alive = RewTerm(func=reward_alive, weight=1.0)
    forward_velocity = RewTerm(
        func=reward_forward_velocity,
        weight=2.0,
        params={"target_vel": 1.5}
    )

    # Penalties
    action_smoothness = RewTerm(
        func=reward_action_smoothness,
        weight=0.1
    )
    energy = RewTerm(
        func=reward_energy_efficiency,
        weight=0.05
    )

    # Termination penalty
    termination = RewTerm(
        func=lambda env: env.reset_buf.float(),
        weight=-100.0
    )
```

### Common Reward Patterns

| Pattern | Use Case | Implementation |
|---------|----------|----------------|
| **Sparse** | Goal reaching | +1 at goal, 0 otherwise |
| **Dense** | Continuous feedback | Distance-based shaping |
| **Curriculum** | Progressive difficulty | Increase threshold over time |
| **Imitation** | Match demonstrations | Minimize action/state difference |

---

## Custom Environments

### Creating a Reach Task

```python title="custom_reach_env.py"
"""Custom reaching environment for robot arm."""
import torch
from omni.isaac.lab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg
from omni.isaac.lab.assets import Articulation, ArticulationCfg
from omni.isaac.lab.managers import SceneEntityCfg

@configclass
class ReachEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for reaching task."""

    # Simulation settings
    sim: SimulationCfg = SimulationCfg(dt=1/120)

    # Scene with robot
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=1024,
        env_spacing=2.5,
    )

    # Robot configuration
    robot: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path="path/to/robot.usd"
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            joint_pos={".*": 0.0},
        ),
    )

    # Observations: joint states + target position
    observations: ObservationsCfg = ObservationsCfg(
        policy=ObservationGroupCfg(
            observations={
                "joint_pos": JointPosCfg(),
                "joint_vel": JointVelCfg(),
                "target_pos": TargetPosCfg(),
            }
        )
    )

    # Actions: joint position targets
    actions: ActionsCfg = ActionsCfg(
        joint_effort=JointEffortActionCfg(
            asset_name="robot",
            joint_names=[".*"],
        )
    )


class ReachEnv(ManagerBasedRLEnv):
    """Custom reaching environment."""

    cfg: ReachEnvCfg

    def __init__(self, cfg: ReachEnvCfg):
        super().__init__(cfg)

        # Target position (randomized per episode)
        self.target_pos = torch.zeros(
            self.num_envs, 3, device=self.device
        )

    def _reset_idx(self, env_ids: torch.Tensor):
        """Reset environments and randomize targets."""
        super()._reset_idx(env_ids)

        # Randomize target position in workspace
        self.target_pos[env_ids] = torch.rand(
            len(env_ids), 3, device=self.device
        ) * 0.4 + torch.tensor([0.3, -0.2, 0.2])

    def _compute_rewards(self) -> torch.Tensor:
        """Compute reward based on distance to target."""
        ee_pos = self.scene["robot"].data.body_pos_w[:, -1]  # End-effector
        distance = torch.norm(ee_pos - self.target_pos, dim=-1)

        # Dense reward: closer is better
        reward = -distance

        # Bonus for reaching target
        reached = distance < 0.05
        reward += reached.float() * 10.0

        return reward
```

### Registering Custom Environment

```python title="__init__.py"
"""Register custom environment."""
import gymnasium as gym
from .custom_reach_env import ReachEnv, ReachEnvCfg

gym.register(
    id="Isaac-Custom-Reach-v0",
    entry_point="omni.isaac.lab_tasks.custom:ReachEnv",
    kwargs={"env_cfg": ReachEnvCfg()},
)
```

---

## Domain Randomization

### Randomization for Sim-to-Real

```python title="domain_randomization.py"
"""Domain randomization configuration."""
from omni.isaac.lab.managers import RandomizationTermCfg

@configclass
class RandomizationCfg:
    """Randomization for sim-to-real transfer."""

    # Physics randomization
    mass_randomization = RandomizationTermCfg(
        func=randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "mass_range": (0.8, 1.2),  # ±20%
        }
    )

    friction_randomization = RandomizationTermCfg(
        func=randomize_physics_material,
        mode="reset",
        params={
            "static_friction_range": (0.5, 1.5),
            "dynamic_friction_range": (0.5, 1.5),
        }
    )

    # Actuator randomization
    motor_strength = RandomizationTermCfg(
        func=randomize_actuator_gains,
        mode="startup",
        params={
            "stiffness_range": (0.9, 1.1),
            "damping_range": (0.9, 1.1),
        }
    )

    # Observation noise
    observation_noise = RandomizationTermCfg(
        func=add_observation_noise,
        mode="interval",
        interval_range_s=(0.0, 0.0),  # Every step
        params={
            "noise_std": 0.01,
        }
    )

    # External disturbances
    push_robot = RandomizationTermCfg(
        func=push_by_setting_velocity,
        mode="interval",
        interval_range_s=(5.0, 10.0),
        params={
            "velocity_range": (-0.5, 0.5),
        }
    )
```

### Curriculum Learning

```python title="curriculum.py"
"""Curriculum learning for progressive training."""
from omni.isaac.lab.managers import CurriculumTermCfg

@configclass
class CurriculumCfg:
    """Progressive difficulty curriculum."""

    terrain_difficulty = CurriculumTermCfg(
        func=terrain_levels_curriculum,
        params={
            "num_levels": 5,
            "success_threshold": 0.8,
        }
    )

    command_velocity = CurriculumTermCfg(
        func=velocity_curriculum,
        params={
            "initial_range": (0.0, 0.5),
            "final_range": (0.0, 2.0),
            "num_steps": 1000,
        }
    )
```

---

## Policy Deployment

### Export Policy for Deployment

```python title="export_policy.py"
"""Export trained policy for deployment."""
import torch
import onnx

def export_to_onnx(checkpoint_path: str, output_path: str):
    """Export PyTorch policy to ONNX."""
    # Load trained policy
    checkpoint = torch.load(checkpoint_path)
    policy = checkpoint["policy"]
    policy.eval()

    # Create dummy input matching observation shape
    dummy_input = torch.randn(1, policy.obs_dim)

    # Export to ONNX
    torch.onnx.export(
        policy,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=13,
        input_names=["observation"],
        output_names=["action"],
        dynamic_axes={
            "observation": {0: "batch"},
            "action": {0: "batch"},
        }
    )

    print(f"Exported to {output_path}")

# Usage
export_to_onnx(
    "logs/rsl_rl/anymal/policy.pt",
    "deployed_policy.onnx"
)
```

### ROS 2 Deployment Node

```python title="policy_node.py"
"""ROS 2 node for deployed policy."""
import rclpy
from rclpy.node import Node
import numpy as np
import onnxruntime as ort
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray

class PolicyNode(Node):
    def __init__(self):
        super().__init__('policy_node')

        # Load ONNX policy
        self.session = ort.InferenceSession("deployed_policy.onnx")
        self.input_name = self.session.get_inputs()[0].name

        # Subscribers
        self.joint_sub = self.create_subscription(
            JointState, '/joint_states',
            self.joint_callback, 10
        )

        # Publisher
        self.action_pub = self.create_publisher(
            Float64MultiArray, '/joint_commands', 10
        )

        # State buffer
        self.joint_pos = np.zeros(12)
        self.joint_vel = np.zeros(12)

        # Control loop at 50 Hz
        self.timer = self.create_timer(0.02, self.control_loop)

    def joint_callback(self, msg):
        """Update joint state from robot."""
        self.joint_pos = np.array(msg.position)
        self.joint_vel = np.array(msg.velocity)

    def control_loop(self):
        """Run policy and publish actions."""
        # Build observation
        obs = np.concatenate([
            self.joint_pos,
            self.joint_vel,
            # Add command velocity, gravity vector, etc.
        ]).astype(np.float32).reshape(1, -1)

        # Run inference
        action = self.session.run(None, {self.input_name: obs})[0]

        # Publish action
        msg = Float64MultiArray()
        msg.data = action.flatten().tolist()
        self.action_pub.publish(msg)


def main():
    rclpy.init()
    node = PolicyNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
```

---

## Exercise 1: Train Cartpole

:::tip Exercise 1: Basic Policy Training
**Objective**: Train and evaluate a cartpole balancing policy.

**Steps**:

1. Launch Isaac Lab with cartpole environment
2. Train for 500 iterations with 1024 environments
3. Monitor training with TensorBoard
4. Visualize the trained policy
5. Experiment with different hyperparameters

**Commands**:
```bash
# Train
python source/standalone/workflows/rl_games/train.py \
  --task Isaac-Cartpole-v0 \
  --num_envs 1024

# Monitor
tensorboard --logdir logs/rl_games/cartpole

# Play
python source/standalone/workflows/rl_games/play.py \
  --task Isaac-Cartpole-v0 \
  --checkpoint <path_to_checkpoint>
```

**Expected Result**: Policy achieves 195+ reward (solved).

**Time Estimate**: 30 minutes
:::

---

## Exercise 2: Quadruped Locomotion

:::tip Exercise 2: Train Walking Robot
**Objective**: Train a quadruped robot to walk forward.

**Steps**:

1. Set up Anymal environment
2. Train with 4096 parallel environments
3. Analyze reward components
4. Test on different terrain types
5. Export policy for deployment

**Training Command**:
```bash
python source/standalone/workflows/rsl_rl/train.py \
  --task Isaac-Anymal-C-Flat-v0 \
  --num_envs 4096 \
  --max_iterations 2000
```

**Analysis Tasks**:
- Plot learning curves for each reward term
- Compare forward velocity vs energy usage
- Test robustness to external pushes

**Time Estimate**: 60 minutes
:::

---

## Exercise 3: Custom Environment

:::tip Exercise 3: Create Your Own Task
**Objective**: Build a custom environment for a specific task.

**Steps**:

1. Choose a task (pushing, picking, balancing)
2. Define observation and action spaces
3. Implement reward function
4. Configure domain randomization
5. Train and iterate on reward design

**Template**:
```python
# Start from existing environment
# Modify rewards and terminations
# Add task-specific observations
```

**Deliverables**:
- Working environment configuration
- Trained policy achieving task
- Documentation of reward design choices

**Time Estimate**: 2-3 hours
:::

---

## Summary

In this chapter, you learned:

- **RL Fundamentals**: States, actions, rewards, and policies
- **Isaac Lab Setup**: Installation and project structure
- **Training**: Using RL Games and RSL-RL for policy learning
- **Reward Engineering**: Designing effective reward functions
- **Custom Environments**: Building task-specific simulations
- **Sim-to-Real**: Domain randomization and policy deployment

Isaac Lab's massively parallel simulation enables training robot policies that would be impractical with traditional approaches. The key to success is careful reward design and thorough domain randomization.

Next, explore [Perception Pipelines](/docs/module-3/perception-pipelines) to build end-to-end systems combining learned policies with real-time perception.

## Further Reading

- [Isaac Lab Documentation](https://isaac-sim.github.io/IsaacLab/)
- [Isaac Lab GitHub](https://github.com/isaac-sim/IsaacLab)
- [RL Games](https://github.com/Denys88/rl_games)
- [RSL-RL](https://github.com/leggedrobotics/rsl_rl)
- [Spinning Up in Deep RL](https://spinningup.openai.com/)
