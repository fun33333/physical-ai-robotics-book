---
title: "Simulation Fundamentals"
sidebar_position: 2
description: "Core concepts of robotics simulation: physics engines, sensor modeling, and sim-to-real transfer."
---

# Simulation Fundamentals

Before diving into specific simulators, it's essential to understand what makes a simulation useful for robotics and AI development. This chapter covers the fundamental concepts that apply across all simulation platforms.

## Overview

In this section, you will:

- Understand physics engine architectures and trade-offs
- Learn sensor modeling approaches and noise characteristics
- Explore domain randomization for robust AI training
- Master sim-to-real transfer techniques
- Choose appropriate simulation fidelity for your use case

## Prerequisites

Before starting, ensure you have completed:

- [Module 1: ROS 2 Fundamentals](/docs/module-1)
- Basic understanding of coordinate frames and transformations

---

## What is Robotics Simulation?

### Definition

A robotics simulation is a software system that models:

1. **Robot kinematics**: How joints move and affect end-effector position
2. **Robot dynamics**: How forces and torques affect motion
3. **Sensors**: What the robot perceives about its environment
4. **Environment**: The world the robot operates in
5. **Physics**: Interactions between robot, objects, and environment

### Simulation Loop

Every simulator runs a core loop:

```
┌─────────────────────────────────────────────────────────────┐
│                     Simulation Loop                          │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   ┌──────────┐    ┌──────────┐    ┌──────────┐             │
│   │  Read    │    │  Physics │    │  Sensor  │             │
│   │ Commands │───▶│   Step   │───▶│  Update  │             │
│   └──────────┘    └──────────┘    └──────────┘             │
│        │                                  │                  │
│        │         ┌──────────┐            │                  │
│        └─────────│  Render  │◀───────────┘                  │
│                  └──────────┘                               │
│                       │                                      │
│                       ▼                                      │
│               [Next Time Step]                               │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Real-Time Factor

The **real-time factor (RTF)** measures simulation speed:

- RTF = 1.0: Simulation runs at real-world speed
- RTF > 1.0: Simulation runs faster than real-time
- RTF < 1.0: Simulation runs slower than real-time

```python title="Calculating RTF"
rtf = simulation_elapsed_time / wall_clock_time

# Example: 10 simulated seconds in 5 wall-clock seconds
rtf = 10.0 / 5.0  # RTF = 2.0 (2x faster than real-time)
```

:::tip Training Speed
For AI training, higher RTF is better. NVIDIA Isaac Sim can achieve RTF > 10,000 for simple physics on modern GPUs.
:::

---

## Physics Engines

### How Physics Engines Work

Physics engines compute how objects move and interact:

1. **Collision detection**: Determine which objects are touching
2. **Contact resolution**: Calculate forces at contact points
3. **Constraint solving**: Apply joint constraints, friction
4. **Integration**: Update positions and velocities

### Common Physics Engines

| Engine | Strengths | Used By |
|--------|-----------|---------|
| **ODE** | Stable, well-tested | Gazebo (default) |
| **Bullet** | Fast, good for many objects | Gazebo, PyBullet |
| **DART** | Accurate dynamics | Gazebo option |
| **PhysX** | GPU-accelerated, real-time | Isaac Sim, Unity |
| **MuJoCo** | Research-focused, accurate contacts | DeepMind |

### Physics Engine Trade-offs

```
        Accuracy
           ▲
           │
    MuJoCo │    DART
           │
           │    ODE
           │              Bullet
           │                      PhysX
           └─────────────────────────────▶ Speed
```

**Choosing a physics engine:**

- **Control algorithm development**: Choose accuracy (MuJoCo, DART)
- **Multi-robot simulation**: Choose speed (Bullet, PhysX)
- **RL training at scale**: Choose GPU acceleration (PhysX)

### Simulation Step Size

The **time step** affects accuracy and stability:

```python title="Step size considerations"
# Small step = more accurate but slower
physics_step = 0.001  # 1ms, 1000 steps per second

# Large step = faster but may be unstable
physics_step = 0.01   # 10ms, 100 steps per second

# Typical robotics simulation
physics_step = 0.001 to 0.004  # 1-4ms
```

:::warning Stability
Large time steps can cause unstable simulations where objects explode or pass through each other. Start with small steps and increase only if needed for speed.
:::

---

## Sensor Modeling

### Sensor Model Components

A realistic sensor model includes:

1. **Ideal measurement**: Perfect sensor reading
2. **Systematic errors**: Bias, scale factor, misalignment
3. **Random errors**: Gaussian noise, quantization
4. **Physical effects**: Occlusion, reflections, motion blur

### Camera Simulation

```python title="Camera noise model"
import numpy as np

def simulate_camera(ideal_image, params):
    """Add realistic noise to camera image."""
    # Gaussian noise (sensor noise)
    noise = np.random.normal(0, params.noise_std, ideal_image.shape)

    # Quantization (8-bit ADC)
    noisy = ideal_image + noise
    quantized = np.clip(noisy, 0, 255).astype(np.uint8)

    # Motion blur (if camera moving)
    if params.exposure_time > 0:
        quantized = apply_motion_blur(quantized, params.velocity)

    return quantized
```

**Camera parameters to model:**

| Parameter | Description | Typical Value |
|-----------|-------------|---------------|
| Resolution | Image size | 640x480 to 4K |
| FOV | Field of view | 60-120 degrees |
| Frame rate | FPS | 30-60 Hz |
| Noise | Sensor noise | SNR 30-50 dB |
| Latency | Processing delay | 10-50 ms |

### LiDAR Simulation

```python title="LiDAR noise model"
def simulate_lidar(ideal_ranges, params):
    """Add realistic noise to LiDAR ranges."""
    ranges = ideal_ranges.copy()

    # Range-dependent Gaussian noise
    noise_std = params.base_noise + params.range_factor * ranges
    noise = np.random.normal(0, noise_std)
    ranges += noise

    # Random dropouts (missed returns)
    dropout_mask = np.random.random(ranges.shape) < params.dropout_rate
    ranges[dropout_mask] = np.inf

    # Beam divergence (averaging at distance)
    # ... additional effects

    return ranges
```

**LiDAR parameters:**

| Parameter | Description | Typical Value |
|-----------|-------------|---------------|
| Channels | Vertical beams | 16, 32, 64, 128 |
| Points/sec | Scan rate | 300K - 2M |
| Range | Max distance | 50-200 m |
| Accuracy | Range error | ±2-5 cm |
| Angular resolution | Beam spacing | 0.1-0.4 degrees |

### IMU Simulation

```python title="IMU noise model"
class IMUSimulator:
    def __init__(self, params):
        self.gyro_bias = np.zeros(3)
        self.accel_bias = np.zeros(3)
        self.params = params

    def update(self, true_angular_vel, true_linear_accel, dt):
        """Simulate IMU measurements with noise."""
        # Bias random walk
        self.gyro_bias += np.random.normal(
            0, self.params.gyro_bias_instability * np.sqrt(dt), 3
        )
        self.accel_bias += np.random.normal(
            0, self.params.accel_bias_instability * np.sqrt(dt), 3
        )

        # Add bias and white noise
        gyro = true_angular_vel + self.gyro_bias + \
               np.random.normal(0, self.params.gyro_noise, 3)

        accel = true_linear_accel + self.accel_bias + \
                np.random.normal(0, self.params.accel_noise, 3)

        return gyro, accel
```

---

## Domain Randomization

### What is Domain Randomization?

Domain randomization trains AI models on varied simulated environments so they generalize to the real world.

**Key insight**: If a model works across many random simulations, it will likely work in reality (which is just one more variation).

### Types of Randomization

```
┌─────────────────────────────────────────────────────────────┐
│                  Domain Randomization                        │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   Visual Randomization          Physics Randomization        │
│   ┌─────────────────────┐      ┌─────────────────────┐      │
│   │ • Textures          │      │ • Mass/Inertia      │      │
│   │ • Lighting          │      │ • Friction          │      │
│   │ • Camera position   │      │ • Joint damping     │      │
│   │ • Colors            │      │ • Motor strength    │      │
│   │ • Backgrounds       │      │ • Sensor noise      │      │
│   └─────────────────────┘      └─────────────────────┘      │
│                                                              │
│   Dynamics Randomization       Environment Randomization     │
│   ┌─────────────────────┐      ┌─────────────────────┐      │
│   │ • Action delays     │      │ • Object positions  │      │
│   │ • Observation noise │      │ • Obstacle shapes   │      │
│   │ • Control frequency │      │ • Terrain           │      │
│   └─────────────────────┘      └─────────────────────┘      │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Implementation Example

```python title="domain_randomization.py"
import numpy as np

class DomainRandomizer:
    """Randomize simulation parameters for robust training."""

    def __init__(self, config):
        self.config = config

    def randomize_physics(self, robot):
        """Randomize robot physical properties."""
        # Mass: ±20%
        for link in robot.links:
            scale = np.random.uniform(0.8, 1.2)
            link.mass *= scale

        # Friction: 0.5 to 1.5
        for link in robot.links:
            link.friction = np.random.uniform(0.5, 1.5)

        # Joint damping: ±30%
        for joint in robot.joints:
            scale = np.random.uniform(0.7, 1.3)
            joint.damping *= scale

    def randomize_visuals(self, scene):
        """Randomize visual appearance."""
        # Lighting direction
        scene.light_direction = np.random.uniform(-1, 1, 3)
        scene.light_direction /= np.linalg.norm(scene.light_direction)

        # Light intensity
        scene.light_intensity = np.random.uniform(0.5, 1.5)

        # Random textures
        for obj in scene.objects:
            obj.texture = np.random.choice(self.config.textures)

    def randomize_sensors(self, robot):
        """Randomize sensor characteristics."""
        # Camera noise level
        robot.camera.noise_std = np.random.uniform(0.01, 0.05)

        # LiDAR dropout rate
        robot.lidar.dropout_rate = np.random.uniform(0.0, 0.1)

        # IMU bias
        robot.imu.gyro_bias = np.random.normal(0, 0.01, 3)
```

### Curriculum Learning

Start simple, increase randomization over training:

```python title="curriculum.py"
def get_randomization_scale(training_step, max_steps):
    """Gradually increase randomization."""
    progress = training_step / max_steps

    # Linear curriculum
    return min(1.0, progress * 2.0)  # Full randomization at 50%

# Usage
scale = get_randomization_scale(current_step, total_steps)
randomizer.randomize_physics(robot, scale=scale)
```

---

## Sim-to-Real Transfer

### The Reality Gap

Simulations never perfectly match reality. Common differences:

| Aspect | Simulation | Reality |
|--------|------------|---------|
| Physics | Approximated | Complex, unknown |
| Sensors | Modeled noise | True noise distribution |
| Actuators | Ideal response | Delays, nonlinearities |
| Environment | Simplified | Cluttered, dynamic |

### Transfer Techniques

**1. System Identification**

Measure real robot properties and match simulation:

```python title="system_identification.py"
def identify_motor_dynamics(real_robot):
    """Measure motor response characteristics."""
    # Send step commands, measure response
    command = 1.0  # rad/s
    real_robot.set_velocity_command(command)

    response = []
    for t in range(100):
        response.append(real_robot.get_velocity())
        time.sleep(0.01)

    # Fit first-order system: tau * dv/dt + v = K * command
    tau, K = fit_first_order(response, command)

    return {'time_constant': tau, 'gain': K}
```

**2. Domain Adaptation**

Adapt model to real domain after simulation training:

```
┌─────────────┐        ┌─────────────┐        ┌─────────────┐
│  Train in   │        │   Fine-tune │        │  Deploy on  │
│ Simulation  │───────▶│  on Real    │───────▶│ Real Robot  │
│             │        │    Data     │        │             │
└─────────────┘        └─────────────┘        └─────────────┘
     100K episodes         100 episodes           Production
```

**3. Reality-Aware Training**

Include real data during simulation training:

```python title="mixed_training.py"
def get_training_batch(sim_buffer, real_buffer, real_ratio=0.1):
    """Mix simulation and real data."""
    batch_size = 256
    real_samples = int(batch_size * real_ratio)
    sim_samples = batch_size - real_samples

    sim_batch = sim_buffer.sample(sim_samples)
    real_batch = real_buffer.sample(real_samples)

    return concatenate(sim_batch, real_batch)
```

---

## Choosing Simulation Fidelity

### Fidelity Levels

| Level | Description | Use Case | Speed |
|-------|-------------|----------|-------|
| **Kinematic** | Positions only, no physics | Path planning | Very fast |
| **Dynamic** | Basic physics, simple contacts | Control testing | Fast |
| **Detailed** | Full physics, friction | RL training | Medium |
| **Photorealistic** | Ray tracing, materials | Perception training | Slow |

### Matching Fidelity to Task

```python title="fidelity_selection.py"
def select_fidelity(task_type):
    """Choose appropriate simulation fidelity."""

    if task_type == 'path_planning':
        return {
            'physics': 'kinematic',
            'rendering': 'none',
            'sensors': 'ideal',
            'expected_rtf': 10000
        }

    elif task_type == 'control_development':
        return {
            'physics': 'dynamic',
            'rendering': 'basic',
            'sensors': 'noisy',
            'expected_rtf': 100
        }

    elif task_type == 'rl_training':
        return {
            'physics': 'detailed',
            'rendering': 'basic',
            'sensors': 'noisy',
            'expected_rtf': 10
        }

    elif task_type == 'perception_training':
        return {
            'physics': 'detailed',
            'rendering': 'photorealistic',
            'sensors': 'realistic',
            'expected_rtf': 0.1
        }
```

---

## Exercise 1: Sensor Noise Analysis

:::tip Exercise 1: Compare Ideal vs Noisy Sensors
**Objective**: Understand how sensor noise affects robot perception.

**Time Estimate**: 30 minutes

**Steps**:

1. Create a Python script that simulates LiDAR readings
2. Generate ideal readings for a simple room (4 walls)
3. Add Gaussian noise with different standard deviations
4. Visualize how noise affects wall detection
5. Calculate error statistics

**Starter Code**:

```python
import numpy as np
import matplotlib.pyplot as plt

def generate_ideal_lidar(room_size, num_beams=360):
    """Generate ideal LiDAR readings for a rectangular room."""
    angles = np.linspace(0, 2*np.pi, num_beams, endpoint=False)
    ranges = []

    for angle in angles:
        # Calculate intersection with walls
        # ... implement ray-wall intersection
        pass

    return angles, np.array(ranges)

def add_noise(ranges, noise_std):
    """Add Gaussian noise to ranges."""
    return ranges + np.random.normal(0, noise_std, ranges.shape)

# Generate and compare
angles, ideal = generate_ideal_lidar((10, 10))
noisy_01 = add_noise(ideal, 0.01)  # 1cm noise
noisy_05 = add_noise(ideal, 0.05)  # 5cm noise

# Visualize in polar coordinates
# ...
```

**Expected Result**: Visualization showing how noise distorts the room shape, with error statistics.
:::

---

## Exercise 2: Domain Randomization Experiment

:::tip Exercise 2: Test Randomization Effects
**Objective**: Observe how domain randomization helps generalization.

**Time Estimate**: 45 minutes

**Steps**:

1. Create a simple simulated sensor that detects objects
2. Train a classifier on fixed simulation settings
3. Train another classifier with domain randomization
4. Test both on "unseen" settings
5. Compare generalization performance

**Key Insight**: The randomized model should perform better on varied test conditions.

**Expected Outcome**:
```
Fixed training accuracy: 95%
Fixed test accuracy (new conditions): 60%

Randomized training accuracy: 85%
Randomized test accuracy (new conditions): 80%
```
:::

---

## Summary

In this chapter, you learned:

- **Physics engines**: ODE, Bullet, PhysX - trade accuracy for speed
- **Sensor modeling**: Add realistic noise for cameras, LiDAR, IMU
- **Domain randomization**: Vary simulation to improve real-world transfer
- **Sim-to-real**: System identification, adaptation, reality-aware training
- **Fidelity selection**: Match simulation complexity to your task

These concepts apply to all simulators. In the next chapter, you'll apply them using [Gazebo Basics](/docs/module-2/gazebo-basics).

## Further Reading

- [OpenAI Domain Randomization Paper](https://arxiv.org/abs/1703.06907)
- [Sim-to-Real Robot Learning Survey](https://arxiv.org/abs/2009.13303)
- [MuJoCo Physics Documentation](https://mujoco.readthedocs.io/)
- [NVIDIA Isaac Sim Replicator](https://developer.nvidia.com/isaac-sim)
