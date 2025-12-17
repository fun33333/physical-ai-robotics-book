---
title: "Module 2: The Digital Twin (Gazebo & Unity)"
sidebar_position: 1
description: "Build high-fidelity simulations for testing and training robotic systems using Gazebo and Unity."
---

# Module 2: The Digital Twin (Gazebo & Unity)

Simulation is the backbone of modern robotics development, enabling rapid iteration, safe testing, and scalable AI training without the cost and risk of physical hardware. This module teaches you to create digital twins - virtual replicas of robots and environments that behave like their physical counterparts.

## Module Overview

A digital twin is more than just a 3D model - it's a complete virtual representation that includes:

- **Physical properties**: Mass, inertia, friction, collision geometry
- **Sensor simulation**: Cameras, LiDAR, IMU with realistic noise models
- **Actuator dynamics**: Motor characteristics, joint limits, control interfaces
- **Environment interaction**: Physics-based contact, lighting, materials

:::info Why Simulation?
Training a robot manipulation policy in the real world might require millions of attempts over weeks or months. In simulation, the same training can happen in hours on a GPU cluster. Companies like Google, NVIDIA, and OpenAI rely heavily on simulation for AI development.
:::

## Learning Objectives

By the end of this module, you will be able to:

1. **Understand simulation trade-offs** - Physics fidelity vs speed, rendering quality vs computation
2. **Build Gazebo environments** - Create worlds, add robots, configure sensors
3. **Model robots with URDF** - Define kinematics, dynamics, and visual properties
4. **Integrate with ROS 2** - Bridge simulation data to ROS 2 topics and services
5. **Use Unity for robotics** - Leverage game engine capabilities for perception training
6. **Implement digital twin workflows** - Sync simulation with real robot state

## Prerequisites

Before starting this module, ensure you have:

- **Module 1 complete**: ROS 2 fundamentals, topics, services, actions
- **Linux environment**: Ubuntu 22.04 (native or WSL2)
- **GPU recommended**: For Unity and faster Gazebo rendering
- **Basic 3D concepts**: Coordinate frames, transformations (helpful but not required)

:::tip Hardware Note
Gazebo runs on CPU and works without a dedicated GPU. Unity requires a GPU for reasonable performance. See the [Hardware Guide](/docs/hardware/software-requirements) for specifications.
:::

## Module Structure

| Section | Focus | Duration |
|---------|-------|----------|
| [Simulation Fundamentals](/docs/module-2/simulation-fundamentals) | Physics engines, sim-to-real, domain randomization | 2-3 hours |
| [Gazebo Basics](/docs/module-2/gazebo-basics) | Installation, SDF, world creation, robot models | 4-5 hours |
| [Gazebo-ROS 2 Integration](/docs/module-2/gazebo-ros2-integration) | Bridges, sensors, control interfaces | 3-4 hours |
| [Unity Robotics](/docs/module-2/unity-robotics) | Unity Hub, URDF import, ROS-TCP connection | 4-5 hours |
| [Digital Twin Workflows](/docs/module-2/digital-twin-workflows) | Synchronization, validation, deployment patterns | 2-3 hours |

**Total estimated time**: 15-20 hours over 2 weeks

## Simulation Landscape

### Gazebo (Open Source)

Gazebo is the traditional choice for ROS-based robotics simulation:

**Strengths:**
- Deep ROS 2 integration
- Large model library
- Physics accuracy (ODE, Bullet, DART)
- Community support

**Considerations:**
- Rendering quality improving but not photorealistic
- GUI can be challenging for complex scenes

### Unity (Game Engine)

Unity brings modern game engine technology to robotics:

**Strengths:**
- Photorealistic rendering (HDRP)
- Diverse asset marketplace
- Excellent for synthetic data generation
- Cross-platform deployment

**Considerations:**
- Learning curve for non-game developers
- ROS integration requires additional setup
- Proprietary (free tier available)

### When to Use Which

| Use Case | Recommended Simulator |
|----------|----------------------|
| ROS 2 development and testing | Gazebo |
| Navigation and SLAM | Gazebo |
| Perception/vision training | Unity |
| Synthetic dataset generation | Unity |
| Quick prototyping | Gazebo |
| Photorealistic visualization | Unity |
| Multi-robot simulation | Gazebo |

## Key Concepts Preview

### URDF: Robot Description

The Unified Robot Description Format (URDF) defines robot structure:

```xml
<robot name="my_robot">
  <link name="base_link">
    <visual>...</visual>
    <collision>...</collision>
    <inertial>...</inertial>
  </link>
  <joint name="wheel_joint" type="continuous">
    <parent link="base_link"/>
    <child link="wheel"/>
    <axis xyz="0 1 0"/>
  </joint>
</robot>
```

### SDF: Simulation Description

Simulation Description Format extends URDF with world and physics:

```xml
<sdf version="1.9">
  <world name="warehouse">
    <physics type="ode">
      <real_time_factor>1.0</real_time_factor>
    </physics>
    <include>
      <uri>model://my_robot</uri>
    </include>
  </world>
</sdf>
```

### ROS 2-Gazebo Bridge

Connect simulation to ROS 2:

```yaml
- ros_topic_name: "/camera/image"
  gz_topic_name: "/world/default/model/robot/link/camera/sensor/camera/image"
  ros_type_name: "sensor_msgs/msg/Image"
  gz_type_name: "gz.msgs.Image"
  direction: GZ_TO_ROS
```

## Hands-On Projects

Throughout this module, you will build:

1. **Custom Robot Model** - URDF with multiple joints and sensors
2. **Warehouse World** - Complete Gazebo environment with obstacles
3. **Sensor Suite** - Camera, LiDAR, and IMU integration
4. **Unity Scene** - Photorealistic environment for perception
5. **Digital Twin** - Synchronized simulation matching physical setup

Each project includes step-by-step instructions and expected outcomes.

## Industry Applications

### Simulation in Production

- **Amazon Robotics**: Warehouse simulation for fleet optimization
- **Tesla**: Autopilot training in simulated driving scenarios
- **Boston Dynamics**: Physics simulation for locomotion development
- **NVIDIA Isaac**: Synthetic data generation for perception models
- **Waymo**: Millions of miles driven in simulation daily

### Research Applications

- **RL Policy Training**: Safe exploration in simulation
- **Failure Analysis**: Test edge cases without hardware damage
- **Algorithm Benchmarking**: Reproducible comparisons
- **Hardware Design**: Virtual prototyping before manufacturing

## Assessment

### Module Exercises

Each section contains hands-on exercises:

- **Guided exercises**: Step-by-step with expected output
- **Build exercises**: Create components from requirements
- **Integration exercises**: Combine multiple concepts

### Module Project

The module culminates in building a complete simulation environment:

- Custom robot model with multiple sensors
- Gazebo world with static and dynamic obstacles
- Full ROS 2 integration for control and sensing
- Optional Unity scene for perception training

See [Projects](/docs/assessments/projects) for detailed requirements.

## Getting Started

Ready to begin? Start with [Simulation Fundamentals](/docs/module-2/simulation-fundamentals) to understand the core concepts that apply across all simulation platforms.

:::note Pacing
If following the 8-week curriculum, this module spans Weeks 3-4. See the [Weekly Breakdown](/docs/weekly-breakdown/week-03-04) for day-by-day guidance.
:::
