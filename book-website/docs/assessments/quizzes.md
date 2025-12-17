---
title: "Knowledge Checks"
sidebar_position: 3
description: "Self-assessment quizzes to verify understanding of key concepts throughout the curriculum."
---

# Knowledge Checks

Knowledge checks are short self-assessments distributed throughout the curriculum to verify your understanding of key concepts. They're designed to identify gaps early, before they compound into larger problems.

These checks cover theoretical concepts, architectural decisions, troubleshooting scenarios, and best practices. They're not graded in a traditional sense but help you gauge your readiness to proceed and identify topics that need review.

Each knowledge check includes explanations for all answers, turning assessment into another learning opportunity. We recommend completing these honestly without looking up answers - they're most valuable as honest self-assessment.

---

## Quiz 1: ROS 2 Fundamentals (Weeks 1-2)

### Concepts Covered

- ROS 2 architecture and computation graph
- Nodes, topics, services, and actions
- QoS policies and DDS
- Lifecycle nodes
- Nav2 and MoveIt 2 basics

### Sample Questions

:::note Question 1
What is the primary difference between ROS 2 topics and services?

A) Topics are synchronous, services are asynchronous
B) Topics use publish-subscribe, services use request-response
C) Topics are for large data, services are for small data
D) Topics require DDS, services don't

<details>
<summary>Answer</summary>

**B) Topics use publish-subscribe, services use request-response**

Topics implement a publish-subscribe pattern where publishers send messages without knowing who (if anyone) is receiving them. Services use request-response where a client sends a request and waits for a response from a specific server. Topics are typically asynchronous (publishers don't wait), while service calls can be synchronous or asynchronous.

</details>
:::

:::note Question 2
Which QoS policy would you use to ensure a late-joining subscriber receives the last published message?

A) Reliability: Best Effort
B) Durability: Transient Local
C) History: Keep All
D) Liveliness: Manual

<details>
<summary>Answer</summary>

**B) Durability: Transient Local**

Transient Local durability means the publisher stores the last message(s) and sends them to new subscribers when they connect. This is useful for "latching" behavior where subscribers need the current state even if they missed the original publication.

</details>
:::

:::note Question 3
In a lifecycle node, what state transition must occur before `on_activate` can be called?

A) Unconfigured → Active
B) Inactive → Active
C) Configured → Active
D) Finalized → Active

<details>
<summary>Answer</summary>

**B) Inactive → Active**

The lifecycle node state machine requires: Unconfigured → (configure) → Inactive → (activate) → Active. You cannot skip states. The `on_configure` callback must succeed before `on_activate` can be called.

</details>
:::

:::note Question 4
What component in Nav2 is responsible for avoiding obstacles detected during navigation?

A) Global Planner
B) Controller Server
C) Costmap 2D
D) BT Navigator

<details>
<summary>Answer</summary>

**B) Controller Server**

While the Costmap 2D provides obstacle information, the Controller Server is responsible for generating commands that follow the path while avoiding obstacles. It runs at high frequency (typically 20Hz) and adjusts velocity commands to avoid collisions detected in the local costmap.

</details>
:::

### Full Quiz

The complete quiz contains 15-20 questions covering all Weeks 1-2 material. Access the full quiz through your course platform.

---

## Quiz 2: Simulation (Weeks 3-4)

### Concepts Covered

- Physics simulation fundamentals
- Gazebo world building and SDF
- URDF robot modeling
- Sensor simulation and noise
- Unity robotics setup
- ROS 2 bridge integration

### Sample Questions

:::note Question 1
Which physics engine does Gazebo Fortress (Ignition) use by default?

A) ODE (Open Dynamics Engine)
B) Bullet
C) DART
D) PhysX

<details>
<summary>Answer</summary>

**C) DART**

Gazebo Fortress uses DART (Dynamic Animation and Robotics Toolkit) as its default physics engine. DART provides accurate multi-body dynamics simulation suitable for robotics applications. Previous versions of Gazebo (Classic) used ODE by default.

</details>
:::

:::note Question 2
What is the primary advantage of using Unity over Gazebo for robotics simulation?

A) Better physics accuracy
B) Native ROS 2 support
C) Photorealistic rendering for perception training
D) Faster simulation speed

<details>
<summary>Answer</summary>

**C) Photorealistic rendering for perception training**

Unity excels at rendering high-quality, photorealistic images, making it ideal for generating synthetic training data for computer vision models. This includes features like ray tracing, HDR lighting, and domain randomization through the Perception package.

</details>
:::

:::note Question 3
In URDF, what element defines the physical properties used for dynamics simulation?

A) `<visual>`
B) `<collision>`
C) `<inertial>`
D) `<joint>`

<details>
<summary>Answer</summary>

**C) `<inertial>`**

The `<inertial>` element defines mass, center of mass (origin), and moment of inertia matrix - the physical properties needed for accurate dynamics simulation. `<visual>` is for rendering, `<collision>` is for collision detection, and `<joint>` defines connections between links.

</details>
:::

:::note Question 4
What is the Real-Time Factor (RTF) in simulation?

A) The frame rate of the simulation display
B) The ratio of simulated time to real wall clock time
C) The physics timestep size
D) The number of simulation iterations per second

<details>
<summary>Answer</summary>

**B) The ratio of simulated time to real wall clock time**

RTF = 1.0 means simulation runs at real-time speed. RTF > 1.0 means faster than real-time, RTF < 1.0 means slower. For RL training, RTF >> 1.0 is desirable. For testing with real hardware, RTF ≈ 1.0 is needed.

</details>
:::

### Full Quiz

Access the complete 15-20 question quiz through your course platform.

---

## Quiz 3: Isaac Platform (Weeks 5-6)

### Concepts Covered

- NVIDIA Omniverse and USD
- Isaac Sim environment creation
- Isaac ROS perception pipelines
- Isaac Lab RL training
- Domain randomization
- Policy deployment

### Sample Questions

:::note Question 1
What file format does NVIDIA Isaac Sim use for storing scenes and assets?

A) URDF
B) SDF
C) USD (Universal Scene Description)
D) FBX

<details>
<summary>Answer</summary>

**C) USD (Universal Scene Description)**

Isaac Sim is built on NVIDIA Omniverse, which uses USD as its native file format. USD was developed by Pixar and provides composition, instancing, and streaming capabilities ideal for large, complex scenes.

</details>
:::

:::note Question 2
What is the primary benefit of Isaac Lab's parallel simulation capability?

A) Higher rendering quality
B) More accurate physics
C) Orders of magnitude faster RL training
D) Better ROS 2 integration

<details>
<summary>Answer</summary>

**C) Orders of magnitude faster RL training**

Isaac Lab can run thousands of simulation environments in parallel on a single GPU, collecting training data much faster than sequential simulation. This enables training complex robot policies in hours instead of days.

</details>
:::

:::note Question 3
In Isaac Lab, what is domain randomization used for?

A) Generating diverse training environments
B) Reducing memory usage
C) Speeding up simulation
D) Improving rendering quality

<details>
<summary>Answer</summary>

**A) Generating diverse training environments**

Domain randomization varies simulation parameters (lighting, textures, physics properties, object positions) during training to help policies generalize to the real world (sim-to-real transfer). A policy trained on diverse simulated conditions is more likely to handle real-world variability.

</details>
:::

:::note Question 4
Which Isaac ROS package provides GPU-accelerated deep learning inference?

A) isaac_ros_common
B) isaac_ros_dnn_inference
C) isaac_ros_visual_slam
D) isaac_ros_apriltag

<details>
<summary>Answer</summary>

**B) isaac_ros_dnn_inference**

The isaac_ros_dnn_inference package provides TensorRT-accelerated inference for deep neural networks. It supports various model formats and enables real-time perception on NVIDIA GPUs.

</details>
:::

### Full Quiz

Access the complete quiz through your course platform.

---

## Quiz 4: VLA & Integration (Weeks 7-8)

### Concepts Covered

- Vision encoder architectures
- Language model integration
- Action representation and prediction
- Diffusion policies
- End-to-end VLA training
- ROS 2 deployment

### Sample Questions

:::note Question 1
What is the main advantage of Vision Transformers (ViT) over CNNs for VLA models?

A) Faster inference
B) Better handling of global context and patch relationships
C) Smaller model size
D) No pre-training required

<details>
<summary>Answer</summary>

**B) Better handling of global context and patch relationships**

Vision Transformers use self-attention to model relationships between all image patches simultaneously, capturing global context better than CNNs which have limited receptive fields. This is particularly valuable for robotics where understanding spatial relationships across the entire scene matters.

</details>
:::

:::note Question 2
What is action chunking in VLA models?

A) Compressing action data
B) Predicting multiple future actions at once
C) Discretizing continuous actions
D) Batching action execution

<details>
<summary>Answer</summary>

**B) Predicting multiple future actions at once**

Action chunking predicts a sequence of future actions (e.g., 16 timesteps) in one forward pass, rather than predicting single actions. This improves temporal consistency and reduces compounding errors from sequential predictions.

</details>
:::

:::note Question 3
Why are diffusion policies effective for robot action generation?

A) They are faster than other methods
B) They can model multi-modal action distributions
C) They require less training data
D) They don't need GPUs

<details>
<summary>Answer</summary>

**B) They can model multi-modal action distributions**

Diffusion policies can represent multiple valid solutions to a task (multi-modality). For example, there might be multiple ways to grasp an object - diffusion models can capture this distribution and sample diverse, valid actions, unlike single-point predictions.

</details>
:::

:::note Question 4
What is the sim-to-real gap, and how do VLA models address it?

A) The gap between simulation speed and real-time; addressed by faster inference
B) The gap between simulated and real sensor data; addressed by domain randomization
C) The gap between training and test data; addressed by more data
D) The gap between planning and execution; addressed by better planners

<details>
<summary>Answer</summary>

**B) The gap between simulated and real sensor data; addressed by domain randomization**

The sim-to-real gap refers to differences between simulation and reality that cause policies to fail when deployed. VLA models address this through domain randomization during training (varying lighting, textures, physics) and by using pre-trained vision encoders that generalize across domains.

</details>
:::

### Full Quiz

Access the complete final quiz through your course platform.

---

## Taking Quizzes

### Best Practices

1. **Attempt without resources first** - Get an honest baseline
2. **Review explanations** - Learn from every question
3. **Retake after review** - Aim for >90% on retakes
4. **Note weak areas** - Focus study on missed topics

### Scoring

| Score | Recommendation |
|-------|----------------|
| 90-100% | Ready to proceed |
| 70-89% | Review specific topics, then proceed |
| 50-69% | Significant review needed |
| under 50% | Re-study module before continuing |

### Using Results

Quiz results should guide your study:

- **Consistently wrong topic**: Re-read corresponding chapter section
- **Careless mistakes**: Slow down, read carefully
- **Conceptual gaps**: Seek additional explanations or ask questions

---

Return to [Assessments Overview](/docs/assessments) or explore [Projects](/docs/assessments/projects).
