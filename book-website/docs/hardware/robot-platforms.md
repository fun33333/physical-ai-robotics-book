---
title: "Robot Platforms Guide"
sidebar_position: 6
description: "Comprehensive guide to selecting appropriate robot platforms for Physical AI applications with technical specifications and use cases"
---

# Robot Platforms Guide

## Overview

Selecting the right robot platform is a critical decision that impacts your learning experience, project scope, and ability to implement Physical AI concepts. This guide will help you understand the different types of robot platforms available, their capabilities, limitations, and ideal use cases.

## Mobile Robot Platforms

Mobile robots form the foundation for learning navigation, mapping, SLAM, and path planning algorithms in Physical AI.

### Differential Drive Platforms

Differential drive robots use two independently controlled wheels to achieve motion and rotation, making them ideal for learning basic mobile robotics concepts.

#### Small-Scale Platforms

| Platform | Description | Cost | Dimensions | Payload | Max Speed | Sensors | Best For |
|----------|-------------|------|------------|---------|-----------|---------|----------|
| TurtleBot 4 Lite | ROS 2 ready educational robot | ~$1200 | 330x330x380mm | 5kg | 0.5m/s | RGB-D camera, IMU | Entry-level learning |
| Clearpath Jackal | All-terrain research platform | ~$4000 | 533x406x203mm | 25kg | 2m/s | Hokuyo LiDAR, Intel RealSense, IMU | Research projects |
| Dagu Wild Thumper | Rugged outdoor platform | ~$200 | 305x254x130mm | 10kg | 1m/s | User added | Outdoor exploration |
| Pololu Zumo | Arduino-based tracked robot | ~$100 | 98x98x30mm | 500g | 0.5m/s | User added | Beginner electronics |

#### Educational Platforms

| Platform | Description | Cost | Dimensions | Payload | Max Speed | Sensors | Best For |
|----------|-------------|------|------------|---------|-----------|---------|----------|
| TurtleBot 4 | Complete ROS 2 platform | ~$2500 | 330x330x380mm | 5kg | 0.5m/s | Intel RealSense D435, IMU, 2D LiDAR | Comprehensive learning |
| MiR100 | Professional service robot | ~$80000 | 810x590x780mm | 100kg | 1.2m/s | Multiple LiDARs, cameras, IMU | Advanced service robotics |
| Thymio II | Educational robot for all ages | ~$150 | 115x115x45mm | 500g | 0.3m/s | Infrared, accelerometer, microphone | Introductory robotics |

### Ackermann Drive Platforms

Ackermann drive robots use car-like steering, making them suitable for more complex navigation tasks.

| Platform | Description | Cost | Dimensions | Payload | Max Speed | Sensors | Best For |
|----------|-------------|------|------------|---------|-----------|---------|----------|
| Donkey Car | DIY self-driving platform | ~$200-500 | Custom | 1-2kg | 2m/s | Camera | Self-driving education |
| AWS DeepRacer | Cloud-trained autonomous car | ~$399 | 400x200x100mm | 1kg | 4m/s | Camera | Reinforcement learning |
| AutonomouStuff ASI-200 | Research-grade platform | ~$15000 | 1.8m long | 100kg | 15m/s | Multiple sensors | Advanced autonomous driving |

### Custom Platforms

Building your own robot platform allows for complete customization but requires more time and expertise:

- **Advantages**: Custom design for specific requirements, cost-effective for unique applications
- **Disadvantages**: Time-consuming, requires mechanical and electrical engineering skills
- **Best for**: Specific research applications, budget-conscious projects

## Manipulation Platforms

Manipulation robots enable learning of grasping, manipulation, and interaction with objects.

### Desktop Arms

| Platform | DOF | Reach (mm) | Payload (g) | Cost | Controllers | Best For |
|----------|-----|------------|-------------|------|-------------|----------|
| WidowX 250 | 5 | 250 | 500 | ~$1200 | ROS, MoveIt! | Academic research |
| WidowX 350 | 5 | 350 | 500 | ~$1500 | ROS, MoveIt! | Academic research |
| Interbotix Arms | 4-7 | 178-500 | 100-500 | ~$800-2000 | ROS, MoveIt! | Educational use |
| Kinova Gen3 | 7 | 850 | 500 | ~$30000 | ROS, Web interface | Advanced manipulation |
| Universal Robots UR3e | 6 | 500 | 3000 | ~$23000 | ROS, Web interface | Industrial applications |

### Mobile Manipulators

| Platform | Description | Cost | Mobile Base | Arm | Best For |
|----------|-------------|------|-------------|-----|----------|
| Fetch Robot | Professional mobile manipulator | ~$40000 | Wheeled base | 7-DOF arm | Research, service tasks |
| Toyota HSR | Research robot | ~$50000 | Omnidirectional base | 5-DOF arm | HRI research |
| Stretch RE1 | Cost-effective mobile manipulator | ~$18000 | Custom mobile base | 6-DOF arm | Service robotics |

### Grippers and End Effectors

The choice of gripper significantly impacts manipulation capabilities:

| Type | Description | Cost | Advantages | Disadvantages | Best For |
|------|-------------|------|------------|---------------|----------|
| Parallel Jaw | Simple 2-finger gripper | $50-500 | Reliable, precise | Limited object types | Rectangular objects |
| Adaptive | Underactuated fingers | $200-2000 | Grasps diverse shapes | Not precise | General purpose |
| Suction Cups | Vacuum-based grip | $100-1000 | Grasps flat objects | Only for specific materials | Manufacturing |
| Soft Actuators | Flexible grippers | $300-3000 | Grasps delicate objects | Complex control | Delicate objects |

## Humanoid Robot Platforms

Humanoid robots are complex platforms suitable for advanced research in locomotion, balance, and human-robot interaction.

| Platform | Description | Cost | DOF | Height (cm) | Sensors | Best For |
|----------|-------------|------|-----|-------------|---------|----------|
| NAO | Small humanoid robot | ~$9000 | 25 | 58 | Cameras, microphones, IMU | HRI, education |
| Pepper | Social humanoid robot | ~$20000 | 20 | 120 | 3D camera, microphones, touch sensors | Social robotics |
| Darwin OP | Research humanoid | ~$10000 | 20 | 46 | Camera, IMU, microphone | Research |
| Unitree H1 | Advanced humanoid | ~$100000 | 23 | 165 | LiDAR, cameras, IMU | Advanced locomotion |
| Boston Dynamics Atlas | State-of-the-art humanoid | ~$1000000 | 28+ | 173 | Cameras, LIDAR, IMU | Advanced research |

## Specialized Platforms

### Aerial Robots (Drones)

| Platform | Description | Cost | Flight Time | Sensors | Best For |
|----------|-------------|------|-------------|---------|----------|
| DJI Matrice 300 RTK | Industrial drone | ~$15000 | 55 min | Multiple cameras, LiDAR | Aerial inspection |
| PX4-based drone | Open-source drone | ~$2000-5000 | 15-45 min | Custom | Custom applications |
| DJI Tello | Educational drone | ~$100 | 13 min | Camera | Learning drone programming |

### Underwater Robots (ROVs/AUVs)

| Platform | Description | Cost | Depth Rating | Sensors | Best For |
|----------|-------------|------|--------------|---------|----------|
| BlueROV2 | Open-source ROV | ~$4000 | 100m | Cameras, sonar, IMU | Underwater inspection |
| OpenROV Trident | Educational ROV | ~$1800 | 100m | HD camera | Underwater exploration |

## Platform Selection Criteria

:::warning Budget Considerations
Always account for the total cost of ownership, not just the initial purchase price. Consider ongoing costs like maintenance, upgrades, and potential replacement parts when selecting a robot platform.
:::

### Application Requirements

#### Research vs. Educational Use

**Research Platforms:**
- Require high precision and repeatability
- Need extensive sensor integration capabilities
- Support advanced control algorithms
- Examples: Fetch, Kinova Gen3, Clearpath platforms

**Educational Platforms:**
- Emphasize ease of use and learning
- Include extensive documentation and tutorials
- Support standard frameworks (ROS, MoveIt!)
- Examples: TurtleBot, Thymio, Interbotix arms

#### Indoor vs. Outdoor Operation

**Indoor Platforms:**
- Focus on precision and obstacle avoidance
- Operate in controlled environments
- May prioritize sensors over mobility hardware
- Examples: TurtleBot, Fetch, PR2

**Outdoor Platforms:**
- Require robust construction and weather resistance
- Need GPS and outdoor navigation capabilities
- Higher power requirements for all-terrain mobility
- Examples: Clearpath Husky, Jackal, AWS DeepRacer

### Budget Considerations

#### Tier 1: Budget-Friendly (under $1000)
- **Mobile**: Pololu Zumo, LEGO Mindstorms, custom Arduino robots
- **Manipulation**: OWI-535 robotic arm, UGS-25 manipulator arm
- **Sensors**: Basic RGB cameras, ultrasonic sensors, IMU modules
- **Best for**: Introduction to robotics concepts

#### Tier 2: Mid-Range ($1000-$10000)
- **Mobile**: TurtleBot 4 Lite, Dagu platforms
- **Manipulation**: WidowX arms, Interbotix arms
- **Sensors**: Intel RealSense, basic LiDARs
- **Best for**: College-level robotics courses

#### Tier 3: High-End ($10000+)
- **Mobile**: Clearpath Jackal, Fetch, MiR platforms
- **Manipulation**: Kinova Gen3, Franka Emika Panda
- **Sensors**: High-end LiDARs, thermal cameras
- **Best for**: Advanced research projects

### Technical Capabilities

#### Degree of Freedom (DOF) Requirements

- **Low DOF (1-3)**: Simple tasks, single joint applications
- **Medium DOF (4-6)**: Basic manipulation, reaching tasks
- **High DOF (7+)**: Human-like manipulation, complex tasks

#### Computational Requirements

- **Lightweight**: Basic controllers, single-board computers
- **Medium**: Edge AI chips (Jetson, Coral), standard PCs
- **High**: High-end GPUs, cloud computation, specialized hardware

#### Payload Requirements

- **Light (under 1kg)**: Small manipulators, mobile platforms
- **Medium (1-10kg)**: Mid-size manipulators, mobile manipulators
- **Heavy (10kg+)**: Industrial arms, large platforms

## Integration Considerations

### Software Compatibility

#### ROS Support
- Ensure the platform has ROS/ROS2 support
- Check for active maintenance and updates
- Verify availability of simulation models (Gazebo)

#### API and Documentation
- Comprehensive APIs for all functions
- Well-documented examples and tutorials
- Active community support

### Hardware Integration

#### Sensor Mounting
- Built-in mounting points
- Expandable sensor integration
- Cable management solutions

#### Power Management
- Built-in battery management
- Power distribution systems
- External power options

#### Communication
- Multiple interface options (USB, Ethernet, WiFi)
- Real-time communication capabilities
- Remote operation support

## DIY vs. Off-the-Shelf Platforms

### Advantages of Off-the-Shelf Platforms

- **Proven Reliability**: Tested and validated designs
- **Comprehensive Support**: Documentation, tutorials, community
- **Warranty**: Manufacturer support and replacement
- **Time-Saving**: Immediate use without construction

### Advantages of DIY Platforms

- **Customization**: Tailored to specific requirements
- **Cost-Effective**: Potentially lower cost for unique applications
- **Learning**: Deep understanding of robot systems
- **Flexibility**: Easy modification and extension

### Hybrid Approach

Consider building on top of existing platforms:
- Start with proven base platform
- Add custom sensors or end effectors
- Modify software for specific applications

## Platform Maintenance and Support

### Maintenance Requirements

- Regular calibration of sensors and actuators
- Battery replacement and care
- Mechanical wear inspection
- Software updates and security patches

### Support Resources

- Manufacturer documentation and forums
- Community support and tutorials
- Academic papers and implementations
- Third-party accessories and add-ons

## Testing and Validation

### Initial Setup Testing
1. Verify all hardware components function correctly
2. Test all sensors and actuators individually
3. Validate communication and control systems
4. Confirm safety mechanisms work properly

### Performance Validation
- Test navigation accuracy and reliability
- Validate manipulation precision
- Verify system stability under load
- Confirm safety features in emergency situations

---

:::tip Exercise 5: Robot Platform Selection
**Objective**: Select an appropriate robot platform for a specific application in Physical AI

**Time Estimate**: 60 minutes

**Steps**:
1. Define a specific application scenario (e.g., warehouse navigation, object manipulation, human-robot interaction)
2. List the technical requirements for your application
3. Research 3-5 robot platforms that could potentially meet these requirements
4. Compare the platforms based on cost, capabilities, and ease of use
5. Make your selection with justification for why it's the best fit
6. Outline a basic integration plan for your application

**Expected Result**: A detailed platform selection report with comparison tables, technical analysis, and implementation plan

**Hints**:
- Consider both current needs and future expandability
- Factor in development timeline and expertise level
- Account for the total cost of ownership, not just initial purchase
:::