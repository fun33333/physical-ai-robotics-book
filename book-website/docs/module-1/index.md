---
title: "Module 1: The Robotic Nervous System (ROS 2)"
sidebar_position: 1
description: "Master ROS 2, the de facto standard middleware for building modular, distributed robotic systems."
---

# Module 1: The Robotic Nervous System (ROS 2)

Robot Operating System 2 (ROS 2) serves as the nervous system of modern robotic applications, providing the communication infrastructure that connects sensors, actuators, and AI components into a cohesive system. This module provides a comprehensive introduction to ROS 2 architecture and development.

## Module Overview

ROS 2 is not actually an operating system - it is a middleware framework that provides:

- **Communication infrastructure** for distributed robotic systems
- **Hardware abstraction** that works across different robot platforms
- **Software ecosystem** with thousands of reusable packages
- **Development tools** for debugging, visualization, and simulation

:::info Why ROS 2?
Over 80% of robotics companies use ROS. Learning ROS 2 is essential for any career in robotics, whether at startups, established companies, or research labs. It is the common language that robotics engineers speak.
:::

## Learning Objectives

By the end of this module, you will be able to:

1. **Understand ROS 2 architecture** - Nodes, executors, and the DDS middleware
2. **Implement communication patterns** - Topics for streaming, services for requests, actions for long tasks
3. **Manage node lifecycles** - Build robust systems with proper startup/shutdown behavior
4. **Use ROS 2 tools** - CLI commands, visualization (rviz2), introspection, and logging
5. **Configure navigation** - Set up Nav2 for autonomous mobile robots
6. **Plan robot motions** - Use MoveIt 2 for manipulation tasks

## Prerequisites

Before starting this module, ensure you have:

- **Python experience**: Comfortable with functions, classes, and callbacks
- **Linux basics**: Terminal navigation, file operations, environment variables
- **Development setup**: Ubuntu 22.04 (or WSL2) with ROS 2 Humble installed

:::tip Installation Help
See the [Hardware Guide](/docs/hardware/software-requirements) for detailed installation instructions. ROS 2 Humble is the recommended LTS version.
:::

## Module Structure

This module is divided into five sections, each building on the previous:

| Section | Focus | Duration |
|---------|-------|----------|
| [ROS 2 Fundamentals](/docs/module-1/ros2-fundamentals) | Installation, architecture, first nodes | 3-4 hours |
| [Nodes, Topics, Services](/docs/module-1/ros2-nodes-topics-services) | Core communication patterns | 4-5 hours |
| [Actions and Lifecycle](/docs/module-1/ros2-actions-lifecycle) | Long-running tasks, managed nodes | 3-4 hours |
| [Navigation Stack](/docs/module-1/ros2-navigation) | Autonomous navigation with Nav2 | 4-5 hours |
| [Manipulation](/docs/module-1/ros2-manipulation) | Robot arm control with MoveIt 2 | 4-5 hours |

**Total estimated time**: 18-23 hours over 2 weeks

## Key Concepts Preview

### Nodes: The Building Blocks

ROS 2 applications are composed of nodes - independent processes that perform specific functions:

- **Sensor nodes** read data from cameras, LiDAR, IMUs
- **Processing nodes** filter, transform, and analyze data
- **Planning nodes** decide what actions to take
- **Control nodes** send commands to motors and actuators

Nodes communicate through well-defined interfaces, making systems modular and testable.

### Communication Patterns

ROS 2 provides three primary communication mechanisms:

| Pattern | Use Case | Example |
|---------|----------|---------|
| **Topics** | Continuous data streams | Camera images, sensor readings |
| **Services** | Request-response queries | Get robot state, trigger calibration |
| **Actions** | Long-running tasks with feedback | Navigate to goal, pick up object |

### The ROS 2 Graph

All nodes and their connections form the ROS 2 graph:

```
[Camera Node] --/image--> [Perception Node] --/objects--> [Planning Node]
                                                              |
[Motor Node] <--/cmd_vel-- [Control Node] <--/goal----------+
```

You will learn to design, implement, and debug these graphs throughout this module.

## Development Environment

### Recommended Setup

- **IDE**: VS Code with ROS extension
- **Terminal**: Multiple terminals for running nodes
- **Visualization**: rviz2 for 3D visualization, rqt for GUI tools

### Workspace Organization

ROS 2 uses workspaces to organize code:

```
ros2_ws/
  src/
    my_robot_package/
      my_robot_package/
        __init__.py
        my_node.py
      resource/
      test/
      package.xml
      setup.py
  build/
  install/
  log/
```

You will create this structure in the first section.

## Hands-On Projects

Throughout this module, you will build:

1. **Publisher/Subscriber System** - Stream sensor data between nodes
2. **Service-Based Robot Controller** - Query and command robot state
3. **Action-Based Navigator** - Send goals with progress feedback
4. **Autonomous Mobile Robot** - Full Nav2 integration in simulation
5. **Robot Arm Controller** - Basic manipulation with MoveIt 2

Each project includes starter code, step-by-step instructions, and expected outputs.

## Industry Context

### Where ROS 2 is Used

- **Warehouse robots**: Amazon, Locus Robotics, 6 River Systems
- **Autonomous vehicles**: Autoware, Apollo ecosystem integrations
- **Research platforms**: TurtleBot, PR2, Fetch, many custom platforms
- **Industrial automation**: Integration with PLC systems, factory robots
- **Service robots**: Hospital delivery, retail assistance, cleaning

### Career Relevance

ROS 2 skills are explicitly requested in job postings for:

- Robotics Software Engineer
- Perception Engineer
- Motion Planning Engineer
- Systems Integration Engineer
- Field Applications Engineer

## Assessment

### Module Exercises

Each section contains hands-on exercises:

- **Guided exercises**: Step-by-step with expected output
- **Challenge exercises**: Apply concepts to new scenarios
- **Debugging exercises**: Fix broken code to reinforce understanding

### Module Project

The module culminates in a project that demonstrates:

- Proper ROS 2 package structure
- Multiple communicating nodes
- At least two communication patterns
- Integration with simulation (covered in Module 2)

See [Projects](/docs/assessments/projects) for detailed requirements.

## Getting Started

Ready to begin? Start with [ROS 2 Fundamentals](/docs/module-1/ros2-fundamentals) to set up your environment and create your first ROS 2 nodes.

:::note Pacing
If you are following the 8-week curriculum, this module spans Weeks 1-2. See the [Weekly Breakdown](/docs/weekly-breakdown/week-01-02) for day-by-day guidance.
:::
