---
title: "Gazebo Basics"
sidebar_position: 3
description: "Getting started with Gazebo: installation, world creation, robot modeling with SDF/URDF."
---

# Gazebo Basics

Gazebo has been the standard robotics simulator for over a decade, and the new Gazebo (formerly Ignition Gazebo) brings modern architecture, improved rendering, and better scalability. This chapter covers everything you need to start building simulations in Gazebo.

## Overview

In this section, you will:

- Install and configure Gazebo Harmonic for ROS 2 Humble
- Understand the Gazebo architecture and GUI
- Create simulation worlds using SDF (Simulation Description Format)
- Build robot models with URDF and convert to SDF
- Add sensors including cameras, LiDAR, and IMUs
- Use Gazebo plugins to extend functionality

## Prerequisites

Before starting, ensure you have:

- [Module 1: ROS 2 Fundamentals](/docs/module-1) completed
- Ubuntu 22.04 (native or WSL2)
- At least 8GB RAM and 20GB disk space
- GPU recommended but not required

---

## Installing Gazebo

### Gazebo Harmonic (Recommended for ROS 2 Humble)

Gazebo Harmonic is the latest LTS release compatible with ROS 2 Humble:

```bash title="Install Gazebo Harmonic"
# Add Gazebo repository
sudo wget https://packages.osrfoundation.org/gazebo.gpg -O /usr/share/keyrings/pkgs-osrf-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/pkgs-osrf-archive-keyring.gpg] http://packages.osrfoundation.org/gazebo/ubuntu-stable $(lsb_release -cs) main" | sudo tee /etc/apt/sources.list.d/gazebo-stable.list > /dev/null

# Install Gazebo Harmonic
sudo apt update
sudo apt install gz-harmonic

# Install ROS 2 integration packages
sudo apt install ros-humble-ros-gz
```

**Verify installation:**

```bash
gz sim --version
# Expected: Gazebo Sim, version 8.x.x
```

### First Launch

Launch Gazebo with a sample world:

```bash
gz sim shapes.sdf
```

You should see a world with basic geometric shapes (box, sphere, cylinder).

---

## Gazebo Architecture

### Core Components

```
┌─────────────────────────────────────────────────────────────────┐
│                      Gazebo Architecture                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   ┌──────────────┐    ┌──────────────┐    ┌──────────────┐     │
│   │   Server     │    │   GUI        │    │   Plugins    │     │
│   │  (Physics)   │◄──►│  (Rendering) │◄──►│  (Custom)    │     │
│   └──────────────┘    └──────────────┘    └──────────────┘     │
│          │                   │                   │              │
│          └───────────────────┴───────────────────┘              │
│                              │                                   │
│                     ┌────────▼────────┐                         │
│                     │   gz Transport   │                         │
│                     │   (Messaging)    │                         │
│                     └─────────────────┘                         │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

| Component | Purpose |
|-----------|---------|
| **Server** | Runs physics simulation, handles collisions |
| **GUI** | Renders 3D scene, provides user interface |
| **Plugins** | Extend functionality (sensors, controllers) |
| **Transport** | Internal messaging between components |

### Headless Mode

For training and CI/CD, run without GUI:

```bash
# Headless simulation (no rendering)
gz sim -s shapes.sdf

# With specific physics step size
gz sim -s --physics-rate=1000 shapes.sdf
```

---

## Simulation Description Format (SDF)

SDF is Gazebo's native format for describing worlds and models. It's more expressive than URDF and supports complete simulation configuration.

### World Structure

```xml title="my_world.sdf"
<?xml version="1.0" ?>
<sdf version="1.9">
  <world name="my_world">
    <!-- Physics configuration -->
    <physics type="ode">
      <max_step_size>0.001</max_step_size>
      <real_time_factor>1.0</real_time_factor>
    </physics>

    <!-- Lighting -->
    <light type="directional" name="sun">
      <cast_shadows>true</cast_shadows>
      <pose>0 0 10 0 0 0</pose>
      <diffuse>0.8 0.8 0.8 1</diffuse>
      <specular>0.2 0.2 0.2 1</specular>
      <direction>-0.5 0.1 -0.9</direction>
    </light>

    <!-- Ground plane -->
    <model name="ground_plane">
      <static>true</static>
      <link name="link">
        <collision name="collision">
          <geometry>
            <plane>
              <normal>0 0 1</normal>
              <size>100 100</size>
            </plane>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <plane>
              <normal>0 0 1</normal>
              <size>100 100</size>
            </plane>
          </geometry>
          <material>
            <ambient>0.8 0.8 0.8 1</ambient>
          </material>
        </visual>
      </link>
    </model>

    <!-- Include a robot model -->
    <include>
      <uri>model://my_robot</uri>
      <pose>0 0 0.5 0 0 0</pose>
    </include>

  </world>
</sdf>
```

### Key SDF Elements

| Element | Purpose | Example |
|---------|---------|---------|
| `<world>` | Top-level container | Contains all models, lights, physics |
| `<model>` | A simulated object | Robot, obstacle, sensor |
| `<link>` | Rigid body with mass | Base link, wheel, arm |
| `<joint>` | Connection between links | Revolute, prismatic, fixed |
| `<collision>` | Physics collision geometry | Simplified shapes |
| `<visual>` | Rendered appearance | Detailed meshes |
| `<sensor>` | Simulated sensor | Camera, LiDAR, IMU |
| `<plugin>` | Custom behavior | Controllers, bridges |

---

## Creating Robot Models

### URDF for ROS 2 Robots

While Gazebo uses SDF, most ROS 2 robots are defined in URDF. Gazebo automatically converts URDF to SDF.

```xml title="simple_robot.urdf"
<?xml version="1.0"?>
<robot name="simple_robot">
  <!-- Base link -->
  <link name="base_link">
    <visual>
      <geometry>
        <box size="0.5 0.3 0.1"/>
      </geometry>
      <material name="blue">
        <color rgba="0 0 0.8 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <box size="0.5 0.3 0.1"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="5.0"/>
      <inertia ixx="0.05" ixy="0" ixz="0"
               iyy="0.1" iyz="0" izz="0.1"/>
    </inertial>
  </link>

  <!-- Left wheel -->
  <link name="left_wheel">
    <visual>
      <geometry>
        <cylinder radius="0.1" length="0.05"/>
      </geometry>
      <material name="black">
        <color rgba="0.1 0.1 0.1 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <cylinder radius="0.1" length="0.05"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.5"/>
      <inertia ixx="0.001" ixy="0" ixz="0"
               iyy="0.001" iyz="0" izz="0.002"/>
    </inertial>
  </link>

  <!-- Left wheel joint -->
  <joint name="left_wheel_joint" type="continuous">
    <parent link="base_link"/>
    <child link="left_wheel"/>
    <origin xyz="0 0.175 0" rpy="-1.5707 0 0"/>
    <axis xyz="0 0 1"/>
  </joint>

  <!-- Right wheel (similar structure) -->
  <link name="right_wheel">
    <visual>
      <geometry>
        <cylinder radius="0.1" length="0.05"/>
      </geometry>
      <material name="black">
        <color rgba="0.1 0.1 0.1 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <cylinder radius="0.1" length="0.05"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.5"/>
      <inertia ixx="0.001" ixy="0" ixz="0"
               iyy="0.001" iyz="0" izz="0.002"/>
    </inertial>
  </link>

  <joint name="right_wheel_joint" type="continuous">
    <parent link="base_link"/>
    <child link="right_wheel"/>
    <origin xyz="0 -0.175 0" rpy="-1.5707 0 0"/>
    <axis xyz="0 0 1"/>
  </joint>

  <!-- Caster wheel -->
  <link name="caster">
    <visual>
      <geometry>
        <sphere radius="0.05"/>
      </geometry>
    </visual>
    <collision>
      <geometry>
        <sphere radius="0.05"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.1"/>
      <inertia ixx="0.0001" ixy="0" ixz="0"
               iyy="0.0001" iyz="0" izz="0.0001"/>
    </inertial>
  </link>

  <joint name="caster_joint" type="fixed">
    <parent link="base_link"/>
    <child link="caster"/>
    <origin xyz="-0.2 0 -0.05"/>
  </joint>
</robot>
```

### Converting URDF to SDF

```bash
# Convert URDF to SDF for inspection
gz sdf -p simple_robot.urdf > simple_robot.sdf
```

### Visualizing URDF in RViz2

Before simulation, verify your URDF in RViz2:

```bash
# Install visualization tools
sudo apt install ros-humble-urdf-tutorial

# View URDF
ros2 launch urdf_tutorial display.launch.py model:=simple_robot.urdf
```

---

## Adding Sensors

### Camera Sensor

```xml title="Adding a camera to URDF"
<!-- Camera link -->
<link name="camera_link">
  <visual>
    <geometry>
      <box size="0.02 0.05 0.02"/>
    </geometry>
  </visual>
  <inertial>
    <mass value="0.01"/>
    <inertia ixx="0.00001" ixy="0" ixz="0"
             iyy="0.00001" iyz="0" izz="0.00001"/>
  </inertial>
</link>

<joint name="camera_joint" type="fixed">
  <parent link="base_link"/>
  <child link="camera_link"/>
  <origin xyz="0.25 0 0.05" rpy="0 0 0"/>
</joint>

<!-- Gazebo camera sensor -->
<gazebo reference="camera_link">
  <sensor name="camera" type="camera">
    <always_on>true</always_on>
    <update_rate>30</update_rate>
    <camera>
      <horizontal_fov>1.047</horizontal_fov>
      <image>
        <width>640</width>
        <height>480</height>
        <format>R8G8B8</format>
      </image>
      <clip>
        <near>0.1</near>
        <far>100</far>
      </clip>
    </camera>
  </sensor>
</gazebo>
```

### LiDAR Sensor

```xml title="Adding LiDAR to URDF"
<link name="lidar_link">
  <visual>
    <geometry>
      <cylinder radius="0.05" length="0.04"/>
    </geometry>
    <material name="dark_grey">
      <color rgba="0.2 0.2 0.2 1"/>
    </material>
  </visual>
  <inertial>
    <mass value="0.2"/>
    <inertia ixx="0.0001" ixy="0" ixz="0"
             iyy="0.0001" iyz="0" izz="0.0001"/>
  </inertial>
</link>

<joint name="lidar_joint" type="fixed">
  <parent link="base_link"/>
  <child link="lidar_link"/>
  <origin xyz="0 0 0.1" rpy="0 0 0"/>
</joint>

<gazebo reference="lidar_link">
  <sensor name="lidar" type="gpu_lidar">
    <always_on>true</always_on>
    <update_rate>10</update_rate>
    <lidar>
      <scan>
        <horizontal>
          <samples>360</samples>
          <resolution>1</resolution>
          <min_angle>-3.14159</min_angle>
          <max_angle>3.14159</max_angle>
        </horizontal>
      </scan>
      <range>
        <min>0.1</min>
        <max>30.0</max>
        <resolution>0.01</resolution>
      </range>
    </lidar>
  </sensor>
</gazebo>
```

### IMU Sensor

```xml title="Adding IMU to URDF"
<gazebo reference="base_link">
  <sensor name="imu" type="imu">
    <always_on>true</always_on>
    <update_rate>100</update_rate>
    <imu>
      <angular_velocity>
        <x>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>0.01</stddev>
          </noise>
        </x>
        <y>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>0.01</stddev>
          </noise>
        </y>
        <z>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>0.01</stddev>
          </noise>
        </z>
      </angular_velocity>
      <linear_acceleration>
        <x>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>0.1</stddev>
          </noise>
        </x>
        <y>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>0.1</stddev>
          </noise>
        </y>
        <z>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>0.1</stddev>
          </noise>
        </z>
      </linear_acceleration>
    </imu>
  </sensor>
</gazebo>
```

---

## Gazebo Plugins

Plugins extend Gazebo functionality. Here are common plugin patterns:

### Differential Drive Plugin

```xml title="Add differential drive control"
<gazebo>
  <plugin filename="gz-sim-diff-drive-system"
          name="gz::sim::systems::DiffDrive">
    <left_joint>left_wheel_joint</left_joint>
    <right_joint>right_wheel_joint</right_joint>
    <wheel_separation>0.35</wheel_separation>
    <wheel_radius>0.1</wheel_radius>
    <odom_publish_frequency>50</odom_publish_frequency>
    <topic>cmd_vel</topic>
  </plugin>
</gazebo>
```

### Joint State Publisher

```xml title="Publish joint states"
<gazebo>
  <plugin filename="gz-sim-joint-state-publisher-system"
          name="gz::sim::systems::JointStatePublisher">
    <joint_name>left_wheel_joint</joint_name>
    <joint_name>right_wheel_joint</joint_name>
  </plugin>
</gazebo>
```

---

## Exercise 1: Build Your First Gazebo World

:::tip Exercise 1: Create a Simple Warehouse World
**Objective**: Build a Gazebo world with obstacles for robot navigation testing.

**Time Estimate**: 45 minutes

**Steps**:

1. Create a new SDF file named `warehouse.sdf`
2. Add a ground plane with gray material
3. Add directional lighting (sun)
4. Add at least 4 box obstacles of different sizes
5. Add at least 2 cylinder obstacles (representing pillars)
6. Save and launch with `gz sim warehouse.sdf`

**Starter Template**:

```xml
<?xml version="1.0" ?>
<sdf version="1.9">
  <world name="warehouse">
    <!-- Add physics configuration -->

    <!-- Add lighting -->

    <!-- Add ground plane -->

    <!-- Add obstacle 1: Large box (2m x 1m x 1m) at position (3, 2, 0.5) -->

    <!-- Add more obstacles... -->

  </world>
</sdf>
```

**Expected Result**: A Gazebo window showing a warehouse-like environment with scattered obstacles suitable for navigation testing.

**Success Criteria**:
- World loads without errors
- Ground plane is visible
- All obstacles are positioned correctly
- Scene is properly lit
:::

---

## Exercise 2: Create a Two-Wheeled Robot

:::tip Exercise 2: Build a Differential Drive Robot
**Objective**: Create a complete robot model with differential drive.

**Time Estimate**: 60 minutes

**Steps**:

1. Create a URDF file with base_link, two wheels, and a caster
2. Add proper inertial properties to all links
3. Add a differential drive Gazebo plugin
4. Spawn the robot in an empty world
5. Control the robot using `gz topic -t /cmd_vel`

**Testing Command**:

```bash
# Publish velocity command
gz topic -t /cmd_vel -m gz.msgs.Twist -p 'linear: {x: 0.5}, angular: {z: 0.3}'
```

**Expected Result**: Robot moves forward while turning, demonstrating successful differential drive control.
:::

---

## Troubleshooting

### Common Issues

| Problem | Cause | Solution |
|---------|-------|----------|
| Robot falls through floor | Missing collision geometry | Add `<collision>` to ground |
| Robot explodes on spawn | Overlapping collisions | Adjust spawn position |
| Wheels don't turn | Joint axis incorrect | Check `<axis>` direction |
| No sensor data | Plugin not loaded | Verify plugin filename |
| Slow simulation | Step size too small | Increase `max_step_size` |

### Debugging Commands

```bash
# List all topics
gz topic -l

# Echo a topic
gz topic -e -t /world/default/model/robot/joint_state

# Get model info
gz model -m my_robot --pose

# Check simulation stats
gz stats
```

---

## Summary

In this chapter, you learned:

- **Installation**: Set up Gazebo Harmonic with ROS 2 Humble
- **Architecture**: Server, GUI, plugins, and transport system
- **SDF Worlds**: Create environments with physics, lighting, and obstacles
- **Robot Models**: Build URDF robots with proper inertial properties
- **Sensors**: Add cameras, LiDAR, and IMU with noise models
- **Plugins**: Use differential drive and joint state publishers

These fundamentals prepare you for the next chapter: [Gazebo-ROS 2 Integration](/docs/module-2/gazebo-ros2-integration), where you'll connect your simulations to ROS 2.

## Further Reading

- [Gazebo Harmonic Documentation](https://gazebosim.org/docs/harmonic)
- [SDF Format Specification](http://sdformat.org/spec)
- [URDF Tutorials](http://wiki.ros.org/urdf/Tutorials)
- [Gazebo Sim Plugins](https://gazebosim.org/api/sim/8/namespacegz_1_1sim_1_1systems.html)
