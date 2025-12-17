---
title: "Hands-on Projects"
sidebar_position: 2
description: "Detailed descriptions of hands-on projects that demonstrate practical Physical AI skills."
---

# Hands-on Projects

Projects are the heart of assessment in this curriculum. Each module culminates in a practical project that demonstrates mastery of the covered material and produces artifacts you can include in your portfolio.

## Project Overview

| Project | Module | Focus | Difficulty |
|---------|--------|-------|------------|
| [ROS 2 Robot Controller](#project-1-ros-2-robot-controller) | Module 1 | Communication patterns, system integration | Intermediate |
| [Simulation Environment](#project-2-simulation-environment) | Module 2 | Gazebo, URDF, sensor simulation | Intermediate |
| [Isaac Perception Pipeline](#project-3-isaac-perception-pipeline) | Module 3 | GPU-accelerated perception | Advanced |
| [VLA Demo System](#project-4-vla-demo-system) | Module 4 | Vision-language-action integration | Advanced |
| [Capstone: Autonomous Assistant](#capstone-project) | All | Full integration | Expert |

---

## Project 1: ROS 2 Robot Controller

**Module**: 1 - The Robotic Nervous System (ROS 2)

### Overview

Build a complete ROS 2 application that demonstrates mastery of core communication patterns. You will create a multi-node system that simulates a robot with sensors, a controller, and a mission system that coordinates autonomous behavior.

### Learning Objectives

By completing this project, you will demonstrate:

- Proper ROS 2 package structure and build configuration
- Implementation of all three communication patterns (topics, services, actions)
- Lifecycle node management for controlled startup/shutdown
- System integration across multiple communicating nodes
- Use of ROS 2 tools for debugging and introspection

### System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     Robot Controller System                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   ┌──────────────┐        Topics         ┌──────────────┐       │
│   │   Sensor     │ ───────────────────▶  │  Controller  │       │
│   │   Nodes      │  /sensors/battery     │     Node     │       │
│   │              │  /sensors/position    │              │       │
│   └──────────────┘                       └──────┬───────┘       │
│                                                 │               │
│                                          Services/Actions       │
│                                                 │               │
│   ┌──────────────┐        Actions        ┌──────▼───────┐       │
│   │   Mission    │ ◀─────────────────── │    Robot     │       │
│   │   Planner    │  /navigate_to_goal   │   State      │       │
│   │              │  (feedback/result)    │   Server     │       │
│   └──────────────┘                       └──────────────┘       │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Requirements

#### Functional Requirements

1. **Sensor Simulation Node** (Lifecycle Node)
   - Publish simulated battery level (0-100%) on `/sensors/battery`
   - Publish simulated position (x, y, theta) on `/sensors/position`
   - Support lifecycle transitions (configure, activate, deactivate)
   - Battery should decrease over time when active

2. **Robot State Server Node**
   - Provide `/robot/get_state` service returning current robot state
   - Provide `/robot/set_mode` service to change operating mode (idle, patrol, charge)
   - Maintain internal state machine for mode transitions

3. **Navigation Action Server**
   - Accept `/navigate_to_goal` action with target position
   - Publish feedback with current position and distance remaining
   - Support goal cancellation
   - Simulate movement at configurable speed

4. **Mission Controller Node**
   - Subscribe to sensor topics
   - Monitor battery and trigger charging mode when low (< 20%)
   - Execute patrol mission: visit 3 waypoints in sequence
   - Use action client to send navigation goals
   - Log mission progress

#### Technical Requirements

- All nodes in a single package named `robot_controller`
- Use Python (rclpy) for implementation
- Include launch file that starts all nodes
- Include parameter file for configurable values
- Pass `colcon build` and `colcon test` without errors

### Deliverables

- [ ] ROS 2 package `robot_controller` with proper structure
- [ ] Sensor simulation node with lifecycle support
- [ ] Robot state server with services
- [ ] Navigation action server
- [ ] Mission controller node
- [ ] Launch file (`robot_system.launch.py`)
- [ ] Parameter file (`robot_params.yaml`)
- [ ] README with architecture diagram and usage instructions

### Rubric

| Criterion | Points | Description |
|-----------|--------|-------------|
| **Sensor Node** | 15 | Lifecycle-managed, publishes battery and position |
| **State Server** | 15 | Services work correctly, state machine logic |
| **Action Server** | 20 | Navigation with feedback, cancellation support |
| **Mission Controller** | 20 | Coordinates system, handles battery warnings |
| **Code Quality** | 15 | Clean code, comments, follows ROS 2 conventions |
| **Documentation** | 10 | README, architecture diagram, usage examples |
| **Bonus Features** | +10 | Additional features beyond requirements |

**Total**: 100 points (110 with bonus)
**Passing**: 70 points

### Suggested Approach

1. **Week 1**: Create package structure, implement sensor node
2. **Week 1**: Add state server with services
3. **Week 2**: Implement action server
4. **Week 2**: Build mission controller, integrate system

### Extension Challenges

- Add a second robot and implement multi-robot coordination
- Implement obstacle detection that pauses navigation
- Add RViz visualization markers for robot position
- Create a GUI using rqt for mission control
- Add unit tests for all nodes

### Starter Code

```python title="robot_controller/sensor_node.py"
import rclpy
from rclpy.lifecycle import Node as LifecycleNode
from rclpy.lifecycle import State, TransitionCallbackReturn
from std_msgs.msg import Float32
from geometry_msgs.msg import Pose2D


class SensorNode(LifecycleNode):
    """Simulated sensor node with lifecycle management."""

    def __init__(self):
        super().__init__('sensor_node')
        # TODO: Initialize publishers (but don't create until configure)
        self.battery_pub = None
        self.position_pub = None
        self.timer = None

    def on_configure(self, state: State) -> TransitionCallbackReturn:
        # TODO: Create publishers, declare parameters
        self.get_logger().info('Configuring...')
        return TransitionCallbackReturn.SUCCESS

    def on_activate(self, state: State) -> TransitionCallbackReturn:
        # TODO: Create timer, start publishing
        self.get_logger().info('Activating...')
        return TransitionCallbackReturn.SUCCESS

    def on_deactivate(self, state: State) -> TransitionCallbackReturn:
        # TODO: Stop timer
        self.get_logger().info('Deactivating...')
        return TransitionCallbackReturn.SUCCESS

    def publish_sensors(self):
        # TODO: Publish battery and position
        pass


def main(args=None):
    rclpy.init(args=args)
    node = SensorNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
```

---

## Project 2: Simulation Environment

**Module**: 2 - Digital Twins and Simulation

### Overview

Create a complete simulation environment for a mobile robot in Gazebo, including URDF robot description, sensor simulation, and ROS 2 integration. The environment should enable testing of navigation and perception algorithms without physical hardware.

### Learning Objectives

By completing this project, you will demonstrate:

- Creating valid URDF robot descriptions with proper kinematics and dynamics
- Configuring Gazebo sensors (camera, LiDAR, IMU) with realistic noise models
- Setting up ROS 2-Gazebo bridges for sensor data and control
- Building custom Gazebo worlds with static and dynamic obstacles
- Implementing hardware abstraction for simulation/real portability

### System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                   Simulation Environment System                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   ┌──────────────────────────────────────────────────────┐      │
│   │                    Gazebo Sim                         │      │
│   │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  │      │
│   │  │   Robot     │  │   World     │  │   Physics   │  │      │
│   │  │   Model     │  │   (SDF)     │  │   Engine    │  │      │
│   │  └─────────────┘  └─────────────┘  └─────────────┘  │      │
│   └──────────────────────────────────────────────────────┘      │
│                              │                                   │
│                       ros_gz_bridge                              │
│                              │                                   │
│   ┌──────────────────────────────────────────────────────┐      │
│   │                      ROS 2                            │      │
│   │  /camera/image    /scan    /imu    /cmd_vel    /odom │      │
│   └──────────────────────────────────────────────────────┘      │
│                              │                                   │
│   ┌──────────────────────────────────────────────────────┐      │
│   │              Application Nodes                        │      │
│   │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  │      │
│   │  │  Teleop     │  │  Obstacle   │  │   RViz2     │  │      │
│   │  │  Control    │  │  Detector   │  │   Viz       │  │      │
│   │  └─────────────┘  └─────────────┘  └─────────────┘  │      │
│   └──────────────────────────────────────────────────────┘      │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Requirements

#### Robot Model Requirements

1. **URDF Robot Description**
   - Differential drive base with two wheels and caster
   - Proper `<inertial>` properties on all links
   - Correct `<collision>` geometry for physics
   - Visually distinct materials/colors

2. **Sensor Configuration**
   - RGB camera: 640x480, 30Hz, mounted on front
   - 2D LiDAR: 360° scan, 10Hz, mounted on top
   - IMU: 100Hz, attached to base_link
   - All sensors with appropriate noise models

3. **Control Interface**
   - Differential drive plugin accepting `/cmd_vel`
   - Odometry output on `/odom`
   - Joint state publishing for wheel positions

#### World Requirements

1. **Environment Design**
   - Indoor warehouse-style environment (~20m x 20m)
   - At least 10 static obstacles (shelves, boxes, walls)
   - At least 2 different floor textures
   - Proper lighting with shadows

2. **Testing Scenarios**
   - Clear path corridor for basic navigation
   - Tight passage requiring precise control
   - Open area with scattered obstacles

#### Integration Requirements

1. **Bridge Configuration**
   - YAML configuration file for all topic bridges
   - Proper frame_id configuration for all sensors
   - Clock synchronization enabled

2. **Launch Files**
   - `simulation.launch.py`: Start Gazebo, spawn robot, launch bridges
   - `visualization.launch.py`: RViz2 with pre-configured displays
   - Support for `headless:=true` parameter

### Deliverables

- [ ] URDF file with complete robot description (`robot.urdf.xacro`)
- [ ] Gazebo-specific sensor/plugin configurations (`robot_gazebo.xacro`)
- [ ] World file with warehouse environment (`warehouse.sdf`)
- [ ] Bridge configuration file (`bridge_config.yaml`)
- [ ] Launch files for simulation and visualization
- [ ] RViz2 configuration file with all sensor displays
- [ ] README with setup instructions and screenshots
- [ ] Short video (1-2 min) demonstrating the environment

### Rubric

| Criterion | Points | Description |
|-----------|--------|-------------|
| **URDF Quality** | 20 | Valid URDF, proper inertials, collision geometry |
| **Sensor Config** | 20 | Camera, LiDAR, IMU working with noise models |
| **World Design** | 15 | Realistic environment with varied obstacles |
| **ROS 2 Integration** | 20 | Bridges work, topics publish correctly |
| **Launch Files** | 10 | Clean launch structure, configurable parameters |
| **Documentation** | 10 | Clear README, architecture diagram, screenshots |
| **Bonus Features** | +10 | Dynamic obstacles, multiple robots, Unity version |

**Total**: 100 points (110 with bonus)
**Passing**: 70 points

### Suggested Approach

1. **Days 1-2**: Create basic URDF with base and wheels
2. **Days 3-4**: Add sensors (camera, LiDAR, IMU)
3. **Days 5-6**: Build warehouse world
4. **Days 7-8**: Configure bridges and launch files
5. **Days 9-10**: Polish, document, and test

### Extension Challenges

- Add a manipulator arm to the robot
- Implement dynamic obstacles that move
- Create a Unity version for perception training
- Add multiple robots with different configurations
- Implement simulation-to-real validation metrics

### Starter Code

```xml title="robot.urdf.xacro"
<?xml version="1.0"?>
<robot xmlns:xacro="http://www.ros.org/wiki/xacro" name="sim_robot">
  <!-- Properties -->
  <xacro:property name="base_width" value="0.3"/>
  <xacro:property name="base_length" value="0.5"/>
  <xacro:property name="base_height" value="0.1"/>
  <xacro:property name="wheel_radius" value="0.1"/>
  <xacro:property name="wheel_width" value="0.05"/>

  <!-- Base Link -->
  <link name="base_link">
    <visual>
      <geometry>
        <box size="${base_length} ${base_width} ${base_height}"/>
      </geometry>
      <material name="blue">
        <color rgba="0.0 0.0 0.8 1.0"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <box size="${base_length} ${base_width} ${base_height}"/>
      </geometry>
    </collision>
    <inertial>
      <!-- TODO: Add proper inertial properties -->
      <mass value="5.0"/>
      <inertia ixx="0.1" ixy="0" ixz="0" iyy="0.1" iyz="0" izz="0.1"/>
    </inertial>
  </link>

  <!-- TODO: Add wheels with joints -->
  <!-- TODO: Add caster wheel -->
  <!-- TODO: Add sensor links (camera_link, lidar_link) -->

  <!-- Include Gazebo-specific configurations -->
  <xacro:include filename="$(find my_robot)/urdf/robot_gazebo.xacro"/>
</robot>
```

```yaml title="bridge_config.yaml"
# ROS 2-Gazebo Bridge Configuration
# Direction: GZ_TO_ROS for sensors, ROS_TO_GZ for commands

# Camera
- ros_topic_name: "/camera/image_raw"
  gz_topic_name: "/world/warehouse/model/sim_robot/link/camera_link/sensor/camera/image"
  ros_type_name: "sensor_msgs/msg/Image"
  gz_type_name: "gz.msgs.Image"
  direction: GZ_TO_ROS

# TODO: Add LiDAR bridge
# TODO: Add IMU bridge
# TODO: Add cmd_vel bridge (ROS_TO_GZ)
# TODO: Add odometry bridge
# TODO: Add clock bridge
```

---

## Project 3: Isaac Perception Pipeline

**Module**: 3 - The AI-Robot Brain (NVIDIA Isaac)

### Overview

Build a GPU-accelerated perception pipeline using Isaac ROS that processes camera and depth data to detect and localize objects in real-time. The system should demonstrate the performance benefits of GPU acceleration while integrating with a standard ROS 2 navigation stack.

### Learning Objectives

By completing this project, you will demonstrate:

- Setting up Isaac ROS development environment with Docker
- Deploying pre-trained detection models using TensorRT
- Implementing multi-sensor fusion for 3D object localization
- Building real-time object tracking across frames
- Optimizing perception pipelines for production deployment
- Integrating GPU-accelerated perception with ROS 2 systems

### System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                   Isaac Perception Pipeline                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   ┌──────────────┐     ┌──────────────┐     ┌──────────────┐   │
│   │   RGB-D      │     │   Isaac ROS  │     │   NITROS     │   │
│   │   Camera     │────▶│   Container  │────▶│   Pipeline   │   │
│   └──────────────┘     └──────────────┘     └──────────────┘   │
│                                                    │            │
│   ┌────────────────────────────────────────────────┼──────────┐│
│   │                  NITROS Layer                  │          ││
│   │  ┌─────────────┐ ┌─────────────┐ ┌────────────▼────────┐ ││
│   │  │  Rectify    │ │  DetectNet  │ │    Depth           │ ││
│   │  │  (GPU)      │ │  (TensorRT) │ │    Fusion          │ ││
│   │  └──────┬──────┘ └──────┬──────┘ └─────────┬──────────┘ ││
│   └─────────┼───────────────┼──────────────────┼────────────┘│
│             │               │                  │              │
│   ┌─────────▼───────────────▼──────────────────▼────────────┐│
│   │                    Fusion Node                           ││
│   │  • 2D→3D projection  • Multi-object tracking            ││
│   │  • Temporal filtering • Scene graph output              ││
│   └─────────────────────────────────────────────────────────┘│
│                              │                                │
│   ┌──────────────────────────▼──────────────────────────────┐│
│   │                   ROS 2 Interface                        ││
│   │  /detections_3d  /tracked_objects  /scene_graph         ││
│   └─────────────────────────────────────────────────────────┘│
│                                                                │
└─────────────────────────────────────────────────────────────────┘
```

### Requirements

#### Core Pipeline Requirements

1. **Camera Input Processing**
   - Support Intel RealSense D435i (or simulated equivalent)
   - RGB image at 640x480, 30 Hz minimum
   - Depth-aligned image for 3D projection
   - Proper camera intrinsic calibration

2. **Object Detection**
   - Deploy PeopleNet or custom detection model
   - TensorRT optimization for inference
   - Detection output at 20+ FPS on RTX 2070+
   - Support for at least 3 object classes

3. **3D Localization**
   - Project 2D detections to 3D using depth
   - Output bounding boxes in camera frame
   - Transform to robot base_link frame
   - Handle missing depth gracefully

4. **Object Tracking**
   - Persistent IDs across frames
   - Kalman filter-based state estimation
   - Handle occlusions (3+ frame memory)
   - Velocity estimation for tracked objects

#### Performance Requirements

| Metric | Target | Minimum |
|--------|--------|---------|
| End-to-end latency | < 50ms | < 100ms |
| Detection rate | 30 FPS | 20 FPS |
| Tracking accuracy | > 90% | > 80% |
| GPU memory | < 4GB | < 6GB |

#### Integration Requirements

1. **ROS 2 Topics**
   - `/camera/color/image_raw` - Input RGB
   - `/camera/depth/image_rect` - Input depth
   - `/perception/detections_2d` - 2D detections
   - `/perception/detections_3d` - 3D detections
   - `/perception/tracked_objects` - Tracked objects with IDs

2. **Services**
   - `/perception/reset_tracking` - Clear all tracks
   - `/perception/get_scene` - Return current scene graph

3. **Launch Configuration**
   - Single launch file for complete pipeline
   - Support `simulation:=true` for testing
   - Configurable detection confidence threshold
   - Enable/disable tracking mode

### Deliverables

- [ ] Isaac ROS Docker workspace setup
- [ ] Detection model deployment (TensorRT engine)
- [ ] Fusion node (Python or C++) for 3D localization
- [ ] Tracking node with Kalman filter implementation
- [ ] Launch files and configuration
- [ ] RViz2 visualization config showing detections
- [ ] Performance benchmark results
- [ ] README with setup and usage instructions
- [ ] Video demonstration (2-3 minutes)

### Rubric

| Criterion | Points | Description |
|-----------|--------|-------------|
| **Detection Pipeline** | 20 | DetectNet running with TensorRT, 20+ FPS |
| **3D Localization** | 20 | Accurate projection, proper transforms |
| **Object Tracking** | 20 | Persistent IDs, handles occlusions |
| **Performance** | 15 | Meets latency and throughput targets |
| **Integration** | 10 | Clean ROS 2 interfaces, launch files |
| **Documentation** | 10 | Clear setup, architecture diagram |
| **Bonus Features** | +15 | Custom model, scene graph, Isaac Sim integration |

**Total**: 100 points (115 with bonus)
**Passing**: 70 points

### Suggested Approach

1. **Days 1-2**: Set up Isaac ROS Docker environment
2. **Days 3-4**: Deploy and test DetectNet
3. **Days 5-6**: Implement 3D projection and fusion
4. **Days 7-8**: Add object tracking
5. **Days 9-10**: Optimize, benchmark, and document

### Extension Challenges

- Train custom detection model for specific objects
- Add semantic segmentation for scene understanding
- Implement multi-camera fusion
- Create Isaac Sim test environment
- Build scene graph with spatial relations
- Deploy on Jetson for embedded testing

### Starter Code

```python title="perception_node.py"
"""Isaac ROS perception pipeline node."""
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from vision_msgs.msg import Detection3DArray, Detection3D
from tf2_ros import Buffer, TransformListener
import numpy as np

class PerceptionNode(Node):
    """GPU-accelerated perception pipeline."""

    def __init__(self):
        super().__init__('perception_pipeline')

        # Parameters
        self.declare_parameter('confidence_threshold', 0.5)
        self.declare_parameter('tracking_enabled', True)

        # TF2 for coordinate transforms
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Camera intrinsics (populated from CameraInfo)
        self.camera_matrix = None

        # Subscribers
        self.detection_sub = self.create_subscription(
            Detection2DArray, '/detectnet/detections',
            self.detection_callback, 10
        )
        self.depth_sub = self.create_subscription(
            Image, '/camera/depth/image_rect',
            self.depth_callback, 10
        )
        self.info_sub = self.create_subscription(
            CameraInfo, '/camera/color/camera_info',
            self.info_callback, 10
        )

        # Publishers
        self.detection_3d_pub = self.create_publisher(
            Detection3DArray, '/perception/detections_3d', 10
        )

        # State
        self.latest_depth = None
        self.tracker = None  # TODO: Initialize tracker

        self.get_logger().info('Perception node initialized')

    def info_callback(self, msg):
        """Store camera intrinsics."""
        self.camera_matrix = np.array(msg.k).reshape(3, 3)

    def depth_callback(self, msg):
        """Store latest depth image."""
        # TODO: Convert to numpy array
        pass

    def detection_callback(self, msg):
        """Process 2D detections and project to 3D."""
        if self.camera_matrix is None or self.latest_depth is None:
            return

        detections_3d = Detection3DArray()
        detections_3d.header = msg.header

        for det_2d in msg.detections:
            # TODO: Project 2D bbox center to 3D using depth
            # TODO: Create Detection3D message
            # TODO: Transform to base_link frame
            pass

        # TODO: Update tracker with detections

        self.detection_3d_pub.publish(detections_3d)


def main(args=None):
    rclpy.init(args=args)
    node = PerceptionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
```

```yaml title="perception_launch_config.yaml"
/**:
  ros__parameters:
    # Detection
    confidence_threshold: 0.5
    max_detections: 50

    # Tracking
    tracking_enabled: true
    max_track_age: 5
    min_track_hits: 3

    # 3D Localization
    depth_scale: 0.001  # mm to meters
    max_depth: 5.0
    min_depth: 0.3

    # Frame IDs
    camera_frame: "camera_color_optical_frame"
    base_frame: "base_link"
```

---

## Project 4: VLA Demo System

**Module**: 4 - Vision-Language-Action Models

### Overview

Implement a demonstration system that uses vision-language models to interpret natural language commands and generate robot actions based on visual scene understanding. This project will help you understand how state-of-the-art models like RT-2 and Octo combine perception, language, and action.

### Learning Objectives

By completing this project, you will demonstrate:

- Integration of vision and language models for robotics
- Understanding of action tokenization and execution
- End-to-end inference pipeline implementation
- Evaluation of model performance on robotic tasks
- Troubleshooting of complex AI systems

### System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    VLA Demo System                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   ┌─────────────┐     ┌──────────────────┐     ┌─────────────┐ │
│   │   Camera    │────▶│   VLA Model      │────▶│  Action     │ │
│   │   (Image)   │     │ (Vision-Language │     │  Decoder    │ │
│   └─────────────┘     │  -Action)        │     │             │ │
│                       └──────────────────┘     └─────────────┘ │
│                                │                        │      │
│   ┌─────────────────┐          │                        ▼      │
│   │  Natural        │─────────▶│               ┌──────────────┐│
│   │  Language       │          │               │  Robot       ││
│   │  Command        │          │               │  Execution   ││
│   └─────────────────┘          │               │    Node      ││
│                                └───────────────┤              ││
│                                                └──────────────┘│
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Requirements

#### Core Pipeline Requirements

1. **Vision Input Processing**
   - Support RGB camera input (640x480 minimum)
   - Preprocess images for VLA model input
   - Handle variable lighting conditions
   - Integrate with ROS 2 image transport

2. **Language Processing**
   - Accept natural language commands via text input
   - Parse and interpret command intent
   - Map language to potential robot actions
   - Handle ambiguous or complex commands

3. **VLA Model Integration**
   - Deploy pre-trained VLA model (RT-2, Octo, or similar)
   - Handle model inference efficiently
   - Convert model outputs to robot actions
   - Implement appropriate action discretization

4. **Action Execution**
   - Convert predicted actions to robot control commands
   - Handle multiple action modalities (navigation, manipulation)
   - Implement safety checks and validation
   - Provide feedback on action success/failure

#### Integration Requirements

1. **ROS 2 Topics**
   - `/vla/command` - Natural language commands
   - `/camera/image_raw` - Input images
   - `/vla/actions` - Predicted robot actions
   - `/vla/status` - System status and feedback

2. **Performance Requirements**
   - End-to-end latency under 2 seconds for inference
   - Handle 1Hz command frequency minimum
   - GPU memory usage under 8GB (if using GPU)
   - CPU usage under 50% during inference

3. **Launch Configuration**
   - Single launch file for complete pipeline
   - Support for `simulation:=true` parameter
   - Configurable model selection
   - Enable/disable visualization options

### Deliverables

- [ ] VLA model integration with ROS 2 interface
- [ ] Image preprocessing pipeline
- [ ] Natural language command parser
- [ ] Action decoder for robot execution
- [ ] Launch files and configuration
- [ ] RViz2 visualization config for action planning
- [ ] Performance benchmark results
- [ ] README with setup and usage instructions
- [ ] Video demonstration (2-3 minutes)

### Rubric

| Criterion | Points | Description |
|-----------|--------|-------------|
| **VLA Model Integration** | 25 | Successfully deployed and running VLA model |
| **Vision Processing** | 20 | Proper image handling and preprocessing |
| **Language Understanding** | 20 | Accurate command interpretation |
| **Action Execution** | 15 | Actions properly converted to robot commands |
| **Performance** | 10 | Meets latency and resource requirements |
| **Integration** | 5 | Clean ROS 2 interfaces, launch files |
| **Documentation** | 5 | Clear setup, architecture diagram |
| **Bonus Features** | +15 | Advanced language understanding, multimodal input, etc. |

**Total**: 100 points (115 with bonus)
**Passing**: 70 points

### Suggested Approach

1. **Days 1-2**: Set up VLA model environment and test inference
2. **Days 3-4**: Build image preprocessing pipeline
3. **Days 5-6**: Implement language command processing
4. **Days 7-8**: Connect model outputs to robot actions
5. **Days 9-10**: Test, integrate, and optimize performance

### Extension Challenges

- Implement few-shot learning capabilities
- Add multi-step command handling
- Integrate with a real robot (not just simulation)
- Add uncertainty quantification to predictions
- Create a web interface for text commands

### Starter Code Structure

```python title="vla_demo_node.py"
"""VLA Demo System node."""
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
import numpy as np
import torch

class VLADemoNode(Node):
    """Vision-Language-Action demonstration system."""

    def __init__(self):
        super().__init__('vla_demo_system')

        # Parameters
        self.declare_parameter('model_name', 'octo-base')
        self.declare_parameter('confidence_threshold', 0.5)

        # Subscribers
        self.image_sub = self.create_subscription(
            Image, '/camera/image_raw', self.image_callback, 10
        )
        self.command_sub = self.create_subscription(
            String, '/vla/command', self.command_callback, 10
        )

        # Publishers
        # TODO: Define publishers for actions and status

        # Initialize VLA model
        self.load_vla_model()

        # State
        self.current_image = None
        self.pending_command = None

        self.get_logger().info('VLA Demo System initialized')

    def load_vla_model(self):
        """Load pre-trained VLA model."""
        # TODO: Load model (Octo, RT-2, or equivalent)
        pass

    def image_callback(self, msg):
        """Process incoming image."""
        # TODO: Store image and perform inference if command is pending
        pass

    def command_callback(self, msg):
        """Process incoming language command."""
        # TODO: Store command and perform inference if image is available
        pass

    def run_vla_inference(self, image, command):
        """Run VLA model inference."""
        # TODO: Process image and command through VLA model
        # TODO: Convert outputs to action space
        pass


def main(args=None):
    rclpy.init(args=args)
    node = VLADemoNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
```

```yaml title="vla_demo_config.yaml"
/**:
  ros__parameters:
    # VLA Model
    model_name: "octo-base"  # or "rt-2"
    model_path: "/path/to/pretrained/model"

    # Processing
    image_resize: [128, 128]  # for model input
    confidence_threshold: 0.5

    # Actions
    max_action_steps: 10
    action_space: "discrete"  # or "continuous"
```

### Resources

- [Octo Repository](https://github.com/octo-models)
- [RT-2 Paper](https://arxiv.org/abs/2307.15818)
- [VLA Models Overview](https://huggingface.co/papers/2310.12945)

---

## Capstone Project

**All Modules**

### Overview

Design and implement an autonomous assistant robot that combines all skills learned throughout the course. The robot should navigate an environment, perceive objects, understand natural language commands, and perform manipulation tasks in response to high-level instructions.

### Learning Objectives

By completing this capstone project, you will demonstrate:

- Integration of all Physical AI components into a cohesive system
- End-to-end system design and implementation
- Troubleshooting complex multi-component systems
- Professional-level documentation and presentation
- Application of AI techniques to real-world robotics problems

### System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                  Autonomous Assistant System                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │                High-Level Controller                      │   │
│   │  • Command interpretation                                │   │
│   │  • Task planning                                         │   │
│   │  • Behavior arbitration                                  │   │
│   └─────────────┬─────────────────────────────────────────────┘   │
│                 │                                                 │
│   ┌─────────────▼─────────────┐  ┌─────────────────────────────┐ │
│   │   VLA Command Handler     │  │     Navigation System       │ │
│   │ • Natural language input  │  │ • Map building (SLAM)       │ │
│   │ • Intent interpretation   │  │ • Path planning             │ │
│   │ • Action sequence gen.    │  │ • Path following            │ │
│   └─────────────┬─────────────┘  └─────────────┬───────────────┘ │
│                 │                              │                 │
│   ┌─────────────▼─────────────┐  ┌─────────────▼───────────────┐ │
│   │    Perception System      │  │    Manipulation System      │ │
│   │ • Object detection        │  │ • Grasp planning            │ │
│   │ • Scene understanding     │  │ • Trajectory execution      │ │
│   │ • State estimation        │  │ • Pick-and-place            │ │
│   └───────────────────────────┘  └─────────────────────────────┘ │
│                              │                                   │
│   ┌──────────────────────────▼────────────────────────────────┐ │
│   │                   Robot Platform                          │ │
│   │  • Mobile base with differential drive                    │ │
│   │  • Manipulator arm with end-effector                      │ │
│   │  • RGB-D camera, LiDAR, IMU                               │ │
│   │  • ROS 2 integration                                      │ │
│   └───────────────────────────────────────────────────────────┘ │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Requirements

#### Core System Requirements

1. **Command Processing**
   - Accept high-level natural language commands
   - Parse and decompose complex tasks
   - Generate executable action sequences
   - Handle command clarification queries

2. **Autonomous Navigation**
   - Build map of environment (SLAM)
   - Plan paths to specified locations
   - Navigate while avoiding obstacles
   - Handle dynamic obstacles if present

3. **Perception System**
   - Detect and identify objects in the environment
   - Estimate object poses for manipulation
   - Track objects and environment changes
   - Integrate multiple sensor modalities

4. **Manipulation Capabilities**
   - Execute pick-and-place operations
   - Handle objects of various shapes/sizes
   - Plan collision-free trajectories
   - Execute precise manipulations

#### Integration Requirements

1. **ROS 2 Architecture**
   - Modular node design with clear interfaces
   - Proper error handling and recovery
   - Appropriate message types and topics
   - Efficient state management

2. **Performance Requirements**
   - Complete task within 10 minutes average
   - Navigation success rate > 90%
   - Object manipulation success rate > 75%
   - System uptime > 95% during operation

3. **Safety & Robustness**
   - Emergency stop functionality
   - Collision avoidance during navigation and manipulation
   - Graceful degradation when components fail
   - Safe operation limits enforced

### Deliverables

- [ ] Complete integrated robot system
- [ ] High-level command processing module
- [ ] Autonomous navigation with SLAM
- [ ] Object perception and detection
- [ ] Manipulation system with pick/place
- [ ] System integration and orchestration
- [ ] Comprehensive documentation
- [ ] 10-minute technical presentation
- [ ] Video demonstration of complete system
- [ ] Source code with unit tests

### Rubric

| Criterion | Points | Description |
|-----------|--------|-------------|
| **System Integration** | 25 | All components work together seamlessly |
| **Command Processing** | 15 | Natural language commands interpreted and executed |
| **Navigation** | 15 | Successful autonomous navigation and mapping |
| **Perception** | 15 | Accurate object detection and state estimation |
| **Manipulation** | 15 | Successful pick-and-place operations |
| **Performance** | 10 | Meets specified performance requirements |
| **Documentation** | 5 | Complete technical documentation |
| **Presentation** | 5 | Clear explanation of design and implementation |
| **Bonus Features** | +10 | Advanced capabilities, exceptional implementation |

**Total**: 105 points (115 with bonus)
**Passing**: 75 points

### Suggested Approach

1. **Week 1**: System architecture design and component integration plan
2. **Week 2**: Implement basic command processing and navigation
3. **Week 3**: Add perception and manipulation capabilities
4. **Week 4**: Integrate components and optimize system performance
5. **Week 5**: Final testing, documentation, and presentation preparation

### Extension Challenges

- Multi-object manipulation sequences
- Human-robot interaction and collaboration
- Learning from demonstration
- Adaptive behavior based on environment
- Integration with cloud services or external APIs

### Architecture Considerations

- Use behavior trees or state machines for task orchestration
- Implement proper error handling and recovery mechanisms
- Design modular components that can be independently tested
- Plan for resource constraints on robot computer
- Consider real-time performance requirements

### Resources

- [Navigation2 Documentation](https://navigation.ros.org/)
- [MoveIt! Motion Planning](https://moveit.ros.org/)
- [Behavior Trees in Robotics](https://www.cs.unc.edu/~jcarback/publications/ICRA14_bts.pdf)
- [ROS 2 Design](https://design.ros2.org/)

---

## Submission Guidelines

### Format

- Submit via GitHub repository
- Include all source code, configuration files, and documentation
- Provide clear build and run instructions
- Include video demonstration (2-5 minutes)

### Grading Timeline

- Projects submitted by deadline: Full credit eligible
- Up to 1 week late: Maximum 90% credit
- Up to 2 weeks late: Maximum 70% credit
- Beyond 2 weeks: Not accepted

### Academic Integrity

- All code must be your own or properly attributed
- Collaboration is encouraged for learning, not copying
- Use of AI assistants must be disclosed
- External libraries and references must be cited
