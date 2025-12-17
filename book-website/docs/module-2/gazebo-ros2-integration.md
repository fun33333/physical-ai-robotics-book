---
title: "Gazebo-ROS 2 Integration"
sidebar_position: 4
description: "Connecting Gazebo simulations to ROS 2: bridges, plugins, and seamless hardware abstraction."
---

# Gazebo-ROS 2 Integration

The power of Gazebo for robotics development comes from its seamless integration with ROS 2. This chapter teaches you to bridge your simulations to the ROS 2 ecosystem, enabling the same code to run on simulated and real robots.

## Overview

In this section, you will:

- Understand the ros_gz bridge architecture and message mapping
- Configure bidirectional topic bridges for sensors and commands
- Set up ROS 2 launch files for simulation
- Use Gazebo plugins with ROS 2 interfaces
- Implement hardware abstraction for sim/real portability
- Debug common integration issues

## Prerequisites

Before starting, ensure you have:

- [Gazebo Basics](/docs/module-2/gazebo-basics) completed
- Gazebo Harmonic and ros-humble-ros-gz installed
- A working robot URDF with sensors

---

## The ros_gz Bridge

### Architecture

The ros_gz bridge translates messages between Gazebo Transport and ROS 2:

```
┌─────────────────────────────────────────────────────────────────┐
│                    ros_gz Bridge Architecture                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   ┌──────────────────┐              ┌──────────────────┐        │
│   │    Gazebo Sim    │              │      ROS 2       │        │
│   │   (gz Transport) │              │   (DDS/rmw)      │        │
│   └────────┬─────────┘              └────────┬─────────┘        │
│            │                                  │                  │
│            │    ┌────────────────────┐       │                  │
│            └───►│   ros_gz_bridge    │◄──────┘                  │
│                 │                    │                          │
│                 │  • Message convert │                          │
│                 │  • Topic mapping   │                          │
│                 │  • QoS handling    │                          │
│                 └────────────────────┘                          │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Bridge Directions

| Direction | Use Case | Example |
|-----------|----------|---------|
| `GZ_TO_ROS` | Sensor data to ROS 2 | Camera images, LiDAR scans |
| `ROS_TO_GZ` | Commands to Gazebo | Velocity commands, joint positions |
| `BIDIRECTIONAL` | Two-way communication | Clock synchronization |

---

## Installing the Bridge

```bash title="Install ros_gz packages"
# Core bridge packages
sudo apt install ros-humble-ros-gz-bridge
sudo apt install ros-humble-ros-gz-sim
sudo apt install ros-humble-ros-gz-image

# Additional utilities
sudo apt install ros-humble-ros-gz-interfaces
```

---

## Configuring Topic Bridges

### YAML Configuration

Create a bridge configuration file:

```yaml title="bridge_config.yaml"
# Sensor bridges (Gazebo to ROS 2)
- ros_topic_name: "/camera/image_raw"
  gz_topic_name: "/world/default/model/robot/link/camera_link/sensor/camera/image"
  ros_type_name: "sensor_msgs/msg/Image"
  gz_type_name: "gz.msgs.Image"
  direction: GZ_TO_ROS

- ros_topic_name: "/camera/camera_info"
  gz_topic_name: "/world/default/model/robot/link/camera_link/sensor/camera/camera_info"
  ros_type_name: "sensor_msgs/msg/CameraInfo"
  gz_type_name: "gz.msgs.CameraInfo"
  direction: GZ_TO_ROS

- ros_topic_name: "/scan"
  gz_topic_name: "/world/default/model/robot/link/lidar_link/sensor/lidar/scan"
  ros_type_name: "sensor_msgs/msg/LaserScan"
  gz_type_name: "gz.msgs.LaserScan"
  direction: GZ_TO_ROS

- ros_topic_name: "/imu"
  gz_topic_name: "/world/default/model/robot/link/base_link/sensor/imu/imu"
  ros_type_name: "sensor_msgs/msg/Imu"
  gz_type_name: "gz.msgs.IMU"
  direction: GZ_TO_ROS

# Command bridges (ROS 2 to Gazebo)
- ros_topic_name: "/cmd_vel"
  gz_topic_name: "/model/robot/cmd_vel"
  ros_type_name: "geometry_msgs/msg/Twist"
  gz_type_name: "gz.msgs.Twist"
  direction: ROS_TO_GZ

# Clock synchronization (bidirectional)
- ros_topic_name: "/clock"
  gz_topic_name: "/clock"
  ros_type_name: "rosgraph_msgs/msg/Clock"
  gz_type_name: "gz.msgs.Clock"
  direction: GZ_TO_ROS
```

### Running the Bridge

```bash title="Launch bridge with config"
# Using configuration file
ros2 run ros_gz_bridge parameter_bridge --ros-args -p config_file:=bridge_config.yaml

# Quick bridge for single topic
ros2 run ros_gz_bridge parameter_bridge /cmd_vel@geometry_msgs/msg/Twist]gz.msgs.Twist
```

---

## ROS 2 Launch Files

### Complete Simulation Launch

```python title="simulation.launch.py"
import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node

def generate_launch_description():
    # Package directories
    pkg_ros_gz_sim = get_package_share_directory('ros_gz_sim')
    pkg_my_robot = get_package_share_directory('my_robot_description')

    # Launch arguments
    world_file = LaunchConfiguration('world')
    use_sim_time = LaunchConfiguration('use_sim_time', default='true')

    # Declare arguments
    declare_world_arg = DeclareLaunchArgument(
        'world',
        default_value=PathJoinSubstitution([pkg_my_robot, 'worlds', 'warehouse.sdf']),
        description='Path to world file'
    )

    # Start Gazebo
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            os.path.join(pkg_ros_gz_sim, 'launch', 'gz_sim.launch.py')
        ]),
        launch_arguments={
            'gz_args': ['-r ', world_file],
        }.items()
    )

    # Spawn robot from URDF
    spawn_robot = Node(
        package='ros_gz_sim',
        executable='create',
        arguments=[
            '-name', 'my_robot',
            '-topic', 'robot_description',
            '-x', '0.0',
            '-y', '0.0',
            '-z', '0.5',
        ],
        output='screen',
    )

    # Robot state publisher
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        parameters=[{
            'robot_description': open(
                os.path.join(pkg_my_robot, 'urdf', 'robot.urdf')
            ).read(),
            'use_sim_time': use_sim_time,
        }],
        output='screen',
    )

    # Bridge configuration
    bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        parameters=[{
            'config_file': os.path.join(pkg_my_robot, 'config', 'bridge.yaml'),
            'use_sim_time': use_sim_time,
        }],
        output='screen',
    )

    # RViz2 for visualization
    rviz = Node(
        package='rviz2',
        executable='rviz2',
        arguments=['-d', os.path.join(pkg_my_robot, 'rviz', 'simulation.rviz')],
        parameters=[{'use_sim_time': use_sim_time}],
        output='screen',
    )

    return LaunchDescription([
        declare_world_arg,
        gazebo,
        spawn_robot,
        robot_state_publisher,
        bridge,
        rviz,
    ])
```

### Launch Commands

```bash title="Running the simulation"
# Build your package first
colcon build --packages-select my_robot_description

# Source the workspace
source install/setup.bash

# Launch simulation
ros2 launch my_robot_description simulation.launch.py

# Launch with custom world
ros2 launch my_robot_description simulation.launch.py world:=/path/to/custom.sdf
```

---

## Sensor Integration

### Camera with Image Bridge

```xml title="camera_gazebo.urdf.xacro"
<?xml version="1.0"?>
<robot xmlns:xacro="http://www.ros.org/wiki/xacro">
  <xacro:macro name="camera_sensor" params="parent_link">
    <link name="camera_link">
      <visual>
        <geometry>
          <box size="0.02 0.05 0.02"/>
        </geometry>
      </visual>
      <collision>
        <geometry>
          <box size="0.02 0.05 0.02"/>
        </geometry>
      </collision>
      <inertial>
        <mass value="0.01"/>
        <inertia ixx="1e-6" ixy="0" ixz="0" iyy="1e-6" iyz="0" izz="1e-6"/>
      </inertial>
    </link>

    <joint name="camera_joint" type="fixed">
      <parent link="${parent_link}"/>
      <child link="camera_link"/>
      <origin xyz="0.2 0 0.1" rpy="0 0 0"/>
    </joint>

    <!-- Gazebo sensor configuration -->
    <gazebo reference="camera_link">
      <sensor name="camera" type="camera">
        <always_on>true</always_on>
        <update_rate>30</update_rate>
        <visualize>true</visualize>
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
          <noise>
            <type>gaussian</type>
            <mean>0.0</mean>
            <stddev>0.007</stddev>
          </noise>
        </camera>
        <!-- Topic will be: /world/{world}/model/{model}/link/camera_link/sensor/camera/image -->
      </sensor>
    </gazebo>
  </xacro:macro>
</robot>
```

### LiDAR with LaserScan Bridge

```xml title="lidar_gazebo.urdf.xacro"
<?xml version="1.0"?>
<robot xmlns:xacro="http://www.ros.org/wiki/xacro">
  <xacro:macro name="lidar_sensor" params="parent_link">
    <link name="lidar_link">
      <visual>
        <geometry>
          <cylinder radius="0.05" length="0.04"/>
        </geometry>
        <material name="dark_grey">
          <color rgba="0.2 0.2 0.2 1"/>
        </material>
      </visual>
      <collision>
        <geometry>
          <cylinder radius="0.05" length="0.04"/>
        </geometry>
      </collision>
      <inertial>
        <mass value="0.2"/>
        <inertia ixx="1e-4" ixy="0" ixz="0" iyy="1e-4" iyz="0" izz="1e-4"/>
      </inertial>
    </link>

    <joint name="lidar_joint" type="fixed">
      <parent link="${parent_link}"/>
      <child link="lidar_link"/>
      <origin xyz="0 0 0.15" rpy="0 0 0"/>
    </joint>

    <gazebo reference="lidar_link">
      <sensor name="lidar" type="gpu_lidar">
        <always_on>true</always_on>
        <update_rate>10</update_rate>
        <visualize>true</visualize>
        <lidar>
          <scan>
            <horizontal>
              <samples>360</samples>
              <resolution>1.0</resolution>
              <min_angle>-3.14159</min_angle>
              <max_angle>3.14159</max_angle>
            </horizontal>
            <vertical>
              <samples>1</samples>
              <resolution>1</resolution>
              <min_angle>0</min_angle>
              <max_angle>0</max_angle>
            </vertical>
          </scan>
          <range>
            <min>0.1</min>
            <max>30.0</max>
            <resolution>0.01</resolution>
          </range>
          <noise>
            <type>gaussian</type>
            <mean>0.0</mean>
            <stddev>0.02</stddev>
          </noise>
        </lidar>
      </sensor>
    </gazebo>
  </xacro:macro>
</robot>
```

---

## Control Integration

### Differential Drive Controller

```xml title="Adding diff_drive to URDF for ROS 2"
<gazebo>
  <!-- Differential drive plugin -->
  <plugin filename="gz-sim-diff-drive-system"
          name="gz::sim::systems::DiffDrive">
    <left_joint>left_wheel_joint</left_joint>
    <right_joint>right_wheel_joint</right_joint>
    <wheel_separation>0.35</wheel_separation>
    <wheel_radius>0.1</wheel_radius>
    <max_linear_acceleration>1.0</max_linear_acceleration>
    <max_angular_acceleration>2.0</max_angular_acceleration>
    <odom_publish_frequency>50</odom_publish_frequency>

    <!-- Topic names (Gazebo side) -->
    <topic>cmd_vel</topic>
    <odom_topic>odom</odom_topic>
    <tf_topic>tf</tf_topic>

    <!-- Frame IDs -->
    <frame_id>odom</frame_id>
    <child_frame_id>base_link</child_frame_id>
  </plugin>

  <!-- Joint state publisher -->
  <plugin filename="gz-sim-joint-state-publisher-system"
          name="gz::sim::systems::JointStatePublisher">
    <joint_name>left_wheel_joint</joint_name>
    <joint_name>right_wheel_joint</joint_name>
  </plugin>
</gazebo>
```

### Bridge for Control Topics

```yaml title="control_bridge.yaml"
# Velocity commands
- ros_topic_name: "/cmd_vel"
  gz_topic_name: "/model/my_robot/cmd_vel"
  ros_type_name: "geometry_msgs/msg/Twist"
  gz_type_name: "gz.msgs.Twist"
  direction: ROS_TO_GZ

# Odometry
- ros_topic_name: "/odom"
  gz_topic_name: "/model/my_robot/odom"
  ros_type_name: "nav_msgs/msg/Odometry"
  gz_type_name: "gz.msgs.Odometry"
  direction: GZ_TO_ROS

# Joint states
- ros_topic_name: "/joint_states"
  gz_topic_name: "/world/default/model/my_robot/joint_state"
  ros_type_name: "sensor_msgs/msg/JointState"
  gz_type_name: "gz.msgs.Model"
  direction: GZ_TO_ROS

# TF (transforms)
- ros_topic_name: "/tf"
  gz_topic_name: "/model/my_robot/tf"
  ros_type_name: "tf2_msgs/msg/TFMessage"
  gz_type_name: "gz.msgs.Pose_V"
  direction: GZ_TO_ROS
```

---

## Hardware Abstraction Pattern

### The Key Principle

Write your ROS 2 nodes to be simulation-agnostic:

```python title="robot_controller.py"
#!/usr/bin/env python3
"""
Robot controller that works identically in simulation and on real hardware.
The only difference is the underlying data source (Gazebo bridge vs real drivers).
"""
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry

class RobotController(Node):
    def __init__(self):
        super().__init__('robot_controller')

        # Parameters
        self.declare_parameter('use_sim_time', False)
        self.declare_parameter('max_linear_speed', 0.5)
        self.declare_parameter('max_angular_speed', 1.0)

        # Publishers - same topic names for sim and real
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        # Subscribers - same topic names for sim and real
        self.scan_sub = self.create_subscription(
            LaserScan, '/scan', self.scan_callback, 10
        )
        self.odom_sub = self.create_subscription(
            Odometry, '/odom', self.odom_callback, 10
        )

        # Control loop
        self.timer = self.create_timer(0.1, self.control_loop)

        self.latest_scan = None
        self.latest_odom = None

        self.get_logger().info('Robot controller initialized')

    def scan_callback(self, msg: LaserScan):
        self.latest_scan = msg

    def odom_callback(self, msg: Odometry):
        self.latest_odom = msg

    def control_loop(self):
        """Main control logic - works in sim or real."""
        if self.latest_scan is None:
            return

        cmd = Twist()
        max_linear = self.get_parameter('max_linear_speed').value
        max_angular = self.get_parameter('max_angular_speed').value

        # Simple obstacle avoidance
        min_distance = min(self.latest_scan.ranges)

        if min_distance < 0.5:
            # Obstacle close - turn
            cmd.linear.x = 0.0
            cmd.angular.z = max_angular
        else:
            # Clear path - go forward
            cmd.linear.x = max_linear
            cmd.angular.z = 0.0

        self.cmd_pub.publish(cmd)


def main(args=None):
    rclpy.init(args=args)
    controller = RobotController()
    rclpy.spin(controller)
    controller.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
```

### Launch File Abstraction

```python title="robot.launch.py"
"""
Launch file that can start simulation or real robot based on parameters.
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    # Declare simulation argument
    use_sim = LaunchConfiguration('use_sim', default='true')

    declare_use_sim = DeclareLaunchArgument(
        'use_sim',
        default_value='true',
        description='Use simulation (true) or real robot (false)'
    )

    # Common nodes (run in both sim and real)
    common_nodes = GroupAction([
        Node(
            package='my_robot_control',
            executable='robot_controller',
            parameters=[{
                'use_sim_time': use_sim,
            }],
            output='screen',
        ),
    ])

    # Simulation-only nodes
    sim_nodes = GroupAction(
        condition=IfCondition(use_sim),
        actions=[
            # Gazebo and bridge launched separately or included here
        ]
    )

    # Real robot nodes
    real_nodes = GroupAction(
        condition=UnlessCondition(use_sim),
        actions=[
            Node(
                package='my_robot_driver',
                executable='motor_driver',
                output='screen',
            ),
            Node(
                package='my_robot_driver',
                executable='sensor_driver',
                output='screen',
            ),
        ]
    )

    return LaunchDescription([
        declare_use_sim,
        common_nodes,
        sim_nodes,
        real_nodes,
    ])
```

---

## Debugging Integration Issues

### Common Problems and Solutions

| Problem | Symptoms | Solution |
|---------|----------|----------|
| No data on ROS 2 topics | `ros2 topic list` shows topic but `echo` shows nothing | Check Gazebo topic name in `gz topic -l` |
| Wrong frame IDs | TF errors in RViz2 | Verify `frame_id` in sensor config |
| Time synchronization | Stale data warnings | Enable `use_sim_time` parameter |
| Bridge crashes | Segfault on startup | Check message type compatibility |
| Missing transforms | "Could not find transform" | Add `robot_state_publisher` node |

### Debugging Commands

```bash title="Debugging toolkit"
# List all Gazebo topics
gz topic -l

# Echo a Gazebo topic
gz topic -e -t /world/default/model/my_robot/link/camera_link/sensor/camera/image

# List ROS 2 topics
ros2 topic list

# Check bridge status
ros2 node info /ros_gz_bridge

# Verify transforms
ros2 run tf2_tools view_frames

# Check message types match
ros2 interface show sensor_msgs/msg/Image
gz msg --info gz.msgs.Image
```

### Visualization in RViz2

```bash title="Verify sensor data in RViz2"
# Launch RViz2 with sim time
ros2 run rviz2 rviz2 --ros-args -p use_sim_time:=true

# Add displays:
# - LaserScan on /scan
# - Image on /camera/image_raw
# - TF
# - RobotModel
```

---

## Exercise 1: Complete Sensor Bridge

:::tip Exercise 1: Bridge All Robot Sensors
**Objective**: Configure a bridge for a robot with camera, LiDAR, and IMU.

**Time Estimate**: 30 minutes

**Steps**:

1. Create a `sensor_bridge.yaml` configuration file
2. Add bridges for:
   - Camera image and camera_info
   - LiDAR scan
   - IMU data
3. Launch Gazebo with your robot
4. Run the bridge
5. Verify all topics appear in ROS 2

**Verification**:

```bash
# Check all topics are publishing
ros2 topic hz /camera/image_raw
ros2 topic hz /scan
ros2 topic hz /imu
```

**Expected Result**: All three sensors publishing at expected rates (camera ~30Hz, LiDAR ~10Hz, IMU ~100Hz).
:::

---

## Exercise 2: Teleoperation Setup

:::tip Exercise 2: Keyboard Control of Simulated Robot
**Objective**: Set up keyboard teleoperation for your simulated robot.

**Time Estimate**: 20 minutes

**Steps**:

1. Ensure differential drive plugin is configured
2. Add cmd_vel bridge (ROS_TO_GZ direction)
3. Launch simulation and bridge
4. Run teleop_twist_keyboard
5. Control the robot with keyboard

**Commands**:

```bash
# Terminal 1: Launch simulation
ros2 launch my_robot_description simulation.launch.py

# Terminal 2: Keyboard teleop
ros2 run teleop_twist_keyboard teleop_twist_keyboard --ros-args -r cmd_vel:=/cmd_vel
```

**Expected Result**: Robot responds to WASD/arrow keys, moving and turning in Gazebo.
:::

---

## Exercise 3: RViz2 Visualization

:::tip Exercise 3: Visualize Simulation in RViz2
**Objective**: Configure RViz2 to show robot model, sensors, and transforms.

**Time Estimate**: 25 minutes

**Steps**:

1. Launch your simulation with all bridges
2. Open RViz2 with `use_sim_time:=true`
3. Add TF display and set Fixed Frame to "odom"
4. Add RobotModel display
5. Add LaserScan display for /scan
6. Add Image display for /camera/image_raw
7. Save the configuration for future use

**Expected Result**: RViz2 shows the robot model with live sensor data overlaid on the 3D view.
:::

---

## Summary

In this chapter, you learned:

- **Bridge Architecture**: ros_gz connects Gazebo and ROS 2 via message translation
- **Configuration**: YAML files define topic mappings and directions
- **Launch Files**: Python launch files orchestrate simulation startup
- **Sensor Integration**: Camera, LiDAR, and IMU bridges with proper frame IDs
- **Control Integration**: Differential drive and joint state bridges
- **Hardware Abstraction**: Write code that works in simulation and on real robots
- **Debugging**: Tools and techniques for troubleshooting integration issues

With Gazebo-ROS 2 integration mastered, you can now develop and test robotics applications entirely in simulation before deploying to hardware.

Next, explore [Unity for Robotics](/docs/module-2/unity-robotics) for photorealistic simulation and synthetic data generation.

## Further Reading

- [ros_gz Documentation](https://github.com/gazebosim/ros_gz)
- [ROS 2 Launch Tutorial](https://docs.ros.org/en/humble/Tutorials/Intermediate/Launch/Launch-Main.html)
- [Gazebo-ROS 2 Migration Guide](https://gazebosim.org/docs/harmonic/ros2_integration)
- [TF2 Documentation](https://docs.ros.org/en/humble/Concepts/About-Tf2.html)
