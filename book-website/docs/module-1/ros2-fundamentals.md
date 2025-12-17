---
title: "ROS 2 Fundamentals"
sidebar_position: 2
description: "Introduction to ROS 2 architecture, installation, workspace setup, and core concepts."
---

# ROS 2 Fundamentals

ROS 2 represents a significant evolution from its predecessor, designed from the ground up for real-time systems, multi-robot scenarios, and production deployments. This chapter covers the foundational concepts you need to understand before diving into more advanced topics.

## Overview

In this section, you will:

- Install ROS 2 Humble on Ubuntu 22.04
- Understand the DDS middleware architecture
- Set up your first ROS 2 workspace
- Create and build packages with colcon
- Use essential CLI tools for introspection
- Configure Quality of Service (QoS) policies

## Prerequisites

Before starting, ensure you have:

- Ubuntu 22.04 LTS (native or WSL2 on Windows)
- Basic terminal familiarity (cd, ls, sudo)
- Python 3.10+ installed
- At least 10GB free disk space

:::info WSL2 Users
ROS 2 works well in WSL2. For GUI applications (rviz2, rqt), install an X server like VcXsrv or use WSLg on Windows 11.
:::

## ROS 2 Architecture

### Why ROS 2?

ROS 1 served the robotics community well for over a decade, but it had limitations:

| ROS 1 Limitation | ROS 2 Solution |
|------------------|----------------|
| Single Master (roscore) | Decentralized discovery |
| No real-time support | Real-time capable |
| Linux only | Cross-platform (Linux, Windows, macOS) |
| Limited security | Built-in DDS security |
| Single-robot focus | Multi-robot native |

### The DDS Middleware

ROS 2 uses Data Distribution Service (DDS) as its communication layer. DDS is an industry standard for real-time, distributed systems used in aerospace, defense, and financial trading.

```
┌─────────────────────────────────────────────────────────┐
│                    Your Application                      │
├─────────────────────────────────────────────────────────┤
│                      ROS 2 API                          │
│              (rclpy, rclcpp, rcl)                       │
├─────────────────────────────────────────────────────────┤
│                    RMW Layer                            │
│           (ROS Middleware Interface)                    │
├─────────────────────────────────────────────────────────┤
│                   DDS Implementation                     │
│        (Fast DDS, Cyclone DDS, Connext)                 │
└─────────────────────────────────────────────────────────┘
```

**Key DDS concepts:**

- **Domain**: Logical network partition (default: 0)
- **Participant**: An application in the DDS network
- **Topic**: Named data channel with specific type
- **QoS**: Quality of Service policies for reliability, history, etc.

:::tip Default DDS
ROS 2 Humble uses Fast DDS by default. You can switch implementations by setting `RMW_IMPLEMENTATION` environment variable.
:::

## Installation

### Installing ROS 2 Humble

ROS 2 Humble Hawksbill is the current LTS release, supported until May 2027.

**Step 1: Set up sources**

```bash title="Terminal"
# Enable Universe repository
sudo apt install software-properties-common
sudo add-apt-repository universe

# Add ROS 2 GPG key
sudo apt update && sudo apt install curl -y
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg

# Add repository to sources list
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null
```

**Step 2: Install ROS 2**

```bash title="Terminal"
sudo apt update
sudo apt upgrade

# Desktop install (recommended) - includes rviz2, rqt, demos
sudo apt install ros-humble-desktop

# Or minimal install (no GUI tools)
# sudo apt install ros-humble-ros-base
```

**Step 3: Source the setup file**

```bash title="Terminal"
# Source ROS 2 for current terminal
source /opt/ros/humble/setup.bash

# Add to .bashrc for automatic sourcing
echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
```

**Step 4: Install development tools**

```bash title="Terminal"
sudo apt install python3-colcon-common-extensions python3-rosdep python3-vcstool
```

**Step 5: Initialize rosdep**

```bash title="Terminal"
sudo rosdep init
rosdep update
```

### Verify Installation

Test your installation with the demo nodes:

```bash title="Terminal 1"
ros2 run demo_nodes_cpp talker
```

```bash title="Terminal 2"
ros2 run demo_nodes_cpp listener
```

**Expected Output:**

```
[INFO] [talker]: Publishing: "Hello World: 0"
[INFO] [talker]: Publishing: "Hello World: 1"
...
```

```
[INFO] [listener]: I heard: [Hello World: 0]
[INFO] [listener]: I heard: [Hello World: 1]
...
```

If you see messages being published and received, congratulations! ROS 2 is working.

## Workspace Setup

### Understanding Workspaces

A ROS 2 workspace is a directory containing packages you're developing. The structure is:

```
ros2_ws/                    # Workspace root
├── src/                    # Source code (your packages)
│   ├── package_1/
│   └── package_2/
├── build/                  # Build artifacts (auto-generated)
├── install/                # Installed packages (auto-generated)
└── log/                    # Build logs (auto-generated)
```

### Create Your Workspace

```bash title="Terminal"
# Create workspace directory
mkdir -p ~/ros2_ws/src
cd ~/ros2_ws

# Build empty workspace (creates build, install, log)
colcon build

# Source the workspace
source install/setup.bash
```

:::warning Source Order Matters
Always source ROS 2 first (`/opt/ros/humble/setup.bash`), then your workspace (`install/setup.bash`). Your workspace overlays the base installation.
:::

### Create Your First Package

ROS 2 supports Python and C++ packages. Let's create a Python package:

```bash title="Terminal"
cd ~/ros2_ws/src

# Create package with dependencies
ros2 pkg create --build-type ament_python my_first_package \
    --dependencies rclpy std_msgs
```

This creates the following structure:

```
my_first_package/
├── my_first_package/
│   └── __init__.py
├── resource/
│   └── my_first_package
├── test/
│   ├── test_copyright.py
│   ├── test_flake8.py
│   └── test_pep257.py
├── package.xml              # Package metadata and dependencies
└── setup.py                 # Python package configuration
```

## Your First ROS 2 Node

Let's create a simple publisher node that sends messages.

**Step 1: Create the node file**

```python title="my_first_package/my_publisher.py"
import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class MinimalPublisher(Node):
    """A minimal ROS 2 publisher node."""

    def __init__(self):
        # Initialize the node with name 'minimal_publisher'
        super().__init__('minimal_publisher')

        # Create publisher on 'topic' with queue size 10
        self.publisher_ = self.create_publisher(String, 'topic', 10)

        # Create timer that calls callback every 0.5 seconds
        timer_period = 0.5  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.count = 0

        self.get_logger().info('Publisher node started')

    def timer_callback(self):
        """Publish a message on each timer tick."""
        msg = String()
        msg.data = f'Hello World: {self.count}'
        self.publisher_.publish(msg)
        self.get_logger().info(f'Publishing: "{msg.data}"')
        self.count += 1


def main(args=None):
    # Initialize ROS 2 communication
    rclpy.init(args=args)

    # Create and spin the node
    node = MinimalPublisher()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # Clean shutdown
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

**Step 2: Register the entry point**

Edit `setup.py` to add the console script:

```python title="setup.py (entry_points section)"
entry_points={
    'console_scripts': [
        'my_publisher = my_first_package.my_publisher:main',
    ],
},
```

**Step 3: Build and run**

```bash title="Terminal"
cd ~/ros2_ws
colcon build --packages-select my_first_package
source install/setup.bash
ros2 run my_first_package my_publisher
```

**Expected Output:**

```
[INFO] [minimal_publisher]: Publisher node started
[INFO] [minimal_publisher]: Publishing: "Hello World: 0"
[INFO] [minimal_publisher]: Publishing: "Hello World: 1"
[INFO] [minimal_publisher]: Publishing: "Hello World: 2"
```

## Essential CLI Tools

ROS 2 provides powerful command-line tools for introspection and debugging.

### Node Introspection

```bash title="Terminal"
# List all running nodes
ros2 node list

# Get info about a specific node
ros2 node info /minimal_publisher
```

### Topic Introspection

```bash title="Terminal"
# List all topics
ros2 topic list

# Show topic type
ros2 topic info /topic

# Echo messages on a topic
ros2 topic echo /topic

# Publish a message from CLI
ros2 topic pub /topic std_msgs/msg/String "data: 'Hello from CLI'"

# Check publishing rate
ros2 topic hz /topic
```

### Interface Inspection

```bash title="Terminal"
# List all message types
ros2 interface list

# Show message definition
ros2 interface show std_msgs/msg/String

# Show package interfaces
ros2 interface package std_msgs
```

### Running and Managing Nodes

```bash title="Terminal"
# Run a node
ros2 run <package_name> <executable_name>

# Run with parameters
ros2 run <package_name> <executable_name> --ros-args -p param_name:=value

# Run with remapping
ros2 run <package_name> <executable_name> --ros-args -r old_topic:=new_topic
```

## Quality of Service (QoS)

QoS policies control communication behavior. Key policies include:

| Policy | Options | Description |
|--------|---------|-------------|
| **Reliability** | RELIABLE / BEST_EFFORT | Guarantee delivery or drop |
| **Durability** | VOLATILE / TRANSIENT_LOCAL | Keep data for late joiners |
| **History** | KEEP_LAST(n) / KEEP_ALL | How much history to store |
| **Depth** | Integer | Queue size for KEEP_LAST |

### Common QoS Profiles

```python title="QoS Examples"
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

# Sensor data (best effort, small queue)
sensor_qos = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    history=HistoryPolicy.KEEP_LAST,
    depth=5
)

# Reliable communication (guaranteed delivery)
reliable_qos = QoSProfile(
    reliability=ReliabilityPolicy.RELIABLE,
    history=HistoryPolicy.KEEP_LAST,
    depth=10
)
```

:::warning QoS Compatibility
Publisher and subscriber QoS must be compatible. A RELIABLE subscriber cannot connect to a BEST_EFFORT publisher.
:::

## Environment Variables

Important ROS 2 environment variables:

| Variable | Purpose | Example |
|----------|---------|---------|
| `ROS_DOMAIN_ID` | Network partition | `export ROS_DOMAIN_ID=42` |
| `ROS_LOCALHOST_ONLY` | Disable network | `export ROS_LOCALHOST_ONLY=1` |
| `RMW_IMPLEMENTATION` | DDS selection | `export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp` |
| `RCUTILS_COLORIZED_OUTPUT` | Colored logs | `export RCUTILS_COLORIZED_OUTPUT=1` |

---

## Exercise 1: Create a Subscriber Node

:::tip Exercise 1: Create a Subscriber Node
**Objective**: Complete the publisher-subscriber pair by creating a subscriber.

**Time Estimate**: 20 minutes

**Steps**:

1. In the same package, create `my_subscriber.py`:

```python title="my_first_package/my_subscriber.py"
import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class MinimalSubscriber(Node):
    def __init__(self):
        super().__init__('minimal_subscriber')
        # TODO: Create a subscription to 'topic'
        # Hint: use self.create_subscription(String, 'topic', callback, 10)
        pass

    def listener_callback(self, msg):
        # TODO: Log the received message
        pass


def main(args=None):
    rclpy.init(args=args)
    node = MinimalSubscriber()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
```

2. Add the entry point to `setup.py`
3. Build with `colcon build`
4. Run publisher in one terminal, subscriber in another

**Expected Result**: Subscriber prints messages from publisher:
```
[INFO] [minimal_subscriber]: I heard: "Hello World: 0"
[INFO] [minimal_subscriber]: I heard: "Hello World: 1"
```

**Hints**:
- The callback receives a message object with `.data` attribute
- Use `self.get_logger().info()` for logging
:::

---

## Exercise 2: Explore with CLI Tools

:::tip Exercise 2: CLI Exploration
**Objective**: Use ROS 2 CLI tools to inspect the running system.

**Time Estimate**: 15 minutes

**Steps**:

1. Start the publisher node
2. In another terminal, use these commands:
   - `ros2 node list` - Find your node
   - `ros2 topic list` - Find the topic
   - `ros2 topic info /topic` - Check the message type
   - `ros2 topic echo /topic` - See live messages
   - `ros2 topic hz /topic` - Measure publish rate
3. Publish a message from CLI while subscriber is running

**Expected Result**: You understand how to inspect any ROS 2 system.

**Challenge**: Can you change the publish rate using `ros2 topic hz` to verify your timer is working correctly?
:::

---

## Summary

In this chapter, you learned:

- **ROS 2 architecture**: DDS-based middleware with decentralized discovery
- **Installation**: Setting up ROS 2 Humble on Ubuntu 22.04
- **Workspaces**: Creating and building packages with colcon
- **Node creation**: Writing Python nodes with publishers
- **CLI tools**: Introspecting nodes, topics, and messages
- **QoS**: Configuring communication reliability and history

These fundamentals are the building blocks for everything that follows. In the next chapter, you'll dive deeper into [Nodes, Topics & Services](/docs/module-1/ros2-nodes-topics-services) to master the core communication patterns.

## Further Reading

- [ROS 2 Documentation](https://docs.ros.org/en/humble/)
- [DDS Specification](https://www.omg.org/spec/DDS/)
- [ROS 2 Design Articles](https://design.ros2.org/)
- [colcon Documentation](https://colcon.readthedocs.io/)
