---
title: "Nodes, Topics & Services"
sidebar_position: 3
description: "Deep dive into ROS 2 communication patterns: nodes, topics for streaming data, and services for request-response."
---

# Nodes, Topics & Services

The node is the fundamental unit of computation in ROS 2, representing a single process that performs a specific function. This chapter explores how nodes communicate through topics (publish-subscribe pattern) and services (request-response pattern).

## Overview

In this section, you will:

- Understand node design principles and best practices
- Implement publishers and subscribers for streaming data
- Create services for request-response communication
- Work with custom message and service types
- Debug communication issues using introspection tools

## Prerequisites

Before starting, ensure you have completed [ROS 2 Fundamentals](/docs/module-1/ros2-fundamentals):

- Working ROS 2 Humble installation
- Familiarity with workspace and package creation
- Basic understanding of the ROS 2 architecture

## Nodes: The Building Blocks

### What is a Node?

A node is a single-purpose, modular process that performs a specific function. Nodes communicate with each other through well-defined interfaces, creating a distributed system that is:

- **Modular**: Each node handles one responsibility
- **Reusable**: Nodes can be used in different robot configurations
- **Testable**: Individual nodes can be tested in isolation
- **Fault-tolerant**: One node crashing doesn't bring down the whole system

### Node Design Principles

**Single Responsibility**: Each node should do one thing well.

```
✓ Good: CameraNode → reads camera, publishes images
✗ Bad:  RobotNode → reads camera, plans path, controls motors, logs data
```

**Loose Coupling**: Nodes communicate through topics/services, not direct function calls.

**Composability**: Nodes can be combined in different ways for different applications.

### Node Anatomy

Every ROS 2 Python node follows this structure:

```python title="node_template.py"
import rclpy
from rclpy.node import Node


class MyNode(Node):
    def __init__(self):
        super().__init__('my_node_name')
        # Initialize publishers, subscribers, timers, services
        self.get_logger().info('Node initialized')

    def some_callback(self):
        # Handle events
        pass


def main(args=None):
    rclpy.init(args=args)
    node = MyNode()
    try:
        rclpy.spin(node)  # Process callbacks until shutdown
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

### The Node Lifecycle

```
┌─────────────┐
│   Create    │ ─── rclpy.init() + Node()
└──────┬──────┘
       ▼
┌─────────────┐
│    Spin     │ ─── rclpy.spin() processes callbacks
└──────┬──────┘
       ▼
┌─────────────┐
│   Destroy   │ ─── node.destroy_node() + rclpy.shutdown()
└─────────────┘
```

---

## Topics: Streaming Data

Topics provide **publish-subscribe** communication, ideal for continuous data streams like sensor readings.

### How Topics Work

```
┌──────────┐          /camera/image          ┌──────────────┐
│  Camera  │ ────────────────────────────────▶│  Perception  │
│   Node   │  Publisher        Subscriber    │     Node     │
└──────────┘                                  └──────────────┘
                              │
                              ▼
                        ┌──────────────┐
                        │    Display   │
                        │     Node     │
                        └──────────────┘
```

**Key characteristics:**

- **Many-to-many**: Multiple publishers and subscribers per topic
- **Asynchronous**: Publishers don't wait for subscribers
- **Unidirectional**: Data flows one way
- **Typed**: Each topic has a specific message type

### Implementing a Publisher

```python title="sensor_publisher.py"
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Temperature


class TemperatureSensor(Node):
    """Publishes simulated temperature readings."""

    def __init__(self):
        super().__init__('temperature_sensor')

        # Create publisher
        # Arguments: message_type, topic_name, queue_size
        self.publisher = self.create_publisher(
            Temperature,
            'sensor/temperature',
            10
        )

        # Create timer for periodic publishing
        self.timer = self.create_timer(1.0, self.publish_temperature)
        self.temperature = 20.0

        self.get_logger().info('Temperature sensor started')

    def publish_temperature(self):
        """Publish current temperature reading."""
        msg = Temperature()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'sensor_frame'
        msg.temperature = self.temperature
        msg.variance = 0.5  # Measurement uncertainty

        self.publisher.publish(msg)
        self.get_logger().info(f'Published: {msg.temperature:.1f}°C')

        # Simulate temperature changes
        import random
        self.temperature += random.uniform(-0.5, 0.5)


def main(args=None):
    rclpy.init(args=args)
    node = TemperatureSensor()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
```

### Implementing a Subscriber

```python title="temperature_monitor.py"
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Temperature


class TemperatureMonitor(Node):
    """Subscribes to temperature readings and monitors for alerts."""

    def __init__(self):
        super().__init__('temperature_monitor')

        # Create subscription
        self.subscription = self.create_subscription(
            Temperature,
            'sensor/temperature',
            self.temperature_callback,
            10
        )

        self.alert_threshold = 25.0
        self.get_logger().info('Temperature monitor started')

    def temperature_callback(self, msg: Temperature):
        """Process incoming temperature readings."""
        temp = msg.temperature

        if temp > self.alert_threshold:
            self.get_logger().warn(f'HIGH TEMP ALERT: {temp:.1f}°C')
        else:
            self.get_logger().info(f'Temperature: {temp:.1f}°C')


def main(args=None):
    rclpy.init(args=args)
    node = TemperatureMonitor()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
```

**Expected Output:**

```
[INFO] [temperature_sensor]: Published: 20.3°C
[INFO] [temperature_monitor]: Temperature: 20.3°C
[INFO] [temperature_sensor]: Published: 20.8°C
[INFO] [temperature_monitor]: Temperature: 20.8°C
[WARN] [temperature_monitor]: HIGH TEMP ALERT: 25.2°C
```

### Common Message Types

ROS 2 provides standard message types in several packages:

| Package | Common Messages | Use Case |
|---------|----------------|----------|
| `std_msgs` | Bool, Int32, Float64, String | Simple data |
| `geometry_msgs` | Point, Pose, Twist, Transform | Spatial data |
| `sensor_msgs` | Image, LaserScan, Imu, PointCloud2 | Sensor data |
| `nav_msgs` | Odometry, Path, OccupancyGrid | Navigation |

```python title="Using geometry_msgs"
from geometry_msgs.msg import Twist

cmd = Twist()
cmd.linear.x = 1.0   # Forward velocity (m/s)
cmd.linear.y = 0.0
cmd.linear.z = 0.0
cmd.angular.x = 0.0
cmd.angular.y = 0.0
cmd.angular.z = 0.5  # Rotational velocity (rad/s)
```

### Topic Best Practices

:::tip Topic Naming Conventions
- Use lowercase with underscores: `/robot/cmd_vel`
- Include namespace for organization: `/sensor/lidar/points`
- Use standard names when applicable: `/cmd_vel`, `/odom`, `/scan`
:::

:::warning Avoid Common Pitfalls
- **Don't create subscribers in callbacks** - Creates memory leaks
- **Don't block in callbacks** - Use timers or separate threads
- **Mind your queue sizes** - Too small loses messages, too large uses memory
:::

---

## Services: Request-Response

Services provide **synchronous, request-response** communication, ideal for discrete operations like triggering behaviors or querying state.

### How Services Work

```
┌──────────┐    Request     ┌──────────────┐
│  Client  │ ─────────────▶ │    Server    │
│   Node   │                │     Node     │
│          │ ◀───────────── │              │
└──────────┘    Response    └──────────────┘
```

**Key characteristics:**

- **One-to-one**: Single server per service name
- **Synchronous**: Client waits for response
- **Bidirectional**: Request goes in, response comes back
- **Typed**: Request and response have specific types

### When to Use Services vs Topics

| Use Topics When... | Use Services When... |
|-------------------|---------------------|
| Continuous data streams | Discrete operations |
| Multiple consumers | Single request-response |
| Fire-and-forget | Need confirmation |
| High frequency data | Infrequent requests |

**Examples:**

- Topic: Camera images, laser scans, odometry
- Service: Take photo, calibrate sensor, get robot state

### Implementing a Service Server

```python title="robot_state_server.py"
import rclpy
from rclpy.node import Node
from std_srvs.srv import Trigger
from example_interfaces.srv import SetBool


class RobotStateServer(Node):
    """Provides services to query and control robot state."""

    def __init__(self):
        super().__init__('robot_state_server')

        # Robot state
        self.is_enabled = False
        self.battery_level = 85.0

        # Create services
        self.enable_srv = self.create_service(
            SetBool,
            'robot/enable',
            self.enable_callback
        )

        self.status_srv = self.create_service(
            Trigger,
            'robot/status',
            self.status_callback
        )

        self.get_logger().info('Robot state server ready')

    def enable_callback(self, request, response):
        """Handle enable/disable requests."""
        self.is_enabled = request.data
        state = 'ENABLED' if self.is_enabled else 'DISABLED'

        response.success = True
        response.message = f'Robot is now {state}'

        self.get_logger().info(f'Robot {state}')
        return response

    def status_callback(self, request, response):
        """Handle status requests."""
        state = 'enabled' if self.is_enabled else 'disabled'

        response.success = True
        response.message = f'Robot {state}, battery: {self.battery_level}%'

        return response


def main(args=None):
    rclpy.init(args=args)
    node = RobotStateServer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
```

### Implementing a Service Client

```python title="robot_controller.py"
import rclpy
from rclpy.node import Node
from std_srvs.srv import Trigger
from example_interfaces.srv import SetBool


class RobotController(Node):
    """Client that controls robot through services."""

    def __init__(self):
        super().__init__('robot_controller')

        # Create service clients
        self.enable_client = self.create_client(SetBool, 'robot/enable')
        self.status_client = self.create_client(Trigger, 'robot/status')

        # Wait for services to be available
        while not self.enable_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for robot/enable service...')

        self.get_logger().info('Robot controller ready')

    def enable_robot(self, enable: bool):
        """Send enable/disable request."""
        request = SetBool.Request()
        request.data = enable

        future = self.enable_client.call_async(request)
        rclpy.spin_until_future_complete(self, future)

        response = future.result()
        self.get_logger().info(f'Enable response: {response.message}')
        return response.success

    def get_status(self):
        """Query robot status."""
        request = Trigger.Request()

        future = self.status_client.call_async(request)
        rclpy.spin_until_future_complete(self, future)

        response = future.result()
        self.get_logger().info(f'Status: {response.message}')
        return response.message


def main(args=None):
    rclpy.init(args=args)
    controller = RobotController()

    # Example usage
    controller.get_status()
    controller.enable_robot(True)
    controller.get_status()
    controller.enable_robot(False)

    controller.destroy_node()
    rclpy.shutdown()
```

**Expected Output:**

```
[INFO] [robot_controller]: Robot controller ready
[INFO] [robot_controller]: Status: Robot disabled, battery: 85%
[INFO] [robot_state_server]: Robot ENABLED
[INFO] [robot_controller]: Enable response: Robot is now ENABLED
[INFO] [robot_controller]: Status: Robot enabled, battery: 85%
[INFO] [robot_state_server]: Robot DISABLED
[INFO] [robot_controller]: Enable response: Robot is now DISABLED
```

### Common Service Types

| Package | Service Types | Use Case |
|---------|--------------|----------|
| `std_srvs` | Trigger, SetBool, Empty | Simple operations |
| `example_interfaces` | AddTwoInts, SetBool | Examples/tutorials |
| `nav2_msgs` | GetCostmap, ClearCostmap | Navigation |
| `moveit_msgs` | GetPositionFK, GetPositionIK | Manipulation |

### CLI Service Tools

```bash title="Terminal"
# List all services
ros2 service list

# Show service type
ros2 service type /robot/enable

# Show service interface
ros2 interface show example_interfaces/srv/SetBool

# Call a service from CLI
ros2 service call /robot/enable example_interfaces/srv/SetBool "{data: true}"

# Call Trigger service (no request fields)
ros2 service call /robot/status std_srvs/srv/Trigger
```

---

## Custom Messages and Services

When standard types don't fit your needs, create custom interfaces.

### Creating a Custom Message

**Step 1: Create msg directory and file**

```
my_robot_interfaces/
├── msg/
│   └── RobotStatus.msg
├── srv/
├── CMakeLists.txt
└── package.xml
```

```text title="msg/RobotStatus.msg"
# Robot status message

std_msgs/Header header

# Robot identification
string robot_name
string robot_id

# State information
bool is_enabled
float32 battery_percentage
float32 cpu_temperature

# Position (optional)
geometry_msgs/Pose pose
```

**Step 2: Update package.xml**

```xml title="package.xml"
<buildtool_depend>rosidl_default_generators</buildtool_depend>
<exec_depend>rosidl_default_runtime</exec_depend>
<member_of_group>rosidl_interface_packages</member_of_group>

<depend>std_msgs</depend>
<depend>geometry_msgs</depend>
```

**Step 3: Update CMakeLists.txt**

```cmake title="CMakeLists.txt"
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/RobotStatus.msg"
  DEPENDENCIES std_msgs geometry_msgs
)
```

**Step 4: Build and use**

```bash title="Terminal"
cd ~/ros2_ws
colcon build --packages-select my_robot_interfaces
source install/setup.bash
```

```python title="Using custom message"
from my_robot_interfaces.msg import RobotStatus

msg = RobotStatus()
msg.robot_name = 'my_robot'
msg.is_enabled = True
msg.battery_percentage = 85.0
```

### Creating a Custom Service

```text title="srv/GetRobotStatus.srv"
# Request
string robot_id
---
# Response
bool success
string message
my_robot_interfaces/RobotStatus status
```

Update CMakeLists.txt to include the service:

```cmake
rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/RobotStatus.msg"
  "srv/GetRobotStatus.srv"
  DEPENDENCIES std_msgs geometry_msgs
)
```

---

## Combining Topics and Services

Real nodes often use both patterns. Here's an example that monitors sensors and provides a status service:

```python title="sensor_hub.py"
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Temperature, BatteryState
from std_srvs.srv import Trigger


class SensorHub(Node):
    """Aggregates sensor data and provides status service."""

    def __init__(self):
        super().__init__('sensor_hub')

        # Store latest readings
        self.temperature = None
        self.battery = None

        # Subscribe to sensors
        self.temp_sub = self.create_subscription(
            Temperature, 'sensor/temperature',
            self.temp_callback, 10
        )

        self.battery_sub = self.create_subscription(
            BatteryState, 'sensor/battery',
            self.battery_callback, 10
        )

        # Provide status service
        self.status_srv = self.create_service(
            Trigger, 'sensor_hub/status',
            self.status_callback
        )

        # Publish aggregated status
        self.status_pub = self.create_publisher(
            Temperature,  # Using for demo
            'sensor_hub/aggregated',
            10
        )

        self.get_logger().info('Sensor hub initialized')

    def temp_callback(self, msg):
        self.temperature = msg.temperature

    def battery_callback(self, msg):
        self.battery = msg.percentage

    def status_callback(self, request, response):
        temp_str = f'{self.temperature:.1f}°C' if self.temperature else 'N/A'
        bat_str = f'{self.battery:.0f}%' if self.battery else 'N/A'

        response.success = True
        response.message = f'Temp: {temp_str}, Battery: {bat_str}'
        return response


def main(args=None):
    rclpy.init(args=args)
    node = SensorHub()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
```

---

## Exercise 1: Build a Command System

:::tip Exercise 1: Robot Command System
**Objective**: Create a complete command system using topics and services.

**Time Estimate**: 45 minutes

**Requirements**:

1. **Command Publisher Node** (`command_sender.py`):
   - Publishes `geometry_msgs/Twist` commands to `/cmd_vel`
   - Use keyboard input or timer to send different commands
   - Forward, backward, rotate left, rotate right, stop

2. **Robot Simulator Node** (`robot_simulator.py`):
   - Subscribes to `/cmd_vel`
   - Maintains position (x, y, theta)
   - Provides service `/robot/get_pose` returning current pose
   - Logs received commands

**Steps**:

1. Create a new package: `ros2 pkg create --build-type ament_python robot_commands --dependencies rclpy geometry_msgs std_srvs`
2. Implement `command_sender.py` with timer-based commands
3. Implement `robot_simulator.py` with subscriber and service
4. Build and test with both nodes running

**Expected Behavior**:
- Command sender publishes velocity commands
- Robot simulator receives and logs commands
- Service call returns current robot pose

**Challenge**: Add a `/robot/reset_pose` service to reset the robot to origin.
:::

---

## Exercise 2: Multi-Sensor Aggregator

:::tip Exercise 2: Sensor Aggregation
**Objective**: Practice subscribing to multiple topics and providing service.

**Time Estimate**: 30 minutes

**Requirements**:

Create a `sensor_aggregator` node that:
1. Subscribes to three topics:
   - `/sensor/temperature` (Temperature)
   - `/sensor/humidity` (Float32)
   - `/sensor/pressure` (Float32)
2. Provides a `/sensors/status` service (Trigger)
3. The service returns all current readings in the message

**Starter Code**:

```python
# Create fake sensor publishers for testing
# In one terminal:
ros2 topic pub /sensor/temperature sensor_msgs/msg/Temperature "{temperature: 25.0}"
ros2 topic pub /sensor/humidity std_msgs/msg/Float32 "{data: 65.0}"
ros2 topic pub /sensor/pressure std_msgs/msg/Float32 "{data: 1013.25}"
```

**Expected Service Response**:
```
success: True
message: "Temp: 25.0°C, Humidity: 65.0%, Pressure: 1013.25 hPa"
```
:::

---

## Debugging Communication

### Common Issues and Solutions

| Problem | Possible Cause | Solution |
|---------|---------------|----------|
| No messages received | Wrong topic name | Check with `ros2 topic list` |
| Subscriber not connecting | QoS mismatch | Match publisher QoS settings |
| Service times out | Server not running | Check with `ros2 service list` |
| Messages delayed | Callback blocking | Move heavy work to separate thread |

### Visualization Tools

```bash title="Terminal"
# View node graph
ros2 run rqt_graph rqt_graph

# Monitor topics in real-time
ros2 run rqt_topic rqt_topic

# Service caller GUI
ros2 run rqt_service_caller rqt_service_caller
```

---

## Summary

In this chapter, you learned:

- **Nodes**: Single-purpose processes that form the building blocks of ROS 2 applications
- **Topics**: Publish-subscribe pattern for streaming data (sensors, commands)
- **Services**: Request-response pattern for discrete operations (queries, triggers)
- **Custom interfaces**: Creating message and service types for your application
- **Design patterns**: When to use topics vs services, naming conventions

These communication patterns are the foundation of all ROS 2 applications. In the next chapter, you'll learn about [Actions and Lifecycle Nodes](/docs/module-1/ros2-actions-lifecycle) for handling long-running tasks and managed node states.

## Further Reading

- [ROS 2 Concepts: Topics](https://docs.ros.org/en/humble/Concepts/About-Topics.html)
- [ROS 2 Concepts: Services](https://docs.ros.org/en/humble/Concepts/About-Services.html)
- [Writing a Simple Publisher and Subscriber](https://docs.ros.org/en/humble/Tutorials/Writing-A-Simple-Py-Publisher-And-Subscriber.html)
- [Creating Custom Messages](https://docs.ros.org/en/humble/Tutorials/Custom-ROS2-Interfaces.html)
