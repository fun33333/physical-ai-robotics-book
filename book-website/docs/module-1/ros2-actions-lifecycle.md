---
title: "Actions & Lifecycle Nodes"
sidebar_position: 4
description: "Advanced ROS 2 patterns: actions for long-running tasks and lifecycle nodes for managed state transitions."
---

# Actions & Lifecycle Nodes

While topics and services cover most communication needs, some operations require more sophisticated patterns. Actions provide goal-based, preemptable, long-running task execution with feedback. Lifecycle nodes add state management for graceful startup, shutdown, and error recovery.

## Overview

In this section, you will:

- Understand when to use actions vs topics/services
- Implement action servers and clients
- Handle goal preemption and cancellation
- Create lifecycle-managed nodes
- Compose robust systems with proper initialization order

## Prerequisites

Before starting, ensure you have completed:

- [ROS 2 Fundamentals](/docs/module-1/ros2-fundamentals)
- [Nodes, Topics & Services](/docs/module-1/ros2-nodes-topics-services)

---

## Actions: Long-Running Tasks

### Why Actions?

Topics and services have limitations for certain operations:

| Pattern | Limitation for Long Tasks |
|---------|--------------------------|
| Topics | No confirmation, no progress feedback |
| Services | Blocking - client waits for entire duration |

**Actions solve this with:**

- **Goal**: What the robot should achieve
- **Feedback**: Progress updates during execution
- **Result**: Final outcome when complete
- **Preemption**: Ability to cancel mid-execution

### Action Architecture

```
┌─────────────┐                              ┌─────────────┐
│   Action    │  ─────── Goal ─────────────▶ │   Action    │
│   Client    │  ◀────── Feedback ────────── │   Server    │
│             │  ◀────── Result ──────────── │             │
│             │  ─────── Cancel ───────────▶ │             │
└─────────────┘                              └─────────────┘
```

### When to Use Actions

| Use Actions For | Don't Use Actions For |
|-----------------|----------------------|
| Navigation goals (go to pose) | Continuous sensor data |
| Manipulation tasks (pick object) | Simple state queries |
| Calibration procedures | Toggle operations |
| Any task > 1 second | Immediate responses |

### Common Action Types

| Package | Action | Description |
|---------|--------|-------------|
| `action_tutorials_interfaces` | Fibonacci | Tutorial example |
| `nav2_msgs` | NavigateToPose | Navigation goal |
| `moveit_msgs` | MoveGroup | Manipulation goal |
| `control_msgs` | FollowJointTrajectory | Joint motion |

---

## Implementing an Action Server

Let's create an action server for a "count to N" task with progress feedback.

### Define the Action Interface

First, create a custom action (or use existing ones):

```text title="action/CountTo.action"
# Goal: target number to count to
int32 target_number
---
# Result: final count and time taken
int32 final_count
float32 time_elapsed
---
# Feedback: current progress
int32 current_count
float32 percent_complete
```

### Action Server Implementation

```python title="count_action_server.py"
import time
import rclpy
from rclpy.action import ActionServer, GoalResponse, CancelResponse
from rclpy.action.server import ServerGoalHandle
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node

from action_tutorials_interfaces.action import Fibonacci


class CountActionServer(Node):
    """Action server that counts to a target number with feedback."""

    def __init__(self):
        super().__init__('count_action_server')

        # Create action server with callbacks
        self._action_server = ActionServer(
            self,
            Fibonacci,  # Using Fibonacci action for demo
            'count_to',
            execute_callback=self.execute_callback,
            goal_callback=self.goal_callback,
            cancel_callback=self.cancel_callback,
            callback_group=ReentrantCallbackGroup()
        )

        self.get_logger().info('Count action server ready')

    def goal_callback(self, goal_request):
        """Accept or reject incoming goal."""
        target = goal_request.order
        self.get_logger().info(f'Received goal: count to {target}')

        # Validate goal
        if target < 0:
            self.get_logger().warn('Rejecting negative target')
            return GoalResponse.REJECT

        if target > 100:
            self.get_logger().warn('Rejecting target > 100')
            return GoalResponse.REJECT

        return GoalResponse.ACCEPT

    def cancel_callback(self, goal_handle):
        """Handle cancellation requests."""
        self.get_logger().info('Received cancel request')
        return CancelResponse.ACCEPT

    async def execute_callback(self, goal_handle: ServerGoalHandle):
        """Execute the counting task."""
        self.get_logger().info('Executing goal...')

        target = goal_handle.request.order
        feedback_msg = Fibonacci.Feedback()
        result = Fibonacci.Result()

        # Initialize sequence for Fibonacci example
        feedback_msg.partial_sequence = [0, 1]

        start_time = time.time()

        for i in range(1, target):
            # Check for cancellation
            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                self.get_logger().info('Goal canceled')
                result.sequence = feedback_msg.partial_sequence
                return result

            # Simulate work
            time.sleep(0.5)

            # Update feedback
            feedback_msg.partial_sequence.append(
                feedback_msg.partial_sequence[-1] +
                feedback_msg.partial_sequence[-2]
            )

            self.get_logger().info(f'Progress: {i}/{target}')
            goal_handle.publish_feedback(feedback_msg)

        # Complete successfully
        goal_handle.succeed()

        result.sequence = feedback_msg.partial_sequence
        elapsed = time.time() - start_time
        self.get_logger().info(f'Goal completed in {elapsed:.1f}s')

        return result


def main(args=None):
    rclpy.init(args=args)
    node = CountActionServer()

    # Use multi-threaded executor for concurrent goal handling
    executor = MultiThreadedExecutor()
    executor.add_node(node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

---

## Implementing an Action Client

```python title="count_action_client.py"
import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node

from action_tutorials_interfaces.action import Fibonacci


class CountActionClient(Node):
    """Action client that sends counting goals."""

    def __init__(self):
        super().__init__('count_action_client')
        self._action_client = ActionClient(
            self,
            Fibonacci,
            'count_to'
        )
        self.get_logger().info('Count action client ready')

    def send_goal(self, target: int):
        """Send a counting goal."""
        self.get_logger().info(f'Waiting for action server...')
        self._action_client.wait_for_server()

        goal_msg = Fibonacci.Goal()
        goal_msg.order = target

        self.get_logger().info(f'Sending goal: count to {target}')

        # Send goal with callbacks
        self._send_goal_future = self._action_client.send_goal_async(
            goal_msg,
            feedback_callback=self.feedback_callback
        )
        self._send_goal_future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        """Handle goal acceptance/rejection."""
        goal_handle = future.result()

        if not goal_handle.accepted:
            self.get_logger().warn('Goal rejected')
            return

        self.get_logger().info('Goal accepted')

        # Wait for result
        self._get_result_future = goal_handle.get_result_async()
        self._get_result_future.add_done_callback(self.result_callback)

    def feedback_callback(self, feedback_msg):
        """Handle feedback during execution."""
        sequence = feedback_msg.feedback.partial_sequence
        self.get_logger().info(f'Feedback: {sequence}')

    def result_callback(self, future):
        """Handle final result."""
        result = future.result().result
        self.get_logger().info(f'Result: {result.sequence}')
        rclpy.shutdown()


def main(args=None):
    rclpy.init(args=args)
    client = CountActionClient()

    # Send goal
    client.send_goal(10)

    rclpy.spin(client)


if __name__ == '__main__':
    main()
```

**Expected Output:**

```
[INFO] [count_action_server]: Received goal: count to 10
[INFO] [count_action_client]: Goal accepted
[INFO] [count_action_server]: Progress: 1/10
[INFO] [count_action_client]: Feedback: [0, 1, 1]
[INFO] [count_action_server]: Progress: 2/10
[INFO] [count_action_client]: Feedback: [0, 1, 1, 2]
...
[INFO] [count_action_server]: Goal completed in 4.5s
[INFO] [count_action_client]: Result: [0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55]
```

### Canceling Goals

```python title="cancel_goal.py"
def cancel_goal(self, goal_handle):
    """Cancel an active goal."""
    self.get_logger().info('Canceling goal...')
    future = goal_handle.cancel_goal_async()
    future.add_done_callback(self.cancel_done_callback)

def cancel_done_callback(self, future):
    cancel_response = future.result()
    if len(cancel_response.goals_canceling) > 0:
        self.get_logger().info('Goal canceled successfully')
    else:
        self.get_logger().warn('Goal cancellation failed')
```

### CLI Action Tools

```bash title="Terminal"
# List actions
ros2 action list

# Show action info
ros2 action info /count_to

# Send goal from CLI
ros2 action send_goal /count_to action_tutorials_interfaces/action/Fibonacci "{order: 5}"

# Send goal with feedback display
ros2 action send_goal /count_to action_tutorials_interfaces/action/Fibonacci "{order: 5}" --feedback
```

---

## Lifecycle Nodes

### Why Lifecycle Nodes?

Standard nodes have a simple lifecycle: create → spin → destroy. This causes problems:

- **No initialization order**: Nodes start randomly
- **No graceful shutdown**: Abrupt termination
- **No error recovery**: Crash and restart is the only option

**Lifecycle nodes add managed states:**

```
┌─────────────────────────────────────────────────────────────┐
│                    LIFECYCLE STATES                          │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   [Unconfigured] ──configure──▶ [Inactive]                  │
│        ▲                           │                         │
│        │                      activate                       │
│     cleanup                        ▼                         │
│        │                       [Active]                      │
│        │                           │                         │
│        └──────── deactivate ◀──────┘                        │
│                                                              │
│   Any state can transition to [Finalized] via shutdown       │
└─────────────────────────────────────────────────────────────┘
```

### Lifecycle States

| State | Description |
|-------|-------------|
| **Unconfigured** | Node created but not configured |
| **Inactive** | Configured but not processing |
| **Active** | Fully operational |
| **Finalized** | Shutting down |

### Lifecycle Transitions

| Transition | From → To | Purpose |
|------------|-----------|---------|
| `configure` | Unconfigured → Inactive | Load parameters, setup |
| `activate` | Inactive → Active | Start processing |
| `deactivate` | Active → Inactive | Pause processing |
| `cleanup` | Inactive → Unconfigured | Release resources |
| `shutdown` | Any → Finalized | Terminate node |

---

## Implementing a Lifecycle Node

```python title="lifecycle_camera.py"
import rclpy
from rclpy.lifecycle import Node as LifecycleNode
from rclpy.lifecycle import State, TransitionCallbackReturn
from sensor_msgs.msg import Image
from std_msgs.msg import Header


class LifecycleCamera(LifecycleNode):
    """Lifecycle-managed camera node."""

    def __init__(self, node_name='lifecycle_camera'):
        super().__init__(node_name)
        self.get_logger().info('Node created (Unconfigured)')

        # Declare but don't create resources yet
        self._publisher = None
        self._timer = None
        self._frame_count = 0

    def on_configure(self, state: State) -> TransitionCallbackReturn:
        """
        Configure callback - load parameters, create publishers.
        Called when transitioning from Unconfigured to Inactive.
        """
        self.get_logger().info('Configuring...')

        # Declare and get parameters
        self.declare_parameter('frame_rate', 30.0)
        self.declare_parameter('resolution', [640, 480])

        self._frame_rate = self.get_parameter('frame_rate').value
        self._resolution = self.get_parameter('resolution').value

        # Create publisher (but don't publish yet)
        self._publisher = self.create_lifecycle_publisher(
            Image,
            'camera/image',
            10
        )

        self.get_logger().info(
            f'Configured: {self._resolution[0]}x{self._resolution[1]} @ {self._frame_rate}fps'
        )
        return TransitionCallbackReturn.SUCCESS

    def on_activate(self, state: State) -> TransitionCallbackReturn:
        """
        Activate callback - start processing.
        Called when transitioning from Inactive to Active.
        """
        self.get_logger().info('Activating...')

        # Create timer to publish images
        timer_period = 1.0 / self._frame_rate
        self._timer = self.create_timer(timer_period, self.publish_image)

        self.get_logger().info('Camera activated - publishing images')
        return super().on_activate(state)

    def on_deactivate(self, state: State) -> TransitionCallbackReturn:
        """
        Deactivate callback - stop processing but keep configuration.
        Called when transitioning from Active to Inactive.
        """
        self.get_logger().info('Deactivating...')

        # Stop timer but keep publisher
        if self._timer:
            self._timer.cancel()
            self._timer = None

        self.get_logger().info('Camera deactivated - stopped publishing')
        return super().on_deactivate(state)

    def on_cleanup(self, state: State) -> TransitionCallbackReturn:
        """
        Cleanup callback - release all resources.
        Called when transitioning from Inactive to Unconfigured.
        """
        self.get_logger().info('Cleaning up...')

        # Destroy publisher
        if self._publisher:
            self.destroy_publisher(self._publisher)
            self._publisher = None

        self._frame_count = 0
        self.get_logger().info('Cleanup complete')
        return TransitionCallbackReturn.SUCCESS

    def on_shutdown(self, state: State) -> TransitionCallbackReturn:
        """
        Shutdown callback - final cleanup before destruction.
        Can be called from any state.
        """
        self.get_logger().info('Shutting down...')

        # Ensure all resources released
        if self._timer:
            self._timer.cancel()
        if self._publisher:
            self.destroy_publisher(self._publisher)

        self.get_logger().info('Shutdown complete')
        return TransitionCallbackReturn.SUCCESS

    def publish_image(self):
        """Publish camera frame (simulated)."""
        if self._publisher is None or not self._publisher.is_activated:
            return

        msg = Image()
        msg.header = Header()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'camera_link'
        msg.height = self._resolution[1]
        msg.width = self._resolution[0]
        msg.encoding = 'rgb8'
        msg.step = self._resolution[0] * 3
        msg.data = bytes(self._resolution[0] * self._resolution[1] * 3)

        self._publisher.publish(msg)
        self._frame_count += 1

        if self._frame_count % 30 == 0:
            self.get_logger().info(f'Published frame {self._frame_count}')


def main(args=None):
    rclpy.init(args=args)
    node = LifecycleCamera()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

### Managing Lifecycle from CLI

```bash title="Terminal"
# Check current state
ros2 lifecycle get /lifecycle_camera

# Trigger transitions
ros2 lifecycle set /lifecycle_camera configure
ros2 lifecycle set /lifecycle_camera activate
ros2 lifecycle set /lifecycle_camera deactivate
ros2 lifecycle set /lifecycle_camera cleanup
ros2 lifecycle set /lifecycle_camera shutdown

# List available transitions
ros2 lifecycle list /lifecycle_camera
```

### Managing Lifecycle Programmatically

```python title="lifecycle_manager.py"
import rclpy
from rclpy.node import Node
from lifecycle_msgs.srv import ChangeState, GetState
from lifecycle_msgs.msg import Transition


class LifecycleManager(Node):
    """Manages lifecycle of other nodes."""

    def __init__(self):
        super().__init__('lifecycle_manager')

        # Create service clients for managed node
        self.change_state_client = self.create_client(
            ChangeState,
            '/lifecycle_camera/change_state'
        )
        self.get_state_client = self.create_client(
            GetState,
            '/lifecycle_camera/get_state'
        )

    def configure(self):
        """Transition to configured state."""
        return self.change_state(Transition.TRANSITION_CONFIGURE)

    def activate(self):
        """Transition to active state."""
        return self.change_state(Transition.TRANSITION_ACTIVATE)

    def deactivate(self):
        """Transition to inactive state."""
        return self.change_state(Transition.TRANSITION_DEACTIVATE)

    def change_state(self, transition_id: int) -> bool:
        """Send state change request."""
        if not self.change_state_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().error('Service not available')
            return False

        request = ChangeState.Request()
        request.transition.id = transition_id

        future = self.change_state_client.call_async(request)
        rclpy.spin_until_future_complete(self, future)

        return future.result().success
```

---

## Composing Lifecycle Systems

### Launch File with Lifecycle Management

```python title="lifecycle_launch.py"
from launch import LaunchDescription
from launch_ros.actions import LifecycleNode
from launch_ros.events.lifecycle import ChangeState
from launch.actions import EmitEvent, RegisterEventHandler
from launch.event_handlers import OnProcessStart
from lifecycle_msgs.msg import Transition


def generate_launch_description():
    # Create lifecycle node
    camera_node = LifecycleNode(
        package='my_camera_pkg',
        executable='lifecycle_camera',
        name='camera',
        namespace='',
        output='screen'
    )

    # Auto-configure on start
    configure_event = EmitEvent(
        event=ChangeState(
            lifecycle_node_matcher=lambda node: node == camera_node,
            transition_id=Transition.TRANSITION_CONFIGURE
        )
    )

    # Auto-activate after configure
    activate_event = EmitEvent(
        event=ChangeState(
            lifecycle_node_matcher=lambda node: node == camera_node,
            transition_id=Transition.TRANSITION_ACTIVATE
        )
    )

    return LaunchDescription([
        camera_node,
        RegisterEventHandler(
            OnProcessStart(
                target_action=camera_node,
                on_start=[configure_event, activate_event]
            )
        )
    ])
```

### System Startup Order

For complex systems, use lifecycle to ensure proper initialization:

```
1. Sensors configure → activate
2. Perception nodes configure → wait for sensors → activate
3. Planning nodes configure → wait for perception → activate
4. Control nodes configure → wait for planning → activate
```

---

## Exercise 1: Navigation Action Client

:::tip Exercise 1: Navigation Action
**Objective**: Create an action client for simulated navigation.

**Time Estimate**: 40 minutes

**Requirements**:

1. **Action Server** (`nav_server.py`):
   - Accept NavigateToPose goals (or use Fibonacci for simplicity)
   - Simulate movement by incrementally updating position
   - Publish feedback with current position and distance remaining
   - Support cancellation

2. **Action Client** (`nav_client.py`):
   - Send goal position
   - Display feedback updates
   - Ability to cancel mid-navigation

**Steps**:

1. Create package with action dependencies
2. Implement server with simulated movement loop
3. Implement client with feedback display
4. Test navigation with cancellation

**Expected Behavior**:
- Client sends goal (x: 10, y: 5)
- Server publishes feedback: "Position: (2, 1), Distance: 9.2m"
- On cancel: Server stops, returns partial result

**Challenge**: Add obstacle detection that pauses navigation and sends special feedback.
:::

---

## Exercise 2: Lifecycle Sensor Node

:::tip Exercise 2: Managed Sensor
**Objective**: Create a lifecycle-managed sensor with proper state handling.

**Time Estimate**: 35 minutes

**Requirements**:

Create `lifecycle_lidar.py` that:
1. **Unconfigured**: Node exists but no resources
2. **Configure**: Declare parameters (scan_rate, range_max), create publisher
3. **Activate**: Start publishing fake LaserScan messages
4. **Deactivate**: Stop publishing, keep configuration
5. **Cleanup**: Release all resources

**Test Procedure**:
```bash
# Terminal 1: Start node
ros2 run my_pkg lifecycle_lidar

# Terminal 2: Manage lifecycle
ros2 lifecycle set /lifecycle_lidar configure
ros2 topic list  # Should see /scan topic
ros2 lifecycle set /lifecycle_lidar activate
ros2 topic echo /scan  # Should see messages
ros2 lifecycle set /lifecycle_lidar deactivate
ros2 topic echo /scan  # Should stop
```

**Expected Behavior**:
- Only publishes when Active
- Parameters only readable after Configure
- Clean state transitions with logging
:::

---

## Best Practices

### Action Design

- **Keep goals simple**: One clear objective per action
- **Provide useful feedback**: Progress percentage, estimated time, current state
- **Handle cancellation gracefully**: Clean up resources, return partial results
- **Set reasonable timeouts**: Prevent stuck goals

### Lifecycle Design

- **Fail fast in configure**: Validate all parameters early
- **Keep activate/deactivate fast**: Heavy work in configure
- **Clean up completely**: No resource leaks between configure cycles
- **Log state transitions**: Essential for debugging

### Error Handling

```python title="Error handling example"
def on_configure(self, state: State) -> TransitionCallbackReturn:
    try:
        # Attempt configuration
        self.setup_hardware()
        return TransitionCallbackReturn.SUCCESS
    except HardwareError as e:
        self.get_logger().error(f'Configuration failed: {e}')
        return TransitionCallbackReturn.FAILURE
    except Exception as e:
        self.get_logger().error(f'Unexpected error: {e}')
        return TransitionCallbackReturn.ERROR
```

---

## Summary

In this chapter, you learned:

- **Actions**: Goal-based communication with feedback and cancellation for long-running tasks
- **Action servers**: Implementing task execution with progress updates
- **Action clients**: Sending goals and handling feedback/results
- **Lifecycle nodes**: Managed state transitions for robust systems
- **System composition**: Coordinating startup order and dependencies

These advanced patterns are essential for building production-ready robotic systems. In the next chapter, you'll apply these concepts to [Navigation with Nav2](/docs/module-1/ros2-navigation).

## Further Reading

- [ROS 2 Actions Tutorial](https://docs.ros.org/en/humble/Tutorials/Intermediate/Writing-an-Action-Server-Client/Py.html)
- [Managed Nodes Design](https://design.ros2.org/articles/node_lifecycle.html)
- [Navigation2 Action Servers](https://navigation.ros.org/concepts/index.html)
- [Lifecycle Node Documentation](https://docs.ros.org/en/humble/Concepts/About-Managed-Nodes.html)
