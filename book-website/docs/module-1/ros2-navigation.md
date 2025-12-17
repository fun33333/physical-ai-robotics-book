---
title: "Navigation Stack (Nav2)"
sidebar_position: 5
description: "Autonomous mobile robot navigation using Nav2: localization, path planning, and obstacle avoidance."
---

# Navigation Stack (Nav2)

Nav2 is the premier navigation framework for ROS 2, providing a complete solution for autonomous mobile robot navigation. This chapter covers the architecture and components needed to make a robot navigate from point A to point B while avoiding obstacles.

## Overview

In this section, you will:

- Understand the Nav2 architecture and component interactions
- Configure localization with AMCL
- Set up costmaps for obstacle representation
- Use global and local path planners
- Control navigation with behavior trees
- Send navigation goals programmatically

## Prerequisites

Before starting, ensure you have completed:

- [ROS 2 Fundamentals](/docs/module-1/ros2-fundamentals)
- [Actions & Lifecycle Nodes](/docs/module-1/ros2-actions-lifecycle)

Additionally, you need:

```bash title="Install Nav2"
sudo apt install ros-humble-navigation2 ros-humble-nav2-bringup
sudo apt install ros-humble-turtlebot3-gazebo ros-humble-turtlebot3-navigation2
```

---

## Nav2 Architecture

### System Overview

Nav2 is a modular framework composed of lifecycle-managed nodes communicating via actions, topics, and services:

```
┌─────────────────────────────────────────────────────────────────┐
│                        BT Navigator                              │
│                    (Behavior Tree Engine)                        │
└────────────────────────────┬────────────────────────────────────┘
                             │ Actions
         ┌───────────────────┼───────────────────┐
         ▼                   ▼                   ▼
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│   Controller    │  │     Planner     │  │    Recovery     │
│    Server       │  │     Server      │  │    Server       │
│  (Local Plan)   │  │  (Global Plan)  │  │   (Behaviors)   │
└────────┬────────┘  └────────┬────────┘  └────────┬────────┘
         │                    │                    │
         ▼                    ▼                    ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Costmap 2D                                  │
│              (Global + Local Costmaps)                          │
└─────────────────────────────────────────────────────────────────┘
         ▲                    ▲
         │                    │
┌─────────────────┐  ┌─────────────────┐
│   Localization  │  │     Sensors     │
│  (AMCL / SLAM)  │  │  (LaserScan)    │
└─────────────────┘  └─────────────────┘
```

### Key Components

| Component | Purpose | Default Plugin |
|-----------|---------|----------------|
| **BT Navigator** | Orchestrates navigation behavior | NavigateToPose BT |
| **Planner Server** | Computes global path | NavFn Planner |
| **Controller Server** | Follows path locally | DWB Controller |
| **Costmap 2D** | Represents obstacles | Voxel Layer |
| **AMCL** | Localizes robot on map | Adaptive MCL |
| **Recovery Server** | Handles failures | Spin, Backup, Wait |

### Data Flow

1. **Goal received** → BT Navigator accepts NavigateToPose action
2. **Global planning** → Planner Server computes path on global costmap
3. **Local control** → Controller Server follows path on local costmap
4. **Feedback loop** → Position updates from localization
5. **Recovery** → If stuck, recovery behaviors attempt to unstick

---

## Localization with AMCL

### What is Localization?

Localization answers: "Where am I on the map?" AMCL (Adaptive Monte Carlo Localization) uses particle filters to estimate robot pose.

```
┌─────────────┐     LaserScan      ┌─────────────┐
│   LiDAR     │ ─────────────────▶ │    AMCL     │
└─────────────┘                    └──────┬──────┘
                                          │
┌─────────────┐     Odometry              │  Pose +
│  Encoders   │ ─────────────────▶        │  Particles
└─────────────┘                           ▼
                                   ┌─────────────┐
┌─────────────┐      Map           │   TF Tree   │
│  Map Server │ ─────────────────▶ │  map→odom   │
└─────────────┘                    └─────────────┘
```

### AMCL Parameters

```yaml title="amcl_params.yaml"
amcl:
  ros__parameters:
    use_sim_time: true

    # Particle filter settings
    min_particles: 500
    max_particles: 2000

    # Motion model (differential drive)
    robot_model_type: "nav2_amcl::DifferentialMotionModel"
    alpha1: 0.2   # Rotation noise from rotation
    alpha2: 0.2   # Rotation noise from translation
    alpha3: 0.2   # Translation noise from translation
    alpha4: 0.2   # Translation noise from rotation

    # Laser model
    laser_model_type: "likelihood_field"
    laser_max_range: 12.0
    laser_min_range: 0.1
    max_beams: 60

    # Update thresholds
    update_min_a: 0.2    # Radians before update
    update_min_d: 0.25   # Meters before update

    # Initial pose (optional)
    set_initial_pose: true
    initial_pose:
      x: 0.0
      y: 0.0
      yaw: 0.0
```

### Setting Initial Pose

```python title="set_initial_pose.py"
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseWithCovarianceStamped


class InitialPoseSetter(Node):
    def __init__(self):
        super().__init__('initial_pose_setter')
        self.publisher = self.create_publisher(
            PoseWithCovarianceStamped,
            '/initialpose',
            10
        )

    def set_pose(self, x: float, y: float, yaw: float):
        """Set robot's initial pose estimate."""
        import math
        msg = PoseWithCovarianceStamped()
        msg.header.frame_id = 'map'
        msg.header.stamp = self.get_clock().now().to_msg()

        msg.pose.pose.position.x = x
        msg.pose.pose.position.y = y
        msg.pose.pose.position.z = 0.0

        # Convert yaw to quaternion
        msg.pose.pose.orientation.z = math.sin(yaw / 2.0)
        msg.pose.pose.orientation.w = math.cos(yaw / 2.0)

        # Covariance (uncertainty)
        msg.pose.covariance[0] = 0.25   # x variance
        msg.pose.covariance[7] = 0.25   # y variance
        msg.pose.covariance[35] = 0.07  # yaw variance

        self.publisher.publish(msg)
        self.get_logger().info(f'Set initial pose: ({x}, {y}, {yaw})')
```

---

## Costmaps

### Understanding Costmaps

Costmaps represent the world as a grid where each cell has a cost value:

| Cost Value | Meaning |
|------------|---------|
| 0 | Free space |
| 1-252 | Inflation (proximity to obstacles) |
| 253 | Inscribed obstacle (robot would touch) |
| 254 | Lethal obstacle (collision) |
| 255 | Unknown |

### Global vs Local Costmaps

| Costmap | Purpose | Size | Update Rate |
|---------|---------|------|-------------|
| **Global** | Planning full path | Full map | Slow (when map changes) |
| **Local** | Immediate obstacle avoidance | Robot-centered window | Fast (sensor rate) |

### Costmap Configuration

```yaml title="costmap_params.yaml"
global_costmap:
  global_costmap:
    ros__parameters:
      update_frequency: 1.0
      publish_frequency: 1.0
      global_frame: map
      robot_base_frame: base_link

      resolution: 0.05  # 5cm per cell

      plugins: ["static_layer", "obstacle_layer", "inflation_layer"]

      static_layer:
        plugin: "nav2_costmap_2d::StaticLayer"
        map_subscribe_transient_local: true

      obstacle_layer:
        plugin: "nav2_costmap_2d::ObstacleLayer"
        enabled: true
        observation_sources: scan
        scan:
          topic: /scan
          max_obstacle_height: 2.0
          clearing: true
          marking: true
          data_type: "LaserScan"

      inflation_layer:
        plugin: "nav2_costmap_2d::InflationLayer"
        cost_scaling_factor: 3.0
        inflation_radius: 0.55  # Robot radius + safety margin

local_costmap:
  local_costmap:
    ros__parameters:
      update_frequency: 5.0
      publish_frequency: 2.0
      global_frame: odom
      robot_base_frame: base_link

      rolling_window: true
      width: 3
      height: 3
      resolution: 0.05

      plugins: ["voxel_layer", "inflation_layer"]

      voxel_layer:
        plugin: "nav2_costmap_2d::VoxelLayer"
        enabled: true
        observation_sources: scan
        scan:
          topic: /scan
          max_obstacle_height: 2.0
          clearing: true
          marking: true
```

---

## Path Planning

### Global Planner

The global planner computes a path from start to goal on the global costmap.

**NavFn Planner** (Dijkstra/A*):

```yaml title="planner_params.yaml"
planner_server:
  ros__parameters:
    planner_plugins: ["GridBased"]

    GridBased:
      plugin: "nav2_navfn_planner/NavfnPlanner"
      tolerance: 0.5           # Goal tolerance
      use_astar: true          # A* vs Dijkstra
      allow_unknown: true      # Plan through unknown space
```

**Smac Planners** (lattice-based, better for non-circular robots):

```yaml
GridBased:
  plugin: "nav2_smac_planner/SmacPlannerHybrid"
  minimum_turning_radius: 0.4
  motion_model_for_search: "DUBIN"
```

### Local Controller

The controller follows the global path while reacting to local obstacles.

**DWB Controller** (Dynamic Window Approach):

```yaml title="controller_params.yaml"
controller_server:
  ros__parameters:
    controller_frequency: 20.0

    controller_plugins: ["FollowPath"]

    FollowPath:
      plugin: "dwb_core::DWBLocalPlanner"

      # Velocity limits
      min_vel_x: 0.0
      max_vel_x: 0.26
      max_vel_theta: 1.0
      min_speed_xy: 0.0
      max_speed_xy: 0.26

      # Acceleration limits
      acc_lim_x: 2.5
      acc_lim_theta: 3.2
      decel_lim_x: -2.5

      # Trajectory generation
      vx_samples: 20
      vy_samples: 5
      vtheta_samples: 20
      sim_time: 1.7

      # Critics (scoring functions)
      critics: ["RotateToGoal", "Oscillation", "BaseObstacle",
                "GoalAlign", "PathAlign", "PathDist", "GoalDist"]
```

**Regulated Pure Pursuit** (smoother paths):

```yaml
FollowPath:
  plugin: "nav2_regulated_pure_pursuit_controller::RegulatedPurePursuitController"
  desired_linear_vel: 0.5
  lookahead_dist: 0.6
  min_lookahead_dist: 0.3
  max_lookahead_dist: 0.9
```

---

## Behavior Trees

### What are Behavior Trees?

Nav2 uses behavior trees to orchestrate navigation logic. BTs are more flexible than finite state machines:

```
[Root]
  └── [Sequence]
        ├── [ComputePathToPose]
        ├── [FollowPath]
        └── [Recovery] (on failure)
              ├── [ClearCostmap]
              ├── [Spin]
              └── [BackUp]
```

### Default Navigate to Pose BT

```xml title="navigate_to_pose_w_replanning_and_recovery.xml"
<root main_tree_to_execute="MainTree">
  <BehaviorTree ID="MainTree">
    <RecoveryNode number_of_retries="6" name="NavigateRecovery">
      <PipelineSequence name="NavigateWithReplanning">
        <RateController hz="1.0">
          <ComputePathToPose goal="{goal}" path="{path}"
                             planner_id="GridBased"/>
        </RateController>
        <FollowPath path="{path}" controller_id="FollowPath"/>
      </PipelineSequence>
      <SequenceStar name="RecoveryActions">
        <ClearEntireCostmap name="ClearGlobalCostmap"
                            service_name="/global_costmap/clear_entirely_global_costmap"/>
        <ClearEntireCostmap name="ClearLocalCostmap"
                            service_name="/local_costmap/clear_entirely_local_costmap"/>
        <Spin spin_dist="1.57"/>
        <Wait wait_duration="5"/>
        <BackUp backup_dist="0.3" backup_speed="0.05"/>
      </SequenceStar>
    </RecoveryNode>
  </BehaviorTree>
</root>
```

### BT Navigator Configuration

```yaml title="bt_navigator_params.yaml"
bt_navigator:
  ros__parameters:
    use_sim_time: true
    global_frame: map
    robot_base_frame: base_link
    odom_topic: /odom

    # BT file to use
    default_bt_xml_filename: "navigate_to_pose_w_replanning_and_recovery.xml"

    # Plugin libraries
    plugin_lib_names:
      - nav2_compute_path_to_pose_action_bt_node
      - nav2_follow_path_action_bt_node
      - nav2_spin_action_bt_node
      - nav2_wait_action_bt_node
      - nav2_back_up_action_bt_node
      - nav2_clear_costmap_service_bt_node
```

---

## Sending Navigation Goals

### Using the Action Client

```python title="nav_goal_sender.py"
import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import NavigateToPose


class NavGoalSender(Node):
    """Send navigation goals to Nav2."""

    def __init__(self):
        super().__init__('nav_goal_sender')
        self._action_client = ActionClient(
            self,
            NavigateToPose,
            'navigate_to_pose'
        )
        self.get_logger().info('Waiting for Nav2...')
        self._action_client.wait_for_server()
        self.get_logger().info('Nav2 ready!')

    def send_goal(self, x: float, y: float, yaw: float = 0.0):
        """Send a navigation goal."""
        import math

        goal_msg = NavigateToPose.Goal()
        goal_msg.pose.header.frame_id = 'map'
        goal_msg.pose.header.stamp = self.get_clock().now().to_msg()

        goal_msg.pose.pose.position.x = x
        goal_msg.pose.pose.position.y = y
        goal_msg.pose.pose.position.z = 0.0

        goal_msg.pose.pose.orientation.z = math.sin(yaw / 2.0)
        goal_msg.pose.pose.orientation.w = math.cos(yaw / 2.0)

        self.get_logger().info(f'Sending goal: ({x}, {y})')

        self._send_goal_future = self._action_client.send_goal_async(
            goal_msg,
            feedback_callback=self.feedback_callback
        )
        self._send_goal_future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().warn('Goal rejected')
            return

        self.get_logger().info('Goal accepted')
        self._get_result_future = goal_handle.get_result_async()
        self._get_result_future.add_done_callback(self.result_callback)

    def feedback_callback(self, feedback_msg):
        feedback = feedback_msg.feedback
        current = feedback.current_pose.pose.position
        remaining = feedback.distance_remaining
        self.get_logger().info(
            f'Position: ({current.x:.2f}, {current.y:.2f}), '
            f'Distance remaining: {remaining:.2f}m'
        )

    def result_callback(self, future):
        result = future.result().result
        self.get_logger().info('Navigation complete!')


def main(args=None):
    rclpy.init(args=args)
    node = NavGoalSender()

    # Send goal
    node.send_goal(2.0, 1.0)

    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
```

### Waypoint Following

```python title="waypoint_follower.py"
from nav2_msgs.action import FollowWaypoints
from geometry_msgs.msg import PoseStamped


class WaypointFollower(Node):
    """Follow multiple waypoints sequentially."""

    def __init__(self):
        super().__init__('waypoint_follower')
        self._action_client = ActionClient(
            self,
            FollowWaypoints,
            'follow_waypoints'
        )

    def follow_waypoints(self, waypoints: list):
        """Send list of waypoints to follow."""
        goal_msg = FollowWaypoints.Goal()

        for wp in waypoints:
            pose = PoseStamped()
            pose.header.frame_id = 'map'
            pose.header.stamp = self.get_clock().now().to_msg()
            pose.pose.position.x = wp[0]
            pose.pose.position.y = wp[1]
            pose.pose.orientation.w = 1.0
            goal_msg.poses.append(pose)

        self.get_logger().info(f'Following {len(waypoints)} waypoints')
        return self._action_client.send_goal_async(goal_msg)
```

---

## Running Nav2 with TurtleBot3

### Launch the Simulation

```bash title="Terminal 1: Launch Gazebo"
export TURTLEBOT3_MODEL=burger
ros2 launch turtlebot3_gazebo turtlebot3_world.launch.py
```

```bash title="Terminal 2: Launch Nav2"
export TURTLEBOT3_MODEL=burger
ros2 launch turtlebot3_navigation2 navigation2.launch.py use_sim_time:=true
```

```bash title="Terminal 3: Launch RViz2"
ros2 launch nav2_bringup rviz_launch.py
```

### In RViz2:

1. Click **2D Pose Estimate** → Click on map to set initial pose
2. Wait for AMCL particles to converge
3. Click **2D Goal Pose** → Click destination
4. Watch robot navigate!

### CLI Navigation

```bash title="Terminal"
# Send goal via CLI
ros2 action send_goal /navigate_to_pose nav2_msgs/action/NavigateToPose \
  "{pose: {header: {frame_id: 'map'}, pose: {position: {x: 2.0, y: 1.0}}}}"
```

---

## Common Issues and Solutions

| Problem | Cause | Solution |
|---------|-------|----------|
| Robot doesn't move | Localization lost | Set initial pose, check TF |
| Path not found | Obstacles blocking | Clear costmaps, check inflation |
| Robot oscillates | Controller tuning | Reduce velocity, adjust critics |
| Recovery loops | Stuck in corner | Increase recovery behaviors |
| Costmap artifacts | Sensor noise | Adjust obstacle parameters |

### Debugging Commands

```bash title="Terminal"
# Check transforms
ros2 run tf2_tools view_frames
ros2 run tf2_ros tf2_echo map base_link

# Monitor topics
ros2 topic echo /amcl_pose
ros2 topic echo /local_costmap/costmap
ros2 topic echo /plan

# Check lifecycle states
ros2 lifecycle list /amcl
ros2 lifecycle get /planner_server
```

---

## Exercise 1: Multi-Goal Navigation

:::tip Exercise 1: Patrol Robot
**Objective**: Create a patrol robot that visits multiple locations in sequence.

**Time Estimate**: 45 minutes

**Requirements**:

1. Launch TurtleBot3 with Nav2
2. Create `patrol_node.py` that:
   - Defines 4 patrol waypoints
   - Navigates to each in sequence
   - Loops indefinitely
   - Handles navigation failures gracefully

**Steps**:

1. Define waypoints as a list of (x, y, yaw) tuples
2. Use FollowWaypoints action or sequential NavigateToPose
3. Add logic to restart patrol on completion
4. Add logging for each waypoint reached

**Expected Behavior**:
```
[INFO] Starting patrol
[INFO] Navigating to waypoint 1: (1.0, 0.0)
[INFO] Reached waypoint 1
[INFO] Navigating to waypoint 2: (1.0, 1.0)
...
[INFO] Patrol complete, restarting
```

**Challenge**: Add a pause at each waypoint and publish a "checkpoint reached" message.
:::

---

## Exercise 2: Custom Recovery Behavior

:::tip Exercise 2: Smart Recovery
**Objective**: Implement custom recovery when navigation fails.

**Time Estimate**: 30 minutes

**Requirements**:

Modify the navigation client to:
1. Detect when navigation fails (result not success)
2. Attempt recovery actions:
   - First: Clear costmaps and retry
   - Second: Move backward 0.5m and retry
   - Third: Spin 180 degrees and retry
   - Finally: Report failure to user

**Starter Approach**:

```python
def navigate_with_recovery(self, x, y, max_retries=3):
    for attempt in range(max_retries):
        result = self.send_goal_sync(x, y)
        if result.success:
            return True

        self.get_logger().warn(f'Attempt {attempt+1} failed, recovering...')
        self.execute_recovery(attempt)

    return False
```

**Expected Behavior**:
- Robot attempts navigation
- On failure, tries recovery behaviors
- Reports which recovery helped (if any)
:::

---

## Summary

In this chapter, you learned:

- **Nav2 Architecture**: Modular, lifecycle-managed navigation stack
- **Localization**: AMCL particle filter for map-based positioning
- **Costmaps**: Grid representation for planning and obstacle avoidance
- **Path Planning**: Global (NavFn, Smac) and local (DWB, RPP) planners
- **Behavior Trees**: Flexible navigation logic orchestration
- **Goal Sending**: Programmatic navigation with feedback

Nav2 provides everything needed for autonomous mobile robot navigation. In the next chapter, you'll learn about [Manipulation with MoveIt 2](/docs/module-1/ros2-manipulation) for robot arm control.

## Further Reading

- [Navigation2 Documentation](https://navigation.ros.org/)
- [Nav2 Configuration Guide](https://navigation.ros.org/configuration/index.html)
- [Behavior Tree Tutorial](https://navigation.ros.org/behavior_trees/index.html)
- [TurtleBot3 Navigation](https://emanual.robotis.com/docs/en/platform/turtlebot3/navigation/)
