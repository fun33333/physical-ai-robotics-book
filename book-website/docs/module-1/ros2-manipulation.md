---
title: "Manipulation (MoveIt 2)"
sidebar_position: 6
description: "Robot arm manipulation using MoveIt 2: motion planning, collision avoidance, and grasp generation."
---

# Manipulation (MoveIt 2)

MoveIt 2 is the most widely used software for manipulation, incorporating motion planning, kinematics, collision checking, and control. This chapter introduces the MoveIt 2 ecosystem for programming robot arms and hands.

## Overview

In this section, you will:

- Understand the MoveIt 2 architecture and components
- Configure MoveIt 2 for a robot arm
- Plan and execute collision-free motions
- Use the MoveIt Python API for manipulation tasks
- Implement basic pick-and-place operations

## Prerequisites

Before starting, ensure you have completed:

- [ROS 2 Fundamentals](/docs/module-1/ros2-fundamentals)
- [Actions & Lifecycle Nodes](/docs/module-1/ros2-actions-lifecycle)

Additionally, install MoveIt 2:

```bash title="Install MoveIt 2"
sudo apt install ros-humble-moveit ros-humble-moveit-visual-tools
sudo apt install ros-humble-moveit-planners-ompl ros-humble-moveit-simple-controller-manager
```

For tutorials, install the Panda robot:

```bash title="Install Panda Robot (Tutorial Robot)"
sudo apt install ros-humble-moveit-resources-panda-moveit-config
```

---

## MoveIt 2 Architecture

### System Overview

MoveIt 2 provides a complete manipulation pipeline:

```
┌─────────────────────────────────────────────────────────────────┐
│                     MoveIt 2 Architecture                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   ┌───────────────┐    ┌───────────────┐    ┌───────────────┐   │
│   │  Move Group   │    │   Planning    │    │   Execution   │   │
│   │   Interface   │───▶│   Pipeline    │───▶│   Interface   │   │
│   └───────────────┘    └───────────────┘    └───────────────┘   │
│          │                    │                    │             │
│          ▼                    ▼                    ▼             │
│   ┌───────────────┐    ┌───────────────┐    ┌───────────────┐   │
│   │    Robot      │    │    Motion     │    │  Controller   │   │
│   │  Description  │    │   Planners    │    │   Manager     │   │
│   │    (URDF)     │    │   (OMPL)      │    │               │   │
│   └───────────────┘    └───────────────┘    └───────────────┘   │
│          │                    │                    │             │
│          ▼                    ▼                    ▼             │
│   ┌───────────────┐    ┌───────────────┐    ┌───────────────┐   │
│   │   Collision   │    │   Planning    │    │   Trajectory  │   │
│   │   Detection   │    │    Scene      │    │   Execution   │   │
│   └───────────────┘    └───────────────┘    └───────────────┘   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Key Components

| Component | Purpose |
|-----------|---------|
| **Move Group** | High-level interface for planning and execution |
| **Planning Scene** | World representation with robot and obstacles |
| **Motion Planners** | Algorithms to compute collision-free paths (OMPL) |
| **Kinematics** | Forward/inverse kinematics solvers |
| **Collision Detection** | FCL-based collision checking |
| **Controller Manager** | Interfaces with robot hardware controllers |

### Core Concepts

**Planning Group**: A named set of joints that move together (e.g., "arm", "gripper")

**End Effector**: The tool at the end of the arm (gripper, vacuum, tool)

**Planning Scene**: The world model including robot state, obstacles, and attached objects

**Trajectory**: Time-parameterized sequence of joint configurations

---

## Robot Description (URDF/SRDF)

### Understanding URDF

The Unified Robot Description Format describes robot kinematics and dynamics:

```xml title="simple_arm.urdf"
<?xml version="1.0"?>
<robot name="simple_arm">

  <!-- Base Link -->
  <link name="base_link">
    <visual>
      <geometry>
        <box size="0.1 0.1 0.1"/>
      </geometry>
    </visual>
    <collision>
      <geometry>
        <box size="0.1 0.1 0.1"/>
      </geometry>
    </collision>
  </link>

  <!-- Link 1 -->
  <link name="link_1">
    <visual>
      <geometry>
        <cylinder length="0.5" radius="0.05"/>
      </geometry>
      <origin xyz="0 0 0.25" rpy="0 0 0"/>
    </visual>
  </link>

  <!-- Joint 1 (Revolute) -->
  <joint name="joint_1" type="revolute">
    <parent link="base_link"/>
    <child link="link_1"/>
    <origin xyz="0 0 0.05" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-3.14" upper="3.14" effort="100" velocity="1.0"/>
  </joint>

  <!-- End Effector Link -->
  <link name="end_effector">
    <visual>
      <geometry>
        <sphere radius="0.02"/>
      </geometry>
    </visual>
  </link>

  <joint name="ee_joint" type="fixed">
    <parent link="link_1"/>
    <child link="end_effector"/>
    <origin xyz="0 0 0.5" rpy="0 0 0"/>
  </joint>

</robot>
```

### SRDF for MoveIt

The Semantic Robot Description Format adds MoveIt-specific information:

```xml title="simple_arm.srdf"
<?xml version="1.0"?>
<robot name="simple_arm">

  <!-- Planning Groups -->
  <group name="arm">
    <joint name="joint_1"/>
    <joint name="joint_2"/>
    <joint name="joint_3"/>
  </group>

  <group name="gripper">
    <joint name="finger_joint_1"/>
    <joint name="finger_joint_2"/>
  </group>

  <!-- End Effector -->
  <end_effector name="gripper" parent_link="link_3" group="gripper"/>

  <!-- Named Poses -->
  <group_state name="home" group="arm">
    <joint name="joint_1" value="0"/>
    <joint name="joint_2" value="-1.57"/>
    <joint name="joint_3" value="0"/>
  </group_state>

  <group_state name="ready" group="arm">
    <joint name="joint_1" value="0"/>
    <joint name="joint_2" value="0"/>
    <joint name="joint_3" value="0"/>
  </group_state>

  <!-- Disable Collision Between Adjacent Links -->
  <disable_collisions link1="base_link" link2="link_1" reason="Adjacent"/>
  <disable_collisions link1="link_1" link2="link_2" reason="Adjacent"/>

</robot>
```

---

## MoveIt Setup Assistant

The Setup Assistant generates MoveIt configuration packages:

```bash title="Launch Setup Assistant"
ros2 launch moveit_setup_assistant setup_assistant.launch.py
```

**Steps:**

1. Load URDF or existing config
2. Generate Self-Collision Matrix
3. Define Planning Groups
4. Define Robot Poses
5. Configure End Effectors
6. Define Passive Joints
7. Configure ROS 2 Controllers
8. Generate Package

---

## Motion Planning with Python

### MoveIt Python Interface

```python title="moveit_interface.py"
import rclpy
from rclpy.node import Node
from moveit.planning import MoveItPy
from moveit.core.robot_state import RobotState
from geometry_msgs.msg import Pose, PoseStamped
import numpy as np


class ManipulationNode(Node):
    """Basic manipulation using MoveIt Python API."""

    def __init__(self):
        super().__init__('manipulation_node')

        # Initialize MoveIt
        self.moveit = MoveItPy(node_name="moveit_py")
        self.arm = self.moveit.get_planning_component("panda_arm")
        self.planning_scene = self.moveit.get_planning_scene_monitor()

        self.get_logger().info('MoveIt interface ready')

    def move_to_named_pose(self, pose_name: str) -> bool:
        """Move to a pre-defined named pose."""
        self.arm.set_start_state_to_current_state()
        self.arm.set_goal_state(configuration_name=pose_name)

        # Plan
        plan_result = self.arm.plan()

        if plan_result:
            self.get_logger().info(f'Moving to {pose_name}')
            self.moveit.execute(plan_result.trajectory, controllers=[])
            return True
        else:
            self.get_logger().warn(f'Planning to {pose_name} failed')
            return False

    def move_to_pose(self, pose: Pose) -> bool:
        """Move end effector to target pose."""
        self.arm.set_start_state_to_current_state()
        self.arm.set_goal_state(
            pose_stamped_msg=PoseStamped(
                header={'frame_id': 'panda_link0'},
                pose=pose
            ),
            pose_link="panda_link8"
        )

        plan_result = self.arm.plan()

        if plan_result:
            self.get_logger().info('Executing motion plan')
            self.moveit.execute(plan_result.trajectory, controllers=[])
            return True
        return False

    def move_to_joint_positions(self, joint_values: list) -> bool:
        """Move to specific joint configuration."""
        robot_state = RobotState(self.moveit.get_robot_model())
        robot_state.set_joint_group_positions("panda_arm", joint_values)

        self.arm.set_start_state_to_current_state()
        self.arm.set_goal_state(robot_state=robot_state)

        plan_result = self.arm.plan()

        if plan_result:
            self.moveit.execute(plan_result.trajectory, controllers=[])
            return True
        return False
```

### Using Move Group Interface (Traditional)

```python title="move_group_interface.py"
import rclpy
from rclpy.node import Node
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import (
    MotionPlanRequest,
    Constraints,
    PositionConstraint,
    OrientationConstraint,
    BoundingVolume
)
from shape_msgs.msg import SolidPrimitive
from geometry_msgs.msg import Pose, PoseStamped
from rclpy.action import ActionClient


class MoveGroupClient(Node):
    """Action client for MoveGroup."""

    def __init__(self):
        super().__init__('move_group_client')
        self._action_client = ActionClient(
            self,
            MoveGroup,
            'move_action'
        )
        self.get_logger().info('Waiting for MoveGroup action server...')
        self._action_client.wait_for_server()
        self.get_logger().info('Connected to MoveGroup')

    def plan_to_pose(self, target_pose: Pose, planning_group: str = 'panda_arm'):
        """Plan motion to target pose."""
        goal = MoveGroup.Goal()

        # Motion plan request
        goal.request.group_name = planning_group
        goal.request.num_planning_attempts = 10
        goal.request.allowed_planning_time = 5.0

        # Set target pose constraint
        constraints = Constraints()

        # Position constraint
        pos_constraint = PositionConstraint()
        pos_constraint.header.frame_id = 'panda_link0'
        pos_constraint.link_name = 'panda_link8'

        # Target region (small box around target)
        bounding_volume = BoundingVolume()
        primitive = SolidPrimitive()
        primitive.type = SolidPrimitive.BOX
        primitive.dimensions = [0.01, 0.01, 0.01]
        bounding_volume.primitives.append(primitive)

        primitive_pose = Pose()
        primitive_pose.position = target_pose.position
        primitive_pose.orientation.w = 1.0
        bounding_volume.primitive_poses.append(primitive_pose)

        pos_constraint.constraint_region = bounding_volume
        pos_constraint.weight = 1.0
        constraints.position_constraints.append(pos_constraint)

        # Orientation constraint
        orient_constraint = OrientationConstraint()
        orient_constraint.header.frame_id = 'panda_link0'
        orient_constraint.link_name = 'panda_link8'
        orient_constraint.orientation = target_pose.orientation
        orient_constraint.absolute_x_axis_tolerance = 0.1
        orient_constraint.absolute_y_axis_tolerance = 0.1
        orient_constraint.absolute_z_axis_tolerance = 0.1
        orient_constraint.weight = 1.0
        constraints.orientation_constraints.append(orient_constraint)

        goal.request.goal_constraints.append(constraints)

        # Send goal
        future = self._action_client.send_goal_async(goal)
        return future
```

---

## Planning Scene Management

### Adding Collision Objects

```python title="collision_objects.py"
from moveit_msgs.msg import CollisionObject, PlanningScene
from shape_msgs.msg import SolidPrimitive
from geometry_msgs.msg import Pose


class SceneManager:
    """Manage planning scene objects."""

    def __init__(self, node):
        self.node = node
        self.scene_pub = node.create_publisher(
            PlanningScene,
            '/planning_scene',
            10
        )

    def add_box(self, name: str, pose: Pose, size: list):
        """Add a box obstacle to the scene."""
        co = CollisionObject()
        co.header.frame_id = 'panda_link0'
        co.id = name
        co.operation = CollisionObject.ADD

        # Define box shape
        box = SolidPrimitive()
        box.type = SolidPrimitive.BOX
        box.dimensions = size  # [x, y, z]

        co.primitives.append(box)
        co.primitive_poses.append(pose)

        # Publish to planning scene
        scene = PlanningScene()
        scene.world.collision_objects.append(co)
        scene.is_diff = True

        self.scene_pub.publish(scene)
        self.node.get_logger().info(f'Added box: {name}')

    def add_table(self):
        """Add a table to the scene."""
        pose = Pose()
        pose.position.x = 0.5
        pose.position.y = 0.0
        pose.position.z = 0.2
        pose.orientation.w = 1.0

        self.add_box('table', pose, [0.6, 1.0, 0.4])

    def remove_object(self, name: str):
        """Remove an object from the scene."""
        co = CollisionObject()
        co.header.frame_id = 'panda_link0'
        co.id = name
        co.operation = CollisionObject.REMOVE

        scene = PlanningScene()
        scene.world.collision_objects.append(co)
        scene.is_diff = True

        self.scene_pub.publish(scene)

    def attach_object(self, object_name: str, link_name: str = 'panda_link8'):
        """Attach an object to the robot (for grasping)."""
        # Object is now part of robot, moves with it
        co = CollisionObject()
        co.header.frame_id = link_name
        co.id = object_name
        co.operation = CollisionObject.ADD

        scene = PlanningScene()
        scene.robot_state.attached_collision_objects.append(co)
        scene.is_diff = True

        self.scene_pub.publish(scene)
```

---

## Pick and Place Operations

### Basic Pick and Place

```python title="pick_and_place.py"
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose
import time


class PickAndPlace(Node):
    """Simple pick and place demonstration."""

    def __init__(self):
        super().__init__('pick_and_place')
        # Initialize MoveIt interface (from previous examples)
        self.manipulation = ManipulationNode()
        self.scene = SceneManager(self)

    def pick(self, object_pose: Pose):
        """Pick up an object."""
        self.get_logger().info('Starting pick operation')

        # 1. Open gripper
        self.open_gripper()

        # 2. Move to pre-grasp pose (above object)
        pre_grasp = Pose()
        pre_grasp.position.x = object_pose.position.x
        pre_grasp.position.y = object_pose.position.y
        pre_grasp.position.z = object_pose.position.z + 0.15  # 15cm above
        pre_grasp.orientation.x = 1.0  # Pointing down
        pre_grasp.orientation.w = 0.0

        self.manipulation.move_to_pose(pre_grasp)

        # 3. Move to grasp pose
        grasp_pose = Pose()
        grasp_pose.position = object_pose.position
        grasp_pose.orientation = pre_grasp.orientation

        self.manipulation.move_to_pose(grasp_pose)

        # 4. Close gripper
        self.close_gripper()
        time.sleep(0.5)  # Wait for grasp

        # 5. Attach object to gripper in planning scene
        self.scene.attach_object('target_object')

        # 6. Lift object
        self.manipulation.move_to_pose(pre_grasp)

        self.get_logger().info('Pick complete')

    def place(self, place_pose: Pose):
        """Place the held object."""
        self.get_logger().info('Starting place operation')

        # 1. Move to pre-place pose
        pre_place = Pose()
        pre_place.position.x = place_pose.position.x
        pre_place.position.y = place_pose.position.y
        pre_place.position.z = place_pose.position.z + 0.15
        pre_place.orientation.x = 1.0
        pre_place.orientation.w = 0.0

        self.manipulation.move_to_pose(pre_place)

        # 2. Move to place pose
        self.manipulation.move_to_pose(place_pose)

        # 3. Open gripper
        self.open_gripper()
        time.sleep(0.5)

        # 4. Detach object from gripper
        self.scene.remove_object('target_object')

        # 5. Retreat
        self.manipulation.move_to_pose(pre_place)

        self.get_logger().info('Place complete')

    def open_gripper(self):
        """Open the gripper."""
        # Implementation depends on gripper type
        self.manipulation.move_gripper(0.04)  # 4cm open

    def close_gripper(self):
        """Close the gripper."""
        self.manipulation.move_gripper(0.0)  # Closed

    def move_gripper(self, width: float):
        """Move gripper to specified width."""
        # Use gripper action or joint control
        pass


def main(args=None):
    rclpy.init(args=args)
    node = PickAndPlace()

    # Define pick and place locations
    pick_pose = Pose()
    pick_pose.position.x = 0.5
    pick_pose.position.y = 0.0
    pick_pose.position.z = 0.1

    place_pose = Pose()
    place_pose.position.x = 0.3
    place_pose.position.y = 0.3
    place_pose.position.z = 0.1

    # Execute pick and place
    node.pick(pick_pose)
    node.place(place_pose)

    node.destroy_node()
    rclpy.shutdown()
```

---

## Cartesian Path Planning

### Linear Motions

```python title="cartesian_path.py"
from moveit_msgs.msg import RobotTrajectory
from geometry_msgs.msg import Pose
import copy


def compute_cartesian_path(move_group, waypoints: list, eef_step: float = 0.01):
    """
    Compute a Cartesian path through waypoints.

    Args:
        move_group: MoveIt planning component
        waypoints: List of Pose objects
        eef_step: Step size for interpolation (meters)

    Returns:
        tuple: (trajectory, fraction of path achieved)
    """
    # Get current pose
    current_pose = move_group.get_current_pose().pose

    # Create waypoints list
    poses = [current_pose] + waypoints

    # Plan Cartesian path
    (plan, fraction) = move_group.compute_cartesian_path(
        poses,
        eef_step,          # End effector step
        0.0,               # Jump threshold (0 = disabled)
        avoid_collisions=True
    )

    return plan, fraction


def draw_rectangle(move_group, width: float, height: float):
    """Draw a rectangle in Cartesian space."""
    waypoints = []

    # Get starting pose
    start = move_group.get_current_pose().pose

    # Point 1: Right
    pose1 = copy.deepcopy(start)
    pose1.position.y += width / 2
    waypoints.append(pose1)

    # Point 2: Forward
    pose2 = copy.deepcopy(pose1)
    pose2.position.x += height
    waypoints.append(pose2)

    # Point 3: Left
    pose3 = copy.deepcopy(pose2)
    pose3.position.y -= width
    waypoints.append(pose3)

    # Point 4: Back
    pose4 = copy.deepcopy(pose3)
    pose4.position.x -= height
    waypoints.append(pose4)

    # Back to start
    waypoints.append(copy.deepcopy(start))

    # Plan and execute
    plan, fraction = compute_cartesian_path(move_group, waypoints)

    if fraction > 0.9:
        move_group.execute(plan, wait=True)
        return True
    return False
```

---

## Common Motion Planning Issues

### Troubleshooting

| Problem | Cause | Solution |
|---------|-------|----------|
| No solution found | IK failure or collision | Try different seed state, adjust pose |
| Jerky motion | Time parameterization | Use TOTG or increase planning time |
| Collision in path | Scene not updated | Sync planning scene, check transforms |
| Controller error | Trajectory too fast | Reduce velocity scaling factor |
| Goal tolerance error | Pose unreachable | Check workspace limits |

### Planning Parameters

```yaml title="moveit_planning.yaml"
move_group:
  ros__parameters:
    # Planning time
    planning_time: 5.0
    num_planning_attempts: 10

    # Velocity and acceleration scaling
    max_velocity_scaling_factor: 0.5
    max_acceleration_scaling_factor: 0.5

    # Planner configuration
    default_planner_config: RRTConnect
    default_planning_pipeline: ompl

    # Trajectory execution
    moveit_manage_controllers: true
    trajectory_execution.allowed_start_tolerance: 0.01
```

---

## Running MoveIt 2 Demo

### Launch Panda Robot

```bash title="Terminal: Launch MoveIt Demo"
ros2 launch moveit_resources_panda_moveit_config demo.launch.py
```

This launches:
- Robot state publisher
- MoveIt move_group node
- RViz with Motion Planning plugin

### In RViz:

1. Use **Motion Planning** panel
2. Set **Start State** to current
3. Set **Goal State** by dragging interactive marker
4. Click **Plan** to compute path
5. Click **Execute** to run motion

---

## Exercise 1: Simple Pick and Place

:::tip Exercise 1: Box Stacking
**Objective**: Stack three boxes using pick and place.

**Time Estimate**: 60 minutes

**Requirements**:

1. Launch MoveIt with Panda robot
2. Add three box objects to the scene at different locations
3. Implement a node that:
   - Picks up each box in sequence
   - Stacks them at a target location
   - Uses appropriate pre-grasp and pre-place poses

**Steps**:

1. Create scene with boxes:
   ```python
   scene.add_box('box1', pose1, [0.05, 0.05, 0.05])
   scene.add_box('box2', pose2, [0.05, 0.05, 0.05])
   scene.add_box('box3', pose3, [0.05, 0.05, 0.05])
   ```

2. Implement pick/place for each box
3. Adjust place height for stacking (5cm, 10cm, 15cm)
4. Handle attached objects correctly

**Expected Behavior**:
- Robot picks box1, places at stack location
- Robot picks box2, places on top of box1
- Robot picks box3, places on top of box2

**Challenge**: Add collision detection and retry logic if planning fails.
:::

---

## Exercise 2: Cartesian Drawing

:::tip Exercise 2: Draw a Shape
**Objective**: Use Cartesian path planning to draw shapes.

**Time Estimate**: 40 minutes

**Requirements**:

Create a node that:
1. Moves to a starting position
2. Computes a Cartesian path for a shape (square, circle, star)
3. Executes the path smoothly
4. Returns to home position

**Steps**:

1. Define waypoints for your shape
2. Use `compute_cartesian_path()` to plan
3. Check fraction completed (should be > 0.95)
4. Execute if planning successful

**Expected Behavior**:
```
[INFO] Planning Cartesian path...
[INFO] Path fraction: 0.98
[INFO] Executing drawing motion
[INFO] Drawing complete
```

**Challenge**: Draw multiple shapes in sequence, add a "pen up/pen down" motion between shapes.
:::

---

## Summary

In this chapter, you learned:

- **MoveIt Architecture**: Planning scene, motion planners, execution pipeline
- **Robot Description**: URDF for kinematics, SRDF for MoveIt configuration
- **Motion Planning**: Using OMPL planners through MoveIt interface
- **Planning Scene**: Managing collision objects and attached objects
- **Pick and Place**: Complete manipulation workflow
- **Cartesian Paths**: Linear motion planning through waypoints

MoveIt 2 provides the tools needed for sophisticated robot manipulation. Combined with the navigation skills from the previous chapter, you can now build complete mobile manipulation applications.

## Further Reading

- [MoveIt 2 Documentation](https://moveit.picknik.ai/)
- [MoveIt 2 Tutorials](https://moveit.picknik.ai/humble/doc/tutorials/tutorials.html)
- [OMPL Motion Planning Library](https://ompl.kavrakilab.org/)
- [MoveIt Task Constructor](https://moveit.picknik.ai/main/doc/tutorials/pick_and_place_with_moveit_task_constructor/pick_and_place_with_moveit_task_constructor.html)

---

## Module 1 Complete

Congratulations! You have completed Module 1: The Robotic Nervous System (ROS 2). You now have:

- **ROS 2 fundamentals**: Installation, workspaces, packages
- **Communication patterns**: Topics, services, actions
- **Advanced patterns**: Lifecycle nodes, behavior trees
- **Navigation**: Nav2 for autonomous mobile robots
- **Manipulation**: MoveIt 2 for robot arm control

Continue to [Module 2: Simulation](/docs/module-2) to learn how to test your robots in virtual environments before deploying to real hardware.
