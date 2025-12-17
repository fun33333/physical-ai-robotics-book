---
title: "Isaac Sim"
sidebar_position: 3
description: "High-fidelity robotics simulation with NVIDIA Isaac Sim: photorealistic rendering and physics."
---

# Isaac Sim

Isaac Sim is NVIDIA's flagship robotics simulator, built on the Omniverse platform. It provides photorealistic rendering, accurate physics, and comprehensive sensor simulation—capabilities that make it ideal for training perception systems and validating robot behaviors before real-world deployment.

## Overview

In this section, you will:

- Navigate the Omniverse interface and Isaac Sim extensions
- Work with USD (Universal Scene Description) format
- Import and configure robot models
- Set up sensors with realistic noise models
- Generate synthetic datasets for training
- Integrate Isaac Sim with ROS 2

## Prerequisites

- Isaac Sim installed via Omniverse Launcher
- NVIDIA GPU with 8GB+ VRAM (12GB+ recommended)
- Basic Python programming skills
- Familiarity with robotics concepts from Modules 1-2

---

## Getting Started

### Launching Isaac Sim

```bash title="Launch Isaac Sim"
# Standard launch
~/.local/share/ov/pkg/isaac_sim-2023.1.1/isaac-sim.sh

# With specific extensions
~/.local/share/ov/pkg/isaac_sim-2023.1.1/isaac-sim.sh \
  --enable omni.isaac.ros2_bridge

# Headless mode (for training)
~/.local/share/ov/pkg/isaac_sim-2023.1.1/isaac-sim.sh --headless
```

### Interface Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                     Isaac Sim Interface                          │
├─────────────────────────────────────────────────────────────────┤
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                     Menu Bar                              │   │
│  │  File  Edit  Create  Window  Isaac Examples  Help         │   │
│  └──────────────────────────────────────────────────────────┘   │
│  ┌────────────┐ ┌──────────────────────────┐ ┌────────────┐    │
│  │            │ │                          │ │            │    │
│  │   Stage    │ │       Viewport           │ │ Property   │    │
│  │   Panel    │ │                          │ │ Panel      │    │
│  │            │ │   3D Scene View          │ │            │    │
│  │ /World     │ │                          │ │ Transform  │    │
│  │  /Robot    │ │        [Robot]           │ │ Physics    │    │
│  │  /Ground   │ │           ↓              │ │ Materials  │    │
│  │  /Lights   │ │        [Floor]           │ │            │    │
│  │            │ │                          │ │            │    │
│  └────────────┘ └──────────────────────────┘ └────────────┘    │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  Timeline  │►  ■  │  0:00:00  ───────────── 0:10:00     │   │
│  └──────────────────────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  Console │ Output logs and Python REPL                    │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

---

## Working with USD

### USD Fundamentals

USD (Universal Scene Description) is Pixar's format for 3D scenes:

```python title="USD structure example"
# USD represents scenes as hierarchical prims
# Each prim has properties, relationships, and metadata

# Example scene hierarchy:
/World                          # Root prim
    /Physics                    # Physics settings
        /PhysicsScene
    /Environment
        /GroundPlane
        /DomeLight
    /Robot
        /base_link              # Robot base
            /left_wheel_joint
            /left_wheel
            /right_wheel_joint
            /right_wheel
        /camera_link
            /Camera
```

### Creating a Scene Programmatically

```python title="create_scene.py"
"""Create an Isaac Sim scene programmatically."""
import omni
from omni.isaac.core import World
from omni.isaac.core.objects import DynamicCuboid
from omni.isaac.core.utils.nucleus import get_assets_root_path
from pxr import Gf

# Initialize world
world = World(stage_units_in_meters=1.0)
world.scene.add_default_ground_plane()

# Add a dynamic cube
cube = world.scene.add(
    DynamicCuboid(
        prim_path="/World/Cube",
        name="my_cube",
        position=Gf.Vec3f(0.0, 0.0, 1.0),
        size=0.5,
        color=Gf.Vec3f(1.0, 0.0, 0.0)  # Red
    )
)

# Add a robot from USD
assets_root = get_assets_root_path()
robot_usd = assets_root + "/Isaac/Robots/Carter/carter_v1.usd"

from omni.isaac.core.utils.stage import add_reference_to_stage
add_reference_to_stage(usd_path=robot_usd, prim_path="/World/Robot")

# Initialize physics
world.reset()

# Simulation loop
for i in range(1000):
    world.step(render=True)

print("Simulation complete!")
```

---

## Importing Robots

### From URDF

```python title="urdf_import.py"
"""Import a robot from URDF."""
from omni.isaac.urdf import _urdf
from omni.isaac.core.utils.extensions import enable_extension

# Enable URDF extension
enable_extension("omni.isaac.urdf")

# Configure import settings
import_config = _urdf.ImportConfig()
import_config.merge_fixed_joints = False
import_config.fix_base = False
import_config.import_inertia_tensor = True
import_config.distance_scale = 1.0
import_config.density = 0.0
import_config.default_drive_type = _urdf.UrdfJointTargetType.JOINT_DRIVE_POSITION
import_config.default_drive_strength = 1000.0
import_config.default_position_drive_damping = 100.0

# Import URDF
urdf_interface = _urdf.acquire_urdf_interface()
result = urdf_interface.parse_urdf(
    urdf_path="/path/to/robot.urdf",
    import_config=import_config
)

# Import to stage
urdf_interface.import_robot(
    urdf_path="/path/to/robot.urdf",
    import_config=import_config,
    prim_path="/World/Robot"
)
```

### From Asset Store

```python title="asset_import.py"
"""Import robot from Isaac Assets."""
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.utils.stage import add_reference_to_stage

assets_root = get_assets_root_path()

# Available robots include:
robots = {
    "carter": "/Isaac/Robots/Carter/carter_v1.usd",
    "jetbot": "/Isaac/Robots/Jetbot/jetbot.usd",
    "franka": "/Isaac/Robots/Franka/franka.usd",
    "ur10": "/Isaac/Robots/UniversalRobots/ur10/ur10.usd",
    "anymal": "/Isaac/Robots/ANYbotics/anymal_c.usd",
}

# Import Franka arm
add_reference_to_stage(
    usd_path=assets_root + robots["franka"],
    prim_path="/World/Franka"
)
```

---

## Sensor Simulation

### Camera Setup

```python title="camera_sensor.py"
"""Configure camera sensor in Isaac Sim."""
from omni.isaac.sensor import Camera
from omni.isaac.core import World

world = World()

# Create camera
camera = Camera(
    prim_path="/World/Robot/camera_link/Camera",
    name="front_camera",
    resolution=(640, 480),
    frequency=30
)

# Configure camera properties
camera.set_focal_length(2.0)  # mm
camera.set_focus_distance(1.0)  # meters
camera.set_horizontal_aperture(5.0)  # mm
camera.set_clipping_range(0.1, 100.0)  # near, far

# Initialize
camera.initialize()
world.reset()

# Get camera data
world.step(render=True)
rgb_data = camera.get_rgba()
depth_data = camera.get_depth()

print(f"RGB shape: {rgb_data.shape}")      # (480, 640, 4)
print(f"Depth shape: {depth_data.shape}")  # (480, 640)
```

### LiDAR Setup

```python title="lidar_sensor.py"
"""Configure LiDAR sensor in Isaac Sim."""
from omni.isaac.range_sensor import LidarRtx
from omni.isaac.core import World

world = World()

# Create LiDAR (RTX-based)
lidar = LidarRtx(
    prim_path="/World/Robot/lidar_link/Lidar",
    name="front_lidar",
    rotation_frequency=10.0,  # Hz
    high_lod=False,  # True for higher quality
    horizontal_fov=360.0,
    vertical_fov=30.0,
    horizontal_resolution=0.5,  # degrees
    vertical_resolution=2.0,
    min_range=0.1,
    max_range=100.0
)

lidar.initialize()
world.reset()

# Get point cloud
world.step(render=True)
point_cloud = lidar.get_point_cloud_data()

print(f"Points: {len(point_cloud)}")
```

### IMU Setup

```python title="imu_sensor.py"
"""Configure IMU sensor in Isaac Sim."""
from omni.isaac.sensor import IMUSensor
from omni.isaac.core import World

world = World()

# Create IMU
imu = IMUSensor(
    prim_path="/World/Robot/base_link/IMU",
    name="base_imu",
    frequency=100,  # Hz
    linear_acceleration_noise=0.1,  # m/s²
    angular_velocity_noise=0.01,  # rad/s
    orientation_noise=0.001  # radians
)

imu.initialize()
world.reset()

# Get IMU data
world.step(render=True)
imu_data = imu.get_current_frame()

print(f"Linear acceleration: {imu_data['linear_acceleration']}")
print(f"Angular velocity: {imu_data['angular_velocity']}")
print(f"Orientation: {imu_data['orientation']}")
```

---

## ROS 2 Integration

### Enable ROS 2 Bridge

```python title="ros2_bridge.py"
"""Enable ROS 2 bridge in Isaac Sim."""
import omni
from omni.isaac.core.utils.extensions import enable_extension

# Enable ROS 2 bridge extension
enable_extension("omni.isaac.ros2_bridge")

# Import ROS 2 components
import rclpy
from rclpy.node import Node
from omni.isaac.ros2_bridge import SimulationContext
```

### Publish Camera to ROS 2

```python title="ros2_camera.py"
"""Publish camera data to ROS 2."""
import omni.graph.core as og
from omni.isaac.core import World
from omni.isaac.ros2_bridge import ROSContext

# Create ROS 2 context
ros_context = ROSContext()

# Create camera publisher using OmniGraph
keys = og.Controller.Keys
og.Controller.edit(
    {"graph_path": "/ActionGraph", "evaluator_name": "execution"},
    {
        keys.CREATE_NODES: [
            ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),
            ("CameraHelper", "omni.isaac.ros2_bridge.ROS2CameraHelper"),
        ],
        keys.CONNECT: [
            ("OnPlaybackTick.outputs:tick", "CameraHelper.inputs:execIn"),
        ],
        keys.SET_VALUES: [
            ("CameraHelper.inputs:cameraPrim", "/World/Robot/Camera"),
            ("CameraHelper.inputs:topicName", "/camera/image_raw"),
            ("CameraHelper.inputs:frameId", "camera_link"),
            ("CameraHelper.inputs:type", "rgb"),
        ],
    }
)

# Also publish camera_info
og.Controller.edit(
    {"graph_path": "/ActionGraph"},
    {
        keys.CREATE_NODES: [
            ("CameraInfoHelper", "omni.isaac.ros2_bridge.ROS2CameraInfoHelper"),
        ],
        keys.CONNECT: [
            ("OnPlaybackTick.outputs:tick", "CameraInfoHelper.inputs:execIn"),
        ],
        keys.SET_VALUES: [
            ("CameraInfoHelper.inputs:cameraPrim", "/World/Robot/Camera"),
            ("CameraInfoHelper.inputs:topicName", "/camera/camera_info"),
            ("CameraInfoHelper.inputs:frameId", "camera_link"),
        ],
    }
)
```

### Publish TF

```python title="ros2_tf.py"
"""Publish TF transforms to ROS 2."""
import omni.graph.core as og

keys = og.Controller.Keys
og.Controller.edit(
    {"graph_path": "/ActionGraph"},
    {
        keys.CREATE_NODES: [
            ("PublishTF", "omni.isaac.ros2_bridge.ROS2PublishTransformTree"),
        ],
        keys.CONNECT: [
            ("OnPlaybackTick.outputs:tick", "PublishTF.inputs:execIn"),
        ],
        keys.SET_VALUES: [
            ("PublishTF.inputs:targetPrims", ["/World/Robot"]),
        ],
    }
)
```

---

## Synthetic Data Generation

### Replicator for Domain Randomization

```python title="replicator_example.py"
"""Generate synthetic data with domain randomization."""
import omni.replicator.core as rep

# Setup scene
with rep.new_layer():
    # Create camera
    camera = rep.create.camera(position=(0, 0, 2), look_at=(0, 0, 0))

    # Create render product
    render_product = rep.create.render_product(camera, (640, 480))

    # Domain randomization
    with rep.trigger.on_frame(num_frames=1000):
        # Randomize lighting
        with rep.get.light():
            rep.modify.pose(
                position=rep.distribution.uniform((-5, -5, 3), (5, 5, 6)),
                rotation=rep.distribution.uniform((0, -90, -90), (0, 90, 90))
            )
            rep.modify.light_intensity(
                rep.distribution.uniform(500, 2000)
            )

        # Randomize object positions
        with rep.get.prims(semantics=[("class", "target")]):
            rep.modify.pose(
                position=rep.distribution.uniform((-1, -1, 0), (1, 1, 0.5)),
                rotation=rep.distribution.uniform((0, 0, 0), (0, 0, 360))
            )

        # Randomize textures
        with rep.get.prims(semantics=[("class", "background")]):
            rep.randomizer.texture(
                textures=rep.distribution.choice([
                    "omni://textures/wood.png",
                    "omni://textures/metal.png",
                    "omni://textures/concrete.png",
                ])
            )

    # Setup output
    writer = rep.WriterRegistry.get("BasicWriter")
    writer.initialize(
        output_dir="./synthetic_data",
        rgb=True,
        bounding_box_2d_tight=True,
        semantic_segmentation=True,
        instance_segmentation=True,
    )
    writer.attach([render_product])

# Run data generation
rep.orchestrator.run()
```

### Output Formats

Isaac Sim can export to various formats:

| Format | Use Case | Output |
|--------|----------|--------|
| **COCO** | Object detection | JSON annotations |
| **KITTI** | Autonomous driving | txt + calibration |
| **Cityscapes** | Segmentation | PNG masks |
| **Custom** | Any | Scriptable |

---

## Exercise 1: Build a Complete Scene

:::tip Exercise 1: Warehouse Simulation
**Objective**: Create a warehouse scene with a mobile robot.

**Steps**:

1. Create new stage in Isaac Sim
2. Add ground plane with warehouse texture
3. Import Carter robot from Isaac Assets
4. Add shelves and boxes as obstacles
5. Set up camera and LiDAR on robot
6. Enable physics and run simulation

**Expected Result**: Robot with working sensors in warehouse environment.

**Time Estimate**: 45 minutes
:::

---

## Exercise 2: ROS 2 Integration

:::tip Exercise 2: ROS 2 Bridge
**Objective**: Stream Isaac Sim sensors to ROS 2.

**Steps**:

1. Enable ROS 2 bridge extension
2. Configure camera publisher
3. Configure LiDAR publisher
4. Add cmd_vel subscriber
5. Run simulation and verify with `ros2 topic list`

**Verification**:
```bash
# In another terminal
ros2 topic list
# Should see /camera/image_raw, /scan, etc.

ros2 topic hz /camera/image_raw
# Should show ~30 Hz
```

**Time Estimate**: 30 minutes
:::

---

## Exercise 3: Synthetic Data Pipeline

:::tip Exercise 3: Generate Training Data
**Objective**: Create synthetic dataset for object detection.

**Steps**:

1. Create scene with target objects (boxes, cylinders)
2. Add semantic labels to objects
3. Configure Replicator with domain randomization
4. Generate 100 images with annotations
5. Verify output in COCO format

**Expected Output**:
- 100 RGB images
- 100 corresponding bounding box annotations
- COCO-format `annotations.json`

**Time Estimate**: 60 minutes
:::

---

## Summary

In this chapter, you learned:

- **Interface**: Navigate Isaac Sim and understand USD structure
- **Robots**: Import from URDF or Isaac Assets
- **Sensors**: Configure camera, LiDAR, and IMU with noise models
- **ROS 2**: Bridge simulation to ROS 2 topics
- **Synthetic Data**: Generate training datasets with domain randomization

Isaac Sim's photorealistic rendering and accurate physics make it invaluable for:
- Training perception models with synthetic data
- Validating robot behaviors before deployment
- Creating digital twins of real environments

Next, learn about [Isaac ROS](/docs/module-3/isaac-ros) for deploying GPU-accelerated perception on real robots.

## Further Reading

- [Isaac Sim Documentation](https://docs.omniverse.nvidia.com/isaacsim)
- [USD Specification](https://openusd.org/release/spec.html)
- [Replicator Documentation](https://docs.omniverse.nvidia.com/replicator)
- [Isaac Sim Tutorials](https://docs.omniverse.nvidia.com/isaacsim/latest/tutorials.html)
