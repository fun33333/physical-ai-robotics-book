---
title: "Isaac ROS"
sidebar_position: 4
description: "GPU-accelerated perception with Isaac ROS: optimized packages for detection, segmentation, and SLAM."
---

# Isaac ROS

Isaac ROS provides GPU-accelerated implementations of common robotics perception algorithms, offering significant speedups over CPU-only alternatives. This chapter shows you how to deploy Isaac ROS packages in your perception pipeline for real-time performance on NVIDIA hardware.

## Overview

In this section, you will:

- Set up Isaac ROS development environment
- Deploy pre-trained detection and segmentation models
- Implement visual SLAM with cuVSLAM
- Build multi-camera perception pipelines
- Optimize inference with TensorRT
- Integrate with existing ROS 2 systems

## Prerequisites

- Ubuntu 22.04 with ROS 2 Humble
- NVIDIA GPU (RTX series or Jetson)
- Docker with NVIDIA Container Toolkit
- NGC account with API key
- Completed [Isaac Platform Overview](/docs/module-3/isaac-platform-overview)

---

## Isaac ROS Architecture

### Package Structure

```
┌─────────────────────────────────────────────────────────────────┐
│                    Isaac ROS Package Hierarchy                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │                   Application Layer                      │   │
│   │  isaac_ros_examples • Your Custom Nodes                  │   │
│   └─────────────────────────────────────────────────────────┘   │
│                              │                                   │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │                  Perception Packages                     │   │
│   │  ┌─────────────┐ ┌──────────────┐ ┌─────────────────┐  │   │
│   │  │ Detection   │ │ Segmentation │ │    SLAM         │  │   │
│   │  │             │ │              │ │                 │  │   │
│   │  │ detectnet   │ │ segformer    │ │ visual_slam     │  │   │
│   │  │ yolov8      │ │ unet        │ │ nvblox          │  │   │
│   │  │ rtdetr      │ │ bi3d         │ │                 │  │   │
│   │  └─────────────┘ └──────────────┘ └─────────────────┘  │   │
│   └─────────────────────────────────────────────────────────┘   │
│                              │                                   │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │                   Core Infrastructure                    │   │
│   │  ┌───────────────┐ ┌─────────────┐ ┌─────────────────┐ │   │
│   │  │ dnn_inference │ │image_pipeline│ │ depth_image_proc│ │   │
│   │  │ (TensorRT)    │ │(GPU resize) │ │  (CUDA accel)   │ │   │
│   │  └───────────────┘ └─────────────┘ └─────────────────┘ │   │
│   └─────────────────────────────────────────────────────────┘   │
│                              │                                   │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │                        NITROS                            │   │
│   │           Zero-copy GPU memory transfer                  │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Key Packages

| Package | Function | Performance |
|---------|----------|-------------|
| `isaac_ros_dnn_inference` | TensorRT inference | Up to 50x vs CPU |
| `isaac_ros_visual_slam` | cuVSLAM | Real-time 60 FPS |
| `isaac_ros_apriltag` | Fiducial detection | 10x vs CPU |
| `isaac_ros_image_pipeline` | Image processing | GPU-accelerated |
| `isaac_ros_depth_image_proc` | Depth processing | CUDA kernels |
| `isaac_ros_detectnet` | Object detection | 30+ FPS on RTX |
| `isaac_ros_segformer` | Semantic segmentation | Real-time |

---

## Setup

### Docker-Based Development

```bash title="Isaac ROS Docker setup"
# 1. Clone Isaac ROS common
mkdir -p ~/workspaces/isaac_ros-dev/src
cd ~/workspaces/isaac_ros-dev/src
git clone https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_common.git

# 2. Clone desired packages
git clone https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_visual_slam.git
git clone https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_object_detection.git
git clone https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_dnn_inference.git

# 3. Launch development container
cd ~/workspaces/isaac_ros-dev/src/isaac_ros_common
./scripts/run_dev.sh

# Inside container:
# 4. Build packages
cd /workspaces/isaac_ros-dev
colcon build --symlink-install

# 5. Source workspace
source install/setup.bash
```

### Native Installation (Advanced)

```bash title="Native Isaac ROS installation"
# Install dependencies
sudo apt install -y \
  ros-humble-vision-msgs \
  ros-humble-cv-bridge \
  ros-humble-image-transport

# Install Isaac ROS apt packages
sudo apt-key adv --fetch-keys https://isaac.download.nvidia.com/isaac-ros/repos.key
echo 'deb https://isaac.download.nvidia.com/isaac-ros/release-3 jammy main' | \
  sudo tee /etc/apt/sources.list.d/isaac-ros.list

sudo apt update
sudo apt install -y \
  ros-humble-isaac-ros-visual-slam \
  ros-humble-isaac-ros-apriltag \
  ros-humble-isaac-ros-dnn-inference
```

---

## Object Detection

### Using DetectNet

```bash title="Launch DetectNet"
# Download model (inside Docker)
mkdir -p /tmp/models
cd /tmp/models
wget https://api.ngc.nvidia.com/v2/models/nvidia/tao/peoplenet/versions/pruned_quantized_decrypted_v2.3.3/files/resnet34_peoplenet_int8.etlt

# Launch detection
ros2 launch isaac_ros_detectnet isaac_ros_detectnet.launch.py \
  model_file_path:=/tmp/models/resnet34_peoplenet_int8.etlt \
  engine_file_path:=/tmp/models/resnet34_peoplenet_int8.plan \
  input_binding_names:=['input_1'] \
  output_binding_names:=['output_cov/Sigmoid', 'output_bbox/BiasAdd'] \
  network_image_width:=960 \
  network_image_height:=544 \
  input_image_topic:=/camera/image_raw
```

### Custom Detection Model

```python title="custom_detection_node.py"
"""Custom detection using Isaac ROS DNN inference."""
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray
from isaac_ros_tensor_list_interfaces.msg import TensorList

class DetectionNode(Node):
    def __init__(self):
        super().__init__('custom_detection')

        # Subscribe to inference output
        self.tensor_sub = self.create_subscription(
            TensorList,
            '/tensor_pub',
            self.tensor_callback,
            10
        )

        # Publish detections
        self.detection_pub = self.create_publisher(
            Detection2DArray,
            '/detections',
            10
        )

        self.get_logger().info('Detection node ready')

    def tensor_callback(self, msg: TensorList):
        """Process inference output tensors."""
        detections = Detection2DArray()
        detections.header = msg.header

        # Parse model-specific output format
        # (Depends on your model architecture)
        for tensor in msg.tensors:
            # Process tensor.data based on model
            pass

        self.detection_pub.publish(detections)


def main():
    rclpy.init()
    node = DetectionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
```

---

## Visual SLAM

### cuVSLAM Setup

cuVSLAM is NVIDIA's GPU-accelerated visual SLAM implementation:

```bash title="Launch cuVSLAM"
# For stereo camera
ros2 launch isaac_ros_visual_slam isaac_ros_visual_slam.launch.py \
  enable_rectified_pose:=True \
  enable_imu:=True \
  enable_slam_visualization:=True \
  camera_info_topic:=/camera/left/camera_info \
  image_topic:=/camera/left/image_raw \
  right_camera_info_topic:=/camera/right/camera_info \
  right_image_topic:=/camera/right/image_raw \
  imu_topic:=/imu

# For RealSense camera
ros2 launch isaac_ros_visual_slam isaac_ros_visual_slam_realsense.launch.py
```

### cuVSLAM Configuration

```yaml title="vslam_config.yaml"
/**:
  ros__parameters:
    # Camera configuration
    base_frame: "base_link"
    map_frame: "map"
    odom_frame: "odom"

    # Feature tracking
    num_cameras: 2
    enable_observations_view: true
    enable_landmarks_view: true

    # IMU integration
    enable_imu_fusion: true
    gyro_noise_density: 0.00016
    gyro_random_walk: 0.000002
    accel_noise_density: 0.0006
    accel_random_walk: 0.0003

    # Performance tuning
    image_jitter_threshold_ms: 35.0
    enable_debug_mode: false
    enable_localization_n_mapping: true
```

### SLAM Output Topics

| Topic | Type | Description |
|-------|------|-------------|
| `/visual_slam/tracking/odometry` | Odometry | 6DOF pose at 60+ Hz |
| `/visual_slam/tracking/slam_path` | Path | Trajectory history |
| `/visual_slam/vis/landmarks_cloud` | PointCloud2 | 3D feature map |
| `/visual_slam/status` | DiagnosticArray | Tracking status |

---

## Semantic Segmentation

### Using SegFormer

```bash title="Launch SegFormer"
# Download model
mkdir -p /tmp/models
# Get model from NGC catalog

# Launch segmentation
ros2 launch isaac_ros_segformer isaac_ros_segformer.launch.py \
  model_file_path:=/tmp/models/segformer.onnx \
  engine_file_path:=/tmp/models/segformer.plan \
  input_image_topic:=/camera/image_raw \
  network_image_width:=1024 \
  network_image_height:=1024
```

### Segmentation Classes

Default SegFormer outputs include:

| Class ID | Label | Color (RGB) |
|----------|-------|-------------|
| 0 | Background | (0, 0, 0) |
| 1 | Person | (255, 0, 0) |
| 2 | Vehicle | (0, 255, 0) |
| 3 | Road | (128, 128, 128) |
| ... | ... | ... |

---

## Multi-Camera Pipeline

### Pipeline Architecture

```python title="multi_camera_pipeline.launch.py"
"""Multi-camera perception pipeline."""
from launch import LaunchDescription
from launch_ros.actions import Node, ComposableNodeContainer
from launch_ros.descriptions import ComposableNode

def generate_launch_description():
    # Define cameras
    cameras = ['front', 'left', 'right']

    nodes = []

    for camera in cameras:
        # Image rectification
        rectify_node = ComposableNode(
            package='isaac_ros_image_pipeline',
            plugin='nvidia::isaac_ros::image_pipeline::RectifyNode',
            name=f'{camera}_rectify',
            parameters=[{
                'output_width': 640,
                'output_height': 480,
            }],
            remappings=[
                ('image_raw', f'/{camera}/image_raw'),
                ('camera_info', f'/{camera}/camera_info'),
                ('image_rect', f'/{camera}/image_rect'),
            ]
        )
        nodes.append(rectify_node)

    # Detection (shared across cameras)
    detection_node = ComposableNode(
        package='isaac_ros_detectnet',
        plugin='nvidia::isaac_ros::detectnet::DetectNetDecoderNode',
        name='detectnet',
        parameters=[{
            'model_file_path': '/tmp/models/peoplenet.plan',
        }]
    )
    nodes.append(detection_node)

    # Container for zero-copy communication
    container = ComposableNodeContainer(
        name='perception_container',
        namespace='',
        package='rclcpp_components',
        executable='component_container_mt',
        composable_node_descriptions=nodes,
        output='screen'
    )

    return LaunchDescription([container])
```

---

## Performance Optimization

### TensorRT Engine Generation

```bash title="Generate TensorRT engine"
# Convert ONNX to TensorRT
/usr/src/tensorrt/bin/trtexec \
  --onnx=/tmp/models/model.onnx \
  --saveEngine=/tmp/models/model.plan \
  --fp16 \
  --workspace=4096 \
  --verbose

# For INT8 quantization (requires calibration data)
/usr/src/tensorrt/bin/trtexec \
  --onnx=/tmp/models/model.onnx \
  --saveEngine=/tmp/models/model_int8.plan \
  --int8 \
  --calib=/tmp/calibration_cache.bin
```

### NITROS Optimization

```python title="Enable NITROS"
"""Enable NITROS for zero-copy GPU transfer."""
from launch_ros.descriptions import ComposableNode

# NITROS-enabled node configuration
node = ComposableNode(
    package='isaac_ros_dnn_inference',
    plugin='nvidia::isaac_ros::dnn_inference::TensorRTNode',
    name='tensorrt_node',
    parameters=[{
        # NITROS is enabled by default in composable nodes
        'enable_nitros': True,
        'nitros_format': 'nitros_tensor_list_nchw_rgb_f32',
    }]
)
```

### Benchmark Results

| Task | CPU (Intel i7) | GPU (RTX 3060) | Speedup |
|------|----------------|----------------|---------|
| YOLOv8 (640x640) | 50ms | 5ms | 10x |
| SegFormer | 200ms | 15ms | 13x |
| Visual SLAM | 100ms | 8ms | 12x |
| Depth estimation | 80ms | 6ms | 13x |

---

## Exercise 1: Object Detection Pipeline

:::tip Exercise 1: Deploy PeopleNet
**Objective**: Set up real-time people detection.

**Steps**:

1. Launch Isaac ROS container
2. Download PeopleNet model from NGC
3. Generate TensorRT engine
4. Launch DetectNet node
5. Visualize detections in RViz2

**Verification**:
```bash
# Check detection rate
ros2 topic hz /detectnet/detections
# Should be 20+ Hz

# View detections
ros2 topic echo /detectnet/detections
```

**Time Estimate**: 30 minutes
:::

---

## Exercise 2: Visual SLAM Integration

:::tip Exercise 2: cuVSLAM with RealSense
**Objective**: Run visual SLAM with Intel RealSense camera.

**Steps**:

1. Connect RealSense D435i camera
2. Launch RealSense ROS 2 driver
3. Configure cuVSLAM for RealSense
4. Run SLAM and build map
5. Save map for localization

**Expected Output**:
- Odometry at 60 Hz
- 3D landmark map
- Trajectory visualization

**Time Estimate**: 45 minutes
:::

---

## Exercise 3: Custom Model Deployment

:::tip Exercise 3: Deploy Your Own Model
**Objective**: Convert and deploy a custom detection model.

**Steps**:

1. Export PyTorch model to ONNX
2. Generate TensorRT engine with `trtexec`
3. Create Isaac ROS launch file
4. Write decoder node for your model output
5. Test with camera input

**Model Export**:
```python
import torch
model = YourModel()
model.load_state_dict(torch.load('weights.pth'))
model.eval()

dummy_input = torch.randn(1, 3, 640, 640)
torch.onnx.export(model, dummy_input, 'model.onnx',
                  opset_version=13,
                  input_names=['input'],
                  output_names=['output'])
```

**Time Estimate**: 90 minutes
:::

---

## Summary

In this chapter, you learned:

- **Setup**: Deploy Isaac ROS via Docker or native installation
- **Detection**: Run real-time object detection with DetectNet
- **SLAM**: Implement visual SLAM with cuVSLAM
- **Segmentation**: Deploy semantic segmentation models
- **Optimization**: Use TensorRT and NITROS for maximum performance

Isaac ROS enables perception capabilities that would be impossible with CPU-only approaches, making real-time robotics applications practical on embedded and desktop hardware.

Next, explore [Isaac Lab](/docs/module-3/isaac-lab) to learn reinforcement learning for robot control.

## Further Reading

- [Isaac ROS Documentation](https://nvidia-isaac-ros.github.io/)
- [Isaac ROS GitHub](https://github.com/NVIDIA-ISAAC-ROS)
- [TensorRT Documentation](https://developer.nvidia.com/tensorrt)
- [cuVSLAM Paper](https://developer.nvidia.com/isaac-ros-visual-slam)
