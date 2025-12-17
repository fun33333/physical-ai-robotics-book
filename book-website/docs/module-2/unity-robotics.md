---
title: "Unity for Robotics"
sidebar_position: 5
description: "Using Unity for robotics simulation: photorealistic rendering, synthetic data, and ROS 2 integration."
---

# Unity for Robotics

Unity brings game engine technology to robotics, offering photorealistic rendering, efficient synthetic data generation, and a massive ecosystem of assets and tools. This chapter introduces Unity for robotics applications, focusing on perception training and ROS 2 integration.

## Overview

In this section, you will:

- Set up Unity for robotics development with the Robotics Hub
- Import and configure robot models (URDF to Unity)
- Configure articulation bodies for physics simulation
- Generate synthetic datasets for perception training
- Integrate Unity with ROS 2 via TCP connector
- Create domain randomization pipelines for robust AI

## Prerequisites

Before starting, ensure you have:

- [Gazebo Basics](/docs/module-2/gazebo-basics) completed
- Ubuntu 22.04 or Windows 10/11
- NVIDIA GPU with 4GB+ VRAM (recommended)
- Unity Hub installed
- Basic understanding of 3D graphics concepts

---

## Why Unity for Robotics?

### Comparison with Gazebo

| Feature | Gazebo | Unity |
|---------|--------|-------|
| Physics accuracy | High | Medium-High |
| Visual quality | Medium | Photorealistic |
| ROS integration | Native | Via connector |
| Asset ecosystem | Robotics-focused | Massive marketplace |
| Synthetic data | Basic | Excellent |
| Learning curve | Moderate | Steeper |
| Cost | Free (OSS) | Free tier available |

### When to Use Unity

Unity excels for:

- **Perception training**: Generate synthetic images for object detection, segmentation
- **Synthetic data at scale**: Domain randomization for robust models
- **Visualization**: Create compelling demos and presentations
- **Cross-platform deployment**: Build for VR, mobile, web
- **Custom environments**: Leverage asset store for diverse scenes

:::info Best Practice
Use Gazebo for control and navigation development, Unity for perception and synthetic data generation. Many teams use both in their pipeline.
:::

---

## Setting Up Unity for Robotics

### Installing Unity Hub

```bash title="Install Unity Hub on Ubuntu"
# Add Unity repository
wget -qO - https://hub.unity3d.com/linux/keys/public | sudo apt-key add -
sudo sh -c 'echo "deb https://hub.unity3d.com/linux/repos/deb stable main" > /etc/apt/sources.list.d/unityhub.list'

# Install Unity Hub
sudo apt update
sudo apt install unityhub

# Launch Unity Hub
unityhub
```

### Recommended Unity Version

For robotics simulation, use Unity **2022.3 LTS** or later:

1. Open Unity Hub
2. Click "Installs" → "Install Editor"
3. Select 2022.3.x LTS
4. Add modules:
   - Linux Build Support (if on Windows/Mac)
   - Visual Studio / VS Code integration

### Installing Robotics Packages

Create a new 3D project, then add packages:

1. Open **Window → Package Manager**
2. Click **+ → Add package from git URL**
3. Add these packages:

```text
# Unity Robotics Hub
https://github.com/Unity-Technologies/Unity-Robotics-Hub.git?path=/com.unity.robotics.ros-tcp-connector

# URDF Importer
https://github.com/Unity-Technologies/URDF-Importer.git?path=/com.unity.robotics.urdf-importer

# Perception (for synthetic data)
com.unity.perception
```

---

## Importing Robot Models

### URDF Import Process

Unity's URDF Importer converts URDF files to Unity GameObjects with ArticulationBodies:

```
┌─────────────────────────────────────────────────────────────────┐
│                    URDF Import Pipeline                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   ┌──────────┐     ┌──────────┐     ┌──────────────────┐       │
│   │  URDF    │     │  Parse   │     │   Unity          │       │
│   │  File    │────▶│  Links   │────▶│   GameObject     │       │
│   │  (.urdf) │     │  Joints  │     │   Hierarchy      │       │
│   └──────────┘     └──────────┘     └──────────────────┘       │
│        │                                     │                  │
│        │           ┌──────────────┐         │                  │
│        └──────────▶│    Meshes    │─────────┘                  │
│                    │  Colliders   │                             │
│                    │  Materials   │                             │
│                    └──────────────┘                             │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Step-by-Step Import

1. **Prepare your URDF**:
   - Ensure all mesh paths are relative
   - Include `<visual>`, `<collision>`, and `<inertial>` tags
   - Package meshes with the URDF file

2. **Import in Unity**:
   ```
   Assets → Import Robot from URDF
   ```

3. **Configure import settings**:

```csharp title="URDF Import Settings (Editor)"
// Access via Import Robot dialog
public class URDFImportSettings
{
    // Physics settings
    public bool useArticulationBodies = true;  // Recommended for robots
    public float defaultMass = 1.0f;           // For links without mass

    // Visual settings
    public bool importVisuals = true;
    public bool importCollisions = true;

    // Joint settings
    public float stiffness = 10000f;
    public float damping = 100f;
}
```

### Articulation Bodies

Unity uses ArticulationBodies for robot physics - a system designed for robotics:

```csharp title="ArticulationBody Configuration"
using UnityEngine;

public class RobotJointController : MonoBehaviour
{
    private ArticulationBody[] joints;

    void Start()
    {
        // Get all articulation bodies in hierarchy
        joints = GetComponentsInChildren<ArticulationBody>();

        foreach (var joint in joints)
        {
            // Configure joint drives for position control
            var drive = joint.xDrive;
            drive.stiffness = 10000f;  // Position gain
            drive.damping = 100f;       // Velocity damping
            drive.forceLimit = 1000f;   // Max torque
            joint.xDrive = drive;
        }
    }

    public void SetJointTarget(int jointIndex, float targetPosition)
    {
        if (jointIndex < joints.Length)
        {
            var drive = joints[jointIndex].xDrive;
            drive.target = targetPosition * Mathf.Rad2Deg;  // Convert to degrees
            joints[jointIndex].xDrive = drive;
        }
    }
}
```

---

## ROS 2 Integration

### ROS TCP Connector Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                  Unity-ROS 2 Communication                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   ┌──────────────┐                    ┌──────────────┐          │
│   │    Unity     │     TCP/IP         │    ROS 2     │          │
│   │  Simulation  │◄──────────────────▶│    Nodes     │          │
│   └──────────────┘                    └──────────────┘          │
│          │                                   │                   │
│          │  ┌─────────────────────────┐     │                   │
│          └──│   ROS TCP Connector     │─────┘                   │
│             │   (Unity Package)       │                         │
│             └─────────────────────────┘                         │
│                        │                                         │
│             ┌─────────────────────────┐                         │
│             │  ROS TCP Endpoint       │                         │
│             │  (ROS 2 Node)           │                         │
│             └─────────────────────────┘                         │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Setting Up ROS TCP Endpoint

On your ROS 2 machine:

```bash title="Install and run ROS TCP Endpoint"
# Install the ROS TCP Endpoint package
cd ~/ros2_ws/src
git clone -b main https://github.com/Unity-Technologies/ROS-TCP-Endpoint.git

# Build
cd ~/ros2_ws
colcon build --packages-select ros_tcp_endpoint

# Source and run
source install/setup.bash
ros2 run ros_tcp_endpoint default_server_endpoint --ros-args -p ROS_IP:=0.0.0.0
```

### Unity-Side Configuration

```csharp title="ROSConnection Setup"
using Unity.Robotics.ROSTCPConnector;
using RosMessageTypes.Geometry;
using RosMessageTypes.Sensor;

public class ROS2Interface : MonoBehaviour
{
    private ROSConnection ros;

    void Start()
    {
        // Get ROSConnection instance
        ros = ROSConnection.GetOrCreateInstance();
        ros.ConnectOnStart = true;

        // Configure connection (also settable in Inspector)
        // ROS IP: 127.0.0.1 (or your ROS machine IP)
        // ROS Port: 10000

        // Register publishers
        ros.RegisterPublisher<TwistMsg>("/cmd_vel");
        ros.RegisterPublisher<ImageMsg>("/camera/image_raw");

        // Register subscribers
        ros.Subscribe<LaserScanMsg>("/scan", OnLaserScan);
        ros.Subscribe<JointStateMsg>("/joint_states", OnJointStates);
    }

    void OnLaserScan(LaserScanMsg msg)
    {
        // Process incoming LiDAR data
        Debug.Log($"Received {msg.ranges.Length} range readings");
    }

    void OnJointStates(JointStateMsg msg)
    {
        // Update robot visualization
        for (int i = 0; i < msg.name.Length; i++)
        {
            Debug.Log($"Joint {msg.name[i]}: {msg.position[i]}");
        }
    }

    public void PublishCmdVel(float linear, float angular)
    {
        var msg = new TwistMsg
        {
            linear = new Vector3Msg { x = linear },
            angular = new Vector3Msg { z = angular }
        };
        ros.Publish("/cmd_vel", msg);
    }
}
```

---

## Synthetic Data Generation

### Unity Perception Package

The Perception package enables large-scale synthetic data generation:

```csharp title="Setting up Perception Camera"
using UnityEngine;
using UnityEngine.Perception.GroundTruth;
using UnityEngine.Perception.Randomization.Scenarios;

public class SyntheticDataCamera : MonoBehaviour
{
    void Start()
    {
        // Add Perception Camera component
        var perceptionCamera = gameObject.AddComponent<PerceptionCamera>();

        // Configure labelers for different annotations
        // - Bounding Box 2D: Object detection
        // - Semantic Segmentation: Pixel-wise labels
        // - Instance Segmentation: Per-object masks
        // - Keypoint: Pose estimation
    }
}
```

### Domain Randomization Setup

```csharp title="RandomizerConfiguration.cs"
using UnityEngine;
using UnityEngine.Perception.Randomization.Randomizers;
using UnityEngine.Perception.Randomization.Parameters;

// Randomize lighting conditions
[AddRandomizerMenu("Perception/Light Randomizer")]
public class LightRandomizer : Randomizer
{
    public FloatParameter intensity = new FloatParameter { value = new UniformSampler(0.5f, 2.0f) };
    public ColorHsvaParameter color = new ColorHsvaParameter();

    protected override void OnIterationStart()
    {
        var lights = tagManager.Query<LightRandomizerTag>();
        foreach (var light in lights)
        {
            var l = light.GetComponent<Light>();
            l.intensity = intensity.Sample();
            l.color = color.Sample();
        }
    }
}

// Randomize object positions
[AddRandomizerMenu("Perception/Pose Randomizer")]
public class PoseRandomizer : Randomizer
{
    public Vector3Parameter position = new Vector3Parameter();
    public Vector3Parameter rotation = new Vector3Parameter();

    protected override void OnIterationStart()
    {
        var objects = tagManager.Query<PoseRandomizerTag>();
        foreach (var obj in objects)
        {
            obj.transform.position = position.Sample();
            obj.transform.eulerAngles = rotation.Sample();
        }
    }
}
```

### Dataset Export

```csharp title="DatasetExporter.cs"
using UnityEngine;
using UnityEngine.Perception.GroundTruth;

public class DatasetExporter : MonoBehaviour
{
    // Perception package automatically exports to:
    // Windows: %USERPROFILE%\AppData\LocalLow\DefaultCompany\ProjectName\
    // Linux: ~/.config/unity3d/DefaultCompany/ProjectName/

    // Output formats:
    // - COCO (object detection)
    // - SOLO (Unity's native format)
    // - Custom (via DatasetConsumer)

    void ConfigureExport()
    {
        // Dataset is exported automatically during simulation
        // Configure in Perception Camera component:
        // - Capture trigger mode (scheduled, manual)
        // - Frames between captures
        // - Start frame
    }
}
```

---

## Simulating Sensors

### Camera Sensor

```csharp title="CameraSensor.cs"
using UnityEngine;
using Unity.Robotics.ROSTCPConnector;
using RosMessageTypes.Sensor;

public class CameraSensor : MonoBehaviour
{
    public Camera sensorCamera;
    public int width = 640;
    public int height = 480;
    public float publishRate = 30f;

    private ROSConnection ros;
    private RenderTexture renderTexture;
    private Texture2D texture2D;
    private float timeSincePublish = 0f;

    void Start()
    {
        ros = ROSConnection.GetOrCreateInstance();
        ros.RegisterPublisher<ImageMsg>("/camera/image_raw");

        // Create render texture
        renderTexture = new RenderTexture(width, height, 24);
        sensorCamera.targetTexture = renderTexture;
        texture2D = new Texture2D(width, height, TextureFormat.RGB24, false);
    }

    void Update()
    {
        timeSincePublish += Time.deltaTime;

        if (timeSincePublish >= 1f / publishRate)
        {
            PublishImage();
            timeSincePublish = 0f;
        }
    }

    void PublishImage()
    {
        // Render camera to texture
        RenderTexture.active = renderTexture;
        texture2D.ReadPixels(new Rect(0, 0, width, height), 0, 0);
        texture2D.Apply();
        RenderTexture.active = null;

        // Convert to ROS message
        var msg = new ImageMsg
        {
            header = new RosMessageTypes.Std.HeaderMsg
            {
                stamp = new RosMessageTypes.BuiltinInterfaces.TimeMsg
                {
                    sec = (int)Time.time,
                    nanosec = (uint)((Time.time % 1) * 1e9)
                },
                frame_id = "camera_link"
            },
            width = (uint)width,
            height = (uint)height,
            encoding = "rgb8",
            step = (uint)(width * 3),
            data = texture2D.GetRawTextureData()
        };

        ros.Publish("/camera/image_raw", msg);
    }
}
```

### LiDAR Sensor

```csharp title="LiDARSensor.cs"
using UnityEngine;
using Unity.Robotics.ROSTCPConnector;
using RosMessageTypes.Sensor;

public class LiDARSensor : MonoBehaviour
{
    public int numRays = 360;
    public float maxRange = 30f;
    public float minRange = 0.1f;
    public float publishRate = 10f;

    private ROSConnection ros;
    private float[] ranges;
    private float timeSincePublish = 0f;

    void Start()
    {
        ros = ROSConnection.GetOrCreateInstance();
        ros.RegisterPublisher<LaserScanMsg>("/scan");
        ranges = new float[numRays];
    }

    void Update()
    {
        timeSincePublish += Time.deltaTime;

        if (timeSincePublish >= 1f / publishRate)
        {
            ScanAndPublish();
            timeSincePublish = 0f;
        }
    }

    void ScanAndPublish()
    {
        float angleIncrement = 2 * Mathf.PI / numRays;

        for (int i = 0; i < numRays; i++)
        {
            float angle = i * angleIncrement;
            Vector3 direction = new Vector3(
                Mathf.Cos(angle),
                0,
                Mathf.Sin(angle)
            );
            direction = transform.TransformDirection(direction);

            RaycastHit hit;
            if (Physics.Raycast(transform.position, direction, out hit, maxRange))
            {
                ranges[i] = Mathf.Max(hit.distance, minRange);
            }
            else
            {
                ranges[i] = float.PositiveInfinity;
            }
        }

        var msg = new LaserScanMsg
        {
            header = new RosMessageTypes.Std.HeaderMsg
            {
                stamp = new RosMessageTypes.BuiltinInterfaces.TimeMsg
                {
                    sec = (int)Time.time,
                    nanosec = (uint)((Time.time % 1) * 1e9)
                },
                frame_id = "lidar_link"
            },
            angle_min = 0f,
            angle_max = 2 * Mathf.PI,
            angle_increment = angleIncrement,
            range_min = minRange,
            range_max = maxRange,
            ranges = ranges
        };

        ros.Publish("/scan", msg);
    }
}
```

---

## Exercise 1: Import a Robot to Unity

:::tip Exercise 1: URDF Import and Configuration
**Objective**: Import a robot URDF into Unity and configure physics.

**Time Estimate**: 45 minutes

**Steps**:

1. Create a new Unity project with 3D template
2. Install URDF Importer package via Package Manager
3. Download a robot URDF (e.g., TurtleBot3, UR5)
4. Import via Assets → Import Robot from URDF
5. Configure ArticulationBody settings for realistic physics
6. Add a ground plane and test that robot doesn't fall through

**Expected Result**: Robot model visible in Unity scene with functioning joints.

**Verification**:
- Robot spawns above ground
- Joints have correct limits
- Physics simulation runs without errors
:::

---

## Exercise 2: ROS 2 Connection

:::tip Exercise 2: Unity-ROS 2 Communication
**Objective**: Establish bidirectional communication between Unity and ROS 2.

**Time Estimate**: 30 minutes

**Steps**:

1. Install ROS TCP Connector in Unity
2. Start ROS TCP Endpoint on ROS 2 machine
3. Configure ROSConnection in Unity (IP, port)
4. Create a publisher for `/cmd_vel`
5. Create a subscriber for `/scan`
6. Test with `ros2 topic echo` and `ros2 topic pub`

**ROS 2 Test Commands**:

```bash
# Echo published messages from Unity
ros2 topic echo /cmd_vel

# Publish test message to Unity
ros2 topic pub /scan sensor_msgs/msg/LaserScan \
  "{ranges: [1.0, 2.0, 3.0]}" --once
```

**Expected Result**: Messages flow bidirectionally between Unity and ROS 2.
:::

---

## Exercise 3: Synthetic Dataset Generation

:::tip Exercise 3: Generate Object Detection Dataset
**Objective**: Create a synthetic dataset for training an object detection model.

**Time Estimate**: 60 minutes

**Steps**:

1. Install Unity Perception package
2. Create a simple scene with objects to detect
3. Add labels to objects (via IdLabelConfig)
4. Configure Perception Camera with BoundingBox2D labeler
5. Create a Fixed Length Scenario (100 iterations)
6. Add randomizers for position, lighting, and camera
7. Run simulation and export dataset

**Configuration**:

```
Scenario Settings:
- Iterations: 100
- Frames per iteration: 1

Randomizers:
- Background: Random textures
- Foreground: Random object positions
- Lighting: Random intensity and color
- Camera: Slight position variation
```

**Expected Result**: 100 images with COCO-format annotations in output folder.
:::

---

## Best Practices

### Performance Optimization

| Technique | Impact | When to Use |
|-----------|--------|-------------|
| Lower resolution textures | High | Large scenes |
| Reduce physics iterations | Medium | Many objects |
| Batch ray casts | High | LiDAR simulation |
| GPU instancing | High | Many similar objects |
| Level of Detail (LOD) | Medium | Large environments |

### Sim-to-Real Tips

1. **Match camera intrinsics**: Use real camera calibration
2. **Add realistic noise**: Gaussian noise to sensors
3. **Vary lighting extensively**: Real world has diverse lighting
4. **Include distractors**: Add irrelevant objects to scenes
5. **Test on edge cases**: Unusual poses, occlusions, lighting

---

## Summary

In this chapter, you learned:

- **Unity Setup**: Install Unity Hub, robotics packages, and configure projects
- **URDF Import**: Convert robot descriptions to Unity with ArticulationBodies
- **ROS 2 Integration**: Connect Unity to ROS 2 via TCP connector
- **Sensor Simulation**: Implement camera and LiDAR sensors
- **Synthetic Data**: Generate labeled datasets with domain randomization
- **Best Practices**: Performance optimization and sim-to-real transfer

Unity complements Gazebo by providing photorealistic rendering for perception training. Many teams use Gazebo for control development and Unity for synthetic data generation.

Next, learn about [Digital Twin Workflows](/docs/module-2/digital-twin-workflows) to combine these tools into production systems.

## Further Reading

- [Unity Robotics Hub](https://github.com/Unity-Technologies/Unity-Robotics-Hub)
- [Unity Perception Package](https://github.com/Unity-Technologies/com.unity.perception)
- [ROS TCP Connector Documentation](https://github.com/Unity-Technologies/ROS-TCP-Connector)
- [Unity Manual - ArticulationBody](https://docs.unity3d.com/Manual/class-ArticulationBody.html)
