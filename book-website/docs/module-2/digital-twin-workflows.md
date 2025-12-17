---
title: "Digital Twin Workflows"
sidebar_position: 6
description: "Best practices for digital twin development: validation, continuous integration, and production deployment."
---

# Digital Twin Workflows

Creating a simulation is just the beginning - this chapter covers the workflows and practices that make digital twins truly valuable for robotics development. From validation to continuous integration, these practices separate hobby projects from production systems.

## Overview

In this section, you will:

- Understand the complete digital twin lifecycle
- Validate simulations against real-world behavior
- Set up CI/CD pipelines with simulation-in-the-loop testing
- Manage simulation assets alongside code
- Scale simulations for parallel training and testing
- Monitor and maintain simulation accuracy over time

## Prerequisites

Before starting, ensure you have completed:

- [Gazebo-ROS 2 Integration](/docs/module-2/gazebo-ros2-integration)
- [Unity for Robotics](/docs/module-2/unity-robotics)
- Basic understanding of Git and CI/CD concepts

---

## What is a Digital Twin?

### Beyond Simulation

A digital twin is more than a simulation - it's a continuously synchronized virtual replica:

```
┌─────────────────────────────────────────────────────────────────┐
│                    Digital Twin Architecture                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   ┌─────────────────┐              ┌─────────────────┐          │
│   │   Physical      │   Real-time  │    Digital      │          │
│   │   Robot         │◄────────────▶│    Twin         │          │
│   └─────────────────┘    Sync      └─────────────────┘          │
│          │                                  │                    │
│          │  Sensors      ┌──────────┐      │  Predictions      │
│          └──────────────▶│  Data    │◀─────┘                    │
│                          │  Lake    │                           │
│                          └────┬─────┘                           │
│                               │                                  │
│                     ┌─────────▼─────────┐                       │
│                     │   Analytics &     │                       │
│                     │   ML Training     │                       │
│                     └───────────────────┘                       │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Digital Twin Maturity Levels

| Level | Description | Capabilities |
|-------|-------------|--------------|
| **L1: Model** | Static 3D model | Visualization only |
| **L2: Simulation** | Physics-based behavior | Algorithm testing |
| **L3: Connected** | Receives real sensor data | Live monitoring |
| **L4: Synchronized** | Bidirectional updates | Predictive maintenance |
| **L5: Autonomous** | Self-updating models | Closed-loop optimization |

:::tip Course Focus
This course focuses on L2-L3 digital twins. Higher levels require production infrastructure beyond our scope.
:::

---

## Validation Workflow

### The Validation Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                    Simulation Validation Pipeline                │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐ │
│   │  Define  │    │  Collect │    │  Run     │    │  Compare │ │
│   │  Metrics │───▶│  Real    │───▶│  Sim     │───▶│  Results │ │
│   │          │    │  Data    │    │  Tests   │    │          │ │
│   └──────────┘    └──────────┘    └──────────┘    └──────────┘ │
│        │                                               │        │
│        │              ┌──────────┐                    │        │
│        └─────────────▶│  Update  │◀───────────────────┘        │
│                       │  Model   │                              │
│                       └──────────┘                              │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Defining Validation Metrics

```python title="validation_metrics.py"
"""Define metrics for simulation validation."""
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class ValidationMetric:
    """A metric for comparing simulation to reality."""
    name: str
    description: str
    tolerance: float  # Acceptable error margin
    unit: str

# Common robotics validation metrics
VALIDATION_METRICS = [
    ValidationMetric(
        name="position_error",
        description="End-effector position accuracy",
        tolerance=0.01,  # 1cm
        unit="meters"
    ),
    ValidationMetric(
        name="velocity_tracking",
        description="Velocity command tracking error",
        tolerance=0.05,  # 5%
        unit="m/s"
    ),
    ValidationMetric(
        name="sensor_latency",
        description="Sensor data delay",
        tolerance=0.010,  # 10ms
        unit="seconds"
    ),
    ValidationMetric(
        name="collision_detection",
        description="Collision detection accuracy",
        tolerance=0.02,  # 2cm
        unit="meters"
    ),
]

def compute_metric(real_data: np.ndarray, sim_data: np.ndarray,
                   metric: ValidationMetric) -> Tuple[float, bool]:
    """
    Compute error between real and simulated data.

    Returns:
        Tuple of (error_value, passes_tolerance)
    """
    if metric.name == "position_error":
        error = np.mean(np.linalg.norm(real_data - sim_data, axis=1))
    elif metric.name == "velocity_tracking":
        error = np.mean(np.abs(real_data - sim_data) / (np.abs(real_data) + 1e-6))
    else:
        error = np.mean(np.abs(real_data - sim_data))

    return error, error <= metric.tolerance
```

### Data Collection Protocol

```python title="data_collection.py"
#!/usr/bin/env python3
"""Collect synchronized data from real and simulated robots."""
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from sensor_msgs.msg import JointState
import json
from datetime import datetime

class DataCollector(Node):
    def __init__(self, source: str):  # 'real' or 'sim'
        super().__init__(f'{source}_data_collector')
        self.source = source
        self.data = {
            'source': source,
            'timestamp': [],
            'odom': [],
            'joint_states': []
        }

        # Subscribe to robot data
        self.odom_sub = self.create_subscription(
            Odometry, '/odom', self.odom_callback, 10
        )
        self.joint_sub = self.create_subscription(
            JointState, '/joint_states', self.joint_callback, 10
        )

        self.get_logger().info(f'Collecting {source} data...')

    def odom_callback(self, msg: Odometry):
        self.data['timestamp'].append(self.get_clock().now().nanoseconds)
        self.data['odom'].append({
            'x': msg.pose.pose.position.x,
            'y': msg.pose.pose.position.y,
            'theta': msg.pose.pose.orientation.z,
            'vx': msg.twist.twist.linear.x,
            'vtheta': msg.twist.twist.angular.z
        })

    def joint_callback(self, msg: JointState):
        self.data['joint_states'].append({
            'names': list(msg.name),
            'positions': list(msg.position),
            'velocities': list(msg.velocity)
        })

    def save_data(self, filename: str):
        with open(filename, 'w') as f:
            json.dump(self.data, f, indent=2)
        self.get_logger().info(f'Saved to {filename}')


def main():
    rclpy.init()
    # Run for real robot:
    # collector = DataCollector('real')
    # Run for simulation:
    collector = DataCollector('sim')
    try:
        rclpy.spin(collector)
    except KeyboardInterrupt:
        collector.save_data(f'data_{collector.source}_{datetime.now():%Y%m%d_%H%M%S}.json')
    finally:
        collector.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

### Comparison Analysis

```python title="validation_analysis.py"
"""Compare real and simulated data."""
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def load_and_align_data(real_file: str, sim_file: str):
    """Load data files and align by timestamp."""
    with open(real_file) as f:
        real_data = json.load(f)
    with open(sim_file) as f:
        sim_data = json.load(f)

    # Convert to numpy arrays
    real_odom = np.array([[d['x'], d['y'], d['vx']] for d in real_data['odom']])
    sim_odom = np.array([[d['x'], d['y'], d['vx']] for d in sim_data['odom']])

    # Interpolate to common timestamps (simplified)
    min_len = min(len(real_odom), len(sim_odom))
    return real_odom[:min_len], sim_odom[:min_len]

def generate_validation_report(real_odom, sim_odom, output_dir: Path):
    """Generate validation report with plots and metrics."""
    output_dir.mkdir(exist_ok=True)

    # Position tracking plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(real_odom[:, 0], real_odom[:, 1], 'b-', label='Real')
    axes[0].plot(sim_odom[:, 0], sim_odom[:, 1], 'r--', label='Sim')
    axes[0].set_xlabel('X (m)')
    axes[0].set_ylabel('Y (m)')
    axes[0].set_title('Trajectory Comparison')
    axes[0].legend()
    axes[0].axis('equal')

    # Position error over time
    pos_error = np.sqrt((real_odom[:, 0] - sim_odom[:, 0])**2 +
                        (real_odom[:, 1] - sim_odom[:, 1])**2)
    axes[1].plot(pos_error)
    axes[1].axhline(y=0.01, color='g', linestyle='--', label='Tolerance')
    axes[1].set_xlabel('Time Step')
    axes[1].set_ylabel('Position Error (m)')
    axes[1].set_title('Position Error Over Time')
    axes[1].legend()

    # Velocity comparison
    axes[2].plot(real_odom[:, 2], 'b-', label='Real Vx')
    axes[2].plot(sim_odom[:, 2], 'r--', label='Sim Vx')
    axes[2].set_xlabel('Time Step')
    axes[2].set_ylabel('Velocity (m/s)')
    axes[2].set_title('Velocity Tracking')
    axes[2].legend()

    plt.tight_layout()
    plt.savefig(output_dir / 'validation_report.png', dpi=150)
    plt.close()

    # Compute summary metrics
    metrics = {
        'mean_position_error': float(np.mean(pos_error)),
        'max_position_error': float(np.max(pos_error)),
        'velocity_rmse': float(np.sqrt(np.mean((real_odom[:, 2] - sim_odom[:, 2])**2))),
        'passed': np.mean(pos_error) < 0.01
    }

    with open(output_dir / 'metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)

    return metrics
```

---

## CI/CD with Simulation-in-the-Loop

### Pipeline Architecture

```yaml title=".github/workflows/simulation-test.yml"
name: Simulation Tests

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  simulation-tests:
    runs-on: ubuntu-22.04
    container:
      image: osrf/ros:humble-desktop

    steps:
      - uses: actions/checkout@v3

      - name: Install dependencies
        run: |
          apt-get update
          apt-get install -y ros-humble-ros-gz ros-humble-nav2-bringup
          rosdep update
          rosdep install --from-paths src -y --ignore-src

      - name: Build workspace
        run: |
          source /opt/ros/humble/setup.bash
          colcon build --packages-select my_robot_description my_robot_tests

      - name: Run simulation tests
        run: |
          source /opt/ros/humble/setup.bash
          source install/setup.bash
          # Start Gazebo in headless mode
          ros2 launch my_robot_description simulation.launch.py headless:=true &
          sleep 10  # Wait for simulation to start
          # Run test suite
          ros2 launch my_robot_tests simulation_tests.launch.py
        timeout-minutes: 30

      - name: Upload test results
        uses: actions/upload-artifact@v3
        with:
          name: test-results
          path: |
            test_results/
            validation_reports/
```

### Test Scenarios

```python title="simulation_tests.py"
#!/usr/bin/env python3
"""Simulation test scenarios for CI/CD pipeline."""
import rclpy
from rclpy.node import Node
import unittest
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
import time

class SimulationTestCase(unittest.TestCase):
    """Base class for simulation tests."""

    @classmethod
    def setUpClass(cls):
        rclpy.init()
        cls.node = rclpy.create_node('simulation_tester')
        cls.cmd_pub = cls.node.create_publisher(Twist, '/cmd_vel', 10)
        cls.latest_odom = None
        cls.odom_sub = cls.node.create_subscription(
            Odometry, '/odom', cls._odom_callback, 10
        )

    @classmethod
    def _odom_callback(cls, msg):
        cls.latest_odom = msg

    def wait_for_odom(self, timeout=5.0):
        """Wait for odometry message."""
        start = time.time()
        while self.latest_odom is None and time.time() - start < timeout:
            rclpy.spin_once(self.node, timeout_sec=0.1)
        self.assertIsNotNone(self.latest_odom, "No odometry received")

    def send_velocity(self, linear: float, angular: float, duration: float):
        """Send velocity command for specified duration."""
        cmd = Twist()
        cmd.linear.x = linear
        cmd.angular.z = angular

        start = time.time()
        while time.time() - start < duration:
            self.cmd_pub.publish(cmd)
            rclpy.spin_once(self.node, timeout_sec=0.05)


class TestBasicMotion(SimulationTestCase):
    """Test basic robot motion."""

    def test_forward_motion(self):
        """Robot should move forward when commanded."""
        self.wait_for_odom()
        initial_x = self.latest_odom.pose.pose.position.x

        # Move forward for 2 seconds at 0.5 m/s
        self.send_velocity(0.5, 0.0, 2.0)
        self.wait_for_odom()

        final_x = self.latest_odom.pose.pose.position.x
        distance = final_x - initial_x

        # Should move approximately 1 meter (0.5 m/s * 2s)
        self.assertGreater(distance, 0.8, "Robot did not move forward enough")
        self.assertLess(distance, 1.2, "Robot moved too far")

    def test_rotation(self):
        """Robot should rotate in place."""
        self.wait_for_odom()

        # Rotate for 3 seconds
        self.send_velocity(0.0, 0.5, 3.0)
        self.wait_for_odom()

        # Check that robot rotated (simplified check)
        # In production, compute actual rotation from quaternion
        self.assertIsNotNone(self.latest_odom)

    def test_stop(self):
        """Robot should stop when commanded zero velocity."""
        self.send_velocity(0.5, 0.0, 1.0)  # Move first
        self.send_velocity(0.0, 0.0, 1.0)  # Stop

        self.wait_for_odom()
        vx = self.latest_odom.twist.twist.linear.x

        self.assertLess(abs(vx), 0.05, "Robot did not stop")


class TestObstacleDetection(SimulationTestCase):
    """Test obstacle detection capabilities."""

    def test_lidar_range(self):
        """LiDAR should detect obstacles within range."""
        # This would subscribe to /scan and verify readings
        pass

    def test_collision_avoidance(self):
        """Robot should stop before hitting obstacle."""
        # This requires spawning obstacle and testing behavior
        pass


if __name__ == '__main__':
    unittest.main()
```

### Automated Regression Testing

```python title="regression_suite.py"
"""Regression test suite comparing simulation versions."""
import subprocess
import json
from pathlib import Path
from datetime import datetime

def run_benchmark_suite(sim_version: str) -> dict:
    """Run standard benchmark and return metrics."""
    results = {
        'version': sim_version,
        'timestamp': datetime.now().isoformat(),
        'tests': {}
    }

    # Test 1: Navigation benchmark
    nav_result = subprocess.run([
        'ros2', 'launch', 'my_robot_tests', 'nav_benchmark.launch.py',
        f'sim_version:={sim_version}'
    ], capture_output=True, timeout=300)
    results['tests']['navigation'] = {
        'passed': nav_result.returncode == 0,
        'time_to_goal': parse_nav_time(nav_result.stdout)
    }

    # Test 2: Manipulation benchmark
    manip_result = subprocess.run([
        'ros2', 'launch', 'my_robot_tests', 'manipulation_benchmark.launch.py',
        f'sim_version:={sim_version}'
    ], capture_output=True, timeout=300)
    results['tests']['manipulation'] = {
        'passed': manip_result.returncode == 0,
        'success_rate': parse_success_rate(manip_result.stdout)
    }

    return results

def compare_versions(baseline: dict, current: dict) -> dict:
    """Compare benchmark results between versions."""
    comparison = {
        'baseline_version': baseline['version'],
        'current_version': current['version'],
        'regressions': [],
        'improvements': []
    }

    for test_name in baseline['tests']:
        base_val = baseline['tests'][test_name]
        curr_val = current['tests'][test_name]

        if not curr_val['passed'] and base_val['passed']:
            comparison['regressions'].append(f"{test_name}: Now failing")
        elif curr_val['passed'] and not base_val['passed']:
            comparison['improvements'].append(f"{test_name}: Now passing")

    return comparison
```

---

## Asset Management

### Directory Structure

```
my_robot_simulation/
├── worlds/
│   ├── warehouse.sdf
│   ├── outdoor.sdf
│   └── test_arena.sdf
├── models/
│   ├── my_robot/
│   │   ├── model.config
│   │   ├── model.sdf
│   │   └── meshes/
│   │       ├── base_link.stl
│   │       └── wheel.stl
│   └── obstacles/
│       ├── box_small/
│       └── cylinder_tall/
├── config/
│   ├── bridge.yaml
│   ├── sensors.yaml
│   └── physics.yaml
├── launch/
│   ├── simulation.launch.py
│   └── test_world.launch.py
├── urdf/
│   ├── robot.urdf.xacro
│   └── sensors.xacro
└── tests/
    ├── simulation_tests.py
    └── validation/
        ├── real_data/
        └── comparison_scripts/
```

### Version Control Best Practices

```gitignore title=".gitignore"
# Gazebo cache
~/.gazebo/

# Build artifacts
build/
install/
log/

# Large binary files (use Git LFS)
*.dae
*.stl
*.obj

# Simulation logs
*.log
gazebo-*.log

# Test artifacts
test_results/
validation_reports/
```

```bash title="Git LFS setup for meshes"
# Install Git LFS
sudo apt install git-lfs
git lfs install

# Track mesh files
git lfs track "*.stl"
git lfs track "*.dae"
git lfs track "*.obj"

# Track large world files
git lfs track "*.world"
git lfs track "worlds/*.sdf"

# Add .gitattributes
git add .gitattributes
git commit -m "Configure Git LFS for simulation assets"
```

---

## Scaling Simulations

### Parallel Simulation for Training

```python title="parallel_simulation.py"
"""Launch multiple simulation instances for parallel training."""
import subprocess
import os
from multiprocessing import Pool
from typing import List, Tuple

def launch_simulation(config: Tuple[int, dict]) -> dict:
    """Launch a single simulation instance."""
    instance_id, params = config

    # Set unique ROS domain ID
    env = os.environ.copy()
    env['ROS_DOMAIN_ID'] = str(100 + instance_id)

    # Launch simulation with unique port
    gz_port = 11345 + instance_id

    cmd = [
        'gz', 'sim', '-s',  # Server only, no GUI
        '-r', params['world_file'],
        '--physics-engine', params.get('physics_engine', 'ode'),
    ]

    proc = subprocess.Popen(
        cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )

    return {
        'instance_id': instance_id,
        'pid': proc.pid,
        'ros_domain_id': env['ROS_DOMAIN_ID'],
        'gz_port': gz_port
    }


def launch_parallel_simulations(num_instances: int, world_file: str) -> List[dict]:
    """Launch multiple simulation instances in parallel."""
    configs = [
        (i, {'world_file': world_file})
        for i in range(num_instances)
    ]

    with Pool(num_instances) as pool:
        results = pool.map(launch_simulation, configs)

    return results


# Usage for RL training
if __name__ == '__main__':
    # Launch 8 parallel simulations for training
    instances = launch_parallel_simulations(8, 'training_world.sdf')
    print(f"Launched {len(instances)} simulation instances")
```

### Cloud-Based Simulation

```yaml title="kubernetes/simulation-deployment.yaml"
apiVersion: apps/v1
kind: Deployment
metadata:
  name: gazebo-simulation
spec:
  replicas: 10  # 10 parallel simulations
  selector:
    matchLabels:
      app: gazebo-sim
  template:
    metadata:
      labels:
        app: gazebo-sim
    spec:
      containers:
      - name: gazebo
        image: osrf/ros:humble-simulation
        command: ["ros2", "launch", "my_robot", "simulation.launch.py", "headless:=true"]
        resources:
          requests:
            memory: "4Gi"
            cpu: "2"
          limits:
            memory: "8Gi"
            cpu: "4"
        env:
        - name: ROS_DOMAIN_ID
          valueFrom:
            fieldRef:
              fieldPath: metadata.name
```

---

## Exercise 1: Validation Pipeline

:::tip Exercise 1: Build a Validation Pipeline
**Objective**: Create a validation pipeline comparing simulation to real data.

**Time Estimate**: 60 minutes

**Steps**:

1. Define 3 validation metrics for your robot
2. Collect 60 seconds of data from simulation
3. Create synthetic "real" data with added noise
4. Implement comparison functions
5. Generate a validation report with plots

**Expected Output**:
- `metrics.json` with pass/fail status
- `validation_report.png` with comparison plots
- Summary of which metrics passed tolerance

**Success Criteria**:
- Position error < 1cm for 90% of time steps
- Velocity tracking error < 5%
:::

---

## Exercise 2: CI/CD Integration

:::tip Exercise 2: Simulation-in-the-Loop CI
**Objective**: Set up automated testing with simulation in a CI pipeline.

**Time Estimate**: 45 minutes

**Steps**:

1. Create a GitHub Actions workflow file
2. Add simulation launch step (headless mode)
3. Create 3 test cases (motion, stopping, rotation)
4. Configure test result artifacts
5. Set up pass/fail criteria

**Test Coverage**:
- Basic motion commands
- Emergency stop response
- Sensor data publication

**Expected Result**: PR checks run simulation tests automatically.
:::

---

## Summary

In this chapter, you learned:

- **Digital Twin Lifecycle**: From static models to synchronized systems
- **Validation**: Define metrics, collect data, compare real vs. sim
- **CI/CD Integration**: Automated testing with simulation-in-the-loop
- **Asset Management**: Version control for simulation files
- **Scaling**: Parallel simulations for training and cloud deployment

These workflows transform simulations from development tools into production infrastructure. Combined with the Gazebo and Unity skills from earlier chapters, you're now equipped to build robust robotics development pipelines.

## Module 2 Completion

Congratulations on completing Module 2! You've learned:

1. Simulation fundamentals and physics engines
2. Gazebo world and robot creation
3. ROS 2 bridge integration
4. Unity for perception and synthetic data
5. Digital twin best practices

Continue to [Module 3: NVIDIA Isaac Platform](/docs/module-3) to explore advanced simulation with GPU acceleration and reinforcement learning.

## Further Reading

- [Digital Twin Consortium](https://www.digitaltwinconsortium.org/)
- [Simulation-Based Testing for Autonomous Vehicles](https://arxiv.org/abs/2003.07739)
- [GitHub Actions for ROS 2](https://github.com/ros-tooling/setup-ros)
- [Kubernetes for Robotics](https://www.robolaunch.io/)
