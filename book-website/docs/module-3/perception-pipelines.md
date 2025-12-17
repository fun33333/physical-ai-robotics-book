---
title: "Perception Pipelines"
sidebar_position: 6
description: "Building end-to-end perception systems: sensor fusion, object tracking, and scene understanding."
---

# Perception Pipelines

Individual perception components must be combined into coherent pipelines that provide robots with understanding of their environment. This chapter teaches you to design and implement complete perception systems that integrate detection, tracking, and scene understanding into real-time workflows.

## Overview

In this section, you will:

- Design modular perception pipeline architectures
- Implement multi-sensor fusion techniques
- Build object tracking systems across frames
- Create scene graphs for semantic understanding
- Optimize pipeline performance for real-time operation
- Debug and visualize perception systems
- Handle perception failures gracefully

## Prerequisites

- Completed [Isaac ROS](/docs/module-3/isaac-ros) chapter
- Understanding of ROS 2 topics, services, and actions
- Familiarity with computer vision concepts
- Basic linear algebra (transforms, matrices)
- Python and C++ programming experience

---

## Pipeline Architecture

### Design Principles

```
┌─────────────────────────────────────────────────────────────────┐
│              Perception Pipeline Architecture                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │                   Sensor Layer                           │   │
│   │  ┌────────┐  ┌────────┐  ┌────────┐  ┌────────┐        │   │
│   │  │RGB Cam │  │Depth   │  │ LiDAR  │  │  IMU   │        │   │
│   │  └────┬───┘  └────┬───┘  └────┬───┘  └────┬───┘        │   │
│   └───────┼───────────┼───────────┼───────────┼─────────────┘   │
│           │           │           │           │                  │
│   ┌───────┴───────────┴───────────┴───────────┴─────────────┐   │
│   │               Synchronization & Preprocessing            │   │
│   │    Time sync • Undistort • Point cloud generation       │   │
│   └───────────────────────────┬─────────────────────────────┘   │
│                               │                                  │
│   ┌───────────────────────────┴─────────────────────────────┐   │
│   │                   Perception Modules                     │   │
│   │  ┌─────────────┐ ┌──────────────┐ ┌─────────────────┐  │   │
│   │  │ Detection   │ │ Segmentation │ │   SLAM/Odom     │  │   │
│   │  └──────┬──────┘ └──────┬───────┘ └────────┬────────┘  │   │
│   └─────────┼───────────────┼──────────────────┼────────────┘   │
│             │               │                  │                 │
│   ┌─────────┴───────────────┴──────────────────┴────────────┐   │
│   │                    Fusion Layer                          │   │
│   │    Object tracking • Pose estimation • Map integration   │   │
│   └───────────────────────────┬─────────────────────────────┘   │
│                               │                                  │
│   ┌───────────────────────────┴─────────────────────────────┐   │
│   │                   Scene Understanding                    │   │
│   │    Scene graph • Semantic map • Spatial reasoning        │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Key Principles

| Principle | Description | Implementation |
|-----------|-------------|----------------|
| **Modularity** | Swap components without rewriting | ROS 2 interfaces |
| **Time Sync** | All sensors aligned temporally | message_filters |
| **Graceful Degradation** | Continue with partial data | Fallback paths |
| **Real-time** | Meet latency requirements | Profile and optimize |
| **Observability** | Monitor system health | Diagnostics, logging |

---

## Sensor Synchronization

### Time Synchronization

```python title="sync_node.py"
"""Multi-sensor synchronization using message_filters."""
import rclpy
from rclpy.node import Node
from message_filters import Subscriber, ApproximateTimeSynchronizer
from sensor_msgs.msg import Image, CameraInfo, PointCloud2
from nav_msgs.msg import Odometry

class SensorSyncNode(Node):
    def __init__(self):
        super().__init__('sensor_sync')

        # Create subscribers
        self.rgb_sub = Subscriber(self, Image, '/camera/color/image_raw')
        self.depth_sub = Subscriber(self, Image, '/camera/depth/image_raw')
        self.info_sub = Subscriber(self, CameraInfo, '/camera/color/camera_info')
        self.lidar_sub = Subscriber(self, PointCloud2, '/lidar/points')

        # Time synchronizer with 50ms tolerance
        self.sync = ApproximateTimeSynchronizer(
            [self.rgb_sub, self.depth_sub, self.info_sub, self.lidar_sub],
            queue_size=10,
            slop=0.05  # 50ms tolerance
        )
        self.sync.registerCallback(self.synced_callback)

        # Publisher for synchronized bundle
        self.synced_pub = self.create_publisher(
            SensorBundle, '/perception/sensor_bundle', 10
        )

        self.get_logger().info('Sensor sync ready')

    def synced_callback(self, rgb_msg, depth_msg, info_msg, lidar_msg):
        """Process synchronized sensor data."""
        # Create synchronized bundle
        bundle = SensorBundle()
        bundle.header.stamp = self.get_clock().now().to_msg()
        bundle.rgb = rgb_msg
        bundle.depth = depth_msg
        bundle.camera_info = info_msg
        bundle.point_cloud = lidar_msg

        self.synced_pub.publish(bundle)


def main():
    rclpy.init()
    node = SensorSyncNode()
    rclpy.spin(node)


if __name__ == '__main__':
    main()
```

### TF Transform Management

```python title="transform_manager.py"
"""Transform management for perception pipeline."""
import rclpy
from rclpy.node import Node
from tf2_ros import Buffer, TransformListener, TransformBroadcaster
from geometry_msgs.msg import TransformStamped
import numpy as np

class TransformManager(Node):
    def __init__(self):
        super().__init__('transform_manager')

        # TF2 setup
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.tf_broadcaster = TransformBroadcaster(self)

        # Frame configuration
        self.declare_parameter('base_frame', 'base_link')
        self.declare_parameter('camera_frame', 'camera_link')
        self.declare_parameter('lidar_frame', 'lidar_link')

    def get_transform(self, target_frame: str, source_frame: str):
        """Get transform between frames."""
        try:
            transform = self.tf_buffer.lookup_transform(
                target_frame,
                source_frame,
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=1.0)
            )
            return transform
        except Exception as e:
            self.get_logger().warn(f'Transform lookup failed: {e}')
            return None

    def transform_point_cloud(self, cloud, target_frame):
        """Transform point cloud to target frame."""
        transform = self.get_transform(target_frame, cloud.header.frame_id)
        if transform is None:
            return None

        # Apply transformation using tf2_sensor_msgs
        from tf2_sensor_msgs import do_transform_cloud
        return do_transform_cloud(cloud, transform)
```

---

## Multi-Sensor Fusion

### Detection Fusion

```python title="detection_fusion.py"
"""Fuse detections from multiple sensors."""
import numpy as np
from dataclasses import dataclass
from typing import List
from scipy.optimize import linear_sum_assignment

@dataclass
class Detection:
    """Unified detection format."""
    bbox_2d: np.ndarray  # [x, y, w, h]
    bbox_3d: np.ndarray  # [x, y, z, l, w, h, yaw]
    confidence: float
    class_id: int
    source: str  # 'camera', 'lidar', 'radar'

class DetectionFuser:
    """Fuse detections from camera and LiDAR."""

    def __init__(self, iou_threshold: float = 0.3):
        self.iou_threshold = iou_threshold

    def fuse(self, camera_dets: List[Detection],
             lidar_dets: List[Detection]) -> List[Detection]:
        """Fuse camera and LiDAR detections."""
        if not camera_dets:
            return lidar_dets
        if not lidar_dets:
            return camera_dets

        # Build cost matrix based on 3D IoU
        cost_matrix = np.zeros((len(camera_dets), len(lidar_dets)))
        for i, cam_det in enumerate(camera_dets):
            for j, lid_det in enumerate(lidar_dets):
                iou = self._compute_3d_iou(cam_det.bbox_3d, lid_det.bbox_3d)
                cost_matrix[i, j] = 1.0 - iou

        # Hungarian algorithm for optimal assignment
        row_indices, col_indices = linear_sum_assignment(cost_matrix)

        fused_detections = []
        matched_camera = set()
        matched_lidar = set()

        for row, col in zip(row_indices, col_indices):
            if cost_matrix[row, col] < (1.0 - self.iou_threshold):
                # Fuse matched detections
                fused = self._merge_detections(
                    camera_dets[row], lidar_dets[col]
                )
                fused_detections.append(fused)
                matched_camera.add(row)
                matched_lidar.add(col)

        # Add unmatched detections
        for i, det in enumerate(camera_dets):
            if i not in matched_camera:
                fused_detections.append(det)

        for j, det in enumerate(lidar_dets):
            if j not in matched_lidar:
                fused_detections.append(det)

        return fused_detections

    def _compute_3d_iou(self, box1: np.ndarray, box2: np.ndarray) -> float:
        """Compute 3D IoU between two bounding boxes."""
        # Simplified axis-aligned IoU
        x1_min, x1_max = box1[0] - box1[3]/2, box1[0] + box1[3]/2
        y1_min, y1_max = box1[1] - box1[4]/2, box1[1] + box1[4]/2
        z1_min, z1_max = box1[2] - box1[5]/2, box1[2] + box1[5]/2

        x2_min, x2_max = box2[0] - box2[3]/2, box2[0] + box2[3]/2
        y2_min, y2_max = box2[1] - box2[4]/2, box2[1] + box2[4]/2
        z2_min, z2_max = box2[2] - box2[5]/2, box2[2] + box2[5]/2

        # Intersection
        xi_min = max(x1_min, x2_min)
        yi_min = max(y1_min, y2_min)
        zi_min = max(z1_min, z2_min)
        xi_max = min(x1_max, x2_max)
        yi_max = min(y1_max, y2_max)
        zi_max = min(z1_max, z2_max)

        if xi_min >= xi_max or yi_min >= yi_max or zi_min >= zi_max:
            return 0.0

        inter_vol = (xi_max - xi_min) * (yi_max - yi_min) * (zi_max - zi_min)
        vol1 = box1[3] * box1[4] * box1[5]
        vol2 = box2[3] * box2[4] * box2[5]

        return inter_vol / (vol1 + vol2 - inter_vol)

    def _merge_detections(self, cam_det: Detection,
                          lid_det: Detection) -> Detection:
        """Merge camera and LiDAR detections."""
        # Use LiDAR for 3D position (more accurate)
        # Use camera for classification (better features)
        return Detection(
            bbox_2d=cam_det.bbox_2d,
            bbox_3d=lid_det.bbox_3d,
            confidence=(cam_det.confidence + lid_det.confidence) / 2,
            class_id=cam_det.class_id,
            source='fused'
        )
```

### Camera-LiDAR Projection

```python title="projection.py"
"""Project between camera and LiDAR coordinate frames."""
import numpy as np

class CameraLidarProjector:
    """Project points between camera and LiDAR frames."""

    def __init__(self, camera_matrix: np.ndarray,
                 extrinsic: np.ndarray):
        """
        Args:
            camera_matrix: 3x3 intrinsic camera matrix
            extrinsic: 4x4 transform from LiDAR to camera frame
        """
        self.K = camera_matrix
        self.T_cam_lidar = extrinsic

    def lidar_to_image(self, points_lidar: np.ndarray) -> np.ndarray:
        """Project LiDAR points onto image plane."""
        # Add homogeneous coordinate
        n_points = points_lidar.shape[0]
        points_h = np.hstack([points_lidar, np.ones((n_points, 1))])

        # Transform to camera frame
        points_cam = (self.T_cam_lidar @ points_h.T).T[:, :3]

        # Filter points behind camera
        valid = points_cam[:, 2] > 0
        points_cam = points_cam[valid]

        # Project to image plane
        points_2d = (self.K @ points_cam.T).T
        points_2d = points_2d[:, :2] / points_2d[:, 2:3]

        return points_2d, valid

    def image_to_ray(self, pixel: np.ndarray) -> np.ndarray:
        """Get ray direction from pixel coordinate."""
        # Unproject pixel to normalized camera coords
        K_inv = np.linalg.inv(self.K)
        pixel_h = np.array([pixel[0], pixel[1], 1.0])
        ray_cam = K_inv @ pixel_h

        # Transform to LiDAR frame
        T_lidar_cam = np.linalg.inv(self.T_cam_lidar)
        ray_lidar = T_lidar_cam[:3, :3] @ ray_cam

        return ray_lidar / np.linalg.norm(ray_lidar)
```

---

## Object Tracking

### Multi-Object Tracker

```python title="tracker.py"
"""Multi-object tracking with Kalman filter."""
import numpy as np
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment
from dataclasses import dataclass, field
from typing import List, Optional
import uuid

@dataclass
class Track:
    """Tracked object state."""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    state: np.ndarray = None  # [x, y, z, vx, vy, vz]
    covariance: np.ndarray = None
    class_id: int = -1
    confidence: float = 0.0
    age: int = 0
    hits: int = 0
    misses: int = 0
    kf: KalmanFilter = None


class MultiObjectTracker:
    """Track multiple objects across frames."""

    def __init__(self,
                 max_age: int = 5,
                 min_hits: int = 3,
                 iou_threshold: float = 0.3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.tracks: List[Track] = []

    def update(self, detections: List[Detection]) -> List[Track]:
        """Update tracks with new detections."""
        # Predict existing tracks
        for track in self.tracks:
            track.kf.predict()
            track.state = track.kf.x.flatten()
            track.covariance = track.kf.P
            track.age += 1

        # Match detections to tracks
        if len(self.tracks) > 0 and len(detections) > 0:
            matched, unmatched_dets, unmatched_trks = self._associate(
                detections, self.tracks
            )
        else:
            matched = []
            unmatched_dets = list(range(len(detections)))
            unmatched_trks = list(range(len(self.tracks)))

        # Update matched tracks
        for det_idx, trk_idx in matched:
            track = self.tracks[trk_idx]
            det = detections[det_idx]

            # Kalman update
            measurement = det.bbox_3d[:3]  # x, y, z
            track.kf.update(measurement)
            track.state = track.kf.x.flatten()
            track.covariance = track.kf.P
            track.hits += 1
            track.misses = 0
            track.confidence = det.confidence
            track.class_id = det.class_id

        # Create new tracks for unmatched detections
        for det_idx in unmatched_dets:
            det = detections[det_idx]
            track = self._create_track(det)
            self.tracks.append(track)

        # Mark missed tracks
        for trk_idx in unmatched_trks:
            self.tracks[trk_idx].misses += 1

        # Remove dead tracks
        self.tracks = [
            t for t in self.tracks
            if t.misses <= self.max_age
        ]

        # Return confirmed tracks
        return [
            t for t in self.tracks
            if t.hits >= self.min_hits
        ]

    def _create_track(self, det: Detection) -> Track:
        """Initialize new track from detection."""
        # 6D Kalman filter: [x, y, z, vx, vy, vz]
        kf = KalmanFilter(dim_x=6, dim_z=3)

        # State transition matrix (constant velocity)
        dt = 0.1  # Assume 10 Hz
        kf.F = np.array([
            [1, 0, 0, dt, 0, 0],
            [0, 1, 0, 0, dt, 0],
            [0, 0, 1, 0, 0, dt],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1],
        ])

        # Measurement matrix
        kf.H = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
        ])

        # Initial state from detection
        kf.x = np.array([
            det.bbox_3d[0], det.bbox_3d[1], det.bbox_3d[2],
            0, 0, 0
        ]).reshape(-1, 1)

        # Covariance matrices
        kf.P *= 10
        kf.R = np.diag([0.1, 0.1, 0.1])
        kf.Q = np.diag([0.01, 0.01, 0.01, 0.1, 0.1, 0.1])

        return Track(
            state=kf.x.flatten(),
            covariance=kf.P,
            class_id=det.class_id,
            confidence=det.confidence,
            hits=1,
            kf=kf
        )

    def _associate(self, detections, tracks):
        """Associate detections with existing tracks."""
        cost_matrix = np.zeros((len(detections), len(tracks)))

        for d, det in enumerate(detections):
            for t, track in enumerate(tracks):
                # Use Mahalanobis distance
                innovation = det.bbox_3d[:3] - track.state[:3]
                S = track.covariance[:3, :3] + np.eye(3) * 0.1
                dist = np.sqrt(innovation @ np.linalg.inv(S) @ innovation)
                cost_matrix[d, t] = dist

        # Hungarian algorithm
        det_indices, trk_indices = linear_sum_assignment(cost_matrix)

        matched = []
        unmatched_dets = list(range(len(detections)))
        unmatched_trks = list(range(len(tracks)))

        for d, t in zip(det_indices, trk_indices):
            if cost_matrix[d, t] < 10.0:  # Gating threshold
                matched.append((d, t))
                unmatched_dets.remove(d)
                unmatched_trks.remove(t)

        return matched, unmatched_dets, unmatched_trks
```

---

## Scene Understanding

### Scene Graph Representation

```python title="scene_graph.py"
"""Scene graph for semantic understanding."""
from dataclasses import dataclass, field
from typing import List, Dict, Optional
import numpy as np

@dataclass
class SceneNode:
    """Node in scene graph."""
    id: str
    class_name: str
    position: np.ndarray
    orientation: np.ndarray
    dimensions: np.ndarray
    confidence: float
    attributes: Dict[str, any] = field(default_factory=dict)
    parent_id: Optional[str] = None


@dataclass
class SceneRelation:
    """Spatial relation between nodes."""
    subject_id: str
    object_id: str
    relation_type: str  # 'on', 'near', 'inside', 'left_of', etc.
    confidence: float


class SceneGraph:
    """Scene graph for semantic scene understanding."""

    def __init__(self):
        self.nodes: Dict[str, SceneNode] = {}
        self.relations: List[SceneRelation] = []

    def add_node(self, node: SceneNode):
        """Add node to scene graph."""
        self.nodes[node.id] = node

    def remove_node(self, node_id: str):
        """Remove node and its relations."""
        if node_id in self.nodes:
            del self.nodes[node_id]
            self.relations = [
                r for r in self.relations
                if r.subject_id != node_id and r.object_id != node_id
            ]

    def add_relation(self, relation: SceneRelation):
        """Add spatial relation."""
        self.relations.append(relation)

    def compute_relations(self):
        """Compute spatial relations between nodes."""
        self.relations = []
        nodes = list(self.nodes.values())

        for i, node1 in enumerate(nodes):
            for node2 in nodes[i+1:]:
                relations = self._compute_pairwise_relations(node1, node2)
                self.relations.extend(relations)

    def _compute_pairwise_relations(self, node1: SceneNode,
                                     node2: SceneNode) -> List[SceneRelation]:
        """Compute relations between two nodes."""
        relations = []
        distance = np.linalg.norm(node1.position - node2.position)

        # Near relation
        if distance < 1.0:  # Within 1 meter
            relations.append(SceneRelation(
                subject_id=node1.id,
                object_id=node2.id,
                relation_type='near',
                confidence=1.0 - distance
            ))

        # On relation (vertical alignment)
        if abs(node1.position[0] - node2.position[0]) < 0.3 and \
           abs(node1.position[1] - node2.position[1]) < 0.3:
            if node1.position[2] > node2.position[2]:
                relations.append(SceneRelation(
                    subject_id=node1.id,
                    object_id=node2.id,
                    relation_type='on',
                    confidence=0.9
                ))

        # Left/Right relations
        relative = node2.position - node1.position
        if abs(relative[1]) > 0.3:
            relation_type = 'right_of' if relative[1] > 0 else 'left_of'
            relations.append(SceneRelation(
                subject_id=node2.id,
                object_id=node1.id,
                relation_type=relation_type,
                confidence=min(abs(relative[1]), 1.0)
            ))

        return relations

    def query(self, query: str) -> List[SceneNode]:
        """Query scene graph (simplified)."""
        # Example: "cup on table"
        tokens = query.lower().split()

        results = []
        for node in self.nodes.values():
            if node.class_name.lower() in tokens:
                results.append(node)

        return results

    def to_dict(self) -> dict:
        """Export scene graph as dictionary."""
        return {
            'nodes': [
                {
                    'id': n.id,
                    'class': n.class_name,
                    'position': n.position.tolist(),
                    'confidence': n.confidence
                }
                for n in self.nodes.values()
            ],
            'relations': [
                {
                    'subject': r.subject_id,
                    'object': r.object_id,
                    'relation': r.relation_type,
                    'confidence': r.confidence
                }
                for r in self.relations
            ]
        }
```

---

## Complete Pipeline

### ROS 2 Pipeline Node

```python title="perception_pipeline.py"
"""Complete perception pipeline ROS 2 node."""
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image, PointCloud2
from vision_msgs.msg import Detection3DArray, Detection3D
from visualization_msgs.msg import MarkerArray, Marker
from geometry_msgs.msg import Point

import numpy as np
from cv_bridge import CvBridge

class PerceptionPipeline(Node):
    """Complete perception pipeline node."""

    def __init__(self):
        super().__init__('perception_pipeline')

        # Components
        self.bridge = CvBridge()
        self.tracker = MultiObjectTracker()
        self.scene_graph = SceneGraph()
        self.fuser = DetectionFuser()

        # Parameters
        self.declare_parameter('detection_threshold', 0.5)
        self.declare_parameter('tracking_enabled', True)

        # Subscribers
        qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT
        )

        self.camera_det_sub = self.create_subscription(
            Detection3DArray, '/camera/detections',
            self.camera_detection_callback, qos
        )
        self.lidar_det_sub = self.create_subscription(
            Detection3DArray, '/lidar/detections',
            self.lidar_detection_callback, qos
        )

        # Publishers
        self.tracks_pub = self.create_publisher(
            Detection3DArray, '/perception/tracks', 10
        )
        self.scene_pub = self.create_publisher(
            MarkerArray, '/perception/scene_graph', 10
        )

        # Buffers
        self.camera_detections = []
        self.lidar_detections = []

        # Timer for pipeline execution
        self.timer = self.create_timer(0.1, self.run_pipeline)

        self.get_logger().info('Perception pipeline initialized')

    def camera_detection_callback(self, msg):
        """Buffer camera detections."""
        self.camera_detections = self._convert_detections(msg)

    def lidar_detection_callback(self, msg):
        """Buffer LiDAR detections."""
        self.lidar_detections = self._convert_detections(msg)

    def _convert_detections(self, msg) -> List[Detection]:
        """Convert ROS message to Detection objects."""
        detections = []
        for det in msg.detections:
            bbox_3d = np.array([
                det.bbox.center.position.x,
                det.bbox.center.position.y,
                det.bbox.center.position.z,
                det.bbox.size.x,
                det.bbox.size.y,
                det.bbox.size.z,
                0.0  # yaw
            ])
            detections.append(Detection(
                bbox_2d=np.zeros(4),
                bbox_3d=bbox_3d,
                confidence=det.results[0].hypothesis.score if det.results else 0.5,
                class_id=int(det.results[0].hypothesis.class_id) if det.results else -1,
                source='ros'
            ))
        return detections

    def run_pipeline(self):
        """Execute perception pipeline."""
        # 1. Fuse detections
        fused = self.fuser.fuse(
            self.camera_detections,
            self.lidar_detections
        )

        # 2. Track objects
        if self.get_parameter('tracking_enabled').value:
            tracks = self.tracker.update(fused)
        else:
            tracks = fused

        # 3. Update scene graph
        self.update_scene_graph(tracks)

        # 4. Publish results
        self.publish_tracks(tracks)
        self.publish_scene_visualization()

        # Clear buffers
        self.camera_detections = []
        self.lidar_detections = []

    def update_scene_graph(self, tracks):
        """Update scene graph with tracked objects."""
        # Remove stale nodes
        current_ids = {t.id for t in tracks}
        stale_ids = [
            nid for nid in self.scene_graph.nodes
            if nid not in current_ids
        ]
        for nid in stale_ids:
            self.scene_graph.remove_node(nid)

        # Update/add nodes
        for track in tracks:
            node = SceneNode(
                id=track.id,
                class_name=self._get_class_name(track.class_id),
                position=track.state[:3],
                orientation=np.array([0, 0, 0, 1]),
                dimensions=np.array([0.5, 0.5, 0.5]),
                confidence=track.confidence
            )
            self.scene_graph.add_node(node)

        # Recompute relations
        self.scene_graph.compute_relations()

    def _get_class_name(self, class_id: int) -> str:
        """Map class ID to name."""
        class_names = {
            0: 'person', 1: 'car', 2: 'bicycle',
            3: 'chair', 4: 'table', 5: 'cup'
        }
        return class_names.get(class_id, 'unknown')

    def publish_tracks(self, tracks):
        """Publish tracked objects."""
        msg = Detection3DArray()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'

        for track in tracks:
            det = Detection3D()
            det.bbox.center.position.x = float(track.state[0])
            det.bbox.center.position.y = float(track.state[1])
            det.bbox.center.position.z = float(track.state[2])
            det.tracking_id = track.id
            msg.detections.append(det)

        self.tracks_pub.publish(msg)

    def publish_scene_visualization(self):
        """Publish scene graph visualization."""
        markers = MarkerArray()

        # Node markers
        for i, node in enumerate(self.scene_graph.nodes.values()):
            marker = Marker()
            marker.header.frame_id = 'base_link'
            marker.id = i
            marker.type = Marker.CUBE
            marker.action = Marker.ADD
            marker.pose.position.x = float(node.position[0])
            marker.pose.position.y = float(node.position[1])
            marker.pose.position.z = float(node.position[2])
            marker.scale.x = 0.3
            marker.scale.y = 0.3
            marker.scale.z = 0.3
            marker.color.a = 0.8
            marker.color.r = 0.0
            marker.color.g = 1.0
            marker.color.b = 0.0
            markers.markers.append(marker)

        self.scene_pub.publish(markers)


def main():
    rclpy.init()
    node = PerceptionPipeline()
    rclpy.spin(node)


if __name__ == '__main__':
    main()
```

### Launch File

```python title="perception_pipeline.launch.py"
"""Launch perception pipeline."""
from launch import LaunchDescription
from launch_ros.actions import Node, ComposableNodeContainer
from launch_ros.descriptions import ComposableNode

def generate_launch_description():
    # Isaac ROS detection
    detection_node = ComposableNode(
        package='isaac_ros_detectnet',
        plugin='nvidia::isaac_ros::detectnet::DetectNetDecoderNode',
        name='detectnet',
        parameters=[{
            'model_file_path': '/models/peoplenet.plan',
        }]
    )

    # Isaac ROS visual SLAM
    slam_node = ComposableNode(
        package='isaac_ros_visual_slam',
        plugin='nvidia::isaac_ros::visual_slam::VisualSlamNode',
        name='visual_slam',
    )

    # GPU container
    container = ComposableNodeContainer(
        name='perception_container',
        namespace='',
        package='rclcpp_components',
        executable='component_container_mt',
        composable_node_descriptions=[detection_node, slam_node],
        output='screen'
    )

    # Perception pipeline
    pipeline = Node(
        package='perception_pipeline',
        executable='perception_pipeline',
        name='perception_pipeline',
        parameters=[{
            'detection_threshold': 0.5,
            'tracking_enabled': True,
        }],
        output='screen'
    )

    return LaunchDescription([container, pipeline])
```

---

## Performance Optimization

### Profiling Tools

```bash title="Profile perception pipeline"
# ROS 2 topic bandwidth
ros2 topic bw /perception/tracks

# ROS 2 topic frequency
ros2 topic hz /perception/tracks

# CPU profiling
perf record -g ros2 run perception_pipeline perception_pipeline
perf report

# GPU profiling
nsys profile ros2 launch perception_pipeline perception.launch.py
```

### Optimization Strategies

| Bottleneck | Solution | Impact |
|------------|----------|--------|
| **Image resize** | GPU-accelerated resize | 5-10x faster |
| **Detection** | TensorRT optimization | 10x faster |
| **Point cloud** | Voxel downsampling | Reduce data 10x |
| **Tracking** | Parallel association | 2-3x faster |
| **Memory** | Pre-allocated buffers | Reduce GC |

---

## Exercise 1: Build Detection Pipeline

:::tip Exercise 1: Multi-Sensor Detection
**Objective**: Create a detection pipeline fusing camera and LiDAR.

**Steps**:

1. Set up synchronized sensor inputs
2. Run Isaac ROS DetectNet on camera images
3. Run PointPillars on LiDAR point clouds
4. Implement detection fusion
5. Visualize fused detections in RViz

**Expected Output**:
- Fused detections at 20+ Hz
- 3D bounding boxes with class labels
- Latency < 100ms

**Time Estimate**: 90 minutes
:::

---

## Exercise 2: Object Tracking

:::tip Exercise 2: Multi-Object Tracker
**Objective**: Implement persistent tracking across frames.

**Steps**:

1. Implement Kalman filter tracker
2. Configure track management parameters
3. Test with moving objects
4. Evaluate tracking metrics (MOTA, IDF1)
5. Handle occlusions and reappearances

**Metrics**:
- Track fragmentation < 10%
- ID switches < 5%
- Tracking accuracy > 80%

**Time Estimate**: 60 minutes
:::

---

## Exercise 3: Scene Graph System

:::tip Exercise 3: Semantic Scene Understanding
**Objective**: Build a scene graph from perception outputs.

**Steps**:

1. Create scene graph data structures
2. Populate from tracked objects
3. Compute spatial relations
4. Implement simple queries
5. Visualize in RViz

**Example Queries**:
- "Find all cups on tables"
- "Objects near the robot"
- "People in front of the robot"

**Time Estimate**: 75 minutes
:::

---

## Summary

In this chapter, you learned:

- **Architecture**: Modular pipeline design principles
- **Synchronization**: Multi-sensor time alignment
- **Fusion**: Combining camera and LiDAR detections
- **Tracking**: Kalman filter-based multi-object tracking
- **Scene Graphs**: Semantic scene representation
- **Optimization**: Profiling and performance tuning

A well-designed perception pipeline transforms raw sensor data into actionable understanding that robots can use for planning and control. The key is balancing accuracy, latency, and robustness.

This completes Module 3 on NVIDIA Isaac. Next, explore [Module 4: Vision-Language-Action Models](/docs/module-4) to learn how foundation models enable natural language robot control.

## Further Reading

- [Isaac ROS Perception](https://github.com/NVIDIA-ISAAC-ROS)
- [ROS 2 Perception Pipeline](https://navigation.ros.org/perception/index.html)
- [Multi-Object Tracking Survey](https://arxiv.org/abs/2006.16567)
- [Scene Graph Generation](https://cs.stanford.edu/~danfei/scene-graph.html)
- [FilterPy Kalman Filter](https://filterpy.readthedocs.io/)
