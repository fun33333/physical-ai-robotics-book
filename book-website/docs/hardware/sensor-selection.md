---
title: "Sensor Selection Guide"
sidebar_position: 5
description: "Comprehensive guide to selecting appropriate sensors for Physical AI applications with technical specifications and use cases"
---

# Sensor Selection Guide

## Overview

Sensors are the eyes, ears, and touch of your robot, providing the perception capabilities essential for Physical AI systems. Selecting the right sensors for your application requires balancing cost, accuracy, reliability, and computational requirements. This guide will help you choose the most appropriate sensors for your specific robotics projects.

## Sensor Categories

### Vision Sensors

Vision sensors enable your robot to perceive its environment in 2D and 3D, forming the foundation of most AI perception systems.

#### RGB Cameras
- **Purpose**: Color image capture for object recognition, navigation, and monitoring
- **Key Specifications**:
  - Resolution (e.g., 640x480, 1280x720, 1920x1080)
  - Frame rate (e.g., 30, 60, 120 fps)
  - Field of view (FOV)
  - Interface (USB, GigE, MIPI)

| Sensor Model | Resolution | Frame Rate | Interface | Price Range | Best Use Case |
|--------------|------------|------------|-----------|-------------|---------------|
| Intel RealSense D435 | 1280x720 RGB, 1280x720 Depth | 30-90 fps | USB 3.0 | $200 | General purpose with depth |
| FLIR Blackfly S | Various options up to 5.1MP | Up to 80 fps | GigE, USB3 | $400-1200 | Industrial machine vision |
| Arducam IMX477 | 8MP | 30 fps | MIPI/USB | $100 | Pi-based systems |
| Basler acA1300-75gc | 1.3MP | 75 fps | GigE | $300 | High-speed applications |

#### RGB-D Cameras
- **Purpose**: Simultaneous color and depth information for 3D scene understanding
- **Key Specifications**:
  - Depth accuracy (mm)
  - Range (meters)
  - Resolution of depth map
  - Operating principle (stereo, structured light, ToF)

| Sensor Model | Depth Accuracy | Range | Resolution | Technology | Price Range |
|--------------|----------------|-------|------------|------------|-------------|
| Intel RealSense D435 | ±2% at 1m | 0.2-10m | 1280x720 | Stereo | $200 |
| Intel RealSense D455 | ±1% at 1m | 0.1-3.5 (close) / 0.3-10 (far) m | 1280x720 | Stereo | $500 |
| Stereolabs ZED 2i | Better outdoor | 0.2-20m | 2208x1242 | Stereo | $600 |
| Orbbec Astra Pro 2 | ±3mm at 1m | 0.6-6m | 640x480 | Structured light | $150 |

#### Thermal Cameras
- **Purpose**: Detecting heat signatures, useful in low-visibility conditions
- **Key Specifications**:
  - Thermal sensitivity
  - Temperature range
  - Resolution (thermal image)

| Sensor Model | Resolution | Sensitivity | Range | Interface | Price Range |
|--------------|------------|-------------|-------|-----------|-------------|
| FLIR Lepton `3.5` | 160x120 | less than 50mK | -10 to +400 degrees C | SPI/USB | $350 |
| Seek Compact Pro (320x240) | 320x240 | less than 50mK | -40 to +330 degrees C | USB | $1000 |

### Range Sensors

Range sensors provide distance measurements that are essential for navigation and mapping.

#### 2D LiDAR
- **Purpose**: Planar distance scanning for navigation and mapping
- **Key Specifications**:
  - Range (meters)
  - Angular resolution (degrees)
  - Scan rate (Hz)
  - Number of points per revolution

| Sensor Model | Range | Resolution | Scan Rate | Points/Rev | Price Range |
|--------------|-------|------------|-----------|------------|-------------|
| Slamtec RPLidar A1 | 6m | 0.44° | 5-15 Hz | ~8000 | $150 |
| Hokuyo URG-04LX-UG01 | 4m | 0.36° | 10 Hz | 682 | $350 |
| SICK TiM571 | 25m | 0.33° | 15 Hz | ~4000 | $1300 |
| Velodyne Puck | 100m | 0.4° vertical | 5-20 Hz | ~300,000 | $4000 |

#### 3D LiDAR
- **Purpose**: Full 3D environment mapping
- **Key Specifications**:
  - Range (meters)
  - Angular resolution (vertical/horizontal)
  - FOV (vertical/horizontal)
  - Point cloud density

| Sensor Model | Range | Vertical Resolution | Horizontal Resolution | FOV | Price Range |
|--------------|-------|-------------------|---------------------|-----|-------------|
| Velodyne VLP-16 | 100m | 2° | 0.1-0.4° | 30° vertical, 360° horizontal | $8000 |
| Ouster OS0-32 | 75m | 0.75° | 1.15° | 90° vertical, 360° horizontal | $6000 |
| Livox Horizon | 260m | 0.18°-3.96° | 0.18°-0.36° | 81.7° vertical, 360° horizontal | $2000 |

#### Time-of-Flight (ToF) Sensors
- **Purpose**: Distance measurement using light pulses
- **Key Specifications**:
  - Range (meters)
  - Resolution (X,Y)
  - Accuracy

| Sensor Model | Range | Resolution | Accuracy | Price Range |
|--------------|-------|------------|----------|-------------|
| PMD O3M324 | 0.1-4m | 176x132 | ±10mm | $500 |
| Intel RealSense L515 | 0.25-9m | 1024x768 | ±1% | $300 |

### Inertial Sensors

Inertial sensors provide information about acceleration, rotation, and orientation, crucial for robot state estimation.

#### IMUs (Inertial Measurement Units)
- **Purpose**: Measure linear acceleration and angular velocity
- **Key Specifications**:
  - Accelerometer range (±g)
  - Gyroscope range (±dps)
  - Noise density
  - Bias instability

| Sensor Model | Accel Range | Gyro Range | Noise Density Accel | Noise Density Gyro | Price Range |
|--------------|-------------|------------|-------------------|------------------|-------------|
| Bosch BNO055 | ±2/4/8/16g | ±2000°/s | 150 μg/√Hz | 11.8 mdps/√Hz | $30 |
| ADIS16470 | ±180g | ±450°/s | 2.8 mg/√Hz | 1.5 mdps/√Hz | $350 |
| VectorNav VN-100 | ±16g | ±2000°/s | less than 0.5 mg/√Hz | less than 0.015 °/s/√Hz | $300 |

#### Magnetometers
- **Purpose**: Measure magnetic fields for compass heading
- **Key Specifications**:
  - Resolution
  - Range
  - Power consumption

| Sensor Model | Resolution | Range | Power | Price Range |
|--------------|------------|-------|-------|-------------|
| Honeywell HMC5883L | 16-bit | ±8.1 Gauss | 100 μA | $10 |
| Bosch BMM150 | 16-bit | ±1300 μT | 8.4 μA | $5 |

### Environmental Sensors

Environmental sensors help robots understand their surroundings beyond visual and motion information.

#### Temperature Sensors
- **Purpose**: Monitor environmental or system temperatures
- **Key Specifications**:
  - Accuracy
  - Range
  - Response time

#### Humidity Sensors
- **Purpose**: Measure environmental humidity
- **Key Specifications**:
  - Accuracy
  - Range
  - Response time

| Sensor Model | Accuracy | Range | Interface | Price Range |
|--------------|----------|-------|-----------|-------------|
| Sensirion SHT30 | ±2% RH | 0-100% RH | I2C | $10 |
| Honeywell HIH6130 | ±2% RH | 0-100% RH | I2C | $8 |

#### Barometric Pressure Sensors
- **Purpose**: Measure atmospheric pressure for altitude estimation
- **Key Specifications**:
  - Accuracy
  - Pressure range
  - Temperature compensation

| Sensor Model | Accuracy | Resolution | Interface | Price Range |
|--------------|----------|------------|-----------|-------------|
| Bosch BMP280 | ±1 Pa | 0.01 Pa | I2C/SPI | $5 |
| TE Connectivity MS5611 | ±1.5 mbar | 0.012 mbar | I2C/SPI | $15 |

## Selection Criteria

:::tip Important Consideration
When selecting sensors, always consider the entire pipeline: sensing → processing → action. A high-resolution sensor is only valuable if your compute platform can process the data in real-time for your application.
:::

### Application-Specific Considerations

#### Navigation and Mapping
- **Primary Sensors**: 2D or 3D LiDAR, RGB-D cameras
- **Secondary Sensors**: IMU for odometry enhancement
- **Key Factors**: Accuracy, range, update rate
- **Example**: Hokuyo URG for indoor navigation, Velodyne for outdoor mapping

#### Manipulation
- **Primary Sensors**: RGB-D cameras, force/torque sensors
- **Secondary Sensors**: Joint encoders, tactile sensors
- **Key Factors**: Accuracy, latency, spatial resolution
- **Example**: Intel RealSense for object recognition, ATI Gamma for force control

#### Human-Robot Interaction
- **Primary Sensors**: RGB cameras, microphones, proximity sensors
- **Secondary Sensors**: Speakers, displays, haptic feedback
- **Key Factors**: Privacy, user-friendliness, responsiveness
- **Example**: Multiple RGB cameras for social robotics

### Environmental Factors

#### Indoor vs. Outdoor
- **Outdoor Challenges**: Sunlight interference, weather, dust
- **Indoor Advantages**: Controlled lighting, stable conditions
- **Sensor Selection**: 
  - Outdoor: LiDAR, thermal cameras, GPS
  - Indoor: IR sensors, structured light cameras

#### Lighting Conditions
- **Low Light**: Thermal cameras, LiDAR, active illumination
- **Harsh Lighting**: Polarizing filters, HDR cameras, LiDAR
- **Variable Lighting**: Adaptive exposure, multiple sensor fusion

#### Temperature and Humidity
- **Extreme Cold**: Heated enclosures, special lubricants
- **High Humidity**: Sealed enclosures, corrosion-resistant materials
- **Large Variations**: Temperature-compensated sensors

### Budget Constraints

#### Tier 1: Budget (under $500)
- IMU: MPU6050 or Bosch BNO055
- Camera: USB webcam or Intel RealSense D435
- Range: HC-SR04 ultrasonic sensors
- Connectivity: USB, WiFi, Bluetooth

#### Tier 2: Recommended ($500-$2000)
- Camera: Intel RealSense D455 or Stereolabs ZED
- LiDAR: Slamtec RPLidar or Hokuyo URG
- IMU: ADIS16470 or VectorNav VN-100
- Connectivity: Ethernet, CAN bus

#### Tier 3: Premium ($2000+)
- Camera: Basler or FLIR industrial cameras
- LiDAR: SICK TiM or Velodyne Puck
- IMU: High-end VectorNav or Epson G370
- Connectivity: Time-synchronized protocols

### Integration Complexity

#### Hardware Integration
- **Connectors**: Standard vs. proprietary
- **Mounting**: Standard brackets vs. custom solutions
- **Power**: Voltage requirements, power consumption
- **Form Factor**: Size constraints in robot design

#### Software Integration
- **Driver Availability**: ROS packages, SDKs, APIs
- **Calibration**: Factory calibrated vs. field calibration
- **Data Formats**: Standardized vs. proprietary
- **Update Rates**: Synchronization requirements

## Sensor Fusion Concepts

### Basic Fusion Approaches

For many robotics applications, using multiple sensors together (sensor fusion) provides more robust and accurate perception than individual sensors.

#### Early Fusion
- Combine sensor data at the raw signal level
- Requires time synchronization
- High data rates but potentially better resolution

#### Late Fusion
- Process sensors independently then combine results
- Lower computational requirements
- More fault-tolerant to individual sensor failures

#### State Vector Fusion
- Combine state estimates from different sensors
- Uses techniques like Kalman filtering
- Common in navigation applications

### Common Fusion Configurations

#### Visual-Inertial Odometry (VIO)
- Combines camera images with IMU data
- Provides robust motion estimation
- Used in applications where wheel odometry is unreliable

#### SLAM with Multiple Sensors
- Combines LiDAR, cameras, and IMUs
- Provides map creation and localization
- Robust to environmental conditions

## Best Practices

### Sensor Placement
- Position sensors to minimize occlusions
- Consider safety: avoid placing fragile sensors in collision zones
- Plan for maintenance access
- Account for robot's center of gravity when mounting heavy sensors

### Calibration Requirements
- Factory-calibrated sensors: Less setup but limited flexibility
- Field-calibrated sensors: More setup but potentially better performance
- Maintain calibration records and schedules

### Mounting Considerations
- Rigid mounting to minimize vibration effects
- Shock absorption for delicate sensors
- Thermal management to avoid heating effects
- Cable management to prevent strain

### Redundancy Planning
- Critical sensors: Consider redundant systems
- Non-critical sensors: Single system with fallbacks
- Budget for redundancy: Factor into overall system cost

## Testing and Validation

### Performance Validation
- Test sensors under expected environmental conditions
- Validate accuracy against known references
- Check for electromagnetic interference
- Verify data rates and communication reliability

### Integration Testing
- Verify sensor data is accessible in your software framework
- Test time synchronization between sensors
- Validate sensor calibration procedures
- Test error handling and fault recovery

---

:::tip Exercise 4: Sensor Selection for a Mobile Robot
**Objective**: Select an appropriate sensor suite for a mobile robot operating in an indoor warehouse environment

**Time Estimate**: 45 minutes

**Steps**:
1. Define your robot's primary functions (navigation, mapping, obstacle avoidance)
2. Identify environmental challenges (lighting, obstacles, space constraints)
3. Create a list of required sensor types
4. Research specific sensor models that meet your requirements
5. Calculate the total cost of your sensor suite
6. Consider backup options for critical sensors

**Expected Result**: A detailed sensor specification document with models, specifications, and justification for your choices

**Hints**:
- Consider the trade-offs between LiDAR and camera sensing
- Account for the robot's computational resources
- Factor in the need for redundancy for critical functions
:::