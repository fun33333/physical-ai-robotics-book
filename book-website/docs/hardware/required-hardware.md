---
title: "Required Hardware Specifications"
sidebar_position: 2
description: "Essential hardware components needed for the Physical AI course with detailed specifications and pricing"
---

# Required Hardware Specifications

## Overview

This document outlines the essential hardware components you'll need to complete the Physical AI course successfully. We've organized the requirements into three tiers to accommodate different budgets and use cases: Budget, Recommended, and Premium.

## Computational Hardware

The computational requirements vary significantly depending on whether you're focusing on simulation, real-time robotics, or machine learning inference.

### GPU Options

| Component | Budget | Recommended | Premium |
|-----------|--------|-------------|---------|
| Graphics Card | GTX 1660 Super (6GB VRAM) (~$230) | RTX 3060 (12GB VRAM) (~$400) | RTX 4080 (16GB VRAM) (~$900) |
| Alternative | Integrated graphics (slow) | RTX 3070 (8GB VRAM) (~$500) | RTX 4090 (24GB VRAM) (~$1600) |

**Notes**:
- Minimum 6GB VRAM required for Isaac Sim
- 12GB+ VRAM recommended for VLA model inference
- Check power supply requirements before purchase

### CPU Options

| Component | Budget | Recommended | Premium |
|-----------|--------|-------------|---------|
| Processor | AMD Ryzen 5 3600 / Intel i5-10400F | AMD Ryzen 7 3700X / Intel i7-10700KF | AMD Ryzen 9 5900X / Intel i9-12900K |
| Cores/Threads | 6C/12T | 8C/16T | 12C/24T or 16C/24T |

### RAM Options

| Component | Budget | Recommended | Premium |
|-----------|--------|-------------|---------|
| Memory | 16GB DDR4-3200MHz | 32GB DDR4-3200MHz | 64GB DDR4-3600MHz |
| Expandability | Limited to 32GB | Upgradable to 64GB | Upgradable to 128GB+ |

### Storage Options

| Component | Budget | Recommended | Premium |
|-----------|--------|-------------|---------|
| Primary SSD | 500GB NVMe SATA | 1TB NVMe PCIe Gen3 | 2TB NVMe PCIe Gen4 |
| Secondary Storage | None | 2TB HDD | 4TB HDD + 1TB NVMe for cache |

## Sensing Hardware

For physical robots, you'll need various sensors to perceive the environment.

### RGB-D Cameras

| Component | Budget | Recommended | Premium |
|-----------|--------|-------------|---------|
| Stereo Camera | Stereo Labs ZED Mini (~$400) | Intel RealSense D455 (~$500) | Stereolabs ZED 2i (~$600) |
| Alternative | Intel RealSense D435 (~$200) | Photoneo MotionCam-3D (~$1000) | Ouster OS-0-64 Gen2 (~$8000) |

### LiDAR Sensors

| Component | Budget | Recommended | Premium |
|-----------|--------|-------------|---------|
| 2D LiDAR | Slamtec RPLidar A1 (~$150) | Hokuyo URG-04LX-UG01 (~$350) | Sick TiM571 (~$1300) |
| 3D LiDAR | Velodyne Puck (~$4000) | Ouster OS0-32 (~$6000) | Livox Horizon (~$2000) |

### IMU Sensors

| Component | Budget | Recommended | Premium |
|-----------|--------|-------------|---------|
| IMU Module | Bosch BNO055 (~$30) | Adafruit BNO085 (~$50) | VectorNav VN-100R (~$300) |

## Actuation Hardware

For robots that physically interact with the environment:

### Servo Motors

| Component | Budget | Recommended | Premium |
|-----------|--------|-------------|---------|
| Standard Servo | SG90 (~$5) | Dynamixel XL430-W250 (~$60) | Dynamixel MX-64AT (~$200) |
| High Torque | DS3218 20kg (~$25) | Dynamixel XH430-V350 (~$100) | Futaba S3003 (~$150) |

### Motor Controllers

| Component | Budget | Recommended | Premium |
|-----------|--------|-------------|---------|
| Servo Controller | PCA9685 16-channel (~$15) | Herkulex DRS-0402 (~$80) | Roboteq ESCAP (~$200) |

## Robot Platforms

### Mobile Robots

| Component | Budget | Recommended | Premium |
|-----------|--------|-------------|---------|
| Differential Drive | Dagu Wild Thumper (~$200) | Clearpath Jackal (~$4000) | Fetch Freight500 (~$25000) |
| Alternative | Pololu Zumo (~$100) | TurtleBot 4 (~$1200) | Boston Dynamics Handle (~$100000) |

### Manipulator Arms

| Component | Budget | Recommended | Premium |
|-----------|--------|-------------|---------|
| Desktop Arm | WidowX 250 (~$1200) | Franka Emika Panda (~$50000) | KUKA LBR iiwa (~$80000) |
| DIY Kit | OpenManipulator (~$300) | Interbotix Arms (~$2000) | Kinova Gen3 (~$30000) |

## Network and Communication

### Network Interface

| Component | Budget | Recommended | Premium |
|-----------|--------|-------------|---------|
| Ethernet Switch | Unmanaged 5-port (~$20) | Managed 8-port (~$80) | Managed PoE+ (~$200) |

### Wireless Options

| Component | Budget | Recommended | Premium |
|-----------|--------|-------------|---------|
| Wi-Fi Dongle | Generic AC (~$15) | ASUS USB-AC53(~$30) | Ubiquiti mFi mPower Pro (~$100) |

## Workshop Tools

### Essential Tools

| Component | Budget | Recommended | Premium |
|-----------|--------|-------------|---------|
| Multimeter | Basic (~$20) | Fluke 17B+ (~$100) | Keysight U1253B (~$300) |
| Soldering Iron | Hakko FX888D (~$100) | Weller WE1010 (~$150) | JBC CDS 20 (~$300) |
| Oscilloscope | DS1054Z (~$400) | Rigol MSO5074 (~$1500) | Keysight 3000T (~$3000) |

## Power Systems

### Power Supplies

| Component | Budget | Recommended | Premium |
|-----------|--------|-------------|---------|
| Bench Power Supply | Siglent SPD3303X (~$150) | Rigol DP832 (~$400) | Keysight E36312A (~$800) |
| DC Distribution | Generic Breadboard (~$5) | MB102 Breadboard (~$10) | Custom PCB Power Dist (~$50) |

### Batteries

| Component | Budget | Recommended | Premium |
|-----------|--------|-------------|---------|
| LiPo Battery | Generic 11.1V 5000mAh (~$30) | Turnigy nano-tech 5000mAh (~$60) | Custom battery pack (~$150) |

## Additional Components

### Cables and Connectors

| Component | Budget | Recommended | Premium |
|-----------|--------|-------------|---------|
| Jumper Wire Set | Assorted kit (~$10) | Premium male/female (~$20) | Custom cable assemblies (~$100) |
| USB Adapters | Generic USB-A to micro-B (~$5) | High-quality cables (~$15) | Isolated converters (~$50) |

### Miscellaneous

| Component | Budget | Recommended | Premium |
|-----------|--------|-------------|---------|
| Prototyping Board | Breadboards (~$10) | Perfboard (~$15) | Custom PCB (~$50) |
| Hardware Kit | Generic M3/M4 screws (~$10) | Delran rod/plate (~$25) | Precision machined (~$100) |

## Cost Summary

Based on the above specifications, here's a rough cost breakdown:

| Configuration | Estimated Cost |
|---------------|----------------|
| Budget Setup | ~$1,500 |
| Recommended Setup | ~$7,500 |
| Premium Setup | ~$50,000+ |

## Purchasing Recommendations

1. Start with a basic computer setup that meets minimum requirements - you can always upgrade later
2. For sensors, begin with a reliable RGB-D camera like the Intel RealSense D435 for versatility
3. Consider buying components from reputable suppliers who offer good technical support
4. For robot platforms, start with simulation before investing in physical hardware

---

:::tip Exercise 1: Hardware Planning
**Objective**: Create a hardware procurement plan for your specific situation

**Time Estimate**: 30 minutes

**Steps**:
1. Determine your budget constraints
2. Identify your primary use case (simulation, mobile robot, manipulator, etc.)
3. List the top 5 components you would purchase first
4. Research current prices from 2-3 suppliers
5. Note any compatibility requirements between components

**Expected Result**: A prioritized hardware wish list with prices and compatibility notes

**Hints**:
- Consider shipping times and return policies
- Look for academic discounts if you're a student
- Factor in taxes and import duties if ordering internationally
:::