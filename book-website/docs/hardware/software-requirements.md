---
title: "Software Requirements"
sidebar_position: 5
description: "Software dependencies, operating systems, and development environment setup for Physical AI."
---

# Software Requirements

Physical AI development requires a carefully configured software environment. This chapter guides you through setting up the operating system, development tools, and dependencies needed for the curriculum. A consistent environment prevents the dreaded "works on my machine" problem.

## Overview

The software stack includes:

1. **Operating System**: Ubuntu 22.04 LTS (primary)
2. **ROS 2**: Humble Hawksbill
3. **Simulation**: Gazebo Harmonic, Isaac Sim
4. **Deep Learning**: PyTorch, CUDA
5. **Development Tools**: VS Code, Git, Docker

---

## Operating System

### Ubuntu 22.04 LTS (Recommended)

Ubuntu 22.04 is the primary supported platform:

| Aspect | Details |
|--------|---------|
| Version | 22.04.3 LTS (Jammy Jellyfish) |
| Support until | April 2027 |
| ROS 2 support | Humble (Tier 1) |
| NVIDIA support | Full driver support |

**Installation options**:

1. **Native install** (recommended for best performance)
2. **Dual boot** with Windows
3. **WSL2** (Windows Subsystem for Linux)
4. **Virtual machine** (VMware, VirtualBox)

### Installation Guide

```bash title="Post-install essentials"
# Update system
sudo apt update && sudo apt upgrade -y

# Install essential tools
sudo apt install -y \
    build-essential \
    cmake \
    git \
    curl \
    wget \
    python3-pip \
    python3-venv \
    software-properties-common

# Set up locale
sudo locale-gen en_US en_US.UTF-8
sudo update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
export LANG=en_US.UTF-8
```

### WSL2 Setup (Windows Users)

For Windows users, WSL2 provides excellent Linux compatibility:

```powershell title="PowerShell (Admin)"
# Enable WSL
wsl --install -d Ubuntu-22.04

# Set WSL2 as default
wsl --set-default-version 2

# Update WSL kernel
wsl --update
```

**WSL2 Limitations**:
- No native GPU for display (use X server)
- Isaac Sim requires native Linux or cloud
- USB device access requires additional setup
- Performance ~90% of native

**WSL2 GPU Setup** (for CUDA):

```bash title="In WSL2 Ubuntu"
# NVIDIA drivers are shared from Windows
# Install CUDA toolkit for WSL
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update
sudo apt install cuda-toolkit-12-2
```

---

## ROS 2 Humble Installation

### Binary Installation (Recommended)

```bash title="Install ROS 2 Humble"
# Set locale
sudo apt update && sudo apt install locales
sudo locale-gen en_US en_US.UTF-8
sudo update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8

# Setup sources
sudo apt install software-properties-common
sudo add-apt-repository universe
sudo apt update && sudo apt install curl -y
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null

# Install ROS 2
sudo apt update
sudo apt install ros-humble-desktop  # Full desktop install

# Install development tools
sudo apt install ros-dev-tools

# Source ROS 2 (add to ~/.bashrc)
echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
source ~/.bashrc
```

### Verify Installation

```bash title="Test ROS 2"
# Terminal 1
ros2 run demo_nodes_cpp talker

# Terminal 2
ros2 run demo_nodes_py listener

# You should see messages being exchanged
```

### Essential ROS 2 Packages

```bash title="Install common packages"
sudo apt install -y \
    ros-humble-navigation2 \
    ros-humble-nav2-bringup \
    ros-humble-slam-toolbox \
    ros-humble-robot-state-publisher \
    ros-humble-joint-state-publisher \
    ros-humble-xacro \
    ros-humble-ros-gz \
    ros-humble-moveit \
    ros-humble-ros2-control \
    ros-humble-ros2-controllers
```

---

## NVIDIA Software Stack

### Driver Installation

```bash title="Install NVIDIA drivers"
# Check recommended driver
ubuntu-drivers devices

# Install recommended driver
sudo ubuntu-drivers autoinstall

# Or install specific version
sudo apt install nvidia-driver-545

# Reboot
sudo reboot

# Verify
nvidia-smi
```

### CUDA Toolkit

```bash title="Install CUDA 12.2"
# Download and install CUDA
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update
sudo apt install cuda-toolkit-12-2

# Add to PATH (add to ~/.bashrc)
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Verify
nvcc --version
```

### cuDNN Installation

```bash title="Install cuDNN"
# Download from NVIDIA (requires account)
# https://developer.nvidia.com/cudnn

# Install downloaded packages
sudo dpkg -i cudnn-local-repo-ubuntu2204-8.9.7.29_1.0-1_amd64.deb
sudo cp /var/cudnn-local-repo-ubuntu2204-8.9.7.29/cudnn-local-*-keyring.gpg /usr/share/keyrings/
sudo apt update
sudo apt install libcudnn8 libcudnn8-dev
```

---

## Python Environment

### Python Setup

```bash title="Python environment"
# Ensure Python 3.10
python3 --version  # Should be 3.10.x

# Install pip
sudo apt install python3-pip

# Create virtual environment for projects
python3 -m venv ~/ros2_ws/venv
source ~/ros2_ws/venv/bin/activate

# Install common packages
pip install --upgrade pip
pip install \
    numpy \
    scipy \
    matplotlib \
    opencv-python \
    torch \
    torchvision \
    transformers
```

### PyTorch with CUDA

```bash title="Install PyTorch with CUDA"
# PyTorch with CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Verify CUDA availability
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python3 -c "import torch; print(f'Device: {torch.cuda.get_device_name(0)}')"
```

---

## Gazebo Installation

### Gazebo Harmonic (New Gazebo)

```bash title="Install Gazebo Harmonic"
# Add Gazebo repository
sudo wget https://packages.osrfoundation.org/gazebo.gpg -O /usr/share/keyrings/pkgs-osrf-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/pkgs-osrf-archive-keyring.gpg] http://packages.osrfoundation.org/gazebo/ubuntu-stable $(lsb_release -cs) main" | sudo tee /etc/apt/sources.list.d/gazebo-stable.list > /dev/null

# Install Gazebo Harmonic
sudo apt update
sudo apt install gz-harmonic

# Verify
gz sim --version
```

### ROS 2-Gazebo Integration

```bash title="Install ros_gz"
sudo apt install ros-humble-ros-gz
```

---

## Isaac Sim Installation

### System Requirements

| Requirement | Minimum | Recommended |
|-------------|---------|-------------|
| GPU | RTX 2070 | RTX 3080+ |
| VRAM | 8GB | 12GB+ |
| RAM | 32GB | 64GB |
| Storage | 50GB | 100GB SSD |
| Driver | 525.60+ | Latest |

### Installation via Omniverse

1. **Download Omniverse Launcher**:
   - Visit [NVIDIA Omniverse](https://www.nvidia.com/en-us/omniverse/)
   - Download the Linux AppImage

2. **Install Launcher**:
   ```bash
   chmod +x omniverse-launcher-linux.AppImage
   ./omniverse-launcher-linux.AppImage
   ```

3. **Install Isaac Sim**:
   - Open Omniverse Launcher
   - Go to Exchange → Isaac Sim
   - Click Install
   - Wait for download (~15GB)

4. **Verify Installation**:
   ```bash
   # Launch Isaac Sim
   ~/.local/share/ov/pkg/isaac_sim-2023.1.1/isaac-sim.sh
   ```

### Isaac Sim with ROS 2

```bash title="Enable ROS 2 bridge"
# In Isaac Sim, enable ROS 2 extension:
# Window → Extensions → Search "ROS2" → Enable

# Or via command line
~/.local/share/ov/pkg/isaac_sim-*/isaac-sim.sh --enable omni.isaac.ros2_bridge
```

---

## Development Tools

### Visual Studio Code

```bash title="Install VS Code"
# Download and install
wget -qO- https://packages.microsoft.com/keys/microsoft.asc | gpg --dearmor > packages.microsoft.gpg
sudo install -D -o root -g root -m 644 packages.microsoft.gpg /etc/apt/keyrings/packages.microsoft.gpg
echo "deb [arch=amd64,arm64,armhf signed-by=/etc/apt/keyrings/packages.microsoft.gpg] https://packages.microsoft.com/repos/code stable main" | sudo tee /etc/apt/sources.list.d/vscode.list > /dev/null
sudo apt update
sudo apt install code
```

**Recommended Extensions**:
- Python
- ROS (by Microsoft)
- C/C++
- XML Tools
- YAML
- GitLens
- Docker

### Docker Setup

```bash title="Install Docker"
# Install Docker
sudo apt install docker.io docker-compose

# Add user to docker group (logout required)
sudo usermod -aG docker $USER

# Enable service
sudo systemctl enable docker
sudo systemctl start docker

# Verify
docker run hello-world
```

### NVIDIA Container Toolkit

For GPU access in Docker containers:

```bash title="Install NVIDIA Container Toolkit"
# Add repository
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

# Install
sudo apt update
sudo apt install nvidia-container-toolkit

# Configure Docker
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# Test
docker run --rm --gpus all nvidia/cuda:12.2.0-base-ubuntu22.04 nvidia-smi
```

---

## Git Configuration

```bash title="Configure Git"
# Set identity
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"

# Set default branch name
git config --global init.defaultBranch main

# Useful aliases
git config --global alias.st status
git config --global alias.co checkout
git config --global alias.br branch
git config --global alias.lg "log --oneline --graph --all"
```

---

## Verification Checklist

Run these commands to verify your setup:

```bash title="verification.sh"
#!/bin/bash
echo "=== System Verification ==="

echo -n "Ubuntu version: "
lsb_release -d | cut -f2

echo -n "Python version: "
python3 --version

echo -n "ROS 2: "
ros2 --version 2>/dev/null || echo "NOT INSTALLED"

echo -n "NVIDIA driver: "
nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null || echo "NOT INSTALLED"

echo -n "CUDA version: "
nvcc --version 2>/dev/null | grep release | awk '{print $6}' || echo "NOT INSTALLED"

echo -n "Gazebo: "
gz sim --version 2>/dev/null | head -1 || echo "NOT INSTALLED"

echo -n "Docker: "
docker --version 2>/dev/null || echo "NOT INSTALLED"

echo -n "PyTorch CUDA: "
python3 -c "import torch; print(torch.cuda.is_available())" 2>/dev/null || echo "NOT INSTALLED"

echo "=== Verification Complete ==="
```

**Expected Output**:
```
=== System Verification ===
Ubuntu version: Ubuntu 22.04.3 LTS
Python version: Python 3.10.12
ROS 2: humble
NVIDIA driver: 545.23.08
CUDA version: V12.2.140
Gazebo: Gazebo Sim, version 8.3.0
Docker: Docker version 24.0.7
PyTorch CUDA: True
=== Verification Complete ===
```

---

## Troubleshooting

### Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| `nvidia-smi` not found | Driver not installed | Install NVIDIA driver |
| ROS 2 commands not found | Not sourced | Add `source /opt/ros/humble/setup.bash` to ~/.bashrc |
| CUDA not detected by PyTorch | Version mismatch | Match PyTorch CUDA version with installed CUDA |
| Gazebo crashes | GPU driver issue | Update NVIDIA driver |
| Permission denied (Docker) | Not in docker group | Run `sudo usermod -aG docker $USER` and logout |

### Getting Help

1. Check logs: `journalctl -xe`
2. ROS 2 issues: [ROS Answers](https://answers.ros.org)
3. NVIDIA issues: [NVIDIA Forums](https://forums.developer.nvidia.com)
4. General: Stack Overflow, GitHub Issues

---

## Summary

### Minimum Software Stack

- Ubuntu 22.04 LTS
- ROS 2 Humble
- Gazebo Harmonic
- Python 3.10 + pip
- Git

### Full Software Stack

- All minimum requirements
- NVIDIA driver 525+
- CUDA 12.x
- cuDNN 8.x
- Isaac Sim 2023.1+
- Docker + NVIDIA Container Toolkit
- VS Code with extensions
- PyTorch with CUDA

### Next Steps

With your software environment configured:

1. Complete the [verification checklist](#verification-checklist)
2. Start with [Module 1: ROS 2 Fundamentals](/docs/module-1)
3. Return here if you encounter setup issues

---

*Software versions current as of December 2024. Always check official documentation for latest installation instructions.*
