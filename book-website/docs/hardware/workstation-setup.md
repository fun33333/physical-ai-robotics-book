---
title: "Workstation Setup Guide"
sidebar_position: 4
description: "Step-by-step guide to setting up your Physical AI development workstation with optimal configurations"
---

# Workstation Setup Guide

## Overview

A properly configured development workstation is essential for productivity in Physical AI projects. This guide will walk you through setting up a computer system optimized for ROS 2, simulation environments, and AI model development.

## Physical Setup Considerations

### Space Requirements

Before diving into software setup, consider your physical workspace needs:

#### Desktop Station Requirements
- **Space**: Minimum 30" x 24" desk surface for monitors and peripherals
- **Height**: Adjustable desk height for ergonomic setup (28-32" standard)
- **Power**: Multiple outlets near computer placement for monitor, lights, and peripherals
- **Ventilation**: Adequate airflow around computer to prevent overheating
- **Cable Management**: Organized system to prevent tangling and tripping hazards

#### Environmental Conditions
- **Temperature**: Room temperature between 68°F and 72°F (20°C-22°C)
- **Humidity**: 30-50% RH to prevent static discharge and corrosion
- **Lighting**: Proper lighting to reduce eye strain without screen glare
- **Acoustic**: Consider sound dampening if fans/cooling is loud

### Ergonomic Setup
- Monitor positioned at eye level, 20-24 inches away
- Keyboard at elbow height with wrist support
- Chair with lower back support adjusted to 90-110 degrees
- Feet flat on floor with thighs parallel to floor

## Hardware Installation

### Installing Components (Desktop)

:::warning Safety Warning
Always ground yourself by touching a metal surface or using an anti-static wrist strap before handling computer components to prevent electrostatic discharge that can damage sensitive electronics.
:::

#### Motherboard, CPU, RAM Installation
1. Ground yourself by touching a metal surface to prevent static discharge
2. Install CPU following manufacturer's alignment marks
3. Apply thermal paste (pea-sized amount) and attach cooler
4. Install RAM in slots following motherboard's dual-channel recommendations
5. Secure motherboard in case with standoffs

#### GPU Installation
1. Remove appropriate PCIe slot covers from case
2. Ensure power supply has adequate wattage (check GPU manufacturer requirements)
3. Insert GPU into PCIe x16 slot and secure with screws
4. Connect required power cables from PSU to GPU
5. Verify secure fit and proper seating

#### Storage Installation
1. Mount drives in appropriate drive bays
2. Connect SATA data cables to motherboard ports
3. Connect SATA power cables from PSU to drives
4. For NVMe drives: Install directly onto motherboard in M.2 slot

### Cable Management Tips
- Use zip ties or velcro straps to bundle cables
- Route cables behind motherboard tray when possible
- Label cables for easy identification
- Leave space for airflow

## Operating System Installation

### Ubuntu 22.04 LTS (Recommended)

Ubuntu 22.04 LTS is the officially supported OS for ROS 2 Humble Hawksbill.

#### Installation Steps
1. Download Ubuntu 22.04 LTS ISO from ubuntu.com
2. Create bootable USB using platform-specific tools:
   - **On Windows**: Use Rufus to create a bootable USB
     - Download Rufus from https://rufus.ie/
     - Select the Ubuntu ISO file
     - Use default settings and click "Start"

   - **On Linux**: Use the `dd` command to create a bootable USB
     ```bash
     sudo dd if=ubuntu-22.04-desktop-amd64.iso of=/dev/sdX bs=4M status=progress
     ```
     Replace `/dev/sdX` with your USB drive identifier.

   - **On macOS**: Use balenaEtcher to create a bootable USB
     - Download balenaEtcher from https://www.balena.io/etcher/
     - Select the Ubuntu ISO file
     - Select your USB drive
     - Click "Flash"

3. Boot from USB drive (may require changing BIOS boot order)
4. Choose "Install Ubuntu" option
5. Select language and timezone
6. During installation type, choose "Normal installation"
7. Ensure "Install third-party software" is checked (for proprietary drivers)
8. Select "Erase disk and install Ubuntu" or manual partitioning if dual-booting
9. Complete user account setup

:::tip Installation Tip
For development work, consider allocating at least 100GB of disk space to your Ubuntu installation. This will provide sufficient room for ROS 2 packages, simulation environments, and other development tools.
:::

### Dual Boot Setup (Windows + Ubuntu)
1. Shrink Windows partition using Disk Manager (minimum 50GB recommended)
2. Boot from Ubuntu installer USB
3. Choose "Install alongside Windows Boot Manager"
4. Adjust partition sizes as needed (Ubuntu minimum 40GB recommended)
5. Complete installation

### Important Post-Installation Steps
1. **System Updates**:
   ```bash
   sudo apt update && sudo apt upgrade -y
   ```

2. **Install Proprietary GPU Drivers**:
   ```bash
   # For NVIDIA cards:
   sudo apt install nvidia-driver-535  # Or latest appropriate driver
   
   # For AMD cards:
   sudo apt install mesa-vulkan-drivers xserver-xorg-video-amdgpu
   ```

3. **Reboot after driver installation**:
   ```bash
   sudo reboot
   ```

4. **Verify GPU Driver Installation**:
   ```bash
   # For NVIDIA:
   nvidia-smi
   
   # For AMD:
   rocm-smi
   ```

## Development Software Installation

### Version Control (Git)
```bash
sudo apt update
sudo apt install git
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"
```

### Essential Development Tools
```bash
sudo apt install build-essential cmake python3-dev python3-pip python3-colcon-common-extensions
```

### Python Environment Management
```bash
# Install virtual environment manager
sudo apt install python3-venv

# Create virtual environment for projects
python3 -m venv ~/robotics_env
source ~/robotics_env/bin/activate
```

## ROS 2 Installation

### Setting up ROS 2 Humble Hawksbill

Follow the official installation guide:
```bash
# Set locale
locale  # Verify LANG=en_US.UTF-8
sudo apt update && sudo apt install locales
sudo locale-gen en_US.UTF-8
export LANG=en_US.UTF-8

# Add ROS 2 apt repository
sudo apt install software-properties-common
sudo add-apt-repository universe
sudo apt update && sudo apt install curl -y
curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key | sudo gpg --dearmor -o /usr/share/keyrings/ros-archive-keyring.gpg

echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null

# Install ROS 2 packages
sudo apt update
sudo apt install ros-humble-desktop
sudo apt install ros-dev-tools
```

**Expected Output**:
```
...
The following NEW packages will be installed:
  ros-humble-desktop ros-dev-tools ...
...
```

:::note ROS 2 Installation Notes
The `ros-humble-desktop` package includes all the necessary packages for general ROS 2 development including Gazebo simulation tools. If you have storage constraints, you can install `ros-humble-ros-base` for a minimal installation, but you'll need to install additional packages as needed for your projects.
:::

### Environment Setup for ROS 2
Add to `~/.bashrc`:
```bash
# Setup ROS 2 environment
source /opt/ros/humble/setup.bash
```

Apply changes:
```bash
source ~/.bashrc
```

## Simulation Environment Setup

### Installing Gazebo (Fortress)
```bash
# Add Gazebo repository
sudo sh -c 'echo "deb http://packages.osrfoundation.org/gazebo/ubuntu-stable `lsb_release -cs` main" > /etc/apt/sources.list.d/gazebo-stable.list'
wget http://packages.osrfoundation.org/gazebo.key -O - | sudo apt-key add -
sudo apt update

# Install Gazebo Fortress
sudo apt install gazebo-fortress
```

### Installing Unity Hub and Editor (Optional)
1. Download Unity Hub from unity3d.com
2. Install Unity Hub:
   ```bash
   chmod +x UnityHub.AppImage
   ./UnityHub.AppImage
   ```
3. Sign in with Unity ID
4. Install latest LTS Unity Editor version (recommended 2022.3.x)

### NVIDIA Isaac Sim Setup (if applicable)
1. Ensure CUDA-compatible GPU with at least 8GB VRAM
2. Have NVIDIA GPU drivers installed (verified with nvidia-smi)
3. Create NVIDIA Developer account
4. Download Isaac Sim from developer.nvidia.com
5. Extract and run:
   ```bash
   cd Isaac-Sim-XXXX.X.X.X
   bash install_dependencies.sh
   python3 -m pip install -e .
   ```

## IDE and Developer Tools

### Installing VS Code
```bash
sudo snap install --classic code
```

### Useful VS Code Extensions for Robotics
- ROS (for VS Code)
- C/C++ (Microsoft)
- Python (Microsoft)
- Pylance (Python extension)
- Docker (Microsoft)
- GitLens (Git supercharged)

### Setting up Terminal
```bash
# Install and configure Oh My Zsh
sudo apt install zsh
sh -c "$(curl -fsSL https://raw.github.com/ohmyzsh/ohmyzsh/master/tools/install.sh)"

# Install zsh-autosuggestions plugin
git clone https://github.com/zsh-users/zsh-autosuggestions ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/zsh-autosuggestions

# Edit ~/.zshrc and add zsh-autosuggestions to plugins
# plugins=(... zsh-autosuggestions ...)
```

## Network Configuration

### Setting Up Static IP (Optional)
For robotics applications, setting a static IP can improve reliability:

Edit network configuration:
```bash
sudo nano /etc/netplan/01-network-manager-all.yaml
```

Example configuration:
```yaml
network:
  version: 2
  renderer: networkd
  ethernets:
    eth0:  # Replace with your interface name (find with ip addr)
      dhcp4: no
      addresses:
        - 192.168.1.100/24
      gateway4: 192.168.1.1
      nameservers:
        addresses: [8.8.8.8, 8.8.4.4]
```

Apply configuration:
```bash
sudo netplan apply
```

### Firewall Configuration for Robotics
```bash
# Allow common ROS ports (default: 11311 and ephemeral range 35800-36800)
sudo ufw enable
sudo ufw allow 11311
sudo ufw allow 35800:36800/tcp
```

## Testing Your Setup

### Basic System Test
```bash
# Check ROS 2 installation
ros2 --version

# Source ROS 2 environment
source /opt/ros/humble/setup.bash

# Test with simple talker/listener example
ros2 run demo_nodes_cpp talker &
ros2 run demo_nodes_py listener
```

### GPU Acceleration Test
```bash
# Verify OpenGL acceleration is working
glxinfo | grep -i opengl

# For NVIDIA, verify CUDA is accessible
nvidia-smi
nvcc --version
```

### Gazebo Test
```bash
# Launch Gazebo Fortress
gazebo
```

## Troubleshooting Common Issues

### Problem: "Unable to initialize graphics" in Gazebo
**Solution**: 
```bash
# Check graphics acceleration
export LIBGL_ALWAYS_SOFTWARE=1
gazebo  # Run once to check if it's graphics-related

# If it works, reinstall graphics drivers
sudo apt remove nvidia-*  # Remove old drivers
sudo apt autoremove
sudo apt install nvidia-driver-XXX  # Replace XXX with appropriate version
sudo reboot
```

### Problem: Permission denied with serial devices
**Solution**:
```bash
# Add user to dialout group
sudo usermod -a -G dialout $USER
# Log out and back in for changes to take effect
```

### Problem: Audio/video devices not working in VM
**Solution**:
- For VirtualBox: Install Guest Additions and enable audio/video passthrough
- For VMware: Install VMware Tools and verify device sharing settings

## Security Considerations

### User Accounts
- Use dedicated user accounts for development rather than root
- Enable automatic updates for security patches
- Use SSH keys instead of passwords for remote access

### Firewall Rules
- Limit exposure to only necessary ports
- Use VPN for remote access to robotics systems
- Regularly review active connections with `netstat -tuln`

## Maintenance and Optimization

### System Monitoring
Install system monitoring tools:
```bash
sudo apt install htop iotop nethogs
```

### Cleanup Commands
```bash
# Clean package cache
sudo apt autoremove && sudo apt autoclean

# Clear temporary files
rm -rf ~/.cache/*

# Clean up Docker (if using)
docker system prune
```

---

:::tip Exercise 3: Workstation Performance Testing
**Objective**: Evaluate your workstation's performance for Physical AI applications

**Time Estimate**: 60 minutes

**Steps**:
1. Benchmark your system using standard tools:
   - CPU: `sysbench cpu --threads=0 run`
   - RAM: `sysbench memory --memory-block-size=1M --memory-total-size=100G run`
   - GPU: Install and run gpu_burn for stress test
2. Test Gazebo simulation performance with a simple robot model
3. Measure ROS 2 communication latency between nodes
4. Try running a basic Isaac Sim scene (if installed)
5. Document any bottlenecks or issues discovered

**Expected Result**: A performance profile of your workstation with recommendations for improvements

**Hints**:
- Run benchmarks when the system is not under other loads
- Record baseline performance for future comparison
- Consider thermal throttling during long-running simulations
:::