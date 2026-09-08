# Weed Rover - ROS System

This branch contains the ROS 2 workspace (`laser_weeder_ws`) for the Weed Rover project. It includes the simulation environment, perception, control, and robot description packages for Gazebo and RViz.

## Workspace Structure
- `weeder_description`: URDF models and Xacro files for the robot.
- `weeder_gazebo`: Gazebo simulation worlds and environments (including the mock farm).
- `weeder_perception`: ROS 2 nodes for computer vision.
- `weeder_control`: ROS 2 nodes for logic and motor control.
- `weeder_teleop`: Teleoperation packages for manual control.
- `weeder_bringup`: Launch files to start the entire system.

## Setup Instructions

1. **Prerequisites**: Ensure you have ROS 2 (Humble/Iron) and Gazebo installed.
2. **Build**:
   ```bash
   cd laser_weeder_ws
   colcon build
   ```
3. **Source**:
   ```bash
   source install/setup.bash
   ```

## Running the Simulation

To launch the full simulation in Gazebo and RViz:
```bash
ros2 launch weeder_bringup simulation.launch.py
```
