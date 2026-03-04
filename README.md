# Vision-Assisted Navigation Simulation

ROS 2 simulation for a TurtleBot navigating with vision-based safety: YOLO object detection and a behavior node that clamps velocity and interacts with Nav2 using 3D detections.

## Overview

- **Gazebo**: TurtleBot3 in a house world with actors.
- **Nav2**: Standard navigation stack with a provided map.
- **YOLO inference** (`yolo_inference_ros`): 2D/3D detections (vision_msgs), optional ByteTrack tracking, depth-based 3D boxes.
- **Vision safety** (`tbot_nav_behavior`): Clamps `cmd_vel` and cancels Nav2 goals using warning/stop distances and corridor logic.

## Main packages

| Package             | Role                                      |
|---------------------|-------------------------------------------|
| `demo_launcher`     | Top-level launch files                    |
| `tbot_gazebo`       | Gazebo world, spawn, robot state publisher |
| `tbot_description` | Robot description (URDF, xacro, RViz)    |
| `tbot_nav`          | Nav2 launch and config                    |
| `yolo_inference_ros`| YOLO detection + tracking node            |
| `tbot_nav_behavior` | Vision safety clamp node                  |

## Quick start

1. Build the workspace: `colcon build --symlink-install`
2. Source: `source install/setup.bash`
3. Start simulation: `ros2 launch demo_launcher start_sim.launch.py world:=resources/gazebo_worlds/turtlebot3_house_actors_dynamic.world`
4. In another terminal, bring up robot + Nav2 + YOLO + vision safety:  
   `ros2 launch demo_launcher bringup_tbot.launch.py use_sim_time:=true`
