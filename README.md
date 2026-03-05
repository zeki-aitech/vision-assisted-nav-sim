# Vision-Assisted Navigation Simulation

ROS 2 simulation for a mobile robot (tbot) navigating with vision-based safety: YOLO object detection and a behavior node that clamps velocity and interacts with Nav2 using 3D detections.

## Overview

- **Gazebo**: a house world with actors.
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

## Development environment (Docker / Dev Container)

The recommended way to develop and run the simulation is with the provided Dev Container (VS Code: “Reopen in Container”). It uses:

- **Image**: `docker/Dockerfile` — ROS 2 Humble with CUDA, plus GUI libraries for Gazebo/RViz.
- **Runtime**: host network, GPU access, X11 forwarding for display, 16 GB shared memory.

If you prefer to build the image yourself: from the repo root,  
`docker build -f docker/Dockerfile --build-arg USERNAME=$(whoami) docker/`

## Quick start

1. Install ROS dependencies (from workspace root):  
   `rosdep install -y --from-paths src --ignore-src`
2. Install Python requirements for YOLO:  
   `pip install -r src/yolo_inference_ros/requirements.txt`
3. Get the YOLO model:  
   `wget https://huggingface.co/Ultralytics/YOLOv8/resolve/main/yolov8n.pt -O yolov8n.pt`  
   Export to TensorRT for faster inference:  
   `yolo export model=yolov8n.pt format=engine device=0`  
   (Add `half=True` for FP16. Use the resulting `yolov8n.engine` path as `model`.)
4. Build the workspace:  
   `colcon build --symlink-install`
5. Source the workspace:  
   `source install/setup.bash`
6. Start the simulation:  
   `ros2 launch demo_launcher start_sim.launch.py world:=resources/gazebo_worlds/turtlebot3_house_actors_dynamic.world`
7. In another terminal, bring up robot, Nav2, YOLO, and vision safety:  
   `ros2 launch demo_launcher bringup_tbot.launch.py use_sim_time:=true map:=/workspaces/vision-assisted-nav-sim/resources/maps/map_house.yaml model:=/path/to/yolov8n.pt`  
   (Use the path to your `.pt` or `.engine` file; omit `model:=...` to use the launch default.)  
