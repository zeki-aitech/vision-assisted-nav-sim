import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.action import ActionClient
import math

from geometry_msgs.msg import Twist, PointStamped
from vision_msgs.msg import Detection3DArray
from visualization_msgs.msg import Marker, MarkerArray
from nav_msgs.msg import Path

from nav2_msgs.action import NavigateToPose
from action_msgs.srv import CancelGoal

from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
import tf2_geometry_msgs 
from builtin_interfaces.msg import Time


class SafetyState:
    CLEAR = "CLEAR"
    WARN = "WARN"
    STOP = "STOP"
    BLIND = "BLIND"


class VisionSafetyClampNode(Node):
    def __init__(self):
        super().__init__('vision_safety_clamp_node')

        # --- Parameters ---
        # Distance (meters) to start slowing down
        self.declare_parameter('warning_distance', 2.5)
        
        # Distance (meters) to trigger E-STOP / Cancel Goal
        self.declare_parameter('stop_distance', 1.5)
        
        # Width (meters) of the safe corridor (left/right)
        self.declare_parameter('robot_corridor_width', 0.8)
        
        # Visual debugging in RViz
        self.declare_parameter('enable_debug', True)
        
        # Max time (seconds) allowed without vision data before stopping
        self.declare_parameter('timeout_sec', 0.5)
        
        # Robot base frame
        self.declare_parameter('target_frame', 'base_footprint') 
        
        # Time (seconds) allowed to rotate in place when a NEW goal is received
        self.declare_parameter('escape_grace_period', 3.0)
        
        # Object detection classes to react to (e.g., ['0'] for person)
        self.declare_parameter('target_classes', ['0'])
        
        # Tolerance (in meters) to distinguish a "New User Goal" from a "Planner Update"
        self.declare_parameter('new_goal_tolerance', 0.1)
        
        # --- Load Parameters ---
        self.warn_dist = self.get_parameter('warning_distance').value
        self.stop_dist = self.get_parameter('stop_distance').value
        self.corridor_width = self.get_parameter('robot_corridor_width').value
        self.enable_debug = self.get_parameter('enable_debug').value
        self.timeout_sec = self.get_parameter('timeout_sec').value
        self.target_frame = self.get_parameter('target_frame').value
        self.target_classes = self.get_parameter('target_classes').value
        self.new_goal_tolerance = self.get_parameter('new_goal_tolerance').value

        # --- State Variables ---
        self.current_min_x = float('inf') 
        self.last_det_time = self.get_clock().now()
        
        self.current_scale_factor = 1.0
        self.safety_state = SafetyState.CLEAR
        self.color_rgb = (0.0, 1.0, 0.0)

        # --- Goal Management State ---
        self.is_paused_by_vision = False
        self.saved_goal_pose = None
        self.saved_goal_frame = 'map'
        
        # Lock to prevent flooding the server with Resume requests
        self.is_resuming = False 
        
        # --- Grace Period State ---
        self.escape_grace_period_sec = self.get_parameter('escape_grace_period').value
        self.escape_deadline = None
        
        # --- TF2 Setup ---
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # --- Control Loop Timer (10 Hz) ---
        self.timer = self.create_timer(0.1, self.timer_state_update)
        
        # --- Subscribers ---
        self.sub_cmd_vel = self.create_subscription(
            Twist, '/cmd_vel', self.nav_cmd_callback, 10)
            
        self.sub_detections = self.create_subscription(
            Detection3DArray, '/detections_3d', self.detections_callback, 10)
        
        self.plan_sub = self.create_subscription(
            Path, '/plan', self.plan_callback, 10)
        
        # --- Publishers ---
        self.pub_cmd_vel = self.create_publisher(Twist, '/cmd_vel_vision_safe', 10)
        
        if self.enable_debug:
            self.pub_markers = self.create_publisher(MarkerArray, '~/safety_markers', 10)

        # --- Nav2 Action Clients ---
        self.nav_to_pose_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        
        # Service client for reliable "Cancel All" functionality
        self.cancel_nav_client = self.create_client(CancelGoal, '/navigate_to_pose/_action/cancel_goal')

        self.get_logger().info("Vision Safety Clamp Node Initialized!")
        self.get_logger().info(f"Frame: {self.target_frame} | Corridor: {self.corridor_width}m")

    def plan_callback(self, msg: Path):
        """ 
        Intercept the Nav2 global path. 
        Only reset the grace period if the GOAL actually changes (User clicked new goal).
        Ignores periodic replanning updates.
        """
        if not msg.poses:
            return
        
        # Get the final pose (the destination) from the new plan
        new_goal_pose = msg.poses[-1].pose
        
        is_new_goal = False
        
        if self.saved_goal_pose is None:
            is_new_goal = True
        else:
            # Calculate distance between old goal and new goal
            dx = new_goal_pose.position.x - self.saved_goal_pose.position.x
            dy = new_goal_pose.position.y - self.saved_goal_pose.position.y
            dist = math.sqrt(dx**2 + dy**2)
            
            # If the destination changed by more than new_goal_tolerance, treat it as a new user command
            if dist > self.new_goal_tolerance: 
                is_new_goal = True

        # Update the stored goal
        self.saved_goal_pose = new_goal_pose
        self.saved_goal_frame = msg.header.frame_id
        
        if is_new_goal:
            self.get_logger().info(f"New Destination! Granting {self.escape_grace_period_sec}s grace period.")
            self.is_paused_by_vision = False 
            self.is_resuming = False # Reset resume lock
            
            # Set deadline to NOW + Grace Period
            self.escape_deadline = self.get_clock().now() + Duration(seconds=self.escape_grace_period_sec)

    def pause_navigation(self):
        """ Cancels the current goal on the Nav2 server. """
        self.is_resuming = False # Reset the resume lock
        
        if not self.saved_goal_pose:
            return

        self.get_logger().warn("Obstacle in STOP zone! Canceling current Nav2 goal...")
        self.is_paused_by_vision = True
        
        if self.cancel_nav_client.wait_for_service(timeout_sec=1.0):
            cancel_msg = CancelGoal.Request()
            # [CRITICAL] Zero out UUID/Stamp to target ALL active goals
            cancel_msg.goal_info.goal_id.uuid = [0] * 16  
            cancel_msg.goal_info.stamp.sec = 0
            cancel_msg.goal_info.stamp.nanosec = 0
            
            self.cancel_nav_client.call_async(cancel_msg)
        else:
            self.get_logger().error("Cancel service not available!")

    def resume_navigation(self):
        """ Resends the saved goal to Nav2. """
        # Prevent 10Hz flooding: Check if we are already resuming
        if not self.saved_goal_pose or self.is_resuming:
            return

        self.get_logger().info("Obstacle cleared! Resuming Nav2 goal...")
        
        # Lock logic immediately
        self.is_resuming = True 
        
        if not self.nav_to_pose_client.wait_for_server(timeout_sec=1.0):
            self.get_logger().error("NavigateToPose action server not available!")
            self.is_resuming = False # Unlock on failure
            return

        goal_msg = NavigateToPose.Goal()
        goal_msg.pose.header.frame_id = self.saved_goal_frame
        goal_msg.pose.header.stamp = self.get_clock().now().to_msg()
        goal_msg.pose.pose = self.saved_goal_pose

        self.is_paused_by_vision = False
        
        # Send goal and use callback to unlock
        send_goal_future = self.nav_to_pose_client.send_goal_async(goal_msg)
        send_goal_future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        """ Unlock the resume logic once Nav2 responds. """
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error("Resume Goal rejected by Nav2!")
            self.is_resuming = False 
            return

        self.get_logger().info("Resume Goal accepted by Nav2.")
        self.is_resuming = False

    def detections_callback(self, msg: Detection3DArray):
        """ Transform obstacles to base_footprint and find closest in corridor. """
        self.last_det_time = self.get_clock().now()
        
        if not msg.detections:
            self.current_min_x = float('inf')
            return

        camera_frame_id = msg.header.frame_id

        try:
            # Look up transform from Camera -> Robot Base
            transform = self.tf_buffer.lookup_transform(
                self.target_frame, 
                camera_frame_id, 
                rclpy.time.Time()
            )
        except Exception as e:
            self.get_logger().warn(f"TF Error: {e}", throttle_duration_sec=2.0)
            return 

        min_forward_dist = float('inf')

        for det in msg.detections:
            # Filter by class ID (if specified)
            if self.target_classes and len(det.results) > 0:
                det_class_id = det.results[0].hypothesis.class_id
                if det_class_id not in self.target_classes:
                    continue 

            p_cam = PointStamped()
            p_cam.header = msg.header
            p_cam.point = det.bbox.center.position

            # Transform point
            p_base = tf2_geometry_msgs.do_transform_point(p_cam, transform)

            forward_dist = p_base.point.x
            lateral_offset = p_base.point.y

            # Check if inside safety corridor
            if abs(lateral_offset) <= (self.corridor_width / 2.0):
                if 0.0 < forward_dist < min_forward_dist:
                    min_forward_dist = forward_dist

        self.current_min_x = min_forward_dist

    def timer_state_update(self):
        """ Main Logic Loop: Checks timers, distances, and triggers Pause/Resume. """
        now = self.get_clock().now()
        
        # Calculate time since last vision update
        time_since_last_det = (now - self.last_det_time).nanoseconds / 1e9

        # --- Check Grace Period ---
        in_grace_period = False
        if self.escape_deadline is not None:
            if now < self.escape_deadline:
                in_grace_period = True
            else:
                self.escape_deadline = None # Timer Expired

        # 1. Watchdog Check (Camera Failure)
        if time_since_last_det > self.timeout_sec:
            if not self.is_paused_by_vision and self.saved_goal_pose is not None:
                self.pause_navigation()
                
            self.current_scale_factor = 0.0
            self.safety_state = SafetyState.BLIND
            self.color_rgb = (1.0, 0.0, 0.0) # RED
            self.get_logger().error("Watchdog Timeout! Stopping robot.", throttle_duration_sec=2.0)
            
        else:
            # 2. Zone Evaluation
            if self.current_min_x <= self.stop_dist:
                # STOP Logic: Pause ONLY if NOT in grace period
                if not self.is_paused_by_vision and self.saved_goal_pose is not None and not in_grace_period:
                    self.pause_navigation()
                
                # If in grace period, we don't pause, but we might still want to slow down visually/safely
                # For now, we enforce stop visually
                self.current_scale_factor = 0.0
                self.safety_state = SafetyState.STOP
                self.color_rgb = (1.0, 0.0, 0.0) # RED
                self.get_logger().warn(f"Stop zone: {self.current_min_x:.2f}m", throttle_duration_sec=1.0)
                
            elif self.current_min_x <= self.warn_dist:
                # WARN Logic: Resume if path clears
                if self.is_paused_by_vision and not self.is_resuming:
                    self.resume_navigation()

                # Linear scaling: 0% at stop_dist -> 100% at warn_dist
                scale = (self.current_min_x - self.stop_dist) / (self.warn_dist - self.stop_dist)
                self.current_scale_factor = max(0.1, min(scale, 1.0))
                self.safety_state = SafetyState.WARN
                self.color_rgb = (1.0, 1.0, 0.0) # YELLOW
                
            else:
                # CLEAR Logic: Resume full speed
                if self.is_paused_by_vision and not self.is_resuming:
                    self.resume_navigation()

                self.current_scale_factor = 1.0
                self.safety_state = SafetyState.CLEAR
                self.color_rgb = (0.0, 1.0, 0.0) # GREEN

        # 3. Publish Markers to RViz
        if self.enable_debug:
            self.publish_debug_markers(now)

    def nav_cmd_callback(self, nav_cmd: Twist):
        """ Modifies incoming velocity based on safety state. """
        safe_cmd = Twist()
        
        # Scale forward linear velocity
        if nav_cmd.linear.x > 0:
            safe_cmd.linear.x = nav_cmd.linear.x * self.current_scale_factor
        else:
            # Allow full backward speed (escape)
            safe_cmd.linear.x = nav_cmd.linear.x 

        safe_cmd.linear.y = nav_cmd.linear.y * self.current_scale_factor
        safe_cmd.linear.z = nav_cmd.linear.z
        
        # Allow rotation! (Critical for spinning in place during grace period)
        safe_cmd.angular.x = nav_cmd.angular.x
        safe_cmd.angular.y = nav_cmd.angular.y
        # Scale angular velocity slightly to avoid jerky turns when close to obstacles
        safe_cmd.angular.z = nav_cmd.angular.z * (0.5 + 0.5 * self.current_scale_factor)

        self.pub_cmd_vel.publish(safe_cmd)

    def publish_debug_markers(self, timestamp):
        """ Draws the safety corridor and status text in RViz. """
        marker_array = MarkerArray()
        lifetime = Duration(seconds=0.0).to_msg()
        zero_time = Time()

        # --- Marker 1: Corridor Floor ---
        corridor = Marker()
        corridor.header.frame_id = self.target_frame
        corridor.header.stamp = zero_time
        corridor.ns = "safety_corridor"
        corridor.id = 0
        corridor.type = Marker.CUBE
        corridor.action = Marker.ADD

        corridor.pose.position.x = self.warn_dist / 2.0  
        corridor.pose.position.y = 0.0                   
        corridor.pose.position.z = 0.05                  
        corridor.pose.orientation.w = 1.0
        
        corridor.scale.x = self.warn_dist
        corridor.scale.y = self.corridor_width
        corridor.scale.z = 0.01

        corridor.color.r = self.color_rgb[0]
        corridor.color.g = self.color_rgb[1]
        corridor.color.b = self.color_rgb[2]
        corridor.color.a = 0.3 
        corridor.lifetime = lifetime
        marker_array.markers.append(corridor)

        # --- Marker 2: HUD Text ---
        hud = Marker()
        hud.header.frame_id = self.target_frame
        hud.header.stamp = zero_time
        hud.ns = "safety_hud"
        hud.id = 1
        hud.type = Marker.TEXT_VIEW_FACING
        hud.action = Marker.ADD

        hud.pose.position.x = 0.0
        hud.pose.position.y = 0.0 
        hud.pose.position.z = 1.2  
        hud.pose.orientation.w = 1.0
        
        hud.scale.z = 0.2 
        
        hud.color.r = self.color_rgb[0]
        hud.color.g = self.color_rgb[1]
        hud.color.b = self.color_rgb[2]
        hud.color.a = 1.0
        
        speed_percent = int(self.current_scale_factor * 100)
        state_text = self.safety_state
        
        if self.safety_state in [SafetyState.WARN, SafetyState.STOP]:
            state_text += f"-({self.current_min_x:.2f}m)"
            
        hud.text = f"[{state_text}|SPEED:{speed_percent}%]"
        
        hud.lifetime = lifetime
        marker_array.markers.append(hud)

        self.pub_markers.publish(marker_array)


def main(args=None):
    rclpy.init(args=args)
    node = VisionSafetyClampNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()