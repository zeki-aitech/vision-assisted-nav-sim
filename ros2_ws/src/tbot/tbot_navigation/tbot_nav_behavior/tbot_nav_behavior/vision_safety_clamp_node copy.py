import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.action import ActionClient

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
        
        # Distance (in meters) directly in front of the robot where it will start 
        # to proportionally slow down. Speed scales linearly from 100% at this 
        # distance down to 0% at the stop_distance.
        self.declare_parameter('warning_distance', 2.5)
        
        # Critical distance (in meters) where the forward velocity (linear.x) 
        # is strictly clamped to 0.0. If the obstacle remains here, it triggers 
        # the Nav2 Action Server pause.
        self.declare_parameter('stop_distance', 1.5)
        
        # The lateral width (in meters) of your safety zone. Obstacles outside 
        # this left/right boundary are completely ignored. For a grass-cutting AMR, 
        # this should be set slightly wider than your physical cutting deck or 
        # wheelbase so you don't react to obstacles you will safely pass beside.
        self.declare_parameter('robot_corridor_width', 0.8)
        
        # Toggles the publishing of the visual MarkerArray to RViz2, which draws 
        # the colored corridor on the ground and the floating status HUD.
        self.declare_parameter('enable_debug', True)
        
        # Watchdog timer limit. If the computer vision pipeline stops publishing 
        # to '/detections_3d' for this many seconds (e.g., if the camera disconnects 
        # or the inference node crashes), the robot enters a BLIND state and halts.
        self.declare_parameter('timeout_sec', 0.5)
        
        # The coordinate frame used to calculate distance. This is usually the center 
        # point of the robot on the ground. It is critical that in this frame, 
        # +X is strictly forward and +Y is strictly lateral (left/right).
        self.declare_parameter('target_frame', 'base_footprint') 
        
        # The "Escape Grace Period" (in seconds). When a NEW navigation goal is 
        # received while the robot is currently blocked, Nav2 is temporarily 
        # unpaused for this duration. During this window, forward linear velocity 
        # remains safely clamped to 0.0 (due to the stop zone scaling factor), 
        # but angular velocity is still permitted. This elegantly allows the 
        # controller to rotate the robot in place to face the new goal, clearing 
        # the obstacle from its safety corridor before we force-pause the system.
        self.declare_parameter('escape_grace_period', 3.0)
        
        # A list of specific class IDs from your object detection model that should 
        # trigger a stop. For example, '0' might be 'person'. If you want the robot 
        # to stop for humans and vehicles, but push through tall grass or ignore 
        # specific other obstacles, you define the danger classes here. 
        # Leaving it as an empty list [] means ALL 3D detections will stop the robot.
        self.declare_parameter('target_classes', ['0'])
        
        #--------------------------------
        
        self.warn_dist = self.get_parameter('warning_distance').value
        self.stop_dist = self.get_parameter('stop_distance').value
        self.corridor_width = self.get_parameter('robot_corridor_width').value
        self.enable_debug = self.get_parameter('enable_debug').value
        self.timeout_sec = self.get_parameter('timeout_sec').value
        self.target_frame = self.get_parameter('target_frame').value
        self.target_classes = self.get_parameter('target_classes').value

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
        
        # --- Grace Period State ---
        self.escape_grace_period = self.get_parameter('escape_grace_period').value
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
        self.cancel_nav_client = self.create_client(CancelGoal, '/navigate_to_pose/_action/cancel_goal')

        self.get_logger().info("Vision Safety Clamp Node Initialized!")
        self.get_logger().info(f"Frame: {self.target_frame} | Corridor: {self.corridor_width}m | Warn: {self.warn_dist}m | Stop: {self.stop_dist}m")

    def plan_callback(self, msg: Path):
        """ Intercept the Nav2 global path to store the true destination. """
        if not msg.poses:
            return
        
        self.saved_goal_pose = msg.poses[-1].pose
        self.saved_goal_frame = msg.header.frame_id
        
        self.get_logger().info(
            f"Saved Goal -> X: {self.saved_goal_pose.position.x:.2f}, "
            f"Y: {self.saved_goal_pose.position.y:.2f}"
        )

        # When a new plan arrives, give the robot a window to turn away from the obstacle
        self.get_logger().warn(f"New plan received! Giving robot {self.escape_grace_period}s to escape.")
        self.is_paused_by_vision = False 
        self.escape_deadline = self.get_clock().now() + Duration(seconds=self.escape_grace_period)

    def pause_navigation(self):
        if not self.saved_goal_pose:
            return

        self.get_logger().warn("Obstacle in STOP zone! Canceling current Nav2 goal...")
        self.is_paused_by_vision = True
        
        if self.cancel_nav_client.wait_for_service(timeout_sec=1.0):
            # Sending an empty Request cancels the current active goal
            cancel_msg = CancelGoal.Request()
            self.cancel_nav_client.call_async(cancel_msg)
        else:
            self.get_logger().error("Cancel service not available!")

    def resume_navigation(self):
        if not self.saved_goal_pose:
            return

        self.get_logger().info("Obstacle cleared! Resuming Nav2 goal...")
        self.is_paused_by_vision = False
        
        if not self.nav_to_pose_client.wait_for_server(timeout_sec=1.0):
            self.get_logger().error("NavigateToPose action server not available!")
            return

        goal_msg = NavigateToPose.Goal()
        goal_msg.pose.header.frame_id = self.saved_goal_frame
        goal_msg.pose.header.stamp = self.get_clock().now().to_msg()
        goal_msg.pose.pose = self.saved_goal_pose

        self.nav_to_pose_client.send_goal_async(goal_msg)

    def detections_callback(self, msg: Detection3DArray):
        """ Transform obstacles to base_footprint, filter corridor, and find closest. """
        self.last_det_time = self.get_clock().now()
        
        if not msg.detections:
            self.current_min_x = float('inf')
            return

        camera_frame_id = msg.header.frame_id

        try:
            transform = self.tf_buffer.lookup_transform(
                self.target_frame, 
                camera_frame_id, 
                rclpy.time.Time()
            )
        except Exception as e:
            self.get_logger().warn(f"TF Error: Could not transform {camera_frame_id} to {self.target_frame}: {e}")
            return 

        min_forward_dist = float('inf')

        for det in msg.detections:
            if self.target_classes and len(det.results) > 0:
                det_class_id = det.results[0].hypothesis.class_id
                if det_class_id not in self.target_classes:
                    continue 

            p_cam = PointStamped()
            p_cam.header = msg.header
            p_cam.point = det.bbox.center.position

            p_base = tf2_geometry_msgs.do_transform_point(p_cam, transform)

            forward_dist = p_base.point.x
            lateral_offset = p_base.point.y

            if abs(lateral_offset) <= (self.corridor_width / 2.0):
                if 0.0 < forward_dist < min_forward_dist:
                    min_forward_dist = forward_dist

        self.current_min_x = min_forward_dist

    def timer_state_update(self):
        now = self.get_clock().now()
        time_since_last_det = (now - self.last_det_time).nanoseconds / 1e9

        # --- Evaluate Grace Period ---
        in_grace_period = False
        if self.escape_deadline is not None:
            if now < self.escape_deadline:
                in_grace_period = True
            else:
                self.get_logger().warn("Escape grace period expired! Target is still blocked.")
                self.escape_deadline = None # Reset

        # 1. Watchdog Check
        if time_since_last_det > self.timeout_sec:
            if not self.is_paused_by_vision and self.saved_goal_pose is not None:
                self.pause_navigation()
                
            self.current_scale_factor = 0.0
            self.safety_state = SafetyState.BLIND
            self.color_rgb = (1.0, 0.0, 0.0)
            self.get_logger().error("Watchdog Timeout! Stopping robot.", throttle_duration_sec=2.0)
            
        else:
            # 2. Zone Evaluation
            if self.current_min_x <= self.stop_dist:
                # PAUSE LOGIC: Only pause if we are NOT in the escape grace period
                if not self.is_paused_by_vision and self.saved_goal_pose is not None and not in_grace_period:
                    self.pause_navigation()

                self.current_scale_factor = 0.0
                self.safety_state = SafetyState.STOP
                self.color_rgb = (1.0, 0.0, 0.0)
                self.get_logger().warn(f"Stop zone breached: {self.current_min_x:.2f}m", throttle_duration_sec=1.0)
                
            elif self.current_min_x <= self.warn_dist:
                # RESUME LOGIC 
                if self.is_paused_by_vision:
                    self.resume_navigation()

                scale = (self.current_min_x - self.stop_dist) / (self.warn_dist - self.stop_dist)
                self.current_scale_factor = max(0.1, min(scale, 1.0))
                self.safety_state = SafetyState.WARN
                self.color_rgb = (1.0, 1.0, 0.0)
                
            else:
                # RESUME LOGIC 
                if self.is_paused_by_vision:
                    self.resume_navigation()

                self.current_scale_factor = 1.0
                self.safety_state = SafetyState.CLEAR
                self.color_rgb = (0.0, 1.0, 0.0)

        # 3. Always Publish Debug Markers
        if self.enable_debug:
            self.publish_debug_markers(now)

    def nav_cmd_callback(self, nav_cmd: Twist):
        """ Pass-through callback: Multiplies incoming vel by current scale factor. """
        safe_cmd = Twist()
        
        if nav_cmd.linear.x > 0:
            safe_cmd.linear.x = nav_cmd.linear.x * self.current_scale_factor
        else:
            safe_cmd.linear.x = nav_cmd.linear.x 

        safe_cmd.linear.y = nav_cmd.linear.y * self.current_scale_factor
        safe_cmd.linear.z = nav_cmd.linear.z
        
        safe_cmd.angular.x = nav_cmd.angular.x
        safe_cmd.angular.y = nav_cmd.angular.y
        safe_cmd.angular.z = nav_cmd.angular.z * (0.5 + 0.5 * self.current_scale_factor)

        self.pub_cmd_vel.publish(safe_cmd)

    def publish_debug_markers(self, timestamp):
        """ Draw the Safety Corridor and HUD on RViz2 in target_frame (base_footprint) """
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
        if self.safety_state == SafetyState.CLEAR:
            hud.text = f"[{self.safety_state}|SPEED:{speed_percent}%]"
        elif self.safety_state in [SafetyState.WARN, SafetyState.STOP]:
            hud.text = f"[{self.safety_state}-({self.current_min_x:.2f}m)|SPEED:{speed_percent}%]"
        elif self.safety_state == SafetyState.BLIND:
            hud.text = f"[{self.safety_state}]"
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