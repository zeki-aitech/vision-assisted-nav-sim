import rclpy
from rclpy.node import Node
from rclpy.duration import Duration

from geometry_msgs.msg import Twist, PointStamped
from vision_msgs.msg import Detection3DArray
from visualization_msgs.msg import Marker, MarkerArray

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
        self.declare_parameter('warning_distance', 2.5)
        self.declare_parameter('stop_distance', 1.0)
        self.declare_parameter('robot_corridor_width', 0.8)
        self.declare_parameter('enable_debug', True)
        self.declare_parameter('timeout_sec', 0.5)
        self.declare_parameter('target_frame', 'base_footprint') # The ground-truth frame for the robot
        
        self.warn_dist = self.get_parameter('warning_distance').value
        self.stop_dist = self.get_parameter('stop_distance').value
        self.corridor_width = self.get_parameter('robot_corridor_width').value
        self.enable_debug = self.get_parameter('enable_debug').value
        self.timeout_sec = self.get_parameter('timeout_sec').value
        self.target_frame = self.get_parameter('target_frame').value

        # --- State Variables ---
        self.current_min_x = float('inf') # We use X now because base_footprint X is forward
        self.last_det_time = self.get_clock().now()
        
        self.current_scale_factor = 1.0
        self.safety_state = SafetyState.CLEAR
        self.color_rgb = (0.0, 1.0, 0.0)

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

        # --- Publishers ---
        self.pub_cmd_vel = self.create_publisher(Twist, '/cmd_vel_vision_safe', 10)
        
        if self.enable_debug:
            self.pub_markers = self.create_publisher(MarkerArray, '~/safety_markers', 10)

        self.get_logger().info("Vision Safety Clamp Node Initialized!")
        self.get_logger().info(f"Frame: {self.target_frame} | Corridor: {self.corridor_width}m | Warn: {self.warn_dist}m | Stop: {self.stop_dist}m")

    def detections_callback(self, msg: Detection3DArray):
        """ Transform obstacles to base_footprint, filter corridor, and find closest. """
        self.last_det_time = self.get_clock().now()
        
        # If there are no detections, simply return (distance stays at infinity)
        if not msg.detections:
            self.current_min_x = float('inf')
            return

        camera_frame_id = msg.header.frame_id

        # 1. Look up the transform from the Camera to the Base Footprint
        try:
            # We use rclpy.time.Time() to get the latest available transform
            transform = self.tf_buffer.lookup_transform(
                self.target_frame, 
                camera_frame_id, 
                rclpy.time.Time()
            )
        except Exception as e:
            self.get_logger().warn(f"TF Error: Could not transform {camera_frame_id} to {self.target_frame}: {e}")
            return # Skip this frame if we don't know where the camera is

        min_forward_dist = float('inf')

        for det in msg.detections:
            # 2. Package the raw optical point into a PointStamped
            p_cam = PointStamped()
            p_cam.header = msg.header
            p_cam.point = det.bbox.center.position

            # 3. Transform the point to base_footprint
            p_base = tf2_geometry_msgs.do_transform_point(p_cam, transform)

            # In base_footprint: X is strictly forward distance, Y is strict left/right offset
            forward_dist = p_base.point.x
            lateral_offset = p_base.point.y

            # 4. THE CORRIDOR FILTER (Using true physical Y offset)
            if abs(lateral_offset) <= (self.corridor_width / 2.0):
                # Ensure it's actually in front of the robot (X > 0)
                if 0.0 < forward_dist < min_forward_dist:
                    min_forward_dist = forward_dist

        self.current_min_x = min_forward_dist

    def timer_state_update(self):
        """ 10Hz Loop: Evaluates safety state, sets scale factor, and ALWAYS publishes markers. """
        now = self.get_clock().now()
        time_since_last_det = (now - self.last_det_time).nanoseconds / 1e9

        # 1. Watchdog Check
        if time_since_last_det > self.timeout_sec:
            self.current_scale_factor = 0.0
            self.safety_state = SafetyState.BLIND
            self.color_rgb = (1.0, 0.0, 0.0)
            self.get_logger().error("Watchdog Timeout! Stopping robot.", throttle_duration_sec=2.0)
            
        else:
            # 2. Zone Evaluation using X (forward distance in base_footprint)
            if self.current_min_x <= self.stop_dist:
                self.current_scale_factor = 0.0
                self.safety_state = SafetyState.STOP
                self.color_rgb = (1.0, 0.0, 0.0)
                self.get_logger().warn(f"Stop zone breached: {self.current_min_x:.2f}m", throttle_duration_sec=1.0)
                
            elif self.current_min_x <= self.warn_dist:
                scale = (self.current_min_x - self.stop_dist) / (self.warn_dist - self.stop_dist)
                self.current_scale_factor = max(0.1, min(scale, 1.0))
                self.safety_state = SafetyState.WARN
                self.color_rgb = (1.0, 1.0, 0.0)
                
            else:
                self.current_scale_factor = 1.0
                self.safety_state = SafetyState.CLEAR
                self.color_rgb = (0.0, 1.0, 0.0)

        # 3. Always Publish Debug Markers
        if self.enable_debug:
            self.publish_debug_markers(now)

    def nav_cmd_callback(self, nav_cmd: Twist):
        """ Pass-through callback: Simply multiplies incoming vel by current scale factor. """
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
        # Lifetime is slightly longer than timer period (0.1s) to prevent flickering
        # lifetime = Duration(seconds=0.2).to_msg() 
        lifetime = Duration(seconds=0.0).to_msg()
        zero_time = Time()

        # --- Marker 1: Corridor Floor ---
        corridor = Marker()
        corridor.header.frame_id = self.target_frame
        # corridor.header.stamp = timestamp.to_msg()
        corridor.header.stamp = zero_time
        corridor.ns = "safety_corridor"
        corridor.id = 0
        corridor.type = Marker.CUBE
        corridor.action = Marker.ADD

        # In base_footprint: X is forward, Y is left/right, Z is up
        corridor.pose.position.x = self.warn_dist / 2.0  # Center of the corridor is halfway forward
        corridor.pose.position.y = 0.0                   # Centered exactly on the robot
        corridor.pose.position.z = 0.05                  # Float 5cm above the floor to avoid z-fighting
        corridor.pose.orientation.w = 1.0
        
        # Scale: X = length (forward), Y = width (lateral), Z = thickness
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
        # hud.header.stamp = timestamp.to_msg()
        hud.header.stamp = zero_time
        hud.ns = "safety_hud"
        hud.id = 1
        hud.type = Marker.TEXT_VIEW_FACING
        hud.action = Marker.ADD

        # Hover directly above the robot
        hud.pose.position.x = 0.0
        hud.pose.position.y = 0.0 
        hud.pose.position.z = 1.2  # 1.2 meters up in the air 
        hud.pose.orientation.w = 1.0
        
        hud.scale.z = 0.2 
        
        hud.color.r = self.color_rgb[0]
        hud.color.g = self.color_rgb[1]
        hud.color.b = self.color_rgb[2]
        hud.color.a = 1.0
        
        speed_percent = int(self.current_scale_factor * 100)
        if self.safety_state == SafetyState.CLEAR:
            hud.text = f"[{self.safety_state}|SPEED:{speed_percent}%]"
        elif self.safety_state == SafetyState.WARN:
            hud.text = f"[{self.safety_state}-({self.current_min_x:.2f}m)|SPEED:{speed_percent}%]"
        elif self.safety_state == SafetyState.STOP:
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