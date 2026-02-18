from typing import List, Dict
from cv_bridge import CvBridge

import rclpy
from rclpy.qos import QoSProfile
from rclpy.qos import QoSHistoryPolicy
from rclpy.qos import QoSDurabilityPolicy
from rclpy.qos import QoSReliabilityPolicy
from rclpy.node import Node

import message_filters
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray
from vision_msgs.msg import Detection3DArray


class YoloInferenceNode(Node):
    def __init__(self):
        super().__init__('yolo_inference_node')
        
        self.declare_parameter("target_frame", "base_link")
        
        self.declare_parameter("image_reliability", QoSReliabilityPolicy.BEST_EFFORT)
        self.declare_parameter("image_info_reliability", QoSReliabilityPolicy.BEST_EFFORT)
        
        
        self.declare_parameter("depth_reliability", QoSReliabilityPolicy.BEST_EFFORT)
        self.declare_parameter("depth_info_reliability", QoSReliabilityPolicy.BEST_EFFORT)
        
        
        #########################################################
        self.target_frame = (
            self.get_parameter("target_frame").get_parameter_value().string_value
        )

        image_reliability = (
            self.get_parameter("image_reliability")
            .get_parameter_value()
            .integer_value
        )
        image_info_reliability = (
            self.get_parameter("image_info_reliability")
            .get_parameter_value()
            .integer_value
        )
        depth_reliability = (
            self.get_parameter("depth_reliability")
            .get_parameter_value()
            .integer_value
        )
        depth_info_reliability = (
            self.get_parameter("depth_info_reliability")
            .get_parameter_value()
            .integer_value
        )
        
        self.image_qos_profile = QoSProfile(
            reliability=image_reliability,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=1,
        )
        self.image_info_qos_profile = QoSProfile(
            reliability=image_info_reliability,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=1,
        )
        self.depth_qos_profile = QoSProfile(
            reliability=depth_reliability,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=1,
        )
        self.depth_info_qos_profile = QoSProfile(
            reliability=depth_info_reliability,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=1,
        )
        
        self.rgb_sub_ = message_filters.create_subscription(
            Image, '/camera/image_raw', self.img_callback, self.image_qos_profile)
        
        self.depth_sub_ = message_filters.create_subscription(
            Image, '/camera/depth/image_raw', self.depth_callback, self.depth_qos_profile)
        
        self.get_logger().info('YoloInferenceNode initialized')
        

    def run(self):
        self.get_logger().info('YoloInferenceNode running')
        
        
def main(args=None):
    rclpy.init(args=args)
    node = YoloInferenceNode()
    node.run()
    rclpy.shutdown()
