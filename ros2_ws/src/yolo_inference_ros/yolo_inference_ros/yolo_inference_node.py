from typing import List, Dict
from cv_bridge import CvBridge

import rclpy
from rclpy.qos import QoSProfile
from rclpy.qos import QoSHistoryPolicy
from rclpy.qos import QoSDurabilityPolicy
from rclpy.qos import QoSReliabilityPolicy
from rclpy.node import Node

import message_filters
from cv_bridge import CvBridge
from tf2_ros.buffer import Buffer
from tf2_ros import TransformException
from tf2_ros.transform_listener import TransformListener

from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray
from vision_msgs.msg import Detection3DArray

import torch
from ultralytics import YOLO


class YoloInferenceNode(Node):
    def __init__(self):
        super().__init__('yolo_inference_node')
        
        self.declare_parameter("enable_3d", False)
        
        self.declare_parameter("model", "yolov8m.pt")
        self.declare_parameter("task", "detect")
        self.declare_parameter("device", "cuda:0")
        self.declare_parameter("fuse_model", False)
        self.declare_parameter("yolo_encoding", "bgr8")
        self.declare_parameter("enable", True)

        self.declare_parameter("threshold", 0.5)
        self.declare_parameter("iou", 0.5)
        self.declare_parameter("imgsz_height", 640)
        self.declare_parameter("imgsz_width", 640)
        self.declare_parameter("half", False)
        self.declare_parameter("max_det", 300)
        self.declare_parameter("augment", False)
        self.declare_parameter("agnostic_nms", False)
        self.declare_parameter("retina_masks", False)
        
        self.declare_parameter("image_reliability", QoSReliabilityPolicy.BEST_EFFORT)

        #########################################################
        self.enable_3d = self.get_parameter("enable_3d").get_parameter_value().bool_value
        
        # model parameters
        self.model = self.get_parameter("model").get_parameter_value().string_value
        self.task = self.get_parameter("task").get_parameter_value().string_value
        self.device = self.get_parameter("device").get_parameter_value().string_value
        self.fuse_model = (
            self.get_parameter("fuse_model").get_parameter_value().bool_value
        )
        self.yolo_encoding = (
            self.get_parameter("yolo_encoding").get_parameter_value().string_value
        )

        # Inference params
        self.threshold = (
            self.get_parameter("threshold").get_parameter_value().double_value
        )
        self.iou = self.get_parameter("iou").get_parameter_value().double_value
        self.imgsz_height = (
            self.get_parameter("imgsz_height").get_parameter_value().integer_value
        )
        self.imgsz_width = (
            self.get_parameter("imgsz_width").get_parameter_value().integer_value
        )
        self.half = self.get_parameter("half").get_parameter_value().bool_value
        self.max_det = self.get_parameter("max_det").get_parameter_value().integer_value
        self.augment = self.get_parameter("augment").get_parameter_value().bool_value
        self.agnostic_nms = (
            self.get_parameter("agnostic_nms").get_parameter_value().bool_value
        )
        self.retina_masks = (
            self.get_parameter("retina_masks").get_parameter_value().bool_value
        )
    
        # ROS params
        self.enable = self.get_parameter("enable").get_parameter_value().bool_value
        self.reliability = (
            self.get_parameter("image_reliability").get_parameter_value().integer_value
        )
        image_reliability = (
            self.get_parameter("image_reliability")
            .get_parameter_value()
            .integer_value
        )
        self.image_qos_profile = QoSProfile(
            reliability=image_reliability,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=1,
        )

        self.rgb_sub_ = message_filters.create_subscription(
            Image, '/camera/image_raw', self.img_callback, self.image_qos_profile)
        
        if self.enable_3d:
            self.init_3d_detection()
        
        self.get_logger().info('YoloInferenceNode initialized')
        
    def init_3d_detection(self):
        
        self.declare_parameter("target_frame", "base_link")
        self.declare_parameter("depth_reliability", QoSReliabilityPolicy.BEST_EFFORT)
        self.declare_parameter("depth_info_reliability", QoSReliabilityPolicy.BEST_EFFORT)
        
        #########################################################
        self.target_frame = (
            self.get_parameter("target_frame").get_parameter_value().string_value
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
        
        self.tf_buffer = Buffer()
        self.cv_bridge = CvBridge()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        self.depth_sub_ = message_filters.create_subscription(
            Image, '/camera/depth/image_raw', self.depth_callback, self.depth_qos_profile)
        

    def run(self):
        self.get_logger().info('YoloInferenceNode running')
        
        
def main(args=None):
    rclpy.init(args=args)
    node = YoloInferenceNode()
    node.run()
    rclpy.shutdown()
