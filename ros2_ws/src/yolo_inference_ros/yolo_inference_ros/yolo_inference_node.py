from typing import List, Dict, Optional
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

from sensor_msgs.msg import Image, CameraInfo
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

        if self.enable_3d:
            self.get_logger().info("Mode: 3D (RGB + Depth + Info)")
            self.init_3d()
            self.tf_buffer = Buffer()
            self.cv_bridge = CvBridge()
            self.tf_listener = TransformListener(self.tf_buffer, self)
            self.image_sub = message_filters.create_subscription(
                self, Image, '/camera/image_raw', qos_profile=self.image_qos_profile)
            self.depth_sub = message_filters.Subscriber(
                self, Image, '/camera/depth/image_raw', qos_profile=self.depth_qos_profile)
            self.depth_info_sub = message_filters.Subscriber(
                self, CameraInfo, '/camera/depth/camera_info', qos_profile=self.depth_info_qos_profile)
            self.ts = message_filters.ApproximateTimeSynchronizer(
                [self.image_sub, self.depth_sub, self.depth_info_sub], 
                queue_size=10, slop=0.5)
            self.ts.registerCallback(self.callback_3d)
            
        else:
            self.get_logger().info("Mode: 2D (RGB Only)")
            self.sub_2d = self.create_subscription(
                Image, 
                '/camera/image_raw', 
                self.callback_2d, 
                self.image_qos_profile
            )
        
        self.init_yolo_model()
        
        self.get_logger().info('YoloInferenceNode initialized')
        
    def init_yolo_model(self):
        self.get_logger().info(f"Loading YOLO model: {self.model} ...")
        
        try:
            # 1. Load Model
            self.yolo_model = YOLO(model=self.model, task=self.task)
            
            # 2. Check Device Safety
            if 'cuda' in self.device and not torch.cuda.is_available():
                self.get_logger().warn(f"CUDA requested but not available! Fallback to CPU.")
                self.device = 'cpu'
            
            # Only move device if it's a PyTorch model, other models will handle device themselves
            if self.model.endswith('.pt'):
                 self.yolo_model.to(self.device)

            self.get_logger().info(f"Model loaded on {self.device}")

        except Exception as e:
            self.get_logger().error(f"CRITICAL: Failed to load YOLO model: {e}")
            raise e
        
        # 3. Fuse Model (Only apply to PyTorch models)
        if self.fuse_model and self.model.endswith('.pt'):
            try:
                self.get_logger().info("Fusing model layers for faster inference...")
                self.yolo_model.fuse()
            except Exception as e:
                self.get_logger().warn(f"Could not fuse model: {e}")

        # 4. WARMUP (Very important for real-time)
        try:
            self.get_logger().info("Warming up model...")
            # Create a dummy input tensor for warmup
            dummy_input = torch.zeros((1, 3, self.imgsz_height, self.imgsz_width)).to(self.device)
            
            # Run inference with dummy input (verbose=False to avoid logging)
            self.yolo_model(dummy_input, verbose=False, device=self.device)
            self.get_logger().info("Warmup complete! System ready.")
            
        except Exception as e:
            self.get_logger().warn(f"Warmup failed (non-critical): {e}")
        

    def init_3d(self):
        
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
        
    def callback_2d(self, image: Image):
        self._process_data(image)
        
    def callback_3d(self, image: Image, depth: Image, depth_info: CameraInfo):
        self._process_data(image, depth, depth_info)
    
    def _process_data(
        self, 
        image_msg, 
        depth_msg: Optional[Image] = None, 
        depth_info_msg: Optional[CameraInfo] = None
    ):
        self.get_logger().info("Processing data...")

        
def main(args=None):
    rclpy.init(args=args)
    node = YoloInferenceNode()
    rclpy.spin(node)
    rclpy.shutdown()
