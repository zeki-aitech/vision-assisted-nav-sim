import math
from typing import List, Dict, Optional

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from rclpy.qos import QoSHistoryPolicy
from rclpy.qos import QoSDurabilityPolicy
from rclpy.qos import QoSReliabilityPolicy

import message_filters
from cv_bridge import CvBridge
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener

from sensor_msgs.msg import Image, CameraInfo
# Note: We do not need geometry_msgs/Pose2D here because we access it via BoundingBox2D
from vision_msgs.msg import (
    ObjectHypothesis,
    ObjectHypothesisWithPose,
    BoundingBox2D,
    Detection2D,
    Detection2DArray,
    BoundingBox3D,
    Detection3D,
    Detection3DArray,
)

import torch
from ultralytics import YOLO
from ultralytics.engine.results import Results
from ultralytics.engine.results import Boxes


class YoloInferenceNode(Node):
    def __init__(self):
        super().__init__('yolo_inference_node')
        
        # --- Parameter Declaration ---
        self.declare_parameter("enable_3d", False)
        self.declare_parameter("enable_debug", False)
        
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

        # --- Get Parameter Values ---
        self.enable_3d = self.get_parameter("enable_3d").get_parameter_value().bool_value
        self.enable_debug = self.get_parameter("enable_debug").get_parameter_value().bool_value
        
        self.model = self.get_parameter("model").get_parameter_value().string_value
        self.task = self.get_parameter("task").get_parameter_value().string_value
        self.device = self.get_parameter("device").get_parameter_value().string_value
        self.fuse_model = self.get_parameter("fuse_model").get_parameter_value().bool_value
        self.yolo_encoding = self.get_parameter("yolo_encoding").get_parameter_value().string_value

        self.threshold = self.get_parameter("threshold").get_parameter_value().double_value
        self.iou = self.get_parameter("iou").get_parameter_value().double_value
        self.imgsz_height = self.get_parameter("imgsz_height").get_parameter_value().integer_value
        self.imgsz_width = self.get_parameter("imgsz_width").get_parameter_value().integer_value
        self.half = self.get_parameter("half").get_parameter_value().bool_value
        self.max_det = self.get_parameter("max_det").get_parameter_value().integer_value
        self.augment = self.get_parameter("augment").get_parameter_value().bool_value
        self.agnostic_nms = self.get_parameter("agnostic_nms").get_parameter_value().bool_value
        self.retina_masks = self.get_parameter("retina_masks").get_parameter_value().bool_value
    
        self.enable = self.get_parameter("enable").get_parameter_value().bool_value
        image_reliability = self.get_parameter("image_reliability").get_parameter_value().integer_value
        
        self.image_qos_profile = QoSProfile(
            reliability=image_reliability,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=1,
        )

        self.cv_bridge = CvBridge()

        # --- Publishers ---
        self.pub_2d = self.create_publisher(Detection2DArray, 'detections_2d', 10)
        
        if self.enable_debug:
            self.get_logger().info("Debug Mode Enabled: Publishing to ~/debug_image")
            self.pub_debug = self.create_publisher(Image, '~/debug_image', 10)

        # --- Subscribers & 3D Setup ---
        if self.enable_3d:
            self.get_logger().info("Mode: 3D (RGB + Depth + Info)")
            self.init_3d()
            self.tf_buffer = Buffer()
            self.tf_listener = TransformListener(self.tf_buffer, self)
            
            self.pub_3d = self.create_publisher(Detection3DArray, 'detections_3d', 10)
            
            # Use message_filters.Subscriber for synchronization
            self.image_sub = message_filters.Subscriber(
                self, Image, '/camera/image_raw', qos_profile=self.image_qos_profile)
            self.depth_sub = message_filters.Subscriber(
                self, Image, '/camera/depth/image_raw', qos_profile=self.depth_qos_profile)
            self.depth_info_sub = message_filters.Subscriber(
                self, CameraInfo, '/camera/depth/camera_info', qos_profile=self.depth_info_qos_profile)
            
            self.ts = message_filters.ApproximateTimeSynchronizer(
                [self.image_sub, self.depth_sub, self.depth_info_sub], 
                queue_size=10, slop=0.1) 
            self.ts.registerCallback(self.callback_3d)
            
        else:
            self.get_logger().info("Mode: 2D (RGB Only)")
            # Use standard subscription for simple 2D mode
            self.sub_2d = self.create_subscription(
                Image, 
                '/camera/image_raw', 
                self.callback_2d, 
                self.image_qos_profile
            )
        
        # --- Initialize Model ---
        self.init_yolo_model()
        self.get_logger().info('YoloInferenceNode initialized')
        
    def init_yolo_model(self):
        self.get_logger().info(f"Loading YOLO model: {self.model} ...")
        
        try:
            self.yolo_model = YOLO(model=self.model, task=self.task)
            
            # Check CUDA availability to prevent crash
            if 'cuda' in self.device and not torch.cuda.is_available():
                self.get_logger().warn(f"CUDA requested but not available! Fallback to CPU.")
                self.device = 'cpu'
            
            # Move to device if using standard PyTorch model
            if self.model.endswith('.pt'):
                 self.yolo_model.to(self.device)

            self.get_logger().info(f"Model loaded on {self.device}")

        except Exception as e:
            self.get_logger().error(f"CRITICAL: Failed to load YOLO model: {e}")
            raise e
        
        # Fuse model for speed (PyTorch only)
        if self.fuse_model and self.model.endswith('.pt'):
            try:
                self.get_logger().info("Fusing model layers for faster inference...")
                self.yolo_model.fuse()
            except Exception as e:
                self.get_logger().warn(f"Could not fuse model: {e}")

        # Warmup
        try:
            self.get_logger().info("Warming up model...")
            dummy_input = torch.zeros((1, 3, self.imgsz_height, self.imgsz_width)).to(self.device)
            self.yolo_model(dummy_input, verbose=False, device=self.device)
            self.get_logger().info("Warmup complete! System ready.")
        except Exception as e:
            self.get_logger().warn(f"Warmup failed (non-critical): {e}")
        

    def init_3d(self):
        self.declare_parameter("target_frame", "base_link")
        self.declare_parameter("depth_reliability", QoSReliabilityPolicy.BEST_EFFORT)
        self.declare_parameter("depth_info_reliability", QoSReliabilityPolicy.BEST_EFFORT)
        
        self.target_frame = self.get_parameter("target_frame").get_parameter_value().string_value
        depth_reliability = self.get_parameter("depth_reliability").get_parameter_value().integer_value
        depth_info_reliability = self.get_parameter("depth_info_reliability").get_parameter_value().integer_value
        
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
        
    def callback_2d(self, image_msg: Image):
        self._process_data(image_msg)
        
    def callback_3d(self, image_msg: Image, depth_msg: Image, depth_info_msg: CameraInfo):
        self._process_data(image_msg, depth_msg, depth_info_msg)
    
    def _process_data(
        self, 
        image_msg: Image, 
        depth_msg: Optional[Image] = None, 
        depth_info_msg: Optional[CameraInfo] = None
    ):
        if not self.enable:
            return

        # 1. Convert Image
        try:
            cv_image = self.cv_bridge.imgmsg_to_cv2(
                image_msg, desired_encoding=self.yolo_encoding
            )
        except Exception as e:
            self.get_logger().error(f"CV Bridge conversion failed: {e}")
            return
        
        # 2. Inference
        inference_results = self.yolo_model.predict(
            source=cv_image,
            verbose=False,
            stream=False,
            conf=self.threshold,
            iou=self.iou,
            imgsz=(self.imgsz_height, self.imgsz_width),
            half=self.half,
            max_det=self.max_det,
            augment=self.augment,
            agnostic_nms=self.agnostic_nms,
            retina_masks=self.retina_masks,
            device=self.device,
        )
        
        results: Results = inference_results[0].cpu()

        # 3. Prepare ROS Messages
        detections_2d_msg = Detection2DArray()
        detections_2d_msg.header = image_msg.header
        
        detections_3d_msg = Detection3DArray() if self.enable_3d else None
        if self.enable_3d:
            detections_3d_msg.header = image_msg.header

        if results.boxes:
            hypothesis_list = self.parse_hypothesis(results)
            boxes_2d_list = self.parse_boxes(results) # FIX IS INSIDE HERE

            # Prepare for 3D calc
            cv_depth = None
            fx = fy = cx = cy = 0.0
            if self.enable_3d and depth_msg and depth_info_msg:
                try:
                    cv_depth = self.cv_bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")
                    fx = depth_info_msg.k[0]
                    cx = depth_info_msg.k[2]
                    fy = depth_info_msg.k[4]
                    cy = depth_info_msg.k[5]
                except Exception as e:
                    self.get_logger().error(f"Depth processing failed: {e}")

            # Loop through detections
            for i in range(len(results.boxes)):
                # --- 2D Detection ---
                det_2d = Detection2D()
                det_2d.header = image_msg.header
                det_2d.bbox = boxes_2d_list[i]
                
                obj_hyp = ObjectHypothesisWithPose()
                obj_hyp.hypothesis.class_id = hypothesis_list[i]["class_id"]
                obj_hyp.hypothesis.score = hypothesis_list[i]["score"]
                
                det_2d.results.append(obj_hyp)
                detections_2d_msg.detections.append(det_2d)

                # --- 3D Detection ---
                if self.enable_3d and cv_depth is not None:
                    # Note: We now use center.position.x because of the fix
                    u = int(boxes_2d_list[i].center.position.x)
                    v = int(boxes_2d_list[i].center.position.y)
                    
                    height, width = cv_depth.shape
                    
                    if 0 <= u < width and 0 <= v < height:
                        depth_val = cv_depth[v, u]
                        
                        # Handle typical depth encodings (mm vs meters)
                        if "16UC1" in depth_msg.encoding:
                            z = depth_val / 1000.0
                        else:
                            z = float(depth_val)
                            
                        if z > 0.0 and not math.isnan(z):
                            det_3d = Detection3D()
                            det_3d.header = image_msg.header
                            
                            # Pinhole Camera Model
                            x = (u - cx) * z / fx
                            y = (v - cy) * z / fy
                            
                            det_3d.bbox.center.position.x = x
                            det_3d.bbox.center.position.y = y
                            det_3d.bbox.center.position.z = z
                            det_3d.bbox.center.orientation.w = 1.0
                            
                            det_3d.bbox.size.x = 0.5 # Placeholder size
                            det_3d.bbox.size.y = 0.5
                            det_3d.bbox.size.z = 0.5
                            
                            det_3d.results.append(obj_hyp)
                            detections_3d_msg.detections.append(det_3d)

        # 4. Publish
        self.pub_2d.publish(detections_2d_msg)
        if self.enable_3d and detections_3d_msg:
            self.pub_3d.publish(detections_3d_msg)
            
        # 5. Publish Debug Image (If Enabled)
        if self.enable_debug:
            try:
                # Plot returns BGR numpy array
                annotated_frame = results.plot()
                debug_msg = self.cv_bridge.cv2_to_imgmsg(annotated_frame, encoding="bgr8")
                debug_msg.header = image_msg.header
                self.pub_debug.publish(debug_msg)
            except Exception as e:
                self.get_logger().error(f"Failed to publish debug image: {e}")
        
    def parse_hypothesis(self, results: Results) -> List[Dict]:
        hypothesis_list = []
        if results.boxes:
            for box_data in results.boxes:
                cls_id = int(box_data.cls)
                hypothesis = {
                    "class_id": str(cls_id), 
                    "class_name": self.yolo_model.names[cls_id],
                    "score": float(box_data.conf),
                }
                hypothesis_list.append(hypothesis)
        return hypothesis_list

    def parse_boxes(self, results: Results) -> List[BoundingBox2D]:
        """
        FIXED: Updated to use vision_msgs/Pose2D structure (center.position.x)
        """
        boxes_list = []
        if results.boxes:
            for box_data in results.boxes:
                msg = BoundingBox2D()

                # xywh: center x, center y, width, height
                box = box_data.xywh[0]
                
                # --- CRITICAL FIX HERE ---
                # In ROS 2 vision_msgs, 'center' is a Pose2D, which contains 'position'
                msg.center.position.x = float(box[0])
                msg.center.position.y = float(box[1])
                msg.center.theta = 0.0 
                
                msg.size_x = float(box[2])
                msg.size_y = float(box[3])

                boxes_list.append(msg)

        return boxes_list
    

def main(args=None):
    rclpy.init(args=args)
    node = YoloInferenceNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()