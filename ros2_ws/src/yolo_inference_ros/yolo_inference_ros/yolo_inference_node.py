import math
from typing import List, Dict, Optional, Tuple

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from rclpy.qos import QoSHistoryPolicy
from rclpy.qos import QoSDurabilityPolicy
from rclpy.qos import QoSReliabilityPolicy
from rclpy.duration import Duration

import message_filters
from cv_bridge import CvBridge
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener

from sensor_msgs.msg import Image, CameraInfo
from vision_msgs.msg import (
    ObjectHypothesis,
    ObjectHypothesisWithPose,
    BoundingBox2D,
    Detection2D,
    Detection2DArray,
    BoundingBox3D,
    Detection3D,
    Detection3DArray,
    LabelInfo, VisionClass
)
from visualization_msgs.msg import Marker, MarkerArray

import torch
from ultralytics import YOLO
from ultralytics.engine.results import Results

import numpy as np

from yolo_inference_ros.depth_processor import DepthProcessor


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
        self.declare_parameter("tracker", "") # path to the tracker config file

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
        self.tracker = self.get_parameter("tracker").get_parameter_value().string_value

        self.threshold = self.get_parameter("threshold").get_parameter_value().double_value
        self.iou = self.get_parameter("iou").get_parameter_value().double_value
        self.imgsz_height = self.get_parameter("imgsz_height").get_parameter_value().integer_value
        self.imgsz_width = self.get_parameter("imgsz_width").get_parameter_value().integer_value
        self.half = self.get_parameter("half").get_parameter_value().bool_value
        self.max_det = self.get_parameter("max_det").get_parameter_value().integer_value
        self.augment = self.get_parameter("augment").get_parameter_value().bool_value
        self.agnostic_nms = self.get_parameter("agnostic_nms").get_parameter_value().bool_value
        self.retina_masks = self.get_parameter("retina_masks").get_parameter_value().bool_value
    
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
        latch_qos = QoSProfile(
            depth=1,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST
        )
        self.pub_label_info = self.create_publisher(LabelInfo, 'labels', latch_qos)
        
        if self.enable_debug:
            self.get_logger().info("Debug Mode Enabled: Publishing to ~/debug_image and ~/debug_markers_3d")
            self.pub_debug = self.create_publisher(Image, '~/debug_image', 10)
            if self.enable_3d:
                self.pub_markers = self.create_publisher(MarkerArray, '~/debug_markers_3d', 10)

        # --- Subscribers & 3D Setup ---
        if self.enable_3d:
            self.get_logger().info("Mode: 3D (RGB + Depth + Info)")
            self.init_3d()
            self.tf_buffer = Buffer()
            self.tf_listener = TransformListener(self.tf_buffer, self)
            
            self.pub_3d = self.create_publisher(Detection3DArray, 'detections_3d', 10)
            
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
            self.sub_2d = self.create_subscription(
                Image, 
                '/camera/image_raw', 
                self.callback_2d, 
                self.image_qos_profile
            )
        
        # --- Initialize Model ---
        self.init_yolo_model()
        self.publish_label_info()
        
        self.get_logger().info('YoloInferenceNode initialized')
        
    def init_yolo_model(self):
        self.get_logger().info(f"Loading YOLO model: {self.model} ...")
        
        try:
            self.yolo_model = YOLO(model=self.model, task=self.task)
            
            if 'cuda' in self.device and not torch.cuda.is_available():
                self.get_logger().warn("CUDA requested but not available! Fallback to CPU.")
                self.device = 'cpu'
            
            if self.model.endswith('.pt'):
                 self.yolo_model.to(self.device)

            self.get_logger().info(f"Model loaded on {self.device}")

        except Exception as e:
            self.get_logger().error(f"CRITICAL: Failed to load YOLO model: {e}")
            raise e
        
        if self.fuse_model and self.model.endswith('.pt'):
            try:
                self.get_logger().info("Fusing model layers for faster inference...")
                self.yolo_model.fuse()
            except Exception as e:
                self.get_logger().warn(f"Could not fuse model: {e}")

        try:
            self.get_logger().info("Warming up model...")
            dummy_input = torch.zeros((1, 3, self.imgsz_height, self.imgsz_width)).to(self.device)
            self.yolo_model(dummy_input, verbose=False, device=self.device)
            self.get_logger().info("Warmup complete! System ready.")
        except Exception as e:
            self.get_logger().warn(f"Warmup failed (non-critical): {e}")

    def init_3d(self):
        self.declare_parameter("depth_image_units_divisor", 1000)
        self.declare_parameter("depth_reliability", QoSReliabilityPolicy.BEST_EFFORT)
        self.declare_parameter("depth_info_reliability", QoSReliabilityPolicy.BEST_EFFORT)
        
        self.depth_image_units_divisor = (
            self.get_parameter("depth_image_units_divisor")
            .get_parameter_value()
            .integer_value
        )
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
        self.depth_processor = DepthProcessor(
            depth_image_units_divisor=self.depth_image_units_divisor
        )

    def publish_label_info(self):
        """
        Publish the dictionary mapping of class IDs to Class names.
        Published once using Transient Local QoS.
        """
        msg = LabelInfo()
        # Note: If no image has arrived yet, clock will be current time
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "yolo_model" # Generic frame for info
        
        # Populate the class mapping from the loaded YOLO model
        for cls_id, cls_name in self.yolo_model.names.items():
            vc = VisionClass()
            vc.class_id = int(cls_id)
            vc.class_name = str(cls_name)
            msg.class_map.append(vc)
            
        msg.threshold = float(self.threshold)
        
        self.pub_label_info.publish(msg)
        self.get_logger().info(f"Published LabelInfo mapping for {len(self.yolo_model.names)} classes to /labels")
        
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

        try:
            cv_image = self.cv_bridge.imgmsg_to_cv2(
                image_msg, desired_encoding=self.yolo_encoding
            )
        except Exception as e:
            self.get_logger().error(f"CV Bridge conversion failed: {e}")
            return
        
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

        detections_2d_msg = Detection2DArray()
        detections_2d_msg.header = image_msg.header
        
        if self.enable_3d:
            detections_3d_msg = Detection3DArray()
            detections_3d_msg.header = image_msg.header

        if results.boxes:
            hypothesis_list = self.parse_hypothesis(results)
            boxes_2d_list = self.parse_boxes(results) 

            for i in range(len(results.boxes)):
                # --- 2D Detection Processing ---
                det_2d = Detection2D()
                det_2d.header = image_msg.header
                det_2d.bbox = boxes_2d_list[i]
                
                obj_hyp = ObjectHypothesisWithPose()
                obj_hyp.hypothesis.class_id = hypothesis_list[i]["class_id"]
                obj_hyp.hypothesis.score = hypothesis_list[i]["score"]
                
                det_2d.results.append(obj_hyp)
                detections_2d_msg.detections.append(det_2d)

                # --- 3D Detection Processing ---
                if self.enable_3d and depth_msg is not None and depth_info_msg is not None:
                    try:
                        depth_image = self.cv_bridge.imgmsg_to_cv2(
                            depth_msg, desired_encoding="passthrough")
                    except Exception as e:
                        self.get_logger().error(f"Depth processing failed: {e}")
                        return None
                    
                    box_3d_data = self.depth_processor.convert_to_3d_bbox(
                        depth_image=depth_image,
                        depth_info=depth_info_msg,
                        center_x=det_2d.bbox.center.position.x,
                        center_y=det_2d.bbox.center.position.y,
                        size_x=det_2d.bbox.size_x,
                        size_y=det_2d.bbox.size_y
                    )
                    
                    if box_3d_data is not None:
                        det_3d_msg = Detection3D()
                        det_3d_msg.header = image_msg.header
                        
                        det_3d_msg.bbox.center.position.x = box_3d_data.x
                        det_3d_msg.bbox.center.position.y = box_3d_data.y
                        det_3d_msg.bbox.center.position.z = box_3d_data.z
                        det_3d_msg.bbox.center.orientation.w = 1.0
                        
                        det_3d_msg.bbox.size.x = box_3d_data.w
                        det_3d_msg.bbox.size.y = box_3d_data.h
                        det_3d_msg.bbox.size.z = box_3d_data.d
                        
                        det_3d_msg.results.append(obj_hyp)
                        detections_3d_msg.detections.append(det_3d_msg)

        # Publish Results
        self.pub_2d.publish(detections_2d_msg)
        if self.enable_3d and detections_3d_msg is not None:
            self.pub_3d.publish(detections_3d_msg)
            
        # Publish Debug Image & Markers (If Enabled)
        if self.enable_debug:
            try:
                # 1. Publish Annotated 2D Image
                annotated_frame = results.plot()
                debug_msg = self.cv_bridge.cv2_to_imgmsg(annotated_frame, encoding="bgr8")
                debug_msg.header = image_msg.header
                self.pub_debug.publish(debug_msg)
                
                # 2. Publish 3D Markers to RViz2
                if self.enable_3d and detections_3d_msg is not None:
                    marker_array = MarkerArray()
                    for i, det_3d in enumerate(detections_3d_msg.detections):
                        # Create box and text markers, extend the array
                        new_markers = self.create_bb_markers(det_3d, color=(0, 255, 0), base_id=i)
                        marker_array.markers.extend(new_markers)
                    self.pub_markers.publish(marker_array)
                    
            except Exception as e:
                self.get_logger().error(f"Failed to publish debug visualizations: {e}")

    def create_bb_markers(self, detection: Detection3D, color: Tuple[int, int, int], base_id: int) -> List[Marker]:
        """
        Create a 3D bounding box AND a text label for RViz visualization.
        """
        markers = []
        bbox = detection.bbox
        lifetime = Duration(seconds=0.5).to_msg()

        # ---------------------------
        # 1. Box Marker (CUBE)
        # ---------------------------
        box_marker = Marker()
        box_marker.header = detection.header
        box_marker.ns = "yolo_3d_boxes"
        box_marker.id = base_id * 2  # Evens for boxes
        box_marker.type = Marker.CUBE
        box_marker.action = Marker.ADD
        box_marker.frame_locked = False

        box_marker.pose.position.x = bbox.center.position.x
        box_marker.pose.position.y = bbox.center.position.y
        box_marker.pose.position.z = bbox.center.position.z
        
        box_marker.pose.orientation.x = bbox.center.orientation.x
        box_marker.pose.orientation.y = bbox.center.orientation.y
        box_marker.pose.orientation.z = bbox.center.orientation.z
        box_marker.pose.orientation.w = bbox.center.orientation.w
        
        box_marker.scale.x = bbox.size.x
        box_marker.scale.y = bbox.size.y
        box_marker.scale.z = bbox.size.z

        box_marker.color.r = color[0] / 255.0
        box_marker.color.g = color[1] / 255.0
        box_marker.color.b = color[2] / 255.0
        box_marker.color.a = 0.4
        box_marker.lifetime = lifetime

        markers.append(box_marker)

        # ---------------------------
        # 2. Text Marker (TEXT_VIEW_FACING)
        # ---------------------------
        if detection.results:
            text_marker = Marker()
            text_marker.header = detection.header
            text_marker.ns = "yolo_3d_labels"
            text_marker.id = (base_id * 2) + 1  # Odds for text
            text_marker.type = Marker.TEXT_VIEW_FACING
            text_marker.action = Marker.ADD
            text_marker.frame_locked = False

            text_marker.pose.position.x = bbox.center.position.x
            text_marker.pose.position.y = bbox.center.position.y - (bbox.size.y / 2.0) - 0.1
            text_marker.pose.position.z = bbox.center.position.z
            
            text_marker.scale.z = 0.15 
            
            text_marker.color.r = 1.0
            text_marker.color.g = 1.0
            text_marker.color.b = 1.0
            text_marker.color.a = 1.0
            
            class_id = detection.results[0].hypothesis.class_id
            class_name = self.yolo_model.names[int(class_id)]
            score = detection.results[0].hypothesis.score
            text_marker.text = f"{class_name}-{class_id}-({score:.2f})"
            
            text_marker.lifetime = lifetime
            markers.append(text_marker)

        return markers

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
        boxes_list = []
        if results.boxes:
            for box_data in results.boxes:
                msg = BoundingBox2D()
                box = box_data.xywh[0]
                
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