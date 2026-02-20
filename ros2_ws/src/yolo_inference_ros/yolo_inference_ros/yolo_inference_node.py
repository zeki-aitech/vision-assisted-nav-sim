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
        if not self.enable:
            return

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
        
        detections_3d_msg = Detection3DArray() if self.enable_3d else None
        if self.enable_3d:
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
                    det_3d_msg = self.convert_to_3d_bbox(depth_msg, depth_info_msg, det_2d)
                    
                    if det_3d_msg is not None:
                        det_3d_msg.header = image_msg.header
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

    def convert_to_3d_bbox(
        self, 
        depth_msg: Image, 
        depth_info_msg: CameraInfo, 
        det_2d_msg: Detection2D
    ) -> Optional[Detection3D]:
        try:
            depth_image = self.cv_bridge.imgmsg_to_cv2(
                depth_msg, desired_encoding="passthrough")
        except Exception as e:
            self.get_logger().error(f"Depth processing failed: {e}")
            return None

        center_x = det_2d_msg.bbox.center.position.x
        center_y = det_2d_msg.bbox.center.position.y
        size_x = det_2d_msg.bbox.size_x
        size_y = det_2d_msg.bbox.size_y

        u_min = max(int(center_x - size_x // 2), 0)
        u_max = min(int(center_x + size_x // 2), depth_image.shape[1] - 1)
        v_min = max(int(center_y - size_y // 2), 0)
        v_max = min(int(center_y + size_y // 2), depth_image.shape[0] - 1)

        if u_max <= u_min or v_max <= v_min:
            return None

        depth_roi = depth_image[v_min:v_max, u_min:u_max]

        roi_h, roi_w = depth_roi.shape
        y_grid, x_grid = np.meshgrid(
            np.arange(roi_h) + v_min, np.arange(roi_w) + u_min, indexing="ij"
        )
        pixel_coords = np.column_stack([x_grid.flatten(), y_grid.flatten()])                    

        if not np.any(np.isfinite(depth_roi)) or not np.any(depth_roi):
            return None    

        valid_depths = depth_roi.flatten()

        try:
            valid_depths = np.asarray(valid_depths, dtype=np.float64)
            if depth_msg.encoding in ["16UC1", "mono16", "16SC1"]:
                valid_depths = valid_depths / float(self.depth_image_units_divisor)
        except (ValueError, TypeError):
            return None

        valid_mask = (valid_depths > 0) & np.isfinite(valid_depths)
        valid_depths = valid_depths[valid_mask]
        valid_coords = pixel_coords[valid_mask] 

        if len(valid_depths) == 0:
            return None
        
        spatial_weights = self._compute_spatial_weights(
            valid_coords, center_x, center_y, size_x, size_y
        )

        z, z_min, z_max = self._compute_depth_bounds_weighted(
            valid_depths, spatial_weights
        )

        if not np.isfinite(z) or z == 0:
            return None

        y_center, y_min, y_max = self._compute_height_bounds(
            valid_coords, valid_depths, spatial_weights, depth_info_msg
        )

        if not all(np.isfinite([y_center, y_min, y_max])):
            return None

        x_center, x_min, x_max = self._compute_width_bounds(
            valid_coords, valid_depths, spatial_weights, depth_info_msg
        )

        if not all(np.isfinite([x_center, x_min, x_max])):
            return None

        x = x_center
        y = y_center
        w = float(x_max - x_min)
        h = float(y_max - y_min)

        det_3d_msg = Detection3D()
        det_3d_msg.bbox.center.position.x = x
        det_3d_msg.bbox.center.position.y = y
        det_3d_msg.bbox.center.position.z = z
        det_3d_msg.bbox.center.orientation.w = 1.0
        
        det_3d_msg.bbox.size.x = w
        det_3d_msg.bbox.size.y = h
        det_3d_msg.bbox.size.z = float(z_max - z_min)
        
        return det_3d_msg

    @staticmethod
    def _compute_spatial_weights(
        coords: np.ndarray, center_x: int, center_y: int, size_x: int, size_y: int
    ) -> np.ndarray:
        """
        Compute spatial weights for depth values based on distance from 2D bbox center.
        Pixels near the center get higher weight to handle occlusions better.

        Args:
            coords: Nx2 array of pixel coordinates [x, y]
            center_x: X coordinate of bbox center
            center_y: Y coordinate of bbox center
            size_x: Width of bbox
            size_y: Height of bbox

        Returns:
            Array of weights (0-1) for each coordinate
        """
        # Compute normalized distance from center
        dx = (coords[:, 0] - center_x) / (size_x / 2 + 1e-6)
        dy = (coords[:, 1] - center_y) / (size_y / 2 + 1e-6)
        normalized_dist = np.sqrt(dx**2 + dy**2)

        # Use Gaussian-like weighting: higher weight at center, lower at edges
        # sigma = 0.8 means ~80% of bbox radius has high weight
        weights = np.exp(-0.5 * (normalized_dist / 0.8) ** 2)

        # Ensure minimum weight of 0.3 to not completely ignore edge pixels
        weights = np.maximum(weights, 0.3)

        return weights

    @staticmethod
    def _compute_height_bounds(
        valid_coords: np.ndarray,
        valid_depths: np.ndarray,
        spatial_weights: np.ndarray,
        depth_info: CameraInfo,
    ) -> Tuple[float, float, float]:
        """
        Compute 3D height (y-axis) statistics from valid depth points.
        Uses actual 3D point positions instead of just projecting 2D bbox.

        Args:
            valid_coords: Nx2 array of pixel coordinates [x, y]
            valid_depths: N array of depth values in meters
            spatial_weights: N array of spatial weights
            depth_info: Camera intrinsic parameters

        Returns:
            Tuple of (y_center, y_min, y_max) in meters
        """
        # Input validations
        try:
            valid_depths = np.asarray(valid_depths, dtype=np.float64)
            spatial_weights = np.asarray(spatial_weights, dtype=np.float64)
        except (ValueError, TypeError):
            return 0.0, 0.0, 0.0

        if len(valid_coords) == 0 or len(valid_depths) == 0:
            return 0.0, 0.0, 0.0

        if len(valid_coords) < 4:
            # Fallback: just use simple projection
            k = depth_info.k
            py, fy = k[5], k[4]

            # Validate camera parameters
            if fy == 0:
                return 0.0, 0.0, 0.0

            # Validate depths are finite
            if not np.all(np.isfinite(valid_depths)):
                return 0.0, 0.0, 0.0

            y_coords_pixel = valid_coords[:, 1]
            y_3d = valid_depths * (y_coords_pixel - py) / fy

            # Validate result
            if not np.all(np.isfinite(y_3d)):
                return 0.0, 0.0, 0.0

            return float(np.median(y_3d)), float(np.min(y_3d)), float(np.max(y_3d))

        # Convert pixel coordinates to 3D y-coordinates
        k = depth_info.k
        py, fy = k[5], k[4]

        # Validate camera parameters
        if fy == 0:
            return 0.0, 0.0, 0.0

        # Validate depths are finite before calculation
        if not np.all(np.isfinite(valid_depths)):
            return 0.0, 0.0, 0.0

        y_coords_pixel = valid_coords[:, 1]
        y_3d = valid_depths * (y_coords_pixel - py) / fy

        # Validate result
        if not np.any(np.isfinite(y_3d)):
            return 0.0, 0.0, 0.0

        # Filter outliers using robust statistics
        # Compute weighted median as reference
        sorted_idx = np.argsort(y_3d)
        sorted_y = y_3d[sorted_idx]
        sorted_weights = spatial_weights[sorted_idx]
        cumsum_weights = np.cumsum(sorted_weights)
        cumsum_weights /= cumsum_weights[-1] if cumsum_weights[-1] > 0 else 1.0
        median_idx = np.searchsorted(cumsum_weights, 0.5)
        y_median = sorted_y[median_idx]

        # Compute MAD (Median Absolute Deviation)
        deviations = np.abs(y_3d - y_median)
        mad = np.median(deviations)

        # Filter outliers: keep points within 4.5*MAD from median
        # Balanced threshold to handle tall objects while avoiding background
        threshold = np.clip(4.5 * mad, 0.06, 0.50)
        valid_mask = deviations <= threshold
        filtered_y = y_3d[valid_mask]
        filtered_weights = spatial_weights[valid_mask]

        # Ensure we have enough points (at least 12% of data)
        if len(filtered_y) < max(4, len(y_3d) * 0.12):
            filtered_y = y_3d
            filtered_weights = spatial_weights

        # Compute weighted center using trimmed mean
        sorted_idx = np.argsort(filtered_y)
        sorted_y = filtered_y[sorted_idx]
        sorted_weights = filtered_weights[sorted_idx]
        cumsum_weights = np.cumsum(sorted_weights)
        cumsum_weights /= cumsum_weights[-1] if cumsum_weights[-1] > 0 else 1.0

        # Trim 5% from each end for robust center estimation
        trim_low_idx = np.searchsorted(cumsum_weights, 0.05)
        trim_high_idx = np.searchsorted(cumsum_weights, 0.95)

        if trim_high_idx > trim_low_idx:
            trimmed_y = sorted_y[trim_low_idx:trim_high_idx]
            trimmed_weights = sorted_weights[trim_low_idx:trim_high_idx]
            if np.sum(trimmed_weights) > 0:
                y_center = np.average(trimmed_y, weights=trimmed_weights)
            else:
                y_center = np.median(filtered_y)
        else:
            y_center = np.median(filtered_y)

        # Compute extent using balanced percentiles (3rd and 97th)
        # Good balance between capturing object extent and avoiding outliers
        sorted_idx = np.argsort(filtered_y)
        sorted_y = filtered_y[sorted_idx]
        sorted_weights = filtered_weights[sorted_idx]
        cumsum_weights = np.cumsum(sorted_weights)
        cumsum_weights /= cumsum_weights[-1] if cumsum_weights[-1] > 0 else 1.0

        p3_idx = np.searchsorted(cumsum_weights, 0.03)
        p97_idx = np.searchsorted(cumsum_weights, 0.97)

        y_min = sorted_y[p3_idx]
        y_max = sorted_y[p97_idx]

        # Ensure minimum height of 2cm
        min_height = 0.02
        if (y_max - y_min) < min_height:
            half_min = min_height / 2
            y_min = y_center - half_min
            y_max = y_center + half_min

        return float(y_center), float(y_min), float(y_max)

    @staticmethod
    def _compute_width_bounds(
        valid_coords: np.ndarray,
        valid_depths: np.ndarray,
        spatial_weights: np.ndarray,
        depth_info: CameraInfo,
    ) -> Tuple[float, float, float]:
        """
        Compute 3D width (x-axis) statistics from valid depth points.
        Uses actual 3D point positions instead of just projecting 2D bbox.

        Args:
            valid_coords: Nx2 array of pixel coordinates [x, y]
            valid_depths: N array of depth values in meters
            spatial_weights: N array of spatial weights
            depth_info: Camera intrinsic parameters

        Returns:
            Tuple of (x_center, x_min, x_max) in meters
        """
        # Input validations
        try:
            valid_depths = np.asarray(valid_depths, dtype=np.float64)
            spatial_weights = np.asarray(spatial_weights, dtype=np.float64)
        except (ValueError, TypeError):
            return 0.0, 0.0, 0.0

        if len(valid_coords) == 0 or len(valid_depths) == 0:
            return 0.0, 0.0, 0.0

        if len(valid_coords) < 4:
            # Fallback: just use simple projection
            k = depth_info.k
            px, fx = k[2], k[0]

            # Validate camera parameters
            if fx == 0:
                return 0.0, 0.0, 0.0

            # Validate depths are finite
            if not np.all(np.isfinite(valid_depths)):
                return 0.0, 0.0, 0.0

            x_coords_pixel = valid_coords[:, 0]
            x_3d = valid_depths * (x_coords_pixel - px) / fx

            # Validate result
            if not np.all(np.isfinite(x_3d)):
                return 0.0, 0.0, 0.0

            return float(np.median(x_3d)), float(np.min(x_3d)), float(np.max(x_3d))

        # Convert pixel coordinates to 3D x-coordinates
        k = depth_info.k
        px, fx = k[2], k[0]

        # Validate camera parameters
        if fx == 0:
            return 0.0, 0.0, 0.0

        # Validate depths are finite before calculation
        if not np.all(np.isfinite(valid_depths)):
            return 0.0, 0.0, 0.0

        x_coords_pixel = valid_coords[:, 0]
        x_3d = valid_depths * (x_coords_pixel - px) / fx

        # Validate result
        if not np.any(np.isfinite(x_3d)):
            return 0.0, 0.0, 0.0

        # Filter outliers using robust statistics
        # Compute weighted median as reference
        sorted_idx = np.argsort(x_3d)
        sorted_x = x_3d[sorted_idx]
        sorted_weights = spatial_weights[sorted_idx]
        cumsum_weights = np.cumsum(sorted_weights)
        cumsum_weights /= cumsum_weights[-1] if cumsum_weights[-1] > 0 else 1.0
        median_idx = np.searchsorted(cumsum_weights, 0.5)
        x_median = sorted_x[median_idx]

        # Compute MAD (Median Absolute Deviation)
        deviations = np.abs(x_3d - x_median)
        mad = np.median(deviations)

        # Adaptive threshold based on depth variance (helps with occlusions)
        # Check if object has varying depth (might indicate occlusion)
        depth_std = np.std(valid_depths)
        if depth_std > 0.15:  # High depth variation - likely occlusion or 3D object
            # Use tighter threshold to avoid including background
            threshold = np.clip(4.0 * mad, 0.06, 0.40)
        else:  # Uniform depth - flat object
            # Can be more permissive
            threshold = np.clip(4.5 * mad, 0.08, 0.50)

        valid_mask = deviations <= threshold
        filtered_x = x_3d[valid_mask]
        filtered_weights = spatial_weights[valid_mask]

        # Ensure we have enough points (at least 12% of data)
        if len(filtered_x) < max(4, len(x_3d) * 0.12):
            filtered_x = x_3d
            filtered_weights = spatial_weights

        # Compute weighted center using trimmed mean
        sorted_idx = np.argsort(filtered_x)
        sorted_x = filtered_x[sorted_idx]
        sorted_weights = filtered_weights[sorted_idx]
        cumsum_weights = np.cumsum(sorted_weights)
        cumsum_weights /= cumsum_weights[-1] if cumsum_weights[-1] > 0 else 1.0

        # Trim 5% from each end for robust center estimation
        trim_low_idx = np.searchsorted(cumsum_weights, 0.05)
        trim_high_idx = np.searchsorted(cumsum_weights, 0.95)

        if trim_high_idx > trim_low_idx:
            trimmed_x = sorted_x[trim_low_idx:trim_high_idx]
            trimmed_weights = sorted_weights[trim_low_idx:trim_high_idx]
            if np.sum(trimmed_weights) > 0:
                x_center = np.average(trimmed_x, weights=trimmed_weights)
            else:
                x_center = np.median(filtered_x)
        else:
            x_center = np.median(filtered_x)

        # Compute extent using balanced percentiles (3rd and 97th)
        # Good balance between capturing object extent and avoiding outliers
        sorted_idx = np.argsort(filtered_x)
        sorted_x = filtered_x[sorted_idx]
        sorted_weights = filtered_weights[sorted_idx]
        cumsum_weights = np.cumsum(sorted_weights)
        cumsum_weights /= cumsum_weights[-1] if cumsum_weights[-1] > 0 else 1.0

        p3_idx = np.searchsorted(cumsum_weights, 0.03)
        p97_idx = np.searchsorted(cumsum_weights, 0.97)

        x_min = sorted_x[p3_idx]
        x_max = sorted_x[p97_idx]

        # Ensure minimum width of 2cm
        min_width = 0.02
        if (x_max - x_min) < min_width:
            half_min = min_width / 2
            x_min = x_center - half_min
            x_max = x_center + half_min

        return float(x_center), float(x_min), float(x_max)

    @staticmethod
    def _compute_depth_bounds_weighted(
        depth_values: np.ndarray, spatial_weights: np.ndarray
    ) -> Tuple[float, float, float]:
        """
        Compute robust depth statistics with spatial weighting to handle occlusions.

        Args:
            depth_values: 1D array of valid depth values (> 0)
            spatial_weights: 1D array of spatial weights (0-1) for each depth

        Returns:
            Tuple of (z_center, z_min, z_max) representing the object's depth
        """
        # Input validations
        try:
            depth_values = np.asarray(depth_values, dtype=np.float64)
            spatial_weights = np.asarray(spatial_weights, dtype=np.float64)
        except (ValueError, TypeError):
            return 0.0, 0.0, 0.0

        if len(depth_values) == 0:
            return 0.0, 0.0, 0.0

        # Validate that all values are finite
        valid_mask = np.isfinite(depth_values) & np.isfinite(spatial_weights)
        depth_values = depth_values[valid_mask]
        spatial_weights = spatial_weights[valid_mask]

        if len(depth_values) == 0:
            return 0.0, 0.0, 0.0

        if len(depth_values) < 4:
            z_center = float(np.median(depth_values))
            return z_center, float(np.min(depth_values)), float(np.max(depth_values))

        # Step 1: Multi-scale histogram analysis for robust mode detection
        depth_range = np.ptp(depth_values)
        if not np.isfinite(depth_range) or depth_range <= 0:
            n_bins = 30
        else:
            n_bins = max(20, min(60, int(depth_range / 0.01)))

        # Create weighted histogram
        hist, bin_edges = np.histogram(depth_values, bins=n_bins, weights=spatial_weights)

        # Smooth histogram to reduce noise while preserving peaks
        if len(hist) >= 5:
            # Simple moving average smoothing
            kernel_size = min(5, len(hist) // 4)
            kernel = np.ones(kernel_size) / kernel_size
            hist_smooth = np.convolve(hist, kernel, mode="same")
        else:
            hist_smooth = hist

        # Find peak (mode) - highest weighted density region
        peak_bin_idx = np.argmax(hist_smooth)
        mode_depth = (bin_edges[peak_bin_idx] + bin_edges[peak_bin_idx + 1]) / 2

        # Step 2: Adaptive outlier filtering with less aggressive thresholds
        deviations = np.abs(depth_values - mode_depth)

        # Compute robust MAD without inverse weighting to avoid over-filtering
        mad = np.median(deviations)

        # More permissive threshold - adjust based on object size and uniformity
        # Check depth distribution uniformity
        q25 = np.percentile(depth_values, 25)
        q75 = np.percentile(depth_values, 75)
        iqr = q75 - q25

        # Adaptive threshold: looser for varied depth, tighter for uniform
        if iqr < 0.03:  # Very uniform depth (<3cm IQR)
            # For flat objects, use tighter bounds
            threshold = np.clip(3.5 * mad, 0.08, 0.30)
        elif iqr < 0.10:  # Moderate variation (<10cm IQR)
            # Standard threshold
            threshold = np.clip(4.0 * mad, 0.12, 0.40)
        else:  # High variation (>10cm IQR)
            # For complex 3D objects, use very permissive bounds
            threshold = np.clip(5.0 * mad, 0.15, 0.60)

        # Keep depths within threshold
        object_mask = deviations <= threshold
        object_depths = depth_values[object_mask]
        object_weights = spatial_weights[object_mask]

        # Fallback if filtering was too aggressive
        min_points = max(6, int(len(depth_values) * 0.15))  # Keep at least 15% of points
        if len(object_depths) < min_points:
            # Use weighted percentiles with wider range
            sorted_idx = np.argsort(depth_values)
            cumsum_weights = np.cumsum(spatial_weights[sorted_idx])
            cumsum_weights /= cumsum_weights[-1]

            # Find 2nd and 85th weighted percentiles (wider range)
            p2_idx = np.searchsorted(cumsum_weights, 0.02)
            p85_idx = np.searchsorted(cumsum_weights, 0.85)

            p2_val = depth_values[sorted_idx[p2_idx]]
            p85_val = depth_values[sorted_idx[p85_idx]]

            object_mask = (depth_values >= p2_val) & (depth_values <= p85_val)
            object_depths = depth_values[object_mask]
            object_weights = spatial_weights[object_mask]

        if len(object_depths) == 0:
            object_depths = depth_values
            object_weights = spatial_weights

        # Step 3: Compute robust weighted center using trimmed mean
        if np.sum(object_weights) > 0:
            # Use weighted average, but trim extreme 2% on each side first
            sorted_idx = np.argsort(object_depths)
            sorted_depths = object_depths[sorted_idx]
            sorted_weights = object_weights[sorted_idx]

            cumsum_weights = np.cumsum(sorted_weights)
            cumsum_weights /= cumsum_weights[-1] if cumsum_weights[-1] > 0 else 1.0

            # Trim 2% from each end
            trim_low_idx = np.searchsorted(cumsum_weights, 0.02)
            trim_high_idx = np.searchsorted(cumsum_weights, 0.98)

            if trim_high_idx > trim_low_idx:
                trimmed_depths = sorted_depths[trim_low_idx:trim_high_idx]
                trimmed_weights = sorted_weights[trim_low_idx:trim_high_idx]

                if np.sum(trimmed_weights) > 0:
                    z_center = np.average(trimmed_depths, weights=trimmed_weights)
                else:
                    z_center = np.median(object_depths)
            else:
                z_center = np.average(object_depths, weights=object_weights)
        else:
            z_center = np.median(object_depths)

        # Step 4: Compute extent using balanced weighted percentiles
        sorted_idx = np.argsort(object_depths)
        cumsum_weights = np.cumsum(object_weights[sorted_idx])
        cumsum_weights /= cumsum_weights[-1] if cumsum_weights[-1] > 0 else 1.0

        # Use 1st and 99th percentiles for depth (slightly more coverage than width/height)
        p1_idx = np.searchsorted(cumsum_weights, 0.01)
        p99_idx = np.searchsorted(cumsum_weights, 0.99)

        z_min = object_depths[sorted_idx[p1_idx]]
        z_max = object_depths[sorted_idx[p99_idx]]

        # Validate and adjust bounds relative to center
        # Ensure center is within bounds (sanity check)
        if z_center < z_min or z_center > z_max:
            # Recompute bounds symmetrically around center
            depth_extent = max(z_max - z_min, 0.02)  # At least 2cm
            z_min = z_center - depth_extent / 2
            z_max = z_center + depth_extent / 2

        # Ensure minimum depth size of 2cm (more realistic for real objects)
        min_depth_size = 0.02
        if (z_max - z_min) < min_depth_size:
            # Expand around center
            half_min = min_depth_size / 2
            z_min = z_center - half_min
            z_max = z_center + half_min

        return float(z_center), float(z_min), float(z_max)



def main(args=None):
    rclpy.init(args=args)
    node = YoloInferenceNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()