import rclpy
from rclpy.node import Node

from yolo_msgs.msg import DetectionArray

class PerceptionNavNode(Node):
    def __init__(self):
        super().__init__('perception_nav_node')

        self.get_logger().info('PerceptionNavNode initialized')
        
        self.yolo_sub_ = self.create_subscription(
            DetectionArray,
            '/yolo/tracking',
            self.yolo_callback,
            10
        )
    
    
def main(args=None):
    rclpy.init(args=args)
    perception_nav_node = PerceptionNavNode()
    rclpy.spin(perception_nav_node)
    perception_nav_node.destroy_node()
    rclpy.shutdown()