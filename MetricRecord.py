import math
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped  # 或者其他消息类型

class DistanceLogger(Node):
    def __init__(self):
        super().__init__("distance_logger")
        self.subscription = self.create_subscription(
            PoseStamped,  # 如果是 PoseStamped
            "/drone0/self_localization/pose",
            self.pose_callback,
            10
        )
        self.prev_pos = None
        self.total_distance = 0.0

    def pose_callback(self, msg):
        x = msg.pose.position.x
        y = msg.pose.position.y
        z = msg.pose.position.z

        if self.prev_pos is not None:
            dist = math.dist((x, y, z), self.prev_pos)
            self.total_distance += dist

        self.prev_pos = (x, y, z)
        self.get_logger().info(f"Current total distance: {self.total_distance:.2f}")

def main(args=None):
    rclpy.init(args=args)
    node = DistanceLogger()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
