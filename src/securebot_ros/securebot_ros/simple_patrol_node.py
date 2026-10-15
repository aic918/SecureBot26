import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist


class SimplePatrolNode(Node):
    def __init__(self):
        super().__init__('simple_patrol_node')

        self.publisher = self.create_publisher(Twist, '/nav_cmd_vel', 10)

        self.declare_parameter('forward_speed', 0.12)
        self.forward_speed = self.get_parameter('forward_speed').value

        self.timer = self.create_timer(0.1, self.publish_patrol_cmd)

        self.get_logger().info('Simple autonomous patrol started.')
        self.get_logger().info('Publishing movement commands to /nav_cmd_vel')

    def publish_patrol_cmd(self):
        cmd = Twist()

        # Simple autonomous behavior: slowly drive forward.
        # Later this gets replaced by obstacle avoidance, SLAM, or Nav2.
        cmd.linear.x = self.forward_speed
        cmd.angular.z = 0.0

        self.publisher.publish(cmd)


def main(args=None):
    rclpy.init(args=args)
    node = SimplePatrolNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
