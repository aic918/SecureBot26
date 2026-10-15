import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool
from geometry_msgs.msg import Twist


class MotionManagerNode(Node):
    def __init__(self):
        super().__init__('motion_manager_node')

        self.nav_cmd_sub = self.create_subscription(
            Twist,
            '/nav_cmd_vel',
            self.nav_cmd_callback,
            10
        )

        self.person_sub = self.create_subscription(
            Bool,
            '/person_detected',
            self.person_callback,
            10
        )

        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        self.latest_nav_cmd = Twist()
        self.person_detected = False

        self.declare_parameter('stop_hold_time', 2.0)
        self.stop_hold_time = self.get_parameter('stop_hold_time').value

        self.last_person_time = None

        self.timer = self.create_timer(0.05, self.publish_final_cmd)

        self.get_logger().info('Motion manager started.')
        self.get_logger().info('Subscribing to /nav_cmd_vel and /person_detected')
        self.get_logger().info('Publishing final motor command to /cmd_vel')

    def nav_cmd_callback(self, msg):
        self.latest_nav_cmd = msg

    def person_callback(self, msg):
        self.person_detected = msg.data

        if self.person_detected:
            self.last_person_time = self.get_clock().now()
            self.get_logger().info('Person detected. Stopping robot.')

    def publish_final_cmd(self):
        final_cmd = Twist()

        should_stop = False

        if self.person_detected:
            should_stop = True

        if self.last_person_time is not None:
            elapsed = (self.get_clock().now() - self.last_person_time).nanoseconds / 1e9
            if elapsed < self.stop_hold_time:
                should_stop = True

        if should_stop:
            final_cmd.linear.x = 0.0
            final_cmd.angular.z = 0.0
        else:
            final_cmd = self.latest_nav_cmd

        self.cmd_pub.publish(final_cmd)


def main(args=None):
    rclpy.init(args=args)
    node = MotionManagerNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
