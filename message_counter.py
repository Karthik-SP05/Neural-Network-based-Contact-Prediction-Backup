import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from rclpy.qos import QoSProfile

class MessageCounterNode(Node):

    def __init__(self):
        super().__init__('message_counter_node')
        self.message_count = 0
        self.subscription = self.create_subscription(
            String,
            '/xlsx_data',  # Replace with your topic name
            self.listener_callback,
            QoSProfile(depth=10)
        )
        self.subscription  # Prevent unused variable warning

    def listener_callback(self, msg):
        self.message_count += 1
        self.get_logger().info(f'Number of messages received: {self.message_count}')

def main(args=None):
    rclpy.init(args=args)
    node = MessageCounterNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
