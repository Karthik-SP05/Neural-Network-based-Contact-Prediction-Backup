# ~/ros2_ws/src/my_ros2_subscriber/my_ros2_subscriber/subscriber.py

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray

class MySubscriber(Node):

    def __init__(self):
        super().__init__('my_subscriber')
        self.subscription = self.create_subscription(
            Float32MultiArray,
            '/svan/actual_feet_data',  # Replace with your actual topic name
            self.listener_callback,
            10)
        self.subscription  # prevent unused variable warning

    def listener_callback(self, msg):
        self.get_logger().info(f'Received message: {msg.data}')

def main(args=None):
    rclpy.init(args=args)
    my_subscriber = MySubscriber()
    rclpy.spin(my_subscriber)
    my_subscriber.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
