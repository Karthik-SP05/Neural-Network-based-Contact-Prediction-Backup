import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import openpyxl

class XlsxPublisher(Node):

    def __init__(self):
        super().__init__('xlsx_publisher')
        self.publisher_ = self.create_publisher(String, 'xlsx_data', 500)
        self.start_row = 6500
        self.end_row = 16199
        self.timer = self.create_timer(0.01, self.timer_callback)  # Publish every second
        self.load_xlsx_file('/home/karthik/NN_Based_Contact_Prediction2/bag_to_mat/Our Data (xlsx)/data3(2)inference/final3_for_inference.xlsx')

    def load_xlsx_file(self, file_path):
        self.workbook = openpyxl.load_workbook(file_path)
        self.sheet = self.workbook.active
        self.rows = list(self.sheet.iter_rows(values_only=True))
        self.current_row = self.start_row  # Start from the 7000th row

    def timer_callback(self):
        if self.current_row <= self.end_row and self.current_row < len(self.rows):
            data = ', '.join(map(str, self.rows[self.current_row]))
            msg = String()
            msg.data = data
            self.publisher_.publish(msg)
            self.get_logger().info(f'Publishing: {msg.data}')
            self.current_row += 1
        else:
            self.get_logger().info('No more data to publish')
            self.timer.cancel()

def main(args=None):
    rclpy.init(args=args)
    node = XlsxPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
