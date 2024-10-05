import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from std_msgs.msg import String
import numpy as np
from collections import deque
import os
import argparse
import glob
import sys
sys.path.append('.')
import yaml
from tqdm import tqdm
import scipy.io as sio

import lcm
# from lcm_types.python import contact_t, leg_control_data_lcmt, microstrain_lcmt
import time

import torch.optim as optim

from contact_cnn import *
from utils.data_handler import *
class BagSubscriberNode(Node):

    def __init__(self, device,config,model):
        super().__init__('bag_subscriber_node')
        self.device = device
        self.config = config
        self.model = model
        # Create subscription to 'your_topic'
        self.infer_results = torch.empty(0,4,dtype=torch.uint8).to(device)
        self.subscription = self.create_subscription(
            String,
            '/xlsx_data',  # Replace with your topic name
            self.listener_callback,
            QoSProfile(depth=100)
        )
        # self.dataset = contact_dataset(label_path=self.config['label_path'],\
        #                         window_size=self.config['window_size'],device=device)

        self.subscription  # Prevent unused variable warning
        # Buffer to store the latest 150 data points
        self.data_buffer = deque(maxlen=self.config['window_size'])
        self.current_start_idx = 0  # Index to keep track of the start of the sliding window
        # self.count = 0
        # self.count1 = 0
        # self.processing = False
    def listener_callback(self, msg):

        ##obtaining the number of times callback is called
        # self.count += 1
        # self.get_logger().info(f"number of callback executions: {self.count}")

        # if self.processing:
        #     return
        # self.processing = True

        # start_time = time.time() #time taken to execute(start time)

        # Get the data from the message
        data_str = msg.data

        # Split the comma-separated string into individual values
        data_values = data_str.split(',')
        
        # Convert the values to floats (adjust data type as needed)
        np_array = np.array([float(value) for value in data_values])
        # self.get_logger().info(f"Received array: {np_array} ")

        # Add the new data point to the buffer
        self.data_buffer.append(np_array)
        
        # Increment the start index to maintain sliding window
        self.current_start_idx += 1
        
        # Check if we have received more than 150 messages
        if len(self.data_buffer) >= self.config['window_size']:
            # Convert the deque to a NumPy array
            # self.get_logger().info(f"Received array: {self.count} ")
            buffer_array = np.array(self.data_buffer)
            # Process the NumPy array
            self.process_np_array(buffer_array)
            # self.get_logger().info(f"length of infer results: {len(self.infer_results)}") #length of infer results after each message process
            # Remove the oldest data point from the buffer
            self.data_buffer.popleft()
            self.current_start_idx -= 1  # Adjust start index accordingly

            # self.subscription.destroy() #checking number of rows in data buffer
            # rclpy.shutdown() #checking number of rows in data buffer
        # self.processing = False

        ## time taken for execution if callback(end time)
        # end_time = time.time()
        # execution_time = end_time - start_time
        # if len(self.data_buffer) == self.config['window_size']-1: self.get_logger().info(f"Execution time of callback: {execution_time:.4f} seconds")

    def process_np_array(self, buffer_array):
        # start_time = time.time()
        # Example processing function
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # print('Using ', device)

        # parser = argparse.ArgumentParser(description='Test the contcat network')
        # parser.add_argument('--config_name', type=str, default=os.path.dirname(os.path.abspath(__file__))+'/../config/inference_one_seq_params.yaml')
        # args = parser.parse_args()

        # config = yaml.load(open(args.config_name))
    
        # dataset = contact_dataset(label_path=config['label_path'],\
        #                         window_size=config['window_size'],device=device)
        # dataloader = DataLoader(dataset=dataset, batch_size=config['batch_size'])

        def decimal2binary(x):
            mask = 2**torch.arange(4-1,-1,-1).to(x.device, x.dtype)

            return x.unsqueeze(-1).bitwise_and(mask).ne(0).byte()

        def inference(np_array, model, device):
            # start_time = time.time()
            with torch.no_grad():
                input_data1 = torch.from_numpy(np_array).type('torch.FloatTensor').to(device)
                this_data = (input_data1[:,:]-torch.mean(input_data1[:,:],dim=0))\
                            /torch.std(input_data1[:,:],dim=0)
                input_data = this_data.unsqueeze(0)
                output = model(input_data)
                _, prediction = torch.max(output,1)
                bin_pred = decimal2binary(prediction)
                # infer_results = torch.cat((infer_results, bin_pred), 0)
                # end_time = time.time()
                # execution_time = end_time - start_time
                # self.get_logger().info(f"Execution time of callback: {execution_time:.4f} seconds")

            return bin_pred  
         
        self.infer_results = torch.cat((self.infer_results, inference(buffer_array, self.model, self.device)), 0)

        # end_time = time.time()
        # execution_time = end_time - start_time
        # self.get_logger().info(f"Execution time of callback: {execution_time:.4f} seconds")

    def print_infer_results(self):
        # Print the accumulated inference results
        print("Infer Results:")
        print(len(self.infer_results))

    def node_destroyer(self):
        self.get_logger().info("Node is shutting down...")
        super().destroy_node()   

    def save2mat(self):

        def decimal2binary(x):
            mask = 2**torch.arange(4-1,-1,-1).to(x.device, x.dtype)
            return x.unsqueeze(-1).bitwise_and(mask).ne(0).byte()

        # mat_raw_data = sio.loadmat(config['mat_data_path'])
        data = np.load(self.config['data_path'])
        label_deci_np = np.load(self.config['label_path'])
        label_deci = torch.from_numpy(label_deci_np)
        label = decimal2binary(label_deci).reshape(-1,4)

        out = {}
        out['contacts_est'] = self.infer_results.cpu().numpy()
        out['contacts_gt'] = label[:,:].numpy()
        out['q'] = data[:,:12]
        out['qd'] = data[:,12:24]
        out['imu_acc'] = data[:,24:27]
        out['imu_omega'] = data[:,27:30]
        out['p'] = data[:,30:42]
        out['v'] = data[:,42:54]

        sio.savemat(self.config['mat_save_path'],out)
        print("Saved data to mat!")
    
    def compute_accuracy(self):
        def decimal2binary(x):
            mask = 2**torch.arange(4-1,-1,-1).to(x.device, x.dtype)
            return x.unsqueeze(-1).bitwise_and(mask).ne(0).byte()
        def binary_to_decimal(tensor):
            # Assuming tensor is a 2D tensor where each row represents a binary number
            num_bits = tensor.size(1)  # Number of bits in each binary number (assuming all rows have the same length)
            powers_of_two = 2 ** torch.arange(num_bits - 1, -1, -1, dtype=torch.float32, device=tensor.device)
            decimal_values = torch.matmul(tensor.float(), powers_of_two)
            return decimal_values.long() 
    
        labels_array = np.load(self.config['label_path'])
        labels_tensor = torch.from_numpy(labels_array).type('torch.LongTensor').to(self.device)
        labels_1Dtensor = labels_tensor.squeeze(1)
        labels_1Dtensor = labels_1Dtensor[150:]
        labels_tensor_binary = decimal2binary(labels_tensor)  #labels_2d_tensor_binary = (86196,1,4)
        labels_2Dtensor_binary_squeezed = labels_tensor_binary.squeeze(1) #squeeze dimension at first index
        bin_gt = labels_2Dtensor_binary_squeezed[150: , :]
        prediction_decimal  = binary_to_decimal(self.infer_results)

        # print(self.infer_results.shape)
        # print(bin_gt.shape)
        correct_per_leg = (self.infer_results==bin_gt[:len(self.infer_results),:]).sum(axis=0).cpu().numpy()
        num_data = self.infer_results.size(0)
        num_correct = (prediction_decimal==labels_1Dtensor[:num_data]).sum(axis=0).cpu().numpy()
        
        # print(prediction_decimal.shape)
        # print(labels_1Dtensor.shape)
        # print(correct_per_leg)
        # print(num_data)

        acc = num_correct/num_data
        acc_per_leg = correct_per_leg/num_data
         
        print("Accuracy of leg 0 is: %.4f" % acc_per_leg[0])
        print("Accuracy of leg 1 is: %.4f" % acc_per_leg[1])
        print("Accuracy of leg 2 is: %.4f" % acc_per_leg[2])
        print("Accuracy of leg 3 is: %.4f" % acc_per_leg[3])
        print("Accuracy in terms of class: %.4f" % acc)
        print("Accuracy is: %.4f" % (np.sum(acc_per_leg)/4.0))


def main(args=None):
    rclpy.init(args=args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    parser = argparse.ArgumentParser(description='Test the contcat network')
    parser.add_argument('--config_name', type=str, default=os.path.dirname(os.path.abspath(__file__))+'/../config/inference_one_seq_params.yaml')
    args = parser.parse_args()

    config = yaml.load(open(args.config_name))

    model = contact_cnn()
    checkpoint = torch.load(config['model_load_path'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.eval().to(device)

    bag_subscriber_node = BagSubscriberNode(device=device, config=config, model=model)

    try:
        rclpy.spin(bag_subscriber_node)
    except KeyboardInterrupt:
        pass
    finally:
        bag_subscriber_node.print_infer_results()
        bag_subscriber_node.compute_accuracy()
        # bag_subscriber_node.save2mat()
        bag_subscriber_node.node_destroyer()
        rclpy.shutdown()
        # bag_subscriber_node.destroy_node()
        # rclpy.shutdown()
    # bag_subscriber_node.destroy_node()
    # rclpy.shutdown()

if __name__ == '__main__':
    main()
