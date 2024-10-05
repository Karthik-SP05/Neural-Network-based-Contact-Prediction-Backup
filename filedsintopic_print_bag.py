#!/usr/bin/env python3

import rospy
import rosbag
from std_msgs.msg import String  # Replace with your message type

def print_topic_fields(bag_file, topic):
    rospy.init_node('rosbag_reader', anonymous=True)
    
    # Open the bag file
    with rosbag.Bag(bag_file, 'r') as bag:
        # Iterate over messages in the bag file
        for _, msg, _ in bag.read_messages(topics=[topic]):
            # Print the timestamp of the message
            rospy.loginfo(f'Message timestamp: {msg.header.stamp}')

            # Print all fields in the message (example for std_msgs/String)
            if isinstance(msg, String):
                rospy.loginfo(f'String message data: {msg.data}')
            else:
                # Modify for other message types as needed
                rospy.loginfo(f'Other message type: {msg}')

if __name__ == '__main__':
    # Specify your bag file path
    bag_file = 'Our_Data_bag/2024-06-26-18-41-54.bag'

    # Specify the topic of interest
    topic = '/svan/imu_data '  # Replace with your topic name

    try:
        print_topic_fields(bag_file, topic)
    except rospy.ROSInterruptException:
        pass
