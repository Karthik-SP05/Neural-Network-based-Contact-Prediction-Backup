import rosbag

# Define the path to your .bag file and the topic of interest
bag_path = 'Our_Data_bag/bc3_hw_t3.bag'
topic_of_interest = 'imu_data'  # Replace with your topic

# Initialize the message count
message_count = 0

# Open the .bag file
with rosbag.Bag(bag_path, 'r') as bag:
    for topic, msg, t in bag.read_messages(topics=[topic_of_interest]):
        message_count += 1

print(f"The topic '{topic_of_interest}' contains {message_count} messages.")