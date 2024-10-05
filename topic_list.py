import rosbag

# Define the path to your .bag file
bag_path = 'Our_Data_bag/bc3_hw_t3.bag'

# Open the .bag file
with rosbag.Bag(bag_path, 'r') as bag:
    # Get the list of topics
    topics = bag.get_type_and_topic_info().topics

    # Print the topics
    print("List of topics in the .bag file:")
    for topic in topics:
        print(topic)
