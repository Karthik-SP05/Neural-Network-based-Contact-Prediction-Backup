import rosbag

# Define the path to your .bag file and the topic of interest
bag_path = 'Our_Data_bag/2024-08-09-11-06-49.bag'
topic_of_interest = '/svan/imu_data'  # Replace with your topic

# Function to recursively get fields of a message
def get_fields(msg, parent_name=''):
    fields = []
    for slot in msg.__slots__:
        field_name = f"{parent_name}.{slot}" if parent_name else slot
        field = getattr(msg, slot)
        if hasattr(field, '__slots__'):  # Nested message
            fields.extend(get_fields(field, field_name))
        else:
            fields.append(field_name)
    return fields

# Open the .bag file
with rosbag.Bag(bag_path, 'r') as bag:
    for topic, msg, t in bag.read_messages(topics=[topic_of_interest]):
        # Get the fields of the message
        fields = get_fields(msg)
        
        # Print the fields
        print(f"Fields in the topic '{topic_of_interest}':")
        for field in fields:
            print(field)
        
        # Break after the first message for demonstration
        break
