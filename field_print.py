import rosbag
import rospy

# Define the path to your .bag file and the topic/field you are interested in
bag_path = 'Our Data (bag)/bc2_hw_t8.bag'
topic_of_interest = '/svan/feet_est'
field_of_interest = 'data'  # Replace with the name of the field you want to print

# Open the .bag file
bag = rosbag.Bag(bag_path)

def print_field(msg, field_path):
    """ Recursively print the value of the specified field in the message. """
    fields = field_path.split('.')
    field_value = msg
    for field in fields:
        field_value = getattr(field_value, field, None)
        if field_value is None:
            break
    return field_value

 #Iterate through the messages in the .bag file
for topic, msg, t in bag.read_messages(topics=[topic_of_interest]):
     #Print the value of the specific field
    field_value = print_field(msg, field_of_interest)
    if field_value is not None:
        print(f"Time: {t.to_sec()}, {field_of_interest}: {field_value}")
    else:
        print(f"Time: {t.to_sec()}, {field_of_interest}: Field not found")
