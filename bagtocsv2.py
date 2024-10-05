import rosbag
import csv
import openpyxl

# Specify the path to the .bag file
bag_file_path = 'Our Data (bag)/bc2_hw_t7.bag'

# Open the bag file
bag = rosbag.Bag(bag_file_path)

# Specify the topics and fields you want to extract
topics_and_fields = {
    '/svan/imu_data': ['angularRate.x','angularRate.y','angularRate.z','acceleration.x','acceleration.y','acceleration.z'],
    '/svan/motor_command_act': ['position','velocity'],
    '/svan/feet_est': ['data'],
}

# Prepare a dictionary to hold the data
data = {topic: {field: [] for field in fields} for topic, fields in topics_and_fields.items()}

# Read the messages from the bag file
for topic, msg, t in bag.read_messages(topics=topics_and_fields.keys()):
    if topic in topics_and_fields:
        for field in topics_and_fields[topic]:
            value = eval(f'msg.{field}')  # Extract the field value dynamically
            data[topic][field].append(value)

# Close the bag file
bag.close()
#for topic, fields in topics_and_fields.items():
# Save the extracted data to CSV files

#csv save
for topic, fields in topics_and_fields.items():
   for field in topics_and_fields[topic]: 
      csv_file_path = f'{topic.replace("/", "_")}.{field}.csv'
      with open(csv_file_path, 'w', newline='') as csvfile:
          csvwriter = csv.writer(csvfile)
          # Write the header
          csvwriter.writerow(['timestamp']+ [field])
          # Write the data
          for i in range(len(data[topic][fields[0]])):
              row = [t.to_sec()]  # Add the timestamp
              row.append(data[topic][field][i])
              csvwriter.writerow(row)

print("Data successfully written to CSV files.")
