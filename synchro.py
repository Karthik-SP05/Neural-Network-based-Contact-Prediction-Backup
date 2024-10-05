import bagpy
from bagpy import bagreader
import pandas as pd

bag = bagreader('Our_Data_bag/bc3_hw_t3.bag')

# Extract CSVs for topics
motor_command_act = bag.message_by_topic('/svan/motor_command_act')
imu_data = bag.message_by_topic('/svan/imu_data_filtered')
actual_feet_data = bag.message_by_topic('/svan/actual_feet_data')

# Read CSV files
motor_command_act = pd.read_csv(motor_command_act)
imu_data = pd.read_csv(imu_data)
actual_feet_data = pd.read_csv(actual_feet_data)


# Convert timestamps to datetime (or use as necessary for your synchronization logic) ['Time'] = pd.to_datfiles['Time'], unit='s')
motor_command_act['Time'] = pd.to_datetime(motor_command_act['Time'], unit='s')
imu_data['Time'] = pd.to_datetime(imu_data['Time'], unit='s')
actual_feet_data['Time'] = pd.to_datetime(actual_feet_data['Time'], unit='s')

# Merge dataframes on time
merged_data = pd.merge_asof(motor_command_act, imu_data, on='Time', direction='nearest')
merged_data = pd.merge_asof(merged_data, actual_feet_data, on='Time', direction='nearest')

# Now merged_data contains synchronized messages, save merged_data to a CSV file
output_file = 'merged_data3.csv'
merged_data.to_csv(output_file, index=False)
print(f"Merged data has been successfully written to {output_file}")