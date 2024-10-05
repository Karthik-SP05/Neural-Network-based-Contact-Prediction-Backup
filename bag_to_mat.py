import rosbag
from scipy.io import savemat
import numpy as np
import pandas as pd

# Load the ROS Bag file
#bag = rosbag.Bag('Our Data (bag)/bc2_hw_t7.bag')  # Replace 'your_bag_file.bag' with the actual file name

# Initialize lists or dictionaries to store data
# For example, if you're interested in storing sensor data, initialize lists to store timestamps and sensor readings
timestamps = []
imu_omega = []
imu_acc = []
q = []
qd = []
p = []
L = 0.4035
W = 0.138
# Iterate over messages in the bag file
#for topic, msg, t in bag.read_messages():
    # Process messages based on your requirements
    # For example, if you're interested in sensor data from a specific topic:
    #if topic == '/svan/imu_data':
        #timestamps.append(msg.header.stamp.to_sec())  # Convert ROS time to seconds
        #imu_omega.append([msg.angularRate.x, msg.angularRate.y, msg.angularRate.z])
        #imu_acc.append([msg.acceleration.x, msg.acceleration.y, msg.acceleration.z])  # Assuming sensor data is stored in msg.data1, msg.data2, msg.data3

excel_file1 = 'Our Data (xlsx)/data7/final(2).xlsx'  # Replace with your actual file name

df = pd.read_excel(excel_file1)
df2 = pd.to_datetime(df['Time'], format='%Y-%m-%d %H:%M:%S.%f')
start_time = df2.iloc[0]
df2 = (df2 - start_time).dt.total_seconds()
timestamps_array = df2.values
# timestamps_array = timestamps_array[5000:9500]

df1 = pd.read_excel(excel_file1, usecols='AD:AF')
df2 = pd.read_excel(excel_file1, usecols='AX:AZ')
df = pd.read_excel(excel_file1, usecols='D:O')

# Initialize an empty list to store the results
p = []
i = 0
while i < 4:
    df.iloc[:,3*i] = df.iloc[:,3*i] - df1.iloc[:,0]
    df.iloc[:,(3*i)+1] = df.iloc[:,(3*i)+1] - df1.iloc[:,1]
    df.iloc[:,(3*i)+2] = df.iloc[:,(3*i)+2] - df1.iloc[:,2]
    if i==0 :
        df.iloc[:,3*i] -= (L/2)
        df.iloc[:,(3*i)+1] += (W/2)
    if i==1 :
        df.iloc[:,3*i] -= (L/2)
        df.iloc[:,(3*i)+1] -= (W/2)
    if i==2 :
        df.iloc[:,3*i] += (L/2)
        df.iloc[:,(3*i)+1] -= (W/2)
    if i==3 :
        df.iloc[:,3*i] += (L/2)
        df.iloc[:,(3*i)+1] += (W/2)
    i += 1
# Iterate over the rows
for index, row in df.iterrows():
    row_list = []
    # Iterate over the columns in the current row
    for col in df.columns:
        row_list.append(row[col])
    p.append(row_list)
#excel_file2 = 'Our Data (xlsx)/data7(without commas)/_svan_motor_command_act.velocity_separated.xlsx'  # Replace with your actual file name
df = pd.read_excel(excel_file1, usecols='P:AA',skiprows=4999, nrows=4500)

# Initialize an empty list to store the results
v = []
i = 0
for i in range(4):
    df.iloc[:,3*i] = df.iloc[:,3*i] - df2.iloc[:,0]
    df.iloc[:,(3*i)+1] = df.iloc[:,(3*i)+1] - df2.iloc[:,1]
    df.iloc[:,(3*i)+2] = df.iloc[:,(3*i)+2] - df2.iloc[:,2]
    i += 1
# Iterate over the rows
for index, row in df.iterrows():
    row_list = []
    # Iterate over the columns in the current row
    for col in df.columns:
        row_list.append(row[col])
    v.append(row_list)

#excel_file3 = 'Our Data (xlsx)/data7(without commas)/_svan_feet_est.position.xlsx'  # Replace with your actual file name
df = pd.read_excel(excel_file1, usecols='DD:DF',skiprows=4999, nrows=4500)

# Initialize an empty list to store the results
imu_omega = []

# Iterate over the rows
for index, row in df.iterrows():
    row_list = []
    # Iterate over the columns in the current row
    for col in df.columns:
        row_list.append(row[col])
    imu_omega.append(row_list)

#excel_file4 = 'Our Data (xlsx)/data7(without commas)/_svan_feet_est.velocity.xlsx'  # Replace with your actual file name
df = pd.read_excel(excel_file1,usecols='DM:DO',skiprows=4999, nrows=4500)

# Initialize an empty list to store the results
imu_acc = []

# Iterate over the rows
for index, row in df.iterrows():
    row_list = []
    # Iterate over the columns in the current row
    for col in df.columns:
        row_list.append(row[col])
    imu_acc.append(row_list)

df = pd.read_excel(excel_file1,usecols='BQ:CB', skiprows=4999, nrows=4500)

# Initialize an empty list to store the results
q = []

# Iterate over the rows
for index, row in df.iterrows():
    row_list = []
    # Iterate over the columns in the current row
    for col in df.columns:
        row_list.append(row[col])
    q.append(row_list)

df = pd.read_excel(excel_file1,usecols='CC:CN',skiprows=4999, nrows=4500)

# Initialize an empty list to store the results
qd = []

# Iterate over the rows
for index, row in df.iterrows():
    row_list = []
    # Iterate over the columns in the current row
    for col in df.columns:
        row_list.append(row[col])
    qd.append(row_list)

input_file = 'Our Data (xlsx)/data54/data54(2)/contacts54(2).xlsx'

# Read the Excel file into a DataFrame
df = pd.read_excel(input_file, header=None)

# Convert the DataFrame to a 2D array
contacts_array = df.values

# Convert lists to NumPy arrays
imu_omega_array = np.array(imu_omega)
imu_acc_array = np.array(imu_acc)
p_array = np.array(p)
q_array = np.array(q)
qd_array = np.array(qd)
v_array = np.array(v)

# Save data as a .mat file
output_file = 'output(5).mat'  # Replace with your desired output file name
savemat(output_file, {'timestamps': timestamps_array, 'imu_acc': imu_acc_array, 'p': p_array, 'q': q_array, 'qd': qd_array, 'imu_omega': imu_omega_array, 'v': v_array, 'contacts': contacts_array})

print(f"Data has been successfully saved to {output_file}")
shape = q_array.shape
print(f"Shape of the q_array: {shape}")
size = q_array.size
print(f"Total number of elements in q_array: {size}")
length = len(q_array)
print(f"Number of rows in q_array: {length}")

shape = contacts_array.shape
print(f"Shape of the contacts_array: {shape}")
size = contacts_array.size
print(f"Total number of elements in contacts_array: {size}")
length = len(contacts_array)
print(f"Number of rows in contacts_array: {length}")

# Close the bag file
bag.close()
