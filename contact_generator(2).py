import numpy as np
from scipy.signal import find_peaks
from scipy.signal import argrelextrema
import pandas as pd
from scipy.signal import butter, filtfilt
from datetime import datetime
import matplotlib.pyplot as plt

df = pd.read_excel('Our Data (xlsx)/data8/final8.xlsx')
num_data = 81868
num_legs = 4
gait = 'trot'

def low_pass_filter(input_array, half_power_freq, fs):
    # Calculate the cutoff frequency (3 dB point) for the filter
    cutoff_freq = half_power_freq / (2 * np.pi)

    # Design the filter coefficients
    order = 2  # Choose the filter order (e.g., 1 for first-order, 2 for second-order)
    b, a = butter(order, cutoff_freq / (fs / 2), btype='low')  # Design the filter

    # Apply the filter to the input array using filtfilt to achieve zero-phase filtering
    filtered_output = filtfilt(b, a, input_array)

    return filtered_output

df2 = pd.to_datetime(df['Time'], format='%Y-%m-%d %H:%M:%S.%f')
start_time = df2.iloc[0]
df2 = (df2 - start_time).dt.total_seconds()
time_array = df2.values
time_interval = np.mean(np.diff(time_array))
fs = 1 / time_interval
#print(fs)
#print(time_array)
contacts = np.zeros((num_data, num_legs), dtype=bool)

for l in range(4):
    foot_height = df.iloc[:,2+(3*(l+1))].to_numpy()#2+(3*(l+1))

    # plt.figure(figsize=(12, 6))
    # plt.plot(time_array[10000:15000], foot_height[10000:15000], label='Filtered Signal', linewidth=2)
    # plt.title('Unfiltered data')
    # plt.xlabel('Time [seconds]')
    # plt.ylabel('Amplitude')
    # plt.legend()
    # plt.grid()
    # plt.show()
    # Set different half power frequency based on gait
    if gait == 'trot':
        half_power_freq = 23
    elif gait in ['pronking', 'gallop']:
        half_power_freq = 80

    # Apply low pass filter to the foot height data
    foot_height = low_pass_filter(foot_height, half_power_freq,fs)

    # plt.figure(figsize=(12, 6))
    # plt.plot(time_array[10000:15000], foot_height[10000:15000], label='Filtered Signal', linewidth=2)
    # plt.title('Filtered Signal Using Low-pass Filter')
    # plt.xlabel('Time [seconds]')
    # plt.ylabel('Amplitude')
    # plt.legend()
    # plt.grid()
    # plt.show()
    # break


    # Extract local maxima and minima
    #local_max = is_local_max(filtered_foot_height)
    #local_min = is_local_min(filtered_foot_height)
    # Extract indices from local maxima and minima
    max_idx = argrelextrema(foot_height, np.greater)[0]
    min_idx = argrelextrema(foot_height, np.less)[0]
    #print(max_idx)
    #print(min_idx)

    num_max = len(max_idx)
    num_min = len(min_idx)
    
    i = 0
    j = 0
    if min_idx[0]>max_idx[0] :
        i = 0
        j = 1
        

    # Process each minimum and maximum
    while i < num_min and j < num_max:
        contact_start = min_idx[i]
        next_peak = max_idx[j]

        # Connect all the local minimum before next peak
        count = 0
        if i==0 and min_idx[i]>max_idx[0] :
            contact_end = min_idx[i]
            contact_start = max_idx[0]+1
            i += 1
        
        while i < num_min and min_idx[i] < next_peak:
            contact_end = min_idx[i]
            i += 1
            count += 1

        # If only one local minimum is found between two peaks, set a conservative amount of data beforehand as contact
        if count == 1:
            contact_start = contact_end - 80
            if contact_start < 0:
                contact_start = 0
        # Mark the contact period
        contacts[contact_start:contact_end, l] = True
        j += 1
#print(contacts)
df = pd.DataFrame(contacts)
output_file = 'contacts(3).xlsx'
df.to_excel(output_file, index=False, header=False)
print(f"2D array has been successfully written to {output_file}")
    