import numpy as np
from scipy.signal import find_peaks
from scipy.signal import argrelextrema
import pandas as pd
from scipy.signal import butter, filtfilt
from datetime import datetime
import matplotlib.pyplot as plt

df = pd.read_excel('Our Data (xlsx)/data3(2)/final3.xlsx',usecols='AW:BH')
df1 = pd.read_excel('Our Data (xlsx)/data3(2)/final3.xlsx')
# num_data = 23000
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

df2 = pd.to_datetime(df1['Time'], format='%Y-%m-%d %H:%M:%S.%f')
df2 = df2.iloc[6500:16200]
start_time = df2.iloc[0]
df2 = (df2 - start_time).dt.total_seconds()
time_array = df2.values
time_interval = np.mean(np.diff(time_array))
fs = 1 / time_interval
#print(fs)
#print(time_array)
contacts = np.zeros((df2.shape[0], num_legs), dtype=bool)
l = 0
for l in range(4) :
    foot_height = df.iloc[6500:16200,(3*(l+1))-1].to_numpy()#2+(3*(l+1))

    # plt.figure(figsize=(12, 6))
    # plt.plot(time_array, foot_height, label='Filtered Signal', linewidth=2)
    # plt.title('Unfiltered data')
    # plt.xlabel('Time [seconds]')
    # plt.ylabel('Amplitude')
    # plt.legend()
    # plt.grid()
    # plt.show()
    # Set different half power frequency based on gait
    if gait == 'trot':
        half_power_freq = 20
    elif gait in ['pronking', 'gallop']:
        half_power_freq = 80

    # Apply low pass filter to the foot height data
    foot_height = low_pass_filter(foot_height, half_power_freq,fs)

    # plt.figure(figsize=(12, 6))
    # plt.plot(time_array, foot_height, label='Filtered Signal', linewidth=2)
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
    print(max_idx)
    print(min_idx)
    # print(foot_height[1501])
    # print(foot_height[1560])
    # print(foot_height[1589])

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
        # if j!=0 :
        #     previous_peak = max_idx[j-1]
        # else :
        #     previous_peak = 1

        # Connect all the local minimum before next peak
        count = 0
        if i==0 and min_idx[i]>max_idx[0] :
            contact_end = min_idx[i]
            diff = round((min_idx[i] - max_idx[0])/3)
            contact_start = min_idx[i] - diff #modify to contact_start = min_idx[i]-40 (if min_idx[i]<40 set contact_start = 1)
            i += 1
        
        while i < num_min and min_idx[i] < next_peak:
            contact_end = min_idx[i]
            if(i!=0): diff = round((min_idx[i] - max_idx[i-1])/3)
            else: diff = 30
            i += 1
            count += 1
        

        # If only one local minimum is found between two peaks, set a conservative amount of data beforehand as contact
        if count == 1:
            # print(i)
            # if i!=0 :
            # print(i)
            contact_start = contact_end - diff #modify to contact_start = contact_end-40
            # else :
            #     contact_start = contact_end - 40
            if contact_start<1 :
                contact_start = 1
        
        # Mark the contact period
        contacts[contact_start:contact_end, l] = True
        j += 1
print(contacts)
df = pd.DataFrame(contacts)
output_file = 'contacts54(2).xlsx'
df.to_excel(output_file, index=False, header=False)
print(f"2D array has been successfully written to {output_file}")
    