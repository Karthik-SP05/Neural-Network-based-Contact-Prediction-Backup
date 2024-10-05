from scipy.signal import butter, filtfilt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_excel('Our Data (xlsx)/data54/final54.xlsx')
print(df.shape)
#df1 = pd.read_excel('Our Data (xlsx)/data54/final54.xlsx',usecols='AP:AR')
#df2 = pd.read_excel('Our Data (xlsx)/data54/final54.xlsx', usecols='AY:BA')
half_power_freq = 20
df3 = pd.to_datetime(df['Time'], format='%Y-%m-%d %H:%M:%S.%f')
start_time = df3.iloc[0]
df3 = (df3 - start_time).dt.total_seconds()
time_array = df3.values
time_interval = np.mean(np.diff(time_array))
fs = 1 / time_interval
print(fs)

def low_pass_filter(input_array, half_power_freq, fs):
    # Calculate the cutoff frequency (3 dB point) for the filter
    cutoff_freq = half_power_freq / (2 * np.pi)

    # Design the filter coefficients
    order = 2  # Choose the filter order (e.g., 1 for first-order, 2 for second-order)
    b, a = butter(order, cutoff_freq / (fs / 2), btype='low')  # Design the filter

    # Apply the filter to the input array using filtfilt to achieve zero-phase filtering
    filtered_output = filtfilt(b, a, input_array)

    return filtered_output

i = 41
while(i<44) :
    #imu_data_arrray = df1.iloc[:,i].to_numpy()
    # plt.figure(figsize=(12, 6))
    # plt.plot(time_array, df.iloc[:,i].to_numpy(), label='Filtered Signal', linewidth=2)
    # plt.title('Unfiltered imu_data')
    # plt.xlabel('Time [seconds]')
    # plt.ylabel('Amplitude')
    # plt.legend()
    # plt.grid()
    # plt.show()
    df.iloc[:,i] = low_pass_filter(df.iloc[:,i].to_numpy(),half_power_freq,fs)
    i += 1
    # break
    #df.iloc[:,i] = imu_data_array

i = 50
while(i<53) :
    #imu_data_arrray = df1.iloc[:,i].to_numpy()
    df.iloc[:,i] = low_pass_filter(df.iloc[:,i].to_numpy(),half_power_freq,fs)
    i += 1

# plt.figure(figsize=(12, 6))
# plt.plot(time_array, df.iloc[:,41].to_numpy(), label='Filtered Signal', linewidth=2)
# plt.title('Filtered Signal Using Low-pass Filter')
# plt.xlabel('Time [seconds]')
# plt.ylabel('Amplitude')
# plt.legend()
# plt.grid()
# plt.show()



output_file = 'final54(2).xlsx'
df.to_excel(output_file, index=False)
print(f'dataframe has been successfully written to {output_file}')