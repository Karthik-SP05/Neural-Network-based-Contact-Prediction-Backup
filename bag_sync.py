import bagpy
from bagpy import bagreader
import pandas as pd

bag = bagreader('/Users/amritanshumanu/Documents/PhD/research/codebase/svan_sw/m2_cs_est_data_trot_x_03_ms.bag')

# Extract CSVs for topics
fc_est = bag.message_by_topic('/svan/fc_est')
pc_ref = bag.message_by_topic('/svan/contact_prob_ref')
x_hat = bag.message_by_topic('/svan/x_hat')
cs_FR = bag.message_by_topic('/svan/contact_state_FR')
cs_FL = bag.message_by_topic('/svan/contact_state_FL')
cs_RL = bag.message_by_topic('/svan/contact_state_RL')
cs_RR = bag.message_by_topic('/svan/contact_state_RR')

# Read CSV files
fc_est = pd.read_csv(fc_est)
pc_ref = pd.read_csv(pc_ref)
x_hat = pd.read_csv(x_hat)
cs_FR = pd.read_csv(cs_FR)
cs_FL = pd.read_csv(cs_FL)
cs_RL = pd.read_csv(cs_RL)
cs_RR = pd.read_csv(cs_RR)

# Convert timestamps to datetime (or use as necessary for your synchronization logic)
fc_est['Time'] = pd.to_datetime(fc_est['Time'], unit='s')
pc_ref['Time'] = pd.to_datetime(pc_ref['Time'], unit='s')
x_hat['Time'] = pd.to_datetime(x_hat['Time'], unit='s')
cs_FR['Time'] = pd.to_datetime(cs_FR['Time'], unit='s')
cs_FL['Time'] = pd.to_datetime(cs_FL['Time'], unit='s')
cs_RL['Time'] = pd.to_datetime(cs_RL['Time'], unit='s')
cs_RR['Time'] = pd.to_datetime(cs_RR['Time'], unit='s')

# Merge dataframes on time
merged_data = pd.merge_asof(fc_est, pc_ref, on='Time', direction='nearest')
merged_data = pd.merge_asof(merged_data, x_hat, on='Time', direction='nearest')
merged_data = pd.merge_asof(merged_data, cs_FR, on='Time', direction='nearest')
merged_data = pd.merge_asof(merged_data, cs_FL, on='Time', direction='nearest')
merged_data = pd.merge_asof(merged_data, cs_RL, on='Time', direction='nearest')
merged_data = pd.merge_asof(merged_data, cs_RR, on='Time', direction='nearest')

# Now `merged_data` contains synchronized messages, save merged_data to a CSV file
output_file = 'merged_data.csv'
merged_data.to_csv(output_file, index=False)

print(f"Merged data has been successfully written to {output_file}")