import numpy as np
import pandas as pd

# Specify the path to your .npy file
npy_file_path = '/home/karthik/NN Based Contact Prediction2/Files/deep-contact-estimator/Processed Data/Our Processed Data/Data7(3)/train_label.npy'
xlsx_file_path = 'data7_train_label.xlsx'

# Load the .npy file
data = np.load(npy_file_path)

# Convert the NumPy array to a pandas DataFrame
df = pd.DataFrame(data)

# Save the DataFrame as an .xlsx file
df.to_excel(xlsx_file_path, index=False)

print(f'File saved as {xlsx_file_path}')
