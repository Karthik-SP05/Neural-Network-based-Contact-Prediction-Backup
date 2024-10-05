import scipy.io
import pandas as pd

# Load the .mat file
mat_file = scipy.io.loadmat('/home/karthik/NN Based Contact Prediction2/Datasets/drive-download-20240605T114221Z-001/grass/mat/grass.mat')

# Print the variable names to identify the 2D array
print(mat_file.keys())

# Assuming the 2D array is stored in a variable called 'your_variable'
data = mat_file['contacts']

# Convert the 2D array to a DataFrame
df = pd.DataFrame(data)

# Save the DataFrame to an Excel file
output_file = 'output.xlsx'
df.to_excel(output_file, index=False, header=False)

print(f"Data has been successfully written to {output_file}")
