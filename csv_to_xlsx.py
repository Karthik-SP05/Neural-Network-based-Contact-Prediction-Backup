import pandas as pd

# Read the CSV file into a pandas DataFrame
df = pd.read_csv('merged_data3.csv')

# Convert DataFrame to Excel
df.to_excel('final3.xlsx', index=False)
