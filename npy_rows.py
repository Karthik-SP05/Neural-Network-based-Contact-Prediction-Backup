import numpy as np

def check_npy_rows(file_path):
    # Load the .npy file
    data = np.load(file_path)
    
    # Check the number of rows (assuming it's a 2D array)
    if len(data.shape) == 2:
        print(f'Number of rows: {data.shape[0]}')
    else:
        print('The array is not 2D, so no rows to count.')
    
# Example usage
file_path = '/home/karthik/NN_Based_Contact_Prediction2/Models and Data/Processed Data/Our Processed Data/Data3(inference(2))model49/output3_inference_label.npy'
check_npy_rows(file_path)