import os
import numpy as np

def extract_for_calibration(data_folder, number_of_rows_to_skip, data_rate, number_of_seconds, amplitude, offset):

    # Get a list of all files in the data folder
    file_list = os.listdir(data_folder)

    # Initialize an empty list to store the data matrices
    data_matrices = []
    mean_values = []

    # Iterate over each file in the folder
    for file_name in file_list:
        # Construct the full file path
        file_path = os.path.join(data_folder, file_name)

        # Check if the file exists
        if os.path.isfile(file_path):
            # export the timestamp
            timestamp = np.loadtxt(file_path, dtype='str', usecols=(0), skiprows=5 + number_of_rows_to_skip)

            # Load the data matrix from the file
            data_matrix = np.loadtxt(file_path, dtype='int', skiprows=5 + number_of_rows_to_skip,
                                     usecols=np.arange(1, data_rate + 1), max_rows=len(timestamp) - 1)

            # Calculate the number of samples to keep
            number_of_samples = number_of_seconds * data_rate

             # Flatten the data matrix
            data_matrix = data_matrix.flatten()

            # Keep only the first number_of_samples samples
            data_matrix = data_matrix[:number_of_samples]

            # Append the mean value to the list of mean values
            mean_values.append(np.mean(data_matrix))

            # Cal
        else:
            print(f"File not found: {file_path}")

    # Convert the list of mean values to a numpy array
    mean_values = np.array(mean_values)

    # Sort the mean values
    mean_values.sort()

    # Create the ramp
    ramp_values = np.linspace(offset, amplitude, len(mean_values))
    return mean_values, ramp_values