import numpy as np

def extract_for_calibration(datastore1, datastore2, datastore3, datastore4, datastore5):

    #Extract data
    # export the dataset(reversed) without the 1st array for a problem
    data_matrix1 = np.loadtxt(datastore1, dtype='int', skiprows=5 + number_of_rows_to_skip,
                             usecols=np.arange(1, data_rate + 1), max_rows=len(timestamp) - 1)
    
    return mean_values

def mean_from_matrix(data_matrix):
    #declaration of the mean values array
    mean_values = np.concatenate(data_matrix)

    mean_value = np.mean(mean_values)

    return mean_value