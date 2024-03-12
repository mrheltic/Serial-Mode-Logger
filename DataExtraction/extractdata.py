import numpy as np

def extract_data(datastore):
    # number of array to skip in the dataset
    number_of_rows_to_skip = 1

    # export the current mode
    current_mode = np.loadtxt(datastore, dtype='str', max_rows=1)[-1]

    # export the k value
    k_value = float(np.loadtxt(datastore, dtype='float', usecols=(1), skiprows=1, max_rows=1))

    # export the offset
    offset = np.loadtxt(datastore, dtype='float', usecols=(1), skiprows=2, max_rows=1)

    # export the data rate
    data_rate = int(np.loadtxt(datastore, dtype='int', usecols=(4), skiprows=3, max_rows=1))

    # export the conversion factor
    factor = np.loadtxt(datastore, dtype='float', usecols=(1), skiprows=4, max_rows=1)

    # export the timestamp
    timestamp = np.loadtxt(datastore, dtype='str', usecols=(0), skiprows=5 + number_of_rows_to_skip)

    # export the dataset(reversed) without the 1st array for a problem
    data_matrix = np.loadtxt(datastore, dtype='int', skiprows=5 + number_of_rows_to_skip,
                             usecols=np.arange(1, data_rate + 1), max_rows=len(timestamp) - 1)

    return current_mode, k_value, offset, data_rate, factor, timestamp, data_matrix