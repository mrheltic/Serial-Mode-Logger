import numpy as np

def convert_data(currentmode, k_value, factor, offset, data_matrix):
    if currentmode == "Voltage":
        gain = k_value * factor
        data_matrix = np.dot(data_matrix, gain) - offset


    elif currentmode == "Current":
        gain = k_value * factor
        data_matrix = np.dot(data_matrix, gain) - offset

    elif currentmode == "Resistance":
        num = factor * 3.3
        den = np.dot(data_matrix, k_value)
        data_matrix = num / den - offset

    data_array = np.concatenate(data_matrix)  # create a 1D array from the matrix

    return data_matrix, data_array