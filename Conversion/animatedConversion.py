import numpy as np

#function to find the maximum value in the dataset
def max_value(data_matrix):
    max=0
    for i in range(1, data_matrix.shape[0]):
        temp= np.max(data_matrix[i,:])
        if(temp>max):
            max=temp
    return max


def convert_data(currentmode, k_value, factor, offset, data_matrix, ax):
    if currentmode == "Voltage":
        gain = k_value * factor
        data_matrix = np.dot(data_matrix, gain) - offset

        # call the function to find the maximum value in the dataset
        max_val = max_value(data_matrix)

        # set the limits of the y axis
        ax.set_ylim(0, max_val + 0.3)

    elif currentmode == "Current":
        gain = k_value * factor
        data_matrix = np.dot(data_matrix, gain) - offset

        # call the function to find the maximum value in the dataset
        max_val = max_value(data_matrix)

        # set the limits of the y axis
        ax.set_ylim(-max_val - 0.3, max_val + 0.3)

    elif currentmode == "Resistance":
        num = factor * 3.3
        den = np.dot(data_matrix, k_value)
        data_matrix = num / den - offset

        # call the function to find the maximum value in the dataset
        max_val = max_value(data_matrix)

        # set the limits of the y axis
        ax.set_ylim(0, max_val + 0.3)

    return data_matrix