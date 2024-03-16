import numpy as np

def extract_ramp(data_array, period, data_rate, amplitude):
    # Calculate points, rounding up to the nearest integer to ensure the period is covered
    points = int(np.ceil(data_rate/period))-1

    # Build the periodical ramp signal
    ramp = np.linspace(0, amplitude, points)

    # Shifting the data array to the right to start from the minimum value
    min_index = np.argmin(data_array)
    data_array_period = data_array[min_index:min_index + points]
    
    return data_array_period, ramp, points