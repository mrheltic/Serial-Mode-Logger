import numpy as np

def extract_period(data_array, period, data_rate, amplitude):

    points = int(np.ceil(data_rate/period))

    # Build the periodical sinusoidal signal considering the amplitude and offset
    sinusoidal = amplitude * np.sin(np.linspace(0, 2*np.pi, points))

    # Find the period in the data array and extract it
    max_index = np.argmax(data_array)
    data_array_period = data_array[max_index:max_index + round(2*points)]

    # Shift the data array to right so that the period starts at the beginning of the array
    data_array_period = np.roll(data_array_period, -abs(np.argmin(data_array_period)))

    for i in range(0, len(data_array_period)):
        if data_array_period[i]*data_array_period[i-1] < 0:
            start_data_array_period = i-1
            break

    # Remove the first part of the data array period
    data_array_period = data_array_period[start_data_array_period:]

    for i in range(int(0.9*points), len(data_array_period)):
        if data_array_period[i]*data_array_period[i-1] < 0:
            end_data_array_period = i-1
            break

    # Implement bisect to iterate and find
    tolerance = 1e-20

    def find_sign_change(data_array_period, start, end, tolerance):
        while end - start > 1:
            mid = (start + end) // 2
            if data_array_period[mid] * data_array_period[start] < 0:
                end = mid
            else:
                start = mid
        return start

    start_data_array_period = find_sign_change(data_array_period, 0, int(0.9*points), tolerance)
    end_data_array_period = find_sign_change(data_array_period, int(0.9*points), len(data_array_period), tolerance)

    data_array_period = data_array_period[:end_data_array_period+(sinusoidal.size - end_data_array_period)]

    return data_array_period, sinusoidal