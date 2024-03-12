from datetime import datetime, timedelta
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from collections import deque
from matplotlib.animation import FuncAnimation
import Conversion.conversion as conversion
import DataExtraction.extractdata as extractdata
import DataExtraction.extractramp as extractramp

# Ramp 1: 4V, 100Hz
# Ramp 2: 3.65V, 100Hz
# Ramp 3: 3.55V, 100Hz

datastore = './Dataset/Ramp/ramp3.txt'

# Create a ramp from 0 to 4, with 100Hz
amplitude = 3.55
period = 10

# Extract the data
currentmode, k_value, offset, data_rate, factor, timestamp, data_matrix = extractdata.extract_data(datastore)

# Set the conversion factor to 1 if you're using the FSR of ADC
factor = 1

# Convert the data
data_matrix, data_array = conversion.convert_data(currentmode, k_value, factor, offset, data_matrix)

data_array_period, ramp, points = extractramp.extract_ramp(data_array, period, data_rate, amplitude)

# Fit a lineer model to the data array period
slope, intercept, r_value, p_value, std_err = stats.linregress(np.arange(points), data_array_period)

# Fit a lineer model to the ramp signal
slope_ramp, intercept_ramp, r_value_ramp, p_value_ramp, std_err_ramp = stats.linregress(np.arange(points), ramp)

# Plot one iteration of data array period and compare it to the generated ramp signal, until the period is found
fig, (ax1, ax2) = plt.subplots(2, 1)

# Plot the linear model of the data array period and the ramp signal
ax1.plot(data_array_period)
ax1.plot(slope * np.arange(len(data_array_period)) + intercept)
ax1.plot(ramp)
ax1.plot(slope_ramp * np.arange(len(data_array_period)) + intercept_ramp)

# Adding labels
ax1.set_xlabel('Sample number')
ax1.set_ylabel('Amplitude (V)')

# Adding title and legend
ax1.set_title('Ramp signal')
ax1.legend(['Data array period', 'Ramp signal'])

# Plot the error between the data array period and the ramp signal both with linear model
ax2.plot((slope_ramp * np.arange(len(data_array_period)) + intercept_ramp) - (slope * np.arange(points) + intercept))

# Adding labels
ax2.set_xlabel('Sample number')
ax2.set_ylabel('Error')

# Adding title
ax2.set_title(' Error between data array period and ramp signal')

plt.tight_layout()
plt.show()
