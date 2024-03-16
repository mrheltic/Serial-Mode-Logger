from datetime import datetime, timedelta
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from collections import deque
from matplotlib.animation import FuncAnimation
import sys
sys.path.insert(0, './')
import Conversion.conversion as conversion
import DataExtraction.extractforcalibration as extractforcalibration

# Ramp 1: 4V, 100Hz
# Ramp 2: 3.65V, 100Hz
# Ramp 3: 3.55V, 100Hz

data_rate = 860
number_of_rows_to_skip = 1
number_of_seconds = 30

# Setting the parameters for the calibration
amplitude = 3.55
offset = 0

folder = './Dataset/Calibration'

# Extract the array of mean values
mean_values, ramp_values = extractforcalibration.extract_for_calibration(folder, number_of_rows_to_skip, data_rate, number_of_seconds, amplitude, offset)

points = len(mean_values)

# Fit a linear model to the data array period
slope, intercept, r_value, p_value, std_err = stats.linregress(mean_values, ramp_values)

# Plot the data array period and the resulting line
plt.figure(1)
plt.clf()
plt.plot(mean_values, ramp_values, 'go', linewidth=1, markeredgecolor='k', markerfacecolor='g', markersize=10)
plt.plot(mean_values, slope * mean_values + intercept)
plt.title('Linear regression')
plt.xlabel('Mean values')
plt.ylabel('Ramp values')
plt.legend(['Data', 'Resulting line'])
plt.show()

# Clear the terminal
print("\033c")

# Print the slope and intercept with 50 digits precision
print(f"Slope: {slope:.50f}")
print(f"Intercept: {intercept:.50f}\n\n")
