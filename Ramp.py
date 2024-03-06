from datetime import datetime, timedelta
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from collections import deque
from matplotlib.animation import FuncAnimation

# Ramp 1: 4V, 100Hz
# Ramp 2: 3.65V, 100Hz
# Ramp 3: 3.55V, 100Hz

datastore = './Dataset/ramp3.txt'

# Number of columns to skip in the data array
number_of_columns_to_skip = 6;

# Create a ramp from 0 to 4, with 100Hz
amplitude = 3.55
period = 10

#export the current mode
currentmode=np.loadtxt(datastore, dtype='str', max_rows=1)[-1]

#export the k value
k_value=float(np.loadtxt(datastore, dtype='float', usecols=(1), skiprows=1,max_rows=1))

#export the offset
offset=np.loadtxt(datastore, dtype='float', usecols=(1), skiprows=2, max_rows=1)

#export the data rate
data_rate=int(np.loadtxt(datastore, dtype='int', usecols=(4), skiprows=3, max_rows=1))

#export the conversion factor
factor=1

#export the timestamp
timestamp = np.loadtxt(datastore, dtype='str', usecols=(0), skiprows=6)

#export the dataset(reversed) without the 1st array for a problem
data_matrix = np.loadtxt(datastore, dtype='int', skiprows=6, usecols=np.arange(1, data_rate+1),max_rows=len(timestamp)-1)

# Remove the first number n column from matrix (various errors)
data_matrix = data_matrix[number_of_columns_to_skip:]

#create a 1D array from the matrix
data_array = np.concatenate(data_matrix)

#apply the conversion factor
gain=k_value*factor
data_matrix = np.dot(data_matrix,gain)-offset;

# Flatten the matrix
data_array = data_matrix.flatten()

# Calculate points, rounding up to the nearest integer to ensure the period is covered
points = int(np.ceil(data_rate/period))

# Build the periodical ramp signal
ramp = np.linspace(0, amplitude, points)

# Shifting the data array to the right to start from the minimum value
min_index = np.argmin(data_array)
data_array_period = data_array[min_index:min_index + points]

# Plot one iteration of data array period and compare it to the generated ramp signal, until the period is found
fig, (ax1, ax2) = plt.subplots(2, 1)

# Plot data array period and ramp signal
ax1.plot(data_array_period)
ax1.plot(ramp)

# Adding labels
ax1.set_xlabel('Sample number')
ax1.set_ylabel('Amplitude (V)')

# Adding title and legend
ax1.set_title('Ramp signal')
ax1.legend(['Data array period', 'Ramp signal'])

# Plot the error between the data array period and the ramp signal
ax2.plot((data_array_period - ramp)**2)

# Adding labels
ax2.set_xlabel('Sample number')
ax2.set_ylabel('Squared Error')

# Adding title
ax2.set_title('Squared Error between data array period and ramp signal')

plt.tight_layout()
plt.show()
