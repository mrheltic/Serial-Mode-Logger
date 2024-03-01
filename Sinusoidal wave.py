from datetime import datetime, timedelta
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from collections import deque
from matplotlib.animation import FuncAnimation

datastore = './Dataset/sinusoidal wave 200hz.txt'

# Create a sinusoidal wave from 0 to 4, 200hz
amplitude = 3.55
offset = 1.775
period = 100


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

# Remove the first column from matrix (various errors)
data_matrix = data_matrix[5:]

#create a 1D array from the matrix
data_array = np.concatenate(data_matrix)

#apply the conversion factor
gain=k_value*factor
data_matrix = np.dot(data_matrix,gain)-offset

# Flatten the matrix
data_array = data_matrix.flatten()

# Build the periodical sinusoidal signal considering the amplitude and offset
sinusoidal = amplitude * np.sin(np.linspace(0, 2*np.pi, period)) + offset

# Extract a period from the data array, from the max value to the next max value
# This is done to find the period of the signal
min_index = np.argmin(data_array)
data_array_period = data_array[min_index:min_index + period]

# Plot one iteration of data array period and compare it to the generated sinusoidal signal, until the period is found
fig, ax = plt.subplots()
ax.plot(data_array_period)
ax.plot(sinusoidal)
plt.show()

# Plot the data array
#fig, ax = plt.subplots()
#ax.plot(data_array)
#plt.show()
