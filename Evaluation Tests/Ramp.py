from datetime import datetime, timedelta
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from collections import deque
from matplotlib.animation import FuncAnimation

datastore = '../dataStorage.txt'


#export the current mode
currentmode=np.loadtxt(datastore, dtype='str', max_rows=1)[-1]

#export the k value
k_value=float(np.loadtxt(datastore, dtype='float', usecols=(1), skiprows=1,max_rows=1))

#export the offset
offset=np.loadtxt(datastore, dtype='float', usecols=(1), skiprows=2, max_rows=1)

#export the data rate
data_rate=int(np.loadtxt(datastore, dtype='int', usecols=(4), skiprows=3, max_rows=1))

#export the conversion factor
factor=np.loadtxt(datastore, dtype='float', usecols=(1), skiprows=4, max_rows=1)

#export the timestamp
timestamp = np.loadtxt(datastore, dtype='str', usecols=(0), skiprows=6)

#export the dataset(reversed) without the 1st array for a problem
data_matrix = np.loadtxt(datastore, dtype='int', skiprows=6, usecols=np.arange(1, data_rate+1),max_rows=len(timestamp)-1)

# Remove the first column from matrix (various errors)
data_matrix = data_matrix[1:]

#create a 1D array from the matrix
data_array = np.concatenate(data_matrix)

