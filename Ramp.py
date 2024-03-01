from datetime import datetime, timedelta
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from collections import deque
from matplotlib.animation import FuncAnimation

datastore = 'dataStorage.txt'

# Initializing period, amplitude, and offset
period = 0.1
amplitude = 6.144
offset = 3.072


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
data_matrix = data_matrix[1:]

#create a 1D array from the matrix
data_array = np.concatenate(data_matrix)

# Build a ramp signal, given the period, amplitude, and offset
ramp = amplitude * (1 + np.sin(2 * np.pi * np.arange(data_rate) / period)) + offset

# Plot the ramp signal
plt.plot(ramp)


#declarations for the dynamic plot
fig, ax = plt.subplots() 
line, = ax.plot([]) 

#set the grid
plt.grid()

#comment the following line to use the plot with interpolation
scatter=ax.scatter([], [])

#set the maximum number of data points to be shown
data_points = deque(maxlen=data_rate) 

#function to find the maximum value in the dataset
def max_value(data_matrix):
    max=0
    for i in range(1, data_matrix.shape[0]):
        temp= np.max(data_matrix[i,:])
        if(temp>max):
            max=temp
    return max

gain=k_value*factor
data_matrix = np.dot(data_matrix,gain)-offset

#call the function to find the maximum value in the dataset
max=max_value(data_matrix)
    
#set the limits of the y axis
ax.set_ylim(0, max + 0.3 ) 

#for each row except the last one
for i in range(1, len(timestamp)):
   
    #create a deltatime based on the difference between the timestamps
    timestamp2=datetime.strptime(timestamp[i], '%H:%M:%S') 
    timestamp1=datetime.strptime(timestamp[i-1], '%H:%M:%S')
    diff=timestamp2-timestamp1

    """
    the second timestamp refers to the start of the first measure of the array so I substract, from the  
    difference, the fraction of time corresponding to the time needed to acquire a single sample.
    this is an approximation, but it is good enough for the purpose of the plot
    """
    diff=diff-diff/data_rate

    #create a timeline from a timestamp to another
    timeline=np.arange(timestamp1, timestamp1+diff, diff/data_rate)
    
    #just to be sure that the timeline has the right shape for the plot
    if(timeline.shape>(data_rate,)):
        timeline=np.delete(timeline, -1)


    #set the limits of the x axis
    ax.set_xlim(timestamp1, timestamp2) 
    
    
    #for each data point in the row
    for j in range(0, data_rate): 

        # add every single value in the timeline to new_x
        new_x = timeline[j]
       

        # add the corresponding value to new_y
        new_y = data_matrix[i-1,j]
        

        #list of tuples
        data_points.append((new_x, new_y)) 
        
        
        # Update the plot with the new data points 
        x_values = [x for x, y in data_points] 
        y_values = [y for x, y in data_points] 

        #comment the following line to use the plot with interpolation
        scatter.set_offsets(list(zip(x_values, y_values))) 

        line.set_data(x_values, y_values) 

        #change the following parameter to adjust the speed of the plot
        plt.pause(0.0000001) 
    


#uncomment the following lines to clear the plot with interpolation 
#line.set_data([], [])          #clear  

#show the plot
plt.show() 

# Evaluate the error between the ramp and the acquired signal
error = np.mean(np.abs(ramp - data_array))
print("Mean error: ", error)

# Visualize the error
plt.plot(np.abs(ramp - data_array))

