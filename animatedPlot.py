from datetime import datetime, timedelta
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from collections import deque 
from matplotlib.animation import FuncAnimation 
import Conversion.animatedConversion as animatedConversion
import Conversion.conversion as conversion
import DataExtraction.extractdata as extractdata

datastore = './Dataset/Ramp/temp/ramp3.ds32'
#number of array to skip in the dataset

# Extract the data
currentmode, k_value, offset, data_rate, factor, timestamp, data_matrix = extractdata.extract_data(datastore)

# Set the conversion factor to 1 if you're using the FSR of ADC
#factor = 1

#create a 1D array from the matrix
data_array = np.concatenate(data_matrix) # It could be changed to .reshape

#declaration of the mean values array
mean_values = np.zeros(data_matrix.shape[0]) 

#declaration of the standard deviation values array
std_values = np.zeros(data_matrix.shape[0]) 


# Calculate the mean and standard deviation for each row
for i in range(1, data_matrix.shape[0]):
    mean_values[i] = np.mean(data_matrix[i,:])
    std_values[i]= np.std(data_matrix[i,:])


#declarations for the dynamic plot
fig, ax = plt.subplots() 
line, = ax.plot([]) 

#set the grid
plt.grid()

#comment the following line to use the plot with interpolation
scatter=ax.scatter([], [])

#set the maximum number of data points to be shown
data_points = deque(maxlen=2*data_rate) 

# Convert the data
data_matrix = animatedConversion.convert_data(currentmode, k_value, factor, offset, data_matrix, ax)

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
        plt.pause(0.01) 
    


#uncomment the following lines to clear the plot with interpolation 
#line.set_data([], [])          #clear  

#show the plot
plt.show(block=False)
plt.close('all')