from datetime import datetime, timedelta
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from collections import deque 


#export the current mode
currentmode=np.loadtxt('data_matrix.txt', dtype='str', max_rows=1)[-1] 

#export the data rate
data_rate=int(np.loadtxt('data_matrix.txt', dtype='int', usecols=(2), skiprows=1,max_rows=1))


#export the offset
offset=np.loadtxt('data_matrix.txt', dtype='float', usecols=(2), skiprows=2, max_rows=1) 

#export the timestamp
timestamp = np.loadtxt('data_matrix.txt', dtype='str', usecols=(0), skiprows=3) 

#export the data matrix
data_matrix = np.loadtxt('data_matrix.txt', dtype='int', usecols=np.arange(1, data_rate+1), skiprows=3) 


#create a 1D array from the matrix
data_array = np.concatenate(data_matrix) 


mean_values = np.zeros(data_matrix.shape[0]) #declaration of the mean values array
std_values = np.zeros(data_matrix.shape[0]) #declaration of the standard deviation values array

# Calculate the mean and standard deviation for each row
for i in range(0, data_matrix.shape[0]):
    mean_values[i] = np.mean(data_matrix[i,:])
    std_values[i]= np.std(data_matrix[i,:])


#declarations for the dynamic plot
fig, ax = plt.subplots() 
line, = ax.plot([]) 

#set the maximum number of data points to be shown
data_points = deque(maxlen=data_rate) 

#set the limits of the y axis
ax.set_ylim(0, 9000) 



#for each row except the last one
for i in range(1, len(timestamp)):
   
    #create a deltatime based on the difference between the timestamps
    timestamp2=datetime.strptime(timestamp[i], '%H:%M:%S') 
    timestamp1=datetime.strptime(timestamp[i-1], '%H:%M:%S')
    diff=timestamp2-timestamp1

    #create a timeline from a timestamp to another
    timeline=np.arange(timestamp1, timestamp1 + diff , diff/data_rate)
    
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
        new_y = data_matrix[i,j]

        #list of tuples
        data_points.append((new_x, new_y)) 
  
        # Update the plot with the new data points 
        x_values = [x for x, y in data_points] 
        y_values = [y for x, y in data_points] 
        line.set_data(x_values, y_values) 

        #pause the plot for a short time
        plt.pause(0.0001) 
  

#create a fake timestamp for the last plot with delta time of 1 second
faketimestamp=timedelta(seconds=1)
finaltimestamp=datetime.strptime(timestamp[-1], '%H:%M:%S')
finaltimeline=np.arange(finaltimestamp, finaltimestamp + faketimestamp, faketimestamp/data_rate)
     
if(timeline.shape>(data_rate,)):
        timeline=np.delete(timeline, -1)
        
#set the limits of the x axis
ax.set_xlim(finaltimestamp, (finaltimestamp + faketimestamp)) 
for j in range(0, data_rate): 

    # add every single value in the timeline to new_x
    new_x = finaltimeline[j]

    # add the corresponding value to new_y
    new_y = data_matrix[-1,j]

    #list of tuples
    data_points.append((new_x, new_y)) 
  
    # Update the plot with the new data points 
    x_values = [x for x, y in data_points] 
    y_values = [y for x, y in data_points] 
    line.set_data(x_values, y_values) 

    #pause the plot for a short time
    plt.pause(0.0001) 

#clear the plot        
line.set_data([], [])

#plot
plt.grid()
plt.show() 



"""""
    #plot the data
    plt.figure(figsize=(10, 5), dpi=100)
    plt.plot(timeline, data_matrix[i,:])
    plt.title('Data graph')
    plt.xlabel('Time (s)')
    plt.ylabel('Value')
    plt.grid()
    plt.show()
"""




"""""
for j in range(0, data_rate): 

        # add every single value in the timeline to new_x
        new_x = timeline[j]

        # add the corresponding value to new_y
        new_y = data_matrix[-1,j]

        #list of tuples
        data_points.append((new_x, new_y)) 
  
        # Update the plot with the new data points 
        x_values = [x for x, y in data_points] 
        y_values = [y for x, y in data_points] 
        line.set_data(x_values, y_values) 

        #pause the plot for a short time
        plt.pause(0.0001) 
  
     
#clear the plot        
line.set_data([], [])


plt.grid()
plt.show() 


plt.figure(figsize=(10, 5), dpi=100)
plt.plot(timeline, data_matrix[-1,:])
plt.title('Data graph')
plt.xlabel('Time (s)')
plt.ylabel('Value')
plt.grid()
plt.show()

"""""
# Create the mean graphs
plt.figure(figsize=(10, 5), dpi=100)
plt.plot(timestamp, mean_values)
plt.title('Mean value over time')
plt.xlabel('Time (s)')
plt.ylabel('Mean value')
plt.grid()


#Create the standard deviation graph with error bands
plt.figure(figsize=(10, 5), dpi=100)

#plt.subplot(2, 1, 2)
plt.plot(timestamp, std_values)
plt.fill_between(timestamp,
                 std_values - stats.t.ppf(0.975, df=data_rate - 1) * std_values / np.sqrt(data_rate),
                 std_values + stats.t.ppf(0.975, df=data_rate - 1) * std_values / np.sqrt(data_rate), color='gray',
                 alpha=0.5)
plt.title('STD with 95% confidence interval')
plt.xlabel('Time (s)')
plt.ylabel('Standard deviation')

# Show the graphs
plt.tight_layout()
plt.grid()

plt.show()


# Calculate the FFT of the data
fft_result = np.fft.fft(data_array)

# Calculate the amplitude of the FFT
fft_amplitude = np.abs(fft_result)

# Create the frequency array
freqs = np.fft.fftfreq(len(data_array))

# Plot the FFT
plt.figure(figsize=(4, 3), dpi=150)
plt.plot(freqs, fft_amplitude)
plt.title('FFT of the data')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')
plt.show()
