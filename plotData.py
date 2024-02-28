from datetime import datetime, timedelta
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from collections import deque 
from matplotlib.animation import FuncAnimation 
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg,
    NavigationToolbar2Tk
)




#export the current mode
currentmode=np.loadtxt('dataStorage.txt', dtype='str', max_rows=1)[-1]

#export the k value
k_value=float(np.loadtxt('dataStorage.txt', dtype='float', usecols=(1), skiprows=1,max_rows=1))

#export the offset
offset=np.loadtxt('dataStorage.txt', dtype='float', usecols=(1), skiprows=2, max_rows=1) 

#export the data rate
data_rate=int(np.loadtxt('dataStorage.txt', dtype='int', usecols=(4), skiprows=3, max_rows=1))

#export the conversion factor
factor=np.loadtxt('dataStorage.txt', dtype='float', usecols=(1), skiprows=4, max_rows=1) 

#export the timestamp
timestamp = np.loadtxt('dataStorage.txt', dtype='str', usecols=(0), skiprows=6) 

#export the dataset(reversed) without the 1st array for a problem
data_matrix = np.loadtxt('dataStorage.txt', dtype='int', skiprows=6, usecols=np.arange(1, data_rate+1),max_rows=len(timestamp)-1)

#create a 1D array from the matrix
data_array = np.concatenate(data_matrix) 

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

#declarations for meanPlot 
#fig2, ax2 = plt.subplots()

#set the grid
#plt.grid()

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

#conversion
if currentmode=="Voltage":
    gain=k_value*factor
    data_matrix = np.dot(data_matrix,gain)-offset

    #call the function to find the maximum value in the dataset
    max=max_value(data_matrix)
    
    #set the limits of the y axis
    ax.set_ylim(0, max + 0.3 ) 

else: 
    if currentmode=="Current":
        gain=k_value*factor
        data_matrix = np.dot(data_matrix,gain)-offset

        #call the function to find the maximum value in the dataset
        max=max_value(data_matrix)

        #set the limits of the y axis
        ax.set_ylim(-max- 0.3, max + 0.3 ) 
        
    if currentmode=="Resistance":
        num=factor*3.3
        den = np.dot(data_matrix,k_value)
        data_matrix= num/den-offset

        #call the function to find the maximum value in the dataset
        max=max_value(data_matrix)

        #set the limits of the y axis
        ax.set_ylim(0, max + 0.3 )    
    

import customtkinter

customtkinter.set_appearance_mode("System")  # Modes: system (default), light, dark
customtkinter.set_default_color_theme("blue")

root = customtkinter.CTk()  # create CTk window like you do with the Tk window
root.geometry("1280x720")
root.resizable(False, False)
root.title("Logger project Interface")



graphframe = customtkinter.CTkFrame(master=root)
graphframe.pack(pady=20, padx=60, fill="both", expand=False)
graphframe.configure(height=400, width=100)

commandframe = customtkinter.CTkFrame(master=root)
commandframe.pack(pady=20, padx=60, fill="both", expand=False)
commandframe.configure(height=200, width=100)


fig=Figure(figsize=(10, 5), dpi=100)



#root = tk.Tk()
#root.geometry("1366x768")
#root.resizable(False, False)
#root.title("Logger project Interface")

def MeanOverTime():
    
    # Tkinter Application
    #meanframe = customtkinter.CTkFrame(master=root)
    #meanframe.pack(pady=20, padx=50, fill="both", expand=False)
    
    #meanframe = tk.Frame(root)
    #meanframe.pack()
    
    fig2=Figure(figsize=(10, 5), dpi=100)



    
    # Create the mean graphs
    ax2 = fig2.add_subplot()
    ax2.plot(timestamp[:(len(timestamp)-1)], mean_values)
    ax2.set_xlabel("Timestamp [s]")
    ax2.set_ylabel("Mean(t)")

    # Create Canvas
    canvas = FigureCanvasTkAgg(fig2, master=graphframe)  
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=0)

    canvas.draw()
    #fig2.clf()
    
    


   # Std_Button = customtkinter.CTkButton(master=root, text="Calculate Standard Deviation Over time", command=StandardDeviationGraph)
    #Std_Button.place(relx=0.3, rely=0.8, anchor=customtkinter.E)

    #plt.title('Mean value over time')
    #plt.xlabel('Time (s)')
    #plt.ylabel('Mean value')
    

w=FigureCanvasTkAgg(fig, master=graphframe).get_tk_widget()

def add_plot():

    canvas = FigureCanvasTkAgg(fig, master=graphframe)  
    w=canvas.get_tk_widget()
    w.pack(side=tk.TOP, fill=tk.BOTH, expand=1)
    #toolbar = NavigationToolbar2Tk(canvas, graphframe) 
    #toolbar.update() 
    canvas.draw()
    return w
    
    



def remove_plot(w):
    w.destroy()
    w=FigureCanvasTkAgg(fig, master=graphframe).get_tk_widget()
    return w
     # here you remove the widget from the tk window
    # w.destroy()

def StandardDeviationGraph(w):
    

    # Tkinter Application
    
    #stdframe = tk.Frame(root)
    #stdframe.pack()
    
    #remove_plot(w)
    # Create the mean graphs
    ax2 = fig.add_subplot()
    ax2.plot(timestamp[:(len(timestamp)-1)], std_values)
    ax2.set_xlabel("'Time (s)'")
    ax2.set_ylabel("Standard deviation")
    ax2.fill_between(timestamp[:(len(timestamp)-1)],
                    std_values - stats.t.ppf(0.975, df=data_rate - 1) * std_values / np.sqrt(data_rate),
                    std_values + stats.t.ppf(0.975, df=data_rate - 1) * std_values / np.sqrt(data_rate), color='gray',
                    alpha=0.5)

    # Create Canvas
    w=add_plot()
    return w
    
    
    #remove_plot(w)
    
    
    
    

    





#Mean_button = customtkinter.CTkButton(master=commandframe, text="Calculate Mean Over Time", command=lambda:[remove_plot(w),MeanOverTime])
#Mean_button.place(relx=0.3, rely=0.9, anchor=customtkinter.E)

Std_Button = customtkinter.CTkButton(master=commandframe, text="Calculate Standard Deviation",command= lambda:[remove_plot(w),StandardDeviationGraph(w)])
Std_Button.place(relx=0.3, rely=0.8, anchor=customtkinter.E)

Quit_button = customtkinter.CTkButton(master=commandframe, text="Quit", command=lambda: root.quit())
Quit_button.place(relx=0.8, rely=0.9, anchor=customtkinter.W)



#mean_button = customtkinter.CTkButton(master=meanframe, text="Calculate Mean Over Time", command=MeanOverTime)



#Mean_button = tk.Button(master=root, text="Mean Value Over time!", command=MeanOverTime)
#Quit_button = tk.Button(master=root, text="Quit", command=lambda: root.quit())

#Std_Button = tk.Button(master=root, text="Standard Deviation Over time!", command=StandardDeviationGraph)

#second_button = tk.Button(text="Seconda Funzione", command=second_function)
#second_button.grid(row=1, column=0, pady=20, sticky="W")

#Std_Button.pack(side=tk.BOTTOM, expand=1, anchor='w')
#Mean_button.pack(side=tk.BOTTOM, expand=1, anchor='w')
#Quit_button.pack(ipadx=5, ipady=5, expand=1, anchor='sw')

root.mainloop()









"""
#for each row except the last one
for i in range(1, len(timestamp)):
   
    #create a deltatime based on the difference between the timestamps
    timestamp2=datetime.strptime(timestamp[i], '%H:%M:%S') 
    timestamp1=datetime.strptime(timestamp[i-1], '%H:%M:%S')
    diff=timestamp2-timestamp1

    
    the second timestamp refers to the start of the first measure of the array so I substract, from the  
    difference, the fraction of time corresponding to the time needed to acquire a single sample.
    this is an approximation, but it is good enough for the purpose of the plot
    
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


# Create the mean graphs
plt.figure(figsize=(10, 5), dpi=100)
plt.plot(timestamp[:(len(timestamp)-1)], mean_values)
plt.title('Mean value over time')
plt.xlabel('Time (s)')
plt.ylabel('Mean value')


#Create the standard deviation graph with error bands
plt.figure(figsize=(10, 5), dpi=100)

#plt.subplot(2, 1, 2)
plt.plot(timestamp[:(len(timestamp)-1)], std_values)

plt.fill_between(timestamp[:(len(timestamp)-1)],
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
freqs = np.fft.fftfreq(data_array.size, 1/data_rate)

# Plot the FFT
plt.figure(figsize=(4, 3), dpi=150)
plt.plot(freqs, fft_amplitude)
plt.title('FFT of the data')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')
plt.grid()
plt.show()
"""