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
import noiseFloor

# Ramp 1: 4V, 100Hz
# Ramp 2: 3.65V, 100Hz
# Ramp 3: 3.55V, 100Hz

datastore = './Dataset/ramp3.txt'

# Number of rows to skip in the data array
number_of_rows_to_skip = 1

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
factor=np.loadtxt(datastore, dtype='float', usecols=(1), skiprows=4, max_rows=1) 

#export the timestamp
timestamp = np.loadtxt(datastore, dtype='str', usecols=(0), skiprows=5+number_of_rows_to_skip) 

#export the dataset(reversed) without the 1st array for a problem
data_matrix = np.loadtxt(datastore, dtype='int', skiprows=5+number_of_rows_to_skip, usecols=np.arange(1, data_rate+1),max_rows=len(timestamp)-1)

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

fig2, ax = plt.subplots() 
line, = ax.plot([]) 

#declarations for meanPlot 
#fig2, ax2 = plt.subplots()

#set the grid
#plt.grid()

#comment the following line to use the plot with interpolation
scatter=ax.scatter([], [])



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
root.geometry("1366x768")
root.resizable(True, True)
root.title("Logger project Performance Interface")



"""
def destroyframe(frame):
    customtkinter.CTkFrame.pack_forget(frame)
    
    #customtkinter.CTkFrame.pack_forget(graphframe)
"""


#createframe(graphframe)

commandframe = customtkinter.CTkFrame(master=root)
commandframe.pack(pady=10, padx=10, fill="both", expand=False,side="left")
commandframe.configure(height=600, width=200)

graphframe = customtkinter.CTkFrame(master=root)
graphframe.pack(pady=10, padx=20, fill="both", expand=False,side="top")
graphframe.configure(height=200, width=300)

graphframe2 = customtkinter.CTkFrame(master=root)
graphframe2.pack(pady=10, padx=20, fill="both", expand=False, side="top")
graphframe2.configure(height=200, width=300)

graphframe3 = customtkinter.CTkFrame(master=root)
graphframe3.pack(pady=10, padx=20, fill="both", expand=False, side="top")
graphframe3.configure(height=200, width=300)


def noise():
    
    # Tkinter Application
    #meanframe = customtkinter.CTkFrame(master=root)
    #meanframe.pack(pady=20, padx=50, fill="both", expand=False)
    
    #meanframe = tk.Frame(root)
    #meanframe.pack()
    [freqs, fft_amplitude]=noiseFloor.plotNoise(k_value,factor,offset,data_matrix,data_array,data_rate)
    print(freqs, fft_amplitude)
    fig2=Figure(figsize=(5, 2), dpi=100)



    
    # Create the mean graphs
    ax2 = fig2.add_subplot()

    ax2.plot(freqs[:data_array.size//2], fft_amplitude[:data_array.size//2])
    
    # Extract the maximum frequency and amplitude
    max_amplitude = np.max(fft_amplitude)

    # Extract the frequency at the maximum amplitude
    max_freq = freqs[np.argmax(fft_amplitude)]

    # Calculate the ENOB of the signal where enob = n - log2(signal power / noise power)
    # Calculate the signal power
    signal_power = np.mean(data_array ** 2)

    # Calculate the noise power
    noise_power = np.mean((data_array - np.mean(data_array)) ** 2)

    # Calculate the ENOB
    enob = (np.log2(signal_power / noise_power))

    # Calculate the SNR of the signal
    snr = (6.02 * enob) - 1.76
    snr2 = 20 * np.log10(signal_power / noise_power)
    # Calculate the SINAD of the signal
    sinad = 1.76 + (6.02 * enob)

    # Add text annotations for ENOB, signal power, and other information
    enob_text = f'ENOB: {enob:.2f} bits'
    signal_power_text = f'Signal Power: {signal_power:.2f}'
    noise_power_text = f'Noise Power: {noise_power:.2f}'
    max_amplitude_text = f'Maximum Amplitude: {max_amplitude} dB'
    max_freq_text = f'Frequency at Maximum Amplitude: {max_freq} Hz'
    snr_text = f'SNR: {snr:.2f} dB'
    sinad_text = f'SINAD: {sinad:.2f} dB'

    ax.text(0.05, 0.95, enob_text, transform=ax.transAxes, fontsize=10, verticalalignment='top')
    ax.text(0.05, 0.90, signal_power_text, transform=ax.transAxes, fontsize=10, verticalalignment='top')
    ax.text(0.05, 0.85, noise_power_text, transform=ax.transAxes, fontsize=10, verticalalignment='top')
    ax.text(0.05, 0.80, max_amplitude_text, transform=ax.transAxes, fontsize=10, verticalalignment='top')
    ax.text(0.05, 0.75, max_freq_text, transform=ax.transAxes, fontsize=10, verticalalignment='top')
    ax.text(0.05, 0.70, snr_text, transform=ax.transAxes, fontsize=10, verticalalignment='top')
    ax.text(0.05, 0.65, sinad_text, transform=ax.transAxes, fontsize=10, verticalalignment='top')


    
    # Create Canvas
    canvas = FigureCanvasTkAgg(fig2, master=graphframe)  
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=0)
    canvas.draw()
    toolbar = NavigationToolbar2Tk(canvas,graphframe)

def noise2():
    
    # Tkinter Application
    #meanframe = customtkinter.CTkFrame(master=root)
    #meanframe.pack(pady=20, padx=50, fill="both", expand=False)
    
    #meanframe = tk.Frame(root)
    #meanframe.pack()
    [freqs, fft_amplitude]=noiseFloor.plotNoise(k_value,factor,offset,data_matrix,data_array,data_rate)
    print(freqs, fft_amplitude)
    fig2=Figure(figsize=(5, 2), dpi=100)



    
    # Create the mean graphs
    ax2 = fig2.add_subplot()

    ax2.plot(freqs[:data_array.size//2], fft_amplitude[:data_array.size//2])
    
    # Extract the maximum frequency and amplitude
    max_amplitude = np.max(fft_amplitude)

    # Extract the frequency at the maximum amplitude
    max_freq = freqs[np.argmax(fft_amplitude)]

    # Calculate the ENOB of the signal where enob = n - log2(signal power / noise power)
    # Calculate the signal power
    signal_power = np.mean(data_array ** 2)

    # Calculate the noise power
    noise_power = np.mean((data_array - np.mean(data_array)) ** 2)

    # Calculate the ENOB
    enob = (np.log2(signal_power / noise_power))

    # Calculate the SNR of the signal
    snr = (6.02 * enob) - 1.76
    snr2 = 20 * np.log10(signal_power / noise_power)
    # Calculate the SINAD of the signal
    sinad = 1.76 + (6.02 * enob)

    # Add text annotations for ENOB, signal power, and other information
    enob_text = f'ENOB: {enob:.2f} bits'
    signal_power_text = f'Signal Power: {signal_power:.2f}'
    noise_power_text = f'Noise Power: {noise_power:.2f}'
    max_amplitude_text = f'Maximum Amplitude: {max_amplitude} dB'
    max_freq_text = f'Frequency at Maximum Amplitude: {max_freq} Hz'
    snr_text = f'SNR: {snr:.2f} dB'
    sinad_text = f'SINAD: {sinad:.2f} dB'

    ax.text(0.05, 0.95, enob_text, transform=ax.transAxes, fontsize=10, verticalalignment='top')
    ax.text(0.05, 0.90, signal_power_text, transform=ax.transAxes, fontsize=10, verticalalignment='top')
    ax.text(0.05, 0.85, noise_power_text, transform=ax.transAxes, fontsize=10, verticalalignment='top')
    ax.text(0.05, 0.80, max_amplitude_text, transform=ax.transAxes, fontsize=10, verticalalignment='top')
    ax.text(0.05, 0.75, max_freq_text, transform=ax.transAxes, fontsize=10, verticalalignment='top')
    ax.text(0.05, 0.70, snr_text, transform=ax.transAxes, fontsize=10, verticalalignment='top')
    ax.text(0.05, 0.65, sinad_text, transform=ax.transAxes, fontsize=10, verticalalignment='top')


    
    # Create Canvas
    canvas = FigureCanvasTkAgg(fig2, master=graphframe2)  
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=0)
    canvas.draw()
    toolbar = NavigationToolbar2Tk(canvas,graphframe2)

Noise_button = customtkinter.CTkButton(master=commandframe, text="Show noise",command=noise)
Noise_button.place(relx=0.0, rely=0.8, anchor=customtkinter.W)

ramp_button= customtkinter.CTkButton(master=commandframe, text="Show ramp",command=noise2)
ramp_button.place(relx=0.0, rely=0.7, anchor=customtkinter.W)

#ramp2_button= customtkinter.CTkButton(master=commandframe, text="Show ramp2",command=noise)
#ramp2_button.place(relx=0.0, rely=0.6, anchor=customtkinter.W)

Quit_button = customtkinter.CTkButton(master=commandframe, text="Quit", command=lambda: root.quit())
Quit_button.place(relx=0.0, rely=0.9, anchor=customtkinter.W)





root.mainloop()