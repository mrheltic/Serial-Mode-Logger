import time
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

# Open the data matrix file and import the data matrix withuo the first column (timestamps)
data_matrix = np.loadtxt('data_matrix.txt', skiprows=1)

# Read the data rate from the file, second line (# Data rate: 860 SPS)
data_rate = int(open('data_matrix.txt').readlines()[1].split(' ')[-2])

# Remove the first column (timestamps) from the data matrix and convert it to a list
data_matrix = data_matrix[:, 1:].tolist()

# Flatten the matrix into a one-dimensional array
data_array = np.concatenate(data_matrix)

# Create a time array. Every timestamp is the first element of the array, like 07:58:41
time_array = np.loadtxt('data_matrix.txt', usecols=(0,), dtype=str)


# Create a graph
plt.figure(figsize=(4, 3), dpi=500)
plt.plot(time_array, data_array)
# plt.plot(time_array, data_array)
plt.title('Data graph')
plt.xlabel('Time (s)')
plt.ylabel('Value')

# Show the graph
plt.show()

# Convert the data_matrix to a numpy array for easier calculations
data_matrix = np.array(data_matrix)

# Calculate the mean and standard deviation for each row
mean_values = np.mean(data_matrix, axis=1)
std_values = np.std(data_matrix, axis=1)

# Create a new figure for the mean and standard deviation graphs
plt.figure(figsize=(4, 3), dpi=150)

# Create the mean graph
plt.subplot(2, 1, 1)
plt.plot(time_array[::data_rate], mean_values)
plt.title('Mean value over time')
plt.xlabel('Time (s)')
plt.ylabel('Mean value')

# Create the standard deviation graph with error bands
plt.subplot(2, 1, 2)
plt.plot(time_array[::data_rate], std_values)
plt.fill_between(time_array[::data_rate],
                 std_values - stats.t.ppf(0.975, df=data_rate - 1) * std_values / np.sqrt(data_rate),
                 std_values + stats.t.ppf(0.975, df=data_rate - 1) * std_values / np.sqrt(data_rate), color='gray',
                 alpha=0.5)
plt.title('STD with 95% confidence interval')
plt.xlabel('Time (s)')
plt.ylabel('Standard deviation')

# Show the graphs
plt.tight_layout()
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
