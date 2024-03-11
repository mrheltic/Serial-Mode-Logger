from datetime import datetime, timedelta
import numpy as np
from collections import deque
from matplotlib.animation import FuncAnimation

import matplotlib.pyplot as plt
import scipy.stats as stats

datastore = './Dataset/sinusoidalwave.txt'

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

# Calculate the FFT of the data
fft_result = np.fft.fft(data_array)

# Calculate the amplitude of the FFT in decibels
fft_amplitude = 20 * np.log10(np.abs(fft_result))

# Create the frequency array
freqs = np.fft.fftfreq(data_array.size, 1/data_rate)

# Remove the fundamental frequency from the frequency array
fft_amplitude[0] = 0

# Plot the FFT showing only the positive frequencies
fig, ax = plt.subplots()
ax.plot(freqs[:data_array.size//2], fft_amplitude[:data_array.size//2])

# Extract the maximum frequency and amplitude
max_amplitude = np.max(fft_amplitude)

# Extract the frequency at the maximum amplitude
max_freq = freqs[np.argmax(fft_amplitude)]

# Extract the harmonics of the max frequency
harmonics = []
for i in range(2, 6):
    harmonics.append(max_freq * i)

# Building an array with the bins of the harmonics
harmonics_bins = []
for harmonic in harmonics:
    harmonics_bins.append(harmonic - max_freq*0.4)
    harmonics_bins.append(harmonic + max_freq*0.4)

# Searching for the maximum amplitude in the harmonics bins and storing them in an array
harmonics_max_amplitude = []
harmonics_max_freq = []
for i in range(0, len(harmonics_bins), 2):
    harmonics_max_amplitude.append(np.max(fft_amplitude[(freqs > harmonics_bins[i]) & (freqs < harmonics_bins[i+1])]))
    harmonics_max_freq.append(freqs[(freqs > harmonics_bins[i]) & (freqs < harmonics_bins[i+1])][np.argmax(fft_amplitude[(freqs > harmonics_bins[i]) & (freqs < harmonics_bins[i+1])])])



# Add in the first place of the array the maximum amplitude and frequency
harmonics_max_amplitude.insert(0, max_amplitude)
harmonics_max_freq.insert(0, max_freq)

# Indicates the harmonics on the plot with a red dot
ax.plot(harmonics_max_freq, harmonics_max_amplitude, 'ro')


# Calculate the signal power from the harmonics
signal_power = 0

for i in range(0, len(harmonics_max_amplitude)):
    signal_power += (harmonics_max_amplitude[i] ** 2) / len(harmonics_max_amplitude)



# Substitute the amplitude of the harmonics with the mean value of the noise
for i in range(0, len(harmonics_max_amplitude)):
    fft_amplitude[(freqs > harmonics_bins[i]) & (freqs < harmonics_bins[i+1])] = np.mean(fft_amplitude[(freqs > harmonics_bins[i]) & (freqs < harmonics_bins[i+1])])

# Calculate the noise power from the other frequencies
noise_power = 0

for i in range(0, len(fft_amplitude)):
    noise_power += (fft_amplitude[i] ** 2) / len(fft_amplitude)

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

plt.show()
