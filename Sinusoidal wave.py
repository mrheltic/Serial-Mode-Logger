from datetime import datetime, timedelta
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from collections import deque
from matplotlib.animation import FuncAnimation

datastore = './Dataset/sinusoidal wave 200hz.txt'

# Create a sinusoidal wave from 0 to 4, 200hz
amplitude = 1.775
offset_sin = 1.775
period = 87


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
data_matrix = np.dot(data_matrix,gain)-offset-offset_sin

# Flatten the matrix
data_array = data_matrix.flatten()

# Build the periodical sinusoidal signal considering the amplitude and offset
sinusoidal = amplitude * np.sin(np.linspace(0, 2*np.pi, period))

# Find the period in the data array and extract it
max_index = np.argmax(data_array)
data_array_period = data_array[max_index:max_index + round(2*period)]

# Shift the data array to right so that the period starts at the beginning of the array
data_array_period = np.roll(data_array_period, -abs(np.argmin(data_array_period)))

for i in range(0, len(data_array_period)):
    if data_array_period[i]*data_array_period[i-1] < 0:
        start_data_array_period = i-1
        break

# Remove the first part of the data array period
data_array_period = data_array_period[start_data_array_period:]

for i in range(int(0.9*period), len(data_array_period)):
    if data_array_period[i]*data_array_period[i-1] < 0:
        end_data_array_period = i-1
        break

# Remove the last part of the data array period
data_array_period = data_array_period[:end_data_array_period+(sinusoidal.size - end_data_array_period)]

# Fit a sinudoidal model to the data array period
slope, intercept, r_value, p_value, std_err = stats.linregress(np.linspace(0, 2*np.pi, data_array_period.size), data_array_period)

# Fit a sinudoidal model to the sinusoidal signal
slope_sinusoidal, intercept_sinusoidal, r_value_sinusoidal, p_value_sinusoidal, std_err_sinusoidal = stats.linregress(np.linspace(0, 2*np.pi, sinusoidal.size), sinusoidal)


# Plot one iteration of data array period and compare it to the generated sinusoidal signal, until the period is found
fig, (ax1, ax2) = plt.subplots(2, 1)

# Adding labels
ax1.set_xlabel('Sample number')
ax1.set_ylabel('Amplitude (V)')

# Adding title and legend
ax1.set_title('Sinusoidal signal')
ax1.legend(['Data array period'], loc='upper right')
ax1.plot(data_array_period)
ax1.plot(sinusoidal)

# Adding labels
ax2.set_xlabel('Sample number')
ax2.set_ylabel('Error')

# Adding title
ax2.set_title('Error between data array period and ramp signal')

# Plot the difference between the data array period and the sinusoidal signal
ax2.plot((data_array_period - sinusoidal))

plt.tight_layout()
plt.show()


# Calculate the FFT of the data
fft_result = np.fft.fft(data_array)

# Calculate the amplitude of the FFT in decibels
fft_amplitude = 20 * np.log10(np.abs(fft_result))

# Create the frequency array
freqs = np.fft.fftfreq(data_array.size, 1/data_rate)

# Plot the FFT showing only the positive frequencies
fig, ax = plt.subplots()

# Extract the maximum frequency and amplitude
max_amplitude = np.max(fft_amplitude)

# Extract the frequency at the maximum amplitude
max_freq = freqs[np.argmax(fft_amplitude)]

# Extract the armonic frequencies
harmonic_freqs = [max_freq]
harmonic_amplitudes = [max_amplitude]

for i in range(2, 10):
    indices = np.where(freqs == i * max_freq)[0]
    if len(indices) > 0:
        harmonic_freqs.append(i * max_freq)
        harmonic_amplitudes.append(fft_amplitude[indices[0]])


# Substitute the armonics with the mean of the nearest frequencies
for i in range(1, len(harmonic_freqs)-1):
    if harmonic_amplitudes[i] < harmonic_amplitudes[i-1] and harmonic_amplitudes[i] < harmonic_amplitudes[i+1]:
        harmonic_amplitudes[i] = (harmonic_amplitudes[i-1] + harmonic_amplitudes[i+1])/2


# Calculate the ENOB of the signal where enob = n - log2(signal power / noise power)
# Calculate the signal power
signal_power = np.mean(data_array ** 2)

# Calculate the noise power considering the 
noise_power = np.mean((data_array - np.mean(data_array)) ** 2)

# Calculate the ENOB
enob = (np.log2(signal_power / noise_power))

# Calculate the SNR of the signal
snr = (6.02 * enob) - 1.76
snr2 = 20 * np.log10(signal_power / noise_power)
# Calculate the SINAD of the signal
sinad = 1.76 + (6.02 * enob)

# Plot the FFT considering only the positive frequencies
ax.plot(freqs[:data_array.size//2], fft_amplitude[:data_array.size//2])

# Plot the armonic frequencies
ax.scatter(harmonic_freqs, harmonic_amplitudes, color='red')

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