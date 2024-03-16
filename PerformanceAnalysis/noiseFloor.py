from datetime import datetime, timedelta
import numpy as np
from collections import deque
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import scipy.stats as stats
import sys
sys.path.insert(0, './')
import Conversion.conversion as conversion
import DataExtraction.extractdata as extractdata

datastore = './Dataset/Noise/shortingnoise.ds32'

# Extract the data
currentmode, k_value, offset, data_rate, factor, timestamp, data_matrix = extractdata.extract_data(datastore)

# Set the conversion factor to 1 if you're using the FSR of ADC
factor = 1

# Convert the data
data_matrix, data_array = conversion.convert_data(currentmode, k_value, factor, offset, data_matrix)

# Calculate the FFT of the data
fft_result = np.fft.fft(data_array)

# Calculate the amplitude of the FFT in decibels
fft_amplitude = 20 * np.log10(np.abs(fft_result))

# Create the frequency array
freqs = np.fft.fftfreq(data_array.size, 1/data_rate)

# Remove the fundamental frequency from the frequency array
fft_amplitude[0] = 0
fft_result[0] = 0

# Create a figure with two subplots
fig, (ax1, ax2) = plt.subplots(2, 1)

# Plot the FFT amplitude in the top subplot
ax1.plot(freqs[:data_array.size//2], fft_amplitude[:data_array.size//2])
ax1.set_ylabel('Amplitude (dB)')

# Plot the FFT result in the bottom subplot
ax2.plot(freqs[:data_array.size//2], np.abs(fft_result)[:data_array.size//2])
ax2.set_xlabel('Frequency (Hz)')
ax2.set_ylabel('Amplitude')

# Adjust the spacing between subplots
plt.subplots_adjust(hspace=0.4)

# Show the plot
plt.show()

'''# Extract the maximum frequency and amplitude
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
ax.text(0.05, 0.65, sinad_text, transform=ax.transAxes, fontsize=10, verticalalignment='top')'''

plt.show()
