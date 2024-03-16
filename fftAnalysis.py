import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import Conversion.conversion as conversion
import DataExtraction.extractdata as extractdata

datastore = './Dataset/SinusoidalWave/sinusoidal9.ds32'

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

'''# Extract the harmonics of the max frequency
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
'''
'''# Calculate the noise power from the other frequencies
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
'''
