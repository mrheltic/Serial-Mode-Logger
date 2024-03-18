import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import Conversion.conversion as conversion
import DataExtraction.extractdata as extractdata


datastore = './Dataset/SinusoidalWave/sinusoidal9.ds32'
# Extract the data
currentmode, k_value, offset, data_rate, factor, timestamp, data_matrix = extractdata.extract_data(datastore)

# Set the conversion factor to 1 if you're using the FSR of ADC
#factor = 1

# Convert the data
data_matrix, data_array = conversion.convert_data(currentmode, k_value, factor, offset, data_matrix)

# Calculate the mean and standard deviation for each row
mean_values = np.zeros(data_matrix.shape[0])
std_values = np.zeros(data_matrix.shape[0])

for i in range(0, data_matrix.shape[0]):
    mean_values[i] = np.mean(data_matrix[i, :])
    std_values[i] = np.std(data_matrix[i, :])

#remove the last value of the timestamp
timestamp = timestamp[:-1]

# Plot the entire dataset
plt.figure(figsize=(10, 5), dpi=100)
#plt.plot(range(len(data_array)), data_array, 'o', markersize=2)
plt.plot(data_array)
plt.title('Data')
plt.xlabel('Sample number')
plt.ylabel('Value (V)')
plt.grid()
plt.show()



# Control if timestamp and mean_values have the same length
if len(timestamp) != len(mean_values) or len(timestamp) != len(std_values) or len(mean_values) != len(std_values):
    timestamp = np.arange(0, len(mean_values), 1)


plt.figure(figsize=(10, 10), dpi=100)

# Create the mean graphs
plt.subplot(2, 1, 1)

plt.plot(timestamp, mean_values)
plt.xticks([min(timestamp),max(timestamp)])
plt.title('Mean value over time')
plt.xlabel('Time (s)')
plt.ylabel('Mean value (V)')
plt.grid()

# Create the standard deviation graph with error bands
plt.subplot(2, 1, 2)
plt.plot(timestamp, std_values)
plt.fill_between(timestamp,
                 std_values - stats.t.ppf(0.975, df=data_rate - 1) * std_values / np.sqrt(data_rate),
                 std_values + stats.t.ppf(0.975, df=data_rate - 1) * std_values / np.sqrt(data_rate), color='gray',
                 alpha=0.5)
plt.xticks([min(timestamp),max(timestamp)])
plt.title('STD with 95% confidence interval')
plt.xlabel('Time (s)')
plt.ylabel('Standard deviation (V)')

# Show the graphs
plt.tight_layout()
plt.grid()

plt.show()
