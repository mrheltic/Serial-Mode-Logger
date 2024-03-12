from datetime import datetime, timedelta
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import Conversion.conversion as conversion
import DataExtraction.extractdata as extractdata


datastore = './Dataset/Ramp/ramp3.txt'

# Extract the data
currentmode, k_value, offset, data_rate, factor, timestamp, data_matrix = extractdata.extract_data(datastore)

# Convert the data
data_matrix, data_array = conversion.convert_data(currentmode, k_value, factor, offset, data_matrix)

# Calculate the mean and standard deviation for each row
mean_values = np.zeros(data_matrix.shape[0])
std_values = np.zeros(data_matrix.shape[0])

for i in range(1, data_matrix.shape[0]):
    mean_values[i] = np.mean(data_matrix[i, :])
    std_values[i] = np.std(data_matrix[i, :])

# Plot the entire dataset
    
plt.figure(figsize=(10, 5), dpi=100)
plt.plot(data_array)
plt.title('Entire dataset')
plt.xlabel('Sample number')
plt.ylabel('Value')
plt.grid()
plt.show()

# Remove the first value of mean_values and std_values
mean_values = mean_values[1:]
std_values = std_values[1:]

# Control if timestamp and mean_values have the same length
if len(timestamp) != len(mean_values) or len(timestamp) != len(std_values) or len(mean_values) != len(std_values):
    timestamp = np.arange(0, len(mean_values), 1)

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
