# Find the index of the minimum value in the data array
min_index = np.argmin(data_array)

# Plot one iteration of data array starting from the minimum value and compare it to the generated ramp
fig, ax = plt.subplots()
ax.plot(ramp, label='Generated Ramp')
shifted_data_array = np.roll(data_array, -min_index)
ax.plot(shifted_data_array[:period], label='Shifted Data Array')
ax.legend()
plt.show()