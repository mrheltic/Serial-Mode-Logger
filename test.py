import serial
import time
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

# Open the serial connection (replace 'COM8' with your serial port)
ser = serial.Serial('COM14', 250000)

# Send the start command to the microcontroller
time.sleep(1)
ser.write(b'F')
time.sleep(3)

# Wait for the microcontroller to send current mode and data rate
while True:
    if ser.in_waiting > 0:
        line = ser.readline().decode('utf-8').strip()
        if line == "START":
            print("START command received!")
            break

# Wait for the microcontroller to send current mode and data rate
while True:
    if ser.in_waiting > 0:
        current_mode = ser.readline().decode('utf-8').strip()
        print("Current mode: " + current_mode)
        data_rate = int(ser.readline().decode('utf-8').strip())
        print("Data rate: ", data_rate)

        print("\nStarting data acquisition...")
        break

# Initialize the matrix and array for the data
data_matrix = []
data_array = []
timestamp_array = []

time_old = time.time()

try:
    while True:
        if ser.in_waiting > 0:
            start_byte = ser.read(1)  # Read the start byte
            if start_byte == b'\xCC':  # Verify the start byte
                high_byte = ser.read(1)  # Read the high byte
                low_byte = ser.read(1)  # Read the low byte
                measurement = (ord(high_byte) << 8) | ord(low_byte)  # Merge the bytes
                data_array.append(measurement)  # Add the measurement to the array
                if len(data_array) == data_rate:  # If the array has reached the desired length
                    data_matrix.append(data_array)  # Add the array to the matrix
                    timestamp_array.append(time.time())
                    data_array = []  # Reset the array
                    new_time = time.time()
                    print('Time:', new_time - time_old)
                    time_old = new_time
except KeyboardInterrupt:
    # When the program is interrupted, save the matrix in a text file
    data_matrix = data_matrix[1:]  # Remove the first row (various errors)
    utils = 'Current mode: ' + current_mode + '\nData rate: ' + str(data_rate) + ' SPS\n\n'
    np.savetxt('data_matrix.txt', data_matrix, header=utils, fmt='%d')
    print("\n\n\n\n\nData saved in 'data_matrix.txt'")
finally:
    ser.close()  # Close the serial connection

    # Flatten the matrix into a one-dimensional array
    data_array = np.concatenate(data_matrix)

    # Create a time array. Every data-rate samples correspond to one second.
    time_array = np.arange(len(data_array)) / data_rate

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
