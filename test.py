import serial
import time
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

# Open the serial connection (replace '/dev/ttyACM0' with your serial port)
ser = serial.Serial('COM14', 250000)


# Send the start command to the microcontroller
time.sleep(1)
ser.write('s'.encode())
time.sleep(3)

# Wait for the microcontroller to send current mode and data rate
while True:
    if ser.in_waiting > 0:
        line = ser.readline().decode('utf-8').rstrip()
        if line == "START":
            print("START command received!")
            break

# Wait for the microcontroller to send current mode and data rate
while True:
    if ser.inWaiting():
        current_mode = ser.readline().decode().strip()
        print("Current mode: " + current_mode)
        data_rate = int(ser.readline().decode().strip())
        print("Data rate: ", data_rate)

        print("\nStarting data acquisition...")
        break


# Initialize the matrix and array for the data
data_matrix = []
data_array = []

try:
    while True:
        if ser.inWaiting():
            start_byte = ser.read(1)  # Read the start byte
            if start_byte == b'\xCC':  # Verify the start byte
                high_byte = ser.read(1)  # Read the high byte
                low_byte = ser.read(1)  # Read the low byte
                measurement = (ord(high_byte) << 8) | ord(low_byte)  # Merge the bytes
                data_array.append(measurement)  # Add the measurement to the array
                if len(data_array) == data_rate:  # If the array has reached the desired length
                    data_matrix.append(data_array)  # Add the array to the matrix
                    data_array = []  # Reset the array
except KeyboardInterrupt:
    # When the program is interrupted, save the matrix in a text file
    data_matrix = data_matrix[1:]  # Remove the first row (various errors)
    utils = 'Current mode: ' + current_mode + '\nData rate: ' + str(data_rate) + ' SPSw\n\n'
    np.savetxt('data_matrix.txt', data_matrix, header=utils, fmt='%d')
    print("\n\n\n\n\nData saved in 'data_matrix.txt'")
finally:
    ser.close()  # Close the serial connection

    # Flatten the matrix into a one-dimensional array
    data_array = np.concatenate(data_matrix)

    # Create a time array. Every data-rate samples correspond to one second.
    time_array = np.arange(len(data_array)) / data_rate

    # Create a graph
    plt.figure(figsize=(10, 6), dpi=500)
    plt.plot(time_array, data_array)
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
    plt.figure(figsize=(10, 6), dpi=500)

    # Create the mean graph
    plt.subplot(2, 1, 1)
    plt.plot(time_array[::data_rate], mean_values)
    plt.title('Media per ogni istante')
    plt.xlabel('Tempo (s)')
    plt.ylabel('Media')

    # Create the standard deviation graph with error bands
    plt.subplot(2, 1, 2)
    plt.plot(time_array[::data_rate], std_values)
    plt.fill_between(time_array[::data_rate],
                     std_values - stats.t.ppf(0.975, df=data_rate - 1) * std_values / np.sqrt(data_rate),
                     std_values + stats.t.ppf(0.975, df=data_rate - 1) * std_values / np.sqrt(data_rate), color='gray',
                     alpha=0.5)
    plt.title('Deviazione standard per ogni istante con bande di errore')
    plt.xlabel('Tempo (s)')
    plt.ylabel('Deviazione standard')

    # Show the graphs
    plt.tight_layout()
    plt.show()