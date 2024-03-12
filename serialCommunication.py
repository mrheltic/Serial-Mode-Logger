import serial
import time
import numpy as np
import os
import matplotlib.pyplot as plt
import scipy.stats as stats

# Initialize the matrix and array for the data
data_matrix = []
data_array = []
time_array = []

# Initialize variables for time evaluation
start_time = time.time()
current_time = time.time()
evaluation_time = []

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
        k_value = float(ser.readline().decode('utf-8').strip())
        print("K value: ", k_value)
        offset = float(ser.readline().decode('utf-8').strip())
        print("O value: ", offset)
        data_rate = int(ser.readline().decode('utf-8').strip())
        print("Data rate: ", data_rate)
        factor= float(ser.readline().decode('utf-8').strip())
        print("Conversion factor: ", factor)
 
        print("\nStarting data acquisition...")
        break

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
                    data_array = []  # Reset the array
                    time_array.append(str(ser.readline().decode('utf-8').strip()))
                    current_time = time.time()
                    evaluation_time.append(current_time - start_time)
                    start_time = current_time
                   
                    
                   
except KeyboardInterrupt:
    # If the user stops the program
    print("Data acquisition stopped by the user")

finally:
    ser.close()  # Close the serial connection

    # Print the current mode and data rate
    utils = "Current measure: " + str(current_mode) + "\n" + "Gain: " + str(k_value) + "\n" + "Offset: " + str(
        offset) + "\n" + "Array length (Sample rate): " + str(data_rate) + "\n" + "factor: " + str(
        factor)

    # Saving the last timestamp for dynamic plot
    last_timestamp = time_array[-1]

    #cast data_matrix to str
    data_matrix = np.array(data_matrix).astype(str)
    time_array = np.array(time_array).astype(str)

    # Merging the time array with the data matrix (time array as first column)
    saving_matrix = np.column_stack((time_array, data_matrix))

    #add a first row into saving_matrix with utils
    #saving_matrix = np.vstack((utils, saving_matrix))

    # Check if the file already exists
    if os.path.exists("dataStorage.txt"):
        # Generate a unique filename
        filename = "dataStorage_" + str(time.time()) + ".txt"
    else:
        filename = "dataStorage.txt"

    # Save the data to the file
    np.savetxt(filename, saving_matrix, delimiter=' ', comments='', fmt='%s', header=utils, encoding='utf-8')
    print("\n\n\n\n\nData saved in '" + filename + "'")

    # Adding the last timestamp for dynamic plot in the file
    with open("dataStorage.txt", "a") as file:
        file.write(str(last_timestamp))

    # Evaluating the mean time for each data acquisition
    mean_time = np.mean(evaluation_time)
    print("Mean time: ", mean_time)