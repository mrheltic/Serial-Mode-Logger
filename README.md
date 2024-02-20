# Code Explanation

This Python script is designed to communicate with Logger via a serial connection, collect data, and visualize it. Here's a detailed breakdown of how the code works:

```python
import serial
import time
import numpy as np
import matplotlib.pyplot as plt
```
These lines import the necessary libraries. `serial` is used for serial communication, `time` for timing functions, `numpy` for data processing, and `matplotlib` for data visualization.

```python
ser = serial.Serial('COM14', 250000)
```
This line opens a serial connection on port 'COM14' with a baud rate of 250000.

```python
time.sleep(1)
ser.write('s'.encode())
time.sleep(3)
```
The program pauses for one second, sends the 's' command to the microcontroller to start data transmission, and then pauses for another three seconds to allow the microcontroller to prepare.

```python
while True:
    if ser.in_waiting > 0:
        line = ser.readline().decode('utf-8').rstrip()
        if line == "START":
            print("START command received!")
            break
```
This block waits for the microcontroller to send the "START" command, indicating it's ready to start data transmission.

```python
while True:
    if ser.inWaiting():
        current_mode = ser.readline().decode().strip()
        print("Current mode: " + current_mode)
        data_rate = int(ser.readline().decode().strip())
        print("Data rate: ", data_rate)

        print("\nStarting data acquisition...")
        break
```
This block reads the current mode and data rate from the microcontroller.

```python
data_matrix = []
data_array = []
```
These lines initialize the matrix and array for storing the data.

```python
try:
    while True:
        if ser.inWaiting():
            start_byte = ser.read(1)  # Read the start byte
            if start_byte == b'\xCC':  # Verify the start byte
                high_byte = ser.read(1)  # Read the high byte
                low_byte = ser.read(1)  # Read the low byte
                measurement = (ord(high_byte) << 8) | ord(low_byte)  # Combine the bytes
                data_array.append(measurement)  # Add the measurement to the array
                if len(data_array) == data_rate:  # If the array has reached the desired length
                    data_matrix.append(data_array)  # Add the array to the matrix
                    data_array = []  # Reset the array
```
This block reads data from the microcontroller. Each data point consists of two bytes, a high byte and a low byte, which are combined to form a single measurement. The measurements are added to `data_array` until its length reaches `data_rate`, at which point `data_array` is added to `data_matrix` and then reset.

```python
except KeyboardInterrupt:
    data_matrix = data_matrix[1:]  # Remove the first row (various errors)
    utils = 'Current mode: ' + current_mode + '\nData rate: ' + str(data_rate) + ' SPSw\n\n'
    np.savetxt('data_matrix.txt', data_matrix, header=utils, fmt='%d')
    print("Data saved in 'data_matrix.txt'")
finally:
    ser.close()  # Close the serial connection
```
If the program is interrupted, it saves the data matrix to a text file and closes the serial connection.

```python
data_array = np.concatenate(data_matrix)
```
This line flattens the data matrix into a one-dimensional array.

```python
time_array = np.arange(len(data_array)) / data_rate
```
This line creates an array of times. Each `data_rate` samples correspond to one second.

```python
plt.figure(figsize=(10, 6), dpi=500)
plt.plot(time_array, data_array)
plt.title('Data Plot')
plt.xlabel('Time (s)')
plt.ylabel('Value')
plt.show()
```
These lines create a plot of the data and display it. The x-axis represents time in seconds, and the y-axis represents the value of the data points.

# Problems

```python
UnicodeDecodeError: 'utf-8' codec can't decode byte 0x86 in position 6: invalid start byte
```


> [!CAUTION]
> This script can generate an error like this after uploading cpp code on the ESP, simply re-execute the script
