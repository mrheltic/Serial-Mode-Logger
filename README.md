# Data Acquisition and Analysis Project

This project is a Python script for acquiring and analyzing data from a microcontroller via a serial connection. The script collects data, performs statistical analysis, and visualizes the results.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Code Explanation](#code-explanation)
- [Performance Evaluation](#performance-evaluation)
- [Problems](#problems)
- [Contributing](#contributing)
- [License](#license)

## Installation

The script requires Python and several libraries. You can install these libraries using pip:

```python
pip install pyserial numpy matplotlib scipy
```

## Usage

The script opens a serial connection, sends a start command to the microcontroller, and waits for the microcontroller to send the current mode and data rate. It then starts acquiring data.

Data is read from the serial connection, and measurements are added to an array. When the array reaches the desired length (defined by the data rate), it is added to a matrix, and the array is reset.

If the script is interrupted (e.g., by a KeyboardInterrupt), it saves the matrix to a text file and closes the serial connection.

The script then performs several analyses on the data:

- It calculates the mean and standard deviation for each row of the matrix.
- It calculates the Fast Fourier Transform (FFT) of the data.

Finally, it creates several graphs:

- A graph of the data over time.
- A graph of the mean value over time.
- A graph of the standard deviation over time, with a 95% confidence interval.
- A graph of the FFT of the data.

## Code explanation

This Python script is designed to acquire and analyze data from a microcontroller via a serial connection. The script collects data, performs statistical analysis, and visualizes the results.

### Importing Libraries

These lines import the necessary libraries. `serial` is used for serial communication, `time` for timing functions, `numpy` for data processing, and `matplotlib` for data visualization, and `scipy` for useful tools about statistics.

```python
import serial
import time
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
```

### Establishing Serial Connection

The script establishes a serial connection with the microcontroller. The port and baud rate are specified in the `serial.Serial()` function:

> [!IMPORTANT]
> the port must be set in accordance with the port used by the microcontroller

```python
ser = serial.Serial('COM14', 250000)
```

### Sending Start Command

The script sends a start command (`'s'`) to the microcontroller and waits for the microcontroller to send the current mode and data rate:

```python
time.sleep(1)
ser.write(b'F')
time.sleep(3)
```

### Receiving Data

The script then enters a loop where it waits for data from the microcontroller. When data is available, it reads the data, decodes it, and checks if it has received the "START" command:

```python
while True:
    if ser.in_waiting > 0:
        line = ser.readline().decode('utf-8').strip()
        if line == "START":
            print("START command received!")
            break
```

### Data Acquisition

The script then enters another loop where it waits for the microcontroller to send the current mode and data rate. It prints these values and starts the data acquisition process:

```python
while True:
    if ser.in_waiting > 0:
        current_mode = ser.readline().decode('utf-8').strip()
        print("Current mode: " + current_mode)
        data_rate = int(ser.readline().decode('utf-8').strip()) # This value is stored to be used for the array's lenght
        print("Data rate: ", data_rate)

        print("\nStarting data acquisition...")
        break
```

### Data Collection

The script initializes a matrix and an array for the data, and a timestamp array. It then enters a loop where it continuously reads data from the microcontroller. When the start byte is verified, it reads the high and low bytes, merges them into a measurement, and adds the measurement to the array. When the array reaches the desired length (defined by the data rate), it adds the array to the matrix and resets the array:

```python
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
```

### Saving Data

If the script is interrupted (e.g., by a KeyboardInterrupt), it saves the matrix to a text file and closes the serial connection:

```python
except KeyboardInterrupt:
    # When the program is interrupted, save the matrix in a text file
    data_matrix = data_matrix[1:]  # Remove the first row (various errors)
    utils = 'Current mode: ' + current_mode + '\nData rate: ' + str(data_rate) + ' SPS\n\n'
    np.savetxt('data_matrix.txt', data_matrix, header=utils, fmt='%d')
    print("\n\n\n\n\nData saved in 'data_matrix.txt'")
finally:
    ser.close()  # Close the serial connection
```

### Data Analysis

The script then performs several analyses on the data:

- It calculates the mean and standard deviation for each row of the matrix.
- It calculates the Fast Fourier Transform (FFT) of the data.

### Data Visualization

Finally, it creates several graphs:

- A graph of the data over time.
- A graph of the mean value over time.
- A graph of the standard deviation over time, with a 95% confidence interval.
- A graph of the FFT of the data.

## Performance Evaluation

```python
While True:
    If there is data waiting in the serial buffer:
        Read the start byte
        If the start byte is equal to '\xCC':
            Read the high byte
            Read the low byte
            Merge the high and low bytes to form the measurement
            Append the measurement to the data array
            If the length of the data array is equal to the data rate:
                Append the data array to the data matrix
                Append the current time to the timestamp array
                Reset the data array
                Calculate the new time
                Print the time difference between the new time and the old time
                Update the old time with the new time

```

This code snippet is an empirical error evaluation, calculating the time taken to fill an array. The time should be as close to 1 as possible, with a maximum tolerance equal to the period that the ADC takes to make the measurement. 

The `while True` loop continuously checks if there is data waiting in the serial buffer. If data is available, it reads the start byte and verifies it. If the start byte is correct, it reads the high and low bytes of the measurement, merges them, and appends the result to the `data_array`. 

When `data_array` reaches the desired length (`data_rate`), it is appended to `data_matrix`, and the current time is appended to `timestamp_array`. `data_array` is then reset for the next set of measurements. 

The time taken to fill the array is printed to the console. This time is expected to be as close to 1 as possible, providing an empirical evaluation of the code's performance.

## Problems

```python
UnicodeDecodeError: 'utf-8' codec can't decode byte 0x86 in position 6: invalid start byte
```


> [!CAUTION]
> This script can generate an error like this after uploading cpp code on the ESP, simply re-execute the script


## Contributing

Contributions are welcome. Please open an issue to discuss your ideas before making changes.

## License

This project is licensed under the MIT License. See the LICENSE file for details.