import numpy as np
#import matplotlib.pyplot as plt
import scipy.stats as stats
import pylab as plt
import scipy.optimize
from datetime import datetime, timedelta


datastore = './Dataset/sinusoidal wave 200hz.txt'

# Create a sinusoidal wave from 0 to 4, 200hz
amplitude = 1.775
offset_sin = 1.775
period = 87


#export the current mode
currentmode=np.loadtxt(datastore, dtype='str', max_rows=1)[-1]

#export the k value
k_value=float(np.loadtxt(datastore, dtype='float', usecols=(1), skiprows=1,max_rows=1))

#export the offset
offset=np.loadtxt(datastore, dtype='float', usecols=(1), skiprows=2, max_rows=1)

#export the data rate
data_rate=int(np.loadtxt(datastore, dtype='int', usecols=(4), skiprows=3, max_rows=1))

#export the conversion factor
factor=1

#export the timestamp
timestamp = np.loadtxt(datastore, dtype='str', usecols=(0), skiprows=6)

#export the dataset(reversed) without the 1st array for a problem
data_matrix = np.loadtxt(datastore, dtype='int', skiprows=6, usecols=np.arange(1, data_rate+1),max_rows=len(timestamp)-1)

# Remove the first column from matrix (various errors)
data_matrix = data_matrix[5:]

#create a 1D array from the matrix
data_array = np.concatenate(data_matrix)

#apply the conversion factor
gain=k_value*factor
data_matrix = np.dot(data_matrix,gain)-offset-offset_sin

# Flatten the matrix
data_array = data_matrix.flatten()


# Build the periodical sinusoidal signal considering the amplitude and offset
sinusoidal = amplitude * np.sin(np.linspace(0, 2*np.pi, period))


# Find the period in the data array and extract it
max_index = np.argmax(data_array)
data_array_period = data_array[max_index:max_index + round(2*period)]

# Shift the data array to right so that the period starts at the beginning of the array
data_array_period = np.roll(data_array_period, -abs(np.argmin(data_array_period)))

for i in range(0, len(data_array_period)):
    if data_array_period[i]*data_array_period[i-1] < 0:
        start_data_array_period = i-1
        break

# Remove the first part of the data array period
data_array_period = data_array_period[start_data_array_period:]

for i in range(int(0.9*period), len(data_array_period)):
    if data_array_period[i]*data_array_period[i-1] < 0:
        end_data_array_period = i-1
        break

# Remove the last part of the data array period
data_array_period = data_array_period[:end_data_array_period+(sinusoidal.size - end_data_array_period)]


def fit_sin(tt, yy):
    '''Fit sin to the input time sequence, and return fitting parameters "amp", "omega", "phase", "offset", "freq", "period" and "fitfunc"'''
    tt = np.array(tt)
    yy = np.array(yy)
    ff = np.fft.fftfreq(len(tt), period)   # assume uniform spacing
    Fyy = abs(np.fft.fft(yy))

    



    guess_freq = abs(ff[np.argmax(Fyy[1:])+1])   # excluding the zero frequency "peak", which is related to offset
    guess_amp =np .std(yy) * 2.**0.5
    guess_offset = np.mean(yy)
    guess = np.array([guess_amp, 2.*np.pi*guess_freq, 0., guess_offset])

    def sinfunc(t, A, w, p, c):  return A * np.sin(w*t + p) + c
    popt, pcov = scipy.optimize.curve_fit(sinfunc, tt, yy, p0=guess)
    A, w, p, c = popt
    f = w/(2.*np.pi)
    fitfunc = lambda t: A * np.sin(w*t + p) + c
    return {"amp": A, "omega": w, "phase": p, "offset": c, "freq": f, "period": 1./f, "fitfunc": fitfunc, "maxcov": np.max(pcov), "rawres": (guess,popt,pcov)}

N, amp, omega, phase, offset, noise = 500, 1., 2., .5, 4., 3
#N, amp, omega, phase, offset, noise = 50, 1., .4, .5, 4., .2
#N, amp, omega, phase, offset, noise = 200, 1., 20, .5, 4., 1
tt = np.linspace(0,10, 87)


yy = data_array_period
res=fit_sin(tt, yy)



# Plot one iteration of data array period and compare it to the generated sinusoidal signal, until the period is found
fig, (ax1, ax2) = plt.subplots(2, 1)



# Adding labels
ax1.set_xlabel('Sample number')
ax1.set_ylabel('Amplitude (V)')

# Adding title and legend
ax1.set_title('Sinusoidal signal')
ax1.legend(['Data array period'], loc='upper right')
plt.plot(tt, res["fitfunc"](tt), "r-", label="y fit curve", linewidth=2)
ax1.plot(sinusoidal)
















# Adding title and legend
ax1.set_title('Sinusoidal signal')
ax1.legend(['Data array period'], loc='upper right')
ax1.plot(data_array_period)
ax1.plot(sinusoidal)

# Adding labels
ax2.set_xlabel('Sample number')
ax2.set_ylabel('Squared Error')

# Adding title
ax2.set_title('Squared Error between data array period and ramp signal')

# Plot the difference between the data array period and the sinusoidal signal
ax2.plot((data_array_period - sinusoidal)**2)

plt.tight_layout()
plt.show()


# Plot the data array
#fig, ax = plt.subplots()
#ax.plot(data_array)
#plt.show()

