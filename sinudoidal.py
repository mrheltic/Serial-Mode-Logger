import numpy as np
#import matplotlib.pyplot as plt
import scipy.stats as stats
import pylab as plt
import scipy.optimize
from datetime import datetime, timedelta
import Conversion.conversion as conversion
import DataExtraction.extractdata as extractdata
import DataExtraction.extractperiod as extractperiod


datastore = './Dataset/SinusoidalWave/sinusoidal7.ds32'

# Create a sinusoidal wave from 0 to 4, 200hz
amplitude = 2
offset_sin = 2
period = 2.15

# Extract the data
currentmode, k_value, offset, data_rate, factor, timestamp, data_matrix = extractdata.extract_data(datastore)

# Set the conversion factor to 1 if you're using the FSR of ADC
factor = 1

# Convert the data
data_matrix, data_array = conversion.convert_data(currentmode, k_value, factor, offset, data_matrix)

# Setting the offset for the sinusoidal wave
data_array = data_array - offset_sin - offset

data_array_period, sinusoidal= extractperiod.extract_period(data_array, period, data_rate, amplitude)



def fit_sin(tt, yy):
    '''Fit sin to the input time sequence, and return fitting parameters "amp", "omega", "phase", "offset", "freq", "period" and "fitfunc"'''
    tt = np.array(tt)
    yy = np.array(yy)
    ff = np.fft.fftfreq(len(tt), tt[1]-tt[0])   # assume uniform spacing
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


tt = np.linspace(0,10, len(data_array_period))



yy = data_array_period
res=fit_sin(tt, yy)
res2=fit_sin(tt, sinusoidal)


fig, (ax1, ax2) = plt.subplots(2, 1)

# Plot the linear model of the data array period and the ramp signal
ax1.plot(res2["fitfunc"](tt), "b-", label="y true curve", linewidth=2, color='blue', linestyle='dashed')
ax1.plot(res['fitfunc'](tt), "g-", label="y fit curve", linewidth=2,color='red')


# Adding labels
ax1.set_xlabel('Sample number')
ax1.set_ylabel('Amplitude (V)')

# Adding title and legend
ax1.set_title('Sinusoidal signal')
ax1.legend(['Data array period', 'Sinusoidal signal'])

# Plot the error between the data array period and the ramp signal both with linear model
ax2.plot((res2["fitfunc"](tt)) - (res["fitfunc"](tt)))

# Adding labels
ax2.set_xlabel('Sample number')
ax2.set_ylabel('Error')

# Adding title
ax2.set_title(' Error between data array period and ramp signal')


plt.tight_layout()
plt.show()


