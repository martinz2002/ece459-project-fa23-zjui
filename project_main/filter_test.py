from scipy.signal import butter, freqs, lfilter, filtfilt
import numpy as np
import matplotlib.pyplot as plt

# create an analog lpf
fs = 1000
b, a = butter(8, [40, 80], btype='band', fs=fs)
duration = 0.5
t = np.arange(0, duration, 1/fs)
message = np.sin(2*np.pi*50*t) + np.sin(2*np.pi*70*t) + np.sin(2*np.pi*90*t)

# filter the signal
filtered = filtfilt(b, a, message)
plt.plot(t, filtered, 'g-', linewidth=2, label='filtered data')
plt.plot(t, message, 'b-', label='data')
plt.xlabel('Time [sec]')
plt.xlim(0, 0.1)
plt.grid()
plt.legend()
plt.show()

# print(w, h)