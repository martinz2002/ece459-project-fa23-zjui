import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, freqz

def butter_lowpass(cutoff_frequency, fs, order=4):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff_frequency / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff_frequency, fs, order=4):
    b, a = butter_lowpass(cutoff_frequency, fs, order=order)
    y = lfilter(b, a, data)
    return y

def plot_frequency_response(cutoff_frequency, fs, order=4):
    b, a = butter_lowpass(cutoff_frequency, fs, order=order)
    w, h = freqz(b, a, worN=8000, fs=fs)

    plt.figure()
    plt.plot(0.5 * fs * w / np.pi, np.abs(h), 'b')
    plt.axvline(cutoff_frequency, color='red', linestyle='--', label='Cutoff Frequency')
    plt.title("Butterworth Lowpass Filter Frequency Response")
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Gain')
    plt.legend()
    plt.grid()
    plt.show()

# Set the cutoff frequency and sampling frequency
cutoff_frequency = 100.0  # Hz
fs = 1000.0  # Hz

# Plot the frequency response
plot_frequency_response(cutoff_frequency, fs)
