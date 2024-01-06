from scipy.io import wavfile
import matplotlib.pyplot as plt
from scipy.fftpack import fft, fftshift
import numpy as np

plt.rcParams['font.family'] = 'Palatino Linotype'
plt.rcParams['legend.loc'] = 'best'
plt.rcParams['axes.grid'] = True
plt.rcParams['figure.figsize'] = [5,2.5]

fs, message = wavfile.read('audio.wav')
t = np.arange(0, len(message)/fs, 1/fs)
f = np.arange(-fs/2, fs/2, fs/len(t))
print(f)
message_spectrum = fftshift(fft(message))

plt.figure()
plt.plot(t, message)
plt.title('Message Signal')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
plt.tight_layout()
plt.show()

plt.figure()
plt.plot(f, np.abs(message_spectrum))
plt.title('Message Signal Spectrum')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Amplitude')
plt.tight_layout()
plt.show()

plt.figure()
plt.plot(f, np.abs(message_spectrum))
plt.title('Message Signal Spectrum (< 8 kHz)')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Amplitude')
plt.xlim([-8000, 8000])
plt.tight_layout()
plt.show()
