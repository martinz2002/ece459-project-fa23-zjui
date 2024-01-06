# simulate an AWGN process
import matplotlib.pyplot as plt
import numpy as np
from scipy.fftpack import fft, fftshift

plt.rcParams['font.family'] = 'Palatino Linotype'
plt.rcParams['legend.loc'] = 'best'
plt.rcParams['axes.grid'] = True
plt.rcParams['figure.figsize'] = [5,2.5]

fs = 1000
N = 1e5

# generate a series of gaussian noise
noise = np.random.normal(size=int(N))
t = np.arange(0, N/fs, 1/fs)

# plot the noise
plt.figure()
plt.plot(t, noise, label='Noise')
plt.plot(t, np.mean(noise) * np.ones(len(t)), '--', label='Noise Mean')
plt.title('Gaussian Noise')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
plt.tight_layout()
plt.legend()
plt.show()

# plot the spectrum of the noise
f = np.linspace(-fs/2, fs/2, len(t))
noise_spectrum = fftshift(fft(noise))
plt.figure()
plt.plot(f, np.abs(noise_spectrum), label='Noise Magnitude')
# plot the average power density
plt.plot(f,np.mean(np.abs(noise_spectrum)) * np.ones(len(f)), '--', label='Average Noise Power Density')
plt.title('Spectrum of Gaussian Noise')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Magnitude')
plt.tight_layout()
plt.legend()
plt.show()
