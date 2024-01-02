from scipy import signal
from matplotlib import pyplot as plt
import numpy as np

plt.rcParams['font.family'] = 'Palatino Linotype'
plt.rcParams['font.size'] = 11
plt.rcParams['figure.figsize'] = [6,3]
plt.rcParams['legend.loc'] = 'best'

for n in range(2, 9):
    b, a = signal.butter(n, 100, 'low', analog=True)
    w, h = signal.freqs(b, a)
    plt.semilogx(w, 20 * np.log10(abs(h)), label='n = %d' % n)

plt.title('Butterworth LPF frequency response')
plt.xlabel('Frequency [rad/sec]')
plt.xlim(40, 300)
plt.ylabel('Amplitude [dB]')
plt.ylim(-80, 10)
# plt.margins(0, 0.1)
plt.grid(which='both', axis='both')
plt.axvline(100, color='green') # cutoff frequency
plt.axhline(-3, color='black', linestyle='--') # -3db
plt.legend()
plt.tight_layout()
plt.show()

for n in range(2, 9):
    b, a = signal.butter(n, 100, 'low', analog=True)
    w, h = signal.freqs(b, a)
    plt.plot(w, 20 * np.abs(h), label='n = %d' % n)

plt.title('Butterworth LPF frequency response')
plt.xlabel('Frequency [rad/sec]')
plt.ylabel('Amplitude')
# plt.margins(0, 0.1)
plt.grid(which='both', axis='both')
plt.axvline(100, color='green') # cutoff frequency
# plt.axhline(-3, color='black', linestyle='--') # -3db
plt.legend()
plt.tight_layout()
plt.show()

for n in range(2, 9):
    b, a = signal.butter(n, [100, 200], btype='bandpass', analog=True)
    w, h = signal.freqs(b, a)
    plt.semilogx(w, 20 * np.log10(abs(h)), label='n = %d' % n)

plt.title('Butterworth BPF frequency response')
plt.xlabel('Frequency [rad/sec]')
plt.ylabel('Amplitude [dB]')
plt.xlim(50, 500)
plt.ylim(-100, 10)
# plt.margins(0, 0.1)
plt.grid(which='both', axis='both')
plt.axvline(100, color='green') # cutoff frequency
plt.axvline(200, color='green') # cutoff frequency
plt.axhline(-3, color='black', linestyle='--') # -3db
plt.legend()
plt.tight_layout()
plt.show()

for n in range(2, 9):
    b, a = signal.butter(n, [100, 200], btype='bandpass', analog=True)
    w, h = signal.freqs(b, a)
    plt.plot(w, 20 * np.abs(h), label='n = %d' % n)

plt.title('Butterworth BPF frequency response')
plt.xlabel('Frequency [rad/sec]')
plt.ylabel('Amplitude')
# plt.margins(0, 0.1)
plt.grid(which='both', axis='both')
plt.axvline(100, color='green') # cutoff frequency
plt.axvline(200, color='green') # cutoff frequency
# plt.axhline(-3, color='black', linestyle='--') # -3db
plt.legend()
plt.tight_layout()
plt.show()