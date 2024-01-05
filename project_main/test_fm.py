import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert
from scipy.io import wavfile

def generate_multi_tone_signal(duration, sample_rate, frequencies, amplitudes):
    # t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    # signal = np.sum([amplitude * np.sin(2 * np.pi * frequency * t) for frequency, amplitude in zip(frequencies, amplitudes)], axis=0)
    # read signal from wav
    sample_rate, signal = wavfile.read('audio.wav')
    duration = len(signal) / sample_rate
    padded_signal = np.zeros(3 * int(sample_rate * duration))
    # signal = np.zeros(int(sample_rate * duration))
    # signal[0:int(0.1*duration*sample_rate)] = 1
    # signal[int(0.3*duration*sample_rate):int(0.4*duration*sample_rate)] = 1
    # signal[int(0.6*duration*sample_rate):int(0.8*duration*sample_rate)] = 1
    padded_signal[int(sample_rate * duration):2*int(sample_rate * duration)] = signal
    t = np.linspace(0, 3*duration, 3*len(signal), endpoint=False)
    return t, padded_signal

def fm_modulation(input_signal, carrier_frequency, modulation_index, sample_rate):
    t = np.linspace(0, len(input_signal) / sample_rate, len(input_signal), endpoint=False)
    phase = 2 * np.pi * carrier_frequency * t + modulation_index * np.cumsum(input_signal) / sample_rate
    modulated_signal = np.sin(phase)
    return t, modulated_signal

def fm_demodulation(modulated_signal, sample_rate):
    analytic_signal = hilbert(modulated_signal)
    instantaneous_phase = np.unwrap(np.angle(analytic_signal))
    instantaneous_frequency = np.diff(instantaneous_phase) / (2 * np.pi) * sample_rate
    return instantaneous_frequency

def plot_signals(t, input_signal, modulated_signal, demodulated_signal, title1, title2, title3):
    plt.figure(figsize=(12, 8))

    plt.subplot(3, 1, 1)
    plt.plot(t, input_signal)
    plt.title(title1)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')

    plt.subplot(3, 1, 2)
    plt.plot(t, modulated_signal)
    plt.title(title2)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')

    plt.subplot(3, 1, 3)
    plt.plot(t[:-1], demodulated_signal)
    plt.title(title3)
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    # plt.ylim(990,1010)

    plt.tight_layout()
    plt.show()

# Parameters
duration = 1  # seconds
sample_rate = 10000  # Hz
carrier_frequency = 1000  # Hz
modulation_index = 5.0
frequencies = [10, 50, 100, 200]  # Hz
amplitudes = [1, 0.5]

# Generate multi-tone input signal
t, input_signal = generate_multi_tone_signal(duration, sample_rate, frequencies, amplitudes)

# FM modulation
t_mod, modulated_signal = fm_modulation(input_signal, carrier_frequency, modulation_index, sample_rate)

# FM demodulation
demodulated_signal = fm_demodulation(modulated_signal, sample_rate)

# Plot signals
plot_signals(t, input_signal, modulated_signal, demodulated_signal,
            'Multi-Tone Input Signal', 'FM Modulated Signal', 'Demodulated Signal')
