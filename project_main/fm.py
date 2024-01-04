from scipy.signal import butter, filtfilt
import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':
    # set parameters
    fs = 44100 # 44.1kHz
    fc = 1e4 # 10kHz
    fm = 1e3 # 1kHz
    beta = 5 # modulation index
    duration = 1 # 1 sec
    t = np.arange(0, duration, 1/fs)
    # create carrier signal
    carrier = np.cos(2*np.pi*fc*t)
    # create modulating signal
    message = np.cos(2*np.pi*fm*t)
    # create FM signal
    integral_of_message = np.