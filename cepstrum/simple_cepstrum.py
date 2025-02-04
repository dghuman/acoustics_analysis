import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mplt
from scipy import signal, fft
import soundfile as sf
import argparse

def load_data():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--infile')
    args = parser.parse_args()
    music_file = args.infile
    return music_file

def cepstrum(waveform, rate):
    t = np.arange(0, len(waveform)/rate, 1/rate)
    P = fft.fft(waveform)
    Cc = fft.ifft(np.log(P))
    return t, Cc, 
    

def main():
    music_file = load_data()
    data, rate = sf.read(music_file)
    data = data[:,1]
    #data = data[int(rate*5):int(rate*5.78)]
    t, Cc = cepstrum(data, rate)
    f, Pxx = signal.periodogram(data, rate, 'flattop', scaling='spectrum')
    fig, axs = plt.subplots(3, 1)
    axs[0].plot(t, data)
    axs[0].set_xlabel('Time (s)')
    axs[0].set_ylabel('Amplitude')
    axs[1].plot(f/1e3, np.log(Pxx))
    axs[1].set_xlabel('Frequency (kHz)')
    axs[1].set_ylabel('Log(fft)')
    axs[2].plot(t, np.abs(Cc))
    axs[2].set_xlabel('Quefrency (s)')
    axs[2].set_ylabel('Cepstrum')
    plt.show()

if __name__ == '__main__':
    main()
