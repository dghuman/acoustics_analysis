import matplotlib.pyplot as plt
import matplotlib as mplt
import numpy as np
import soundfile as sf
import os
import argparse
import hist
from hist import Hist
from scipy import signal

# Simple peakfinder+frequency analysis to see what the peaks look like on an average incoming waveform

def load_data():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--infile')
    args = parser.parse_args()
    music_file = args.infile
    return music_file

def load_dir():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--indir')
    args = parser.parse_args()
    music_files = args.indir
    return music_files

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = signal.butter(N=order, Wn=[lowcut, highcut] , btype='band', fs=fs)
    y = signal.lfilter(b, a, data)
    return y

def freq_spec(waveform, rate):
    f, Pxx_spec = signal.periodogram(waveform, rate, 'flattop', scaling='spectrum')    
    peak_index = np.argmax(Pxx_spec)
    peak_f = f[peak_index]
    peak_value = Pxx_spec[peak_index]
    cut_high = peak_value*0.3
    peaks, properties = signal.find_peaks(x=Pxx_spec, height=cut_high)
    peak_fs = f[peaks]
    peak_values = Pxx_spec[peaks]
    return peak_fs, peak_values, f, Pxx_spec
    
def main():
    music_file = load_data()
    data, rate = sf.read(music_file)
    data = data[:,1]
    data = data[int(rate*21.2):int(rate*21.7)]
    data = butter_bandpass_filter(data, 20e3, 40e3, rate)    
    signal_freq = []
    signal_power = []
    peaks, properties = signal.find_peaks(data, distance=rate*0.05, height=0.009)
    shift = int(rate*10e-3)
    for peak in peaks:
        waveform_i = data[peak-shift:peak+shift]
        peak_f, peak_val, f, Pxx = freq_spec(waveform_i, rate)
        signal_freq.extend(peak_f)
        signal_power.extend(peak_val)
    h = Hist.new.Regular(501, 0, 50000, name='fdata').Weight()
    h.fill(signal_freq, weight=signal_power)
    h.plot1d()
    plt.show()


if __name__ == "__main__":
    main()

