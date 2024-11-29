import matplotlib.pyplot as plt
import matplotlib as mplt
import numpy as np
import soundfile as sf
import os
import argparse
import hist
from hist import Hist
from scipy import signal, optimize
from peak_frequency import load_data, freq_spec, butter_bandpass_filter

def sinusoid(x, a, b, c):
    return a*np.sin(b*x + c)    

def peak_fit(peak_f, Pxx, f):
    def func(x, a, b, c):
        return a*x*x + b*x + c

def main():
    music_file = load_data()
    data, rate = sf.read(music_file)
    data = data[:,1]
    data = butter_bandpass_filter(data, 20e3, 40e3, rate)
    t = np.linspace(0, len(data)/rate, len(data))
    cut_low = int(rate*21.225)
    cut_high = int(rate*21.2255)
    data = data[cut_low:cut_high]
    t = t[cut_low:cut_high]
    #popt, pcov = optimize.curve_fit(sinusoid, t, data)
    #plt.plot(t, sinusoid(t, *popt), label='fit')
    plt.plot(t, data)
    plt.show()
    plt.clf()    
#    peaks, properties = signal.find_peaks(data, distance=rate*0.05, height=0.01) 
#    peak_index = np.argmax(data[peaks])
#    peak = peaks[peak_index]
#    shift = int(rate*5e-3)
#    waveform_i = data[peak-shift:peak+shift]
    peak_f, peak_val, f, Pxx = freq_spec(data, rate)
    plt.plot(f, Pxx, linestyle='-', color='k', label='FFT')
    plt.plot(peak_f,peak_val, linestyle='', marker='x', color='red', label='Freq Peak Found')
    plt.show()
    plt.clf()
    new_t = np.linspace(t[0], t[-1], len(t)*10)
    plt.plot(t, data, linestyle='--', label='data')
    plt.plot(new_t, np.sqrt(peak_val)*np.sin((new_t-7e-6)*peak_f*2*np.pi), label='frequency fit')
    plt.show()

if __name__ == "__main__":
    main()
    
