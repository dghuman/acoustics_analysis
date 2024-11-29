import matplotlib.pyplot as plt
import matplotlib as mplt
import numpy as np
import soundfile as sf
import os
import argparse
import hist
from hist import Hist
from scipy import signal
from peak_frequency import load_data, freq_spec, butter_bandpass_filter
from arrival_time import find_arrival_time
from tqdm import tqdm


def progressive_diff(arr):
    final_array = []
    for i, el in enumerate(arr):
        new_arr = arr[i:] - el
        final_array.extend(new_arr)
    return np.array(final_array)

def main():
    music_file = load_data()
    data, rate = sf.read(music_file)
    threshold = 0.008
    data = data[:,1]
    t = np.linspace(0, len(data)/rate, len(data))
    data = butter_bandpass_filter(data, 20e3, 40e3, rate)
    peaks, properties = signal.find_peaks(data, distance=int(rate*0.05), height=threshold) 
    print(f'Found {len(peaks)} peaks.')
    plt.plot(t, data)
    plt.plot(t[peaks], data[peaks], linestyle='', marker='x')
    plt.plot(t, np.ones(len(t))*threshold, linestyle='-', color='k', alpha=0.8)
    plt.show()
    arrival_times = []
    for peak in peaks:
        t_a = find_arrival_time(data, rate, peak, shift=int(rate*10e-3)) #, height=0.0009
        arrival_times.append(t_a)
    arrival_times = np.array(arrival_times)
    full_diff = progressive_diff(arrival_times)
    h = Hist(hist.axis.Regular(bins=201, start=0, stop=2, name='diff in arrival times')) 
    h.fill(full_diff)
    y, x = h.to_numpy()
    plt.clf()
    plt.hist(full_diff, bins = x)
    plt.title(r'$\Delta$t Hist')
    plt.show()

if __name__ == "__main__":
    main()
