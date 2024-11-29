import matplotlib.pyplot as plt
import matplotlib as mplt
import numpy as np
import soundfile as sf
import os
import argparse
import hist
from hist import Hist
from scipy import signal
from peak_frequency import load_data, freq_spec, butter_bandpass_filter, load_dir
from arrival_time import find_arrival_time, compute_stats
from tqdm import tqdm
import h5py as hp


def progressive_diff(arr):
    final_array = []
    for i, el in enumerate(arr):
        new_arr = arr[i:] - el
        final_array.extend(new_arr)
    return np.array(final_array)

def read_gains(file_list):
    gain_strings = []
    for ifile in file_list:
        gain_index = ifile.find('GAIN') + 4
        gain_strings.append(ifile[gain_index:gain_index+3])
    gain_strings = list(set(gain_strings))
    return gain_strings

def main():
    save = False
    debug = False
    super_threshold = 0.0075
    cut_off_ratio = 0.5
    channel_name = 'ch1'
    channel_num = int(channel_name[-1])
    gain_dict = {
        'x11':1,
        'x21':2,
        'x31':5,
        'x41':10,
        'x51':20,
        'x61':50
    }
    music_dir = load_dir()
    loc_number = int(music_dir[-1])
    music_files = os.listdir(music_dir)
    new_files = [x for x in music_files if channel_name in x]
    gain_strings = read_gains(new_files)
    for gain_set in gain_strings:
        threshold = super_threshold*gain_dict[gain_set]
        arrival_times = []
        for music_file in [x for x in new_files if gain_set in x]:
            data, rate = sf.read(music_dir + '/' + music_file)
            data = data[:,(channel_num-1)%2]
            t = np.linspace(0, len(data)/rate, len(data))
            data = butter_bandpass_filter(data, 20e3, 40e3, rate)
            #        max_height = max(data)
            #        print(f"max height is {max_height}")
            peaks, properties = signal.find_peaks(data, distance=int(rate*0.05), height=threshold)
            if debug:
                plt.plot(t, data)
                plt.plot(t[peaks], data[peaks], linestyle='', marker='x')
                plt.plot(t, np.ones(len(t))*threshold, linestyle='-', color='k', alpha=0.8)
                plt.show()
            for peak in peaks:
                t_a = find_arrival_time(data, rate, peak, shift=int(rate*10e-3), height=0.004) #, height=0.0009
                arrival_times.append(t_a)
        arrival_times = np.array(arrival_times)
        full_diff = progressive_diff(arrival_times)
        h = Hist(hist.axis.Regular(bins=301, start=0, stop=3, name='diff in arrival times')) 
        h.fill(full_diff)
        y, x = h.to_numpy()
        offset = int(0.5/0.01)
        y_cut = y[offset:]
        max_val = max(y_cut)
        peaks, properties = signal.find_peaks(y_cut, distance=4, height=cut_off_ratio*max_val)
        means = np.zeros(len(peaks))
        stds = np.zeros(len(peaks))    
        for i in range(len(peaks)):
            means[i], stds[i] = compute_stats(full_diff, [0.5 + (peaks[i] - 2)*0.01, 0.5 + (peaks[i] + 2)*0.01])
        print(f"Means are {means}")
        print(f"with STDs {stds}")
        print(f"with counts {y_cut[peaks]}")
        if not save:
            plt.clf()
            plt.hist(full_diff, bins = x)
            plt.plot(x[peaks + offset], y[peaks + offset], linestyle='', marker='x')
            plt.title(r'$\Delta$t Hist')
            plt.show()
        else:
            f = hp.File('/home/dilraj/Programs/ocean_test/data/rev4/ANALYSIS/SAANICH/DeltaT_Data.hdf5', 'a')
            grp = f.create_group(f"LOC{loc_number}/CH{channel_num}/{gain_set}")
            grp.create_dataset("DeltaT", data=full_diff)
            grp.create_dataset("Means", data=means)
            grp.create_dataset("STDs", data=stds)
            grp.create_dataset("Counts", data=y_cut[peaks])
            f.close()

if __name__ == "__main__":
    main()
