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
from temporal_diff import progressive_diff

# Tag peaks and estimate arrival time of corresponding waveform

def pulse_info(data, rate, peak, shift, height=0.005, plot=False):
    if peak - shift < 0:
        shift_L = 0
    else:
        shift_L = peak - shift
    if peak + shift > len(data) - 1:
        shift_R = -1
    else:
        shift_R = peak + shift
    waveform_i = data[shift_L:shift_R]
    t_i = np.linspace((shift_L)/rate, (shift_R)/rate, shift*2)        
    peak_f, peak_val, f, Pxx = freq_spec(waveform_i, rate)
    waveform_freq = peak_f[np.argmax(peak_val)]
    waveform_period = 1/waveform_freq
    peaks_i, props_i = signal.find_peaks(waveform_i, height=height)
    arrival_time = t_i[peaks_i[0]] - waveform_period*1
    if plot:        
        plt.plot(t_i, waveform_i)    
        plt.plot(t_i[peaks_i[0]], waveform_i[peaks_i[0]], linestyle='', marker='x', label='threshold peak')
        plt.axvline(t_i[peaks_i[0]] - waveform_period*2, label='estimated arrival time', color='k')
        plt.legend()
        plt.show()
        plt.clf()
    return arrival_time, peak_f, peak_val, f, Pxx

def compute_stats(arrival_time, cuts):
    indices = np.where((arrival_time > cuts[0]) & (arrival_time < cuts[1]))
    std = np.std(arrival_time[indices])
    mean = np.mean(arrival_time[indices])
    return mean, std

def main():
    debug = False
    height = 0.0075
    height_2 = 0.003
    channel_name = 'ch1'
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    music_dir = load_dir()
    music_files = os.listdir(music_dir)
    new_files = [x for x in music_files if channel_name in x]
    t_cut = 2.5
    arrival_times = []
    hists = {
        1:[],
        2:[],
        3:[]
    }
    freqs = {
        1:[],
        2:[],
        3:[]
    }
    for music_file in new_files:
        data, rate = sf.read(music_dir + '/' + music_file)
        data = data[:,1]
        t = np.linspace(0, len(data)/rate, len(data))
        data = butter_bandpass_filter(data, 20e3, 40e3, rate)
        peaks, properties = signal.find_peaks(data, distance=rate*0.05, height=height)#0.0075
        if debug == True:
            plt.plot(t, data, color='k')
            plt.plot(t[peaks], data[peaks], linestyle='', marker='x')
            plt.plot(t, np.ones(len(t))*height, color='k', alpha=0.5)
            plt.show()
            plt.clf()
        t_base = t[peaks][0]
        for peak in peaks:
            t_a, peak_fs, peak_val, f, Pxx = pulse_info(data, rate, peak, shift=int(rate*4e-3), height=height_2)
            if t_a - t_base > t_cut:
                t_base = t_a
            t_a = t_a - t_base
            arrival_times.append(t_a)
            if t_a > 0.5 and t_a < 0.55:
                hists[1].append(t_a)
                freqs[1].extend(peak_fs)
            if t_a > 0.7 and t_a < 0.73:
                hists[2].append(t_a)
                freqs[2].extend(peak_fs)            
            if t_a > 0.9 and t_a < 0.93: 
                hists[3].append(t_a)
                freqs[3].extend(peak_fs)           
    arrival_times = np.array(arrival_times)
    h = Hist(hist.axis.Regular(bins=201, start=0, stop=2, name='arrival times')) #Hist.new.Regular(2001, 0, 2, name='fdata').Double()
    hf = Hist(hist.axis.Regular(bins=61, start=20e3, stop=50e3, name='freqs')) #Hist.new.Regular(2001, 0, 2, name='fdata').Double()
    #    h.fill(arrival_time)
    y, x = h.to_numpy()
    yf, xf = hf.to_numpy()
    fig, axes = plt.subplots(2)
    axes[0].hist(arrival_times, bins=x, color='k', alpha=0.9, label="all times")
    axes[0].hist(hists[1], bins=x, color=colors[0], alpha=0.9, label="B1")
    axes[0].hist(hists[2], bins=x, color=colors[1], alpha=0.9, label="B2")
    axes[0].hist(hists[3], bins=x, color=colors[2], alpha=0.9, label="B3")
    axes[1].hist(freqs[1], bins=xf, color=colors[0], alpha=0.9, label="B1")
    axes[1].hist(freqs[2], bins=xf, color=colors[1], alpha=0.9, label="B2")
    axes[1].hist(freqs[3], bins=xf, color=colors[2], alpha=0.9, label="B3")
    plt.legend()
    plt.suptitle(channel_name)
    plt.show()

if __name__ == "__main__":
    main()

