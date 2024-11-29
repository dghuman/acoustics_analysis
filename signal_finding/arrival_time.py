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

# Tag peaks and estimate arrival time of corresponding waveform

def find_arrival_time(data, rate, peak, shift, height=0.005, plot=False):
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
    return arrival_time

def compute_stats(arrival_time, cuts):
    indices = np.where((arrival_time > cuts[0]) & (arrival_time < cuts[1]))
    std = np.std(arrival_time[indices])
    mean = np.mean(arrival_time[indices])
    return mean, std

def main():
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    music_file = load_data()
    data, rate = sf.read(music_file)
    data = data[:,1]
#    data = data[int(rate*21.2):int(rate*21.7)]
    t = np.linspace(0, len(data)/rate, len(data))
    data = butter_bandpass_filter(data, 20e3, 40e3, rate)
    arrival_time = []
    peaks, properties = signal.find_peaks(data, distance=rate*0.05, height=0.02)#0.0075
    plt.plot(t, data)
    plt.plot(t[peaks], data[peaks], linestyle='', marker='x')
    plt.show()
    plt.clf()
    t_cut = 2.5
    t_base = t[peaks][0]
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
    for peak in peaks:
        t_a = find_arrival_time(data, rate, peak, shift=int(rate*10e-3))
        waveform_i = data[peak-int(rate*10e-3):peak+int(rate*10e-3)]
        peak_fs, peak_values, f, Pxx = freq_spec(waveform_i, rate)
        if t_a - t_base > t_cut:
            t_base = t_a
        t_a = t_a - t_base
        arrival_time.append(t_a)
        if t_a > 0.49 and t_a < 0.52:
            hists[1].append(t_a)
            freqs[1].extend(peak_fs)
        if t_a > 0.85 and t_a < 0.87:
            hists[2].append(t_a)
            freqs[2].extend(peak_fs)            
        if t_a > 0.9 and t_a < 0.93: 
           hists[3].append(t_a)
           freqs[3].extend(peak_fs)           
    arrival_time = np.array(arrival_time)
    h = Hist(hist.axis.Regular(bins=201, start=0, stop=2, name='arrival times')) #Hist.new.Regular(2001, 0, 2, name='fdata').Double()
    hf = Hist(hist.axis.Regular(bins=61, start=20e3, stop=50e3, name='freqs')) #Hist.new.Regular(2001, 0, 2, name='fdata').Double()
#    h.fill(arrival_time)
    y, x = h.to_numpy()
    yf, xf = hf.to_numpy()
    fig, axes = plt.subplots(2)
    axes[0].hist(arrival_time, bins=x, color='k', alpha=0.9, label="all times")
    axes[0].hist(hists[1], bins=x, color=colors[0], alpha=0.9, label="B1")
    axes[0].hist(hists[2], bins=x, color=colors[1], alpha=0.9, label="B2")
    axes[0].hist(hists[3], bins=x, color=colors[2], alpha=0.9, label="B3")
    axes[1].hist(freqs[1], bins=xf, color=colors[0], alpha=0.9, label="B1")
    axes[1].hist(freqs[2], bins=xf, color=colors[1], alpha=0.9, label="B2")
    axes[1].hist(freqs[3], bins=xf, color=colors[2], alpha=0.9, label="B3")
    plt.legend()
    #plt.title('Traveling Window')
    #h.plot1d()
    plt.show()
#    mean_1, std_1 = compute_stats(arrival_time, [0.5,0.55])
#    mean_2, std_2 = compute_stats(arrival_time, [0.7,0.73])
#    mean_3, std_3 = compute_stats(arrival_time, [0.9,0.93])
#    speed_of_sound = 1500 # m/s
#    print(f'mean: {mean_1}, std: {std_1}, corresponding distance: {(mean_1 - 0.240)*speed_of_sound/2} +/- {std_1*speed_of_sound}')
#    print(f'mean: {mean_2}, std: {std_2}, corresponding distance: {(mean_2 - 0.440)*speed_of_sound/2} +/- {std_2*speed_of_sound}')
#    print(f'mean: {mean_3}, std: {std_3}, corresponding distance: {(mean_3 - 0.640)*speed_of_sound/2} +/- {std_3*speed_of_sound}')

if __name__ == "__main__":
    main()

