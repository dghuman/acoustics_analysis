import matplotlib.pyplot as plt
from scipy import signal
import numpy as np
import matplotlib as mplt
import soundfile as sf
import os

data_dir = "/home/dilraj/Programs/ocean_test/data/SAANICH/BOAT_COMPATT/"
data_files = {
    95:["COMPATT_ch1-ch2_1_20240502_120804.flac", "COMPATT_ch3-ch4_1_20240502_120616.flac"],
    197:["COMPATT_ch1-ch2_1_20240502_123212.flac", "COMPATT_ch3-ch4_1_20240502_123307.flac"],
    400:["COMPATT_ch1-ch2_1_20240502_124157.flac", "COMPATT_ch3-ch4_1_20240502_124306.flac"],
    550:["COMPATT_ch1-ch2_1_20240502_132007.flac", "COMPATT_ch3-ch4_1_20240502_132105.flac"],
    750:["COMPATT_ch1-ch2_1_20240502_130539.flac", "COMPATT_ch3-ch4_1_20240502_130814.flac"]
    }

peak_cut = {
    1:0.025,
    2:0.01,
    3:0.03,
    4:0.15
    }

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

# 550 distance might have had a larger gain on Channel 1

peaks = {
    1:[],
    2:[],
    3:[],
    4:[]
}

errors = {
    1:[],
    2:[],
    3:[],
    4:[]
}

for distance in data_files.keys():
    for index, _file in enumerate(data_files[distance]):
        waveforms, rate = sf.read(data_dir + _file)
        if index == 0:
            peak_indices, info = signal.find_peaks(waveforms[:,0], distance=rate*0.8, height=[peak_cut[1],0.3])
            avg_peak = np.mean(np.array(waveforms[:,0][peak_indices]))
            error_peak = np.std(np.array(waveforms[:,0][peak_indices]))
            errors[1].append(error_peak)            
            peaks[1].append(avg_peak)

            
            peak_indices, info = signal.find_peaks(waveforms[:,1], distance=rate*0.8, height=peak_cut[2])
            avg_peak = np.mean(np.array(waveforms[:,1][peak_indices]))
            error_peak = np.std(np.array(waveforms[:,1][peak_indices]))
            errors[2].append(error_peak)            
            peaks[2].append(avg_peak)

        else:
            peak_indices, info = signal.find_peaks(waveforms[:,0], distance=rate*0.8, height=peak_cut[3])
            avg_peak = np.mean(np.array(waveforms[:,0][peak_indices]))
            error_peak = np.std(np.array(waveforms[:,0][peak_indices]))
            errors[3].append(error_peak)            
            peaks[3].append(avg_peak)

            peak_indices, info = signal.find_peaks(waveforms[:,1], distance=rate*0.8, height=peak_cut[4])
            avg_peak = np.mean(np.array(waveforms[:,1][peak_indices]))
            error_peak = np.std(np.array(waveforms[:,1][peak_indices]))
            errors[4].append(error_peak)            
            peaks[4].append(avg_peak)

fig, axes = plt.subplots(2, 1, sharex=True)

distances = list(data_files.keys())
channel_list = [1,2,3,4]
for channel in channel_list:
    axes[0].plot(distances, peaks[channel], label=f'Channel {channel}', marker='.', linestyle='-', color=colors[channel-1])
    axes[0].plot([0,750], [peak_cut[channel], peak_cut[channel]], color=colors[channel-1], linestyle=':')
    axes[1].plot(distances, (peaks[channel])/max(peaks[channel]), marker='.', linestyle='-', color=colors[channel-1])
    axes[1].plot([0,750], [peak_cut[channel]/max(peaks[channel]), peak_cut[channel]/max(peaks[channel])], color=colors[channel-1], linestyle=':')


fig.supxlabel('Distance (m)')
#fig.supylabel("Average Max Amplitude (V)")
axes[0].set_ylabel('Average Max Amplitude (V)')
axes[1].set_ylabel('Re-Scaled Average Max Amplitude (V)')
fig.tight_layout() 
#fig.subplots_adjust(right=0.1)
fig.legend(loc="center right")
plt.show()
plt.clf()

