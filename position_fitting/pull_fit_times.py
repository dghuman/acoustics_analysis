import numpy as np
import h5py as hp
import os, sys
import matplotlib.pyplot as plt
import matplotlib as mplt
import matplotlib.patches as mpatches
from itertools import permutations

h = 8

def d_difference(L):
    H = 145
    num = h*H
    denom = np.sqrt(L**2 + H**2)
    return num/denom

def time_fit(t1, t2):
    p_permutations = list(permutations([0,1,2],3))
    chi2 = 1e6
    p_perm = None
    for perm in p_permutations:
        diff = (t1[list(perm)] - t2)**2
        sum_i = np.sum(diff)
        if chi2 > sum_i:
            chi2 = sum_i
            p_perm = perm
    return p_perm

prop_cycle = plt.rcParams["axes.prop_cycle"]
colors = prop_cycle.by_key()["color"] 

delta_T_file = '/home/dilraj/Programs/ocean_test/data/rev4/ANALYSIS/SAANICH/DeltaT_Data.hdf5'
sonardyne_T_file = '/home/dilraj/Programs/ocean_test/data/rev4/ANALYSIS/SAANICH/sonardyne-dilraj-full.npy'

sonardyne_T = np.load(sonardyne_T_file, allow_pickle=True).item()

c0 = 1500


with hp.File(delta_T_file, 'r') as delta_T_data:
    Data = delta_T_data
    plt.xticks(range(6), list(Data.keys()))    
    for k, loc in enumerate(list(Data.keys())):
        #print(f'LOCATION: {loc}')
        for channel in list(Data[loc].keys()):
#            print(f'Channel: {channel}')
            for gain in list(Data[loc][channel].keys()):
 #               print(f'Gain: {gain}')
                counts = np.array(Data[loc][channel][gain]['Counts'][:])
                times = np.array(Data[loc][channel][gain]['Means'][:])
                rec_std = np.array(Data[loc][channel][gain]['STDs'][:])
                relevant_indices = np.argpartition(counts, -3)[-3:]
                corrected_times  = times[relevant_indices] + h/c0 + d_difference(200)/c0
                time1 = np.zeros(3)
                beacon_std = np.zeros(3)
                for i, key in enumerate(list(sonardyne_T.keys())):
                    time1[i] = (sonardyne_T[key][loc]['tof_mean'])*1e-6 # convert to seconds
                    beacon_std[i] = sonardyne_T[key][loc]['tof_std']
                new_indices = list(time_fit(time1, corrected_times))
                beacon_std = beacon_std[new_indices]
                new_std = np.sqrt(beacon_std**2 + rec_std[relevant_indices]**2)
                diff = corrected_times - time1[new_indices]
                for j, d in enumerate(diff):
                    plt.plot(k, d, color=colors[(new_indices[j])], linestyle='', marker='.')
                    #print(np.abs(corrected_times - time1[new_indices])/new_std)
    beacons = list(sonardyne_T.keys())
    handles = []
    for m, beacon in enumerate(beacons):
        handles.append(mpatches.Patch(color=colors[m], label=beacon))
    plt.legend(handles=handles)
    plt.show()
        
            
