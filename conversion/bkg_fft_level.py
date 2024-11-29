import matplotlib.pyplot as plt
import matplotlib as mplt
import numpy as np
from scipy.signal import periodogram, find_peaks
import soundfile as sf
import os
import pickle
import argparse
import scipy.io.wavfile as wav

#plt.style.use('/home/dilraj/.config/matplotlib/matplotlib.rc')

def load_data():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--infile')
    args = parser.parse_args()
    music_file = args.infile
    return music_file

def main():
    data_file = load_data()
    data, rate = sf.read(data_file)
    data1 = data[:,0]
    data2 = data[:,1]
    wav.write("data1.wav", rate, data1)
    wav.write("data2.wav", rate, data2)    
    

if __name__ == '__main__':
    main()
