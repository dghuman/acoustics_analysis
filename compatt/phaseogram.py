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

# start with one file and plot pulse time modulo 1 second. Only pulses that have a 1 second period will stack up

test_data = "/home/dilraj/Programs/ocean_test/data/SAANICH/BOAT_COMPATT/COMPATT_ch3-ch4_1_20240502_123307.flac"
data, rate = sf.read(test_data)
waveform = data[:,1]

