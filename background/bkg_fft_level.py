import matplotlib.pyplot as plt
import matplotlib as mplt
import numpy as np
from scipy.signal import periodogram, find_peaks
import soundfile as sf
import os
import pickle

#plt.style.use('/home/dilraj/.config/matplotlib/matplotlib.rc')



def set_data(indir):
    true_infiles = os.listdir(indir)
    infiles = [x for x in true_infiles if '0x11' in x or '0x61' in x or '0x71' in x]
    data = {}
    for ifile in infiles:
        gain = ifile[24:28]
        data[gain], rate = sf.read(indir + '/' + ifile)
    return data, rate

def fft(waveform, rate):
    f, Pxx_spec = periodogram(waveform, rate, 'flattop', scaling='spectrum')
    peaks, _ = find_peaks(x=Pxx_spec, distance=800)
    return f, Pxx_spec, peaks

def main():
    data, rate = set_data('/home/dilraj/Programs/ocean_test/data/rev4/SFU/POST_TEST')
    gain_ref = {
        '0x11':1,
        '0x21':2,
        '0x31':5,
        '0x41':10,
        '0x51':20,
        '0x61':50,
        '0x71':100
    }
    fig = plt.figure()
    for key in list(data.keys()):
        f, Pxx_spec, peaks = fft(data[key][:int(rate*1),0], rate)
        plt.semilogy(f, np.sqrt(2)*np.sqrt(Pxx_spec), label=f'Gain of {gain_ref[key]}', alpha=0.5)
    
    plt.xlabel("Frequency (Hz)")
    plt.ylabel(r"Average Spectrum [Vpp]")
    plt.legend()
    plt.show()
    plt.clf()

    #with open('peak_info.pkl', 'wb') as f:
    #    pickle.dump(peak_info, f)

if __name__ == '__main__':
    main()
