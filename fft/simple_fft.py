import numpy as np
import matplotlib as mplt
import matplotlib.pyplot as plt
from scipy import signal
import soundfile as sf
ipmort arg

data, rate = sf.read('/home/dilraj/Programs/ocean_test/software/analysis/fft/2024-10-02_202551_oossg-2mV-cont/g-0x11/ADC_ch-1-2_2024-10-02_202551_amp-ch1-1000mV-ch2-2mV_g-0x11_f-1000.000Hz_dt-50.000ms_oossg-2mV-cont.flac')

waveform = data[:,1]
win = signal.windows.hann(2**19,sym=False) 
N = len(waveform)
SFT = signal.ShortTimeFFT(win=win, hop=int(len(win)/2), fs=rate, scale_to='magnitude')
Sx_fft = SFT.stft(waveform)
Sx_spec = SFT.spectrogram(waveform)

fft_proj = np.abs(np.mean(Sx_fft, axis=1))


figure, axis = plt.subplots(2, figsize=(6., 4.))
axis[0].plot(SFT.f, 20*np.log10(fft_proj))
axis[0].grid()
axis[0].set_xlabel('Frequency (Hz)')
im1 = axis[1].imshow(20*np.log10(Sx_spec), origin='lower', aspect='auto', extent=SFT.extent(N), cmap='jet')
figure.colorbar(im1, label=r"dB $20\log_{10}|S_{x}(t, f)|$")
#axis.set_title('Spectrogram')
axis[1].set(xlabel='Time (s)', ylabel='Frequency (Hz)')
plt.show()
