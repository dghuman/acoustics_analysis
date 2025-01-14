import numpy as np
import matplotlib as mplt
import matplotlib.pyplot as plt
import h5py as hp

import os, sys


def main():
    indir = '/home/dilraj/Programs/ocean_test/data/rev4/ANALYSIS/SAANICH/posterior/'
    infile = indir + 'sample_data.hdf5'
    hfile = hp.File(infile, 'r')
    samplers = list(hfile.keys())
    for sampler in samplers:
        print("Sampler " + sampler)
        elements = list(hfile[sampler]['labels'])
        elements = [x.decode('utf-8') for x in elements]
        x_index = elements.index('xr[0, 0]')
        y_index = elements.index('xr[1, 0]')
        z_index = elements.index('xr[2, 0]')
        h_index = elements.index('h[0]')
        means = hfile[sampler]['means']
        x_value = means[x_index]
        y_value = means[y_index]
        z_value = means[z_index]
        h = means[h_index]
        r = np.array([x_value, y_value, z_value])
        print(f'Receiver pos = {r}')
        print(f'Height fit = {h}')
    hfile.close()


if __name__ == '__main__':
    main()
