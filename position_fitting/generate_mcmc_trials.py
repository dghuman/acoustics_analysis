import os
import copy
import numpy as np
from scipy import signal, stats, constants
from scipy.optimize import minimize, curve_fit
import h5py as hp
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

import arviz as az
import pymc3 as pm
import theano.tensor as tt


def set_fixed_variables():
    global c0
    global d_c0
    global d_spatial
    global d_adc
    global d_sync
    global d_shannon
    global d_t
    global d_beacon
    
    # speed of sound
    c0  = 1492 # m/s
    d_c0 = 5 # m/s

    # acoustic resolutions
    # taken from: https://www.institut3b.physik.rwth-aachen.de/global/show_document.asp?id=aaaaaaaaatmdetx
    # especially Table 4.1, Figure 4.5
    d_spatial = 13e-6 # us
    d_adc = 1.5e-6 # us
    d_sync = 5e-6 # us
    d_shannon = 100e-6 # us
    # this is an estimate based on longest attenuation length in ice = 300m
    # in water: http://resource.npl.co.uk/acoustics/techguides/seaabsorption/ (F=15, T=3, D=2.5, S=35, pH=8)
    # gives sound absorption, a, within [1.392, 1.706] db/km
    # in 1/km this becomes a' = a * (ln(10)/10)
    # for a = 1.5 db/km --> attenuation length = 1 / a' = 1 / (a * (ln(10)/10)) = 2.8km ~ 10 * attenuation length ice
    # so we use d_shannon ~ d_shannon_icecube / 10
    d_t = np.sqrt(d_spatial**2 + d_adc**2 + d_sync**2 + d_shannon**2)

    # beacon placement accuracy
    d_beacon = 10. # m

def gaussian(x, a, mu, sig):
    return a * np.exp(-(x-mu)**2/(2*sig**2))

def fit_gaussian(x, bins=100, r=(), sigma=0.1, tol=0.):
    n, e = np.histogram(x, range=r, bins=bins)
    c    = (e[:-1] + e[1:]) / 2.
    a0   = np.max(n)
    mu0  = c[(n == np.max(n))][0]
    sig0 = sigma
    m = (n >= tol*n)
    popt, pcov = curve_fit(gaussian, c[m], n[m], p0=(a0, mu0, sig0), maxfev=10000)
    return popt

def load_data():
    # Load in Boat Data
    data_file = '/home/dilraj/Programs/ocean_test/data/rev4/ANALYSIS/SAANICH/tof-loc1.pkl'
    boat_data = pd.read_pickle(data_file)

    sample_boat_data = boat_data.loc[2:4, ['id', 'tof', 'utc_time']]

    # Load in Sonardyne Data
    sonardyne_file = '/home/dilraj/Programs/ocean_test/data/rev4/ANALYSIS/SAANICH/df-sonardyne-log.pkl'
    sonardyne_data = pd.read_pickle(sonardyne_file)

    #print(sonardyne_data.keys())

    sonardyne_LOC1 = sonardyne_data.loc[lambda df: df['Location'] == 'LOC1',:] #['R3008', 'R2404', 'R2407']
    sonardyne_LOC1 = sonardyne_LOC1.loc[lambda df: df['R3008'] == df['R3008'], :]

    #sonardyne_LOC1_tof_means = sonardyne_LOC1.mean()

    sample_sonardyne_data = sonardyne_LOC1.loc[84]

    return sample_boat_data, sample_sonardyne_data


if __name__ == '__main__':
    set_fixed_variables()
    test_locations = np.load("rotated_test_locations.npy", allow_pickle=True)
    beacon_locations = np.load("rotated_beacons.npy", allow_pickle=True)
    beacon_locations[:,[1,2]] = beacon_locations[:,[2,1]]
    beacon_labels = ["3008", "2404", "2407"]
    TAT = np.array([0.24, 0.44, 0.64])
    sample_boat_data, sample_sonardyne_data = load_data()
    measured_times = sample_boat_data['tof'].to_numpy().astype(float)
    sonardyne_TOF = (sample_sonardyne_data[['R3008', 'R2404', 'R2407']].to_numpy().astype(float))*1e-6
    sonardyne_one_way = (sonardyne_TOF - TAT)/2
    
    
