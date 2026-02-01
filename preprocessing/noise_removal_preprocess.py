#(BPF) for Band-pass filter
#(HPF) for High pass filter
#(LPF)for Low-Pass Filter 


import wfdb
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
import os

# -------------------------------
# Filtering functions
# -------------------------------
def low_pass_filter(signal, fs, cutoff=40, order=4):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low')
    return filtfilt(b, a, signal)

def high_pass_filter(signal, fs, cutoff=0.5, order=4):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high')
    return filtfilt(b, a, signal)



def band_pass_filter(signal, fs, lowcut=0.5, highcut=40, order=4):
    nyq = 0.5 * fs
    b, a = butter(order, [lowcut/nyq, highcut/nyq], btype="band")
    return filtfilt(b, a, signal)

def apply_filter_multilead(ecg, fs):
    filtered = np.zeros_like(ecg)
    for lead in range(ecg.shape[0]):
        filtered[lead] = band_pass_filter(ecg[lead], fs)
    return filtered
