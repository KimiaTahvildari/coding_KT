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
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, signal)

# -------------------------------
# Dataset path
# -------------------------------
dataset_path = 'afdb/'  # folder containing AFDB .dat files
records = [f.split('.')[0] for f in os.listdir(dataset_path) if f.endswith('.dat')]

# -------------------------------
# Apply filters to each record
# -------------------------------
for record_name in records:
    print(f'Processing record: {record_name}')
    record = wfdb.rdrecord(record_name, pn_dir='afdb')
    ecg_signal = record.p_signal[:, 0]  # first channel
    fs = record.fs
    
    # Apply filters
    ecg_lpf = low_pass_filter(ecg_signal, fs, cutoff=40)
    ecg_hpf = high_pass_filter(ecg_signal, fs, cutoff=0.5)
    ecg_bpf = band_pass_filter(ecg_signal, fs, lowcut=0.5, highcut=40)
    
    # Plot for verification
    plt.figure(figsize=(15, 5))
    plt.plot(ecg_signal, label='Original ECG', alpha=0.5)
    plt.plot(ecg_lpf, label='Low-Pass Filtered', color='orange')
    plt.plot(ecg_hpf, label='High-Pass Filtered', color='green')
    plt.plot(ecg_bpf, label='Band-Pass Filtered', color='red')
    plt.title(f'Record {record_name} - ECG Filtering')
    plt.xlabel('Sample')
    plt.ylabel('Amplitude (mV)')
    plt.legend()
    plt.show()
