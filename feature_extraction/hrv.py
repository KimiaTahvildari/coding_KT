#(MF) for median filter
#(BWF) for Butterworth filter 
#PT for Pan-Tompkins algorithm
#(L-ROS) for combination of local outlier factor 
#(FrST ) for Fractional S-Transform
#(IHR) for instantaneous heart rate


import wfdb
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import medfilt, butter, filtfilt, find_peaks
from sklearn.neighbors import LocalOutlierFactor
from scipy.fftpack import fft
import os

# -------------------------------
# Filtering functions
# -------------------------------
def median_filter(signal, kernel_size=5):
    return medfilt(signal, kernel_size=kernel_size)

def butterworth_filter(signal, fs, lowcut=0.5, highcut=40, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, signal)

# -------------------------------
# Pan-Tompkins QRS detection (simplified)
# -------------------------------
def pan_tompkins(signal, fs):
    # Derivative + Squaring + Moving window integration (simplified)
    diff_signal = np.diff(signal)
    squared_signal = diff_signal**2
    window_size = int(0.150 * fs)  # 150 ms window
    integrated_signal = np.convolve(squared_signal, np.ones(window_size)/window_size, mode='same')
    
    # Detect peaks
    threshold = 0.5 * np.max(integrated_signal)
    peaks, _ = find_peaks(integrated_signal, height=threshold, distance=0.2*fs)
    return peaks

# -------------------------------
# Local Outlier Score (LOF)
# -------------------------------
def local_outlier_scores(signal, n_neighbors=20):
    lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=0.01)
    reshaped_signal = signal.reshape(-1,1)
    scores = -lof.fit_predict(reshaped_signal)  # negative for outlier scoring
    return scores

# -------------------------------
# Fractional S-Transform (simplified version)
# -------------------------------
def fractional_s_transform(signal, fs):
    N = len(signal)
    t = np.arange(N) / fs
    S = fft(signal)
    freqs = np.fft.fftfreq(N, 1/fs)
    return freqs, S

# -------------------------------
# Instantaneous heart rate (IHR)
# -------------------------------
def instantaneous_hr(peaks, fs):
    rr_intervals = np.diff(peaks) / fs  # in seconds
    hr = 60 / rr_intervals  # bpm
    return hr, rr_intervals

# -------------------------------
# Main pipeline
# -------------------------------
dataset_path = 'afdb/'  # folder containing AFDB records
records = [f.split('.')[0] for f in os.listdir(dataset_path) if f.endswith('.dat')]

for record_name in records:
    print(f'Processing record: {record_name}')
    record = wfdb.rdrecord(record_name, pn_dir='afdb')
    ecg_signal = record.p_signal[:, 0]  # first channel
    fs = record.fs
    
    # Median filter
    ecg_mf = median_filter(ecg_signal, kernel_size=5)
    
    # Butterworth filter
    ecg_bwf = butterworth_filter(ecg_mf, fs, lowcut=0.5, highcut=40, order=4)
    
    # Pan-Tompkins
    qrs_peaks = pan_tompkins(ecg_bwf, fs)
    
    # Local Outlier Scores
    lof_scores = local_outlier_scores(ecg_bwf)
    
    # Fractional S-Transform
    freqs, st_signal = fractional_s_transform(ecg_bwf, fs)
    
    # Instantaneous HR
    hr, rr_intervals = instantaneous_hr(qrs_peaks, fs)
    
    # -------------------------------
    # Plot example for verification
    # -------------------------------
    plt.figure(figsize=(15,5))
    plt.plot(ecg_signal, label='Original ECG', alpha=0.5)
    plt.plot(ecg_bwf, label='Filtered ECG', color='red')
    plt.scatter(qrs_peaks, ecg_bwf[qrs_peaks], color='green', label='QRS Peaks')
    plt.title(f'Record {record_name} - ECG Processing Pipeline')
    plt.xlabel('Sample')
    plt.ylabel('Amplitude (mV)')
    plt.legend()
    plt.show()
    
    print(f"Mean HR for record {record_name}: {np.mean(hr):.2f} bpm")

