import wfdb
import numpy as np
import pywt
from scipy.signal import butter, filtfilt, cwt, ricker, welch, stft, spectrogram
from scipy.stats import entropy
import matplotlib.pyplot as plt

# -------------------------------
# 1️⃣ Load ECG
# -------------------------------
def load_ecg(record_name, lead=0, pn_dir='afdb'):
    record = wfdb.rdrecord(record_name, pn_dir=pn_dir)
    signal = record.p_signal[:, lead]
    Fs = record.fs
    return signal, Fs

# -------------------------------
# 2️⃣ Band-pass filter
# -------------------------------
def bandpass_filter(sig, lowcut=0.5, highcut=40, Fs=250, order=4):
    nyq = 0.5 * Fs
    b, a = butter(order, [lowcut/nyq, highcut/nyq], btype='band')
    return filtfilt(b, a, sig)

# -------------------------------
# 3️⃣ FFT
# -------------------------------
def compute_fft(signal, Fs):
    N = len(signal)
    fft_values = np.fft.fft(signal)
    freqs = np.fft.fftfreq(N, 1/Fs)
    pos_freqs = freqs[:N//2]
    pos_fft = np.abs(fft_values[:N//2]) * 2.0 / N
    return pos_freqs, pos_fft

# -------------------------------
# 4️⃣ PSD
# -------------------------------
def compute_psd(signal, Fs, nperseg=1024):
    f_psd, Pxx = welch(signal, fs=Fs, nperseg=nperseg)
    return f_psd, Pxx

# -------------------------------
# 5️⃣ STFT
# -------------------------------
def compute_stft(signal, Fs, nperseg=256, noverlap=128):
    f_stft, t_stft, Zxx = stft(signal, fs=Fs, nperseg=nperseg, noverlap=noverlap)
    return f_stft, t_stft, np.abs(Zxx)

# -------------------------------
# 6️⃣ Spectrogram
# -------------------------------
def compute_spectrogram(signal, Fs, nperseg=256, noverlap=128):
    f_spec, t_spec, Sxx = spectrogram(signal, fs=Fs, nperseg=nperseg, noverlap=noverlap)
    return f_spec, t_spec, Sxx

# -------------------------------
# 7️⃣ Spectral Entropy
# -------------------------------
def spectral_entropy(Pxx):
    psd_norm = Pxx / np.sum(Pxx)
    return entropy(psd_norm)

# -------------------------------
# 8️⃣ Relative Spectral Power
# -------------------------------
def relative_spectral_power(f_psd, Pxx, bands={'VLF':(0.5,4),'LF':(4,15),'HF':(15,40)}):
    total_power = np.sum(Pxx)
    RSP = {}
    for band, (low, high) in bands.items():
        idx = np.logical_and(f_psd >= low, f_psd <= high)
        RSP[band] = np.sum(Pxx[idx]) / total_power
    return RSP

# -------------------------------
# 9️⃣ DWT
# -------------------------------
def compute_dwt(signal, wavelet_name='db4', level=4):
    coeffs = pywt.wavedec(signal, wavelet_name, level=level)
    energy = [np.sum(np.square(c)) for c in coeffs]
    return coeffs, energy

# -------------------------------
# 10️⃣ CWT
# -------------------------------
def compute_cwt(signal, scales=np.arange(1,128)):
    return cwt(signal, ricker, scales)

# -------------------------------
# Example usage
# -------------------------------
record_name = '04015'
signal, Fs = load_ecg(record_name)
signal_filtered = bandpass_filter(signal, Fs=Fs)

# FFT
freqs, fft_mag = compute_fft(signal_filtered, Fs)

# PSD
f_psd, Pxx = compute_psd(signal_filtered, Fs)

# STFT
f_stft, t_stft, STFT_mag = compute_stft(signal_filtered, Fs)

# Spectrogram
f_spec, t_spec, Sxx = compute_spectrogram(signal_filtered, Fs)

# Spectral entropy
SpEn = spectral_entropy(Pxx)

# Relative spectral power
RSP = relative_spectral_power(f_psd, Pxx)

# DWT
coeffs, wavelet_energy = compute_dwt(signal_filtered)

# CWT
cwt_matrix = compute_cwt(signal_filtered)

# Optional: plot CWT scalogram
plt.figure(figsize=(12,4))
plt.imshow(cwt_matrix, extent=[0,len(signal_filtered)/Fs, 128, 1], cmap='jet', aspect='auto')
plt.xlabel('Time [s]')
plt.ylabel('Scale')
plt.title(f'CWT Scalogram (Ricker) for record {record_name}')
plt.colorbar(label='Amplitude')
plt.show()
