################################## (FFT) for fast fourier transform ############################### 
# the record, signal and FS should have been introduced before
from scipy.fft import fft, fftfreq

N = len(signal)
fft_values = fft(signal)
freqs = fftfreq(N, 1/Fs)

# Optional: single-sided magnitude spectrum
pos_freqs = freqs[:N//2]
pos_fft = np.abs(fft_values[:N//2]) * 2.0 / N
                                
################################## (PSD) Power spectral density ####################################
from scipy.signal import welch
import matplotlib.pyplot as plt

f_psd, Pxx = welch(signal, fs=Fs, nperseg=1024)
################################## Shortterm Fourier transform (STFT) #############################
from scipy.signal import stft
import numpy as np

# Example ECG signal and sampling frequency
# signal = your_ecg_signal 
# Fs = 250

# Compute STFT
f_stft, t_stft, Zxx = stft(signal, fs=Fs, nperseg=256, noverlap=128)

# Zxx contains complex STFT values
# f_stft = frequency bins
# t_stft = time bins

# If you want magnitude:
STFT_magnitude = np.abs(Zxx)
#################################### spectrogram ###########################################
from scipy.signal import spectrogram
import matplotlib.pyplot as plt

# Compute Spectrogram
f_spec, t_spec, Sxx = spectrogram(signal, fs=Fs, nperseg=256, noverlap=128)

# Sxx = magnitude squared (power)
plt.figure()
plt.pcolormesh(t_spec, f_spec, 10*np.log10(Sxx), shading='gouraud')  # in dB
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.title('Spectrogram')
plt.colorbar(label='Power (dB)')
plt.ylim([0, 50])  # focus on ECG frequencies
plt.show()

########################################Spectral Entropy (SpEn) ######################################3
from scipy.stats import entropy

# Normalize PSD to get probability distribution
psd_norm = Pxx / np.sum(Pxx)
SpEn = entropy(psd_norm)
print("Spectral Entropy:", SpEn)
# ################################# Relative Spectral Power (RSP) ##############################
# Define frequency bands (Hz)
bands = {'VLF': (0.5, 4), 'LF': (4, 15), 'HF': (15, 40)}
total_power = np.sum(Pxx)

RSP = {}
for band, (low, high) in bands.items():
    idx = np.logical_and(f_psd >= low, f_psd <= high)
    band_power = np.sum(Pxx[idx])
    RSP[band] = band_power / total_power

print("Relative Spectral Power:", RSP)
###### #####  #####  #####  #### ####  ###### #####  ##### #####  ### ####  ##### ####  ##### ####  ##### ####  #### ### #### ####  ######### ####

'''import wfdb
import numpy as np
from scipy.signal import butter, filtfilt, cwt, ricker
import pywt
import matplotlib.pyplot as plt

# -------------------------------
# Step 1: Load ECG record from MIT-BIH AFDB
# -------------------------------
record_name = '04015'  # Change to desired record
record = wfdb.rdrecord(record_name, pn_dir='afdb')
signal = record.p_signal[:, 0]  # Lead 0
Fs = record.fs  # Sampling frequency (usually 250 Hz)

# -------------------------------
# Step 2: Band-pass filter (0.5-40 Hz)
# -------------------------------
def bandpass_filter(sig, lowcut=0.5, highcut=40, Fs=250, order=4):
    nyq = 0.5 * Fs
    b, a = butter(order, [lowcut/nyq, highcut/nyq], btype='band')
    return filtfilt(b, a, sig)

signal_filtered = bandpass_filter(signal, Fs=Fs)'''

############################################# (DWT) for discrete wavelet transform #########################################3
wavelet_name = 'db4'
level = 4
coeffs = pywt.wavedec(signal_filtered, wavelet_name, level=level)

# coeffs[0] = approximation (low freq)
# coeffs[1:] = details (high freq)
# Now you can use coeffs for feature extraction, RSP, entropy, etc.

# Example: compute wavelet energy per level
wavelet_energy = [np.sum(np.square(c)) for c in coeffs]
###############################################CWT for continuos wavlet transform ##########################################3
scales = np.arange(1, 128)
cwt_matrix = cwt(signal_filtered, ricker, scales) 
#scalogram 
plt.figure(figsize=(12, 4))
plt.imshow(cwt_matrix, extent=[0, len(signal_filtered)/Fs, scales[-1], scales[0]],
           cmap='jet', aspect='auto')
plt.xlabel('Time [s]')
plt.ylabel('Scale')
plt.title(f'CWT Scalogram (Ricker) for record {record_name}')
plt.colorbar(label='Amplitude')
plt.show()



