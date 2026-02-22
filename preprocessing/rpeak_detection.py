# This is the script for R peak detection using multiple packages and llibraries 
import neurokit2 as nk
import wfdb.processing as wp
from biosppy.signals import ecg as biosppy_ecg
from scipy.signal import butter, filtfilt, find_peaks


FS = 500
def rpeaks_neurokit(ecg, fs=FS):
    """
    R-peak detection using NeuroKit2
    """
    signals, info = nk.ecg_process(ecg, sampling_rate=fs)
    r_peaks = info["ECG_R_Peaks"]
    return np.array(r_peaks)

# -------------------------------


def rpeaks_wfdb(ecg, fs=FS):
    """
    R-peak detection using WFDB XQRS
    """
    r_peaks = wp.xqrs_detect(ecg, fs=fs)
    return np.array(r_peaks)
# -------------------------------


def rpeaks_biosppy(ecg, fs=FS):
    """
    R-peak detection using BioSPPy
    """
    out = biosppy_ecg.ecg(signal=ecg, sampling_rate=fs, show=False)
    r_peaks = out["rpeaks"]
    return np.array(r_peaks)
# -------------------------------


def rpeaks_pan_tompkins(ecg, fs=FS):
    """
    Simplified Pan–Tompkins-style detector
    """

    # Bandpass filter (5–15 Hz)
    low, high = 5 / (fs / 2), 15 / (fs / 2)
    b, a = butter(1, [low, high], btype="band")
    filtered = filtfilt(b, a, ecg)

    # Derivative
    diff = np.diff(filtered)

    # Squaring
    squared = diff ** 2

    # Moving window integration
    window = int(0.15 * fs)
    mwa = np.convolve(squared, np.ones(window)/window, mode="same")

    # Peak detection
    peaks, _ = find_peaks(mwa, distance=0.3*fs)
    return peaks
# -------------------------------
def detect_rpeaks(ecg, fs=FS, method="neurokit"):
    if method == "neurokit":
        return rpeaks_neurokit(ecg, fs)
    elif method == "wfdb":
        return rpeaks_wfdb(ecg, fs)
    elif method == "biosppy":
        return rpeaks_biosppy(ecg, fs)
    elif method == "pan_tompkins":
        return rpeaks_pan_tompkins(ecg, fs)
    else:
        raise ValueError("Unknown R-peak detection method")
# -------------------------------
#usage example
#r_peaks = detect_rpeaks(lead_ii, method="neurokit")

#rr interval 
#def segment_rr_intervals(r_peaks, fs):
    #rr_intervals = np.diff(r_peaks) / fs  # in seconds
    #return rr_intervals