# preprocessing/segmentation.py
# this module is for choosing the leads (segments )
import wfdb
import numpy as np
import neurokit2 as nk
from .noise_removal_preprocess import apply_filter_multilead

def select_lead_I_II(ecg):
    return ecg[[0, 1], :]

def load_and_preprocess(record_path, fs):
    signal, _ = wfdb.rdsamp(record_path)
    ecg = signal.T
    ecg = apply_filter_multilead(ecg, fs)
    ecg = select_lead_I_II(ecg)
    return ecg



def load_and_preprocess_rr(record_path, fs):
    """
    Returns RR intervals in seconds
    """

    ecg = load_and_preprocess(record_path, fs)  # your existing function

    # Select lead II
    lead = ecg[1] if ecg.ndim == 2 else ecg

    # R-peak detection
    _, info = nk.ecg_process(lead, sampling_rate=fs)
    r_peaks = info["ECG_R_Peaks"]

    # RR intervals
    rr = np.diff(r_peaks) / fs

    return rr
