# preprocessing/segmentation.py

import wfdb
import numpy as np
from .noise_removal_preprocess import apply_filter_multilead

def select_lead_I_II(ecg):
    return ecg[[0, 1], :]

def load_and_preprocess(record_path, fs):
    signal, _ = wfdb.rdsamp(record_path)
    ecg = signal.T
    ecg = apply_filter_multilead(ecg, fs)
    ecg = select_lead_I_II(ecg)
    return ecg
