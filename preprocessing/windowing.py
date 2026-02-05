# preprocessing/windowing.py
# this module is for cut signal into fixed-length chunks

import numpy as np
def extract_windows(ecg, label, window_samples):
    windows, labels = [], []
    for i in range(0, ecg.shape[1], window_samples):
        w = ecg[:, i:i + window_samples]
        if w.shape[1] == window_samples:
            windows.append(w)
            labels.append(label)
    return windows, labels



def extract_rr_sequences(rr, label, seq_len=10, stride=1):
    """
    rr: array of RR intervals
    seq_len: number of RR intervals per sample
    """

    X, y = [], []

    for start in range(0, len(rr) - seq_len + 1, stride):
        rr_seq = rr[start:start + seq_len]

        # Z-score normalization (per segment, as in paper)
        rr_seq = (rr_seq - np.mean(rr_seq)) / (np.std(rr_seq) + 1e-8)

        X.append(rr_seq)
        y.append(label)

    return X, y
