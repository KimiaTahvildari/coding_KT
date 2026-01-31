# (Fast SampEn) for fast sample entropy
# (Esh) Information Entropy
# (CosEn) Coefficient of sample entropy
# (LZC)Lempel-Ziv complexity
# C0-complexity
# (REn) Renyi’s Entropy 

import numpy as np
from scipy.stats import entropy

# -------------------------------
# Fast Sample Entropy (SampEn)
# -------------------------------
def fast_sampen(signal, m=2, r=0.2):
    N = len(signal)
    r *= np.std(signal)
    
    def _phi(m):
        x = np.array([signal[i:i+m] for i in range(N - m + 1)])
        C = np.sum(np.max(np.abs(x[:, None] - x[None, :]), axis=2) <= r, axis=0) - 1
        return np.sum(C) / (N - m + 1)
    
    return -np.log(_phi(m+1) / _phi(m) + 1e-10)  # add small value to avoid log(0)

# -------------------------------
# Information Entropy (Shannon Entropy)
# -------------------------------
def information_entropy(signal, bins=50):
    hist, _ = np.histogram(signal, bins=bins, density=True)
    hist = hist[hist > 0]
    return -np.sum(hist * np.log2(hist))

# -------------------------------
# Coefficient of Sample Entropy (CosEn)
# -------------------------------
def cos_en(signal, m=2, r=0.2):
    se = fast_sampen(signal, m=m, r=r)
    return se / (np.std(signal) + 1e-10)

# -------------------------------
# Lempel-Ziv Complexity (LZC)
# -------------------------------
def lz_complexity(signal):
    # Convert to binary using median threshold
    median = np.median(signal)
    binary_seq = (signal > median).astype(int)
    i, k, l = 0, 1, 1
    c = 1
    n = len(binary_seq)
    while True:
        if binary_seq[i+k-1] != binary_seq[l+k-1]:
            if k > l:
                i = i + 1
                if i == l:
                    c += 1
                    l += k
                    if l + 1 > n:
                        break
                    i = 0
                k = 1
            else:
                k += 1
        else:
            k += 1
            if l + k > n:
                c += 1
                break
    return c

# -------------------------------
# C0-Complexity
# -------------------------------
def c0_complexity(signal):
    # Number of sign changes in first derivative
    diff_signal = np.diff(signal)
    sign_changes = np.sum(diff_signal[:-1] * diff_signal[1:] < 0)
    return sign_changes / len(signal)

# -------------------------------
# Renyi's Entropy (alpha=2 by default)
# -------------------------------
def renyi_entropy(signal, alpha=2, bins=50):
    hist, _ = np.histogram(signal, bins=bins, density=True)
    hist = hist[hist > 0]
    if alpha == 1:
        return -np.sum(hist * np.log(hist))
    else:
        return 1/(1-alpha) * np.log2(np.sum(hist**alpha))
