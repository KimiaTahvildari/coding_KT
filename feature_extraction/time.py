# (CV_RR) for coefficient of variation 
# (dRDDC) for RR interval distribution difference curve 
# the mean standard deviation  
# skewness
# (IFFT)for Inverse Fast Fourier Transform
# (HRV) for heart rate variability ???????
# (meanRR) for Mean RR interval
# (SDNN) for Standard deviation of the RR interval
# (SKRR) for Skewness of RR interval
# (PNN20) for Proportion of the adjacent interval difference greater than 20 ms 
# (RCV𝛥RR) for the Reciprocal of the Coefficient of Variation in 𝛥RR 
# EMD-IMF 


import numpy as np
from scipy.stats import skew
from scipy.fft import ifft
# Optional: For EMD
from PyEMD import EMD  

# Example RR interval-based functions
def CV_RR(rr_intervals):
    return np.std(rr_intervals) / np.mean(rr_intervals)

def dRDDC(rr_intervals):
    hist, bin_edges = np.histogram(rr_intervals, bins='auto', density=True)
    return hist  # You can define distance measure later

def mean_std(rr_intervals):
    return np.mean(rr_intervals), np.std(rr_intervals)

def skewness_rr(rr_intervals):
    return skew(rr_intervals)

def IFFT(signal):
    return np.real(ifft(signal))

def HRV(rr_intervals):
    return np.std(rr_intervals) / np.mean(rr_intervals)  # Example: SDNN / mean

def meanRR(rr_intervals):
    return np.mean(rr_intervals)

def SDNN(rr_intervals):
    return np.std(rr_intervals)

def SKRR(rr_intervals):
    return skew(rr_intervals)

def PNN20(rr_intervals):
    diff = np.diff(rr_intervals)
    return np.sum(np.abs(diff) > 20) / len(diff)

def RCV_deltaRR(rr_intervals):
    diff = np.diff(rr_intervals)
    return 1 / (np.std(diff) / np.mean(diff))

def EMD_IMF(signal):
    emd = EMD()
    imfs = emd.emd(signal)
    return imfs
