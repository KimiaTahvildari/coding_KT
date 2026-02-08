import wfdb
import matplotlib.pyplot as plt
import numpy as np

# Replace with path to your record
record_path = r'data\AFDB\04015'

# Read signal and metadata
record = wfdb.rdrecord(record_path)
print(f"Signal shape: {record.p_signal.shape}")
print(f"Sampling rate: {record.fs}")
print(f"Signal names: {record.sig_name}")

# Plot the ECG
wfdb.plot_wfdb(record=record, title='ECG Signal from 04015')
plt.show()
annotation = wfdb.rdann(record_path, 'atr')
print("Annotation sample indices:", annotation.sample[:10])
print("Annotation symbols:", annotation.symbol[:10])

ecg_lead_0 = record.p_signal[:, 0]  # first lead
fs = record.fs

# Get 10 seconds of ECG starting from 100s
start_sec = 100
duration_sec = 10
start_idx = int(start_sec * fs)
end_idx = start_idx + int(duration_sec * fs)

segment = ecg_lead_0[start_idx:end_idx]

# Plot the segment
plt.figure(figsize=(10, 3))
plt.plot(np.arange(len(segment)) / fs, segment)
plt.title('10-Second Segment from Lead 0')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude (mV)')
plt.grid()
plt.show()

for sample, symbol in zip(annotation.sample, annotation.symbol):
    if symbol == 'A' and start_idx <= sample <= end_idx:
        print("AF is present in this segment!")

    elif symbol == 'N' and start_idx <= sample <= end_idx:
        print("Normal rhythm is present in this segment!")  