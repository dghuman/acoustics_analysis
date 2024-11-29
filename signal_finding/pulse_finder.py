import numpy as np
import matplotlib.pyplot as plt

# Parameters for the sine wave
signal_length = 50  # Number of samples for the known sine wave signal
frequency = 10       # Frequency of the sine wave (in Hz)
sampling_rate = 1000  # Sampling rate (samples per second)
t = np.linspace(0, 1, signal_length, endpoint=False)  # Time vector

# Known sine wave signal
known_signal = np.sin(2 * np.pi * frequency * t)

# Example noisy data with repeating sine wave signals
np.random.seed(42)  # For reproducibility
noise = np.random.normal(0, 0.5, signal_length*10)  # Random noise
#repeating_signal = np.tile(known_signal, 10)  # Repeat the sine wave signal
repeating_signal = np.zeros(len(known_signal)*10)
# Set signal at half-way point
repeating_signal[250:300] = known_signal
#repeating_signal[300:325] = known_signal[:25]
noisy_data = repeating_signal + noise  # Add noise to the repeated signal

# Perform cross-correlation
cross_corr = np.correlate(noisy_data, known_signal, mode='full')

# Plot the noisy data and cross-correlation
fig = plt.figure(figsize=(10, 5))

gs = fig.add_gridspec(2, 1, hspace=0)
axes = gs.subplots(sharex='col')

# Plot the noisy data
#plt.subplot(2, 1, 1)

axes[0].plot(repeating_signal)
axes[0].plot(noisy_data)
#axes[0].set_title("Noisy Data with Repeating Sine Wave Signal")
axes[0].set_xlabel("Sample")
axes[0].set_ylabel("Amplitude")

# Plot the cross-correlation
axes[1].plot(cross_corr)
#axes[1].set_title("Cross-Correlation with Known Sine Wave Signal")
axes[1].set_xlabel("Lag")
axes[1].set_ylabel("Correlation")

plt.show()
