import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
from scipy import signal

# LOAD an AUDIO SIGNAL FROM FILE
def load_audio(filename):
    signal, sample_rate = sf.read(filename)
    # If stereo, convert to mono by averaging channels
    if len(signal.shape) == 2:  # Check if stereo
        signal = signal.mean(axis=1)
    return signal, sample_rate

# GENERATE CARRIER SIGNAL USING SCIPY
def generate_carrier(frequency, duration, sample_rate):
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    carrier = np.cos(2 * np.pi * frequency * t)  # Cosine carrier wave
    return t, carrier

# APPLY AMPLITUDE MODULATION (AM)
def apply_am_modulation(input_signal, carrier_signal):
    input_signal = input_signal / np.max(np.abs(input_signal))  # Normalize input to [-1, 1]
    modulated_signal = (1 + input_signal) * carrier_signal  # AM modulation
    return modulated_signal

# DEMODULATION USING ENVELOPE DETECTION
def demodulate_am(modulated_signal, sample_rate):
    # Envelope detection: Low-pass filter to get the envelope of the modulated signal
    analytic_signal = signal.hilbert(modulated_signal)
    demodulated_signal = np.abs(analytic_signal) 
    
    # Design a low-pass filter to smooth the demodulated signal
    cutoff_freq = 1000 
    nyquist = 0.5 * sample_rate
    normal_cutoff = cutoff_freq / nyquist
    b, a = signal.butter(4, normal_cutoff, btype='low', analog=False) 
    demodulated_signal = signal.filtfilt(b, a, demodulated_signal) 
    return demodulated_signal

# Compute modulation index
def compute_modulation_index(input_signal):
    """Calculate the modulation index (mu) for AM."""
    A_m = np.max(np.abs(input_signal))  # Peak amplitude of message (input) signal
    A_c = 1  # Peak amplitude of carrier signal (normalized)
    modulation_index = A_m / A_c
    return modulation_index

# Step 4: Plot Input, Carrier, Modulated, and Demodulated Signals
def plot_signals(time, input_signal, carrier_signal, modulated_signal, demodulated_signal, zoom_range=None):
    if zoom_range:
        start, end = int(zoom_range[0] * len(time)), int(zoom_range[1] * len(time))
    else:
        start, end = 0, len(time)  # Plot the full range
    
    plt.figure(figsize=(8, 8))  # Smaller figure size for better display
    
    # Plot Input Signal
    plt.subplot(4, 1, 1)
    plt.plot(time[start:end], input_signal[start:end], color='blue')
    plt.title('Input Signal')
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')
    
    # Plot Carrier Signal
    plt.subplot(4, 1, 2)
    plt.plot(time[start:end], carrier_signal[start:end], color='green')
    plt.title('Carrier Signal (10000 Hz)')
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')
    
    # Plot Amplitude Modulated Signal
    plt.subplot(4, 1, 3)
    plt.plot(time[start:end], modulated_signal[start:end], color='red')
    plt.title('Amplitude Modulated Signal')
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')

    # Plot Demodulated Signal
    plt.subplot(4, 1, 4)
    plt.plot(time[start:end], demodulated_signal[start:end], color='purple')
    plt.title('Demodulated Signal')
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')
    
    plt.tight_layout()
    plt.show()

# Step 5: Save the Modulated Signal
def save_signal(filename, signal, sample_rate):
    sf.write(filename, signal, sample_rate)

# ******* Main Execution *******
if __name__ == '__main__':
    # Load the input audio signal (replace 'helloHello.wav' with your actual file)
    input_file = 'helloHello.wav'
    input_signal, sample_rate = load_audio(input_file)
    
    # Generate the carrier signal with a frequency of 10 kHz
    duration = len(input_signal) / sample_rate  # Duration should match the input audio
    carrier_frequency = 10000  # Carrier frequency (Hz), increase for better modulation
    t, carrier_signal = generate_carrier(carrier_frequency, duration, sample_rate)
    
    # Apply amplitude modulation
    modulated_signal = apply_am_modulation(input_signal, carrier_signal)
    
    # Demodulate the modulated signal
    demodulated_signal = demodulate_am(modulated_signal, sample_rate)
    
    # Compute the modulation index
    modulation_index = compute_modulation_index(input_signal)
    print(f'Modulation Index: {modulation_index}')
    
    # Plot the input, carrier, modulated, and demodulated signals
    plot_signals(t, input_signal, carrier_signal, modulated_signal, demodulated_signal, zoom_range=None)  # Use None for full plot
    
    # Save the modulated signal as a new .wav file
    output_file = 'modulated_audio.wav'
    save_signal(output_file, modulated_signal, sample_rate)
