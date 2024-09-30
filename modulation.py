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
    """Generate a sine wave carrier signal using scipy's signal generator."""
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    carrier = np.cos(2 * np.pi * frequency * t)  # Cosine carrier wave
    return t, carrier

# APPLY AMPLITUDE MODULATION (AM)
def apply_am_modulation(input_signal, carrier_signal):
    """Perform amplitude modulation by modulating the carrier signal using the input signal. Normalize the    
       input signal to [-1, 1] for clearer visualization."""
    input_signal = input_signal / np.max(np.abs(input_signal))  # Normalize input to [-1, 1]
    modulated_signal = (1 + input_signal) * carrier_signal  # AM modulation
    return modulated_signal

# Step 4: Plot Input, Carrier, and Modulated Signals (Zoom out)
def plot_signals(time, input_signal, carrier_signal, modulated_signal, zoom_range=None):
    """
    Plot the input, carrier, and modulated signals for comparison.
    Either zoom in on a specific time range or plot the full signals.
    """
    if zoom_range:
        start, end = int(zoom_range[0] * len(time)), int(zoom_range[1] * len(time))
    else:
        start, end = 0, len(time)  # Plot the full range
    
    plt.figure(figsize=(12, 8))
    
    # Plot Input Signal
    plt.subplot(3, 1, 1)
    plt.plot(time[start:end], input_signal[start:end], color='blue')
    plt.title('Input Signal')
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')
    
    # Plot Carrier Signal
    plt.subplot(3, 1, 2)
    plt.plot(time[start:end], carrier_signal[start:end], color='green')
    plt.title('Carrier Signal (10 Hz)')
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')
    
    # Plot Amplitude Modulated Signal
    plt.subplot(3, 1, 3)
    plt.plot(time[start:end], modulated_signal[start:end], color='red')
    plt.title('Amplitude Modulated Signal')
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')
    
    plt.tight_layout()
    plt.show()

# Step 5: Save the Modulated Signal
def save_signal(filename, signal, sample_rate):
    """
    Save the modulated signal to a .wav file.
    """
    sf.write(filename, signal, sample_rate)

# ******* Main Execution *******
if __name__ == '__main__':
    # Load the input audio signal (replace 'helloHello.wav' with your actual file)
    input_file = 'helloHello.wav'
    input_signal, sample_rate = load_audio(input_file)
    
    # Generate the carrier signal with a frequency of 10 Hz
    duration = len(input_signal) / sample_rate  # Duration should match the input audio
    carrier_frequency = 10  # Carrier frequency (Hz)
    t, carrier_signal = generate_carrier(carrier_frequency, duration, sample_rate)
    
    # Apply amplitude modulation
    modulated_signal = apply_am_modulation(input_signal, carrier_signal)
    
    # Plot the input, carrier, and modulated signals (zoomed out or full range)
    plot_signals(t, input_signal, carrier_signal, modulated_signal, zoom_range=None)  # Use None for full plot
    
    # Save the modulated signal as a new .wav file
    output_file = 'modulated_audio.wav'
    save_signal(output_file, modulated_signal, sample_rate)
