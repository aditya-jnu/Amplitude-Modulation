import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf

# Step 1: Generate or Load a Signal
def generate_signal(frequency=440, duration=1, sample_rate=44100):
    """
    Generate a simple sine wave signal.
    frequency: Frequency of the sine wave in Hz (default 440 Hz for A4).
    duration: Duration of the signal in seconds.
    sample_rate: Samples per second (default 44.1 kHz for audio).
    """
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    signal = np.sin(2 * np.pi * frequency * t)  # Simple sine wave
    return t, signal

# Step 2: Define a Carrier Signal
def generate_carrier(frequency=10000, duration=1, sample_rate=44100):
    """
    Generate a carrier signal for modulation.
    frequency: Frequency of the carrier wave in Hz.
    """
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    carrier = np.sin(2 * np.pi * frequency * t)
    return t, carrier

# Step 3: Apply Amplitude Modulation (AM)
def apply_am_modulation(signal, carrier):
    """
    Perform amplitude modulation by modulating the carrier signal using the input signal.
    """
    modulated_signal = signal * carrier  # Amplitude Modulation
    return modulated_signal

# Step 4: Plot and Visualize
def plot_signals(time, original_signal, modulated_signal):
    """
    Plot the original and modulated signals for comparison.
    """
    plt.figure(figsize=(10, 6))
    
    # Plot the original signal
    plt.subplot(2, 1, 1)
    plt.plot(time, original_signal)
    plt.title('Original Signal')
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')
    
    # Plot the modulated signal
    plt.subplot(2, 1, 2)
    plt.plot(time, modulated_signal)
    plt.title('Amplitude Modulated Signal')
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')

    plt.tight_layout()
    plt.show()

# Step 5: Save the Modulated Sound
def save_signal(filename, signal, sample_rate):
    """
    Save the modulated signal to a .wav file.
    """
    sf.write(filename, signal, sample_rate)

# Main Execution
if __name__ == '__main__':
    # Generate original signal (sine wave)
    sample_rate = 44100  # 44.1 kHz sample rate
    duration = 2  # 2 seconds duration
    t, original_signal = generate_signal(frequency=440, duration=duration, sample_rate=sample_rate)
    
    # Generate carrier signal
    _, carrier_signal = generate_carrier(frequency=10000, duration=duration, sample_rate=sample_rate)
    
    # Apply amplitude modulation
    modulated_signal = apply_am_modulation(original_signal, carrier_signal)
    
    # Plot the signals
    plot_signals(t, original_signal, modulated_signal)
    
    # Save the modulated signal to a .wav file
    save_signal("modulated_sound.wav", modulated_signal, sample_rate)
