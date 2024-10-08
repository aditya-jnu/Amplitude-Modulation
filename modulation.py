import numpy as np 
import matplotlib.pyplot as plt
import soundfile as sf
from scipy import signal

# LOAD an AUDIO SIGNAL FROM FILE
def load_audio(filename):
    signal, sample_rate = sf.read(filename)
    if len(signal.shape) == 2:  # Check if stereo
        signal = signal.mean(axis=1)  # Convert to mono
    return signal, sample_rate

# Generate carrier signal
def generate_carrier(frequency, duration, sample_rate):
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    carrier = np.sin(2 * np.pi * frequency * t)  # Sine carrier wave
    return t, carrier

# APPLY AMPLITUDE MODULATION (AM)
def apply_am_modulation(input_signal, carrier_signal, modulation_index):
    input_signal = input_signal / np.max(np.abs(input_signal))  # Normalize input to [-1, 1]
    modulated_signal = (1 + modulation_index * input_signal) * carrier_signal  # AM modulation
    return modulated_signal

# DEMODULATION USING ENVELOPE DETECTION
def demodulate_am(modulated_signal, sample_rate):
    analytic_signal = signal.hilbert(modulated_signal)
    demodulated_signal = np.abs(analytic_signal)  # Envelope detection
    
    # Design a low-pass filter to smooth the demodulated signal
    cutoff_freq = 1000  # Cutoff frequency of the low-pass filter
    nyquist = 0.5 * sample_rate
    normal_cutoff = cutoff_freq / nyquist
    b, a = signal.butter(4, normal_cutoff, btype='low', analog=False)
    demodulated_signal = signal.filtfilt(b, a, demodulated_signal)
    return demodulated_signal

# Plot Input, Carrier, Modulated, and Demodulated Signals
def plot_signals(time, input_signal, carrier_signal, modulated_signal, demodulated_signal, zoom_range=None):
    if zoom_range:
        start, end = int(zoom_range[0] * len(time)), int(zoom_range[1] * len(time))
    else:
        start, end = 0, len(time)  # Plot the full range
    
    plt.figure(figsize=(8, 8))
    
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

# Save the signal to a file
def save_signal(filename, signal, sample_rate):
    sf.write(filename, signal, sample_rate)

# ******* Main Execution *****
if __name__ == '__main__':
    # Load the input audio signal (replace 'helloHello.wav' with your actual file)
    input_file = 'helloHello.wav'  # Replace with your audio file
    input_signal, sample_rate = load_audio(input_file)
    
    # Generate the carrier signal with a frequency of 10 kHz
    duration = len(input_signal) / sample_rate
    carrier_frequency = 10000  # 10 kHz
    t, carrier_signal = generate_carrier(carrier_frequency, duration, sample_rate)
    
    # Test different modulation indices
    modulation_cases = {'Under-Modulation (<1)': 0.5, 
                        'Critical-Modulation (=1)': 1.0, 
                        'Over-Modulation (>1)': 1.5}
    
    for case_name, modulation_index in modulation_cases.items():
        print(f"Processing {case_name} with modulation index {modulation_index}")
        
        # Apply amplitude modulation
        modulated_signal = apply_am_modulation(input_signal, carrier_signal, modulation_index)
        
        # Demodulate the modulated signal
        demodulated_signal = demodulate_am(modulated_signal, sample_rate)
        
        # Plot the signals
        plot_signals(t, input_signal, carrier_signal, modulated_signal, demodulated_signal, zoom_range=None)
        
        # Save the modulated signal as a new .wav file
        output_file_modulated = f'modulated_audio_{case_name.replace(" ", "_").replace("<", "lt").replace(">", "gt")}.wav'
        save_signal(output_file_modulated, modulated_signal, sample_rate)
        
        # Save the demodulated signal as a new .wav file
        output_file_demodulated = f'demodulated_audio_{case_name.replace(" ", "_").replace("<", "lt").replace(">", "gt")}.wav'
        save_signal(output_file_demodulated, demodulated_signal, sample_rate)
