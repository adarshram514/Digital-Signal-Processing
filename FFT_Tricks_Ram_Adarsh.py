#!/usr/bin/env python
# coding: utf-8

# # FFT Tricks
# 
# Adarsh Ram

# In[1]:


import numpy as np
import matplotlib.pyplot as plt

def plot_spectrum(signal, sampling_rate, title):
    N = len(signal)
    spectrum = np.fft.fft(signal)
    frequency = np.fft.fftfreq(N, d=1/sampling_rate)

    plt.plot(frequency, 2.0/N * np.abs(spectrum), label=title)

f1 = 2.0
f2 = 2.5
duration = 5.0 
sampling_rate = 10 * max(f1, f2)

t = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)

signal = np.sin(2 * np.pi * f1 * t) + np.sin(2 * np.pi * f2 * t)

#Time-domain signal plots
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(t, signal, label='Time-Domain Signal')
plt.title('Time-Domain Signal')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')

plt.subplot(2, 1, 2)
plot_spectrum(signal, sampling_rate, 'Not Zero-Padded')

plot_spectrum(signal, sampling_rate, 'Zero-Pad x2')

plot_spectrum(signal, sampling_rate, 'Zero-Pad x8')

plt.annotate(f'f1={f1} Hz', xy=(f1, 0), xytext=(f1, 30), arrowprops=dict(facecolor='black', shrink=0.05))
plt.annotate(f'f2={f2} Hz', xy=(f2, 0), xytext=(f2, 30), arrowprops=dict(facecolor='black', shrink=0.05))

plt.title('Frequency Spectrum')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.legend()
plt.tight_layout()
plt.show()


# In[3]:


import numpy as np
import matplotlib.pyplot as plt

def plot_spectrum(signal, sampling_rate, title):
    N = len(signal)
    spectrum = np.fft.fft(signal, n=4*N)
    frequency = np.fft.fftfreq(4*N, d=1/sampling_rate)

    plt.plot(frequency, 2.0/(4*N) * np.abs(spectrum), label=title)
             
f1 = 2.0
f2 = 2.5
duration = 5.0
sampling_rate = 10 * max(f1, f2)

t = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)

original_signal = np.sin(2 * np.pi * f1 * t) + np.sin(2 * np.pi * f2 * t)

plt.figure(figsize=(12, 10))
plt.subplot(3, 2, 1)
plt.plot(t, original_signal, label='Original Signal')
plt.title('Original Time-Domain Signal')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')

plt.subplot(3, 2, 2)
plot_spectrum(original_signal, sampling_rate, 'Original Spectrum')
plt.title('Original Spectrum')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.legend()

zero_stuffed_signal = np.zeros(2 * len(original_signal))
zero_stuffed_signal[::2] = original_signal

plt.subplot(3, 2, 3)
plt.plot(np.linspace(0, duration, len(zero_stuffed_signal), endpoint=False), zero_stuffed_signal, label='Zero-Stuffed Signal')
plt.title('Zero-Stuffed Time-Domain Signal')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')

plt.subplot(3, 2, 4)
plot_spectrum(zero_stuffed_signal, sampling_rate, 'Zero-Stuffed Spectrum')
plt.title('Zero-Stuffed Spectrum')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.legend()

sign_flipped_signal = (-1) ** np.arange(len(original_signal)) * original_signal

plt.subplot(3, 2, 5)
plt.plot(t, sign_flipped_signal, label='Sign-Flipped Signal')
plt.title('Sign-Flipped Time-Domain Signal')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')

plt.subplot(3, 2, 6)
plot_spectrum(sign_flipped_signal, sampling_rate, 'Sign-Flipped Spectrum')
plt.title('Sign-Flipped Spectrum')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.legend()

plt.tight_layout()
plt.show()


# Zero-stuffing and sign-flipping change the spectrum, but they don't change the frequency axis because they don't change the frequencies of the signal's sinusoids. The frequency axis can be "fixed" by changing the x-axis boundaries in the spectrum plot to show the signal's real frequency range, which might be different after zero-stuffing or sign-flipping. The frequency precision in the spectrum plot is better with zero-padding. This means that the underlying frequencies in the signal are shown more accurately and smoothly, especially when it comes to telling the difference between frequencies that are close together. Increasing the signal duration to many more periods improves the frequency resolution in the spectrum picture, making it easier to tell the difference between frequencies that are close together. On the other hand, decreasing the duration may cause frequency leaks and worse resolution. If the signal is deliberately aliased by using a sampling rate that is too low (below the Nyquist rate), frequencies above half the sampling rate will fold back into the spectrum. This will make aliasing effects and change how the original signal is represented.
