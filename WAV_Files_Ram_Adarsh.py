#!/usr/bin/env python
# coding: utf-8

# # 0. Read Files
# We need to be able to read/write files easily. This notebook gives some useful examples.
# 
# Some of the most important files will be WAV files that contain signals.
# 
# Adarsh Ram

# ## 0.a Imports

# In[1]:


import sys
import scipy as sp
import numpy as np
import matplotlib as mpl
from scipy.fftpack import fft, fftfreq
from scipy import signal

# Print lists nicely
import glob

# Read WAV files
import scipy.io.wavfile as wav

import matplotlib.pyplot as plt
#print(plt.style.available)
plt.style.use('classic')


# ## 0.b Check Versions, etc

# In[2]:


print('Python: \t{:2d}.{:1d}'
      .format(sys.version_info[0], sys.version_info[1]))
print('Matplot:\t',mpl.__version__)
print('Numpy:  \t',np.__version__)
print('SciPy:  \t',sp.__version__)


# # 1. Read a "WAV" file 
# WAV files are probably familiar to you from playing music on the computer, etc. WAV is an audio file-format popularized by Microsoft. In addition to samples of a signal (in various formats), it contains some "meta data" that simplifies things a bit, like sampling rate, type of data, type of compression used, and so on.
# 
# Try these experiments with both of the included files: "sin400.wav" and "chirp.wav"

# ## 1.a Read the original file
# This is a sinusoid at 400Hz, sampled at 2000 samples/sec, and duration of about 0.1 sec

# In[3]:


Fs, signal = wav.read('sin400.wav')


# ## 1.b Write the same data to a different file
# Notic that the sampling rate is being changed (in the WAV file), but the data is the same.

# In[4]:


wav.write('sin400_newFs.wav', Fs // 2, signal)


# # 2. Read the tweaked file and compare with the original

# In[5]:


nFs, nsignal = wav.read('sin400_newFs.wav')


# In[6]:


# Quick plot 
plt.figure(figsize=(7,6))

plt.plot(signal, color='blue', label='original')
plt.plot(nsignal, color='red', label='wrong Fs')

mm = 1.1* max(np.maximum(signal, nsignal))
plt.ylim(-mm,mm) 

plt.grid()
plt.legend()
plt.tight_layout()
plt.show()


# # Questions
# Listen to both files on your computer. You should have a program already installed that can read and play WAV files on your computer.
# 1. How does the modified one compare to the original one?
#     * What happened to the WAV file when we changed its sampling rate?
#     * Why do the plots overlap exactly?
# 2. Create your own WAV file 
#     * Use two sinusoids at different frequencies (like 200Hz and 400Hz)
#     * Mix them together (add them up)
#     * Make sure it lasts a couple of seconds
# 3. Add noise to the signal in your WAV file and listen to it.
#     * Add the noise at a specific "dB value" like "-3 dB" relative to the sinusoid
#     * Try different relative dB values
#     * Describe what happens when you listen to these different files

# 1. As the sample rate of the WAV file was changed, the frequency of the sound in the file "sin400_newFs.wav" also changed. The pitch of the sound changed noticeably when it was played after this change. The waves seem to align perfectly, though, when plotting the changed file against the original one. This is because changing the sampling rate doesn't change the numbers of the data, just when they are taken. So, even though the signal's frequency has changed, the waveform's shape and amplitude have stayed the same. This is why the plots show overlap.

# 2.

# In[7]:


import numpy as np
import scipy.io.wavfile as wav

# Parameters
duration = 2  # Duration of the WAV file in seconds
sampling_rate = 44100  # Sampling rate in Hz

# Time array
t = np.linspace(0, duration, int(sampling_rate * duration))

# Generate sinusoids
freq1 = 200  # Frequency of the first sinusoid in Hz
freq2 = 400  # Frequency of the second sinusoid in Hz
sinusoid1 = np.sin(2 * np.pi * freq1 * t)
sinusoid2 = np.sin(2 * np.pi * freq2 * t)

# Mix sinusoids
mixed_signal = sinusoid1 + sinusoid2

# Normalize the mixed signal to be within the range [-1, 1]
mixed_signal /= np.max(np.abs(mixed_signal))

# Save mixed signal to a WAV file
wav.write('mixed_signal.wav', sampling_rate, mixed_signal.astype(np.float32))


# 3.

# In[ ]:


import numpy as np
import scipy.io.wavfile as wav

def add_noise(signal, target_dB):
    # Calculate the root mean square (RMS) of the signal
    rms_signal = np.sqrt(np.mean(signal**2))
    
    # Calculate the RMS of the noise using the desired dB level relative to the signal
    target_rms_noise = rms_signal * 10**(-target_dB / 20)
    
    # Generate Gaussian white noise with the same length as the signal
    noise = np.random.normal(scale=target_rms_noise, size=len(signal))
    
    # Add noise to the signal
    noisy_signal = signal + noise
    
    return noisy_signal

# Load the original signal from the WAV file
sampling_rate, signal = wav.read('mixed_signal.wav')

# Specify the desired dB level relative to the sinusoid
target_dB_values = [-3, 0, 3]  # Try different relative dB values

# Add noise to the signal at different relative dB levels and save them to separate WAV files
for target_dB in target_dB_values:
    noisy_signal = add_noise(signal, target_dB)
    # Normalize the noisy signal to be within the range [-1, 1]
    noisy_signal /= np.max(np.abs(noisy_signal))
    # Save noisy signal to a WAV file
    wav.write(f'noisy_signal_{target_dB}dB.wav', sampling_rate, noisy_signal.astype(np.float32))

