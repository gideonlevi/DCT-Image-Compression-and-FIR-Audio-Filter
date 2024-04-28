import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import cv2
import math
import scipy.io.wavfile
import tqdm

audio = scipy.io.wavfile.read("./HW2_Mix_2.wav")
sampling_rate = audio[0]
print('Audio sampling rate:', sampling_rate)
audio = np.array(audio[1], dtype=float)

# #TIME DOMAIN SIGNAL
# plt.plot(audio[:])
# plt.ylabel("Amplitude")
# plt.xlabel("Time")
# plt.show()

#FREQUENCY DOMAIN SIGNAL
# Calculate n/2 to normalize the FFT output
n = audio.size
normalize = n/2
fft_signal = np.fft.fft(audio)
fft_freq = np.fft.fftfreq(n, d=1.0/sampling_rate)
# only use the first n/2+1 elements
# fft_freq = fft_freq[:math.ceil(n/2)]
# fft_signal = fft_signal[:math.ceil(n/2)]
plt.plot(fft_freq, np.abs(fft_signal/normalize))
plt.ylabel("Amplitude")
plt.xlabel("Frequency")
plt.xlim(-2000, 2000)
plt.savefig('./output/input.png')
plt.show()


#FIR FILTER
select_filter = int(input('Select filter(1,2,3):'))

if select_filter == 1:
    # FILTER 1 (LOWPASS FILTER)
    cutoff_freq = 380
    cutoff_freq = cutoff_freq/sampling_rate
    w_c = 2*math.pi*cutoff_freq
    filter_order = 32000
    middle = filter_order//2
    low_filter = np.zeros((filter_order), dtype=float)

    for i in range(-middle, middle):
        if i == 0:
            low_filter[middle] = 1
        else:
            low_filter[i+middle] = math.sin(2*math.pi*cutoff_freq*i)/(math.pi*i)
    low_filter[middle] = 2*cutoff_freq

    #multiply the elements of filter by a windowing function
    #blackmann window function
    for i in range(0, filter_order):
        low_filter[i] = low_filter[i] * (0.42 + 0.5*math.cos((2*math.pi*i)/(n-1)) + 0.08*math.cos((4*math.pi*i)/(n-1)))

    #1D CONVOLUTION
    filtered_signal_1 = audio.copy() #just to get array with same size
    # Pad the input signal with zeros
    pad_length = len(low_filter) // 2
    audio_padded = np.pad(audio, (pad_length, pad_length), mode='constant')
    # Compute the convolution
    for i in tqdm.trange(len(audio)):
        filtered_signal_1[i] = np.dot(audio_padded[i:i+len(low_filter)], low_filter)

    #output audio
    scipy.io.wavfile.write('./output/Filter1Lowpass_380.wav', sampling_rate, np.int16(filtered_signal_1))

    #output spectrums
    # Calculate n/2 to normalize the FFT output
    n = filtered_signal_1.size
    normalize = n/2
    fft_signal_1 = np.fft.fft(filtered_signal_1)
    fft_freq_1 = np.fft.fftfreq(n, d=1.0/sampling_rate) #(?)

    plt.plot(fft_freq_1, np.abs(fft_signal_1/normalize))
    plt.ylabel("Amplitude")
    plt.xlabel("Frequency")
    plt.title('Output spectrum by filter 1 (Lowpass)')
    plt.xlim(-2000, 2000)
    plt.savefig('./output/output_by_Filter1Lowpass.png')
    plt.show()

    #filter shape
    plt.plot(low_filter)
    plt.title('Filter 1 (Lowpass) shape')
    plt.xlim(11000, 21000)
    plt.savefig('./output/Filter1Lowpass_shape.png') 
    plt.show()

    #filter spectrums
    n = filter_order
    normalize = n/2
    fft_signal_1 = np.fft.fft(low_filter)
    fft_freq_1 = np.fft.fftfreq(n, d=1.0/sampling_rate) #(?)
    plt.plot(fft_freq_1, np.abs(fft_signal_1/normalize))
    plt.title('Filter 1 (Lowpass) filter spectrum')
    plt.ylabel("Amplitude")
    plt.xlabel("Frequency")
    plt.xlim(-2000, 2000)
    plt.savefig('./output/Filter1Lowpass_spectrum.png')
    plt.show()


    #DOWN SAMPLE 2KHZ
    original_sampling_rate = sampling_rate
    new_sampling_rate = 2000
    downsample_factor = original_sampling_rate / new_sampling_rate
    num_samples_downsampled = int(len(filtered_signal_1)/downsample_factor)
    downsampled_signal = np.zeros((num_samples_downsampled), dtype=float)
    #downsample the signal
    for i in range(num_samples_downsampled):
        downsampled_signal[i] = filtered_signal_1[int(i * downsample_factor)]
    #output audio
    scipy.io.wavfile.write('./output/Filter1Lowpass_380_2khz.wav', new_sampling_rate, np.int16(downsampled_signal))

    
    #ONE-FOLD ECHO
    one_echo_filter = np.zeros((3201), dtype=float)
    one_echo_filter[0] = 1
    one_echo_filter[3200] = 0.8

    one_echo_filtered_signal = filtered_signal_1.copy() #just to get array with same size
    # Pad the input signal with zeros
    pad_length = len(one_echo_filter) // 2
    audio_padded = np.pad(filtered_signal_1, (pad_length, pad_length), mode='constant')
    # Compute the convolution
    for i in tqdm.trange(len(audio)):
        one_echo_filtered_signal[i] = np.dot(audio_padded[i:i+len(one_echo_filter)], one_echo_filter)
    #output audio
    scipy.io.wavfile.write('./output/Echo_one.wav', sampling_rate, np.int16(one_echo_filtered_signal))

    #MULTIPLE-FOLD ECHO
    multi_echo_filter = np.zeros((3201), dtype=float)
    multi_echo_filter[0] = 1
    multi_echo_filter[3200] = -0.8

    multi_echo_filtered_signal = filtered_signal_1.copy() #just to get array with same size
    # Pad the input signal with zeros
    pad_length = len(multi_echo_filter) // 2
    audio_padded = np.pad(filtered_signal_1, (pad_length, pad_length), mode='constant')
    # Compute the convolution
    for i in tqdm.trange(len(audio)):
        multi_echo_filtered_signal[i] = np.dot(audio_padded[i:i+len(multi_echo_filter)], multi_echo_filter)
    #output audio
    scipy.io.wavfile.write('./output/Echo_multiple.wav', sampling_rate, np.int16(multi_echo_filtered_signal))


elif select_filter == 2:
    # FILTER 2 (BANDPASS FILTER)
    freq_1 = 380
    freq_2 = 780
    freq_1 = freq_1/sampling_rate
    freq_2 = freq_2/sampling_rate
    filter_order = 32000
    middle = filter_order//2
    bandpass_filter_2 = np.zeros((filter_order), dtype=float)

    for i in range(-middle, middle):
        if i == 0:
            bandpass_filter_2[middle] = 1
        else:
            bandpass_filter_2[i+middle] = math.sin(2*math.pi*freq_2*i)/(math.pi*i) - math.sin(2*math.pi*freq_1*i)/(math.pi*i)
    bandpass_filter_2[middle] = 2*(freq_2-freq_1)

    #blackmann window function
    for i in range(0, filter_order):
        bandpass_filter_2[i] = bandpass_filter_2[i] * (0.42 + 0.5*math.cos((2*math.pi*i)/(n-1)) + 0.08*math.cos((4*math.pi*i)/(n-1)))

    filtered_signal_2 = audio.copy() #just to get array with same size
    # Pad the input signal with zeros
    pad_length = len(bandpass_filter_2) // 2
    audio_padded = np.pad(audio, (pad_length, pad_length), mode='constant')
    # Compute the convolution
    for i in tqdm.trange(len(audio)):
        filtered_signal_2[i] = np.dot(audio_padded[i:i+len(bandpass_filter_2)], bandpass_filter_2)

    #output audio
    scipy.io.wavfile.write('./output/Filter2Bandpass_380_780.wav', sampling_rate, np.int16(filtered_signal_2))

    #output spectrums
    n = filtered_signal_2.size
    normalize = n/2
    fft_signal_2 = np.fft.fft(filtered_signal_2)
    fft_freq_2 = np.fft.fftfreq(n, d=1.0/sampling_rate) #(?)
    plt.plot(fft_freq_2, np.abs(fft_signal_2/normalize))
    plt.ylabel("Amplitude")
    plt.xlabel("Frequency")
    plt.title('Output spectrum by filter 2 (Bandpass)')
    plt.xlim(-2000, 2000)
    plt.savefig('./output/output_by_Filter2Bandpass.png')
    plt.show()

    #filter shape
    plt.plot(bandpass_filter_2)
    plt.title('Filter 2 (Bandpass) shape')
    plt.xlim(11000, 21000)
    plt.savefig('./output/Filter2Bandpass_shape.png') 
    plt.show()

    #filter spectrums
    n = filter_order
    normalize = n/2
    fft_signal_2 = np.fft.fft(bandpass_filter_2)
    fft_freq_2 = np.fft.fftfreq(n, d=1.0/sampling_rate) #(?)
    plt.plot(fft_freq_2, np.abs(fft_signal_2/normalize))
    plt.title('Filter 2 (Bandpass) spectrum')
    plt.ylabel("Amplitude")
    plt.xlabel("Frequency")
    plt.xlim(-2000, 2000)
    plt.savefig('./output/Filter2Bandpass_spectrum.png')
    plt.show()


    #DOWN SAMPLE 2KHZ
    original_sampling_rate = sampling_rate
    new_sampling_rate = 2000
    downsample_factor = original_sampling_rate / new_sampling_rate
    num_samples_downsampled = int(len(filtered_signal_2)/downsample_factor)
    downsampled_signal = np.zeros((num_samples_downsampled), dtype=float)
    #downsample the signal
    for i in range(num_samples_downsampled):
        downsampled_signal[i] = filtered_signal_2[int(i * downsample_factor)]
    #output audio
    scipy.io.wavfile.write('./output/Filter2Bandpass_380_780_2khz.wav', new_sampling_rate, np.int16(downsampled_signal))



elif select_filter == 3:
    #FILTER 3 (HIGHPASS FILTER)
    cutoff_freq = 780
    cutoff_freq = cutoff_freq/sampling_rate
    w_c = 2*math.pi*cutoff_freq
    filter_order = 32000
    middle = filter_order//2
    highpass_filter = np.zeros((filter_order), dtype=float)

    for i in range(-middle, middle):
        if i == 0:
            highpass_filter[middle] = 1
        else:
            highpass_filter[i+middle] = -math.sin(2*math.pi*cutoff_freq*i)/(math.pi*i)
    highpass_filter[middle] = 1 - 2*cutoff_freq

    #blackmann window function
    for i in range(0, filter_order):
        highpass_filter[i] = highpass_filter[i] * (0.42 + 0.5*math.cos((2*math.pi*i)/(n-1)) + 0.08*math.cos((4*math.pi*i)/(n-1)))

    filtered_signal_3 = audio.copy() #just to get array with same size
    # Pad the input signal with zeros
    pad_length = len(highpass_filter) // 2
    audio_padded = np.pad(audio, (pad_length, pad_length), mode='constant')
    # Compute the convolution
    for i in tqdm.trange(len(audio)):
        filtered_signal_3[i] = np.dot(audio_padded[i:i+len(highpass_filter)], highpass_filter)

    #output audio
    scipy.io.wavfile.write('./output/Filter3Highpass_780.wav', sampling_rate, np.int16(filtered_signal_3))

    #output spectrums
    n = filtered_signal_3.size
    normalize = n/2
    fft_signal_3 = np.fft.fft(filtered_signal_3)
    fft_freq_3 = np.fft.fftfreq(n, d=1.0/sampling_rate) #(?)
    plt.plot(fft_freq_3, np.abs(fft_signal_3/normalize))
    plt.ylabel("Amplitude")
    plt.xlabel("Frequency")
    plt.title('Output spectrum by filter 3 (Highpass)')
    plt.xlim(-2000, 2000)
    plt.savefig('./output/output_by_Filter3Highpass.png')
    plt.show()

    #filter shape
    plt.plot(highpass_filter)
    plt.title('Filter 3 (Highpass) shape')
    plt.xlim(11000, 21000)
    plt.savefig('./output/Filter3Highpass_shape.png') 
    plt.show()

    #filter spectrums
    n = filter_order
    normalize = n/2
    fft_signal_3 = np.fft.fft(highpass_filter)
    fft_freq_3 = np.fft.fftfreq(n, d=1.0/sampling_rate) #(?)
    plt.plot(fft_freq_3, np.abs(fft_signal_3/normalize))
    plt.title('Filter 3 (Highpass) spectrum')
    plt.ylabel("Amplitude")
    plt.xlabel("Frequency")
    plt.xlim(-2000, 2000)
    plt.savefig('./output/Filter3Highpass_spectrum.png')
    plt.show()


    #DOWN SAMPLE 2KHZ
    original_sampling_rate = sampling_rate
    new_sampling_rate = 2000
    downsample_factor = original_sampling_rate / new_sampling_rate
    num_samples_downsampled = int(len(filtered_signal_3)/downsample_factor)
    downsampled_signal = np.zeros((num_samples_downsampled), dtype=float)
    #downsample the signal
    for i in range(num_samples_downsampled):
        downsampled_signal[i] = filtered_signal_3[int(i * downsample_factor)]
    #output audio
    scipy.io.wavfile.write('./output/Filter3Highpass_780_2khz.wav', new_sampling_rate, np.int16(downsampled_signal))