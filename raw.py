import torch
import pandas as pd
import librosa
import numpy as np
from scipy.signal import butter, lfilter
import os
#make path be the route of where the audio and txt files are
path2 = '/Data/FishSound/Keelung Chaojing/all/'
path = '/Data/FishSound/Keelung Chaojing/all'
#csvname is the result of sampling
csvname1 = '/Data/FishSound/Keelung Chaojing/800framechaotest.csv'
csvname2 = '/Data/FishSound/Keelung Chaojing/label800chaotest.csv'
miaonum = 21
lyunum = 17
chaonum = 17 
str_num = chaonum #adjust str_num to different dataset
lowcut = 80  # Low cutoff frequency in Hz
highcut = 7000  # High cutoff frequency in Hz
order = 6  # Filter order

def butter_bandpass_filter(data, lowcut, highcut, fs, order=6):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    y = lfilter(b, a, data)
    return y

def windowing(audio, label):
    audio = path2+audio
    label = path2+label
    aud, sr = librosa.load(audio, sr=48000)
    aud = butter_bandpass_filter(aud, lowcut, highcut, sr, order)
    aud = librosa.resample(aud, orig_sr=sr, target_sr=16000)
    newsr = 16000
    data = pd.read_csv(label, sep="\s")
    F = data.iloc[:, 4:6]
    F = F.drop_duplicates()
    F = F.to_numpy()
    F = F*newsr
    window = np.zeros(len(aud))
    for j in range(len(F)):
        k = round(F[j, 0])
        v = round(F[j, 1])
        num = k
        while num <= v and num >= k:
            window[num] = window[num] + 1
            num += 1
    window = pd.DataFrame(window)
    return window

def raw(audio):
    audio = path2+audio
    aud, sr = librosa.load(audio, sr=48000)
    aud = butter_bandpass_filter(aud, lowcut, highcut, sr, order)
    aud = librosa.resample(aud, orig_sr=sr, target_sr=16000)
    window = pd.DataFrame(aud)
    return window

def windowing1(audio, label):
    audio = path2+audio
    label = path2+label
    aud, sr = librosa.load(audio, sr=48000)
    aud = butter_bandpass_filter(aud, lowcut, highcut, sr, order)
    aud = librosa.resample(aud, orig_sr=sr, target_sr=16000)
    newsr = 16000
    data = pd.read_csv(label, sep="\s")
    if data.iloc[0,3] == 1:
        print(1)
        F = data.iloc[:, 4:6]
        F = F.drop_duplicates()
        F = F.to_numpy()
        F = F*newsr
    else:
        print(2)
        F = data.iloc[:, 3:5]
        F = F.drop_duplicates()
        F = F.to_numpy()
        F = F*newsr
    window = np.zeros(len(aud))
    for j in range(len(F)):
        k = round(F[j, 0])
        v = round(F[j, 1])
        num = k
        if v>len(aud):
            v = len(aud)-1
        while num <= v and num >= k:
            #print(num)
            window[num] = window[num] + 1
            num += 1
    window = pd.DataFrame(window)
    return window

path = path
audio_files = [f for f in os.listdir(path) if f.endswith('.wav')]
label_files = [f for f in os.listdir(path) if f.endswith('.txt')]
con = []
for i in range(len(audio_files)):
    for j in label_files:
        if audio_files[i][0:str_num] == j[0:str_num]:
            con = con + [(audio_files[i], j)]
df_label = pd.DataFrame()
df_raw = pd.DataFrame()
for i, j in con:
    print(i,j)
    dfraw = raw(i)
    df_raw = pd.concat([df_raw, dfraw], axis=0)
    result = windowing1(i, j)
    df_label = pd.concat([df_label, result], axis=0)



def hamming(waveform):

    data_frame = waveform  
    data_array = data_frame.values.reshape(1, -1)
    data_array[np.isnan(data_array)] = 0
    data_array = np.transpose(data_array)
    waveform = torch.Tensor(data_array)
    
    # Step 3: Set the desired size and shift of the window
    window_size = 3200  # Replace this with your desired window size
    hop_length = 3200   # Replace this with your desired shift (hop length) size
    
    
    window = 1
    
    # Apply the sliding window to the audio data
    num_samples = waveform.size(0)
    
    num_windows = 1 + (num_samples - window_size) // hop_length
    
    windowed_waveform = torch.zeros(num_windows, window_size)
    for i in range(num_windows):
        
        start = i * hop_length
        end = start + window_size
        windowed_waveform[i] = waveform[start:end,0] * window
   
    k = windowed_waveform.numpy()
    return k
frame = hamming(df_raw)
df = pd.DataFrame(frame)
csv_filename = csvname1
df.to_csv(csv_filename, index=False) 

frame_label = hamming(df_label)
df_label = pd.DataFrame(frame_label)

def frame_sequence(sequence, frame_size, shift):
    framed_sequences = []
    num_frames = (len(sequence) - frame_size) // shift + 1

    for i in range(num_frames):
        start_idx = i * shift
        end_idx = start_idx + frame_size
        framed_sequences.append(sequence[start_idx:end_idx])
    for frame in framed_sequences:
        if sum(frame) >= 1600:
            for k in frame:
                k == 1
    return framed_sequences



label = []
dff = df_label.values
for l in dff:
    u = np.sum(l)
    if u >= 1600:
        label.append(1)
    else:
        label.append(0)
label = pd.DataFrame(label)
csv_filename2 = csvname2
label.to_csv(csv_filename2, index=False)

