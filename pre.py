import os
import shutil
import pyaudio
import wave
import sys
import librosa
import re
import matplotlib.pyplot as plt
import librosa.display
import numpy as np

def get_emotion(fname):
    x = re.findall("\d\d", fname)

    d = {
        "01" : "1",
        "02" : "2",
        "03" : "3",
        "04" : "4",
        "05" : "5",
        "06" : "6",
        "07" : "7",
        "08" : "8"
    }
    return d[x[2]]


# create sound dataset
f= open("sound.txt","w+")
for fname in os.listdir('../sound_data/'):
    path = os.path.abspath('../sound_data/'+fname)
    f.write(path + ' ' + get_emotion(fname)+'\n')
f.close() 

# create image dataset
for fname in os.listdir('../sound_data/'):
    y, sr = librosa.load('../sound_data/'+ fname)
    S = librosa.feature.melspectrogram(y=y, sr=sr)
    fig = plt.figure(figsize=(4, 4))
    librosa.display.specshow(librosa.power_to_db(S, ref=np.max), fmax=8000)
    plt.axis('off')
    plt.savefig('../image_data/'+os.path.splitext(fname)[0]+'.png', bbox_inches='tight', pad_inches = 0)
    plt.close()

f=open("image.txt", "w+")
for fname in os.listdir('../image_data/'):
    path = os.path.abspath('../image_data/'+fname)
    f.write(path + ' ' + get_emotion(fname)+'\n')
f.close() 

# create number dataset
