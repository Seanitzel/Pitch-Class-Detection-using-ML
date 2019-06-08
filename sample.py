import librosa
import numpy as np

def getMagnitude(filename):
    y, sr = librosa.load(filename)
    D = np.abs(librosa.stft(y))
    return D
