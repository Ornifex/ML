import os
import wave
import subprocess
import librosa
import numpy as np
import pandas as pd
from scipy import stats

print(" Feature Extraction")

def columns():
    feature_sizes = dict(mfcc=20, spectral_centroid=1, spectral_contrast=7)
    moments = ('mean', 'std', 'skew', 'kurtosis', 'median', 'min', 'max')

    columns = []
    for name, size in feature_sizes.items():
        for moment in moments:
            it = ((name, moment, '{:02d}'.format(i+1)) for i in range(size))
            columns.extend(it)

    names = ('feature', 'statistics', 'number')
    columns = pd.MultiIndex.from_tuples(columns, names=names)

    return columns.sort_values()    

def compute_features(tid):

    features = pd.Series(index=columns(), dtype=np.float32, name=song)

    def feature_stats(name, values):
        features[name, 'mean'] = np.mean(values, axis=1)
        features[name, 'std'] = np.std(values, axis=1)
        features[name, 'skew'] = stats.skew(values, axis=1)
        features[name, 'kurtosis'] = stats.kurtosis(values, axis=1)
        features[name, 'median'] = np.median(values, axis=1)
        features[name, 'min'] = np.min(values, axis=1)
        features[name, 'max'] = np.max(values, axis=1)
    
    # read .wav and compute mfcc/spectral_contrast/spectral_centroid features
    x, sr = librosa.load(tid, sr=None, mono=True)
    stft = np.abs(librosa.stft(x, n_fft=2048, hop_length=512))
    mel = librosa.feature.melspectrogram(sr=sr, S=stft**2)

    mfcc = librosa.feature.mfcc(S=librosa.power_to_db(mel), n_mfcc=20)
    spectral_contrast = librosa.feature.spectral_contrast(S=stft, n_bands=6)
    spectral_centroid = librosa.feature.spectral_centroid(S=stft)
    feature_stats('mfcc', mfcc)
    feature_stats('spectral_contrast', spectral_contrast)
    feature_stats('spectral_centroid', spectral_centroid)

    return features

# get current working directory
cwd = os.getcwd()
# get the subdirectories of the cwd
dirs = next(os.walk(cwd))[1]

tracks = []
# go through all dirs and make a list of tracks
for dir in dirs:
    genre = os.path.join(cwd, dir)
    songs = os.listdir(genre)
    tracks = tracks + songs

tracks = pd.DataFrame(tracks)

features = pd.DataFrame(columns=columns(), dtype=np.float32)

for dir in dirs:
    genre = os.path.join(cwd, dir)
    songs = os.listdir(genre)
    print("     Computing " + dir)
    for song in songs:
        fp = os.path.join(genre, song) #.au file
        nfp = os.path.splitext(fp)[0] + ".au" #.wav file
        try:
            features = features.append(compute_features(nfp))
        except:
            print("Oops")


print(" Saving to .csv...")
features.to_csv('features.csv', float_format='%.{}e'.format(10))
