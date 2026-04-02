import librosa
import numpy as np

def extract_features(file_path, max_pad_len=100):
    signal, sr = librosa.load(file_path, duration=2.5)
    mfccs = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=40)
    
    pad_width = max_pad_len - mfccs.shape[1]
    if pad_width > 0:
        mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
    else:
        mfccs = mfccs[:, :max_pad_len]
        
    return mfccs.T 