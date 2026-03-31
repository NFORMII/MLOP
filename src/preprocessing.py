import librosa
import numpy as np

def extract_features(file_path):
    """
    Takes an audio file path, reads the waveform, and extracts 
    MFCCs, Chroma, and Mel Spectrogram features into a 1D numpy array.
    """
    signal, sr = librosa.load(file_path, duration=2.5)
    mfccs = np.mean(librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=13).T, axis=0)
    stft = np.abs(librosa.stft(signal))
    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sr).T, axis=0)
    mel = np.mean(librosa.feature.melspectrogram(y=signal, sr=sr).T, axis=0)
    
    return np.hstack([mfccs, chroma, mel])