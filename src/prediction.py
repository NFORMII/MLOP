import numpy as np
import joblib
from tensorflow.keras.models import load_model
from src.preprocessing import extract_features

# Load your sequence-trained models
model = load_model("models/audio_model.h5")
scaler = joblib.load("models/scaler.pkl")
encoder = joblib.load("models/encoder.pkl")

def make_prediction(file_path):
    features = extract_features(file_path) # Get (100, 40)
    
    # Flatten to scale, then reshape back to (1, 100, 40) for the model
    features_flat = features.reshape(-1, 40)
    features_scaled = scaler.transform(features_flat)
    features_ready = features_scaled.reshape(1, 100, 40)
    
    prediction_array = model.predict(features_ready)
    predicted_index = np.argmax(prediction_array, axis=1)
    predicted_emotion = encoder.inverse_transform(predicted_index)[0]
    
    return {
        "predicted_emotion": predicted_emotion,
        "confidence": float(np.max(prediction_array))
    }