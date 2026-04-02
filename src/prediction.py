import numpy as np
from src.model import model, scaler, encoder
from src.preprocessing import extract_features

def make_prediction(file_path):
    """
    End-to-end prediction: extracts features, scales them, 
    asks the model for a prediction, and then decodes the result.
    """
  
    features = extract_features(file_path)
    features_scaled = scaler.transform(features.reshape(1, -1))
    prediction_array = model.predict(features_scaled)
    predicted_index = np.argmax(prediction_array, axis=1)
    predicted_emotion = encoder.inverse_transform(predicted_index)[0]
    
    confidence = float(np.max(prediction_array))
    
    
    return {
        "predicted_emotion": predicted_emotion,
        "confidence": confidence
    }