import numpy as np
from src.model import model, scaler, encoder
from src.preprocessing import extract_features

def make_prediction(file_path):
    """
    End-to-end prediction: Extracts features, scales them, 
    asks the model for a prediction, and decodes the result.
    """
    # 1. Extract features from the audio file
    features = extract_features(file_path)
    
    # 2. Scale the features
    features_scaled = scaler.transform(features.reshape(1, -1))
    
    # 3. Get the prediction array from the model
    prediction_array = model.predict(features_scaled)
    
    # 4. Find the index of the highest probability
    predicted_index = np.argmax(prediction_array, axis=1)
    
    # 5. Decode the index back to the emotion string
    predicted_emotion = encoder.inverse_transform(predicted_index)[0]
    
    # 6. Get the confidence score
    confidence = float(np.max(prediction_array))
    
    # RETURN AS A DICTIONARY 
    return {
        "predicted_emotion": predicted_emotion,
        "confidence": confidence
    }