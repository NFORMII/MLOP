import os
import shutil
import zipfile
import numpy as np
from tensorflow.keras.models import load_model
import joblib
from src.preprocessing import extract_features

# Load the artifacts once when the module is imported
MODEL_PATH = "models/audio_model.h5"
SCALER_PATH = "models/scaler.pkl"
ENCODER_PATH = "models/encoder.pkl"

model = load_model(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
encoder = joblib.load(ENCODER_PATH)

def retrain_pipeline(zip_path: str):
    """Handles the extraction of bulk data and fine-tunes the existing model."""
    print("🔄 Starting core retraining pipeline...")
    extract_dir = "data/temp_retrain"
    os.makedirs(extract_dir, exist_ok=True)
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
        print(">>> BACKGROUND TASK STARTED: Unzipping files...")
        
    new_data, new_labels = [], []
    for root, dirs, files in os.walk(extract_dir):
        for file in files:
            if file.endswith(".wav"):
                emotion = os.path.basename(root) 
                file_path = os.path.join(root, file)
                new_data.append(extract_features(file_path))
                new_labels.append(emotion)

                print("Feature extraction complete. Loading model and fitting new data...")
    
    if len(new_data) > 0:
        X_new = np.array(new_data)
        y_new = encoder.transform(new_labels) 
        X_new_scaled = scaler.transform(X_new) 

        print(">>> Training complete. Saving model.h5 to disk...")
        
        # Fine-tune the model
        model.fit(X_new_scaled, y_new, epochs=10, batch_size=32, verbose=1)
        model.save(MODEL_PATH)
        print("Core model successfully updated and saved!")
        print("BACKGROUND TASK FULLY COMPLETE!")

    # Cleanup
    shutil.rmtree(extract_dir)
    os.remove(zip_path)