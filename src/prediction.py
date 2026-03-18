import tensorflow as tf
import numpy as np
import os
from keras.models import load_model

# define the absolute path to the model so the API can always find it
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, 'models', '_cassava_model.h5')

# loading the model once into memory when the app starts
try:
    model = load_model(MODEL_PATH)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

CLASS_NAMES = [
    "Cassava Bacterial Blight", 
    "Cassava Brown Streak", 
    "Cassava Green Mottle", 
    "Cassava Mosaic", 
    "Healthy"
]
    
    #this takes a preprocessed image tensor, passes it to the model,and then returns the predicted class and confidence.

def predict(preprocessed_image):

    if model is None:
        return "Model Error", 0.0

    # getting the raw probabilities from the model
    predictions = model.predict(preprocessed_image)
    
    #findding the index of the highest probability
    class_idx = np.argmax(predictions[0])
    
    # getting the actual confidence score (as a standard python float)
    confidence = float(np.max(predictions[0]))
    
    # lastly, returning the string name and the score
    return CLASS_NAMES[class_idx], confidence