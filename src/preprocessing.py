import numpy as np
from PIL import Image
import io

def preprocess_image(image_bytes):
    """
    Takes raw image bytes from an API upload, resizes to 224x224,
    and normalizes the pixels for MobileNetV2.
    """
    #open the image from bytes
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    
    #resize to match MobileNetV2 input shape
    img = img.resize((224, 224))
    
    #convert to numpy array and normalize (0 to 1)
    img_array = np.array(img) / 255.0
    
    #add a batch dimension
    img_batch = np.expand_dims(img_array, axis=0)
    
    return img_batch