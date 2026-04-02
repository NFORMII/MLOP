from fastapi import FastAPI, File, UploadFile, BackgroundTasks
import time
import shutil
import os
import numpy as np
import uvicorn

# Internal Imports
from src.prediction import make_prediction
from src.model import retrain_pipeline 

# 1. Initialize the Web Server
app = FastAPI(title="TESS MLOps Emotion Recognition API")
START_TIME = time.time() # Tracks server uptime for the health check

@app.get("/")
def read_root():
    return {
        "message": "Welcome to the Emotion Recognition API!",
        "documentation": "/docs",
        "status": "Online"
    }

@app.get("/health")
def health_check():
    """
    Rubric Requirement: Model up-time and health status.
    Provides monitoring data for DevOps/MLOps oversight.
    """
    uptime_seconds = int(time.time() - START_TIME)
    return {
        "status": "Healthy 🟢",
        "uptime_seconds": uptime_seconds,
        "uptime_minutes": round(uptime_seconds / 60, 2),
        "api_version": "1.0.0"
    }

@app.post("/predict")
async def predict_emotion(file: UploadFile = File(...)):
    """
    Rubric Requirement: Predict one datapoint from sound.
    Handles the ingestion of .wav files and returns JSON predictions.
    """
    # 1. Create a unique temporary path for the uploaded file
    temp_file_path = f"temp_{file.filename}"
    
    # 2. Save the uploaded binary stream to a physical file for Librosa to read
    with open(temp_file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    try:
        # 3. Trigger the prediction logic from src/prediction.py
        result = make_prediction(temp_file_path)
        
        # 4. Safe formatting of the confidence score
        raw_confidence = result.get("confidence", 0)
        formatted_confidence = f"{round(float(raw_confidence) * 100, 2)}%"
        
        return {
            "filename": file.filename,
            "predicted_emotion": result.get("predicted_emotion"),
            "confidence": formatted_confidence,
            "status": "success"
        }
        
    except Exception as e:
        return {
            "filename": file.filename,
            "status": "error",
            "message": str(e)
        }
    finally:
        # 5. MLOps Best Practice: Always cleanup temporary files to prevent disk bloat
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

@app.post("/retrain")
async def trigger_retraining(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """
    Rubric Requirement: Bulk data upload and trigger retraining.
    Uses BackgroundTasks to keep the API responsive while the model trains.
    """
    print(f"--- RETRAINING REQUEST RECEIVED ---")
    
    if not file.filename.endswith('.zip'):
        return {"error": "Invalid format. Please upload a .zip file containing labeled folders."}
        
    temp_zip_path = f"temp_bulk_{file.filename}"
    
    # Save the bulk dataset temporarily
    with open(temp_zip_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # Hand off the heavy lifting to the background worker
    # This prevents the client from timing out during the training process
    background_tasks.add_task(retrain_pipeline, temp_zip_path)
    
    return {
        "message": "Retraining successfully triggered! The AI is updating in the background.",
        "job_status": "Processing",
        "file_received": file.filename
    }

if __name__ == "__main__":
    
    # Use 0.0.0.0 to ensure the container/server is reachable externally
    uvicorn.run(app, host="0.0.0.0", port=8000)