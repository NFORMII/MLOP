from fastapi import FastAPI, File, UploadFile, BackgroundTasks
import time
import shutil
import os
import numpy as np
import uvicorn


from src.prediction import make_prediction
from src.model import retrain_pipeline 

#Initializing the Web Server
app = FastAPI(title="TESS MLOps Emotion Recognition API")
START_TIME = time.time() #server uptime for the health check

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
    Requirement: Model up-time and health status.
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
    Requirement: Predict one datapoint from sound.
    Handles the ingestion of .wav files and returns JSON predictions.
    """
   
    temp_file_path = f"temp_{file.filename}"
    
  
    with open(temp_file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    try:
        #Triggering the prediction logic from src/prediction.py
        result = make_prediction(temp_file_path)
        
        # formatting of the confidence score
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
    
    # Saving the bulk dataset temporarily
    with open(temp_zip_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # Hand off the heavy lifting to the background worker this will help prevent the client from timing out during the training process
    background_tasks.add_task(retrain_pipeline, temp_zip_path)
    
    return {
        "message": "Retraining successfully triggered! The AI is updating in the background.",
        "job_status": "Processing",
        "file_received": file.filename
    }

if __name__ == "__main__":
    
    # using 0.0.0.0 to ensure the container/server is reachable externally
    uvicorn.run(app, host="0.0.0.0", port=8000)