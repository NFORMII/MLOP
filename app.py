from fastapi import FastAPI, File, UploadFile, BackgroundTasks
import time
import shutil
import os


# Import the logic from your separated modules
from src.prediction import make_prediction
from src.model import retrain_pipeline 

# Initialize the Web Server
app = FastAPI(title="TESS MLOps Pipeline API")
START_TIME = time.time() # Tracks server uptime

@app.get("/")
def read_root():
    return {"message": "Welcome to the Emotion Recognition API! Go to /docs to test it."}

@app.get("/health")
def health_check():
    """Rubric Requirement: Model up-time and health status"""
    uptime_seconds = int(time.time() - START_TIME)
    return {
        "status": "Healthy 🟢",
        "uptime_seconds": uptime_seconds,
        "uptime_minutes": round(uptime_seconds / 60, 2)
    }

@app.post("/predict")
async def predict_emotion(file: UploadFile = File(...)):
    """Rubric Requirement: Predict one datapoint from sound"""
    temp_file_path = f"temp_{file.filename}"
    
    # 1. Save the uploaded file temporarily
    with open(temp_file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    try:
        # 2. Delegate to prediction.py (which should handle preprocessing inside it)
        result = make_prediction(temp_file_path)
        
        # Ensure result is a dictionary
        if isinstance(result, dict):
            return {
                "filename": file.filename,
                "predicted_emotion": result.get("predicted_emotion"),
                "confidence": str(round(result.get("confidence") * 100, 2)) + '%'
            }
        else:
            return {
                "filename": file.filename,
                "error": "Invalid result format from prediction model"
            }
    finally:
        # 3. Clean up the temp file
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

@app.post("/retrain")
async def trigger_retraining(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """Rubric Requirement: Bulk data upload and trigger retraining"""
    if not file.filename.endswith('.zip'):
        return {"error": "Please upload a .zip file containing folders of audio files."}
        
    temp_zip_path = f"temp_bulk_{file.filename}"
    
    # 1. Save the uploaded zip temporarily
    with open(temp_zip_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # 2. Send the heavy lifting to model.py via background tasks
    # We pass the temp_zip_path to retrain_pipeline, which will unzip and train
    background_tasks.add_task(retrain_pipeline, temp_zip_path)
    
    return {
        "message": "Retraining successfully triggered! The AI is updating in the background.",
        "status": "Processing"
    }