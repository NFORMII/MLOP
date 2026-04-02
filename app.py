from fastapi import FastAPI, File, UploadFile, BackgroundTasks
import time
import shutil
import os



from src.prediction import make_prediction
from src.model import retrain_pipeline 

# Initializing the Web Server
app = FastAPI(title="TESS MLOps Pipeline API")
START_TIME = time.time() #helps trackacks server uptime

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
    
    #temporarily save uploaded file
    with open(temp_file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    try:
    
        result = make_prediction(temp_file_path)
        
        #ensuring that result is presented as dictionary
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
        #always cleanup
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)



@app.post("/retrain")
async def trigger_retraining(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """Bulk data upload and trigger retraining"""
    print(f"========== RETRAINING REQUEST RECEIVED ==========")
    print(f"File uploaded: {file.filename}")

    if not file.filename.endswith('.zip'):
        print(" Error: File is not a .zip archive.")
        return {"error": "Please upload a .zip file containing folders of audio files."}
        
    temp_zip_path = f"temp_bulk_{file.filename}"
    
    #temporarily saving the .zip file
    print(f"Step 1: Saving uploaded zip file to {temp_zip_path}...")
    with open(temp_zip_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    print(" Success! Zip file saved temporarily.")
    
    #sending the heavy lifting to model.py through the background tasks
    print(f"Step 2: Handing off {temp_zip_path} to the background retraining task...")
    background_tasks.add_task(retrain_pipeline, temp_zip_path)
    
    print(f"========== API RESPONSE SENT (TRAINING CONTINUES IN BACKGROUND) ==========")
    
    return {
        "message": "Retraining successfully triggered! The AI is updating in the background.",
        "status": "Processing"
    }

# @app.post("/retrain")
# async def trigger_retraining(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
#     """Rubric Requirement: Bulk data upload and trigger retraining"""
#     if not file.filename.endswith('.zip'):
#         return {"error": "Please upload a .zip file containing folders of audio files."}
        
#     temp_zip_path = f"temp_bulk_{file.filename}"
    
#     
#     with open(temp_zip_path, "wb") as buffer:
#         shutil.copyfileobj(file.file, buffer)
    
#     background_tasks.add_task(retrain_pipeline, temp_zip_path)
    
#     return {
#         "message": "Retraining successfully triggered! The AI is updating in the background.",
#         "status": "Processing"
#     }