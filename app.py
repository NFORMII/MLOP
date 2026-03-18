from fastapi import FastAPI, UploadFile, File, BackgroundTasks
from fastapi.responses import JSONResponse
import time
import shutil
import os

from src.preprocessing import preprocess_image
from src.prediction import predict

app = FastAPI(title="Cassava Disease ML Pipeline API")
#calculate latency
START_TIME = time.time()

@app.get("/")
def read_root():
    return {"message": "Welcome to the Cassava ML Pipeline API. Go to /docs to test it out!"}

@app.get("/health")
def health_check():
    """Returns the API status and uptime (Assignment Requirement)"""
    uptime_seconds = time.time() - START_TIME
    uptime_hours = round(uptime_seconds / 3600, 4)
    return {
        "status": "online", 
        "uptime_seconds": round(uptime_seconds, 2),
        "uptime_hours": uptime_hours
    }

@app.post("/predict")
async def make_prediction(file: UploadFile = File(...)):
    """Accepts an image file and returns the model's prediction"""
    try:
        image_bytes = await file.read()
        
        #preprocess the image
        processed_tensor = preprocess_image(image_bytes)
        
        #pass to the model for prediction
        predicted_class, confidence = predict(processed_tensor)
        
        return {
            "filename": file.filename,
            "prediction": predicted_class,
            "confidence": round(confidence, 4)
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"message": f"Prediction failed: {str(e)}"})
    

# this function simulates a background retraining job. in a full production environment, this would call logic in src/model.py
    
def run_retraining_job():
    print(">>> BACKGROUND TASK: Retraining triggered! Processing new data...")
    time.sleep(10)
    print(">>> BACKGROUND TASK: Retraining complete. New model saved.")

@app.post("/retrain")
async def trigger_retrain(background_tasks: BackgroundTasks, files: list[UploadFile] = File(...)):
    #Accepts bulk images, saves them, and triggers retraining
    
    upload_dir = "data/train/uploaded_retrain_data"
    os.makedirs(upload_dir, exist_ok=True)
    
    saved_files = 0
    #save all uploaded files to the server
    for file in files:
        file_location = os.path.join(upload_dir, file.filename)
        with open(file_location, "wb+") as file_object:
            shutil.copyfileobj(file.file, file_object)
        saved_files += 1
    
    #trigger the training function in the background so the user doesn't have to wait
    background_tasks.add_task(run_retraining_job)
    
    return {
        "message": "Retraining successfully triggered.",
        "files_uploaded": saved_files,
        "status": "Training in background..."
    }