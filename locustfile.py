import os
from locust import HttpUser, task, between

#real audio file from your local folder to use for the stress test
sample_dir = "data/train/Angry"

try:
    sample_file = os.path.join(sample_dir, os.listdir(sample_dir)[0])
except Exception:
    sample_file = None

class MLOpsLoadTest(HttpUser):
    # fake users will wait between 1 and 2 seconds between clicks
    wait_time = between(1, 2) 

    @task(1)
    def test_health_endpoint(self):
        """Simulates users checking the system health."""
        self.client.get("/health")

    @task(3)
    def test_prediction_endpoint(self):
        
        file_path = "temp_test_audio.wav" 
        
        with open(file_path, "rb") as file:
            self.client.post("/predict", files={"file": ("test.wav", file, "audio/wav")})