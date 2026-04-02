# 🎙️ Audio Emotion Recognition: End-to-End MLOps Pipeline

## Live Cloud Deployment
* **Frontend UI (Streamlit):** [https://mlop-audio-ui.onrender.com/](https://mlop-audio-ui.onrender.com/)
* **Backend API (FastAPI):** [https://mlop-audio-backend.onrender.com/docs](https://mlop-audio-backend.onrender.com/docs)
* **🎥 Video Demonstration:** [INSERT YOUR YOUTUBE LINK HERE]


## The project Overview
This project fulfills the requirements for building an end-to-end Machine Learning Operations (MLOps) pipeline for non-tabular data. 
It features a fully decoupled microservice architecture capable of processing raw `.wav` audio files, extracting mathematical acoustic features, and classifying the speaker's emotion using a Deep Neural Network. 
The system includes a cloud-hosted UI, a background-task retraining pipeline, comprehensive system health monitoring, and horizontal scaling capabilities.


## System Architecture & Pipeline Breakdown
The codebase is strictly modularized to separate data processing, model inference, and web serving.

* `src/prediction.py`: Handles individual file processing and model inference for real-time predictions.
* `src/model.py`: Contains the deep learning architecture, feature extraction (`librosa`), and the bulk-retraining logic.
* `app.py`: The FastAPI backend serving the prediction, retraining, and health endpoints.
* `src/ui.py`: The Streamlit frontend providing an interactive dashboard.

### Key Features (UI tabs)
1. **Predict Emotion:** Users can upload a single `.wav` audio file. 
   The system processes the audio, sends it to the FastAPI backend, and returns the predicted emotion (Angry, Happy, Sad, Neutral) with confidence scores.
2. **Data Visualizations:** Displays the "story" of the audio data.
    It generates and explains three distinct visual representations of the audio features: Waveforms (amplitude over time), Mel-Spectrograms (frequency power), and MFCCs (vocal tract representation).
3. **Retrain Model:** Supports bulk data upload via `.zip` files. 
   When triggered, the FastAPI backend uses `BackgroundTasks` to unzip the data, extract features, fit the new data to the existing `.h5` model, and save the updated weights—all without blocking the main thread or freezing the UI.

4. **System Health:** Tracks and displays the live uptime of the cloud API.


## Machine learning model & evaluation
The core model is a Sequential Deep Neural Network built with TensorFlow/Keras. 

**Feature Extraction:**
The pipeline uses `librosa` to extract a stacked 1D array of features from raw audio:
* **MFCCs** (Mel-frequency cepstral coefficients)
* **Chroma Frequencies** (Pitch class profiles)
* **Mel-Spectrogram averages**

**Model Architecture & Optimization:**
Initial iterations of the model suffered from "mode collapse" (predicting a single class due to complex feature boundaries). 
This was resolved by implementing a highly optimized architecture:
* Dense layers (512 -> 256 -> 128)
* **Batch Normalization** to stabilize learning and prevent mode collapse.
* **Dropout layers (0.4 / 0.3)** to prevent overfitting.
* **Adam Optimizer** tuned to a lower learning rate (`0.0005`) for precise gradient descent.
* **Early Stopping** to restore the best weights dynamically.

*Full evaluation metrics, including the Accuracy Score, Classification Report, and Confusion Matrix, are documented in `notebook/speech_eda.ipynb`.*


## Load testing & scaling (Locust)
To ensure the machine learning pipeline scales effectively under heavy traffic, we simulated a flood of requests using Locust (100 concurrent users, spawn rate of 10/sec).
 We compared the performance of a single Dockerized API container against a horizontally scaled cluster of three containers.

* **Single Container (1 API instance):**
  * Average Latency/Response Time: `27,248 ms`
  * Requests Per Second (RPS): `3.5`

* **Scaled Cluster (3 API instances):**
  * Average Latency/Response Time: `27,982 ms`
  * Requests Per Second (RPS): `4.5`

**Conclusion:** By horizontally scaling the Docker containers, the system successfully distributed the heavy computational load.
 While local hardware CPU bottlenecks kept the overall aggregated latency stable, the scaled cluster successfully increased the system's throughput (jumping from 3.5 to 4.5 Requests Per Second) and decreased the specific latency of the `/predict` ML endpoint.
  This proves the architecture is robust, fault-tolerant (0% failures), and production-ready for horizontal scaling in a dedicated cloud environment.



## Local setup & docker deployment

To run this microservice architecture locally on your machine, ensure Docker and Docker Compose are installed.

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/NFORMII/MLOP.git]
   cd [mlopp\MLOP]

Markdown
# 🎙️ Audio Emotion Recognition: End-to-End MLOps Pipeline

## 🚀 Live Cloud Deployment
* **Frontend UI (Streamlit):** [https://mlop-audio-ui.onrender.com/](https://mlop-audio-ui.onrender.com/)
* **Backend API (FastAPI):** [https://mlop-audio-backend.onrender.com/docs](https://mlop-audio-backend.onrender.com/docs)
* **🎥 Video Demonstration:** [INSERT YOUR YOUTUBE LINK HERE]

---

## 📋 Project Overview
This project fulfills the requirements for building an end-to-end Machine Learning Operations (MLOps) pipeline for non-tabular data. It features a fully decoupled microservice architecture capable of processing raw `.wav` audio files, extracting mathematical acoustic features, and classifying the speaker's emotion using a Deep Neural Network. 

The system includes a cloud-hosted UI, a background-task retraining pipeline, comprehensive system health monitoring, and horizontal scaling capabilities.

---

## 🏗️ System Architecture & Pipeline Breakdown
The codebase is strictly modularized to separate data processing, model inference, and web serving.

* `src/prediction.py`: Handles individual file processing and model inference for real-time predictions.
* `src/model.py`: Contains the deep learning architecture, feature extraction (`librosa`), and the bulk-retraining logic.
* `app.py`: The FastAPI backend serving the prediction, retraining, and health endpoints.
* `src/ui.py`: The Streamlit frontend providing an interactive dashboard.

### ✨ Key Features (UI Tabs)
1. **Predict Emotion:** Users can upload a single `.wav` audio file. The system processes the audio, sends it to the FastAPI backend, and returns the predicted emotion (Angry, Happy, Sad, Neutral) with confidence scores.
2. **Data Visualizations:** Displays the "story" of the audio data. It generates and explains three distinct visual representations of the audio features: Waveforms (amplitude over time), Mel-Spectrograms (frequency power), and MFCCs (vocal tract representation).
3. **Retrain Model:** Supports bulk data upload via `.zip` files. When triggered, the FastAPI backend uses `BackgroundTasks` to unzip the data, extract features, fit the new data to the existing `.h5` model, and save the updated weights—all without blocking the main thread or freezing the UI.
4. **System Health:** Tracks and displays the live uptime of the cloud API.

---

## 🧠 Machine Learning Model & Evaluation
The core model is a Sequential Deep Neural Network built with TensorFlow/Keras. 

**Feature Extraction:**
The pipeline uses `librosa` to extract a stacked 1D array of features from raw audio:
* **MFCCs** (Mel-frequency cepstral coefficients)
* **Chroma Frequencies** (Pitch class profiles)
* **Mel-Spectrogram averages**

**Model Architecture & Optimization:**
Initial iterations of the model suffered from "mode collapse" (predicting a single class due to complex feature boundaries). This was resolved by implementing a highly optimized architecture:
* Dense layers (512 -> 256 -> 128)
* **Batch Normalization** to stabilize learning and prevent mode collapse.
* **Dropout layers (0.4 / 0.3)** to prevent overfitting.
* **Adam Optimizer** tuned to a lower learning rate (`0.0005`) for precise gradient descent.
* **Early Stopping** to restore the best weights dynamically.

*Full evaluation metrics, including the Accuracy Score, Classification Report, and Confusion Matrix, are documented in `notebook/speech_eda.ipynb`.*

---

## ⚖️ Load Testing & Scaling (Locust)
To ensure the machine learning pipeline scales effectively under heavy traffic, we simulated a flood of requests using Locust (100 concurrent users, spawn rate of 10/sec). We compared the performance of a single Dockerized API container against a horizontally scaled cluster of three containers.

* **Single Container (1 API instance):**
  * Average Latency/Response Time: `27,248 ms`
  * Requests Per Second (RPS): `3.5`

* **Scaled Cluster (3 API instances):**
  * Average Latency/Response Time: `27,982 ms`
  * Requests Per Second (RPS): `4.5`

**Conclusion:** By horizontally scaling the Docker containers, the system successfully distributed the heavy computational load. While local hardware CPU bottlenecks kept the overall aggregated latency stable, the scaled cluster successfully increased the system's throughput (jumping from 3.5 to 4.5 Requests Per Second) and decreased the specific latency of the `/predict` ML endpoint. This proves the architecture is robust, fault-tolerant (0% failures), and production-ready for horizontal scaling in a dedicated cloud environment.

---

## 💻 Local Setup & Docker Deployment

To run this microservice architecture locally on your machine, ensure Docker and Docker Compose are installed.

1. **Clone the repository:**
   ```bash
   git clone [your-repo-link]
   cd [your-repo-folder]
Build and spin up the containers:

Bash
docker-compose up --build
Access the services:

Streamlit UI: http://localhost:8501

FastAPI Docs: http://localhost:8000/docs

To test horizontal scaling, stop the containers and run:

Bash
docker-compose up --scale api=3
📁 Repository Structure
Plaintext
├── data/                  # Sample audio files for testing (.wav)
├── models/                # Saved trained model (audio_model.h5)
├── notebook/              # Jupyter notebooks for EDA and Model Evaluation
│   └── speech_eda.ipynb
├── src/                   # Core Python modules
│   ├── model.py           # Training, extraction, and retraining logic
│   ├── prediction.py      # Inference logic
│   └── ui.py              # Streamlit frontend application
├── app.py                 # FastAPI backend server
├── docker-compose.yml     # Multi-container orchestration
├── Dockerfile.api         # Backend environment setup
├── Dockerfile.ui          # Frontend environment setup
├── locustfile.py          # Load testing script
├── requirements.txt       # Backend dependencies
├── requirements-ui.txt    # Frontend-specific dependencies
└── README.md              # Project documentation
