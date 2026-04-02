import streamlit as st
import requests
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

# 1. Configuration & Constants
API_URL = 'http://127.0.0.1:8000' # FastAPI backend URL

st.set_page_config(page_title="TESS Audio AI", layout="wide", initial_sidebar_state="collapsed")

# 2. Custom Styling (Gold & Black Theme)
custom_css = """
<style>
    .stApp {
        background-color: #0a0a0a;
        color: #f0f0f0;
    }
    
    h1, h2, h3 {
        color: #D4AF37 !important;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }

    .gold-card {
        background-color: #141414;
        border-left: 4px solid #D4AF37;
        padding: 20px;
        border-radius: 4px;
        margin-bottom: 15px;
        box-shadow: 0 4px 15px rgba(212, 175, 55, 0.05);
    }
    .gold-card-title {
        color: #D4AF37;
        font-size: 1.3rem;
        font-weight: 600;
        margin-bottom: 8px;
    }
    .gold-card-text {
        color: #cccccc;
        font-size: 1rem;
        line-height: 1.5;
    }

    .health-box {
        background-color: #141414;
        border: 1px solid #D4AF37;
        border-radius: 8px;
        padding: 25px 10px;
        text-align: center;
    }
    .health-value {
        font-size: 2.5rem;
        color: #D4AF37;
        font-weight: 800;
    }
    .health-label {
        font-size: 0.8rem;
        color: #ffffff;
        text-transform: uppercase;
    }

    div.stButton > button, 
    section[data-testid="stFileUploadDropzone"] button {
        background-color: #000000 !important;
        color: #D4AF37 !important;
        border: 1px solid #D4AF37 !important;
    }

    section[data-testid="stFileUploadDropzone"] {
        background-color: #0a0a0a !important;
        border: 1px dashed #D4AF37 !important;
    }
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

st.title("🎧 Speech Emotion Recognition Pipeline")
st.markdown("*A professional MLOps architecture analyzing the Toronto Emotional Speech Set.*")
st.markdown("---")

# 3. Create Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "🔮 Predict Emotion", 
    "📊 Data Visualizations", 
    "⚙️ Retrain Model", 
    "🏥 System Health"
])

# --- Tab 1: Prediction ---
with tab1:
    st.header("Predict a Single Audio File")
    st.write("Upload a .wav file to hear it and predict the emotion.")
    
    uploaded_file = st.file_uploader("Upload an audio file", type=["wav"], key="predict_upload")
    
    if uploaded_file is not None:
        st.audio(uploaded_file, format="audio/wav")
        
        if st.button("Predict Emotion", use_container_width=True):
            with st.spinner("Analyzing audio features..."):
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "audio/wav")}
                try:
                    response = requests.post(f"{API_URL}/predict", files=files)
                    if response.status_code == 200:
                        result = response.json()
                        st.success(f"**Predicted Emotion:** {result['predicted_emotion']}")
                        st.info(f"**Confidence Score:** {result['confidence']}")
                    else:
                        st.error("Error connecting to the API. Check backend logs.")
                except Exception as e:
                    st.error(f"Failed to connect to backend server. Error: {e}")

# --- Tab 2: Data Visualizations (The "Sound DNA" Tab) ---
with tab2:
    st.header("Feature Interpretations")
    st.write("Explore the mathematical features extracted from raw audio waves.")
    
    st.markdown("---")
    st.subheader("🎵 Dynamic Audio Analysis")
    st.markdown("Upload any audio file to see its 'Sound DNA' (Waveform, Spectrogram, and MFCCs).")

    eda_file = st.file_uploader("Upload a .wav file for visual analysis", type=['wav'], key="eda_visuals")

    if eda_file is not None:
        y, sr = librosa.load(eda_file, duration=2.5)

        # Row 1: Waveform & Spectrogram
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("1. Waveform")
            fig_wave, ax_wave = plt.subplots(figsize=(10, 5))
            librosa.display.waveshow(y, sr=sr, ax=ax_wave, color='#D4AF37')
            ax_wave.set_title("Amplitude vs Time", color="white")
            fig_wave.patch.set_facecolor('#0a0a0a')
            ax_wave.set_facecolor('#0a0a0a')
            st.pyplot(fig_wave)
            st.caption("Raw physical vibration and volume levels.")

        with col2:
            st.subheader("2. Mel-Spectrogram")
            fig_spec, ax_spec = plt.subplots(figsize=(10, 5))
            S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
            S_db = librosa.power_to_db(S, ref=np.max)
            img = librosa.display.specshow(S_db, x_axis='time', y_axis='mel', sr=sr, ax=ax_spec)
            plt.colorbar(img, ax=ax_spec, format='%+2.0f dB')
            ax_spec.set_title("Frequency Intensity", color="white")
            fig_spec.patch.set_facecolor('#0a0a0a')
            st.pyplot(fig_spec)
            st.caption("The 'colors' and energy of the voice frequencies.")

        st.divider()

        # Row 2: MFCCs (The specialized feature)
        st.subheader("3. 🧬 Vocal Tract Fingerprint (MFCCs)")
        st.write("These coefficients represent the biological shape of the vocal tract during speech.")
        
        fig_mfcc, ax_mfcc = plt.subplots(figsize=(12, 4))
        # Extract 40 MFCCs to match our training pipeline
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        img_mfcc = librosa.display.specshow(mfccs, x_axis='time', ax=ax_mfcc)
        plt.colorbar(img_mfcc, ax=ax_mfcc)
        ax_mfcc.set_title("MFCC Coefficients", color="white")
        fig_mfcc.patch.set_facecolor('#0a0a0a')
        st.pyplot(fig_mfcc)
        st.caption("This unique pattern is what the AI model actually 'reads' to classify emotion.")

        st.success("✅ Features successfully extracted and visualized!")
    
    else:
        st.info("Please upload a .wav file in the box above to generate visuals.")
    
    # Feature Explanations (Gold Cards)
    st.markdown("""
        <div class="gold-card">
            <div class="gold-card-title">Understanding the Waveform</div>
            <div class="gold-card-text">
                Angry voices show jagged, high-energy peaks, while Sad/Neutral voices are often compressed and lower in volume.
            </div>
        </div>
        <div class="gold-card">
            <div class="gold-card-title">Understanding the Spectrogram</div>
            <div class="gold-card-text">
                Higher frequencies (brighter colors) indicate excitement or distress, while lower clusters suggest calm or sadness.
            </div>
        </div>
        <div class="gold-card">
            <div class="gold-card-title">Understanding MFCCs</div>
            <div class="gold-card-text">
                MFCCs isolate the physical 'tightness' of the throat and mouth, allowing the AI to detect emotion regardless of background noise.
            </div>
        </div>
    """, unsafe_allow_html=True)

# --- Tab 3: Retraining ---
with tab3:
    st.header("Bulk Upload & Model Retraining")
    st.write("Upload a `.zip` file organized by emotion folders to retrain the model.")
    
    zip_file = st.file_uploader("Upload Bulk Data (.zip)", type=["zip"])
    
    if zip_file is not None:
        if st.button("Trigger Retraining Pipeline", use_container_width=True):
            with st.spinner("Uploading and starting background training..."):
                files = {"file": (zip_file.name, zip_file.getvalue(), "application/zip")}
                try:
                    response = requests.post(f"{API_URL}/retrain", files=files)
                    if response.status_code == 200:
                        st.success(response.json()["message"])
                        st.toast('Retraining triggered!', icon='🔄')
                    else:
                        st.error("Failed to trigger retraining.")
                except Exception as e:
                    st.error(f"Connection failed: {e}")

# --- Tab 4: System Health ---
with tab4:
    st.header("Model Server Health")
    st.write("Live telemetry from the FastAPI backend.")
    
    if st.button("Ping Server for Status", use_container_width=True):
        try:
            response = requests.get(f"{API_URL}/health")
            if response.status_code == 200:
                data = response.json()
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.markdown(f'<div class="health-box"><div class="health-value">{data["status"].split()[0]}</div><div class="health-label">Status</div></div>', unsafe_allow_html=True)
                with c2:
                    st.markdown(f'<div class="health-box"><div class="health-value">{data["uptime_minutes"]}</div><div class="health-label">Uptime (Mins)</div></div>', unsafe_allow_html=True)
                with c3:
                    st.markdown(f'<div class="health-box"><div class="health-value">{data["uptime_seconds"]}</div><div class="health-label">Uptime (Secs)</div></div>', unsafe_allow_html=True)
            else:
                st.error("API error.")
        except Exception as e:
            st.error(f"Cannot connect to API: {e}")