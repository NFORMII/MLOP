import streamlit as st
import requests
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import os

#fastAPI backend URL
API_URL = 'http://127.0.0.1:8000' 

#https://ml-audio-backend.onrender.com

st.set_page_config(page_title="TESS Audio AI", layout="wide", initial_sidebar_state="collapsed")


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

tab1, tab2, tab3, tab4 = st.tabs([
    "🔮 Predict Emotion", 
    "📊 Data Visualizations", 
    "⚙️ Retrain Model", 
    "🏥 System Health"
])

with tab1:#prediction tab
    st.header("Predict a Single Audio File")
    st.write("Upload a .wav file to hear it and predict the emotion.")
    
    uploaded_file = st.file_uploader("Upload an audio file", type=["wav", "ogg"], key="predict_upload")
    
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

#visualizations tab
with tab2:
    st.header("Feature Interpretations & Model Performance")
    
   
    st.subheader("🎵 Dynamic Audio Analysis")
    st.markdown("Upload any audio file to see its 'Sound DNA' (Waveform, Spectrogram, and MFCCs).")

    eda_file = st.file_uploader("Upload a .wav file for visual analysis", type=['wav', 'ogg'], key="eda_visuals")

    if eda_file is not None:
        y, sr = librosa.load(eda_file, duration=2.5)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("1. Waveform")
            fig_wave, ax_wave = plt.subplots(figsize=(10, 5))
            librosa.display.waveshow(y, sr=sr, ax=ax_wave, color='#D4AF37')
            ax_wave.set_title("Amplitude vs Time", color="white")
            fig_wave.patch.set_facecolor('#0a0a0a')
            ax_wave.set_facecolor('#0a0a0a')
            st.pyplot(fig_wave)

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

        st.divider()

        # MFCCs
        st.subheader("3.Vocal Tract Fingerprint (MFCCs)")
        fig_mfcc, ax_mfcc = plt.subplots(figsize=(12, 4))
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        img_mfcc = librosa.display.specshow(mfccs, x_axis='time', ax=ax_mfcc)
        plt.colorbar(img_mfcc, ax=ax_mfcc)
        ax_mfcc.set_title("MFCC Coefficients", color="white")
        fig_mfcc.patch.set_facecolor('#0a0a0a')
        st.pyplot(fig_mfcc)

        st.success("Features successfully extracted and visualized!")
    else:
        st.info("Please upload a .wav file in the box above to generate visuals.")
    
    #feature explanations

    st.markdown("""
        <div class="gold-card">
            <div class="gold-card-title">1. The Waveform (captures for energy and pacing)</div>
            <div class="gold-card-text">
                <b>Concept:</b> This represents the raw physical amplitude of the sound wave over time.<br>
                <b>story it tells:</b> High-arousal emotions like Anger or Happiness produce sharp, jagged amplitude peaks with sudden bursts of acoustic energy.
                In contrast, Sadness yields a smoother, compressed envelope with much lower overall physical power and slower pacing.
            </div>
        </div>
        
        <div class="gold-card">
            <div class="gold-card-title">2. The Mel-Spectrogram (captured for the frequency & timbre)</div>
            <div class="gold-card-text">
                <b>Concept:</b> A visual heat-map of frequency intensities, scaled logarithmically to match how human ears actually perceive sound.<br>
                <b>Story it tells:</b> It reveals the 'color' and pitch variations of the voice. Excitement or distress lights up the higher frequency bands (brighter colors), while calm or sad voices keep their acoustic energy tightly clustered in the lower, darker frequency ranges.
            </div>
        </div>
        
        <div class="gold-card">
            <div class="gold-card-title">3. MFCCs (captures the vocal tract geometry)</div>
            <div class="gold-card-text">
                <b>Concept:</b> Mel-Frequency Cepstral Coefficients act as a mathematical "fingerprint" of the speaker's vocal tract.<br>
                <b>Story it tells:</b> By mathematically stripping away background noise and base pitch, MFCCs isolate the physical shape of the throat, tongue, and lips. It allows the AI to detect the biological "tightness" of an angry throat versus the relaxed, sluggish articulation of a sad voice. This makes it our model's most critical feature.
            </div>
        </div>
    """, unsafe_allow_html=True)

    st.divider()


    st.subheader("Global Model Evaluation Metrics 📉")
    st.write("Official performance results from the last training session.")
    
    base_dir = os.getcwd() 
    evaluation_img_path = os.path.join(base_dir, "assets", "model_evaluation.png")
    
    if os.path.exists(evaluation_img_path):
        st.image(evaluation_img_path, caption="Confusion Matrix and Training Curves", use_container_width=True)
    else:
        st.warning(f"⚠️ Evaluation curves not found at: {evaluation_img_path}")

    # st.subheader("Global Model Evaluation Metrics")
    # st.write("Official performance results from the last training session.")
    
    # current_dir = os.path.dirname(os.path.abspath(__file__))

    # evaluation_img_path = os.path.join(current_dir, "assets", "model_evaluation.png")
    
    # if os.path.exists(evaluation_img_path):
    #     st.image(evaluation_img_path, caption="Confusion Matrix and Training Curves", use_container_width=True)
    # else:
    #     st.warning(f"Evaluation curves not found at: {evaluation_img_path}")
    st.markdown("""
        <div class="gold-card">
            <div class="gold-card-title">1. The confusion matrix (recorded a very good classification)</div>
            <div class="gold-card-text">
                <b>What we see:</b> All predictions fall perfectly on the dark blue diagonal line, with zero numbers in the white off-diagonal boxes.<br>
                <b>The Story:</b> The model made absolutely zero mistakes on the test data.
                It never confused an 'Angry' voice for a 'Happy' one, nor did it mix up 'Sad' and 'Neutral'. It successfully identified the unique mathematical signatures of all four emotions.
            </div>
        </div>
        
        <div class="gold-card">
            <div class="gold-card-title">2. Accuracy & Loss Curves (shows very rapid convergence)</div>
            <div class="gold-card-text">
                <b>What is seen:</b> The accuracy shoots up to 100% (1.0) and the loss drops to zero within the very first 5 to 10 epochs, staying completely flat after that.<br>
                <b>The story it tellls:</b> The model learned the patterns almost immediately. Because the Validation lines (green/orange) perfectly track the Training lines (yellow/red), we can confirm the model did not artificially memorize the data ("overfit"); it genuinely found the underlying rules for these audio files very quickly.
            </div>
        </div>
        
        <div class="gold-card">
            <div class="gold-card-title">3. The Real-World Context (The 100% Caveat)</div>
            <div class="gold-card-text">
                <b>My final machine learning interpretations and nsight:</b> While a 100% score is a massive technical success for our pipeline, in real-world ML, it usually indicates that the dataset is "too perfect." The TESS dataset features professional actors speaking clearly with zero background noise. If we deployed this in a noisy environment (like a busy street) with random speakers, the accuracy would naturally drop. This proves our architecture works flawlessly, but our next MLOps phase would require training on messier, real-world data.
            </div>
        </div>
    """, unsafe_allow_html=True)

with tab3:
    st.header("Bulk Upload & Model Retraining")
    zip_file = st.file_uploader("Upload Bulk Data (.zip)", type=["zip"])
    if zip_file is not None:
        if st.button("Trigger Retraining Pipeline", use_container_width=True):
            with st.spinner("Processing..."):
                files = {"file": (zip_file.name, zip_file.getvalue(), "application/zip")}
                try:
                    response = requests.post(f"{API_URL}/retrain", files=files)
                    if response.status_code == 200:
                        st.success(response.json()["message"])
                    else:
                        st.error("Failed.")
                except Exception as e:
                    st.error(f"Error: {e}")

with tab4:
    st.header("Model Server Health")
    if st.button("Ping Server for Status", use_container_width=True):
        try:
            response = requests.get(f"{API_URL}/health")
            if response.status_code == 200:
                data = response.json()
                c1, c2, c3 = st.columns(3)
                with c1: st.markdown(f'<div class="health-box"><div class="health-value">{data["status"].split()[0]}</div><div class="health-label">Status</div></div>', unsafe_allow_html=True)
                with c2: st.markdown(f'<div class="health-box"><div class="health-value">{data["uptime_minutes"]}</div><div class="health-label">Uptime (Mins)</div></div>', unsafe_allow_html=True)
                with c3: st.markdown(f'<div class="health-box"><div class="health-value">{data["uptime_seconds"]}</div><div class="health-label">Uptime (Secs)</div></div>', unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Offline: {e}")