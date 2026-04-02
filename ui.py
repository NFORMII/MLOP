import streamlit as st
import requests

# The URL for FastAPI backend is running
API_URL = "https://mlop-audio-backend.onrender.com"


st.set_page_config(page_title="TESS Audio AI", layout="wide", initial_sidebar_state="collapsed")


custom_css = """
<style>
    /* Force the main background to deep black and text to off-white */
    .stApp {
        background-color: #0a0a0a;
        color: #f0f0f0;
    }
    
    /* Make all default headers Gold */
    h1, h2, h3 {
        color: #D4AF37 !important;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }

    /* Custom CSS for the Data Visualization Cards */
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

    /* Custom CSS for System Health Metrics */
    .health-box {
        background-color: #141414;
        border: 1px solid #D4AF37;
        border-radius: 8px;
        padding: 25px 10px;
        text-align: center;
        box-shadow: 0 0 20px rgba(212, 175, 55, 0.15);
    }
    .health-value {
        font-size: 3rem;
        color: #D4AF37;
        font-weight: 800;
        margin-bottom: 5px;
    }
    .health-label {
        font-size: 0.9rem;
        color: #ffffff;
        text-transform: uppercase;
        letter-spacing: 1.5px;
    }

   
    div.stButton > button, 
    section[data-testid="stFileUploadDropzone"] button {
        background-color: #000000 !important;
        color: #D4AF37 !important;
        border: 1px solid #D4AF37 !important;
        transition: none 
    }

    
    div.stButton > button:hover, 
    div.stButton > button:active,
    div.stButton > button:focus,
    section[data-testid="stFileUploadDropzone"] button:hover {
        background-color: #000000 !important;
        color: #D4AF37 !important;
        border: 1px solid #D4AF37 !important;
    }

    
    section[data-testid="stFileUploadDropzone"] {
        background-color: #0a0a0a !important;
        border: 1px dashed #D4AF37 !important;
    }
</sty
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)


st.title("🎧 Speech Emotion Recognition Pipeline")
st.markdown("*A professional MLOps architecture analyzing the Toronto Emotional Speech Set.*")
st.markdown("---")

#creating the 4 Tabs required by the Rubric
tab1, tab2, tab3, tab4 = st.tabs([
    "🔮 Predict Emotion", 
    "📊 Data Visualizations", 
    "⚙️ Retrain Model", 
    "🏥 System Health"
])

# model prediction
with tab1:
    st.header("Predict a Single Audio File")
    st.write("Upload a .wav file to hear it and predict the emotion.")
    
    uploaded_file = st.file_uploader("Upload an audio file", type=["wav"])
    
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
                        st.error("Error connecting to the API.")
                except Exception as e:
                    st.error(f"Failed to connect to backend server. Is FastAPI running? Error: {e}")

#data visualization
with tab2:
    st.header("Feature Interpretations")
    st.write("To understand how our Deep Learning model differentiates between human emotions, we mathematically extract 3 specific features from the raw audio waves.")
    st.write("") # Spacer
    
   
    st.markdown("""
        <div class="gold-card">
            <div class="gold-card-title">1. The Waveform (Physical Intensity)</div>
            <div class="gold-card-text">
                <b>What it is:</b> The physical amplitude (loudness) of the sound over time.<br>
                <b>The Story:</b> Angry voices have massive, jagged peaks showing high physical energy, while Sad voices are tightly compressed with very low amplitude.
            </div>
        </div>
        
        <div class="gold-card">
            <div class="gold-card-title">2. Mel Spectrogram (Frequency Mapping)</div>
            <div class="gold-card-text">
                <b>What it is:</b> A visual representation of the spectrum of frequencies as they vary with time.<br>
                <b>The Story:</b> Happy and Angry voices light up the higher frequency bands (brighter colors), whereas Neutral and Sad voices keep their energy clustered in the lower, darker frequencies.
            </div>
        </div>
        
        <div class="gold-card">
            <div class="gold-card-title">3. MFCCs (Vocal Tract Geometry)</div>
            <div class="gold-card-text">
                <b>What it is:</b> Mel-Frequency Cepstral Coefficients represent the actual biological shape of the vocal tract.<br>
                <b>The Story:</b> This is the most critical feature. By isolating how the throat and mouth physically move to produce sound, the AI can detect the physical 'tightness' of anger versus the 'sluggishness' of sadness.
            </div>
        </div>
    """, unsafe_allow_html=True)

# retraining tab
with tab3:
    st.header("Bulk Upload & Model Retraining")
    st.write("Upload a `.zip` file containing new audio data organized into folders to trigger a background retraining pipeline.")
    
    zip_file = st.file_uploader("Upload Bulk Data (.zip)", type=["zip"])
    
    if zip_file is not None:
        if st.button("Trigger Retraining Pipeline", use_container_width=True):
            with st.spinner("Sending bulk data to the server..."):
                files = {"file": (zip_file.name, zip_file.getvalue(), "application/zip")}
                try:
                    response = requests.post(f"{API_URL}/retrain", files=files)
                    if response.status_code == 200:
                        st.success("" + response.json()["message"])
                        st.toast('Retraining triggered in the background!', icon='🔄')
                    else:
                        st.error("Failed to trigger retraining.")
                except Exception as e:
                    st.error("Failed to connect to backend server.")

# health and uptime
with tab4:
    st.header("Model Server Health")
    st.write("Live telemetry from the FastAPI backend.")
    st.write("") 
    
    if st.button("Ping Server for Status", use_container_width=True):
        try:
            response = requests.get(f"{API_URL}/health")
            if response.status_code == 200:
                data = response.json()
                
            
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown(f"""
                        <div class="health-box">
                            <div class="health-value">{data['status'].split()[0]}</div>
                            <div class="health-label">System Status</div>
                        </div>
                    """, unsafe_allow_html=True)
                    
                with col2:
                    st.markdown(f"""
                        <div class="health-box">
                            <div class="health-value">{data['uptime_minutes']}</div>
                            <div class="health-label">Uptime (Minutes)</div>
                        </div>
                    """, unsafe_allow_html=True)
                    
                with col3:
                    st.markdown(f"""
                        <div class="health-box">
                            <div class="health-value">{data['uptime_seconds']}</div>
                            <div class="health-label">Uptime (Seconds)</div>
                        </div>
                    """, unsafe_allow_html=True)
                    
            else:
                st.error("API is returning an error.")
        except requests.exceptions.ConnectionError:
            st.error("💥CRITICAL: Cannot connect to the API. Is your FastAPI server running?")