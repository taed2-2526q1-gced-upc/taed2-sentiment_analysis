import os
import requests
import streamlit as st
from streamlit_mic_recorder import mic_recorder
from pathlib import Path

API_URL = "http://127.0.0.1:8000"
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

st.title("🎧 Emotion & Speech Analysis Demo")

# ==============================
# INPUT SELECTION
# ==============================
st.markdown("### 🎙️ Choose an input method")
method = st.radio("Select input type:", ["Upload file", "Record with mic"])

# Initialize session state
if "file_path" not in st.session_state:
    st.session_state.file_path = None

# --- Upload method ---
if method == "Upload file":
    uploaded_file = st.file_uploader("🎧 Upload an audio file", type=["wav", "mp3"])
    if uploaded_file:
        file_path = UPLOAD_DIR / uploaded_file.name
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.audio(str(file_path), format="audio/wav")
        st.success(f"✅ File saved: {uploaded_file.name}")
        st.session_state.file_path = file_path

# --- Mic method ---
elif method == "Record with mic":
    st.info("🎤 Click below to start recording your voice")
    audio_bytes = mic_recorder(start_prompt="🎙️ Record", stop_prompt="⏹️ Stop", key="recorder")
    if audio_bytes:
        file_path = UPLOAD_DIR / "recorded_audio.wav"

        # Si streamlit_mic_recorder devuelve un dict (como suele hacer)
        if isinstance(audio_bytes, dict) and "bytes" in audio_bytes:
            audio_data = audio_bytes["bytes"]
        else:
            audio_data = audio_bytes

        with open(file_path, "wb") as f:
            f.write(audio_data)

        st.audio(audio_data, format="audio/wav")
        st.success("✅ Recording saved successfully!")
        st.session_state.file_path = file_path

# ==============================
# PROCESSING
# ==============================
if st.session_state.file_path and os.path.exists(st.session_state.file_path):
    st.markdown("---")
    st.subheader("🚀 Run analysis")

    if st.button("Analyze Audio"):
        with st.spinner("Analyzing audio... ⏳"):
            try:
                with open(st.session_state.file_path, "rb") as f:
                    files = {"file": f}

                    # Predict emotion
                    pred = requests.post(f"{API_URL}/predict", files=files).json()
                    f.seek(0)

                    # Audio stats
                    stats = requests.post(f"{API_URL}/audio_stats", files=files).json()
                    f.seek(0)

                    # Transcribe
                    trans = requests.post(f"{API_URL}/transcribe", files=files).json()

            except Exception as e:
                st.error(f"❌ Error connecting to API: {e}")
                st.stop()

        # ==============================
        # RESULTS DISPLAY
        # ==============================
        st.markdown("## 📊 Results")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("### 🎭 Emotion")
            if "emotion" in pred:
                st.metric(label="Emotion", value=pred["emotion"].capitalize())
                st.write(f"Confidence: {pred['confidence']*100:.1f}%")

        with col2:
            st.markdown("### 🎧 Audio Stats")
            st.write(f"Duration: {stats.get('duration_sec', 0):.2f} sec")
            st.write(f"RMS: {stats.get('mean_rms', 0):.4f}")
            st.write(f"Centroid: {stats.get('spectral_centroid', 0):.1f} Hz")
            st.write(f"Roll-off: {stats.get('spectral_rolloff', 0):.1f} Hz")
            st.write(f"ZCR: {stats.get('zero_crossing_rate', 0):.5f}")

        with col3:
            st.markdown("### 🗣️ Transcription")
            st.info(trans.get("transcription", "No transcription"))

else:
    st.info("👆 Upload or record an audio file to begin analysis.")