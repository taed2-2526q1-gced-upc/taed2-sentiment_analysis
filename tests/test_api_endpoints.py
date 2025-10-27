import os
from fastapi.testclient import TestClient
from src.api.api import app

client = TestClient(app)

# Ruta a un archivo de audio de prueba (usa uno pequeño del dataset)
AUDIO_PATH = "data/raw/Audio_Speech_Actors/Actor_01/03-01-01-01-01-01-01.wav"

def test_predict_endpoint():
    """Test /predict endpoint with audio file upload."""
    assert os.path.exists(AUDIO_PATH), f"Audio file not found: {AUDIO_PATH}"

    with open(AUDIO_PATH, "rb") as f:
        response = client.post("/predict", files={"file": ("test.wav", f, "audio/wav")})

    assert response.status_code == 200, f"Error: {response.text}"
    data = response.json()
    assert "emotion" in data
    assert "confidence" in data
    print(f"🎯 Predict → {data}")


def test_audio_stats_endpoint():
    """Test /audio_stats endpoint."""
    assert os.path.exists(AUDIO_PATH)

    with open(AUDIO_PATH, "rb") as f:
        response = client.post("/audio_stats", files={"file": ("test.wav", f, "audio/wav")})

    assert response.status_code == 200, f"Error: {response.text}"
    data = response.json()
    for key in ["duration_sec", "mean_rms", "spectral_centroid", "spectral_rolloff", "zero_crossing_rate"]:
        assert key in data, f"Missing key in /audio_stats: {key}"
    print(f"📊 Audio Stats → {data}")


def test_transcribe_endpoint():
    """Test /transcribe endpoint with Whisper model."""
    assert os.path.exists(AUDIO_PATH)

    with open(AUDIO_PATH, "rb") as f:
        response = client.post("/transcribe", files={"file": ("test.wav", f, "audio/wav")})

    assert response.status_code == 200, f"Error: {response.text}"
    data = response.json()
    assert "transcription" in data, "Transcription not found"
    print(f"🗣️ Transcription → {data['transcription']}")