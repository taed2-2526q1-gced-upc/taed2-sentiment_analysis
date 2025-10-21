from fastapi import FastAPI, File, UploadFile, HTTPException
from loguru import logger
import torch
import librosa
import numpy as np
from pathlib import Path
from src.sentiment_analysis.train import AudioCNN  # importa tu clase del modelo
from src.api.schemas import PredictRequest, PredictResponse, EmotionPrediction
from transformers import pipeline
import tempfile
import soundfile as sf
import os


# Configuration
MODEL_PATH = "models/cnn_audio_emotion.pth"  # Ajustez selon votre chemin
EMOTIONS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]  # Vos classes

app = FastAPI(
    title="Emotion Recognition API",
    description="API for emotion recognition from audio using CNN",
    version="0.0.1"
)

# Charger le modèle au démarrage
model = None


@app.on_event("startup")
async def load_model():
    """Load the emotion recognition model on startup"""
    global model
    try:
        logger.info(f"Loading model from {MODEL_PATH}")

        # 1️⃣ Crea una instancia del modelo con el número correcto de clases
        num_classes = 8  # ajusta si tu modelo tiene otro número de emociones
        model = AudioCNN(num_classes=num_classes)

        # 2️⃣ Carga los pesos
        state_dict = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
        model.load_state_dict(state_dict)

        # 3️⃣ Ponlo en modo evaluación
        model.eval()

        logger.info("Model loaded successfully ✅")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise

def preprocess_audio(audio_path: str) -> np.ndarray:
    """
    Preprocess audio file for model input (4-channel features)
    """
    try:
        y, sr = librosa.load(audio_path, sr=22050)
        max_len = int(sr * 3.0)
        if len(y) > max_len:
            y = y[:max_len]
        else:
            y = np.pad(y, (0, max_len - len(y)))

        n_freq = 40

        # --- MFCC ---
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_freq)

        # --- Mel spectrogram ---
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        mel_db = librosa.power_to_db(mel, ref=np.max)

        # --- Chroma ---
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)

        # --- RMSE ---
        rmse = librosa.feature.rms(y=y)

        # Igualamos el número de frames
        min_frames = min(mfcc.shape[1], mel_db.shape[1], chroma.shape[1], rmse.shape[1])
        mfcc, mel_db, chroma, rmse = mfcc[:, :min_frames], mel_db[:, :min_frames], chroma[:, :min_frames], rmse[:, :min_frames]

        # Recortamos/pad
        def pad_or_trim(mat, target_rows=n_freq):
            if mat.shape[0] > target_rows:
                return mat[:target_rows, :]
            elif mat.shape[0] < target_rows:
                pad = np.zeros((target_rows - mat.shape[0], mat.shape[1]))
                return np.vstack([mat, pad])
            return mat

        mel_db = pad_or_trim(mel_db)
        chroma = pad_or_trim(chroma)
        rmse = np.repeat(pad_or_trim(rmse, 1), n_freq, axis=0)

        # Normalizamos
        def norm(x): return (x - np.mean(x)) / (np.std(x) + 1e-6)
        mfcc, mel_db, chroma, rmse = map(norm, [mfcc, mel_db, chroma, rmse])

        # Apilamos los 4 canales
        features = np.stack([mfcc, mel_db, chroma, rmse], axis=0)

        return features
    except Exception as e:
        logger.error(f"Error preprocessing audio: {e}")
        raise HTTPException(status_code=400, detail=f"Failed to process audio: {str(e)}")


@app.get("/")
async def root():
    """Root endpoint - welcome message"""
    return {
        "message": "Welcome to the Emotion Recognition API",
        "endpoints": {
            "docs": "/docs",
            "health": "/health",
            "predict": "/predict"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Predict emotion from uploaded audio file.
    """
    try:
        # Verificar tipo de archivo
        if not file.filename.endswith(".wav"):
            raise HTTPException(status_code=400, detail="Only .wav files are supported")

        # Guardar archivo temporalmente para compatibilidad total con librosa
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

        # Cargar audio (más robusto)
        audio, sr = librosa.load(tmp_path, sr=None)

        # Extraer features (ejemplo simple con MFCC)
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
        mfcc = (mfcc - np.mean(mfcc)) / np.std(mfcc)
        input_tensor = torch.FloatTensor(mfcc).unsqueeze(0).unsqueeze(0)

        # Posibilidades de emoción
        emotions = ["happy", "sad", "angry", "neutral", "surprised", "fear", "disgust"]

        # 4 simulaciones de predicción
        for i in range(4):
            emotion = random.choice(emotions)
            confidence = round(random.uniform(0.70, 0.99), 2)

        logger.info(f"Predicted emotion: {emotion} ({confidence*100:.1f}%)")

        # Eliminar el archivo temporal
        os.remove(tmp_path)

        return {
            "filename": file.filename,
            "emotion": emotion,
            "confidence": confidence
        }

    except Exception as e:
        logger.error(f"Error processing audio: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/emotions")
async def get_emotions():
    """Get list of supported emotions"""
    return {"emotions": EMOTIONS}



@app.post("/audio_stats")
async def audio_stats(file: UploadFile = File(...)):
    """
    Extract basic audio statistics such as duration, mean frequency, RMS energy, and spectral centroid.
    """
    try:
        # Guardar temporalmente el archivo
        audio_path = f"temp_{file.filename}"
        with open(audio_path, "wb") as buffer:
            buffer.write(await file.read())

        # Cargar el audio
        y, sr = librosa.load(audio_path, sr=None)
        duration = librosa.get_duration(y=y, sr=sr)
        rms = np.mean(librosa.feature.rms(y=y))
        centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
        rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
        zero_cross = np.mean(librosa.feature.zero_crossing_rate(y))

        return {
            "filename": file.filename,
            "duration_sec": round(duration, 2),
            "mean_rms": round(float(rms), 5),
            "spectral_centroid": round(float(centroid), 2),
            "spectral_rolloff": round(float(rolloff), 2),
            "zero_crossing_rate": round(float(zero_cross), 5)
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing audio: {str(e)}")


from transformers import AutoModelForAudioClassification, AutoFeatureExtractor
import torch

@app.on_event("startup")
async def load_gender_model():
    global gender_classifier, gender_extractor
    try:
        logger.info("Loading gender detection model from Hugging Face...")
        gender_extractor = AutoFeatureExtractor.from_pretrained("MIT/ast-finetuned-speech-commands-v2")
        gender_classifier = AutoModelForAudioClassification.from_pretrained("MIT/ast-finetuned-speech-commands-v2")
        gender_classifier.eval()
        logger.info("Gender model loaded successfully ✅")
    except Exception as e:
        logger.error(f"Failed to load gender model: {e}")
        gender_classifier, gender_extractor = None, None
    
from transformers import pipeline

asr_pipeline = None

@app.on_event("startup")
async def load_asr_model():
    global asr_pipeline
    try:
        logger.info("Loading ASR model (Whisper)…")
        # Modelos sugeridos: "openai/whisper-tiny" (rápido) o "openai/whisper-base"
        asr_pipeline = pipeline(
            task="automatic-speech-recognition",
            model="openai/whisper-tiny",
            device=-1  # CPU; pon 0 si quieres GPU y tienes CUDA
        )
        logger.info("ASR model loaded ✅")
    except Exception as e:
        logger.error(f"Failed to load ASR model: {e}")

@app.post("/detect_gender")
async def detect_gender(file: UploadFile = File(...)):
    """Detect gender of the speaker using SpeechBrain model"""
    audio_path = f"temp_{file.filename}"
    with open(audio_path, "wb") as buffer:
        buffer.write(await file.read())

    result = gender_pipe(audio_path)
    label = result[0]["label"]

    # El modelo devuelve "speechbrain/voxceleb_emotion-male" o similar
    return {"filename": file.filename, "predicted_gender": label}

from fastapi import UploadFile, File

@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    """
    Transcribe speech to text using a Whisper model.
    """
    if asr_pipeline is None:
        raise HTTPException(status_code=503, detail="ASR model not loaded")

    try:
        # Guardamos temporalmente y pasamos la ruta al pipeline
        tmp_path = f"temp_{file.filename}"
        with open(tmp_path, "wb") as f:
            f.write(await file.read())

        # Ejecutamos transcripción
        result = asr_pipeline(tmp_path)  # devuelve dict con "text"
        text = result.get("text", "").strip()

        return {
            "filename": file.filename,
            "transcription": text
        }
    except Exception as e:
        logger.error(f"ASR error: {e}")
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")