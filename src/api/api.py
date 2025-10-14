from fastapi import FastAPI, HTTPException
from loguru import logger
import torch
import librosa
import numpy as np
from pathlib import Path
from src.sentiment_analysis.train import AudioCNN  # importa tu clase del modelo
from src.api.schemas import PredictRequest, PredictResponse, EmotionPrediction

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

@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    """
    Predict emotions from audio files
    
    Args:
        request: PredictRequest containing list of audio paths
    
    Returns:
        PredictResponse with emotion predictions
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    predictions = []
    
    for audio_input in request.audios:
        try:
            # Vérifier que le fichier existe
            audio_path = Path(audio_input.audio_path)
            if not audio_path.exists():
                raise HTTPException(
                    status_code=404, 
                    detail=f"Audio file not found: {audio_input.audio_path}"
                )
            
            # Prétraiter l'audio
            features = preprocess_audio(str(audio_path))
            
            # Convertir en tensor
            input_tensor = torch.FloatTensor(features).unsqueeze(0)
            
            # Prédiction
            with torch.no_grad():
                output = model(input_tensor)
                probabilities = torch.softmax(output, dim=1)
                confidence, predicted_idx = torch.max(probabilities, 1)
            
            # Créer la réponse
            prediction = EmotionPrediction(
                emotion=EMOTIONS[predicted_idx.item()],
                confidence=float(confidence.item())
            )
            predictions.append(prediction)
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            raise HTTPException(
                status_code=500, 
                detail=f"Prediction failed: {str(e)}"
            )
    
    return PredictResponse(predictions=predictions)

@app.get("/emotions")
async def get_emotions():
    """Get list of supported emotions"""
    return {"emotions": EMOTIONS}