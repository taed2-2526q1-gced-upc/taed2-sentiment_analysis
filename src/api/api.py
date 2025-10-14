from fastapi import FastAPI, HTTPException
from loguru import logger
import torch
import librosa
import numpy as np
from pathlib import Path

from src.api.schemas import PredictRequest, PredictResponse, EmotionPrediction

# Configuration
MODEL_PATH = "models/emotion_model.pth"  # Ajustez selon votre chemin
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
        model = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
        model.eval()
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise

def preprocess_audio(audio_path: str) -> np.ndarray:
    """
    Preprocess audio file for model input
    
    Args:
        audio_path: Path to the audio file
    
    Returns:
        Preprocessed audio features
    """
    try:
        # Charger l'audio
        audio, sr = librosa.load(audio_path, sr=16000)  # Ajustez le sample rate
        
        # Extraire les features (MFCC, mel-spectrogram, etc.)
        # Adaptez selon ce que votre modèle attend
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
        
        # Normalisation ou autres transformations
        mfcc = (mfcc - np.mean(mfcc)) / np.std(mfcc)
        
        return mfcc
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