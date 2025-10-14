from pydantic import BaseModel, Field
from typing import List

class AudioInput(BaseModel):
    """Input schema for emotion prediction from audio"""
    audio_path: str = Field(..., description="Path to the audio file")
    # ou si vous recevez l'audio en base64:
    # audio_data: str = Field(..., description="Base64 encoded audio data")

class EmotionPrediction(BaseModel):
    """Single emotion prediction result"""
    emotion: str = Field(..., description="Predicted emotion")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")

class PredictRequest(BaseModel):
    """Request schema for prediction endpoint"""
    audios: List[AudioInput] = Field(..., description="List of audio inputs")

class PredictResponse(BaseModel):
    """Response schema for prediction endpoint"""
    predictions: List[EmotionPrediction]