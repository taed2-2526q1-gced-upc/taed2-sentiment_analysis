import numpy as np
import pytest
from pathlib import Path
from sentiment_analysis.prepare_dataset import (
    pad_or_trim,
    extract_features,
)

# =====================================================
# 🔹 TEST pad_or_trim()
# =====================================================

def test_pad_or_trim_trims_matrix():
    mat = np.ones((50, 10))
    trimmed = pad_or_trim(mat, target_rows=40)
    assert trimmed.shape == (40, 10)
    assert np.all(trimmed == 1)

def test_pad_or_trim_pads_matrix():
    mat = np.ones((30, 5))
    padded = pad_or_trim(mat, target_rows=40)
    assert padded.shape == (40, 5)
    # las primeras 30 filas son 1s, el resto ceros
    assert np.all(padded[:30] == 1)
    assert np.all(padded[30:] == 0)

def test_pad_or_trim_no_change():
    mat = np.ones((40, 20))
    same = pad_or_trim(mat, target_rows=40)
    assert np.array_equal(same, mat)

# =====================================================
# 🔹 TEST extract_features()
# =====================================================

def test_extract_features_output_shape(tmp_path, monkeypatch):
    """
    Creamos un audio sintético pequeño y verificamos que extract_features
    devuelve un tensor (4, N_FREQ, frames)
    """
    # Creamos un archivo WAV temporal con ruido blanco
    import soundfile as sf
    sample_rate = 22050
    y = np.random.randn(sample_rate * 1)  # 1 segundo de ruido
    wav_path = tmp_path / "fake_audio.wav"
    sf.write(wav_path, y, sample_rate)

    # Ejecutamos la función real
    features = extract_features(wav_path)
    assert features is not None
    assert isinstance(features, np.ndarray)
    # 4 canales (MFCC, Mel, Chroma, RMSE)
    assert features.shape[0] == 4
    # Las frecuencias deben ser N_FREQ (40)
    assert features.shape[1] == 40
    # frames (número de columnas) > 0
    assert features.shape[2] > 0

def test_extract_features_handles_invalid_file(tmp_path):
    bad_path = tmp_path / "missing.wav"
    result = extract_features(bad_path)
    assert result is None

