# sentiment_analysis/prepare_dataset.py
"""
Prepara el dataset de audio para entrenamiento de modelos de emoción (SER).
Extrae MFCC, Mel spectrogram, Chroma y RMSE, y genera arrays listos para CNNs.
"""

import os
import json
import numpy as np
import librosa
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

# ================================
# CONFIG
# ================================
BASE_DIR = Path(__file__).resolve().parents[1]
AUDIO_DIR = BASE_DIR / "data" / "raw" / "Audio_Speech_Actors"
METADATA_PATH = BASE_DIR / "data" / "processed" / "audio_metadata_complete.json"
OUTPUT_PATH = BASE_DIR / "data" / "processed"
SAMPLE_RATE = 22050
MAX_DURATION = 3.0  # segundos
MAX_LEN = int(SAMPLE_RATE * MAX_DURATION)

N_FREQ = 40  # tamaño uniforme para eje de frecuencia


# ================================
# FUNCIONES DE EXTRACCIÓN
# ================================
def pad_or_trim(matrix, target_rows=N_FREQ):
    """Recorta o rellena el número de filas de una matriz."""
    current_rows = matrix.shape[0]
    if current_rows > target_rows:
        return matrix[:target_rows, :]
    elif current_rows < target_rows:
        pad = np.zeros((target_rows - current_rows, matrix.shape[1]))
        return np.vstack([matrix, pad])
    else:
        return matrix


def extract_features(audio_path):
    """Extrae MFCC, Mel spectrogram, Chroma y RMSE de un archivo de audio."""
    try:
        y, sr = librosa.load(audio_path, sr=SAMPLE_RATE)
        if len(y) > MAX_LEN:
            y = y[:MAX_LEN]
        else:
            y = np.pad(y, (0, MAX_LEN - len(y)))

        # --- MFCC ---
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_FREQ)

        # --- Mel spectrogram ---
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        mel_db = librosa.power_to_db(mel, ref=np.max)

        # --- Chroma ---
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)

        # --- RMSE ---
        rmse = librosa.feature.rms(y=y)

        # Igualamos número de frames (columnas)
        min_frames = min(mfcc.shape[1], mel_db.shape[1], chroma.shape[1], rmse.shape[1])
        mfcc, mel_db, chroma, rmse = mfcc[:, :min_frames], mel_db[:, :min_frames], chroma[:, :min_frames], rmse[:, :min_frames]

        # Recortamos o rellenamos filas a N_FREQ
        mel_db = pad_or_trim(mel_db, N_FREQ)
        chroma = pad_or_trim(chroma, N_FREQ)
        rmse = np.repeat(pad_or_trim(rmse, 1), N_FREQ, axis=0)  # replicamos a 40 filas

        # Normalizamos
        def norm(x):
            return (x - np.mean(x)) / (np.std(x) + 1e-6)

        mfcc, mel_db, chroma, rmse = map(norm, [mfcc, mel_db, chroma, rmse])

        # Empilamos como canales → (n_canales, n_features, n_frames)
        features = np.stack([mfcc, mel_db, chroma, rmse], axis=0)

        return features

    except Exception as e:
        print(f"Error procesando {audio_path}: {e}")
        return None


# ================================
# MAIN PIPELINE
# ================================
def main():
    print(f"📂 Leyendo metadata desde: {METADATA_PATH}")
    with open(METADATA_PATH, "r") as f:
        metadata = json.load(f)

    X, y = [], []

    print("🎧 Extrayendo features de audios...")
    for rel_path, info in tqdm(metadata.items()):
        audio_path = AUDIO_DIR / rel_path
        emotion = info["emotion"]

        features = extract_features(audio_path)
        if features is not None:
            X.append(features)
            y.append(emotion)

    X = np.array(X)
    y = np.array(y)

    print(f"✅ Features extraídas: {X.shape}")
    print(f"✅ Etiquetas: {len(y)} muestras")

    # Codificamos emociones
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # Split 70 / 15 / 15
    X_train, X_temp, y_train, y_temp = train_test_split(X, y_encoded, test_size=0.3, stratify=y_encoded, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    out_file = OUTPUT_PATH / "audio_features_multichannel.npz"
    np.savez_compressed(
        out_file,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        classes=le.classes_,
    )

    print(f"💾 Dataset guardado en: {out_file}")
    print("✅ Todo listo para entrenar la CNN 🚀")


if __name__ == "__main__":
    main()
