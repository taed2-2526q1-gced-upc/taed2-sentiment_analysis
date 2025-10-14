import json
import pandas as pd
from pathlib import Path
from sentiment_analysis.decode_audio_emotions import (
    decode_ravdess_filename,
    create_audio_emotion_mapping,
    create_simple_emotion_dict,
    save_mappings,
    analyze_dataset,
)

# =====================================================
# 🔹 TEST decode_ravdess_filename()
# =====================================================

def test_decode_ravdess_filename_valid():
    filename = "03-01-05-02-01-01-12.wav"
    meta = decode_ravdess_filename(filename)

    assert meta["valid"] is True
    assert meta["emotion"] == "angry"
    assert meta["intensity"] == "strong"
    assert meta["actor"] == "Actor_12"
    assert meta["statement"] == "Kids are talking by the door"


def test_decode_ravdess_filename_invalid():
    filename = "badfilename.wav"
    meta = decode_ravdess_filename(filename)

    assert meta["valid"] is False
    assert meta["emotion"] == "unknown"
    assert meta["actor"] == "unknown"

# =====================================================
# 🔹 TEST create_audio_emotion_mapping()
# =====================================================

def test_create_audio_emotion_mapping(tmp_path):
    # Simulamos estructura de carpetas con WAVs falsos
    actor_dir = tmp_path / "Actor_01"
    actor_dir.mkdir()
    fake_file = actor_dir / "03-01-05-02-01-01-12.wav"
    fake_file.touch()

    mapping = create_audio_emotion_mapping(tmp_path)

    assert len(mapping) == 1
    key = next(iter(mapping))
    data = mapping[key]
    assert data["actor_folder"] == "Actor_01"
    assert data["emotion"] == "angry"
    assert data["valid"] is True

# =====================================================
# 🔹 TEST create_simple_emotion_dict()
# =====================================================

def test_create_simple_emotion_dict():
    mock_mapping = {
        "Actor_01/03-01-05-02-01-01-12.wav": {"emotion": "angry"},
        "Actor_02/03-01-03-01-01-01-08.wav": {"emotion": "happy"},
    }
    simple = create_simple_emotion_dict(mock_mapping)

    assert list(simple.values()) == ["angry", "happy"]

# =====================================================
# 🔹 TEST save_mappings() y analyze_dataset()
# =====================================================

def test_save_and_analyze_mappings(tmp_path):
    # Creamos un mapping de ejemplo
    mock_mapping = {
        "Actor_01/file1.wav": {
            "filename": "file1.wav",
            "emotion": "angry",
            "intensity": "normal",
            "actor": "Actor_01",
            "valid": True,
        },
        "Actor_02/file2.wav": {
            "filename": "file2.wav",
            "emotion": "happy",
            "intensity": "strong",
            "actor": "Actor_02",
            "valid": True,
        },
    }

    # Guardamos en una carpeta temporal
    df = save_mappings(mock_mapping, output_dir=tmp_path)

    # Verificamos los archivos generados
    json_full = tmp_path / "audio_metadata_complete.json"
    json_simple = tmp_path / "audio_emotion_simple.json"
    csv_file = tmp_path / "audio_metadata.csv"

    assert json_full.exists()
    assert json_simple.exists()
    assert csv_file.exists()

    # Verificamos contenido básico
    with open(json_simple) as f:
        simple = json.load(f)
    assert "Actor_01/file1.wav" in simple

    # Verificamos que analyze_dataset devuelve los counts
    emotion_counts, intensity_counts = analyze_dataset(df)
    assert "angry" in emotion_counts.index
    assert "normal" in intensity_counts.index
