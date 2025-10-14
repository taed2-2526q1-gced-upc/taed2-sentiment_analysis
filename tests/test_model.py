import torch
import numpy as np
from sentiment_analysis.train import AudioCNN, MODEL_PATH

def test_saved_model_exists():
    """Comprueba que el modelo entrenado existe en la ruta esperada."""
    assert MODEL_PATH.exists(), f"El modelo no se encontró en {MODEL_PATH}"

def test_model_load_and_forward():
    """Verifica que el modelo guardado puede cargarse y realizar inferencias."""
    num_classes = 8  # ajusta según tus clases reales
    model = AudioCNN(num_classes=num_classes)
    
    # Cargamos los pesos guardados
    state_dict = torch.load(MODEL_PATH, map_location="cpu")
    model.load_state_dict(state_dict)

    # Hacemos una inferencia de prueba
    X_fake = torch.randn(1, 4, 40, 130)
    output = model(X_fake)

    # Comprobamos forma de salida
    assert output.shape == (1, num_classes), "La salida del modelo tiene dimensiones incorrectas"
    assert torch.isfinite(output).all(), "La salida del modelo contiene valores no finitos"

def test_model_parameters_count():
    """Valida que el modelo tiene el número esperado de parámetros (indicativo de arquitectura correcta)."""
    num_classes = 8
    model = AudioCNN(num_classes=num_classes)
    params = sum(p.numel() for p in model.parameters())
    assert params > 10_000, "El número de parámetros parece demasiado bajo, posible error en la carga del modelo"
