import torch
import numpy as np
from torch.utils.data import DataLoader
from sentiment_analysis.train import (
    AudioEmotionDataset,
    AudioCNN,
    train_model,
    evaluate_model,
)

# =====================================================
# 🔹 TEST AudioEmotionDataset
# =====================================================

def test_audio_emotion_dataset_shapes():
    X = np.random.randn(8, 4, 40, 130).astype(np.float32)
    y = np.arange(8)

    dataset = AudioEmotionDataset(X, y)
    x0, y0 = dataset[0]

    # Formas correctas
    assert x0.shape == (4, 40, 130)
    assert isinstance(y0.item(), int)
    assert len(dataset) == 8


# =====================================================
# 🔹 TEST AudioCNN forward()
# =====================================================

def test_audiocnn_forward_pass():
    num_classes = 8
    model = AudioCNN(num_classes=num_classes)

    X_fake = torch.randn(2, 4, 40, 130)  # batch=2
    output = model(X_fake)

    # Debe devolver logits (batch, num_classes)
    assert output.shape == (2, num_classes)
    assert torch.is_tensor(output)


# =====================================================
# 🔹 TEST entrenamiento y evaluación (mini dataset)
# =====================================================

def test_train_and_evaluate(tmp_path):
    # Datos pequeños para un entrenamiento rápido
    X = np.random.randn(16, 4, 40, 130).astype(np.float32)
    y = np.random.randint(0, 4, size=(16,))
    dataset = AudioEmotionDataset(X, y)

    # Mini loaders
    train_loader = DataLoader(dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(dataset, batch_size=4)

    model = AudioCNN(num_classes=4)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Entrenamos solo 1 época para comprobar que no falla
    train_model(model, train_loader, val_loader, criterion, optimizer, epochs=1)

    acc = evaluate_model(model, val_loader)
    assert 0.0 <= acc <= 1.0  # accuracy válido

    # Guardar el modelo
    model_path = tmp_path / "cnn_test.pth"
    torch.save(model.state_dict(), model_path)
    assert model_path.exists()
