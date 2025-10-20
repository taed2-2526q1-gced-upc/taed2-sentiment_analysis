"""
Entrenamiento de una CNN para el reconocimiento de emociones en audio (SER).
Carga las features multicanal (MFCC, Mel, Chroma, RMSE) y entrena una CNN básica.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from tqdm import tqdm

# ================================
# CONFIG
# ================================
BASE_DIR = Path(__file__).resolve().parents[2]  
DATA_PATH = BASE_DIR / "data" / "processed" / "audio_features_multichannel.npz"
MODEL_PATH = BASE_DIR / "models" / "cnn_audio_emotion.pth"
EPOCHS = 30
BATCH_SIZE = 32
LR = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ================================
# DATASET
# ================================
class AudioEmotionDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]
        return x, self.y[idx]


# ================================
# MODELO CNN
# ================================
class AudioCNN(nn.Module):
    def __init__(self, num_classes):
        super(AudioCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(4, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(64 * 5 * 16, num_classes)  # ajusta según tus dimensiones

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        return self.fc(x)


# ================================
# TRAINING LOOP
# ================================
def train_model(model, train_loader, val_loader, criterion, optimizer, epochs=EPOCHS):
    # Aseguramos que modelo y datos están en el mismo dispositivo
    device = next(model.parameters()).device
    model.train()

    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0
        loop = tqdm(train_loader, leave=False)

        for X_batch, y_batch in loop:
            # Mover batch al mismo dispositivo que el modelo
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds = outputs.argmax(dim=1)
            correct += (preds == y_batch).sum().item()
            total += y_batch.size(0)

            loop.set_description(f"Epoch [{epoch+1}/{epochs}]")
            loop.set_postfix(loss=loss.item())

        acc = correct / total
        val_acc = evaluate_model(model, val_loader)
        print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(train_loader):.4f} | Train Acc: {acc:.3f} | Val Acc: {val_acc:.3f}")


def evaluate_model(model, loader):
    # Detectamos el dispositivo actual del modelo
    device = next(model.parameters()).device
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            preds = outputs.argmax(dim=1)
            correct += (preds == y_batch).sum().item()
            total += y_batch.size(0)
    return correct / total


# ================================
# MAIN
# ================================
from codecarbon import EmissionsTracker

def main():
    print("📂 Cargando dataset...")
    data = np.load(DATA_PATH, allow_pickle=True)

    X_train, y_train = data["X_train"], data["y_train"]
    X_val, y_val = data["X_val"], data["y_val"]
    X_test, y_test = data["X_test"], data["y_test"]
    classes = data["classes"]

    print(f"✅ Dataset cargado: {X_train.shape[0]} muestras de entrenamiento, {len(classes)} clases")

    train_loader = DataLoader(AudioEmotionDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(AudioEmotionDataset(X_val, y_val), batch_size=BATCH_SIZE)
    test_loader = DataLoader(AudioEmotionDataset(X_test, y_test), batch_size=BATCH_SIZE)

    model = AudioCNN(num_classes=len(classes)).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # =========================================================
    # 🟢 INICIO DEL TRACKER DE CODECARBON
    # =========================================================
    tracker = EmissionsTracker(
        project_name="EmotionRecognitionTraining",
        output_dir=BASE_DIR / "reports",
        measure_power_secs=5,  # frecuencia de medición
        log_level="info",
        save_to_file=True
    )
    tracker.start()

    print("🚀 Entrenando modelo...")
    train_model(model, train_loader, val_loader, criterion, optimizer)

    emissions = tracker.stop()
    print(f"🌱 Emisiones totales del entrenamiento: {emissions:.6f} kgCO₂eq")

    # =========================================================
    # 🧪 Evaluación final
    # =========================================================
    test_acc = evaluate_model(model, test_loader)
    print(f"✅ Accuracy en test: {test_acc:.3f}")

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"💾 Modelo guardado en {MODEL_PATH}")

if __name__ == "__main__":
    main()