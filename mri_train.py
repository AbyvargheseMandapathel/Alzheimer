import torch
import torch.nn as nn
import torch.optim as optim
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics import (
    precision_recall_fscore_support,
    accuracy_score,
    roc_auc_score,
    confusion_matrix
)

from data.dataset import get_dataloaders
from models.cnn_model import AlzheimerResNet


def train_mri_model(data_dir, num_epochs=20, batch_size=32, lr=1e-4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Training MRI Model on: {device}")

    # Create directories
    os.makedirs("weights", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    # Data
    print("📦 Loading MRI Dataset...")
    train_loader, val_loader = get_dataloaders(data_dir, batch_size=batch_size)

    # Model
    print("🧠 Initializing ResNet...")
    model = AlzheimerResNet(num_classes=3).to(device)

    # Loss & Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3, verbose=True
    )

    best_f1 = 0.0
    metrics_history = []

    # Training loop
    for epoch in range(num_epochs):
        print(f"\n--- Epoch {epoch+1}/{num_epochs} ---")

        # ================= TRAIN =================
        model.train()
        running_loss = 0.0
        train_preds = []
        train_labels = []

        for images, labels in tqdm(train_loader, desc="Training"):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            train_preds.extend(predicted.cpu().numpy())
            train_labels.extend(labels.cpu().numpy())

        train_acc = accuracy_score(train_labels, train_preds)

        # ================= VALIDATION =================
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_labels = []
        val_probs = []

        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc="Validating"):
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                probs = torch.softmax(outputs, dim=1)

                val_loss += loss.item()

                _, predicted = torch.max(outputs, 1)

                val_preds.extend(predicted.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
                val_probs.extend(probs.cpu().numpy())

        # ================= METRICS =================
        val_acc = accuracy_score(val_labels, val_preds)

        precision, recall, f1, _ = precision_recall_fscore_support(
            val_labels, val_preds, average='weighted', zero_division=0
        )

        try:
            auc = roc_auc_score(val_labels, np.array(val_probs), multi_class='ovr')
        except Exception:
            auc = 0.0  # fallback if AUC fails

        cm = confusion_matrix(val_labels, val_preds)

        print(f"\n📊 Epoch {epoch+1} Summary:")
        print(f"Train Loss: {running_loss/len(train_loader):.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val Loss  : {val_loss/len(val_loader):.4f} | Val Acc  : {val_acc:.4f}")
        print(f"Precision : {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f} | AUC: {auc:.4f}")

        # ================= SAVE METRICS =================
        metrics_history.append({
            "round": epoch + 1,
            "train_loss": running_loss / len(train_loader),
            "val_loss": val_loss / len(val_loader),
            "train_accuracy": train_acc,
            "val_accuracy": val_acc,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "auc": auc,
            "confusion_matrix": cm.tolist()
        })

        # ================= SAVE BEST MODEL =================
        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), "weights/best_centralized_model.pth")
            print(f"✅ New best model saved (F1: {best_f1:.4f})")

        scheduler.step(val_acc)

    # ================= SAVE CSV ONCE =================
    df = pd.DataFrame(metrics_history)
    df.to_csv("results/mri_metrics.csv", index=False)

    print(f"\n🎉 Training Finished. Best F1 Score: {best_f1:.4f}")
    print("📁 Metrics saved to results/mri_metrics.csv")


if __name__ == "__main__":
    DATASET_PATH = "mri/Data"

    if os.path.exists(DATASET_PATH):
        train_mri_model(DATASET_PATH)
    else:
        print(f"❌ Error: Dataset path {DATASET_PATH} not found!")
