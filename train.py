import torch
import torch.nn as nn
import torch.optim as optim
import os
import csv
import numpy as np
from tqdm import tqdm
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

from data.dataset import get_dataloaders
from models.cnn_model import AlzheimerResNet


def train_model(data_dir, num_epochs=5, batch_size=32, device="cuda"):
    print("Initializing Data Loaders...")
    train_loader, val_loader = get_dataloaders(data_dir, batch_size=batch_size)

    print(f"Loading Model to {device}...")
    model = AlzheimerResNet(num_classes=3).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)

    os.makedirs("weights", exist_ok=True)

    # CSV setup
    csv_file = "mri_metrics.csv"
    file_exists = os.path.isfile(csv_file)

    with open(csv_file, mode="a", newline="") as f:
        writer = csv.writer(f)

        if not file_exists:
            writer.writerow([
                "epoch", "val_accuracy", "precision", "recall",
                "f1", "auc", "confusion_matrix"
            ])

        best_acc = 0.0

        print("Starting Training...")
        for epoch in range(num_epochs):
            # ================= TRAIN =================
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0

            for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]"):
                images, labels = images.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)

                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            train_acc = 100 * correct / total
            print(f"Epoch {epoch+1} - Train Loss: {running_loss/len(train_loader):.4f}, Train Acc: {train_acc:.2f}%")

            # ================= VALIDATION =================
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0

            all_preds = []
            all_labels = []
            all_probs = []

            with torch.no_grad():
                for images, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]"):
                    images, labels = images.to(device), labels.to(device)

                    outputs = model(images)
                    loss = criterion(outputs, labels)

                    probs = torch.softmax(outputs, dim=1)

                    val_loss += loss.item()
                    _, predicted = torch.max(outputs, 1)

                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()

                    all_preds.extend(predicted.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
                    all_probs.extend(probs.cpu().numpy())

            val_acc = 100 * val_correct / val_total

            # ================= METRICS =================
            precision = precision_score(all_labels, all_preds, average="weighted", zero_division=0)
            recall = recall_score(all_labels, all_preds, average="weighted", zero_division=0)
            f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)

            try:
                auc = roc_auc_score(all_labels, np.array(all_probs), multi_class="ovr")
            except:
                auc = 0.0

            cm = confusion_matrix(all_labels, all_preds)

            print(
                f"Epoch {epoch+1} - Val Loss: {val_loss/len(val_loader):.4f}, "
                f"Val Acc: {val_acc:.2f}% | Precision: {precision:.4f} | "
                f"Recall: {recall:.4f} | F1: {f1:.4f} | AUC: {auc:.4f}"
            )

            # ================= SAVE TO CSV =================
            writer.writerow([
                epoch + 1,
                val_acc,
                precision,
                recall,
                f1,
                auc,
                cm.tolist()
            ])

            # ================= SAVE BEST MODEL =================
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(model.state_dict(), "weights/best_centralized_model.pth")
                print("=> Saved optimal model")

    print(f"Training Complete. Best Val Accuracy: {best_acc:.2f}%")


if __name__ == "__main__":
    dataset_path = "mri/Data"
    device_to_use = "cuda" if torch.cuda.is_available() else "cpu"

    train_model(
        data_dir=dataset_path,
        num_epochs=10,
        batch_size=32,
        device=device_to_use
    )
