import torch
import torch.nn as nn
import torch.optim as optim
import os
import copy
import csv
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

from data.dataset import get_dataloaders
from models.cnn_model import AlzheimerResNet


# -------------------------
# 🔹 Split dataset into clients
# -------------------------
def split_clients(dataset, num_clients):
    size = len(dataset) // num_clients
    return torch.utils.data.random_split(dataset, [size] * num_clients)


# -------------------------
# 🔹 Federated Averaging
# -------------------------
def federated_average(models):
    avg_model = copy.deepcopy(models[0])

    for key in avg_model.state_dict().keys():
        avg_model.state_dict()[key].data = torch.stack(
            [m.state_dict()[key].float() for m in models]
        ).mean(0)

    return avg_model


# -------------------------
# 🔹 Local client training
# -------------------------
def train_local(model, loader, device):
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

    return model


# -------------------------
# 🔹 Evaluation
# -------------------------
def evaluate(model, loader, device):
    model.eval()

    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    acc = np.mean(np.array(all_preds) == np.array(all_labels))
    precision = precision_score(all_labels, all_preds, average="weighted", zero_division=0)
    recall = recall_score(all_labels, all_preds, average="weighted", zero_division=0)
    f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)

    try:
        auc = roc_auc_score(all_labels, np.array(all_probs), multi_class="ovr")
    except:
        auc = 0.0

    cm = confusion_matrix(all_labels, all_preds)

    return acc, precision, recall, f1, auc, cm


# -------------------------
# 🔥 FEDERATED TRAINING
# -------------------------
def federated_training(data_dir, rounds=10, num_clients=3, batch_size=32):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    os.makedirs("weights", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    # Load data
    train_loader, val_loader = get_dataloaders(data_dir, batch_size=batch_size)

    dataset = train_loader.dataset

    # Split into clients
    client_sets = split_clients(dataset, num_clients)

    client_loaders = [
        torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True)
        for ds in client_sets
    ]

    # Global model
    global_model = AlzheimerResNet(num_classes=3).to(device)

    # CSV file
    csv_path = "results/federated_metrics.csv"

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "accuracy", "precision", "recall",
            "f1", "auc", "confusion_matrix", "round"
        ])

    best_f1 = 0

    # -------------------------
    # 🔁 Federated rounds
    # -------------------------
    for r in range(1, rounds + 1):
        print(f"\n🌐 Round {r}/{rounds}")

        local_models = []

        # Train each client
        for i, loader in enumerate(client_loaders):
            print(f"Client {i+1} training...")

            local_model = copy.deepcopy(global_model)
            local_model = train_local(local_model, loader, device)

            local_models.append(local_model)

        # Aggregate
        global_model = federated_average(local_models)

        # Evaluate global model
        acc, precision, recall, f1, auc, cm = evaluate(global_model, val_loader, device)

        print(f"Global → Acc: {acc:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}")

        # Save metrics
        with open(csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                round(acc, 3),
                round(precision, 3),
                round(recall, 3),
                round(f1, 3),
                round(auc, 3),
                cm.tolist(),
                r
            ])

        # Save best model
        if f1 > best_f1:
            best_f1 = f1
            torch.save(global_model.state_dict(), "weights/best_federated_model.pth")
            print("✅ Saved best global model")

    print("\n🎉 Federated Training Complete!")


# -------------------------
# ▶️ RUN
# -------------------------
if __name__ == "__main__":
    federated_training(
        data_dir="mri/Data",
        rounds=10,
        num_clients=3,
        batch_size=32
    )
