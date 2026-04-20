import torch
import torch.nn as nn
import torch.optim as optim
import os
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from data.dataset import get_dataloaders
from models.cnn_model import AlzheimerResNet

def train_mri_model(data_dir, num_epochs=20, batch_size=32, lr=1e-4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Training MRI Model on: {device}")
    
    # Create results and weights directories
    os.makedirs("weights", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    
    # 1. Initialize Data Loaders
    print("📦 Loading MRI Dataset...")
    train_loader, val_loader = get_dataloaders(data_dir, batch_size=batch_size)
    
    # 2. Initialize Model (ResNet50)
    print("🧠 Initializing ResNet50 Architecture...")
    model = AlzheimerResNet(num_classes=3).to(device)
    
    # 3. Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, verbose=True)
    
    best_acc = 0.0
    metrics_history = []
    
    # 4. Training Loop
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        all_preds = []
        all_labels = []
        
        print(f"\n--- Epoch {epoch+1}/{num_epochs} ---")
        train_pbar = tqdm(train_loader, desc=f"Training")
        for images, labels in train_pbar:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            train_pbar.set_postfix({'loss': f"{loss.item():.4f}"})
            
        train_acc = accuracy_score(all_labels, all_preds)
        
        # 5. Validation
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_labels = []
        
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc="Validating"):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                val_preds.extend(predicted.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
                
        val_acc = accuracy_score(val_labels, val_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(val_labels, val_preds, average='weighted', zero_division=0)
        
        print(f"Epoch {epoch+1} Summary:")
        print(f"Train Loss: {running_loss/len(train_loader):.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val Loss  : {val_loss/len(val_loader):.4f} | Val Acc  : {val_acc:.4f}")
        print(f"Precision : {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}")
        
        # 6. Save Metrics to results/training_metrics.csv
        metrics_history.append({
            'round': epoch + 1, # 'round' for compatibility with app.py's view
            'loss': val_loss / len(val_loader),
            'accuracy': val_acc,
            'precision': precision,
            'recall': recall,
            'f1': f1
        })
        pd.DataFrame(metrics_history).to_csv("results/training_metrics.csv", index=False)
        
        # 7. Save Best Model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "weights/best_centralized_model.pth")
            print(f"✅ New best model saved with Acc: {best_acc:.4f}")
            
        scheduler.step(val_acc)

    print(f"\n🎉 Training Finished. Best Val Accuracy: {best_acc:.4f}")

if __name__ == "__main__":
    DATASET_PATH = "mri/Data"
    if os.path.exists(DATASET_PATH):
        train_mri_model(DATASET_PATH)
    else:
        print(f"❌ Error: Dataset path {DATASET_PATH} not found!")
