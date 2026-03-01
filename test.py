import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from data.dataset import get_dataloaders
from models.cnn_model import AlzheimerResNet

def evaluate_model(data_dir, model_path="weights/best_centralized_model.pth"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _, val_loader = get_dataloaders(data_dir, batch_size=32)
    
    model = AlzheimerResNet(num_classes=3).to(device)
    try:
        model.load_state_dict(torch.load(model_path))
        print(f"Loaded weights from {model_path}")
    except FileNotFoundError:
        print(f"Weights {model_path} not found. Ensure you have trained the model first.")
        return
        
    model.eval()
    
    all_preds = []
    all_labels = []
    
    print("Evaluating...")
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    # Calculate Metrics
    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, average='weighted')
    rec = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')
    
    print("\n--- Evaluation Metrics ---")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    
    # Plot Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    class_names = ['CN', 'MCI', 'AD']
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix based on Validation Set')
    plt.savefig('confusion_matrix.png')
    plt.show()
    
if __name__ == "__main__":
    dataset_path = "mri/Data"
    # Choose between centralized or federated evaluating
    # evaluate_model(dataset_path, "weights/best_centralized_model.pth")
    evaluate_model(dataset_path, "weights/best_federated_model.pth")
