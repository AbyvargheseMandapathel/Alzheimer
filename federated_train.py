import torch
import torch.nn as nn
import os
from data.dataset import get_federated_dataloaders
from models.cnn_model import AlzheimerResNet
from federated.client import FederatedClient
from federated.server import federated_average

def simulate_federated_learning(data_dir, num_rounds=10, num_clients=3, local_epochs=2, batch_size=32):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # 1. Initialize Global Model
    global_model = AlzheimerResNet(num_classes=3).to(device)
    
    # 2. Get client data loaders and global validation loader
    print(f"Splitting data for {num_clients} clients...")
    client_loaders, val_loader = get_federated_dataloaders(data_dir, num_clients=num_clients, batch_size=batch_size)
    
    # Initialize clients
    clients = [FederatedClient(client_id=i, dataloader=client_loaders[i], device=device) for i in range(num_clients)]
    
    criterion = nn.CrossEntropyLoss()
    os.makedirs("weights", exist_ok=True)
    best_acc = 0.0
    
    # Optimizer and Scheduler for the global model conceptually 
    # (actually used for local updates, but we track global stats)
    optimizer = torch.optim.Adam(global_model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    
    # 3. Federated Rounds
    for r in range(num_rounds):
        print(f"\n--- Round {r+1}/{num_rounds} ---")
        client_weights = []
        client_losses = []
        
        # Local Training -> Simulate parallel client training sequentially
        for client in clients:
            print(f"Training Client {client.client_id}...")
            weights, loss = client.train(global_model, local_epochs=local_epochs, lr=1e-4)
            client_weights.append(weights)
            client_losses.append(loss)
            
        print(f"Average Local Train Loss: {sum(client_losses)/len(client_losses):.4f}")
        
        # Aggregation
        print("Aggregating weights on Central Server...")
        avg_weights = federated_average(client_weights)
        
        # Update Global Model
        global_model.load_state_dict(avg_weights)
        
        # Evaluation step using Global validation dataset
        global_model.to(device)
        global_model.eval()
        
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                
                outputs = global_model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
        val_acc = 100 * val_correct / val_total
        print(f"Round {r+1} Global Val Loss: {val_loss/len(val_loader):.4f}, Global Val Acc: {val_acc:.2f}% (LR: {scheduler.get_last_lr()[0]:.6f})")
        
        # Step the scheduler
        scheduler.step()
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(global_model.state_dict(), "weights/best_federated_model.pth")
            print(f"=> Saved new optimal federated model with accuracy: {best_acc:.2f}%")

    print(f"\nFederated Training Complete. Best Val Accuracy: {best_acc:.2f}%")

if __name__ == "__main__":
    dataset_path = "mri/Data"
    # Target 50 rounds for high accuracy convergence
    simulate_federated_learning(dataset_path, num_rounds=50, num_clients=3, local_epochs=2)
