import torch
import torch.nn as nn
import torch.optim as optim
import copy
from tqdm import tqdm

class FederatedClient:
    def __init__(self, client_id, dataloader, device="cuda"):
        self.client_id = client_id
        self.dataloader = dataloader
        self.device = device
        self.criterion = nn.CrossEntropyLoss()

    def train(self, global_model, local_epochs=1, lr=1e-4):
        """
        Trains the global model locally for `local_epochs` epochs.
        Returns the updated local weights and the local loss.
        """
        model = copy.deepcopy(global_model).to(self.device)
        model.train()
        
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        
        running_loss = 0.0
        
        for epoch in range(local_epochs):
             for images, labels in self.dataloader:
                images, labels = images.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(images)
                loss = self.criterion(outputs, labels)
                
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                
        # Calculate average loss over local epochs
        avg_loss = running_loss / (len(self.dataloader) * local_epochs)
        
        # Return state dict (weights)
        return model.cpu().state_dict(), avg_loss
