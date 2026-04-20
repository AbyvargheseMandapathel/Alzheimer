import torch
import torch.nn as nn
from torchvision import models

class AlzheimerResNet(nn.Module):
    """Modern ResNet50-based model for high accuracy"""
    def __init__(self, num_classes=3, pretrained=True):
        super(AlzheimerResNet, self).__init__()
        # Load pre-trained ResNet50 for deeper feature extraction
        if pretrained:
            self.model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        else:
            self.model = models.resnet50(weights=None)
            
        # Unfreeze more layers (ResNet50 is deeper, unfreezing the last ~100 params
        # ensures layer3 and layer4 are trainable)
        for param in list(self.model.parameters())[:-100]:
            param.requires_grad = False
            
        # Modify the final classification layer
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(num_ftrs, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        return self.model(x)

    def get_last_conv_layer(self):
        """Helper method to get the last convolutional layer for Grad-CAM"""
        return self.model.layer4[-1].conv2

class AlzheimerResNet18(nn.Module):
    """Legacy ResNet18-based model for backward compatibility with 44MB weights"""
    def __init__(self, num_classes=3, pretrained=True):
        super(AlzheimerResNet18, self).__init__()
        if pretrained:
            self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        else:
            self.model = models.resnet18(weights=None)
            
        # The legacy weights expect a specific Sequential structure in fc:
        # Index 0: Dropout, Index 1: Linear(512, 3)
        num_ftrs = self.model.fc.in_features  # 512 for ResNet18
        self.model.fc = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(num_ftrs, num_classes)
        )
        
    def forward(self, x):
        return self.model(x)

    def get_last_conv_layer(self):
        """Helper method to get the last convolutional layer for Grad-CAM"""
        return self.model.layer4[-1].conv1


