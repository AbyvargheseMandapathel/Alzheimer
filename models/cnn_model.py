import torch
import torch.nn as nn
from torchvision import models

class AlzheimerResNet(nn.Module):
    def __init__(self, num_classes=3, pretrained=True):
        super(AlzheimerResNet, self).__init__()
        # Load pre-trained ResNet18
        if pretrained:
            self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        else:
            self.model = models.resnet18(weights=None)
            
        # Unfreeze more layers to adapt to MRI specific features
        for param in list(self.model.parameters())[:-30]:
            param.requires_grad = False
            
        # Modify the final classification layer
        # Output features format: 3 classes (CN, MCI, AD)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_ftrs, num_classes)
        )
        
    def forward(self, x):
        return self.model(x)

    def get_last_conv_layer(self):
        """Helper method to get the last convolutional layer for Grad-CAM"""
        return self.model.layer4[-1].conv2
