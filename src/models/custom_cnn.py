import torch
import torch.nn as nn
import torch.nn.functional as F

class CustomCNN(nn.Module):
    """
    A simple custom CNN to meet the following constraints:
    - Max 3 Conv Layers
    - Max 2 FC Layers
    - Max 50,000 parameters total
    """
    def __init__(self, in_channels: int = 1, num_classes: int = 10):
        super(CustomCNN, self).__init__()
        
        # Layer 1: Conv -> ReLU -> Pool
        # Input: in_channels x 28 x 28
        # Output: 16 x 14 x 14
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Layer 2: Conv -> ReLU -> Pool
        # Input: 16 x 14 x 14
        # Output: 32 x 7 x 7
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # We stop at 2 conv layers to stay well under params
        # Flatten size: 32 * 7 * 7 = 1568
        
        # Layer 3: FC -> ReLU
        # Input: 1568, Output: 16
        self.fc1 = nn.Linear(1568, 16)
        
        # Layer 4: FC (Output)
        # Input: 16, Output: num_classes
        self.fc2 = nn.Linear(16, num_classes)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        
        x = torch.flatten(x, 1) # Flatten all dimensions except batch
        
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x
