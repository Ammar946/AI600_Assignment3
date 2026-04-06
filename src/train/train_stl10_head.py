import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torchvision.models import ResNet18_Weights
from tqdm import tqdm

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.stl10 import get_stl10_dataloaders
from utils.utils import set_seed, count_parameters, AverageMeter

def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    loss_meter = AverageMeter()
    correct = 0
    total = 0
    
    for inputs, targets in tqdm(dataloader, desc="Training STL-10 Head", leave=False):
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        loss_meter.update(loss.item(), inputs.size(0))
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
    acc = 100. * correct / total
    return loss_meter.avg, acc

def eval_epoch(model, dataloader, criterion, device):
    model.eval()
    loss_meter = AverageMeter()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in tqdm(dataloader, desc="Evaluating", leave=False):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            loss_meter.update(loss.item(), inputs.size(0))
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
    acc = 100. * correct / total
    return loss_meter.avg, acc

def main():
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    os.makedirs('artifacts', exist_ok=True)
    
    train_loader, test_loader = get_stl10_dataloaders(batch_size=64)
    print("STL-10 Dataloaders initialized.")
    
    # 1. Load Pretrained ResNet-18
    model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    
    # 2. Freeze the backbone
    for param in model.parameters():
        param.requires_grad = False
        
    # 3. Replace classification head
    # The new linear layer will have requires_grad=True automatically
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 10)
    
    model = model.to(device)
    
    trainable_params = count_parameters(model)
    print(f"Num Trainable Parameters (Head only): {trainable_params}")
    
    criterion = nn.CrossEntropyLoss()
    # Note: Only passing model.fc.parameters() is slightly more robust
    optimizer = optim.Adam(model.fc.parameters(), lr=1e-3)
    
    num_epochs = 10
    best_test_acc = 0.0
    
    print("Starting Training (Head Only)...")
    for epoch in range(num_epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        test_loss, test_acc = eval_epoch(model, test_loader, criterion, device)
        
        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% - Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")
        
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            # Save the full model (useful for GradCAM later)
            torch.save(model.state_dict(), 'artifacts/best_stl10_resnet18.pth')
            print(f"Saved new best model with Test Acc {test_acc:.2f}%")

    print(f"\nFinal Best Test Accuracy: {best_test_acc:.2f}%")

if __name__ == '__main__':
    main()
