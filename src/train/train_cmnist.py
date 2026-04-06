import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.custom_cnn import CustomCNN
from data.cmnist import get_cmnist_dataloaders
from utils.utils import set_seed, AverageMeter

def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    loss_meter = AverageMeter()
    correct = 0
    total = 0
    
    for inputs, targets in tqdm(dataloader, desc="Training (C-MNIST)", leave=False):
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

def eval_epoch(model, dataloader, criterion, device, desc="Validating"):
    model.eval()
    loss_meter = AverageMeter()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in tqdm(dataloader, desc=desc, leave=False):
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
    
    # Dataloaders
    train_loader, test_biased_loader, test_unbiased_loader = get_cmnist_dataloaders(batch_size=64)
    print("C-MNIST Dataloaders initialized.")
    
    # Model (Note: in_channels=3 for RGB)
    model = CustomCNN(in_channels=3, num_classes=10).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    num_epochs = 10
    
    print("Training directly on C-MNIST Train (Biased) Set for 10 epochs...")
    for epoch in range(num_epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        
    torch.save(model.state_dict(), 'artifacts/cmnist_cnn.pth')
    
    print("\nTraining Complete. Evaluating...")
    
    # Evaluate Biased
    loss_b, acc_b = eval_epoch(model, test_biased_loader, criterion, device, desc="Eval Biased")
    print(f"Accuracy on Biased Test Set: {acc_b:.2f}%")
    
    # Evaluate Unbiased
    loss_u, acc_u = eval_epoch(model, test_unbiased_loader, criterion, device, desc="Eval Unbiased")
    print(f"Accuracy on Unbiased Test Set: {acc_u:.2f}%")
    
    print("\nNote: Compare the difference between biased and unbiased accuracy for Q1.3.")

if __name__ == '__main__':
    main()
