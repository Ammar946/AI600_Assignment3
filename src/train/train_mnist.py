import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt

# Adjust imports according to the package structure
import sys
# Add parent dir to path so we can import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.custom_cnn import CustomCNN
from data.mnist import get_mnist_dataloaders
from utils.utils import set_seed, count_parameters, AverageMeter

def plot_curves(train_losses, val_losses, train_accs, val_accs, save_path):
    epochs = range(1, len(train_losses) + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Loss plot
    ax1.plot(epochs, train_losses, 'b-', label='Train')
    ax1.plot(epochs, val_losses, 'r-', label='Val')
    ax1.set_title('Loss vs. Epochs')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Accuracy plot
    ax2.plot(epochs, train_accs, 'b-', label='Train')
    ax2.plot(epochs, val_accs, 'r-', label='Val')
    ax2.set_title('Accuracy vs. Epochs')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Saved learning curves to {save_path}")

def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    loss_meter = AverageMeter()
    correct = 0
    total = 0
    
    for inputs, targets in tqdm(dataloader, desc="Training", leave=False):
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
    # Setup
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create artifact dir
    os.makedirs('artifacts', exist_ok=True)
    
    # Dataloaders
    train_loader, val_loader, test_loader = get_mnist_dataloaders(batch_size=64)
    print("Dataloaders initialized.")
    
    # Model
    model = CustomCNN(in_channels=1, num_classes=10).to(device)
    params = count_parameters(model)
    print(f"Model initialized. Trainable parameters: {params}")
    assert params <= 50000, f"Error: Parameter count {params} exceeds 50,000 limit."
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    num_epochs = 10
    best_val_acc = 0.0
    
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    
    for epoch in range(num_epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = eval_epoch(model, val_loader, criterion, device)
        
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        print(f"Epoch {epoch+1}/{num_epochs} - "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% - "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'artifacts/best_mnist_cnn.pth')
            print(f"Saved new best model with Val Acc {val_acc:.2f}%")
            
    # Plot curves
    plot_curves(train_losses, val_losses, train_accs, val_accs, 'artifacts/mnist_learning_curves.png')
    
    # Test Evaluation
    print("\nEvaluating on Standard MNIST Test Set...")
    model.load_state_dict(torch.load('artifacts/best_mnist_cnn.pth', weights_only=True))
    test_loss, test_acc = eval_epoch(model, test_loader, criterion, device, desc="Testing")
    print(f"Final Test Accuracy: {test_acc:.2f}%")

if __name__ == '__main__':
    main()
