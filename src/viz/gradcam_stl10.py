import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from torchvision.models import ResNet18_Weights
import matplotlib.pyplot as plt
import numpy as np

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.stl10 import get_stl10_dataloaders

def denormalize(tensor_img):
    """Reverses the ImageNet normalization for visualization"""
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    img = tensor_img.numpy().transpose(1, 2, 0)
    img = std * img + mean
    img = np.clip(img, 0, 1)
    return img

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load Model
    model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 10)
    
    # Load best weights
    weight_path = 'artifacts/best_stl10_resnet18.pth'
    if not os.path.exists(weight_path):
        print(f"Error: Could not find {weight_path}. Make sure training completes first.")
        return
        
    model.load_state_dict(torch.load(weight_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    
    # Target layer for ResNet18 GradCAM
    target_layers = [model.layer4[-1]]
    
    # Create CAM object
    cam = GradCAM(model=model, target_layers=target_layers)
    
    # Load Data
    _, test_loader = get_stl10_dataloaders(batch_size=1)
    
    correct_samples = []
    incorrect_samples = []
    
    # Find 2 correct and 2 incorrect classifications
    print("Searching for 2 correct and 2 incorrect samples...")
    for inputs, targets in test_loader:
        if len(correct_samples) >= 2 and len(incorrect_samples) >= 2:
            break
            
        inputs_to_eval = inputs.to(device)
        targets_to_eval = targets.to(device)
        
        with torch.no_grad():
            outputs = model(inputs_to_eval)
            _, preds = outputs.max(1)
            
        is_correct = preds.item() == targets.item()
        
        # We need the original unnormalized image for the overlay later
        orig_img = denormalize(inputs[0])
        
        if is_correct and len(correct_samples) < 2:
            correct_samples.append((inputs_to_eval, orig_img, preds.item(), targets.item()))
        elif not is_correct and len(incorrect_samples) < 2:
            incorrect_samples.append((inputs_to_eval, orig_img, preds.item(), targets.item()))

    samples = correct_samples + incorrect_samples
    
    # Class names for STL-10
    classes = ['airplane', 'bird', 'car', 'cat', 'deer', 'dog', 'horse', 'monkey', 'ship', 'truck']
    
    # Setup plot
    fig, axes = plt.subplots(2, 4, figsize=(15, 8))
    
    print("Generating Heatmaps...")
    for i, (input_tensor, orig_img, pred_idx, true_idx) in enumerate(samples):
        # Generate GradCAM mask
        grayscale_cam = cam(input_tensor=input_tensor, targets=None)[0, :]
        
        # Overlay on original image
        visualization = show_cam_on_image(orig_img, grayscale_cam, use_rgb=True)
        
        # Raw Image Plot (top row)
        ax = axes[0, i]
        ax.imshow(orig_img)
        ax.axis('off')
        status = "Correct" if pred_idx == true_idx else "Incorrect"
        ax.set_title(f"[{status}]\nTrue: {classes[true_idx]}\nPred: {classes[pred_idx]}")
        
        # GradCAM Overlay Plot (bottom row)
        ax = axes[1, i]
        ax.imshow(visualization)
        ax.axis('off')
        ax.set_title("GradCAM Focus")
        
    plt.tight_layout()
    plt.savefig('artifacts/stl10_gradcam_overlays.png')
    print("Saved GradCAM overlays to artifacts/stl10_gradcam_overlays.png")

if __name__ == '__main__':
    main()
