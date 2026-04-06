import os
import torch
import matplotlib.pyplot as plt
import sys
import numpy as np

# Adjust imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.custom_cnn import CustomCNN

def plot_first_layer_filters(model_path, save_path):
    print(f"Loading weights from {model_path}")
    
    model = CustomCNN(in_channels=1, num_classes=10)
    # Weights_only=True for safety
    model.load_state_dict(torch.load(model_path, weights_only=True, map_location='cpu'))
    
    # Extract weights of the first conv layer
    # model.conv1.weight has shape (out_channels, in_channels, kernel_height, kernel_width) -> (16, 1, 3, 3)
    filters = model.conv1.weight.detach().cpu().numpy()
    
    # Normalize filters for visualization
    f_min, f_max = filters.min(), filters.max()
    filters = (filters - f_min) / (f_max - f_min)
    
    n_filters = filters.shape[0]
    # We have 16 filters. Let's arrange them in a 4x4 grid.
    cols = 4
    rows = int(np.ceil(n_filters / cols))
    
    fig, axes = plt.subplots(rows, cols, figsize=(8, 8))
    axes = axes.flatten()
    
    for i in range(n_filters):
        # We only have 1 input channel, so we select index 0 for the in_channel
        # filter shape is (1, 3, 3). We squeeze to (3, 3).
        f = filters[i, 0, :, :]
        axes[i].imshow(f, cmap='PiYG')
        axes[i].axis('off')
        axes[i].set_title(f"Filter {i+1}", fontsize=10)
        
    # Hide any unused subplots
    for i in range(n_filters, len(axes)):
        axes[i].axis('off')
        
    plt.suptitle("First Layer Filters", fontsize=16)
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Saved filter visualizations to {save_path}")

if __name__ == '__main__':
    os.makedirs('artifacts', exist_ok=True)
    plot_first_layer_filters('artifacts/best_mnist_cnn.pth', 'artifacts/first_conv_filters.png')
