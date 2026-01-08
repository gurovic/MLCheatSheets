#!/usr/bin/env python3
"""
Generate matplotlib illustrations for Neural Network Architecture cheatsheets.
This script creates high-quality visualizations and encodes them as base64 for inline HTML embedding.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle, FancyBboxPatch, FancyArrowPatch, Circle, Wedge, Arrow
from matplotlib.colors import LinearSegmentedColormap
import base64
from io import BytesIO
import warnings
warnings.filterwarnings('ignore')

# Set consistent style
sns.set_style("whitegrid")
DPI_VALUE = 300
plt.rcParams['figure.dpi'] = DPI_VALUE
plt.rcParams['savefig.dpi'] = DPI_VALUE
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9

def fig_to_base64(fig):
    """Convert matplotlib figure to base64 string."""
    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=DPI_VALUE)
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()
    plt.close(fig)
    return img_base64

# ============================================================================
# CNN ARCHITECTURES ILLUSTRATIONS
# ============================================================================

def generate_lenet_architecture():
    """Visualize LeNet-5 architecture."""
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 6)
    ax.axis('off')
    
    layers = [
        {'x': 0.5, 'y': 1.5, 'w': 1.2, 'h': 3.0, 'color': '#e3f2fd', 'label': 'Input\n32×32×1'},
        {'x': 2.2, 'y': 1.3, 'w': 1.1, 'h': 3.4, 'color': '#bbdefb', 'label': 'C1\n28×28×6'},
        {'x': 3.8, 'y': 1.8, 'w': 0.8, 'h': 2.4, 'color': '#90caf9', 'label': 'S2\n14×14×6'},
        {'x': 5.1, 'y': 1.6, 'w': 0.9, 'h': 2.8, 'color': '#64b5f6', 'label': 'C3\n10×10×16'},
        {'x': 6.5, 'y': 2.1, 'w': 0.7, 'h': 1.8, 'color': '#42a5f5', 'label': 'S4\n5×5×16'},
        {'x': 7.7, 'y': 2.0, 'w': 0.8, 'h': 2.0, 'color': '#2196f3', 'label': 'C5\n1×1×120'},
        {'x': 9.0, 'y': 2.5, 'w': 0.3, 'h': 1.0, 'color': '#ffa726', 'label': 'F6\n84'},
        {'x': 9.8, 'y': 2.5, 'w': 0.3, 'h': 1.0, 'color': '#ff9800', 'label': 'Output\n10'},
    ]
    
    for layer in layers:
        rect = FancyBboxPatch((layer['x'], layer['y']), layer['w'], layer['h'],
                             boxstyle="round,pad=0.05", 
                             facecolor=layer['color'], 
                             edgecolor='black', linewidth=1.5)
        ax.add_patch(rect)
        
        label_parts = layer['label'].split('\n')
        ax.text(layer['x'] + layer['w']/2, layer['y'] + layer['h']/2, 
               label_parts[0], ha='center', va='center', fontsize=9, fontweight='bold')
        if len(label_parts) > 1:
            ax.text(layer['x'] + layer['w']/2, layer['y'] + layer['h']/2 - 0.3, 
                   label_parts[1], ha='center', va='center', fontsize=8)
    
    for i in range(len(layers) - 1):
        x1 = layers[i]['x'] + layers[i]['w']
        y1 = layers[i]['y'] + layers[i]['h']/2
        x2 = layers[i+1]['x']
        y2 = layers[i+1]['y'] + layers[i+1]['h']/2
        
        arrow = FancyArrowPatch((x1, y1), (x2, y2),
                               arrowstyle='->', mutation_scale=15, 
                               linewidth=1.5, color='gray')
        ax.add_patch(arrow)
    
    ax.text(4.0, 0.5, 'Извлечение признаков', ha='center', fontsize=11, 
           fontweight='bold', color='#1976d2')
    ax.text(9.4, 0.5, 'Классификация', ha='center', fontsize=11, 
           fontweight='bold', color='#f57c00')
    
    plt.title('LeNet-5 Architecture (1998)', fontsize=14, fontweight='bold', pad=20)
    
    return fig_to_base64(fig)

def generate_vgg_block():
    """Visualize VGG block structure."""
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 7)
    ax.axis('off')
    
    # VGG Block
    blocks = [
        {'x': 1, 'y': 2, 'w': 2, 'h': 3, 'color': '#bbdefb', 'label': 'Conv 3×3\n64 filters\nReLU'},
        {'x': 3.5, 'y': 2, 'w': 2, 'h': 3, 'color': '#bbdefb', 'label': 'Conv 3×3\n64 filters\nReLU'},
        {'x': 6, 'y': 2.5, 'w': 1.5, 'h': 2, 'color': '#90caf9', 'label': 'MaxPool\n2×2'},
    ]
    
    for block in blocks:
        rect = FancyBboxPatch((block['x'], block['y']), block['w'], block['h'],
                             boxstyle="round,pad=0.1", 
                             facecolor=block['color'], 
                             edgecolor='black', linewidth=2)
        ax.add_patch(rect)
        
        for i, line in enumerate(block['label'].split('\n')):
            ax.text(block['x'] + block['w']/2, 
                   block['y'] + block['h']/2 + 0.3 - i*0.35, 
                   line, ha='center', va='center', fontsize=9, fontweight='bold')
    
    # Arrows
    arrow1 = FancyArrowPatch((3, 3.5), (3.5, 3.5),
                            arrowstyle='->', mutation_scale=20, 
                            linewidth=2, color='black')
    ax.add_patch(arrow1)
    
    arrow2 = FancyArrowPatch((5.5, 3.5), (6, 3.5),
                            arrowstyle='->', mutation_scale=20, 
                            linewidth=2, color='black')
    ax.add_patch(arrow2)
    
    ax.text(6, 1, 'VGG Block: Многократные Conv 3×3 + MaxPool', 
           ha='center', fontsize=12, fontweight='bold')
    
    plt.title('VGG Building Block', fontsize=14, fontweight='bold', pad=20)
    
    return fig_to_base64(fig)

def generate_alexnet_vs_vgg():
    """Compare AlexNet vs VGG architecture philosophies."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))
    
    # AlexNet
    ax1.set_xlim(0, 14)
    ax1.set_ylim(0, 6)
    ax1.axis('off')
    
    alex_layers = [
        {'x': 0.5, 'y': 1.5, 'w': 1.5, 'h': 3, 'color': '#e3f2fd', 'label': 'Input\n224×224×3'},
        {'x': 2.5, 'y': 1.2, 'w': 1.2, 'h': 3.6, 'color': '#bbdefb', 'label': 'Conv\n11×11\nstride=4'},
        {'x': 4.2, 'y': 1.6, 'w': 0.9, 'h': 2.8, 'color': '#90caf9', 'label': 'MaxPool\n3×3'},
        {'x': 5.6, 'y': 1.8, 'w': 1.0, 'h': 2.4, 'color': '#64b5f6', 'label': 'Conv\n5×5'},
        {'x': 7.1, 'y': 2.0, 'w': 0.8, 'h': 2.0, 'color': '#42a5f5', 'label': 'MaxPool\n3×3'},
        {'x': 8.4, 'y': 2.2, 'w': 0.9, 'h': 1.6, 'color': '#2196f3', 'label': 'Conv\n3×3'},
        {'x': 9.8, 'y': 2.5, 'w': 0.4, 'h': 1.0, 'color': '#ffa726', 'label': 'FC'},
        {'x': 10.7, 'y': 2.5, 'w': 0.4, 'h': 1.0, 'color': '#ff9800', 'label': 'FC'},
        {'x': 11.6, 'y': 2.5, 'w': 0.4, 'h': 1.0, 'color': '#fb8c00', 'label': 'Out'},
    ]
    
    for layer in alex_layers:
        rect = FancyBboxPatch((layer['x'], layer['y']), layer['w'], layer['h'],
                             boxstyle="round,pad=0.05", 
                             facecolor=layer['color'], 
                             edgecolor='black', linewidth=1.5)
        ax1.add_patch(rect)
        
        for i, line in enumerate(layer['label'].split('\n')):
            ax1.text(layer['x'] + layer['w']/2, 
                    layer['y'] + layer['h']/2 + 0.25 - i*0.3, 
                    line, ha='center', va='center', fontsize=8, fontweight='bold')
    
    for i in range(len(alex_layers) - 1):
        x1 = alex_layers[i]['x'] + alex_layers[i]['w']
        y1 = alex_layers[i]['y'] + alex_layers[i]['h']/2
        x2 = alex_layers[i+1]['x']
        y2 = alex_layers[i+1]['y'] + alex_layers[i+1]['h']/2
        arrow = FancyArrowPatch((x1, y1), (x2, y2),
                               arrowstyle='->', mutation_scale=12, 
                               linewidth=1.2, color='gray')
        ax1.add_patch(arrow)
    
    ax1.text(7, 0.3, 'AlexNet: Большие фильтры (11×11, 5×5)', 
            ha='center', fontsize=12, fontweight='bold', color='#1976d2')
    
    # VGG
    ax2.set_xlim(0, 14)
    ax2.set_ylim(0, 6)
    ax2.axis('off')
    
    vgg_layers = [
        {'x': 0.5, 'y': 1.5, 'w': 1.5, 'h': 3, 'color': '#e3f2fd', 'label': 'Input\n224×224×3'},
        {'x': 2.5, 'y': 1.5, 'w': 0.8, 'h': 3, 'color': '#bbdefb', 'label': 'Conv\n3×3'},
        {'x': 3.8, 'y': 1.5, 'w': 0.8, 'h': 3, 'color': '#bbdefb', 'label': 'Conv\n3×3'},
        {'x': 5.1, 'y': 1.9, 'w': 0.6, 'h': 2.2, 'color': '#90caf9', 'label': 'Pool\n2×2'},
        {'x': 6.2, 'y': 1.9, 'w': 0.7, 'h': 2.2, 'color': '#64b5f6', 'label': 'Conv\n3×3'},
        {'x': 7.4, 'y': 1.9, 'w': 0.7, 'h': 2.2, 'color': '#64b5f6', 'label': 'Conv\n3×3'},
        {'x': 8.6, 'y': 2.2, 'w': 0.5, 'h': 1.6, 'color': '#42a5f5', 'label': 'Pool\n2×2'},
        {'x': 9.6, 'y': 2.5, 'w': 0.4, 'h': 1.0, 'color': '#ffa726', 'label': 'FC'},
        {'x': 10.5, 'y': 2.5, 'w': 0.4, 'h': 1.0, 'color': '#ff9800', 'label': 'FC'},
        {'x': 11.4, 'y': 2.5, 'w': 0.4, 'h': 1.0, 'color': '#fb8c00', 'label': 'Out'},
    ]
    
    for layer in vgg_layers:
        rect = FancyBboxPatch((layer['x'], layer['y']), layer['w'], layer['h'],
                             boxstyle="round,pad=0.05", 
                             facecolor=layer['color'], 
                             edgecolor='black', linewidth=1.5)
        ax2.add_patch(rect)
        
        for i, line in enumerate(layer['label'].split('\n')):
            ax2.text(layer['x'] + layer['w']/2, 
                    layer['y'] + layer['h']/2 + 0.15 - i*0.25, 
                    line, ha='center', va='center', fontsize=8, fontweight='bold')
    
    for i in range(len(vgg_layers) - 1):
        x1 = vgg_layers[i]['x'] + vgg_layers[i]['w']
        y1 = vgg_layers[i]['y'] + vgg_layers[i]['h']/2
        x2 = vgg_layers[i+1]['x']
        y2 = vgg_layers[i+1]['y'] + vgg_layers[i+1]['h']/2
        arrow = FancyArrowPatch((x1, y1), (x2, y2),
                               arrowstyle='->', mutation_scale=12, 
                               linewidth=1.2, color='gray')
        ax2.add_patch(arrow)
    
    ax2.text(7, 0.3, 'VGG: Только маленькие фильтры (3×3), больше глубины', 
            ha='center', fontsize=12, fontweight='bold', color='#1976d2')
    
    plt.suptitle('AlexNet vs VGG: Эволюция CNN архитектур', fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    return fig_to_base64(fig)

# ============================================================================
# RESNET ILLUSTRATIONS
# ============================================================================

def generate_residual_block():
    """Visualize ResNet residual block."""
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Main path
    main_blocks = [
        {'x': 4, 'y': 7.5, 'w': 2, 'h': 0.8, 'color': '#bbdefb', 'label': 'Conv 3×3'},
        {'x': 4, 'y': 6.2, 'w': 2, 'h': 0.6, 'color': '#e8f5e9', 'label': 'BatchNorm'},
        {'x': 4, 'y': 5.2, 'w': 2, 'h': 0.6, 'color': '#fff9c4', 'label': 'ReLU'},
        {'x': 4, 'y': 4.0, 'w': 2, 'h': 0.8, 'color': '#bbdefb', 'label': 'Conv 3×3'},
        {'x': 4, 'y': 2.7, 'w': 2, 'h': 0.6, 'color': '#e8f5e9', 'label': 'BatchNorm'},
    ]
    
    for block in main_blocks:
        rect = FancyBboxPatch((block['x'], block['y']), block['w'], block['h'],
                             boxstyle="round,pad=0.05", 
                             facecolor=block['color'], 
                             edgecolor='black', linewidth=2)
        ax.add_patch(rect)
        ax.text(block['x'] + block['w']/2, block['y'] + block['h']/2, 
               block['label'], ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Draw arrows for main path
    for i in range(len(main_blocks) - 1):
        y1 = main_blocks[i]['y']
        y2 = main_blocks[i+1]['y'] + main_blocks[i+1]['h']
        ax.arrow(5, y1, 0, y2 - y1 + 0.1, head_width=0.2, head_length=0.1, 
                fc='black', ec='black', linewidth=2)
    
    # Skip connection (identity)
    ax.plot([2, 2, 8, 8], [8.5, 1.5, 1.5, 2.2], 'r-', linewidth=3, label='Skip Connection')
    ax.arrow(8, 2.2, 0, 0.3, head_width=0.2, head_length=0.1, 
            fc='red', ec='red', linewidth=3)
    
    # Addition
    addition_circle = Circle((5, 1.5), 0.4, facecolor='#ffccbc', edgecolor='black', linewidth=2)
    ax.add_patch(addition_circle)
    ax.text(5, 1.5, '+', ha='center', va='center', fontsize=16, fontweight='bold')
    
    # Input/Output labels
    ax.text(5, 9, 'Input x', ha='center', fontsize=11, fontweight='bold',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax.text(2, 5, 'Identity\nShortcut', ha='center', fontsize=10, fontweight='bold', color='red')
    
    # Final ReLU
    final_relu = FancyBboxPatch((4, 0.3), 2, 0.6,
                               boxstyle="round,pad=0.05", 
                               facecolor='#fff9c4', 
                               edgecolor='black', linewidth=2)
    ax.add_patch(final_relu)
    ax.text(5, 0.6, 'ReLU', ha='center', va='center', fontsize=10, fontweight='bold')
    
    ax.arrow(5, 1.1, 0, -0.5, head_width=0.2, head_length=0.1, 
            fc='black', ec='black', linewidth=2)
    
    ax.text(5, -0.3, 'Output F(x) + x', ha='center', fontsize=11, fontweight='bold',
           bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
    
    plt.title('ResNet Residual Block: y = F(x) + x', fontsize=14, fontweight='bold', pad=20)
    
    return fig_to_base64(fig)

def generate_resnet_degradation():
    """Visualize the degradation problem that ResNet solves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plain network - degradation problem
    depths = [18, 34, 50, 101, 152]
    plain_train_error = [28, 25, 24, 26, 29]  # Error increases with depth
    plain_test_error = [30, 28, 27, 30, 34]
    
    ax1.plot(depths, plain_train_error, 'o-', linewidth=2, markersize=8, 
            label='Training Error', color='#2196f3')
    ax1.plot(depths, plain_test_error, 's-', linewidth=2, markersize=8, 
            label='Test Error', color='#f44336')
    ax1.set_xlabel('Количество слоев', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Ошибка (%)', fontsize=11, fontweight='bold')
    ax1.set_title('Plain Network\n(Проблема деградации)', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([20, 36])
    
    # ResNet - no degradation
    resnet_train_error = [25, 22, 19, 17, 16]  # Continues to improve
    resnet_test_error = [27, 24, 21, 20, 19]
    
    ax2.plot(depths, resnet_train_error, 'o-', linewidth=2, markersize=8, 
            label='Training Error', color='#2196f3')
    ax2.plot(depths, resnet_test_error, 's-', linewidth=2, markersize=8, 
            label='Test Error', color='#f44336')
    ax2.set_xlabel('Количество слоев', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Ошибка (%)', fontsize=11, fontweight='bold')
    ax2.set_title('ResNet\n(Решение с Skip Connections)', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([14, 30])
    
    plt.suptitle('ResNet: Решение проблемы деградации глубоких сетей', 
                fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    return fig_to_base64(fig)

def generate_skip_connections():
    """Visualize different types of skip connections."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    
    for idx, (ax, title, connection_type) in enumerate(zip(axes, 
        ['Identity Shortcut', 'Projection Shortcut', 'Dense Connections'],
        ['identity', 'projection', 'dense'])):
        
        ax.set_xlim(0, 6)
        ax.set_ylim(0, 8)
        ax.axis('off')
        ax.set_title(title, fontsize=12, fontweight='bold')
        
        # Layers
        layers_y = [6, 4.5, 3, 1.5]
        for i, y in enumerate(layers_y):
            rect = FancyBboxPatch((2, y-0.4), 2, 0.8,
                                 boxstyle="round,pad=0.05", 
                                 facecolor='#bbdefb', 
                                 edgecolor='black', linewidth=1.5)
            ax.add_patch(rect)
            ax.text(3, y, f'Layer {i+1}', ha='center', va='center', 
                   fontsize=9, fontweight='bold')
        
        # Draw connections
        if connection_type == 'identity':
            # Skip every other layer
            ax.plot([5, 5], [5.6, 1.1], 'r-', linewidth=2.5)
            ax.arrow(5, 1.1, -0.3, 0, head_width=0.2, head_length=0.15, 
                    fc='red', ec='red', linewidth=2.5)
            ax.text(5.3, 3.5, 'Skip', ha='left', fontsize=9, color='red', fontweight='bold')
            
        elif connection_type == 'projection':
            # Skip with 1x1 conv
            ax.plot([5, 5], [5.6, 2.5], 'r-', linewidth=2.5)
            proj_box = FancyBboxPatch((4.5, 3.5), 1, 0.5,
                                     boxstyle="round,pad=0.05", 
                                     facecolor='#ffccbc', 
                                     edgecolor='red', linewidth=2)
            ax.add_patch(proj_box)
            ax.text(5, 3.75, '1×1 Conv', ha='center', va='center', 
                   fontsize=8, fontweight='bold')
            ax.arrow(5, 2.5, -0.3, -0.6, head_width=0.2, head_length=0.15, 
                    fc='red', ec='red', linewidth=2.5)
            
        else:  # dense
            # Connect to all previous layers
            for i in range(len(layers_y)):
                for j in range(i):
                    y1 = layers_y[j] + 0.4
                    y2 = layers_y[i] - 0.4
                    ax.plot([1.5, 1.5], [y1, y2], 'r-', linewidth=1.5, alpha=0.6)
                    ax.arrow(1.5, y2, 0.3, 0, head_width=0.15, head_length=0.1, 
                            fc='red', ec='red', linewidth=1.5, alpha=0.6)
        
        # Main path arrows
        for i in range(len(layers_y) - 1):
            y1 = layers_y[i] - 0.4
            y2 = layers_y[i+1] + 0.4
            ax.arrow(3, y1, 0, y2 - y1 + 0.05, head_width=0.2, head_length=0.1, 
                    fc='black', ec='black', linewidth=1.5)
    
    plt.suptitle('Типы Skip Connections в глубоких сетях', 
                fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    return fig_to_base64(fig)


# ============================================================================
# INCEPTION & EFFICIENTNET ILLUSTRATIONS
# ============================================================================

def generate_inception_module():
    """Visualize Inception module structure."""
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Input
    input_box = FancyBboxPatch((5, 8.5), 2, 0.8,
                              boxstyle="round,pad=0.1", 
                              facecolor='#e3f2fd', 
                              edgecolor='black', linewidth=2)
    ax.add_patch(input_box)
    ax.text(6, 8.9, 'Input', ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Four parallel paths
    paths = [
        {'x': 0.5, 'label': '1×1', 'color': '#bbdefb'},
        {'x': 3.5, 'label': '1×1→3×3', 'color': '#90caf9'},
        {'x': 6.5, 'label': '1×1→5×5', 'color': '#64b5f6'},
        {'x': 9.5, 'label': '3×3 pool→1×1', 'color': '#42a5f5'},
    ]
    
    for path in paths:
        # Arrow from input
        ax.arrow(6, 8.5, path['x'] + 1 - 6, -1.5, head_width=0.15, head_length=0.1, 
                fc='gray', ec='gray', linewidth=1.5, alpha=0.7)
        
        if path['label'] == '1×1':
            # Single conv
            rect = FancyBboxPatch((path['x'], 5.5), 2, 1,
                                 boxstyle="round,pad=0.1", 
                                 facecolor=path['color'], 
                                 edgecolor='black', linewidth=1.5)
            ax.add_patch(rect)
            ax.text(path['x'] + 1, 6, 'Conv\n1×1', ha='center', va='center', 
                   fontsize=9, fontweight='bold')
        elif '→' in path['label']:
            # Two convs
            parts = path['label'].split('→')
            rect1 = FancyBboxPatch((path['x'], 6.2), 2, 0.6,
                                  boxstyle="round,pad=0.05", 
                                  facecolor=path['color'], 
                                  edgecolor='black', linewidth=1.5)
            ax.add_patch(rect1)
            ax.text(path['x'] + 1, 6.5, f'Conv\n{parts[0]}', ha='center', va='center', 
                   fontsize=8, fontweight='bold')
            
            ax.arrow(path['x'] + 1, 6.2, 0, -0.3, head_width=0.15, head_length=0.08, 
                    fc='black', ec='black', linewidth=1.2)
            
            rect2 = FancyBboxPatch((path['x'], 5.0), 2, 0.6,
                                  boxstyle="round,pad=0.05", 
                                  facecolor=path['color'], 
                                  edgecolor='black', linewidth=1.5)
            ax.add_patch(rect2)
            ax.text(path['x'] + 1, 5.3, f'Conv\n{parts[1]}', ha='center', va='center', 
                   fontsize=8, fontweight='bold')
        else:
            # Pool + conv
            rect1 = FancyBboxPatch((path['x'], 6.2), 2, 0.6,
                                  boxstyle="round,pad=0.05", 
                                  facecolor='#90caf9', 
                                  edgecolor='black', linewidth=1.5)
            ax.add_patch(rect1)
            ax.text(path['x'] + 1, 6.5, 'MaxPool\n3×3', ha='center', va='center', 
                   fontsize=8, fontweight='bold')
            
            ax.arrow(path['x'] + 1, 6.2, 0, -0.3, head_width=0.15, head_length=0.08, 
                    fc='black', ec='black', linewidth=1.2)
            
            rect2 = FancyBboxPatch((path['x'], 5.0), 2, 0.6,
                                  boxstyle="round,pad=0.05", 
                                  facecolor=path['color'], 
                                  edgecolor='black', linewidth=1.5)
            ax.add_patch(rect2)
            ax.text(path['x'] + 1, 5.3, 'Conv\n1×1', ha='center', va='center', 
                   fontsize=8, fontweight='bold')
        
        # Arrow to concat
        y_start = 5.0 if '→' in path['label'] or 'pool' in path['label'] else 5.5
        ax.arrow(path['x'] + 1, y_start, 6 - (path['x'] + 1), -1.5, 
                head_width=0.15, head_length=0.1, 
                fc='gray', ec='gray', linewidth=1.5, alpha=0.7)
    
    # Concatenate
    concat_box = FancyBboxPatch((4.5, 2.5), 3, 0.8,
                               boxstyle="round,pad=0.1", 
                               facecolor='#ffeb3b', 
                               edgecolor='black', linewidth=2)
    ax.add_patch(concat_box)
    ax.text(6, 2.9, 'Filter Concatenation', ha='center', va='center', 
           fontsize=10, fontweight='bold')
    
    # Output
    ax.arrow(6, 2.5, 0, -0.5, head_width=0.2, head_length=0.1, 
            fc='black', ec='black', linewidth=2)
    
    output_box = FancyBboxPatch((5, 1.2), 2, 0.6,
                               boxstyle="round,pad=0.1", 
                               facecolor='#c8e6c9', 
                               edgecolor='black', linewidth=2)
    ax.add_patch(output_box)
    ax.text(6, 1.5, 'Output', ha='center', va='center', fontsize=10, fontweight='bold')
    
    ax.text(0.5, 9.5, 'Параллельные свертки разных размеров', 
           ha='left', fontsize=10, fontweight='bold', style='italic')
    
    plt.title('Inception Module: Multi-scale Feature Extraction', 
             fontsize=14, fontweight='bold', pad=20)
    
    return fig_to_base64(fig)

def generate_efficientnet_scaling():
    """Visualize EfficientNet compound scaling."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Baseline network
    ax = axes[0, 0]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    ax.set_title('Baseline Network', fontsize=12, fontweight='bold')
    
    # Draw simple network
    for i, y in enumerate([8, 6, 4, 2]):
        rect = FancyBboxPatch((3, y-0.5), 4, 1,
                             boxstyle="round,pad=0.1", 
                             facecolor='#bbdefb', 
                             edgecolor='black', linewidth=1.5)
        ax.add_patch(rect)
        ax.text(5, y, f'Layer {i+1}', ha='center', va='center', fontsize=9, fontweight='bold')
        if i < 3:
            ax.arrow(5, y-0.5, 0, -0.8, head_width=0.3, head_length=0.15, 
                    fc='black', ec='black', linewidth=1.5)
    
    ax.text(5, 0.5, 'd=1.0, w=1.0, r=1.0', ha='center', fontsize=9, style='italic')
    
    # Depth scaling
    ax = axes[0, 1]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    ax.set_title('Depth Scaling (d↑)', fontsize=12, fontweight='bold', color='#f44336')
    
    for i, y in enumerate([9, 7.5, 6, 4.5, 3, 1.5]):
        rect = FancyBboxPatch((3, y-0.4), 4, 0.8,
                             boxstyle="round,pad=0.1", 
                             facecolor='#bbdefb', 
                             edgecolor='black', linewidth=1.5)
        ax.add_patch(rect)
        ax.text(5, y, f'Layer {i+1}', ha='center', va='center', fontsize=8, fontweight='bold')
        if i < 5:
            ax.arrow(5, y-0.4, 0, -0.6, head_width=0.3, head_length=0.15, 
                    fc='black', ec='black', linewidth=1.5)
    
    ax.text(5, 0.5, 'Больше слоёв', ha='center', fontsize=9, style='italic', color='#f44336')
    
    # Width scaling
    ax = axes[1, 0]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    ax.set_title('Width Scaling (w↑)', fontsize=12, fontweight='bold', color='#4caf50')
    
    for i, y in enumerate([8, 6, 4, 2]):
        rect = FancyBboxPatch((2, y-0.5), 6, 1,
                             boxstyle="round,pad=0.1", 
                             facecolor='#bbdefb', 
                             edgecolor='black', linewidth=1.5)
        ax.add_patch(rect)
        ax.text(5, y, f'Layer {i+1} (wider)', ha='center', va='center', fontsize=9, fontweight='bold')
        if i < 3:
            ax.arrow(5, y-0.5, 0, -0.8, head_width=0.3, head_length=0.15, 
                    fc='black', ec='black', linewidth=1.5)
    
    ax.text(5, 0.5, 'Больше каналов', ha='center', fontsize=9, style='italic', color='#4caf50')
    
    # Resolution scaling
    ax = axes[1, 1]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    ax.set_title('Resolution Scaling (r↑)', fontsize=12, fontweight='bold', color='#2196f3')
    
    # Show increasing input size
    sizes = [2, 3, 4, 3]
    for i, (y, size) in enumerate(zip([8, 6, 4, 2], sizes)):
        x_offset = (4 - size) / 2
        rect = FancyBboxPatch((3 + x_offset, y-0.5), size, 1,
                             boxstyle="round,pad=0.1", 
                             facecolor='#bbdefb', 
                             edgecolor='black', linewidth=1.5)
        ax.add_patch(rect)
        ax.text(5, y, f'Layer {i+1}', ha='center', va='center', fontsize=9, fontweight='bold')
        if i < 3:
            ax.arrow(5, y-0.5, 0, -0.8, head_width=0.3, head_length=0.15, 
                    fc='black', ec='black', linewidth=1.5)
    
    ax.text(5, 0.5, 'Выше разрешение входа', ha='center', fontsize=9, style='italic', color='#2196f3')
    
    plt.suptitle('EfficientNet: Compound Scaling (d × w × r)', 
                fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    return fig_to_base64(fig)

# ============================================================================
# TRANSFORMER ILLUSTRATIONS
# ============================================================================

def generate_attention_mechanism():
    """Visualize attention mechanism with query, key, value."""
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Input sequence
    input_words = ['The', 'cat', 'sat', 'on', 'mat']
    for i, word in enumerate(input_words):
        x = 1 + i * 2
        rect = FancyBboxPatch((x-0.4, 8.5), 0.8, 0.8,
                             boxstyle="round,pad=0.05", 
                             facecolor='#e3f2fd', 
                             edgecolor='black', linewidth=1.5)
        ax.add_patch(rect)
        ax.text(x, 8.9, word, ha='center', va='center', fontsize=9, fontweight='bold')
    
    # Q, K, V transformations
    qkv_y = 6.5
    qkv_labels = ['Query (Q)', 'Key (K)', 'Value (V)']
    qkv_colors = ['#ffcdd2', '#c5e1a5', '#bbdefb']
    
    for i, (label, color) in enumerate(zip(qkv_labels, qkv_colors)):
        x = 2 + i * 3.5
        rect = FancyBboxPatch((x-0.8, qkv_y-0.4), 1.6, 0.8,
                             boxstyle="round,pad=0.05", 
                             facecolor=color, 
                             edgecolor='black', linewidth=1.5)
        ax.add_patch(rect)
        ax.text(x, qkv_y, label, ha='center', va='center', fontsize=9, fontweight='bold')
        
        # Arrows from input
        ax.arrow(5, 8.5, x - 5, -1.3, head_width=0.15, head_length=0.1, 
                fc='gray', ec='gray', linewidth=1.2, alpha=0.6)
    
    # Attention scores computation
    score_box = FancyBboxPatch((3, 4.5), 6, 1,
                              boxstyle="round,pad=0.1", 
                              facecolor='#fff9c4', 
                              edgecolor='black', linewidth=2)
    ax.add_patch(score_box)
    ax.text(6, 5.3, 'Attention(Q, K, V) = softmax(QK^T / √d_k) V', 
           ha='center', va='center', fontsize=10, fontweight='bold')
    ax.text(6, 4.8, 'Вычисление весов внимания', 
           ha='center', va='center', fontsize=9, style='italic')
    
    # Arrows to attention
    for x in [2, 5.5, 9]:
        ax.arrow(x, qkv_y-0.4, 6 - x, -1.4, head_width=0.15, head_length=0.1, 
                fc='black', ec='black', linewidth=1.5, alpha=0.7)
    
    # Attention weights visualization
    ax.text(1, 3, 'Веса внимания:', ha='left', fontsize=9, fontweight='bold')
    
    # Heatmap-like representation
    words_short = ['The', 'cat', 'sat']
    for i, word in enumerate(words_short):
        ax.text(2.5 + i * 1.5, 2.5, word, ha='center', fontsize=8, fontweight='bold')
        ax.text(1, 2 - i * 0.4, word, ha='right', fontsize=8, fontweight='bold')
    
    # Simplified attention matrix
    attention_weights = np.array([[0.1, 0.3, 0.6],
                                  [0.2, 0.7, 0.1],
                                  [0.3, 0.2, 0.5]])
    
    for i in range(3):
        for j in range(3):
            alpha = attention_weights[i, j]
            rect = Rectangle((2.5 + j * 1.5 - 0.4, 2 - i * 0.4 - 0.15), 0.8, 0.3,
                           facecolor=plt.cm.Blues(alpha), 
                           edgecolor='black', linewidth=0.5)
            ax.add_patch(rect)
            ax.text(2.5 + j * 1.5, 2 - i * 0.4, f'{alpha:.1f}',
                   ha='center', va='center', fontsize=7)
    
    # Output
    ax.arrow(6, 4.5, 0, -1, head_width=0.2, head_length=0.1, 
            fc='black', ec='black', linewidth=2)
    
    output_box = FancyBboxPatch((4.5, 0.2), 3, 0.6,
                               boxstyle="round,pad=0.1", 
                               facecolor='#c8e6c9', 
                               edgecolor='black', linewidth=2)
    ax.add_patch(output_box)
    ax.text(6, 0.5, 'Context-aware Output', ha='center', va='center', 
           fontsize=10, fontweight='bold')
    
    plt.title('Attention Mechanism: Query, Key, Value', fontsize=14, fontweight='bold', pad=20)
    
    return fig_to_base64(fig)

def generate_multi_head_attention():
    """Visualize multi-head attention."""
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Input
    input_box = FancyBboxPatch((6, 8.5), 2, 0.8,
                              boxstyle="round,pad=0.1", 
                              facecolor='#e3f2fd', 
                              edgecolor='black', linewidth=2)
    ax.add_patch(input_box)
    ax.text(7, 8.9, 'Input Embeddings', ha='center', va='center', 
           fontsize=10, fontweight='bold')
    
    # Multiple attention heads
    n_heads = 4
    colors = ['#ffcdd2', '#c5e1a5', '#bbdefb', '#fff9c4']
    
    for i in range(n_heads):
        x = 1.5 + i * 3
        
        # Head box
        head_box = FancyBboxPatch((x-0.8, 5), 2, 2.5,
                                 boxstyle="round,pad=0.1", 
                                 facecolor=colors[i], 
                                 edgecolor='black', linewidth=1.5, alpha=0.7)
        ax.add_patch(head_box)
        ax.text(x + 0.2, 7.2, f'Head {i+1}', ha='center', va='top', 
               fontsize=9, fontweight='bold')
        
        # Q, K, V inside head
        for j, label in enumerate(['Q', 'K', 'V']):
            y = 6.5 - j * 0.6
            mini_box = FancyBboxPatch((x-0.6, y-0.2), 0.4, 0.4,
                                     boxstyle="round,pad=0.02", 
                                     facecolor='white', 
                                     edgecolor='black', linewidth=1)
            ax.add_patch(mini_box)
            ax.text(x-0.4, y, label, ha='center', va='center', fontsize=7, fontweight='bold')
        
        # Attention operation
        ax.text(x + 0.2, 5.5, 'Attention', ha='center', va='center', 
               fontsize=8, style='italic')
        
        # Arrow from input
        ax.arrow(7, 8.5, x + 0.2 - 7, -1, head_width=0.15, head_length=0.1, 
                fc='gray', ec='gray', linewidth=1.2, alpha=0.6)
        
        # Arrow to concat
        ax.arrow(x + 0.2, 5, 7 - (x + 0.2), -1.5, head_width=0.15, head_length=0.1, 
                fc='gray', ec='gray', linewidth=1.2, alpha=0.6)
    
    # Concatenate
    concat_box = FancyBboxPatch((5, 2.5), 4, 0.8,
                               boxstyle="round,pad=0.1", 
                               facecolor='#ffeb3b', 
                               edgecolor='black', linewidth=2)
    ax.add_patch(concat_box)
    ax.text(7, 2.9, 'Concatenate Heads', ha='center', va='center', 
           fontsize=10, fontweight='bold')
    
    # Linear projection
    ax.arrow(7, 2.5, 0, -0.4, head_width=0.2, head_length=0.1, 
            fc='black', ec='black', linewidth=2)
    
    linear_box = FancyBboxPatch((5.5, 1.2), 3, 0.6,
                               boxstyle="round,pad=0.1", 
                               facecolor='#ce93d8', 
                               edgecolor='black', linewidth=2)
    ax.add_patch(linear_box)
    ax.text(7, 1.5, 'Linear Projection', ha='center', va='center', 
           fontsize=10, fontweight='bold')
    
    # Output
    ax.arrow(7, 1.2, 0, -0.3, head_width=0.2, head_length=0.1, 
            fc='black', ec='black', linewidth=2)
    
    output_box = FancyBboxPatch((6, 0.2), 2, 0.5,
                               boxstyle="round,pad=0.1", 
                               facecolor='#c8e6c9', 
                               edgecolor='black', linewidth=2)
    ax.add_patch(output_box)
    ax.text(7, 0.45, 'Output', ha='center', va='center', fontsize=10, fontweight='bold')
    
    ax.text(0.5, 9.5, f'Multi-Head Attention: {n_heads} параллельных механизмов внимания', 
           ha='left', fontsize=11, fontweight='bold', style='italic')
    
    plt.title('Multi-Head Attention Architecture', fontsize=14, fontweight='bold', pad=20)
    
    return fig_to_base64(fig)


def generate_transformer_architecture():
    """Visualize complete transformer encoder-decoder architecture."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 10))
    
    # Encoder
    ax1.set_xlim(0, 6)
    ax1.set_ylim(0, 12)
    ax1.axis('off')
    ax1.set_title('Encoder', fontsize=13, fontweight='bold')
    
    encoder_components = [
        {'y': 10.5, 'label': 'Input\nEmbedding', 'color': '#e3f2fd'},
        {'y': 9.2, 'label': 'Positional\nEncoding', 'color': '#fff9c4'},
        {'y': 7.5, 'label': 'Multi-Head\nAttention', 'color': '#bbdefb'},
        {'y': 6.2, 'label': 'Add & Norm', 'color': '#c8e6c9'},
        {'y': 4.9, 'label': 'Feed\nForward', 'color': '#ce93d8'},
        {'y': 3.6, 'label': 'Add & Norm', 'color': '#c8e6c9'},
    ]
    
    for comp in encoder_components:
        rect = FancyBboxPatch((1.5, comp['y']-0.5), 3, 0.9,
                             boxstyle="round,pad=0.1", 
                             facecolor=comp['color'], 
                             edgecolor='black', linewidth=1.5)
        ax1.add_patch(rect)
        ax1.text(3, comp['y'], comp['label'], ha='center', va='center', 
                fontsize=9, fontweight='bold')
        
    # Arrows
    for i in range(len(encoder_components) - 1):
        y1 = encoder_components[i]['y'] - 0.5
        y2 = encoder_components[i+1]['y'] + 0.4
        ax1.arrow(3, y1, 0, y2 - y1 + 0.05, head_width=0.25, head_length=0.1, 
                 fc='black', ec='black', linewidth=1.5)
    
    # Skip connections
    ax1.plot([0.8, 0.8, 4.7, 4.7], [7.9, 6.7, 6.7, 6.2], 'r--', linewidth=2, alpha=0.6)
    ax1.plot([0.8, 0.8, 4.7, 4.7], [6.6, 5.4, 5.4, 4.9], 'r--', linewidth=2, alpha=0.6)
    
    ax1.text(3, 2.8, '× N layers', ha='center', fontsize=10, 
            style='italic', fontweight='bold')
    ax1.text(3, 1.5, 'Context для\nDecoder', ha='center', fontsize=9, 
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Decoder
    ax2.set_xlim(0, 6)
    ax2.set_ylim(0, 12)
    ax2.axis('off')
    ax2.set_title('Decoder', fontsize=13, fontweight='bold')
    
    decoder_components = [
        {'y': 10.5, 'label': 'Output\nEmbedding', 'color': '#e3f2fd'},
        {'y': 9.2, 'label': 'Positional\nEncoding', 'color': '#fff9c4'},
        {'y': 7.8, 'label': 'Masked\nMulti-Head\nAttention', 'color': '#ffcdd2'},
        {'y': 6.5, 'label': 'Add & Norm', 'color': '#c8e6c9'},
        {'y': 5.2, 'label': 'Cross\nAttention', 'color': '#bbdefb'},
        {'y': 3.9, 'label': 'Add & Norm', 'color': '#c8e6c9'},
        {'y': 2.6, 'label': 'Feed\nForward', 'color': '#ce93d8'},
        {'y': 1.3, 'label': 'Add & Norm', 'color': '#c8e6c9'},
    ]
    
    for comp in decoder_components:
        height = 0.9 if 'Masked' not in comp['label'] else 1.1
        y_offset = 0 if 'Masked' not in comp['label'] else -0.1
        rect = FancyBboxPatch((1.5, comp['y']-0.5+y_offset), 3, height,
                             boxstyle="round,pad=0.1", 
                             facecolor=comp['color'], 
                             edgecolor='black', linewidth=1.5)
        ax2.add_patch(rect)
        ax2.text(3, comp['y'], comp['label'], ha='center', va='center', 
                fontsize=8, fontweight='bold')
    
    # Arrows
    for i in range(len(decoder_components) - 1):
        y1 = decoder_components[i]['y'] - 0.5
        y2 = decoder_components[i+1]['y'] + 0.4
        ax2.arrow(3, y1, 0, y2 - y1 + 0.05, head_width=0.25, head_length=0.1, 
                 fc='black', ec='black', linewidth=1.5)
    
    # Skip connections  
    ax2.plot([0.8, 0.8, 4.7, 4.7], [8.2, 7.0, 7.0, 6.5], 'r--', linewidth=2, alpha=0.6)
    ax2.plot([0.8, 0.8, 4.7, 4.7], [6.9, 4.4, 4.4, 3.9], 'r--', linewidth=2, alpha=0.6)
    ax2.plot([0.8, 0.8, 4.7, 4.7], [4.3, 3.1, 3.1, 2.6], 'r--', linewidth=2, alpha=0.6)
    
    # Encoder context to cross-attention
    ax2.arrow(-0.5, 5.2, 1.8, 0, head_width=0.15, head_length=0.15, 
             fc='blue', ec='blue', linewidth=2)
    ax2.text(-0.5, 5.6, 'From\nEncoder', ha='center', fontsize=8, color='blue', fontweight='bold')
    
    ax2.text(3, 0.3, '× N layers', ha='center', fontsize=10, 
            style='italic', fontweight='bold')
    
    plt.suptitle('Transformer Architecture: Encoder-Decoder', 
                fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    return fig_to_base64(fig)

# ============================================================================
# AUTOENCODER & VAE ILLUSTRATIONS
# ============================================================================

def generate_autoencoder_architecture():
    """Visualize autoencoder structure."""
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis('off')
    
    # Encoder
    encoder_layers = [
        {'x': 1, 'y': 2, 'w': 1.5, 'h': 4, 'label': 'Input\n784', 'color': '#e3f2fd'},
        {'x': 3, 'y': 2.5, 'w': 1.3, 'h': 3, 'label': 'Hidden\n512', 'color': '#bbdefb'},
        {'x': 4.8, 'y': 3, 'w': 1.1, 'h': 2, 'label': 'Hidden\n256', 'color': '#90caf9'},
        {'x': 6.3, 'y': 3.5, 'w': 0.8, 'h': 1, 'label': 'Code\n32', 'color': '#64b5f6'},
    ]
    
    # Decoder (mirror of encoder)
    decoder_layers = [
        {'x': 7.6, 'y': 3, 'w': 1.1, 'h': 2, 'label': 'Hidden\n256', 'color': '#90caf9'},
        {'x': 9.2, 'y': 2.5, 'w': 1.3, 'h': 3, 'label': 'Hidden\n512', 'color': '#bbdefb'},
        {'x': 11, 'y': 2, 'w': 1.5, 'h': 4, 'label': 'Output\n784', 'color': '#c8e6c9'},
    ]
    
    all_layers = encoder_layers + decoder_layers
    
    for layer in all_layers:
        rect = FancyBboxPatch((layer['x'], layer['y']), layer['w'], layer['h'],
                             boxstyle="round,pad=0.1", 
                             facecolor=layer['color'], 
                             edgecolor='black', linewidth=2)
        ax.add_patch(rect)
        
        for i, line in enumerate(layer['label'].split('\n')):
            ax.text(layer['x'] + layer['w']/2, 
                   layer['y'] + layer['h']/2 + 0.2 - i*0.35, 
                   line, ha='center', va='center', fontsize=9, fontweight='bold')
    
    # Arrows
    for i in range(len(all_layers) - 1):
        x1 = all_layers[i]['x'] + all_layers[i]['w']
        y1 = all_layers[i]['y'] + all_layers[i]['h']/2
        x2 = all_layers[i+1]['x']
        y2 = all_layers[i+1]['y'] + all_layers[i+1]['h']/2
        
        arrow = FancyArrowPatch((x1, y1), (x2, y2),
                               arrowstyle='->', mutation_scale=20, 
                               linewidth=2, color='gray')
        ax.add_patch(arrow)
    
    # Labels
    ax.text(3.5, 0.8, 'Encoder\n(Compression)', ha='center', fontsize=12, 
           fontweight='bold', color='#1976d2')
    ax.text(6.7, 0.8, 'Latent\nSpace', ha='center', fontsize=12, 
           fontweight='bold', color='#f57c00')
    ax.text(9.5, 0.8, 'Decoder\n(Reconstruction)', ha='center', fontsize=12, 
           fontweight='bold', color='#388e3c')
    
    # Brackets
    ax.plot([1, 5.8], [0.5, 0.5], 'k-', linewidth=2)
    ax.plot([1, 1], [0.4, 0.6], 'k-', linewidth=2)
    ax.plot([5.8, 5.8], [0.4, 0.6], 'k-', linewidth=2)
    
    ax.plot([7.6, 12.5], [0.5, 0.5], 'k-', linewidth=2)
    ax.plot([7.6, 7.6], [0.4, 0.6], 'k-', linewidth=2)
    ax.plot([12.5, 12.5], [0.4, 0.6], 'k-', linewidth=2)
    
    plt.title('Autoencoder Architecture', fontsize=14, fontweight='bold', pad=20)
    
    return fig_to_base64(fig)

def generate_vae_architecture():
    """Visualize VAE with reparameterization trick."""
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Encoder
    encoder_layers = [
        {'x': 1, 'y': 4, 'w': 1.5, 'h': 2, 'label': 'Input\nx', 'color': '#e3f2fd'},
        {'x': 3, 'y': 4.2, 'w': 1.3, 'h': 1.6, 'label': 'Encoder', 'color': '#bbdefb'},
    ]
    
    for layer in encoder_layers:
        rect = FancyBboxPatch((layer['x'], layer['y']), layer['w'], layer['h'],
                             boxstyle="round,pad=0.1", 
                             facecolor=layer['color'], 
                             edgecolor='black', linewidth=2)
        ax.add_patch(rect)
        ax.text(layer['x'] + layer['w']/2, layer['y'] + layer['h']/2, 
               layer['label'], ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Arrow to mu and sigma
    ax.arrow(4.3, 5, 0.5, 0, head_width=0.2, head_length=0.1, 
            fc='black', ec='black', linewidth=2)
    
    # Mu and Sigma
    mu_box = FancyBboxPatch((5.3, 5.5), 1.2, 0.8,
                           boxstyle="round,pad=0.1", 
                           facecolor='#ffcdd2', 
                           edgecolor='black', linewidth=2)
    ax.add_patch(mu_box)
    ax.text(5.9, 5.9, 'μ', ha='center', va='center', fontsize=12, fontweight='bold')
    ax.text(5.9, 5.6, '(mean)', ha='center', va='center', fontsize=8)
    
    sigma_box = FancyBboxPatch((5.3, 3.9), 1.2, 0.8,
                              boxstyle="round,pad=0.1", 
                              facecolor='#c5e1a5', 
                              edgecolor='black', linewidth=2)
    ax.add_patch(sigma_box)
    ax.text(5.9, 4.3, 'σ', ha='center', va='center', fontsize=12, fontweight='bold')
    ax.text(5.9, 4.0, '(std)', ha='center', va='center', fontsize=8)
    
    # Reparameterization trick
    ax.arrow(6.5, 5.9, 0.8, 0.3, head_width=0.15, head_length=0.1, 
            fc='purple', ec='purple', linewidth=2)
    ax.arrow(6.5, 4.3, 0.8, 0.9, head_width=0.15, head_length=0.1, 
            fc='purple', ec='purple', linewidth=2)
    
    # Epsilon (noise)
    epsilon_box = FancyBboxPatch((5.3, 2.5), 1.2, 0.6,
                                boxstyle="round,pad=0.05", 
                                facecolor='#fff9c4', 
                                edgecolor='black', linewidth=1.5)
    ax.add_patch(epsilon_box)
    ax.text(5.9, 2.8, 'ε ~ N(0,1)', ha='center', va='center', fontsize=9, fontweight='bold')
    
    ax.arrow(6.5, 2.8, 0.8, 2.4, head_width=0.15, head_length=0.1, 
            fc='orange', ec='orange', linewidth=2, linestyle='--')
    
    # z = mu + sigma * epsilon
    z_box = FancyBboxPatch((7.5, 4.7), 1.5, 1,
                          boxstyle="round,pad=0.1", 
                          facecolor='#ce93d8', 
                          edgecolor='black', linewidth=2)
    ax.add_patch(z_box)
    ax.text(8.25, 5.4, 'z = μ + σε', ha='center', va='center', fontsize=10, fontweight='bold')
    ax.text(8.25, 4.95, '(latent)', ha='center', va='center', fontsize=8, style='italic')
    
    # Decoder
    decoder_box = FancyBboxPatch((9.5, 4.2), 1.3, 1.6,
                                boxstyle="round,pad=0.1", 
                                facecolor='#bbdefb', 
                                edgecolor='black', linewidth=2)
    ax.add_patch(decoder_box)
    ax.text(10.15, 5, 'Decoder', ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Output
    output_box = FancyBboxPatch((11.3, 4), 1.5, 2,
                               boxstyle="round,pad=0.1", 
                               facecolor='#c8e6c9', 
                               edgecolor='black', linewidth=2)
    ax.add_patch(output_box)
    ax.text(12.05, 5, "x'\n(recon)", ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Arrows
    ax.arrow(9, 5.2, 0.4, -0.2, head_width=0.2, head_length=0.1, 
            fc='black', ec='black', linewidth=2)
    ax.arrow(10.8, 5, 0.4, 0, head_width=0.2, head_length=0.1, 
            fc='black', ec='black', linewidth=2)
    
    # Loss components
    ax.text(7, 1.5, 'Loss = Reconstruction Loss + KL Divergence', 
           ha='center', fontsize=11, fontweight='bold',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
    ax.text(3, 7.5, 'Reparameterization Trick:', ha='left', fontsize=10, 
           fontweight='bold', color='purple')
    ax.text(3, 7, 'Позволяет градиентам проходить через семплирование', 
           ha='left', fontsize=9, style='italic')
    
    plt.title('Variational Autoencoder (VAE)', fontsize=14, fontweight='bold', pad=20)
    
    return fig_to_base64(fig)

# ============================================================================
# GAN ILLUSTRATIONS
# ============================================================================

def generate_gan_architecture():
    """Visualize GAN generator and discriminator."""
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Generator
    gen_layers = [
        {'x': 1, 'y': 6, 'w': 1, 'h': 0.8, 'label': 'Noise z\n~N(0,1)', 'color': '#fff9c4'},
        {'x': 2.5, 'y': 5.8, 'w': 1, 'h': 1.2, 'label': 'Dense', 'color': '#bbdefb'},
        {'x': 4, 'y': 5.5, 'w': 1, 'h': 1.8, 'label': 'Upsample', 'color': '#90caf9'},
        {'x': 5.5, 'y': 5.2, 'w': 1, 'h': 2.4, 'label': 'Conv', 'color': '#64b5f6'},
        {'x': 7, 'y': 5, 'w': 1, 'h': 2.8, 'label': 'Fake\nImage', 'color': '#ffcdd2'},
    ]
    
    for layer in gen_layers:
        rect = FancyBboxPatch((layer['x'], layer['y']), layer['w'], layer['h'],
                             boxstyle="round,pad=0.1", 
                             facecolor=layer['color'], 
                             edgecolor='black', linewidth=2)
        ax.add_patch(rect)
        
        for i, line in enumerate(layer['label'].split('\n')):
            ax.text(layer['x'] + layer['w']/2, 
                   layer['y'] + layer['h']/2 + 0.15 - i*0.3, 
                   line, ha='center', va='center', fontsize=9, fontweight='bold')
    
    # Arrows for generator
    for i in range(len(gen_layers) - 1):
        x1 = gen_layers[i]['x'] + gen_layers[i]['w']
        y1 = gen_layers[i]['y'] + gen_layers[i]['h']/2
        x2 = gen_layers[i+1]['x']
        y2 = gen_layers[i+1]['y'] + gen_layers[i+1]['h']/2
        
        arrow = FancyArrowPatch((x1, y1), (x2, y2),
                               arrowstyle='->', mutation_scale=20, 
                               linewidth=2, color='gray')
        ax.add_patch(arrow)
    
    ax.text(4, 8.5, 'Generator G', ha='center', fontsize=13, 
           fontweight='bold', color='#1976d2',
           bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    # Real image
    real_img_box = FancyBboxPatch((7, 2), 1, 2.8,
                                 boxstyle="round,pad=0.1", 
                                 facecolor='#c8e6c9', 
                                 edgecolor='black', linewidth=2)
    ax.add_patch(real_img_box)
    ax.text(7.5, 3.4, 'Real\nImage', ha='center', va='center', fontsize=9, fontweight='bold')
    
    # Arrows to discriminator
    ax.arrow(8, 6.4, 1, -0.9, head_width=0.2, head_length=0.1, 
            fc='red', ec='red', linewidth=2.5)
    ax.text(8.5, 5.8, 'Fake', ha='center', fontsize=9, color='red', fontweight='bold')
    
    ax.arrow(8, 3.4, 1, 0, head_width=0.2, head_length=0.1, 
            fc='green', ec='green', linewidth=2.5)
    ax.text(8.5, 2.9, 'Real', ha='center', fontsize=9, color='green', fontweight='bold')
    
    # Discriminator
    disc_layers = [
        {'x': 9.5, 'y': 4, 'w': 1, 'h': 2, 'label': 'Conv', 'color': '#ce93d8'},
        {'x': 11, 'y': 4.3, 'w': 1, 'h': 1.4, 'label': 'Conv', 'color': '#ba68c8'},
        {'x': 12.5, 'y': 4.6, 'w': 1, 'h': 0.8, 'label': 'Dense', 'color': '#ab47bc'},
    ]
    
    for layer in disc_layers:
        rect = FancyBboxPatch((layer['x'], layer['y']), layer['w'], layer['h'],
                             boxstyle="round,pad=0.1", 
                             facecolor=layer['color'], 
                             edgecolor='black', linewidth=2)
        ax.add_patch(rect)
        ax.text(layer['x'] + layer['w']/2, layer['y'] + layer['h']/2, 
               layer['label'], ha='center', va='center', fontsize=9, fontweight='bold')
    
    # Arrows for discriminator
    for i in range(len(disc_layers) - 1):
        x1 = disc_layers[i]['x'] + disc_layers[i]['w']
        y1 = disc_layers[i]['y'] + disc_layers[i]['h']/2
        x2 = disc_layers[i+1]['x']
        y2 = disc_layers[i+1]['y'] + disc_layers[i+1]['h']/2
        
        arrow = FancyArrowPatch((x1, y1), (x2, y2),
                               arrowstyle='->', mutation_scale=20, 
                               linewidth=2, color='gray')
        ax.add_patch(arrow)
    
    ax.text(11, 8.5, 'Discriminator D', ha='center', fontsize=13, 
           fontweight='bold', color='#7b1fa2',
           bbox=dict(boxstyle='round', facecolor='plum', alpha=0.5))
    
    # Output probability
    output_box = FancyBboxPatch((12.5, 2.5), 1, 0.8,
                               boxstyle="round,pad=0.1", 
                               facecolor='#fff9c4', 
                               edgecolor='black', linewidth=2)
    ax.add_patch(output_box)
    ax.text(13, 2.9, 'P(real)', ha='center', va='center', fontsize=9, fontweight='bold')
    
    ax.arrow(13, 4.6, 0, -1.2, head_width=0.2, head_length=0.1, 
            fc='black', ec='black', linewidth=2)
    
    # Training info
    ax.text(7, 0.8, 'Adversarial Training:', ha='center', fontsize=11, fontweight='bold')
    ax.text(7, 0.3, 'G учится обманывать D, D учится различать real/fake', 
           ha='center', fontsize=9, style='italic')
    
    plt.title('Generative Adversarial Network (GAN)', fontsize=14, fontweight='bold', pad=20)
    
    return fig_to_base64(fig)


def generate_diffusion_process():
    """Visualize diffusion forward and reverse process."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    # Forward process (adding noise)
    ax1.set_xlim(0, 14)
    ax1.set_ylim(0, 6)
    ax1.axis('off')
    ax1.set_title('Forward Process: Постепенное добавление шума', 
                 fontsize=12, fontweight='bold')
    
    n_steps = 5
    for i in range(n_steps):
        x = 1 + i * 2.8
        
        # Create noisy image representation
        noise_level = i / (n_steps - 1)
        rect = FancyBboxPatch((x, 2), 2, 2,
                             boxstyle="round,pad=0.1", 
                             facecolor=plt.cm.gray(1 - noise_level * 0.7), 
                             edgecolor='black', linewidth=2)
        ax1.add_patch(rect)
        ax1.text(x + 1, 1.5, f't = {i}', ha='center', fontsize=9, fontweight='bold')
        
        if i == 0:
            ax1.text(x + 1, 4.5, 'x₀\n(real)', ha='center', fontsize=9, fontweight='bold')
        elif i == n_steps - 1:
            ax1.text(x + 1, 4.5, 'xₜ\n(noise)', ha='center', fontsize=9, fontweight='bold')
        else:
            ax1.text(x + 1, 4.5, f'x_{i}', ha='center', fontsize=9)
        
        # Arrow to next step
        if i < n_steps - 1:
            ax1.arrow(x + 2.2, 3, 0.4, 0, head_width=0.3, head_length=0.15, 
                     fc='red', ec='red', linewidth=2)
            ax1.text(x + 2.4, 3.5, 'q(xₜ|xₜ₋₁)', ha='center', fontsize=7, color='red')
    
    # Reverse process (denoising)
    ax2.set_xlim(0, 14)
    ax2.set_ylim(0, 6)
    ax2.axis('off')
    ax2.set_title('Reverse Process: Обучаемое удаление шума (Denoising)', 
                 fontsize=12, fontweight='bold')
    
    for i in range(n_steps-1, -1, -1):
        x = 1 + (n_steps - 1 - i) * 2.8
        
        # Create denoising representation
        noise_level = i / (n_steps - 1)
        rect = FancyBboxPatch((x, 2), 2, 2,
                             boxstyle="round,pad=0.1", 
                             facecolor=plt.cm.gray(1 - noise_level * 0.7), 
                             edgecolor='black', linewidth=2)
        ax2.add_patch(rect)
        ax2.text(x + 1, 1.5, f't = {i}', ha='center', fontsize=9, fontweight='bold')
        
        if i == n_steps - 1:
            ax2.text(x + 1, 4.5, 'xₜ\n(noise)', ha='center', fontsize=9, fontweight='bold')
        elif i == 0:
            ax2.text(x + 1, 4.5, 'x₀\n(generated)', ha='center', fontsize=9, fontweight='bold', color='green')
        else:
            ax2.text(x + 1, 4.5, f'x_{i}', ha='center', fontsize=9)
        
        # Arrow to next step with UNet
        if i > 0:
            ax2.arrow(x + 2.2, 3, 0.4, 0, head_width=0.3, head_length=0.15, 
                     fc='blue', ec='blue', linewidth=2)
            ax2.text(x + 2.4, 3.5, 'pθ(xₜ₋₁|xₜ)', ha='center', fontsize=7, color='blue')
            ax2.text(x + 2.4, 2.7, 'UNet', ha='center', fontsize=7, 
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
    
    plt.suptitle('Diffusion Models: Forward & Reverse Process', 
                fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    return fig_to_base64(fig)

def generate_capsule_network():
    """Visualize capsule network structure."""
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Input
    input_box = FancyBboxPatch((0.5, 4.5), 1.5, 1,
                              boxstyle="round,pad=0.1", 
                              facecolor='#e3f2fd', 
                              edgecolor='black', linewidth=2)
    ax.add_patch(input_box)
    ax.text(1.25, 5, 'Input\nImage', ha='center', va='center', fontsize=9, fontweight='bold')
    
    # Conv layer
    conv_box = FancyBboxPatch((2.5, 4.3), 1.5, 1.4,
                             boxstyle="round,pad=0.1", 
                             facecolor='#bbdefb', 
                             edgecolor='black', linewidth=2)
    ax.add_patch(conv_box)
    ax.text(3.25, 5, 'Conv\nLayer', ha='center', va='center', fontsize=9, fontweight='bold')
    
    ax.arrow(2, 5, 0.4, 0, head_width=0.2, head_length=0.1, 
            fc='black', ec='black', linewidth=2)
    
    # Primary capsules
    ax.text(5.5, 8.5, 'Primary Capsules', ha='center', fontsize=11, fontweight='bold')
    
    n_primary = 4
    for i in range(n_primary):
        y = 7 - i * 1.2
        
        # Capsule as cylinder/vector
        circle = Circle((5.5, y), 0.3, facecolor='#90caf9', edgecolor='black', linewidth=1.5)
        ax.add_patch(circle)
        
        # Vector representation
        ax.arrow(5.5, y, 0, -0.6, head_width=0.15, head_length=0.1, 
                fc='#f57c00', ec='#f57c00', linewidth=2)
        
        ax.text(4.5, y, f'Cap {i+1}', ha='right', fontsize=8)
    
    ax.arrow(4, 5, 0.8, 1, head_width=0.2, head_length=0.1, 
            fc='gray', ec='gray', linewidth=1.5, alpha=0.7)
    
    # Routing arrows
    ax.text(7.5, 8.5, 'Dynamic Routing', ha='center', fontsize=11, fontweight='bold', color='purple')
    
    for i in range(n_primary):
        y1 = 7 - i * 1.2
        for j in range(3):
            y2 = 7 - j * 1.5
            ax.plot([5.8, 8.7], [y1, y2], 'purple', linewidth=1, alpha=0.5)
    
    # Digit capsules
    ax.text(9.5, 8.5, 'Digit Capsules', ha='center', fontsize=11, fontweight='bold')
    
    n_digit = 3
    for i in range(n_digit):
        y = 7 - i * 1.5
        
        # Larger capsule for digit
        circle = Circle((9.5, y), 0.4, facecolor='#c8e6c9', edgecolor='black', linewidth=2)
        ax.add_patch(circle)
        
        # Vector representation
        ax.arrow(9.5, y, 0, -0.8, head_width=0.2, head_length=0.12, 
                fc='#388e3c', ec='#388e3c', linewidth=2.5)
        
        ax.text(10.5, y, f'Digit {i}', ha='left', fontsize=9, fontweight='bold')
    
    # Capsule properties
    ax.text(6, 1.5, 'Capsule = Vector:', ha='center', fontsize=10, fontweight='bold')
    ax.text(6, 1, '• Length = probability', ha='left', fontsize=8)
    ax.text(6, 0.5, '• Direction = properties', ha='left', fontsize=8)
    
    plt.title('Capsule Network Architecture', fontsize=14, fontweight='bold', pad=20)
    
    return fig_to_base64(fig)

def generate_vision_transformer():
    """Visualize Vision Transformer (ViT) architecture."""
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Input image
    img_box = FancyBboxPatch((0.5, 4), 2, 2,
                            boxstyle="round,pad=0.1", 
                            facecolor='#e3f2fd', 
                            edgecolor='black', linewidth=2)
    ax.add_patch(img_box)
    ax.text(1.5, 5, 'Image\n224×224', ha='center', va='center', fontsize=9, fontweight='bold')
    
    # Patch splitting
    ax.arrow(2.5, 5, 0.8, 0, head_width=0.2, head_length=0.1, 
            fc='black', ec='black', linewidth=2)
    
    # Patches
    patch_box = FancyBboxPatch((3.5, 3.5), 2, 3,
                              boxstyle="round,pad=0.1", 
                              facecolor='#bbdefb', 
                              edgecolor='black', linewidth=2)
    ax.add_patch(patch_box)
    ax.text(4.5, 6.2, 'Split to Patches', ha='center', fontsize=9, fontweight='bold')
    
    # Draw small patches
    for i in range(3):
        for j in range(3):
            small_rect = Rectangle((3.7 + j*0.6, 4 + i*0.6), 0.5, 0.5,
                                  facecolor='white', edgecolor='gray', linewidth=1)
            ax.add_patch(small_rect)
    
    ax.text(4.5, 3.7, 'N=196 patches\n(16×16 each)', ha='center', fontsize=7, style='italic')
    
    # Linear projection
    ax.arrow(5.5, 5, 0.7, 0, head_width=0.2, head_length=0.1, 
            fc='black', ec='black', linewidth=2)
    
    proj_box = FancyBboxPatch((6.4, 4.2), 1.2, 1.6,
                             boxstyle="round,pad=0.1", 
                             facecolor='#90caf9', 
                             edgecolor='black', linewidth=2)
    ax.add_patch(proj_box)
    ax.text(7, 5, 'Linear\nProjection', ha='center', va='center', fontsize=8, fontweight='bold')
    
    # Add positional encoding
    ax.arrow(7.6, 5, 0.7, 0, head_width=0.2, head_length=0.1, 
            fc='black', ec='black', linewidth=2)
    
    pos_box = FancyBboxPatch((8.5, 4.5), 1.2, 1,
                            boxstyle="round,pad=0.1", 
                            facecolor='#fff9c4', 
                            edgecolor='black', linewidth=2)
    ax.add_patch(pos_box)
    ax.text(9.1, 5, '+ Pos\nEmbed', ha='center', va='center', fontsize=8, fontweight='bold')
    
    # CLS token
    cls_box = FancyBboxPatch((8.5, 6.2), 1.2, 0.6,
                            boxstyle="round,pad=0.05", 
                            facecolor='#ffccbc', 
                            edgecolor='red', linewidth=2)
    ax.add_patch(cls_box)
    ax.text(9.1, 6.5, '[CLS]', ha='center', va='center', fontsize=8, fontweight='bold')
    
    ax.arrow(9.1, 6.2, 0, -0.6, head_width=0.15, head_length=0.08, 
            fc='red', ec='red', linewidth=1.5)
    
    # Transformer encoder
    ax.arrow(9.7, 5, 0.5, 0, head_width=0.2, head_length=0.1, 
            fc='black', ec='black', linewidth=2)
    
    transformer_box = FancyBboxPatch((10.4, 3), 2.5, 4,
                                    boxstyle="round,pad=0.1", 
                                    facecolor='#ce93d8', 
                                    edgecolor='black', linewidth=2)
    ax.add_patch(transformer_box)
    ax.text(11.65, 6.5, 'Transformer Encoder', ha='center', fontsize=10, fontweight='bold')
    
    # Transformer blocks
    encoder_blocks = ['Multi-Head\nAttention', 'MLP', 'Layer Norm']
    for i, block in enumerate(encoder_blocks):
        y = 5.8 - i * 1.2
        mini_box = FancyBboxPatch((10.7, y-0.4), 1.9, 0.8,
                                 boxstyle="round,pad=0.05", 
                                 facecolor='white', 
                                 edgecolor='black', linewidth=1)
        ax.add_patch(mini_box)
        ax.text(11.65, y, block, ha='center', va='center', fontsize=7, fontweight='bold')
    
    ax.text(11.65, 3.3, '× L layers', ha='center', fontsize=8, style='italic')
    
    # Classification head
    ax.text(11.65, 1.8, 'Extract [CLS] token', ha='center', fontsize=9, fontweight='bold', color='red')
    
    class_box = FancyBboxPatch((10.9, 0.5), 1.5, 0.8,
                              boxstyle="round,pad=0.1", 
                              facecolor='#c8e6c9', 
                              edgecolor='black', linewidth=2)
    ax.add_patch(class_box)
    ax.text(11.65, 0.9, 'MLP Head', ha='center', va='center', fontsize=9, fontweight='bold')
    
    ax.arrow(11.65, 3, 0, -0.8, head_width=0.2, head_length=0.1, 
            fc='red', ec='red', linewidth=2)
    
    # Output
    ax.text(11.65, 0, 'Class Predictions', ha='center', fontsize=9, fontweight='bold',
           bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
    
    plt.title('Vision Transformer (ViT): Image as Sequence of Patches', 
             fontsize=14, fontweight='bold', pad=20)
    
    return fig_to_base64(fig)

# ============================================================================
# MAIN FUNCTION
# ============================================================================

def generate_all_illustrations():
    """Generate all Neural Architecture illustrations."""
    print("Generating Neural Architecture illustrations...")
    print("=" * 70)
    
    illustrations = {}
    
    print("\n1. CNN Architectures:")
    illustrations['lenet_architecture'] = generate_lenet_architecture()
    print("  ✓ LeNet-5 architecture")
    
    illustrations['vgg_block'] = generate_vgg_block()
    print("  ✓ VGG block structure")
    
    illustrations['alexnet_vs_vgg'] = generate_alexnet_vs_vgg()
    print("  ✓ AlexNet vs VGG comparison")
    
    print("\n2. ResNet:")
    illustrations['residual_block'] = generate_residual_block()
    print("  ✓ Residual block")
    
    illustrations['resnet_degradation'] = generate_resnet_degradation()
    print("  ✓ Degradation problem solution")
    
    illustrations['skip_connections'] = generate_skip_connections()
    print("  ✓ Skip connection types")
    
    print("\n3. Inception & EfficientNet:")
    illustrations['inception_module'] = generate_inception_module()
    print("  ✓ Inception module")
    
    illustrations['efficientnet_scaling'] = generate_efficientnet_scaling()
    print("  ✓ EfficientNet compound scaling")
    
    print("\n4. Transformers & Attention:")
    illustrations['attention_mechanism'] = generate_attention_mechanism()
    print("  ✓ Attention mechanism")
    
    illustrations['multi_head_attention'] = generate_multi_head_attention()
    print("  ✓ Multi-head attention")
    
    illustrations['transformer_architecture'] = generate_transformer_architecture()
    print("  ✓ Transformer encoder-decoder")
    
    print("\n5. Vision Transformers:")
    illustrations['vision_transformer'] = generate_vision_transformer()
    print("  ✓ Vision Transformer (ViT)")
    
    print("\n6. Autoencoders & VAE:")
    illustrations['autoencoder_architecture'] = generate_autoencoder_architecture()
    print("  ✓ Autoencoder architecture")
    
    illustrations['vae_architecture'] = generate_vae_architecture()
    print("  ✓ VAE with reparameterization")
    
    print("\n7. GANs:")
    illustrations['gan_architecture'] = generate_gan_architecture()
    print("  ✓ GAN architecture")
    
    print("\n8. Diffusion Models:")
    illustrations['diffusion_process'] = generate_diffusion_process()
    print("  ✓ Diffusion forward/reverse process")
    
    print("\n9. Capsule Networks:")
    illustrations['capsule_network'] = generate_capsule_network()
    print("  ✓ Capsule network")
    
    print("\n" + "=" * 70)
    print(f"✓ Successfully generated {len(illustrations)} illustrations!")
    print("=" * 70)
    
    return illustrations

if __name__ == '__main__':
    illustrations = generate_all_illustrations()
    print(f"\nGenerated {len(illustrations)} illustrations")
    print("Illustrations are base64 encoded and ready for HTML embedding")
