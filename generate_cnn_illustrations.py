#!/usr/bin/env python3
"""
Generate matplotlib illustrations for CNN and Pooling cheatsheets.
This script creates high-quality visualizations and encodes them as base64 for inline HTML embedding.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle, FancyBboxPatch, FancyArrowPatch
from matplotlib.colors import LinearSegmentedColormap
import base64
from io import BytesIO
import warnings
warnings.filterwarnings('ignore')

# Set consistent style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9

def fig_to_base64(fig):
    """Convert matplotlib figure to base64 string."""
    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=300)
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()
    plt.close(fig)
    return img_base64

# ============================================================================
# CNN BASICS ILLUSTRATIONS
# ============================================================================

def generate_convolution_operation():
    """Visualize 2D convolution operation."""
    # Create sample input and kernel
    np.random.seed(42)
    input_size = 6
    kernel_size = 3
    
    # Create input image (simple pattern)
    input_img = np.zeros((input_size, input_size))
    input_img[1:5, 2:4] = 1.0  # Vertical line
    input_img += np.random.randn(input_size, input_size) * 0.1
    
    # Edge detection kernel (vertical edges)
    kernel = np.array([[-1, 0, 1],
                       [-2, 0, 2],
                       [-1, 0, 1]])
    
    # Compute convolution at one position
    output_size = input_size - kernel_size + 1
    output = np.zeros((output_size, output_size))
    
    for i in range(output_size):
        for j in range(output_size):
            region = input_img[i:i+kernel_size, j:j+kernel_size]
            output[i, j] = np.sum(region * kernel)
    
    fig = plt.figure(figsize=(14, 4))
    
    # Input image
    ax1 = plt.subplot(1, 4, 1)
    im1 = ax1.imshow(input_img, cmap='gray', vmin=-1, vmax=2)
    ax1.set_title('Входное изображение\n(6×6)', fontsize=11, fontweight='bold')
    ax1.set_xticks([])
    ax1.set_yticks([])
    # Highlight convolution region
    rect = Rectangle((1.5, 0.5), kernel_size-1, kernel_size-1, 
                     linewidth=2, edgecolor='red', facecolor='none')
    ax1.add_patch(rect)
    plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    
    # Kernel
    ax2 = plt.subplot(1, 4, 2)
    im2 = ax2.imshow(kernel, cmap='RdBu_r', vmin=-2, vmax=2)
    ax2.set_title('Фильтр (Kernel)\n(3×3)', fontsize=11, fontweight='bold')
    for i in range(kernel_size):
        for j in range(kernel_size):
            text = ax2.text(j, i, f'{kernel[i, j]:.0f}',
                          ha="center", va="center", color="black", fontsize=9)
    ax2.set_xticks([])
    ax2.set_yticks([])
    plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    
    # Convolution operation
    ax3 = plt.subplot(1, 4, 3)
    region = input_img[0:kernel_size, 1:1+kernel_size]
    im3 = ax3.imshow(region * kernel, cmap='RdBu_r', vmin=-2, vmax=2)
    ax3.set_title('Поэлементное\nумножение', fontsize=11, fontweight='bold')
    ax3.set_xticks([])
    ax3.set_yticks([])
    result_val = np.sum(region * kernel)
    ax3.text(0.5, -0.15, f'Сумма = {result_val:.2f}', 
            transform=ax3.transAxes, ha='center', fontsize=10, 
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
    
    # Output feature map
    ax4 = plt.subplot(1, 4, 4)
    im4 = ax4.imshow(output, cmap='viridis')
    ax4.set_title('Выходная карта признаков\n(4×4)', fontsize=11, fontweight='bold')
    ax4.set_xticks([])
    ax4.set_yticks([])
    plt.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)
    
    plt.suptitle('Операция свертки (Convolution)', fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    return fig_to_base64(fig)

def generate_feature_maps():
    """Visualize multiple feature maps from different filters."""
    np.random.seed(42)
    
    # Create a simple input image with different patterns
    input_img = np.zeros((8, 8))
    # Vertical line
    input_img[1:7, 3] = 1.0
    # Horizontal line
    input_img[5, 1:7] = 1.0
    # Add some noise
    input_img += np.random.randn(8, 8) * 0.1
    
    # Different kernels
    kernels = {
        'Вертикальные\nрёбра': np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]),
        'Горизонтальные\nрёбра': np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]]),
        'Диагональные\nрёбра': np.array([[0, 1, 1], [-1, 0, 1], [-1, -1, 0]]),
        'Размытие': np.ones((3, 3)) / 9
    }
    
    fig = plt.figure(figsize=(14, 10))
    
    # Show input (spanning two rows in first column)
    ax0 = plt.subplot(4, 4, 1)
    ax0.imshow(input_img, cmap='gray')
    ax0.set_title('Входное\nизображение', fontsize=11, fontweight='bold')
    ax0.set_xticks([])
    ax0.set_yticks([])
    
    # Show each kernel and its output
    for idx, (name, kernel) in enumerate(kernels.items()):
        # Show kernel in top row
        ax_k = plt.subplot(4, 4, idx + 2)
        im = ax_k.imshow(kernel, cmap='RdBu_r', vmin=-1, vmax=1)
        ax_k.set_title(f'Фильтр:\n{name}', fontsize=10, fontweight='bold')
        ax_k.set_xticks([])
        ax_k.set_yticks([])
        
        # Compute output
        output_size = input_img.shape[0] - kernel.shape[0] + 1
        output = np.zeros((output_size, output_size))
        for i in range(output_size):
            for j in range(output_size):
                region = input_img[i:i+3, j:j+3]
                output[i, j] = np.sum(region * kernel)
        
        # Show output in third row
        ax_o = plt.subplot(4, 4, idx + 10)
        ax_o.imshow(output, cmap='viridis')
        ax_o.set_title(f'Карта признаков', fontsize=10)
        ax_o.set_xticks([])
        ax_o.set_yticks([])
    
    plt.suptitle('Различные фильтры извлекают различные признаки', 
                fontsize=13, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    return fig_to_base64(fig)

def generate_cnn_architecture():
    """Visualize CNN architecture layers."""
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 6)
    ax.axis('off')
    
    # Define layer positions and sizes
    layers = [
        {'x': 0.5, 'y': 1.5, 'w': 1.5, 'h': 3, 'color': '#e3f2fd', 'label': 'Input\n32×32×3', 'size': '32×32'},
        {'x': 2.5, 'y': 1.3, 'w': 1.3, 'h': 3.4, 'color': '#bbdefb', 'label': 'Conv1\n32×32×32', 'size': '32×32'},
        {'x': 4.3, 'y': 1.8, 'w': 1.0, 'h': 2.4, 'color': '#90caf9', 'label': 'Pool1\n16×16×32', 'size': '16×16'},
        {'x': 5.8, 'y': 1.6, 'w': 1.1, 'h': 2.8, 'color': '#64b5f6', 'label': 'Conv2\n16×16×64', 'size': '16×16'},
        {'x': 7.4, 'y': 2.1, 'w': 0.8, 'h': 1.8, 'color': '#42a5f5', 'label': 'Pool2\n8×8×64', 'size': '8×8'},
        {'x': 8.7, 'y': 2.0, 'w': 0.9, 'h': 2.0, 'color': '#2196f3', 'label': 'Conv3\n8×8×128', 'size': '8×8'},
        {'x': 10.1, 'y': 2.6, 'w': 0.5, 'h': 0.8, 'color': '#1e88e5', 'label': 'Pool3\n4×4×128', 'size': '4×4'},
        {'x': 11.1, 'y': 2.5, 'w': 0.3, 'h': 1.0, 'color': '#ffa726', 'label': 'Flatten\n2048', 'size': ''},
        {'x': 11.9, 'y': 2.5, 'w': 0.3, 'h': 1.0, 'color': '#ff9800', 'label': 'Dense\n128', 'size': ''},
        {'x': 12.7, 'y': 2.5, 'w': 0.3, 'h': 1.0, 'color': '#fb8c00', 'label': 'Output\n10', 'size': ''},
    ]
    
    # Draw layers
    for layer in layers:
        rect = FancyBboxPatch((layer['x'], layer['y']), layer['w'], layer['h'],
                             boxstyle="round,pad=0.05", 
                             facecolor=layer['color'], 
                             edgecolor='black', linewidth=1.5)
        ax.add_patch(rect)
        
        # Add label
        label_parts = layer['label'].split('\n')
        ax.text(layer['x'] + layer['w']/2, layer['y'] + layer['h']/2, 
               label_parts[0], ha='center', va='center', fontsize=9, fontweight='bold')
        if len(label_parts) > 1:
            ax.text(layer['x'] + layer['w']/2, layer['y'] + layer['h']/2 - 0.3, 
                   label_parts[1], ha='center', va='center', fontsize=8)
    
    # Add arrows between layers
    for i in range(len(layers) - 1):
        x1 = layers[i]['x'] + layers[i]['w']
        y1 = layers[i]['y'] + layers[i]['h']/2
        x2 = layers[i+1]['x']
        y2 = layers[i+1]['y'] + layers[i+1]['h']/2
        
        arrow = FancyArrowPatch((x1, y1), (x2, y2),
                               arrowstyle='->', mutation_scale=15, 
                               linewidth=1.5, color='gray')
        ax.add_patch(arrow)
    
    # Add annotations
    ax.text(3.0, 0.5, 'Извлечение признаков', ha='center', fontsize=11, 
           fontweight='bold', color='#1976d2')
    ax.text(11.5, 0.5, 'Классификация', ha='center', fontsize=11, 
           fontweight='bold', color='#f57c00')
    
    # Add bracket for feature extraction
    ax.plot([2.3, 10.8], [0.7, 0.7], 'k-', linewidth=1.5)
    ax.plot([2.3, 2.3], [0.6, 0.8], 'k-', linewidth=1.5)
    ax.plot([10.8, 10.8], [0.6, 0.8], 'k-', linewidth=1.5)
    
    # Add bracket for classification
    ax.plot([10.9, 13.2], [0.7, 0.7], 'k-', linewidth=1.5)
    ax.plot([10.9, 10.9], [0.6, 0.8], 'k-', linewidth=1.5)
    ax.plot([13.2, 13.2], [0.6, 0.8], 'k-', linewidth=1.5)
    
    plt.title('Типичная архитектура CNN', fontsize=14, fontweight='bold', pad=20)
    
    return fig_to_base64(fig)

# ============================================================================
# POOLING ILLUSTRATIONS
# ============================================================================

def generate_pooling_comparison():
    """Compare Max Pooling vs Average Pooling."""
    # Create sample input
    np.random.seed(42)
    input_data = np.array([[1, 3, 2, 4],
                          [5, 6, 7, 8],
                          [9, 2, 1, 3],
                          [4, 5, 6, 7]])
    
    # Max pooling
    max_pool = np.array([[6, 8],
                        [9, 7]])
    
    # Average pooling
    avg_pool = np.array([[3.75, 5.25],
                        [5.0, 4.25]])
    
    fig = plt.figure(figsize=(14, 5))
    
    # Input
    ax1 = plt.subplot(1, 4, 1)
    im1 = ax1.imshow(input_data, cmap='Blues', vmin=0, vmax=9)
    ax1.set_title('Входная карта признаков\n(4×4)', fontsize=11, fontweight='bold')
    for i in range(4):
        for j in range(4):
            text = ax1.text(j, i, f'{input_data[i, j]:.0f}',
                          ha="center", va="center", color="black", fontsize=10, fontweight='bold')
    
    # Add grid to show pooling regions
    for i in [0, 2, 4]:
        ax1.axhline(y=i-0.5, color='red', linewidth=2)
        ax1.axvline(x=i-0.5, color='red', linewidth=2)
    
    ax1.set_xticks([])
    ax1.set_yticks([])
    plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    
    # Show max pooling regions
    ax2 = plt.subplot(1, 4, 2)
    im2 = ax2.imshow(input_data, cmap='Blues', vmin=0, vmax=9, alpha=0.3)
    ax2.set_title('Max Pooling (2×2)\nВыбор максимума', fontsize=11, fontweight='bold')
    
    # Highlight max values in each region
    regions = [
        ((0, 0, 2, 2), (0, 1)),  # Top-left region, max at (0,1)
        ((0, 2, 2, 4), (1, 3)),  # Top-right region, max at (1,3)
        ((2, 0, 4, 2), (2, 0)),  # Bottom-left region, max at (2,0)
        ((2, 2, 4, 4), (1, 2)),  # Bottom-right region, max at (1,2)
    ]
    
    for (r1, c1, r2, c2), (max_r, max_c) in regions:
        rect = Rectangle((c1-0.5, r1-0.5), c2-c1, r2-r1,
                        linewidth=2, edgecolor='red', facecolor='none')
        ax2.add_patch(rect)
        # Circle the max value
        circle = plt.Circle((max_c, max_r), 0.3, color='red', fill=False, linewidth=2)
        ax2.add_patch(circle)
    
    for i in range(4):
        for j in range(4):
            ax2.text(j, i, f'{input_data[i, j]:.0f}',
                    ha="center", va="center", color="black", fontsize=10)
    ax2.set_xticks([])
    ax2.set_yticks([])
    
    # Max pooling result
    ax3 = plt.subplot(1, 4, 3)
    im3 = ax3.imshow(max_pool, cmap='Greens', vmin=0, vmax=9)
    ax3.set_title('Результат Max Pooling\n(2×2)', fontsize=11, fontweight='bold')
    for i in range(2):
        for j in range(2):
            text = ax3.text(j, i, f'{max_pool[i, j]:.0f}',
                          ha="center", va="center", color="black", fontsize=12, fontweight='bold')
    ax3.set_xticks([])
    ax3.set_yticks([])
    plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
    
    # Average pooling result
    ax4 = plt.subplot(1, 4, 4)
    im4 = ax4.imshow(avg_pool, cmap='Oranges', vmin=0, vmax=9)
    ax4.set_title('Результат Average Pooling\n(2×2)', fontsize=11, fontweight='bold')
    for i in range(2):
        for j in range(2):
            text = ax4.text(j, i, f'{avg_pool[i, j]:.2f}',
                          ha="center", va="center", color="black", fontsize=11, fontweight='bold')
    ax4.set_xticks([])
    ax4.set_yticks([])
    plt.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)
    
    plt.suptitle('Сравнение Max Pooling и Average Pooling', fontsize=13, fontweight='bold', y=1.00)
    plt.tight_layout()
    
    return fig_to_base64(fig)

def generate_global_pooling():
    """Visualize Global Average Pooling."""
    np.random.seed(42)
    
    # Create sample feature maps
    feature_maps = []
    for _ in range(3):
        fm = np.random.rand(4, 4) * 10
        feature_maps.append(fm)
    
    # Compute global average for each
    global_avgs = [np.mean(fm) for fm in feature_maps]
    
    fig = plt.figure(figsize=(14, 4))
    
    # Show feature maps
    for idx, (fm, avg) in enumerate(zip(feature_maps, global_avgs)):
        ax = plt.subplot(1, 5, idx + 1)
        im = ax.imshow(fm, cmap='viridis', vmin=0, vmax=10)
        ax.set_title(f'Карта признаков {idx+1}\n(4×4)', fontsize=10, fontweight='bold')
        ax.set_xticks([])
        ax.set_yticks([])
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        # Show average value below
        ax.text(0.5, -0.15, f'Среднее: {avg:.2f}', 
               transform=ax.transAxes, ha='center', fontsize=9,
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
    # Show result vector
    ax_result = plt.subplot(1, 5, 5)
    result_vec = np.array(global_avgs).reshape(-1, 1)
    im = ax_result.imshow(result_vec, cmap='plasma', vmin=0, vmax=10, aspect='auto')
    ax_result.set_title('Результат\nGlobal Avg Pool', fontsize=10, fontweight='bold')
    for i in range(3):
        ax_result.text(0, i, f'{global_avgs[i]:.2f}',
                      ha="center", va="center", color="white", fontsize=10, fontweight='bold')
    ax_result.set_xticks([])
    ax_result.set_yticks([])
    ax_result.set_ylabel('Вектор признаков', fontsize=10)
    
    plt.suptitle('Global Average Pooling: (batch, C, H, W) → (batch, C)', 
                fontsize=12, fontweight='bold', y=1.00)
    plt.tight_layout()
    
    return fig_to_base64(fig)

# ============================================================================
# 1D & 3D CNN ILLUSTRATIONS
# ============================================================================

def generate_1d_convolution():
    """Visualize 1D convolution on time series."""
    np.random.seed(42)
    
    # Generate sample time series
    t = np.linspace(0, 4*np.pi, 50)
    signal = np.sin(t) + 0.5*np.sin(3*t) + np.random.randn(50)*0.1
    
    # 1D kernel (simple moving average)
    kernel = np.array([0.2, 0.6, 0.2])
    
    # Apply convolution
    output = np.convolve(signal, kernel, mode='valid')
    
    fig = plt.figure(figsize=(14, 8))
    
    # Original signal
    ax1 = plt.subplot(3, 1, 1)
    ax1.plot(signal, 'b-', linewidth=2, label='Исходный сигнал')
    ax1.set_title('1D Сигнал (временной ряд)', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Время')
    ax1.set_ylabel('Значение')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Highlight convolution region
    start_idx = 10
    ax1.axvspan(start_idx, start_idx + len(kernel), alpha=0.3, color='red', 
               label='Окно свертки')
    
    # Kernel
    ax2 = plt.subplot(3, 1, 2)
    ax2.stem(kernel, basefmt=' ', linefmt='r-', markerfmt='ro')
    ax2.set_title('1D Kernel (фильтр)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Позиция')
    ax2.set_ylabel('Вес')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 0.8])
    for i, val in enumerate(kernel):
        ax2.text(i, val + 0.05, f'{val:.1f}', ha='center', fontsize=10)
    
    # Output
    ax3 = plt.subplot(3, 1, 3)
    ax3.plot(signal, 'b-', alpha=0.3, linewidth=1, label='Исходный')
    ax3.plot(range(len(kernel)//2, len(kernel)//2 + len(output)), 
            output, 'g-', linewidth=2, label='После свертки (сглаженный)')
    ax3.set_title('Результат 1D свертки', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Время')
    ax3.set_ylabel('Значение')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    plt.suptitle('1D Convolution для временных рядов', fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    return fig_to_base64(fig)

def generate_3d_convolution():
    """Visualize 3D convolution concept for video."""
    fig = plt.figure(figsize=(14, 6))
    ax = fig.add_subplot(111, projection='3d')
    
    # Draw input volume (video frames)
    def draw_cube(ax, x, y, z, dx, dy, dz, color, alpha=0.3, label=None):
        """Draw a 3D cube."""
        xx = [x, x, x+dx, x+dx, x, x, x+dx, x+dx]
        yy = [y, y+dy, y+dy, y, y, y+dy, y+dy, y]
        zz = [z, z, z, z, z+dz, z+dz, z+dz, z+dz]
        
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        
        # Define the 6 faces of the cube
        faces = [
            [xx[0:4]],  # bottom
            [[xx[4], xx[5], xx[6], xx[7]]],  # top
            [[xx[0], xx[1], xx[5], xx[4]]],  # front
            [[xx[2], xx[3], xx[7], xx[6]]],  # back
            [[xx[0], xx[3], xx[7], xx[4]]],  # left
            [[xx[1], xx[2], xx[6], xx[5]]],  # right
        ]
        
        for face_verts in faces:
            face_verts = [[xx[j], yy[j], zz[j]] for j in [0, 1, 2, 3]]
            poly = Poly3DCollection([face_verts], alpha=alpha, facecolor=color, 
                                   edgecolor='black', linewidth=1)
            ax.add_collection3d(poly)
    
    # Input volume (video: time x height x width)
    draw_cube(ax, 0, 0, 0, 8, 6, 6, 'lightblue', alpha=0.3)
    ax.text(4, 3, -1, 'Input\n16×224×224', ha='center', fontsize=10, fontweight='bold')
    ax.text(4, -1, 3, 'Время (кадры)', ha='center', fontsize=9, style='italic')
    ax.text(-1, 3, 3, 'H', ha='center', fontsize=9, style='italic')
    ax.text(4, 7, 3, 'W', ha='center', fontsize=9, style='italic')
    
    # 3D Kernel
    draw_cube(ax, 10, 2, 2, 2, 2, 2, 'orange', alpha=0.5)
    ax.text(11, 3, 0.5, '3D Kernel\n3×3×3', ha='center', fontsize=10, fontweight='bold')
    
    # Arrow
    from mpl_toolkits.mplot3d import proj3d
    ax.plot([8.5, 9.5], [3, 3], [3, 3], 'k-', linewidth=2)
    ax.plot([9.5], [3], [3], 'k>', markersize=10)
    
    # Output volume
    draw_cube(ax, 14, 0.5, 0.5, 7, 5, 5, 'lightgreen', alpha=0.3)
    ax.text(17.5, 3, -1, 'Output\n14×222×222', ha='center', fontsize=10, fontweight='bold')
    
    ax.set_xlim([0, 22])
    ax.set_ylim([0, 8])
    ax.set_zlim([0, 8])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.view_init(elev=20, azim=45)
    
    plt.title('3D Convolution для видео (D×H×W)', fontsize=14, fontweight='bold', pad=20)
    
    return fig_to_base64(fig)

def generate_receptive_field_comparison():
    """Compare receptive fields for different convolution types."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    
    # 1D CNN receptive field
    ax1 = axes[0]
    time_steps = np.arange(10)
    signal = np.sin(time_steps)
    
    ax1.plot(time_steps, signal, 'b-o', linewidth=2, markersize=8, label='Сигнал')
    ax1.axvspan(3, 6, alpha=0.3, color='red', label='Receptive field\n(kernel=3)')
    ax1.set_title('1D CNN\nReceptive Field', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Время')
    ax1.set_ylabel('Значение')
    ax1.legend(loc='upper right', fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    # 2D CNN receptive field
    ax2 = axes[1]
    img = np.random.rand(8, 8)
    ax2.imshow(img, cmap='gray', alpha=0.5)
    
    # Draw receptive field
    rect = Rectangle((2.5, 2.5), 3, 3, linewidth=3, edgecolor='red', 
                     facecolor='red', alpha=0.3, label='Receptive field\n(3×3)')
    ax2.add_patch(rect)
    ax2.set_title('2D CNN\nReceptive Field', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Ширина')
    ax2.set_ylabel('Высота')
    ax2.legend(loc='upper right', fontsize=8)
    ax2.set_xticks([])
    ax2.set_yticks([])
    
    # 3D CNN receptive field
    ax3 = axes[2]
    ax3 = fig.add_subplot(1, 3, 3, projection='3d')
    
    # Draw a simple 3D grid
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    
    # Define cube for receptive field
    r = [0, 3]
    X, Y = np.meshgrid(r, r)
    
    # Draw edges of the cube
    for s, e in [([0, 0, 0], [3, 0, 0]), ([0, 0, 0], [0, 3, 0]), ([0, 0, 0], [0, 0, 3]),
                 ([3, 0, 0], [3, 3, 0]), ([3, 0, 0], [3, 0, 3]),
                 ([0, 3, 0], [3, 3, 0]), ([0, 3, 0], [0, 3, 3]),
                 ([0, 0, 3], [3, 0, 3]), ([0, 0, 3], [0, 3, 3]),
                 ([3, 3, 0], [3, 3, 3]), ([3, 0, 3], [3, 3, 3]), ([0, 3, 3], [3, 3, 3])]:
        ax3.plot([s[0], e[0]], [s[1], e[1]], [s[2], e[2]], 'r-', linewidth=3)
    
    # Fill the cube
    vertices = [[0, 0, 0], [3, 0, 0], [3, 3, 0], [0, 3, 0],
                [0, 0, 3], [3, 0, 3], [3, 3, 3], [0, 3, 3]]
    
    ax3.set_xlim([0, 5])
    ax3.set_ylim([0, 5])
    ax3.set_zlim([0, 5])
    ax3.set_xlabel('W')
    ax3.set_ylabel('H')
    ax3.set_zlabel('D (время)')
    ax3.set_title('3D CNN\nReceptive Field\n(3×3×3)', fontsize=12, fontweight='bold')
    
    plt.suptitle('Сравнение Receptive Fields', fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    return fig_to_base64(fig)

# ============================================================================
# MAIN FUNCTION
# ============================================================================

def generate_all_illustrations():
    """Generate all CNN and Pooling illustrations."""
    print("Generating CNN and Pooling illustrations...")
    
    illustrations = {}
    
    print("  - Generating CNN basics illustrations...")
    illustrations['cnn_convolution'] = generate_convolution_operation()
    print("    ✓ Convolution operation")
    
    illustrations['cnn_feature_maps'] = generate_feature_maps()
    print("    ✓ Feature maps")
    
    illustrations['cnn_architecture'] = generate_cnn_architecture()
    print("    ✓ CNN architecture")
    
    print("  - Generating Pooling illustrations...")
    illustrations['pooling_comparison'] = generate_pooling_comparison()
    print("    ✓ Max vs Average pooling")
    
    illustrations['pooling_global'] = generate_global_pooling()
    print("    ✓ Global pooling")
    
    print("  - Generating 1D/3D CNN illustrations...")
    illustrations['1d_convolution'] = generate_1d_convolution()
    print("    ✓ 1D convolution")
    
    illustrations['3d_convolution'] = generate_3d_convolution()
    print("    ✓ 3D convolution")
    
    illustrations['receptive_field'] = generate_receptive_field_comparison()
    print("    ✓ Receptive field comparison")
    
    print("All illustrations generated successfully!")
    return illustrations

if __name__ == '__main__':
    illustrations = generate_all_illustrations()
    print(f"\nGenerated {len(illustrations)} illustrations")
    print("Illustrations are base64 encoded and ready for HTML embedding")
