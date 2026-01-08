#!/usr/bin/env python3
"""
Generate matplotlib illustrations for meta-learning and few-shot learning cheatsheets.
This script creates high-quality visualizations and encodes them as base64 for inline HTML embedding.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
from matplotlib.collections import LineCollection
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
# META-LEARNING ILLUSTRATIONS
# ============================================================================

def generate_maml_inner_outer_loop():
    """Generate visualization of MAML inner and outer loops."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis('off')
    
    # Title
    ax.text(5, 5.5, 'MAML: Inner и Outer Loops', 
            ha='center', va='top', fontsize=14, weight='bold')
    
    # Initial parameters θ
    ax.add_patch(Circle((1, 4), 0.3, color='#1a5fb4', alpha=0.7))
    ax.text(1, 4, 'θ', ha='center', va='center', color='white', weight='bold', fontsize=12)
    ax.text(1, 3.3, 'Мета-параметры', ha='center', va='top', fontsize=9)
    
    # Task 1 branch
    arrow1 = FancyArrowPatch((1.3, 4.2), (3, 4.5), 
                            arrowstyle='->', mutation_scale=20, linewidth=2, color='#26a269')
    ax.add_patch(arrow1)
    ax.text(2.15, 4.6, 'Inner Loop\n(Task 1)', ha='center', fontsize=8)
    
    ax.add_patch(Circle((3.5, 4.5), 0.25, color='#26a269', alpha=0.7))
    ax.text(3.5, 4.5, "θ'₁", ha='center', va='center', color='white', weight='bold', fontsize=10)
    ax.text(3.5, 3.95, 'Support set\nадаптация', ha='center', va='top', fontsize=7)
    
    # Task 2 branch
    arrow2 = FancyArrowPatch((1.3, 4), (3, 3.5), 
                            arrowstyle='->', mutation_scale=20, linewidth=2, color='#e66100')
    ax.add_patch(arrow2)
    ax.text(2.15, 3.7, 'Inner Loop\n(Task 2)', ha='center', fontsize=8)
    
    ax.add_patch(Circle((3.5, 3.5), 0.25, color='#e66100', alpha=0.7))
    ax.text(3.5, 3.5, "θ'₂", ha='center', va='center', color='white', weight='bold', fontsize=10)
    ax.text(3.5, 2.95, 'Support set\nадаптация', ha='center', va='top', fontsize=7)
    
    # Task 3 branch
    arrow3 = FancyArrowPatch((1.3, 3.8), (3, 2.5), 
                            arrowstyle='->', mutation_scale=20, linewidth=2, color='#9c27b0')
    ax.add_patch(arrow3)
    ax.text(2.15, 2.8, 'Inner Loop\n(Task 3)', ha='center', fontsize=8)
    
    ax.add_patch(Circle((3.5, 2.5), 0.25, color='#9c27b0', alpha=0.7))
    ax.text(3.5, 2.5, "θ'₃", ha='center', va='center', color='white', weight='bold', fontsize=10)
    ax.text(3.5, 1.95, 'Support set\nадаптация', ha='center', va='top', fontsize=7)
    
    # Query set evaluation boxes
    ax.add_patch(FancyBboxPatch((4.5, 4.25), 1.5, 0.5, 
                                boxstyle="round,pad=0.05", 
                                edgecolor='#26a269', facecolor='#f0f9f4', linewidth=2))
    ax.text(5.25, 4.5, 'Query Loss 1', ha='center', va='center', fontsize=8)
    
    ax.add_patch(FancyBboxPatch((4.5, 3.25), 1.5, 0.5, 
                                boxstyle="round,pad=0.05", 
                                edgecolor='#e66100', facecolor='#fff3e0', linewidth=2))
    ax.text(5.25, 3.5, 'Query Loss 2', ha='center', va='center', fontsize=8)
    
    ax.add_patch(FancyBboxPatch((4.5, 2.25), 1.5, 0.5, 
                                boxstyle="round,pad=0.05", 
                                edgecolor='#9c27b0', facecolor='#f3e5f5', linewidth=2))
    ax.text(5.25, 2.5, 'Query Loss 3', ha='center', va='center', fontsize=8)
    
    # Arrows to query losses
    for y_pos in [4.5, 3.5, 2.5]:
        arrow = FancyArrowPatch((3.75, y_pos), (4.5, y_pos), 
                               arrowstyle='->', mutation_scale=15, linewidth=1.5, color='gray')
        ax.add_patch(arrow)
    
    # Outer loop convergence
    outer_arrow = FancyArrowPatch((6.5, 3.5), (8, 3.5), 
                                 arrowstyle='->', mutation_scale=25, linewidth=3, color='#1a5fb4')
    ax.add_patch(outer_arrow)
    ax.text(7.25, 3.85, 'Outer Loop', ha='center', fontsize=9, weight='bold')
    ax.text(7.25, 3.15, 'Мета-градиент', ha='center', fontsize=8)
    
    # Updated θ
    ax.add_patch(Circle((9, 3.5), 0.35, color='#1a5fb4', alpha=0.9))
    ax.text(9, 3.5, 'θ*', ha='center', va='center', color='white', weight='bold', fontsize=12)
    ax.text(9, 2.7, 'Обновленные\nпараметры', ha='center', va='top', fontsize=9)
    
    return fig_to_base64(fig)

def generate_nway_kshot_illustration():
    """Generate N-way K-shot learning illustration."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # 5-way 1-shot
    ax = axes[0]
    ax.set_xlim(0, 6)
    ax.set_ylim(0, 6)
    ax.axis('off')
    ax.set_title('5-way 1-shot Learning', fontsize=12, weight='bold')
    
    colors = ['#e63946', '#f4a261', '#2a9d8f', '#264653', '#8b5cf6']
    class_names = ['Класс A', 'Класс B', 'Класс C', 'Класс D', 'Класс E']
    
    # Support set (1 example per class)
    ax.text(0.5, 5.5, 'Support Set (K=1):', fontsize=10, weight='bold')
    for i, (color, name) in enumerate(zip(colors, class_names)):
        y_pos = 5 - i * 0.7
        ax.add_patch(Circle((1, y_pos), 0.2, color=color, alpha=0.7, edgecolor='black', linewidth=1.5))
        ax.text(1.6, y_pos, name, va='center', fontsize=9)
    
    # Query set
    ax.text(3.5, 5.5, 'Query Set:', fontsize=10, weight='bold')
    query_positions = [4.5, 5, 4, 4.5, 4.8]
    for i, (color, y_pos) in enumerate(zip(colors, query_positions)):
        y = 5 - i * 0.7
        ax.add_patch(Circle((4, y), 0.15, color=color, alpha=0.5, edgecolor='gray', linewidth=1))
    
    ax.text(3, 0.8, 'Задача: классифицировать\nquery по 1 примеру/класс', 
            ha='center', fontsize=9, style='italic', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    # 5-way 5-shot
    ax = axes[1]
    ax.set_xlim(0, 6)
    ax.set_ylim(0, 6)
    ax.axis('off')
    ax.set_title('5-way 5-shot Learning', fontsize=12, weight='bold')
    
    # Support set (5 examples per class)
    ax.text(0.5, 5.5, 'Support Set (K=5):', fontsize=10, weight='bold')
    for i, (color, name) in enumerate(zip(colors, class_names)):
        y_pos = 5 - i * 0.7
        # Draw 5 examples per class
        for j in range(5):
            x_pos = 0.7 + j * 0.25
            size = 0.12
            ax.add_patch(Circle((x_pos, y_pos), size, color=color, alpha=0.7, edgecolor='black', linewidth=1))
        ax.text(2.2, y_pos, name, va='center', fontsize=9)
    
    # Query set
    ax.text(3.5, 5.5, 'Query Set:', fontsize=10, weight='bold')
    for i, (color, y_pos) in enumerate(zip(colors, query_positions)):
        y = 5 - i * 0.7
        ax.add_patch(Circle((4, y), 0.15, color=color, alpha=0.5, edgecolor='gray', linewidth=1))
    
    ax.text(3, 0.8, 'Задача: классифицировать\nquery по 5 примерам/класс', 
            ha='center', fontsize=9, style='italic', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
    
    plt.tight_layout()
    return fig_to_base64(fig)

def generate_meta_learning_performance():
    """Generate meta-learning vs standard learning performance curves."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Simulated data
    episodes = np.arange(0, 101)
    
    # Standard learning (slow improvement)
    standard = 50 + 30 * (1 - np.exp(-episodes / 40)) + np.random.normal(0, 2, len(episodes))
    standard = np.clip(standard, 40, 85)
    
    # Meta-learning (fast adaptation)
    meta = 40 + 45 * (1 - np.exp(-episodes / 15)) + np.random.normal(0, 2, len(episodes))
    meta = np.clip(meta, 35, 90)
    
    # Plot
    ax.plot(episodes, standard, 'o-', label='Standard Learning', 
            linewidth=2, markersize=4, alpha=0.7, color='#e63946')
    ax.plot(episodes, meta, 's-', label='Meta-Learning (MAML)', 
            linewidth=2, markersize=4, alpha=0.7, color='#2a9d8f')
    
    ax.set_xlabel('Количество эпизодов обучения', fontsize=11)
    ax.set_ylabel('Точность на новых задачах (%)', fontsize=11)
    ax.set_title('Сравнение скорости адаптации: Meta-Learning vs Standard Learning', 
                fontsize=12, weight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(30, 95)
    
    # Add annotation
    ax.annotate('Быстрая адаптация!', 
                xy=(20, meta[20]), xytext=(40, 75),
                arrowprops=dict(arrowstyle='->', color='#2a9d8f', lw=2),
                fontsize=10, color='#2a9d8f', weight='bold')
    
    return fig_to_base64(fig)

def generate_prototypical_network_visualization():
    """Generate prototypical network concept visualization."""
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.set_xlim(-1, 10)
    ax.set_ylim(-1, 8)
    ax.axis('off')
    
    # Title
    ax.text(4.5, 7.5, 'Prototypical Networks: Классификация по прототипам', 
            ha='center', fontsize=13, weight='bold')
    
    # Support examples
    ax.text(1, 6.5, 'Support Set:', fontsize=11, weight='bold')
    
    colors = ['#e63946', '#2a9d8f', '#f4a261']
    class_names = ['Класс 1', 'Класс 2', 'Класс 3']
    
    # Draw support examples and compute prototypes
    prototype_positions = []
    for i, (color, name) in enumerate(zip(colors, class_names)):
        y_base = 5.5 - i * 2
        
        # Support examples
        examples_x = [0.5, 1, 1.5]
        for x in examples_x:
            ax.add_patch(Circle((x, y_base), 0.15, color=color, alpha=0.6, edgecolor='black', linewidth=1.5))
        
        # Arrow to embedding space
        ax.annotate('', xy=(3, y_base), xytext=(2, y_base),
                   arrowprops=dict(arrowstyle='->', lw=2, color='gray'))
        ax.text(2.5, y_base + 0.3, 'Encoder', fontsize=8, ha='center')
        
        # Embeddings in feature space
        for j, x_offset in enumerate([-0.15, 0, 0.15]):
            ax.add_patch(Circle((3.5 + x_offset, y_base + (j-1)*0.12), 0.1, 
                               color=color, alpha=0.5, edgecolor=color, linewidth=1))
        
        # Prototype (mean)
        proto_x, proto_y = 5, y_base
        ax.add_patch(Circle((proto_x, proto_y), 0.25, color=color, alpha=0.9, 
                           edgecolor='black', linewidth=2.5))
        ax.text(proto_x, proto_y, 'P', ha='center', va='center', 
               color='white', weight='bold', fontsize=11)
        prototype_positions.append((proto_x, proto_y))
        
        ax.text(5.7, proto_y, f'{name}\nпрототип', va='center', fontsize=9)
        
        # Arrow from embeddings to prototype
        ax.annotate('', xy=(4.75, proto_y), xytext=(3.8, y_base),
                   arrowprops=dict(arrowstyle='->', lw=1.5, color=color, linestyle='--'))
    
    # Query example
    ax.text(7, 6.5, 'Query:', fontsize=11, weight='bold')
    query_y = 4
    ax.add_patch(Circle((7.5, query_y), 0.2, color='purple', alpha=0.7, 
                       edgecolor='black', linewidth=2))
    ax.text(7.5, 3.5, 'Новый\nпример', ha='center', fontsize=9)
    
    # Query embedding
    ax.annotate('', xy=(8.5, query_y), xytext=(7.8, query_y),
               arrowprops=dict(arrowstyle='->', lw=2, color='gray'))
    ax.add_patch(Circle((8.8, query_y), 0.15, color='purple', alpha=0.7, 
                       edgecolor='purple', linewidth=1.5))
    
    # Distance calculations to prototypes
    for i, (proto_x, proto_y) in enumerate(prototype_positions):
        # Draw distance line
        distance = np.sqrt((8.8 - proto_x)**2 + (query_y - proto_y)**2)
        ax.plot([8.8, proto_x], [query_y, proto_y], 
               linestyle=':', linewidth=1.5, color=colors[i], alpha=0.6)
        
        # Distance label
        mid_x, mid_y = (8.8 + proto_x) / 2, (query_y + proto_y) / 2
        ax.text(mid_x, mid_y, f'd{i+1}', fontsize=8, 
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, pad=0.2))
    
    # Prediction
    ax.text(4.5, 0.5, 'Предсказание = argmin(distance)', 
           ha='center', fontsize=10, weight='bold',
           bbox=dict(boxstyle='round', facecolor='#e6f0ff', alpha=0.8, pad=0.5))
    
    return fig_to_base64(fig)

# ============================================================================
# FEW-SHOT LEARNING ILLUSTRATIONS
# ============================================================================

def generate_siamese_network_architecture():
    """Generate Siamese network architecture illustration."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis('off')
    
    # Title
    ax.text(5, 5.7, 'Siamese Network Architecture', ha='center', fontsize=13, weight='bold')
    
    # Input images
    ax.add_patch(FancyBboxPatch((0.5, 3.5), 1, 1, boxstyle="round,pad=0.05", 
                                edgecolor='#1a5fb4', facecolor='#e6f0ff', linewidth=2))
    ax.text(1, 4, 'Image 1', ha='center', va='center', fontsize=9, weight='bold')
    
    ax.add_patch(FancyBboxPatch((0.5, 1), 1, 1, boxstyle="round,pad=0.05", 
                                edgecolor='#1a5fb4', facecolor='#e6f0ff', linewidth=2))
    ax.text(1, 1.5, 'Image 2', ha='center', va='center', fontsize=9, weight='bold')
    
    # Twin networks (shared weights)
    for y_pos, label in [(4, 'CNN 1'), (1.5, 'CNN 2')]:
        # Convolution layers
        for i, x in enumerate([2.2, 2.8, 3.4]):
            width = 0.4 - i * 0.05
            height = 0.8 - i * 0.1
            y_offset = (0.8 - height) / 2
            ax.add_patch(FancyBboxPatch((x, y_pos - 0.4 + y_offset), width, height, 
                                       boxstyle="round,pad=0.02", 
                                       edgecolor='#26a269', facecolor='#f0f9f4', linewidth=1.5))
    
    # Shared weights indicator
    ax.plot([2.5, 2.5], [2.2, 3.3], 'k--', linewidth=2, alpha=0.5)
    ax.text(2.5, 2.75, 'Общие\nвеса', ha='center', fontsize=8, 
           bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))
    
    # Feature vectors
    ax.add_patch(FancyBboxPatch((4.2, 3.7), 0.6, 0.6, 
                                edgecolor='#e66100', facecolor='#fff3e0', linewidth=2))
    ax.text(4.5, 4, 'f(x₁)', ha='center', va='center', fontsize=10, weight='bold')
    
    ax.add_patch(FancyBboxPatch((4.2, 1.2), 0.6, 0.6, 
                                edgecolor='#e66100', facecolor='#fff3e0', linewidth=2))
    ax.text(4.5, 1.5, 'f(x₂)', ha='center', va='center', fontsize=10, weight='bold')
    
    # Distance calculation
    ax.add_patch(Circle((6, 2.75), 0.5, color='#9c27b0', alpha=0.7))
    ax.text(6, 2.75, '||f(x₁) -\nf(x₂)||', ha='center', va='center', 
           color='white', fontsize=9, weight='bold')
    ax.text(6, 1.9, 'Distance', ha='center', fontsize=8)
    
    # Arrows
    for y in [4, 1.5]:
        ax.annotate('', xy=(4.2, y), xytext=(3.9, y),
                   arrowprops=dict(arrowstyle='->', lw=1.5, color='gray'))
        ax.annotate('', xy=(5.5, 2.75 if y > 2 else 2.75), xytext=(4.8, y),
                   arrowprops=dict(arrowstyle='->', lw=1.5, color='gray'))
    
    # Output
    ax.add_patch(FancyBboxPatch((7.5, 2.3), 1.5, 0.9, 
                                boxstyle="round,pad=0.1", 
                                edgecolor='#e63946', facecolor='#fdf0f2', linewidth=2))
    ax.text(8.25, 2.75, 'Similarity\nScore', ha='center', va='center', 
           fontsize=10, weight='bold')
    
    ax.annotate('', xy=(7.5, 2.75), xytext=(6.5, 2.75),
               arrowprops=dict(arrowstyle='->', lw=2, color='gray'))
    
    # Labels
    ax.text(8.25, 1.5, 'Similar: > threshold\nDifferent: < threshold', 
           ha='center', fontsize=8, style='italic')
    
    return fig_to_base64(fig)

def generate_contrastive_triplet_loss():
    """Generate illustration of contrastive and triplet loss."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Contrastive Loss
    ax = axes[0]
    ax.set_xlim(-1, 5)
    ax.set_ylim(-1, 5)
    ax.axis('off')
    ax.set_title('Contrastive Loss', fontsize=12, weight='bold')
    
    # Positive pair
    ax.add_patch(Circle((1, 3.5), 0.3, color='#2a9d8f', alpha=0.7, edgecolor='black', linewidth=2))
    ax.text(1, 3.5, 'x₁', ha='center', va='center', color='white', weight='bold')
    ax.add_patch(Circle((2.5, 3.5), 0.3, color='#2a9d8f', alpha=0.7, edgecolor='black', linewidth=2))
    ax.text(2.5, 3.5, 'x₂', ha='center', va='center', color='white', weight='bold')
    
    ax.plot([1.3, 2.2], [3.5, 3.5], 'g-', linewidth=3)
    ax.text(1.75, 3.8, 'minimize', ha='center', fontsize=9, color='green', weight='bold')
    ax.text(1.75, 2.9, 'Same class (y=0)', ha='center', fontsize=8, style='italic')
    
    # Negative pair
    ax.add_patch(Circle((1, 1), 0.3, color='#e63946', alpha=0.7, edgecolor='black', linewidth=2))
    ax.text(1, 1, 'x₁', ha='center', va='center', color='white', weight='bold')
    ax.add_patch(Circle((3.5, 1), 0.3, color='#f4a261', alpha=0.7, edgecolor='black', linewidth=2))
    ax.text(3.5, 1, 'x₃', ha='center', va='center', color='white', weight='bold')
    
    ax.plot([1.3, 3.2], [1, 1], 'r--', linewidth=3)
    ax.text(2.25, 1.35, 'maximize', ha='center', fontsize=9, color='red', weight='bold')
    ax.text(2.25, 0.5, 'Different class (y=1)', ha='center', fontsize=8, style='italic')
    
    # Triplet Loss
    ax = axes[1]
    ax.set_xlim(-1, 6)
    ax.set_ylim(-1, 5)
    ax.axis('off')
    ax.set_title('Triplet Loss', fontsize=12, weight='bold')
    
    # Anchor
    ax.add_patch(Circle((1, 2.5), 0.35, color='#1a5fb4', alpha=0.8, edgecolor='black', linewidth=2))
    ax.text(1, 2.5, 'A', ha='center', va='center', color='white', weight='bold', fontsize=12)
    ax.text(1, 1.8, 'Anchor', ha='center', fontsize=9, weight='bold')
    
    # Positive
    ax.add_patch(Circle((3, 3.5), 0.3, color='#26a269', alpha=0.7, edgecolor='black', linewidth=2))
    ax.text(3, 3.5, 'P', ha='center', va='center', color='white', weight='bold', fontsize=11)
    ax.text(3, 4.2, 'Positive', ha='center', fontsize=9, weight='bold')
    
    # Negative
    ax.add_patch(Circle((3, 1.5), 0.3, color='#e63946', alpha=0.7, edgecolor='black', linewidth=2))
    ax.text(3, 1.5, 'N', ha='center', va='center', color='white', weight='bold', fontsize=11)
    ax.text(3, 0.8, 'Negative', ha='center', fontsize=9, weight='bold')
    
    # Distance lines
    ax.plot([1.35, 2.7], [2.7, 3.3], 'g-', linewidth=2.5)
    ax.text(2, 3.3, 'd(A,P)', fontsize=9, color='green', weight='bold',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax.plot([1.35, 2.7], [2.3, 1.7], 'r-', linewidth=2.5)
    ax.text(2, 1.7, 'd(A,N)', fontsize=9, color='red', weight='bold',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Loss formula
    ax.text(4.5, 2.5, 'Loss = max(0,\nd(A,P) - d(A,N)\n+ margin)', 
           ha='center', va='center', fontsize=9,
           bbox=dict(boxstyle='round', facecolor='#e6f0ff', alpha=0.9, pad=0.5))
    
    ax.text(3, -0.2, 'Goal: d(A,P) < d(A,N)', ha='center', fontsize=9, 
           weight='bold', style='italic')
    
    plt.tight_layout()
    return fig_to_base64(fig)

def generate_few_shot_accuracy_comparison():
    """Generate few-shot learning accuracy comparison across methods."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    methods = ['Siamese\nNetworks', 'Matching\nNetworks', 'Prototypical\nNetworks', 
               'MAML', 'Relation\nNetworks']
    one_shot = [68.5, 72.3, 75.8, 78.2, 76.4]
    five_shot = [82.1, 84.7, 87.3, 89.5, 88.1]
    
    x = np.arange(len(methods))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, one_shot, width, label='1-shot', 
                   color='#e63946', alpha=0.8, edgecolor='black', linewidth=1)
    bars2 = ax.bar(x + width/2, five_shot, width, label='5-shot', 
                   color='#2a9d8f', alpha=0.8, edgecolor='black', linewidth=1)
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                   f'{height:.1f}%', ha='center', va='bottom', fontsize=8)
    
    ax.set_xlabel('Методы', fontsize=11, weight='bold')
    ax.set_ylabel('Точность на mini-ImageNet (%)', fontsize=11, weight='bold')
    ax.set_title('Сравнение методов Few-Shot Learning (5-way classification)', 
                fontsize=12, weight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=9)
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(60, 95)
    
    return fig_to_base64(fig)

def generate_episodic_training():
    """Generate episodic training illustration."""
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 7)
    ax.axis('off')
    
    # Title
    ax.text(5.5, 6.7, 'Episodic Training: N-way K-shot Episodes', 
           ha='center', fontsize=13, weight='bold')
    
    # Dataset
    ax.add_patch(FancyBboxPatch((0.2, 4.5), 2, 1.8, boxstyle="round,pad=0.1", 
                                edgecolor='#1a5fb4', facecolor='#e6f0ff', linewidth=2))
    ax.text(1.2, 6, 'Dataset', ha='center', fontsize=10, weight='bold')
    ax.text(1.2, 5.5, 'Множество\nклассов', ha='center', fontsize=8)
    
    # Sample N classes
    ax.annotate('', xy=(2.8, 5.4), xytext=(2.3, 5.4),
               arrowprops=dict(arrowstyle='->', lw=2, color='gray'))
    ax.text(2.55, 5.7, 'Sample\nN классов', ha='center', fontsize=8)
    
    # Episode
    ax.add_patch(FancyBboxPatch((3, 3.5), 4.5, 3, boxstyle="round,pad=0.1", 
                                edgecolor='#26a269', facecolor='#f0f9f4', linewidth=3))
    ax.text(5.25, 6.2, 'Episode (5-way 5-shot)', ha='center', fontsize=11, weight='bold')
    
    # Support set
    ax.text(3.5, 5.5, 'Support Set:', fontsize=9, weight='bold')
    colors = ['#e63946', '#f4a261', '#2a9d8f', '#264653', '#8b5cf6']
    for i, color in enumerate(colors):
        y = 5.2 - i * 0.35
        for j in range(5):
            ax.add_patch(Circle((4 + j * 0.25, y), 0.08, color=color, alpha=0.7))
    
    ax.text(6.2, 4.2, 'K=5 examples\nper class', fontsize=8, style='italic')
    
    # Query set
    ax.text(3.5, 3.8, 'Query Set:', fontsize=9, weight='bold')
    for i, color in enumerate(colors):
        y = 3.6 - i * 0.2
        for j in range(3):
            ax.add_patch(Circle((4.2 + j * 0.25, y), 0.06, color=color, alpha=0.5,
                               edgecolor='gray', linewidth=0.5))
    
    # Model
    ax.annotate('', xy=(8.5, 5), xytext=(7.6, 5),
               arrowprops=dict(arrowstyle='->', lw=2.5, color='#1a5fb4'))
    
    ax.add_patch(FancyBboxPatch((8.5, 4.3), 1.5, 1.4, boxstyle="round,pad=0.1", 
                                edgecolor='#9c27b0', facecolor='#f3e5f5', linewidth=2))
    ax.text(9.25, 5, 'Model', ha='center', va='center', fontsize=10, weight='bold')
    
    # Loss
    ax.add_patch(Circle((9.25, 3.3), 0.4, color='#e66100', alpha=0.7))
    ax.text(9.25, 3.3, 'Loss', ha='center', va='center', color='white', 
           fontsize=9, weight='bold')
    ax.annotate('', xy=(9.25, 3.7), xytext=(9.25, 4.3),
               arrowprops=dict(arrowstyle='->', lw=2, color='gray'))
    
    # Update
    ax.annotate('', xy=(2, 2.5), xytext=(9, 2.5),
               arrowprops=dict(arrowstyle='->', lw=3, color='#1a5fb4',
                             connectionstyle="arc3,rad=.3"))
    ax.text(5.5, 1.8, 'Update Model', ha='center', fontsize=10, weight='bold')
    
    # Iteration
    ax.text(1, 2, 'Repeat for\nmany episodes', ha='center', fontsize=9, style='italic',
           bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))
    
    # Meta-train vs Meta-test
    ax.text(5.5, 0.8, 'Meta-train: используем train классы  |  Meta-test: unseen классы', 
           ha='center', fontsize=9, weight='bold',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5, pad=0.5))
    
    return fig_to_base64(fig)

# ============================================================================
# MAIN FUNCTION
# ============================================================================

def generate_all_illustrations():
    """Generate all meta-learning and few-shot learning illustrations."""
    print("Generating meta-learning and few-shot learning illustrations...")
    
    illustrations = {}
    
    # Meta-learning illustrations
    print("  - MAML inner/outer loop...")
    illustrations['maml_inner_outer_loop'] = generate_maml_inner_outer_loop()
    
    print("  - N-way K-shot illustration...")
    illustrations['nway_kshot'] = generate_nway_kshot_illustration()
    
    print("  - Meta-learning performance...")
    illustrations['meta_learning_performance'] = generate_meta_learning_performance()
    
    print("  - Prototypical network visualization...")
    illustrations['prototypical_network'] = generate_prototypical_network_visualization()
    
    # Few-shot learning illustrations
    print("  - Siamese network architecture...")
    illustrations['siamese_network'] = generate_siamese_network_architecture()
    
    print("  - Contrastive and triplet loss...")
    illustrations['contrastive_triplet_loss'] = generate_contrastive_triplet_loss()
    
    print("  - Few-shot accuracy comparison...")
    illustrations['few_shot_accuracy'] = generate_few_shot_accuracy_comparison()
    
    print("  - Episodic training...")
    illustrations['episodic_training'] = generate_episodic_training()
    
    print("✓ All illustrations generated successfully!")
    return illustrations

if __name__ == '__main__':
    illustrations = generate_all_illustrations()
    print(f"\nGenerated {len(illustrations)} illustrations")
    for key in illustrations:
        print(f"  - {key}: {len(illustrations[key])} bytes (base64)")
