#!/usr/bin/env python3
"""
Generate matplotlib illustrations for transfer learning cheatsheets.
This script creates high-quality visualizations and encodes them as base64 for inline HTML embedding.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification, make_blobs
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import norm
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
# TRANSFER LEARNING CONCEPT ILLUSTRATIONS
# ============================================================================

def generate_transfer_learning_concept():
    """Generate visualization of transfer learning concept (source → target)."""
    np.random.seed(42)
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    # Source domain - large labeled dataset
    X_source = np.random.randn(200, 2) * 1.5 + np.array([2, 2])
    y_source = (X_source[:, 0] + X_source[:, 1] > 4).astype(int)
    
    axes[0].scatter(X_source[y_source==0, 0], X_source[y_source==0, 1], 
                   c='blue', alpha=0.6, s=50, edgecolors='black', linewidth=0.5, label='Класс 0')
    axes[0].scatter(X_source[y_source==1, 0], X_source[y_source==1, 1], 
                   c='red', alpha=0.6, s=50, edgecolors='black', linewidth=0.5, label='Класс 1')
    axes[0].set_title('Source Domain\n(большой размеченный датасет)', fontsize=11, fontweight='bold')
    axes[0].set_xlabel('Признак 1')
    axes[0].set_ylabel('Признак 2')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Transfer arrow
    axes[1].text(0.5, 0.6, 'Transfer\nLearning', ha='center', va='center',
                fontsize=20, fontweight='bold', color='#1a5fb4')
    axes[1].arrow(0.3, 0.5, 0.35, 0, head_width=0.1, head_length=0.08, 
                 fc='#1a5fb4', ec='#1a5fb4', linewidth=3)
    axes[1].text(0.5, 0.3, 'Перенос\nзнаний', ha='center', va='center',
                fontsize=12, style='italic', color='gray')
    axes[1].set_xlim(0, 1)
    axes[1].set_ylim(0, 1)
    axes[1].axis('off')
    
    # Target domain - small labeled dataset
    X_target = np.random.randn(40, 2) * 1.2 + np.array([2.5, 2.5])
    y_target = (X_target[:, 0] + X_target[:, 1] > 5).astype(int)
    
    # Add many unlabeled points
    X_target_unlabeled = np.random.randn(150, 2) * 1.2 + np.array([2.5, 2.5])
    
    axes[2].scatter(X_target_unlabeled[:, 0], X_target_unlabeled[:, 1],
                   c='lightgray', alpha=0.3, s=30, label='Неразмеченные')
    axes[2].scatter(X_target[y_target==0, 0], X_target[y_target==0, 1], 
                   c='blue', alpha=0.8, s=80, edgecolors='black', linewidth=1, label='Класс 0')
    axes[2].scatter(X_target[y_target==1, 0], X_target[y_target==1, 1], 
                   c='red', alpha=0.8, s=80, edgecolors='black', linewidth=1, label='Класс 1')
    axes[2].set_title('Target Domain\n(малый размеченный датасет)', fontsize=11, fontweight='bold')
    axes[2].set_xlabel('Признак 1')
    axes[2].set_ylabel('Признак 2')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig_to_base64(fig)

def generate_transfer_learning_types():
    """Generate visualization of different transfer learning types."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    np.random.seed(42)
    
    # Inductive Transfer
    X1 = np.random.randn(100, 2) + np.array([0, 0])
    y1 = (X1[:, 0] + X1[:, 1] > 0).astype(int)
    X2 = np.random.randn(50, 2) + np.array([0.5, 0.5])
    y2 = (X2[:, 0] * 2 + X2[:, 1] > 1).astype(int)
    
    axes[0, 0].scatter(X1[y1==0, 0], X1[y1==0, 1], c='blue', alpha=0.4, s=40, label='Source: Задача A')
    axes[0, 0].scatter(X1[y1==1, 0], X1[y1==1, 1], c='red', alpha=0.4, s=40)
    axes[0, 0].scatter(X2[y2==0, 0], X2[y2==0, 1], c='green', alpha=0.7, s=60, 
                      edgecolors='black', linewidth=1, marker='s', label='Target: Задача B')
    axes[0, 0].scatter(X2[y2==1, 0], X2[y2==1, 1], c='orange', alpha=0.7, s=60,
                      edgecolors='black', linewidth=1, marker='s')
    axes[0, 0].set_title('Inductive Transfer\n(Разные задачи)', fontsize=11, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Transductive Transfer
    X_s = np.random.randn(100, 2) + np.array([0, 0])
    y_s = (X_s[:, 0] + X_s[:, 1] > 0).astype(int)
    X_t = np.random.randn(100, 2) * 1.5 + np.array([2, 2])
    y_t = (X_t[:, 0] + X_t[:, 1] > 4).astype(int)
    
    axes[0, 1].scatter(X_s[y_s==0, 0], X_s[y_s==0, 1], c='blue', alpha=0.4, s=40, label='Source Domain')
    axes[0, 1].scatter(X_s[y_s==1, 0], X_s[y_s==1, 1], c='red', alpha=0.4, s=40)
    axes[0, 1].scatter(X_t[y_t==0, 0], X_t[y_t==0, 1], c='blue', alpha=0.7, s=60,
                      edgecolors='black', linewidth=1, marker='s', label='Target Domain')
    axes[0, 1].scatter(X_t[y_t==1, 0], X_t[y_t==1, 1], c='red', alpha=0.7, s=60,
                      edgecolors='black', linewidth=1, marker='s')
    axes[0, 1].set_title('Transductive Transfer\n(Та же задача, разные домены)', fontsize=11, fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Feature-based Transfer
    axes[1, 0].text(0.5, 0.7, 'Feature-based Transfer', ha='center', fontsize=14, fontweight='bold')
    axes[1, 0].text(0.5, 0.55, '1. Извлечение признаков из source', ha='center', fontsize=10)
    axes[1, 0].arrow(0.3, 0.5, 0.4, 0, head_width=0.03, head_length=0.05, fc='blue', ec='blue')
    axes[1, 0].text(0.5, 0.40, '2. Применение к target данным', ha='center', fontsize=10)
    axes[1, 0].arrow(0.3, 0.35, 0.4, 0, head_width=0.03, head_length=0.05, fc='green', ec='green')
    axes[1, 0].text(0.5, 0.20, '3. Обучение на новых признаках', ha='center', fontsize=10)
    axes[1, 0].set_xlim(0, 1)
    axes[1, 0].set_ylim(0, 1)
    axes[1, 0].axis('off')
    
    # Parameter Transfer
    axes[1, 1].text(0.5, 0.7, 'Parameter Transfer', ha='center', fontsize=14, fontweight='bold')
    
    # Draw network diagram
    layer_positions = [0.15, 0.35, 0.55, 0.75]
    y_pos = 0.4
    
    for i, x in enumerate(layer_positions):
        circle = plt.Circle((x, y_pos), 0.04, color='skyblue', ec='black', linewidth=2)
        axes[1, 1].add_patch(circle)
        if i < len(layer_positions) - 1:
            axes[1, 1].arrow(x + 0.04, y_pos, 0.12, 0, head_width=0.02, head_length=0.03, 
                           fc='gray', ec='gray', linewidth=1.5)
    
    axes[1, 1].text(0.5, 0.25, 'Инициализация параметрами\nиз source модели', 
                   ha='center', fontsize=10, style='italic')
    axes[1, 1].text(0.5, 0.10, '→ Fine-tuning на target', ha='center', fontsize=10, color='green')
    axes[1, 1].set_xlim(0, 1)
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    return fig_to_base64(fig)

def generate_domain_shift():
    """Generate visualization of domain shift problem."""
    np.random.seed(42)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Source domain distributions
    X_source = np.random.randn(200, 2) * 1.0 + np.array([0, 0])
    y_source = (X_source[:, 0] + X_source[:, 1] > 0).astype(int)
    
    # Target domain distributions (shifted)
    X_target = np.random.randn(200, 2) * 1.2 + np.array([2, 1])
    y_target = (X_target[:, 0] + X_target[:, 1] > 3).astype(int)
    
    # Plot distributions
    axes[0].scatter(X_source[y_source==0, 0], X_source[y_source==0, 1],
                   c='blue', alpha=0.5, s=50, label='Source: Класс 0')
    axes[0].scatter(X_source[y_source==1, 0], X_source[y_source==1, 1],
                   c='red', alpha=0.5, s=50, label='Source: Класс 1')
    axes[0].scatter(X_target[y_target==0, 0], X_target[y_target==0, 1],
                   c='cyan', alpha=0.5, s=50, marker='s', label='Target: Класс 0')
    axes[0].scatter(X_target[y_target==1, 0], X_target[y_target==1, 1],
                   c='orange', alpha=0.5, s=50, marker='s', label='Target: Класс 1')
    
    axes[0].set_title('Domain Shift Problem\nP_source(X) ≠ P_target(X)', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Признак 1')
    axes[0].set_ylabel('Признак 2')
    axes[0].legend(loc='upper left', fontsize=8)
    axes[0].grid(True, alpha=0.3)
    
    # Feature distributions comparison
    feature_vals = np.linspace(-4, 6, 100)
    
    # Source distribution
    source_dist = norm.pdf(feature_vals, 0, 1.0)
    # Target distribution (shifted)
    target_dist = norm.pdf(feature_vals, 2, 1.2)
    
    axes[1].fill_between(feature_vals, source_dist, alpha=0.4, color='blue', label='P_source(X)')
    axes[1].fill_between(feature_vals, target_dist, alpha=0.4, color='orange', label='P_target(X)')
    axes[1].plot(feature_vals, source_dist, 'b-', linewidth=2)
    axes[1].plot(feature_vals, target_dist, 'orange', linewidth=2)
    
    # Add vertical lines for means
    axes[1].axvline(0, color='blue', linestyle='--', alpha=0.7, linewidth=1.5, label='μ_source')
    axes[1].axvline(2, color='orange', linestyle='--', alpha=0.7, linewidth=1.5, label='μ_target')
    
    axes[1].set_title('Распределения признаков\nв source и target доменах', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Значение признака')
    axes[1].set_ylabel('Плотность вероятности')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig_to_base64(fig)

# ============================================================================
# DOMAIN ADAPTATION ILLUSTRATIONS
# ============================================================================

def generate_domain_adaptation_methods():
    """Generate visualization comparing domain adaptation approaches."""
    np.random.seed(42)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Generate source and target data
    X_source = np.random.randn(150, 2) * 1.0 + np.array([0, 0])
    X_target = np.random.randn(150, 2) * 1.2 + np.array([2.5, 1.5])
    
    # No adaptation
    axes[0, 0].scatter(X_source[:, 0], X_source[:, 1], c='blue', alpha=0.5, s=40, label='Source')
    axes[0, 0].scatter(X_target[:, 0], X_target[:, 1], c='orange', alpha=0.5, s=40, label='Target')
    axes[0, 0].set_title('Без адаптации\n(Domain Shift)', fontsize=11, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_xlabel('Признак 1')
    axes[0, 0].set_ylabel('Признак 2')
    
    # Feature alignment (shift target towards source)
    X_target_aligned = (X_target - X_target.mean(axis=0)) + X_source.mean(axis=0)
    axes[0, 1].scatter(X_source[:, 0], X_source[:, 1], c='blue', alpha=0.5, s=40, label='Source')
    axes[0, 1].scatter(X_target_aligned[:, 0], X_target_aligned[:, 1], c='orange', alpha=0.5, s=40, label='Target (aligned)')
    axes[0, 1].set_title('Feature Alignment\n(Выравнивание признаков)', fontsize=11, fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_xlabel('Признак 1')
    axes[0, 1].set_ylabel('Признак 2')
    
    # MMD visualization (show kernel distances)
    from sklearn.metrics.pairwise import rbf_kernel
    
    # Sample points for visualization
    sample_s = X_source[:50]
    sample_t = X_target[:50]
    
    K_st = rbf_kernel(sample_s, sample_t, gamma=0.5)
    
    im = axes[1, 0].imshow(K_st, cmap='RdYlBu_r', aspect='auto', interpolation='nearest')
    axes[1, 0].set_title('MMD: Kernel Similarity\nSource vs Target', fontsize=11, fontweight='bold')
    axes[1, 0].set_xlabel('Target samples')
    axes[1, 0].set_ylabel('Source samples')
    plt.colorbar(im, ax=axes[1, 0], label='Similarity')
    
    # Instance weighting
    # Calculate weights based on distance to target distribution
    weights = np.exp(-0.5 * np.sum((X_source - X_target.mean(axis=0))**2, axis=1))
    weights = weights / weights.max()
    
    scatter = axes[1, 1].scatter(X_source[:, 0], X_source[:, 1], c=weights, 
                                cmap='viridis', s=50, alpha=0.7, edgecolors='black', linewidth=0.5)
    axes[1, 1].scatter(X_target[:, 0], X_target[:, 1], c='red', alpha=0.3, s=30, 
                      marker='x', label='Target distribution')
    axes[1, 1].set_title('Instance Weighting\n(Взвешивание source примеров)', fontsize=11, fontweight='bold')
    axes[1, 1].set_xlabel('Признак 1')
    axes[1, 1].set_ylabel('Признак 2')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=axes[1, 1], label='Вес примера')
    
    plt.tight_layout()
    return fig_to_base64(fig)

def generate_mmd_illustration():
    """Generate Maximum Mean Discrepancy visualization."""
    np.random.seed(42)
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    # Three scenarios: high MMD, medium MMD, low MMD
    scenarios = [
        ('Высокий MMD\n(сильное различие)', 0, 3.0),
        ('Средний MMD\n(умеренное различие)', 0, 1.5),
        ('Низкий MMD\n(слабое различие)', 0, 0.5)
    ]
    
    for idx, (title, source_mean, shift) in enumerate(scenarios):
        X_source = np.random.randn(100, 2) + np.array([source_mean, source_mean])
        X_target = np.random.randn(100, 2) + np.array([source_mean + shift, source_mean + shift])
        
        axes[idx].scatter(X_source[:, 0], X_source[:, 1], c='blue', alpha=0.5, s=50, label='Source')
        axes[idx].scatter(X_target[:, 0], X_target[:, 1], c='orange', alpha=0.5, s=50, label='Target')
        
        # Calculate and display MMD
        from sklearn.metrics.pairwise import rbf_kernel
        K_ss = rbf_kernel(X_source, X_source, gamma=1.0)
        K_tt = rbf_kernel(X_target, X_target, gamma=1.0)
        K_st = rbf_kernel(X_source, X_target, gamma=1.0)
        
        mmd_squared = K_ss.mean() + K_tt.mean() - 2 * K_st.mean()
        mmd = np.sqrt(max(0, mmd_squared))
        
        axes[idx].set_title(f'{title}\nMMD² = {mmd_squared:.3f}', fontsize=11, fontweight='bold')
        axes[idx].set_xlabel('Признак 1')
        axes[idx].set_ylabel('Признак 2')
        axes[idx].legend()
        axes[idx].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig_to_base64(fig)

# ============================================================================
# CNN TRANSFER LEARNING ILLUSTRATIONS
# ============================================================================

def generate_cnn_transfer_architecture():
    """Generate CNN transfer learning architecture visualization."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # Feature Extraction
    ax = axes[0]
    ax.text(0.5, 0.95, 'Feature Extraction: Заморозка backbone', 
           ha='center', va='top', fontsize=13, fontweight='bold', transform=ax.transAxes)
    
    # Draw layers
    layer_info = [
        (0.05, 'Input\n224×224×3', 'lightblue'),
        (0.15, 'Conv1\nЗаморожен', 'lightgray'),
        (0.25, 'Conv2\nЗаморожен', 'lightgray'),
        (0.35, 'Conv3\nЗаморожен', 'lightgray'),
        (0.45, 'Conv4\nЗаморожен', 'lightgray'),
        (0.55, 'Conv5\nЗаморожен', 'lightgray'),
        (0.70, 'Flatten', 'lightyellow'),
        (0.85, 'FC\nОбучается', 'lightgreen')
    ]
    
    for x, label, color in layer_info:
        rect = plt.Rectangle((x, 0.3), 0.08, 0.4, facecolor=color, 
                            edgecolor='black', linewidth=2, transform=ax.transAxes)
        ax.add_patch(rect)
        ax.text(x + 0.04, 0.5, label, ha='center', va='center', fontsize=9, 
               transform=ax.transAxes, fontweight='bold')
        
        if x < 0.8:
            ax.arrow(x + 0.09, 0.5, 0.04, 0, head_width=0.05, head_length=0.015,
                    fc='black', ec='black', transform=ax.transAxes)
    
    # Add legend
    ax.text(0.05, 0.15, '❄️ Заморожены: параметры не обновляются', 
           fontsize=10, color='gray', transform=ax.transAxes)
    ax.text(0.05, 0.08, '🔥 Обучаются: только последний слой', 
           fontsize=10, color='green', transform=ax.transAxes)
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    # Fine-tuning
    ax = axes[1]
    ax.text(0.5, 0.95, 'Fine-tuning: Обучение всей модели или части слоев', 
           ha='center', va='top', fontsize=13, fontweight='bold', transform=ax.transAxes)
    
    layer_info_ft = [
        (0.05, 'Input\n224×224×3', 'lightblue'),
        (0.15, 'Conv1\nLR=1e-5', 'lightyellow'),
        (0.25, 'Conv2\nLR=1e-5', 'lightyellow'),
        (0.35, 'Conv3\nLR=1e-4', 'yellow'),
        (0.45, 'Conv4\nLR=1e-4', 'yellow'),
        (0.55, 'Conv5\nLR=1e-3', 'orange'),
        (0.70, 'Flatten', 'lightyellow'),
        (0.85, 'FC\nLR=1e-2', 'lightgreen')
    ]
    
    for x, label, color in layer_info_ft:
        rect = plt.Rectangle((x, 0.3), 0.08, 0.4, facecolor=color, 
                            edgecolor='black', linewidth=2, transform=ax.transAxes)
        ax.add_patch(rect)
        ax.text(x + 0.04, 0.5, label, ha='center', va='center', fontsize=9, 
               transform=ax.transAxes, fontweight='bold')
        
        if x < 0.8:
            ax.arrow(x + 0.09, 0.5, 0.04, 0, head_width=0.05, head_length=0.015,
                    fc='black', ec='black', transform=ax.transAxes)
    
    # Add legend
    ax.text(0.05, 0.15, '🔥 Все слои обучаются', fontsize=10, color='green', transform=ax.transAxes)
    ax.text(0.05, 0.08, '📊 Discriminative LR: меньше для ранних слоев', 
           fontsize=10, color='orange', transform=ax.transAxes)
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    plt.tight_layout()
    return fig_to_base64(fig)

def generate_finetuning_strategies():
    """Generate visualization of different fine-tuning strategies."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    strategies = [
        ('Freeze All → Train Head', [0, 0, 0, 0, 0, 1]),
        ('Gradual Unfreezing', [0, 0, 0, 1, 1, 1]),
        ('Freeze Early Layers', [0, 0, 1, 1, 1, 1]),
        ('Train All Layers', [1, 1, 1, 1, 1, 1])
    ]
    
    for idx, (title, trainable) in enumerate(strategies):
        ax = axes[idx // 2, idx % 2]
        
        # Draw layers as blocks
        num_layers = len(trainable)
        for i, is_trainable in enumerate(trainable):
            x = i / num_layers
            color = 'lightgreen' if is_trainable else 'lightgray'
            edge_color = 'green' if is_trainable else 'gray'
            linewidth = 2 if is_trainable else 1
            
            rect = plt.Rectangle((x, 0.3), 0.15, 0.4, facecolor=color, 
                                edgecolor=edge_color, linewidth=linewidth, 
                                transform=ax.transAxes)
            ax.add_patch(rect)
            
            # Label
            label = 'FC' if i == num_layers - 1 else f'L{i+1}'
            ax.text(x + 0.075, 0.5, label, ha='center', va='center', 
                   fontsize=10, fontweight='bold', transform=ax.transAxes)
        
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.text(0.5, 0.15, '⬜ Заморожен   🟩 Обучается', 
               ha='center', fontsize=9, transform=ax.transAxes)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
    
    plt.tight_layout()
    return fig_to_base64(fig)

def generate_learning_rate_schedule():
    """Generate learning rate schedule for discriminative fine-tuning."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    layers = ['Conv1', 'Conv2', 'Conv3', 'Conv4', 'Conv5', 'FC1', 'FC2', 'FC3 (Head)']
    learning_rates = [1e-5, 2e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3, 5e-3]
    
    colors = plt.cm.YlOrRd(np.linspace(0.3, 0.9, len(layers)))
    bars = ax.barh(layers, learning_rates, color=colors, edgecolor='black', linewidth=1.5)
    
    # Add value labels
    for i, (bar, lr) in enumerate(zip(bars, learning_rates)):
        width = bar.get_width()
        ax.text(width * 1.1, bar.get_y() + bar.get_height()/2, 
               f'{lr:.0e}', ha='left', va='center', fontsize=10, fontweight='bold')
    
    ax.set_xlabel('Learning Rate', fontsize=12, fontweight='bold')
    ax.set_ylabel('Слой', fontsize=12, fontweight='bold')
    ax.set_title('Discriminative Learning Rates\n(меньше для ранних слоев, больше для поздних)', 
                fontsize=13, fontweight='bold')
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    return fig_to_base64(fig)

# ============================================================================
# TRANSFER LEARNING PERFORMANCE ILLUSTRATIONS
# ============================================================================

def generate_transfer_vs_scratch():
    """Generate comparison of transfer learning vs training from scratch."""
    np.random.seed(42)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    epochs = np.arange(1, 51)
    
    # Training curves
    # From scratch - slower convergence, needs more epochs
    scratch_train = 0.95 - 0.85 * np.exp(-epochs / 20) + np.random.randn(50) * 0.01
    scratch_val = 0.85 - 0.75 * np.exp(-epochs / 25) + np.random.randn(50) * 0.015
    
    # Transfer learning - faster convergence, better final accuracy
    transfer_train = 0.98 - 0.70 * np.exp(-epochs / 8) + np.random.randn(50) * 0.008
    transfer_val = 0.92 - 0.65 * np.exp(-epochs / 10) + np.random.randn(50) * 0.012
    
    axes[0].plot(epochs, scratch_train, 'b-', linewidth=2, label='Train (from scratch)')
    axes[0].plot(epochs, scratch_val, 'b--', linewidth=2, label='Val (from scratch)')
    axes[0].plot(epochs, transfer_train, 'g-', linewidth=2, label='Train (transfer)')
    axes[0].plot(epochs, transfer_val, 'g--', linewidth=2, label='Val (transfer)')
    
    axes[0].set_xlabel('Эпоха', fontsize=11)
    axes[0].set_ylabel('Точность', fontsize=11)
    axes[0].set_title('Transfer Learning vs Training from Scratch\nСкорость сходимости', 
                     fontsize=12, fontweight='bold')
    axes[0].legend(loc='lower right')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim([0.4, 1.0])
    
    # Data efficiency
    data_sizes = np.array([10, 50, 100, 500, 1000, 5000, 10000])
    
    # Accuracy as function of data size
    scratch_acc = 0.5 + 0.35 * np.log10(data_sizes) / np.log10(10000)
    transfer_acc = 0.7 + 0.25 * np.log10(data_sizes) / np.log10(10000)
    
    axes[1].plot(data_sizes, scratch_acc, 'bo-', linewidth=2, markersize=8, label='From scratch')
    axes[1].plot(data_sizes, transfer_acc, 'go-', linewidth=2, markersize=8, label='Transfer learning')
    
    # Highlight advantage with smaller datasets
    axes[1].fill_between(data_sizes[:4], scratch_acc[:4], transfer_acc[:4], 
                        alpha=0.3, color='green', label='Transfer advantage')
    
    axes[1].set_xlabel('Размер датасета (примеры)', fontsize=11)
    axes[1].set_ylabel('Точность', fontsize=11)
    axes[1].set_title('Data Efficiency\nПреимущество на малых датасетах', 
                     fontsize=12, fontweight='bold')
    axes[1].set_xscale('log')
    axes[1].legend(loc='lower right')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim([0.45, 1.0])
    
    plt.tight_layout()
    return fig_to_base64(fig)

def generate_negative_transfer():
    """Generate illustration of negative transfer phenomenon."""
    np.random.seed(42)
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Similarity score between source and target (0 = very different, 1 = very similar)
    similarity = np.linspace(0, 1, 100)
    
    # Performance curves
    baseline = np.ones_like(similarity) * 0.65  # No transfer baseline
    
    # Transfer performance: good when similar, bad when very different
    transfer_perf = 0.5 + 0.45 * similarity - 0.2 * (1 - similarity)**2
    
    ax.plot(similarity, baseline, 'k--', linewidth=2, label='Baseline (без transfer)')
    ax.plot(similarity, transfer_perf, 'b-', linewidth=3, label='С transfer learning')
    
    # Highlight regions
    # Negative transfer region
    neg_region = similarity < 0.4
    ax.fill_between(similarity[neg_region], baseline[neg_region], transfer_perf[neg_region], 
                   alpha=0.3, color='red', label='Negative Transfer')
    
    # Positive transfer region
    pos_region = similarity > 0.4
    ax.fill_between(similarity[pos_region], baseline[pos_region], transfer_perf[pos_region], 
                   alpha=0.3, color='green', label='Positive Transfer')
    
    # Add annotations
    ax.annotate('❌ Negative Transfer\nПерформанс хуже baseline', 
               xy=(0.2, 0.55), xytext=(0.15, 0.45),
               fontsize=10, ha='center',
               bbox=dict(boxstyle='round', facecolor='pink', alpha=0.7),
               arrowprops=dict(arrowstyle='->', color='red', lw=2))
    
    ax.annotate('✅ Positive Transfer\nПерформанс лучше baseline', 
               xy=(0.8, 0.88), xytext=(0.75, 0.95),
               fontsize=10, ha='center',
               bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7),
               arrowprops=dict(arrowstyle='->', color='green', lw=2))
    
    ax.set_xlabel('Схожесть source и target доменов', fontsize=12, fontweight='bold')
    ax.set_ylabel('Перформанс на target', fontsize=12, fontweight='bold')
    ax.set_title('Negative Transfer: когда transfer learning вредит\nВажность выбора похожего source домена', 
                fontsize=13, fontweight='bold')
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0.4, 1.0])
    
    plt.tight_layout()
    return fig_to_base64(fig)

# ============================================================================
# TRADABOOST ILLUSTRATION
# ============================================================================

def generate_tradaboost_process():
    """Generate TrAdaBoost weight update visualization."""
    np.random.seed(42)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    
    # Sample data
    n_source = 100
    n_target = 30
    
    # Initial state
    ax = axes[0, 0]
    X_source = np.random.randn(n_source, 2) * 1.0 + np.array([0, 0])
    X_target = np.random.randn(n_target, 2) * 1.2 + np.array([2, 1.5])
    
    weights_source_init = np.ones(n_source) / n_source
    weights_target_init = np.ones(n_target) / n_target
    
    scatter_s = ax.scatter(X_source[:, 0], X_source[:, 1], 
                          c='blue', s=weights_source_init * 5000, alpha=0.5, label='Source')
    scatter_t = ax.scatter(X_target[:, 0], X_target[:, 1], 
                          c='orange', s=weights_target_init * 5000, alpha=0.7, 
                          edgecolors='black', linewidth=1, label='Target')
    ax.set_title('Итерация 0: Начальные веса\n(все примеры равнозначны)', fontsize=11, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # After iteration 1
    ax = axes[0, 1]
    # Simulate: source points far from target get smaller weights
    distances_to_target = np.min(np.linalg.norm(X_source[:, np.newaxis, :] - X_target[np.newaxis, :, :], axis=2), axis=1)
    weights_source_1 = np.exp(-distances_to_target / 2)
    weights_source_1 = weights_source_1 / weights_source_1.sum() * 0.5
    weights_target_1 = np.ones(n_target) / n_target * 0.5
    
    scatter_s = ax.scatter(X_source[:, 0], X_source[:, 1], 
                          c='blue', s=weights_source_1 * 5000, alpha=0.5, label='Source')
    scatter_t = ax.scatter(X_target[:, 0], X_target[:, 1], 
                          c='orange', s=weights_target_1 * 5000, alpha=0.7,
                          edgecolors='black', linewidth=1, label='Target')
    ax.set_title('Итерация 1: Обновление весов\n(уменьшаем вес далеких source)', fontsize=11, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # After iteration 5
    ax = axes[1, 0]
    weights_source_5 = np.exp(-distances_to_target / 1.5)
    weights_source_5 = weights_source_5 / weights_source_5.sum() * 0.3
    weights_target_5 = np.ones(n_target) / n_target * 0.7
    
    scatter_s = ax.scatter(X_source[:, 0], X_source[:, 1], 
                          c='blue', s=weights_source_5 * 5000, alpha=0.5, label='Source')
    scatter_t = ax.scatter(X_target[:, 0], X_target[:, 1], 
                          c='orange', s=weights_target_5 * 5000, alpha=0.7,
                          edgecolors='black', linewidth=1, label='Target')
    ax.set_title('Итерация 5: Фокус на target\n(target примеры важнее)', fontsize=11, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Weight evolution plot
    ax = axes[1, 1]
    iterations = np.arange(0, 11)
    avg_source_weight = [1.0, 0.8, 0.65, 0.52, 0.42, 0.35, 0.30, 0.26, 0.23, 0.21, 0.20]
    avg_target_weight = [1.0, 1.1, 1.25, 1.42, 1.61, 1.82, 2.05, 2.30, 2.56, 2.84, 3.13]
    
    ax.plot(iterations, avg_source_weight, 'b-o', linewidth=2, markersize=6, label='Средний вес source')
    ax.plot(iterations, avg_target_weight, 'orange', linewidth=2, marker='s', markersize=6, label='Средний вес target')
    
    ax.set_xlabel('Итерация TrAdaBoost', fontsize=11)
    ax.set_ylabel('Относительный вес', fontsize=11)
    ax.set_title('Эволюция весов\n(target становится важнее)', fontsize=11, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig_to_base64(fig)

# ============================================================================
# SELF-TRAINING ILLUSTRATION
# ============================================================================

def generate_self_training():
    """Generate self-training process visualization."""
    np.random.seed(42)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    
    # Initial state: few labeled, many unlabeled
    X_source = np.random.randn(100, 2) * 1.0 + np.array([0, 0])
    y_source = (X_source[:, 0] + X_source[:, 1] > 0).astype(int)
    
    X_target_labeled = np.random.randn(20, 2) * 1.2 + np.array([2, 1.5])
    y_target_labeled = (X_target_labeled[:, 0] + X_target_labeled[:, 1] > 3.5).astype(int)
    
    X_target_unlabeled = np.random.randn(80, 2) * 1.2 + np.array([2, 1.5])
    
    # Step 1: Initial model
    ax = axes[0, 0]
    ax.scatter(X_source[y_source==0, 0], X_source[y_source==0, 1], 
              c='blue', alpha=0.3, s=30, label='Source: Класс 0')
    ax.scatter(X_source[y_source==1, 0], X_source[y_source==1, 1], 
              c='red', alpha=0.3, s=30, label='Source: Класс 1')
    ax.scatter(X_target_labeled[y_target_labeled==0, 0], X_target_labeled[y_target_labeled==0, 1],
              c='blue', s=100, edgecolors='black', linewidth=2, marker='s', label='Target labeled: 0')
    ax.scatter(X_target_labeled[y_target_labeled==1, 0], X_target_labeled[y_target_labeled==1, 1],
              c='red', s=100, edgecolors='black', linewidth=2, marker='s', label='Target labeled: 1')
    ax.scatter(X_target_unlabeled[:, 0], X_target_unlabeled[:, 1],
              c='lightgray', s=50, alpha=0.5, marker='o', label='Target unlabeled')
    ax.set_title('Шаг 1: Начальная модель\n(обучена на source + немного target)', fontsize=10, fontweight='bold')
    ax.legend(loc='upper left', fontsize=7)
    ax.grid(True, alpha=0.3)
    
    # Step 2: Predict on unlabeled
    ax = axes[0, 1]
    pseudo_labels = (X_target_unlabeled[:, 0] + X_target_unlabeled[:, 1] > 3.5).astype(int)
    confidence = np.random.uniform(0.5, 1.0, len(X_target_unlabeled))
    high_conf_mask = confidence > 0.85
    
    ax.scatter(X_source[y_source==0, 0], X_source[y_source==0, 1], 
              c='blue', alpha=0.2, s=20)
    ax.scatter(X_source[y_source==1, 0], X_source[y_source==1, 1], 
              c='red', alpha=0.2, s=20)
    ax.scatter(X_target_labeled[:, 0], X_target_labeled[:, 1],
              c=['blue' if y == 0 else 'red' for y in y_target_labeled], 
              s=100, edgecolors='black', linewidth=2, marker='s')
    
    # Show predictions with confidence
    colors = ['blue' if y == 0 else 'red' for y in pseudo_labels]
    alphas = confidence
    for i, (x, y, c, a, high_conf) in enumerate(zip(X_target_unlabeled[:, 0], X_target_unlabeled[:, 1], 
                                                     colors, alphas, high_conf_mask)):
        marker = 'D' if high_conf else 'o'
        size = 80 if high_conf else 40
        ax.scatter(x, y, c=c, s=size, alpha=a, marker=marker, edgecolors='green' if high_conf else None,
                  linewidth=2 if high_conf else 0)
    
    ax.set_title('Шаг 2: Предсказание на unlabeled\n(◆ = высокая уверенность)', fontsize=10, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Step 3: Add confident predictions
    ax = axes[1, 0]
    X_new_labeled = X_target_unlabeled[high_conf_mask]
    y_new_labeled = pseudo_labels[high_conf_mask]
    X_still_unlabeled = X_target_unlabeled[~high_conf_mask]
    
    ax.scatter(X_source[y_source==0, 0], X_source[y_source==0, 1], 
              c='blue', alpha=0.2, s=20)
    ax.scatter(X_source[y_source==1, 0], X_source[y_source==1, 1], 
              c='red', alpha=0.2, s=20)
    
    # Original labeled
    ax.scatter(X_target_labeled[:, 0], X_target_labeled[:, 1],
              c=['blue' if y == 0 else 'red' for y in y_target_labeled], 
              s=100, edgecolors='black', linewidth=2, marker='s', label='Исходные метки')
    
    # Newly added pseudo-labels
    ax.scatter(X_new_labeled[y_new_labeled==0, 0], X_new_labeled[y_new_labeled==0, 1],
              c='blue', s=80, marker='D', edgecolors='green', linewidth=2, label='Псевдометки: 0')
    ax.scatter(X_new_labeled[y_new_labeled==1, 0], X_new_labeled[y_new_labeled==1, 1],
              c='red', s=80, marker='D', edgecolors='green', linewidth=2, label='Псевдометки: 1')
    
    # Still unlabeled
    ax.scatter(X_still_unlabeled[:, 0], X_still_unlabeled[:, 1],
              c='lightgray', s=40, alpha=0.5, label='Остальные unlabeled')
    
    ax.set_title(f'Шаг 3: Добавление псевдометок\n(+{high_conf_mask.sum()} примеров)', fontsize=10, fontweight='bold')
    ax.legend(loc='upper left', fontsize=7)
    ax.grid(True, alpha=0.3)
    
    # Step 4: Progress plot
    ax = axes[1, 1]
    iterations = np.arange(0, 11)
    labeled_count = [20, 35, 48, 59, 68, 75, 81, 86, 90, 93, 95]
    accuracy = [0.70, 0.73, 0.76, 0.79, 0.82, 0.84, 0.86, 0.87, 0.88, 0.89, 0.89]
    
    ax1 = ax
    ax2 = ax1.twinx()
    
    line1 = ax1.plot(iterations, labeled_count, 'g-o', linewidth=2, markersize=6, label='Размеченных примеров')
    line2 = ax2.plot(iterations, accuracy, 'b-s', linewidth=2, markersize=6, label='Точность')
    
    ax1.set_xlabel('Итерация self-training', fontsize=11)
    ax1.set_ylabel('Количество размеченных', fontsize=11, color='g')
    ax2.set_ylabel('Точность', fontsize=11, color='b')
    ax1.tick_params(axis='y', labelcolor='g')
    ax2.tick_params(axis='y', labelcolor='b')
    
    # Combine legends
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='center right')
    
    ax.set_title('Прогресс self-training\n(больше данных → лучше качество)', fontsize=10, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig_to_base64(fig)

# ============================================================================
# MAIN GENERATION FUNCTION
# ============================================================================

def generate_all_illustrations():
    """Generate all transfer learning illustrations and return as dictionary."""
    print("Generating transfer learning illustrations...")
    
    illustrations = {}
    
    print("  - Transfer Learning Concepts...")
    illustrations['tl_concept'] = generate_transfer_learning_concept()
    illustrations['tl_types'] = generate_transfer_learning_types()
    illustrations['domain_shift'] = generate_domain_shift()
    
    print("  - Domain Adaptation...")
    illustrations['da_methods'] = generate_domain_adaptation_methods()
    illustrations['mmd'] = generate_mmd_illustration()
    
    print("  - CNN Transfer Learning...")
    illustrations['cnn_architecture'] = generate_cnn_transfer_architecture()
    illustrations['finetuning_strategies'] = generate_finetuning_strategies()
    illustrations['learning_rate_schedule'] = generate_learning_rate_schedule()
    
    print("  - Performance Comparisons...")
    illustrations['transfer_vs_scratch'] = generate_transfer_vs_scratch()
    illustrations['negative_transfer'] = generate_negative_transfer()
    
    print("  - Algorithms...")
    illustrations['tradaboost'] = generate_tradaboost_process()
    illustrations['self_training'] = generate_self_training()
    
    print("✓ All illustrations generated successfully!")
    return illustrations

if __name__ == '__main__':
    # Test generation
    illustrations = generate_all_illustrations()
    print(f"\nGenerated {len(illustrations)} illustrations")
    for key in illustrations.keys():
        print(f"  - {key}")
