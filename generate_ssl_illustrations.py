#!/usr/bin/env python3
"""
Generate matplotlib illustrations for Self-Supervised and Semi-Supervised Learning cheatsheets.
This script creates high-quality visualizations and encodes them as base64 for inline HTML embedding.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification, make_blobs, make_moons
from sklearn.preprocessing import StandardScaler
from sklearn.semi_supervised import LabelPropagation, SelfTrainingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
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
# SELF-SUPERVISED LEARNING ILLUSTRATIONS
# ============================================================================

def generate_rotation_prediction():
    """Generate rotation prediction visualization."""
    np.random.seed(42)
    
    fig, axes = plt.subplots(1, 4, figsize=(12, 3))
    
    # Create a simple image pattern
    x = np.linspace(-1, 1, 100)
    y = np.linspace(-1, 1, 100)
    X, Y = np.meshgrid(x, y)
    Z = np.sin(3*X) * np.cos(3*Y) + X*Y
    
    rotations = [0, 90, 180, 270]
    for i, (ax, rot) in enumerate(zip(axes, rotations)):
        if rot == 0:
            img = Z
        elif rot == 90:
            img = np.rot90(Z, k=1)
        elif rot == 180:
            img = np.rot90(Z, k=2)
        else:  # 270
            img = np.rot90(Z, k=3)
        
        ax.imshow(img, cmap='viridis', interpolation='bilinear')
        ax.set_title(f'{rot}°', fontweight='bold')
        ax.axis('off')
    
    fig.suptitle('Rotation Prediction: Pretext Task', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    return fig_to_base64(fig)

def generate_jigsaw_puzzles():
    """Generate jigsaw puzzle visualization."""
    np.random.seed(42)
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    # Create a pattern
    x = np.linspace(0, 1, 90)
    y = np.linspace(0, 1, 90)
    X, Y = np.meshgrid(x, y)
    Z = np.sin(5*np.pi*X) * np.cos(5*np.pi*Y) + 0.5*X + 0.5*Y
    
    # Original image divided into patches
    ax = axes[0]
    ax.imshow(Z, cmap='plasma')
    # Draw grid
    for i in range(1, 3):
        ax.axhline(y=30*i, color='white', linewidth=2)
        ax.axvline(x=30*i, color='white', linewidth=2)
    ax.set_title('Original (3x3 grid)', fontweight='bold')
    ax.axis('off')
    
    # Shuffled patches
    ax = axes[1]
    patches = []
    for i in range(3):
        for j in range(3):
            patch = Z[i*30:(i+1)*30, j*30:(j+1)*30]
            patches.append((patch, i*3+j))
    
    # Shuffle
    np.random.shuffle(patches)
    shuffled = np.zeros_like(Z)
    for idx, (patch, orig_idx) in enumerate(patches):
        i = idx // 3
        j = idx % 3
        shuffled[i*30:(i+1)*30, j*30:(j+1)*30] = patch
    
    ax.imshow(shuffled, cmap='plasma')
    for i in range(1, 3):
        ax.axhline(y=30*i, color='white', linewidth=2)
        ax.axvline(x=30*i, color='white', linewidth=2)
    ax.set_title('Shuffled (restore order)', fontweight='bold')
    ax.axis('off')
    
    fig.suptitle('Jigsaw Puzzles: Pretext Task', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    return fig_to_base64(fig)

def generate_colorization():
    """Generate colorization visualization."""
    np.random.seed(42)
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    # Create a colored pattern
    x = np.linspace(0, 2*np.pi, 100)
    y = np.linspace(0, 2*np.pi, 100)
    X, Y = np.meshgrid(x, y)
    
    R = np.sin(X) * 0.5 + 0.5
    G = np.cos(Y) * 0.5 + 0.5
    B = np.sin(X+Y) * 0.5 + 0.5
    
    colored = np.stack([R, G, B], axis=-1)
    grayscale = np.mean(colored, axis=-1)
    
    axes[0].imshow(grayscale, cmap='gray')
    axes[0].set_title('Input: Grayscale', fontweight='bold')
    axes[0].axis('off')
    
    axes[1].imshow(colored)
    axes[1].set_title('Predict: Color', fontweight='bold')
    axes[1].axis('off')
    
    fig.suptitle('Colorization: Pretext Task', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    return fig_to_base64(fig)

def generate_mae_masking():
    """Generate Masked Autoencoder visualization."""
    np.random.seed(42)
    
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    # Create image-like pattern
    x = np.linspace(0, 1, 112)
    y = np.linspace(0, 1, 112)
    X, Y = np.meshgrid(x, y)
    Z = np.sin(10*X) * np.cos(10*Y) + 0.3*X*Y
    
    # Original
    axes[0].imshow(Z, cmap='viridis')
    axes[0].set_title('Original Image', fontweight='bold')
    axes[0].axis('off')
    
    # Masked (75% masking)
    masked = Z.copy()
    mask = np.random.rand(112, 112) < 0.75
    masked[mask] = np.nan
    
    axes[1].imshow(masked, cmap='viridis')
    axes[1].set_title('Masked (75%)', fontweight='bold')
    axes[1].axis('off')
    
    # Reconstructed (with slight noise)
    reconstructed = Z + np.random.randn(112, 112) * 0.1
    axes[2].imshow(reconstructed, cmap='viridis')
    axes[2].set_title('Reconstructed', fontweight='bold')
    axes[2].axis('off')
    
    fig.suptitle('Masked Autoencoder (MAE)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    return fig_to_base64(fig)

def generate_ssl_comparison():
    """Generate comparison of self-supervised methods."""
    methods = ['Rotation', 'Jigsaw', 'Colorization', 'MAE', 'Contrastive']
    accuracy = [72, 69, 65, 84, 85]  # Example accuracies on downstream task
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = sns.color_palette("husl", len(methods))
    bars = ax.barh(methods, accuracy, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    
    # Add values on bars
    for i, (bar, acc) in enumerate(zip(bars, accuracy)):
        ax.text(acc + 1, i, f'{acc}%', va='center', fontweight='bold')
    
    ax.set_xlabel('ImageNet Top-1 Accuracy (%)', fontweight='bold')
    ax.set_title('Self-Supervised Methods: Downstream Task Performance', fontsize=13, fontweight='bold')
    ax.set_xlim(0, 100)
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    return fig_to_base64(fig)

# ============================================================================
# SEMI-SUPERVISED LEARNING ILLUSTRATIONS
# ============================================================================

def generate_self_training_process():
    """Generate self-training process visualization."""
    np.random.seed(42)
    
    # Generate data
    X, y = make_classification(n_samples=200, n_features=2, n_redundant=0,
                               n_informative=2, n_clusters_per_class=1, random_state=42)
    
    # Simulate labeled/unlabeled split
    n_labeled = 40
    labeled_idx = np.random.choice(len(X), n_labeled, replace=False)
    unlabeled_idx = np.array([i for i in range(len(X)) if i not in labeled_idx])
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Step 1: Initial labeled data
    ax = axes[0]
    ax.scatter(X[labeled_idx, 0], X[labeled_idx, 1], 
              c=y[labeled_idx], cmap='coolwarm', s=100, alpha=0.8,
              edgecolors='black', linewidth=1.5, label='Labeled')
    ax.scatter(X[unlabeled_idx, 0], X[unlabeled_idx, 1], 
              c='gray', s=50, alpha=0.3, label='Unlabeled')
    ax.set_title('Step 1: Initial Labeled Data', fontweight='bold')
    ax.legend()
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    
    # Step 2: Pseudo-labels added
    ax = axes[1]
    # Simulate high-confidence predictions
    confident_idx = unlabeled_idx[:30]
    remaining_unlabeled = unlabeled_idx[30:]
    
    ax.scatter(X[labeled_idx, 0], X[labeled_idx, 1], 
              c=y[labeled_idx], cmap='coolwarm', s=100, alpha=0.8,
              edgecolors='black', linewidth=1.5, label='Labeled')
    ax.scatter(X[confident_idx, 0], X[confident_idx, 1], 
              c=y[confident_idx], cmap='coolwarm', s=80, alpha=0.5,
              edgecolors='green', linewidth=2, marker='s', label='Pseudo-labeled')
    ax.scatter(X[remaining_unlabeled, 0], X[remaining_unlabeled, 1], 
              c='gray', s=50, alpha=0.3, label='Unlabeled')
    ax.set_title('Step 2: Add Pseudo-labels', fontweight='bold')
    ax.legend()
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    
    # Step 3: Final iteration
    ax = axes[2]
    ax.scatter(X[labeled_idx, 0], X[labeled_idx, 1], 
              c=y[labeled_idx], cmap='coolwarm', s=100, alpha=0.8,
              edgecolors='black', linewidth=1.5, label='Original Labeled')
    ax.scatter(X[unlabeled_idx, 0], X[unlabeled_idx, 1], 
              c=y[unlabeled_idx], cmap='coolwarm', s=80, alpha=0.6,
              edgecolors='green', linewidth=1.5, marker='s', label='All Pseudo-labeled')
    ax.set_title('Step 3: Converged', fontweight='bold')
    ax.legend()
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    
    fig.suptitle('Self-Training Process', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    return fig_to_base64(fig)

def generate_label_propagation():
    """Generate label propagation visualization."""
    np.random.seed(42)
    
    # Create data with clear structure
    X, y = make_moons(n_samples=200, noise=0.1, random_state=42)
    
    # Label only a few points
    n_labeled = 10
    labeled_indices = [10, 50, 70, 120, 140, 160, 20, 90, 110, 180]
    
    y_train = np.full(len(y), -1)
    y_train[labeled_indices] = y[labeled_indices]
    
    # Apply Label Propagation
    lp = LabelPropagation(kernel='knn', n_neighbors=7)
    lp.fit(X, y_train)
    y_pred = lp.transduction_
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Before propagation
    ax = axes[0]
    unlabeled = y_train == -1
    ax.scatter(X[unlabeled, 0], X[unlabeled, 1], c='gray', s=50, alpha=0.3, label='Unlabeled')
    ax.scatter(X[labeled_indices, 0], X[labeled_indices, 1], 
              c=y[labeled_indices], cmap='coolwarm', s=200, alpha=0.9,
              edgecolors='black', linewidth=2, marker='*', label='Labeled')
    ax.set_title('Before: Few Labeled Points', fontweight='bold')
    ax.legend()
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    
    # After propagation
    ax = axes[1]
    scatter = ax.scatter(X[:, 0], X[:, 1], c=y_pred, cmap='coolwarm', 
                        s=50, alpha=0.6, edgecolors='white', linewidth=0.5)
    ax.scatter(X[labeled_indices, 0], X[labeled_indices, 1], 
              c=y[labeled_indices], cmap='coolwarm', s=200, alpha=1.0,
              edgecolors='black', linewidth=2, marker='*', label='Original Labels')
    ax.set_title('After: Labels Propagated', fontweight='bold')
    ax.legend()
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    plt.colorbar(scatter, ax=ax, label='Class')
    
    fig.suptitle('Label Propagation', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    return fig_to_base64(fig)

def generate_ssl_performance():
    """Generate semi-supervised learning performance comparison."""
    np.random.seed(42)
    
    labeled_percentages = [5, 10, 20, 50, 100]
    supervised_accuracy = [55, 65, 75, 85, 90]
    semi_supervised_accuracy = [68, 76, 82, 88, 91]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(labeled_percentages, supervised_accuracy, 'o-', 
            linewidth=2.5, markersize=10, label='Supervised Only', color='#2ecc71')
    ax.plot(labeled_percentages, semi_supervised_accuracy, 's-', 
            linewidth=2.5, markersize=10, label='Semi-Supervised', color='#3498db')
    
    # Highlight the gap
    for x, y1, y2 in zip(labeled_percentages, supervised_accuracy, semi_supervised_accuracy):
        ax.vlines(x, y1, y2, colors='red', linestyles='dashed', alpha=0.5, linewidth=1.5)
    
    ax.set_xlabel('Labeled Data (%)', fontweight='bold')
    ax.set_ylabel('Accuracy (%)', fontweight='bold')
    ax.set_title('Semi-Supervised vs Supervised Learning', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    ax.set_ylim(50, 95)
    
    plt.tight_layout()
    return fig_to_base64(fig)

# ============================================================================
# CONTRASTIVE LEARNING ILLUSTRATIONS
# ============================================================================

def generate_contrastive_pairs():
    """Generate visualization of positive and negative pairs."""
    np.random.seed(42)
    
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    
    # Create base patterns
    x = np.linspace(0, 1, 50)
    y = np.linspace(0, 1, 50)
    X, Y = np.meshgrid(x, y)
    
    # Original image
    Z_original = np.sin(8*np.pi*X) * np.cos(8*np.pi*Y)
    
    # Positive pairs (augmentations)
    axes[0, 0].imshow(Z_original, cmap='viridis')
    axes[0, 0].set_title('Original', fontweight='bold', color='green')
    axes[0, 0].axis('off')
    
    # Aug 1: crop and color
    Z_aug1 = Z_original[5:45, 5:45] * 1.2
    axes[0, 1].imshow(Z_aug1, cmap='plasma')
    axes[0, 1].set_title('Aug 1 (Crop+Color)', fontweight='bold', color='green')
    axes[0, 1].axis('off')
    
    # Aug 2: flip and blur
    Z_aug2 = np.flip(Z_original, axis=1)
    from scipy.ndimage import gaussian_filter
    Z_aug2 = gaussian_filter(Z_aug2, sigma=1)
    axes[0, 2].imshow(Z_aug2, cmap='viridis')
    axes[0, 2].set_title('Aug 2 (Flip+Blur)', fontweight='bold', color='green')
    axes[0, 2].axis('off')
    
    # Add text
    axes[0, 1].text(0.5, -0.15, '✓ Positive Pair', transform=axes[0, 1].transAxes,
                   ha='center', fontsize=12, fontweight='bold', color='green')
    
    # Negative examples (different images)
    Z_neg1 = np.cos(10*np.pi*X) * np.sin(5*np.pi*Y)
    Z_neg2 = X * Y * 5
    Z_neg3 = np.sin(3*np.pi*(X+Y)) * np.cos(3*np.pi*(X-Y))
    
    axes[1, 0].imshow(Z_neg1, cmap='magma')
    axes[1, 0].set_title('Different Image 1', fontweight='bold', color='red')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(Z_neg2, cmap='cividis')
    axes[1, 1].set_title('Different Image 2', fontweight='bold', color='red')
    axes[1, 1].axis('off')
    
    axes[1, 2].imshow(Z_neg3, cmap='twilight')
    axes[1, 2].set_title('Different Image 3', fontweight='bold', color='red')
    axes[1, 2].axis('off')
    
    axes[1, 1].text(0.5, -0.15, '✗ Negative Pairs', transform=axes[1, 1].transAxes,
                   ha='center', fontsize=12, fontweight='bold', color='red')
    
    fig.suptitle('Contrastive Learning: Positive vs Negative Pairs', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    return fig_to_base64(fig)

def generate_embedding_space():
    """Generate visualization of learned embedding space."""
    np.random.seed(42)
    
    # Simulate embeddings for 5 classes
    n_samples_per_class = 50
    n_classes = 5
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Before contrastive learning (random embeddings)
    ax = axes[0]
    embeddings_before = []
    labels = []
    for i in range(n_classes):
        emb = np.random.randn(n_samples_per_class, 2) * 2 + np.random.randn(2) * 3
        embeddings_before.append(emb)
        labels.extend([i] * n_samples_per_class)
    embeddings_before = np.vstack(embeddings_before)
    labels = np.array(labels)
    
    scatter = ax.scatter(embeddings_before[:, 0], embeddings_before[:, 1],
                        c=labels, cmap='tab10', s=50, alpha=0.6, edgecolors='white', linewidth=0.5)
    ax.set_title('Before: Random Embeddings', fontweight='bold', fontsize=12)
    ax.set_xlabel('Dimension 1')
    ax.set_ylabel('Dimension 2')
    ax.grid(alpha=0.3)
    
    # After contrastive learning (clustered embeddings)
    ax = axes[1]
    embeddings_after = []
    centers = np.array([[0, 0], [3, 3], [-3, 3], [3, -3], [-3, -3]])
    for i, center in enumerate(centers):
        emb = np.random.randn(n_samples_per_class, 2) * 0.4 + center
        embeddings_after.append(emb)
    embeddings_after = np.vstack(embeddings_after)
    
    scatter = ax.scatter(embeddings_after[:, 0], embeddings_after[:, 1],
                        c=labels, cmap='tab10', s=50, alpha=0.7, edgecolors='black', linewidth=0.5)
    # Draw circles around clusters
    for center in centers:
        circle = plt.Circle(center, 1.2, color='gray', fill=False, linestyle='--', linewidth=2, alpha=0.5)
        ax.add_patch(circle)
    
    ax.set_title('After: Clustered Embeddings', fontweight='bold', fontsize=12)
    ax.set_xlabel('Dimension 1')
    ax.set_ylabel('Dimension 2')
    ax.grid(alpha=0.3)
    
    fig.suptitle('Effect of Contrastive Learning on Embedding Space', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    return fig_to_base64(fig)

def generate_ntxent_loss():
    """Generate NT-Xent loss temperature effect visualization."""
    np.random.seed(42)
    
    # Simulate similarity matrix
    batch_size = 8
    similarities = np.random.randn(batch_size, batch_size) * 0.5
    # Make diagonal more positive (self-similarity)
    np.fill_diagonal(similarities, 1.0)
    # Make one positive pair
    similarities[0, 4] = 0.8
    similarities[4, 0] = 0.8
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    temperatures = [0.1, 0.5, 1.0]
    
    for ax, temp in zip(axes, temperatures):
        # Apply temperature scaling
        scaled = similarities / temp
        
        im = ax.imshow(scaled, cmap='RdYlGn', vmin=-2, vmax=10)
        ax.set_title(f'Temperature τ={temp}', fontweight='bold')
        ax.set_xlabel('Sample Index')
        ax.set_ylabel('Sample Index')
        
        # Annotate
        for i in range(batch_size):
            for j in range(batch_size):
                text = ax.text(j, i, f'{scaled[i, j]:.1f}',
                             ha="center", va="center", color="black", fontsize=8)
        
        # Highlight positive pair
        from matplotlib.patches import Rectangle
        rect1 = Rectangle((3.5, -0.5), 1, 1, linewidth=3, edgecolor='blue', facecolor='none')
        rect2 = Rectangle((-0.5, 3.5), 1, 1, linewidth=3, edgecolor='blue', facecolor='none')
        ax.add_patch(rect1)
        ax.add_patch(rect2)
    
    fig.colorbar(im, ax=axes, label='Similarity / Temperature', fraction=0.046, pad=0.04)
    fig.suptitle('NT-Xent Loss: Temperature Effect', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    return fig_to_base64(fig)

def generate_simclr_architecture():
    """Generate SimCLR architecture diagram."""
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Draw architecture flow
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
    
    # Input image
    img_box = FancyBboxPatch((0.5, 7), 1.5, 2, boxstyle="round,pad=0.1", 
                             edgecolor='black', facecolor='lightblue', linewidth=2)
    ax.add_patch(img_box)
    ax.text(1.25, 8, 'Input\nImage', ha='center', va='center', fontweight='bold', fontsize=10)
    
    # Augmentation 1
    aug1_box = FancyBboxPatch((2.5, 8), 1.5, 1, boxstyle="round,pad=0.1",
                             edgecolor='green', facecolor='lightgreen', linewidth=2)
    ax.add_patch(aug1_box)
    ax.text(3.25, 8.5, 'Aug 1', ha='center', va='center', fontweight='bold', fontsize=9)
    
    # Augmentation 2
    aug2_box = FancyBboxPatch((2.5, 7), 1.5, 1, boxstyle="round,pad=0.1",
                             edgecolor='green', facecolor='lightgreen', linewidth=2)
    ax.add_patch(aug2_box)
    ax.text(3.25, 7.5, 'Aug 2', ha='center', va='center', fontweight='bold', fontsize=9)
    
    # Encoder 1
    enc1_box = FancyBboxPatch((4.5, 8), 1.5, 1, boxstyle="round,pad=0.1",
                             edgecolor='blue', facecolor='lightcyan', linewidth=2)
    ax.add_patch(enc1_box)
    ax.text(5.25, 8.5, 'Encoder\nf(·)', ha='center', va='center', fontweight='bold', fontsize=9)
    
    # Encoder 2
    enc2_box = FancyBboxPatch((4.5, 7), 1.5, 1, boxstyle="round,pad=0.1",
                             edgecolor='blue', facecolor='lightcyan', linewidth=2)
    ax.add_patch(enc2_box)
    ax.text(5.25, 7.5, 'Encoder\nf(·)', ha='center', va='center', fontweight='bold', fontsize=9)
    
    # Projection 1
    proj1_box = FancyBboxPatch((6.5, 8), 1.5, 1, boxstyle="round,pad=0.1",
                              edgecolor='purple', facecolor='plum', linewidth=2)
    ax.add_patch(proj1_box)
    ax.text(7.25, 8.5, 'Projection\ng(·)', ha='center', va='center', fontweight='bold', fontsize=9)
    
    # Projection 2
    proj2_box = FancyBboxPatch((6.5, 7), 1.5, 1, boxstyle="round,pad=0.1",
                              edgecolor='purple', facecolor='plum', linewidth=2)
    ax.add_patch(proj2_box)
    ax.text(7.25, 7.5, 'Projection\ng(·)', ha='center', va='center', fontweight='bold', fontsize=9)
    
    # Loss
    loss_box = FancyBboxPatch((8.5, 7.5), 1.2, 1, boxstyle="round,pad=0.1",
                             edgecolor='red', facecolor='lightcoral', linewidth=2)
    ax.add_patch(loss_box)
    ax.text(9.1, 8, 'NT-Xent\nLoss', ha='center', va='center', fontweight='bold', fontsize=10)
    
    # Arrows
    arrows = [
        ((2, 8), (2.5, 8.5)),
        ((2, 8), (2.5, 7.5)),
        ((4, 8.5), (4.5, 8.5)),
        ((4, 7.5), (4.5, 7.5)),
        ((6, 8.5), (6.5, 8.5)),
        ((6, 7.5), (6.5, 7.5)),
        ((8, 8.5), (8.5, 8.2)),
        ((8, 7.5), (8.5, 7.8)),
    ]
    
    for start, end in arrows:
        arrow = FancyArrowPatch(start, end, arrowstyle='->', 
                               mutation_scale=20, linewidth=2, color='black')
        ax.add_patch(arrow)
    
    # Add labels
    ax.text(5, 9.5, 'SimCLR Architecture', ha='center', fontsize=14, fontweight='bold')
    ax.text(5, 6.5, 'Same weights (shared encoder)', ha='center', fontsize=10, 
           style='italic', color='gray')
    
    # Add annotations
    ax.text(3.25, 9.2, 'Random crop, color jitter,\nflip, blur', ha='center', 
           fontsize=8, style='italic', color='green')
    ax.text(7.25, 9.2, 'MLP: 2048→128', ha='center', 
           fontsize=8, style='italic', color='purple')
    
    plt.tight_layout()
    return fig_to_base64(fig)

def generate_contrastive_methods_comparison():
    """Generate comparison of contrastive learning methods."""
    methods = ['SimCLR', 'MoCo v2', 'BYOL', 'SwAV', 'Barlow Twins']
    accuracy = [76.5, 71.1, 74.3, 75.3, 73.2]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6']
    bars = ax.bar(methods, accuracy, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    
    # Add values on bars
    for bar, acc in zip(bars, accuracy):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{acc}%', ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    ax.set_ylabel('ImageNet Top-1 Accuracy (%)', fontweight='bold', fontsize=11)
    ax.set_title('Contrastive Learning Methods Comparison', fontsize=13, fontweight='bold')
    ax.set_ylim(0, 85)
    ax.grid(axis='y', alpha=0.3)
    ax.set_axisbelow(True)
    
    plt.xticks(rotation=15, ha='right')
    plt.tight_layout()
    return fig_to_base64(fig)

# ============================================================================
# MAIN GENERATION FUNCTION
# ============================================================================

def generate_all_illustrations():
    """Generate all illustrations and return as dictionary."""
    print("Generating Self-Supervised Learning illustrations...")
    illustrations = {}
    
    # Self-Supervised Learning
    print("  - Rotation prediction...")
    illustrations['rotation_prediction'] = generate_rotation_prediction()
    
    print("  - Jigsaw puzzles...")
    illustrations['jigsaw_puzzles'] = generate_jigsaw_puzzles()
    
    print("  - Colorization...")
    illustrations['colorization'] = generate_colorization()
    
    print("  - MAE masking...")
    illustrations['mae_masking'] = generate_mae_masking()
    
    print("  - SSL methods comparison...")
    illustrations['ssl_comparison'] = generate_ssl_comparison()
    
    # Semi-Supervised Learning
    print("\nGenerating Semi-Supervised Learning illustrations...")
    print("  - Self-training process...")
    illustrations['self_training'] = generate_self_training_process()
    
    print("  - Label propagation...")
    illustrations['label_propagation'] = generate_label_propagation()
    
    print("  - SSL performance comparison...")
    illustrations['ssl_performance'] = generate_ssl_performance()
    
    # Contrastive Learning
    print("\nGenerating Contrastive Learning illustrations...")
    print("  - Contrastive pairs...")
    illustrations['contrastive_pairs'] = generate_contrastive_pairs()
    
    print("  - Embedding space...")
    illustrations['embedding_space'] = generate_embedding_space()
    
    print("  - NT-Xent loss...")
    illustrations['ntxent_loss'] = generate_ntxent_loss()
    
    print("  - SimCLR architecture...")
    illustrations['simclr_architecture'] = generate_simclr_architecture()
    
    print("  - Contrastive methods comparison...")
    illustrations['contrastive_comparison'] = generate_contrastive_methods_comparison()
    
    print("\n✓ All illustrations generated successfully!")
    return illustrations

if __name__ == '__main__':
    illustrations = generate_all_illustrations()
    print(f"\nGenerated {len(illustrations)} illustrations")
    for key in illustrations.keys():
        print(f"  - {key}")
