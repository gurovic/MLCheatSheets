#!/usr/bin/env python3
"""
Generate matplotlib illustrations for dimensionality reduction cheatsheets.
This script creates high-quality visualizations and encodes them as base64 for inline HTML embedding.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.manifold import TSNE, Isomap, LocallyLinearEmbedding, MDS
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_blobs, make_classification, make_swiss_roll, load_digits
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
# PCA ILLUSTRATIONS
# ============================================================================

def generate_pca_scree_plot():
    """Generate scree plot showing explained variance."""
    np.random.seed(42)
    X, _ = make_classification(n_samples=300, n_features=10, n_informative=8, 
                                n_redundant=2, random_state=42)
    X_scaled = StandardScaler().fit_transform(X)
    
    pca = PCA()
    pca.fit(X_scaled)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Individual variance
    ax1.bar(range(1, len(pca.explained_variance_ratio_) + 1),
            pca.explained_variance_ratio_, alpha=0.7, color='steelblue')
    ax1.set_xlabel('Номер компоненты')
    ax1.set_ylabel('Объясненная дисперсия')
    ax1.set_title('Scree Plot: Дисперсия по компонентам')
    ax1.grid(True, alpha=0.3)
    
    # Cumulative variance
    cumsum = np.cumsum(pca.explained_variance_ratio_)
    ax2.plot(range(1, len(cumsum) + 1), cumsum, 'bo-', linewidth=2, markersize=6)
    ax2.axhline(y=0.95, color='r', linestyle='--', linewidth=2, label='95% дисперсии')
    ax2.axhline(y=0.90, color='orange', linestyle='--', linewidth=1.5, alpha=0.7, label='90% дисперсии')
    ax2.set_xlabel('Число компонент')
    ax2.set_ylabel('Накопленная дисперсия')
    ax2.set_title('Накопленная объясненная дисперсия')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1.05])
    
    plt.tight_layout()
    return fig_to_base64(fig)

def generate_pca_projection_2d():
    """Generate 2D PCA projection visualization."""
    np.random.seed(42)
    X, y = make_classification(n_samples=300, n_features=10, n_informative=3,
                                n_redundant=2, n_classes=3, n_clusters_per_class=1,
                                random_state=42)
    X_scaled = StandardScaler().fit_transform(X)
    
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    fig, ax = plt.subplots(figsize=(10, 7))
    
    scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis',
                        alpha=0.6, s=50, edgecolors='w', linewidth=0.5)
    
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} дисперсии)')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} дисперсии)')
    ax.set_title('PCA: Проекция в 2D пространство')
    ax.grid(True, alpha=0.3)
    
    # Add origin lines
    ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)
    ax.axvline(x=0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)
    
    plt.colorbar(scatter, ax=ax, label='Класс')
    
    return fig_to_base64(fig)

def generate_pca_3d_projection():
    """Generate 3D PCA projection visualization."""
    np.random.seed(42)
    X, y = make_classification(n_samples=300, n_features=10, n_informative=3,
                                n_redundant=2, n_classes=3, n_clusters_per_class=1,
                                random_state=42)
    X_scaled = StandardScaler().fit_transform(X)
    
    pca = PCA(n_components=3)
    X_pca = pca.fit_transform(X_scaled)
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2],
                        c=y, cmap='viridis', alpha=0.6, s=50,
                        edgecolors='w', linewidth=0.5)
    
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
    ax.set_zlabel(f'PC3 ({pca.explained_variance_ratio_[2]:.1%})')
    ax.set_title('PCA: Проекция в 3D пространство')
    
    plt.colorbar(scatter, ax=ax, label='Класс', pad=0.1)
    
    return fig_to_base64(fig)

def generate_pca_components_visualization():
    """Generate visualization of principal component directions."""
    np.random.seed(42)
    # Generate 2D data for easier visualization
    mean = [0, 0]
    cov = [[3, 2], [2, 2]]
    X = np.random.multivariate_normal(mean, cov, 300)
    X_scaled = StandardScaler().fit_transform(X)
    
    pca = PCA(n_components=2)
    pca.fit(X_scaled)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot data points
    ax.scatter(X_scaled[:, 0], X_scaled[:, 1], alpha=0.5, s=30, label='Данные')
    
    # Plot principal components as arrows
    mean = X_scaled.mean(axis=0)
    for i, (comp, var) in enumerate(zip(pca.components_, pca.explained_variance_)):
        ax.arrow(mean[0], mean[1], comp[0] * 3 * np.sqrt(var), comp[1] * 3 * np.sqrt(var),
                head_width=0.2, head_length=0.2, fc=f'C{i+1}', ec=f'C{i+1}',
                linewidth=3, label=f'PC{i+1} ({pca.explained_variance_ratio_[i]:.1%})')
    
    ax.set_xlabel('Признак 1 (масштабированный)')
    ax.set_ylabel('Признак 2 (масштабированный)')
    ax.set_title('Главные компоненты PCA\n(направления максимальной дисперсии)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    
    return fig_to_base64(fig)

# ============================================================================
# t-SNE ILLUSTRATIONS
# ============================================================================

def generate_tsne_perplexity_comparison():
    """Generate t-SNE visualization with different perplexity values."""
    np.random.seed(42)
    digits = load_digits()
    X = digits.data[:500]
    y = digits.target[:500]
    X_scaled = StandardScaler().fit_transform(X)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    perplexities = [5, 30, 50, 100]
    
    for idx, perplexity in enumerate(perplexities):
        ax = axes[idx // 2, idx % 2]
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
        X_tsne = tsne.fit_transform(X_scaled)
        
        scatter = ax.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='tab10',
                           alpha=0.6, s=30, edgecolors='w', linewidth=0.3)
        ax.set_title(f'Perplexity = {perplexity}')
        ax.set_xlabel('t-SNE 1')
        ax.set_ylabel('t-SNE 2')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig_to_base64(fig)

def generate_tsne_visualization():
    """Generate t-SNE 2D visualization."""
    np.random.seed(42)
    digits = load_digits()
    X = digits.data[:600]
    y = digits.target[:600]
    X_scaled = StandardScaler().fit_transform(X)
    
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    X_tsne = tsne.fit_transform(X_scaled)
    
    fig, ax = plt.subplots(figsize=(12, 9))
    
    scatter = ax.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='tab10',
                        alpha=0.7, s=50, edgecolors='w', linewidth=0.5)
    
    # Add labels for each class
    for i in range(10):
        mask = y == i
        center = X_tsne[mask].mean(axis=0)
        ax.text(center[0], center[1], str(i), fontsize=20, fontweight='bold',
               ha='center', va='center', color='white',
               bbox=dict(boxstyle='circle,pad=0.3', facecolor='black', alpha=0.7))
    
    ax.set_xlabel('t-SNE компонента 1')
    ax.set_ylabel('t-SNE компонента 2')
    ax.set_title('t-SNE: Визуализация цифр MNIST')
    ax.grid(True, alpha=0.3)
    
    cbar = plt.colorbar(scatter, ax=ax, label='Цифра')
    cbar.set_ticks(range(10))
    
    return fig_to_base64(fig)

def generate_tsne_vs_pca():
    """Generate comparison of t-SNE vs PCA."""
    np.random.seed(42)
    digits = load_digits()
    X = digits.data[:500]
    y = digits.target[:500]
    X_scaled = StandardScaler().fit_transform(X)
    
    # PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    # t-SNE
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    X_tsne = tsne.fit_transform(X_scaled)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # PCA plot
    scatter1 = ax1.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='tab10',
                          alpha=0.6, s=30, edgecolors='w', linewidth=0.3)
    ax1.set_xlabel('PC1')
    ax1.set_ylabel('PC2')
    ax1.set_title('PCA (линейное снижение размерности)')
    ax1.grid(True, alpha=0.3)
    
    # t-SNE plot
    scatter2 = ax2.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='tab10',
                          alpha=0.6, s=30, edgecolors='w', linewidth=0.3)
    ax2.set_xlabel('t-SNE 1')
    ax2.set_ylabel('t-SNE 2')
    ax2.set_title('t-SNE (нелинейное снижение размерности)')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig_to_base64(fig)

# ============================================================================
# UMAP ILLUSTRATIONS
# ============================================================================

def generate_umap_comparison():
    """Generate UMAP visualization with different n_neighbors."""
    np.random.seed(42)
    digits = load_digits()
    X = digits.data[:500]
    y = digits.target[:500]
    X_scaled = StandardScaler().fit_transform(X)
    
    try:
        from umap import UMAP
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        n_neighbors_values = [5, 15, 50, 100]
        
        for idx, n_neighbors in enumerate(n_neighbors_values):
            ax = axes[idx // 2, idx % 2]
            umap = UMAP(n_components=2, n_neighbors=n_neighbors, random_state=42)
            X_umap = umap.fit_transform(X_scaled)
            
            scatter = ax.scatter(X_umap[:, 0], X_umap[:, 1], c=y, cmap='tab10',
                               alpha=0.6, s=30, edgecolors='w', linewidth=0.3)
            ax.set_title(f'n_neighbors = {n_neighbors}')
            ax.set_xlabel('UMAP 1')
            ax.set_ylabel('UMAP 2')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig_to_base64(fig)
    except ImportError:
        # Fallback: create a text-based figure
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, 'UMAP library not installed\npip install umap-learn',
               ha='center', va='center', fontsize=16)
        ax.axis('off')
        return fig_to_base64(fig)

def generate_umap_visualization():
    """Generate UMAP 2D visualization."""
    np.random.seed(42)
    digits = load_digits()
    X = digits.data[:600]
    y = digits.target[:600]
    X_scaled = StandardScaler().fit_transform(X)
    
    try:
        from umap import UMAP
        
        umap = UMAP(n_components=2, n_neighbors=15, random_state=42)
        X_umap = umap.fit_transform(X_scaled)
        
        fig, ax = plt.subplots(figsize=(12, 9))
        
        scatter = ax.scatter(X_umap[:, 0], X_umap[:, 1], c=y, cmap='tab10',
                            alpha=0.7, s=50, edgecolors='w', linewidth=0.5)
        
        # Add labels for each class
        for i in range(10):
            mask = y == i
            center = X_umap[mask].mean(axis=0)
            ax.text(center[0], center[1], str(i), fontsize=20, fontweight='bold',
                   ha='center', va='center', color='white',
                   bbox=dict(boxstyle='circle,pad=0.3', facecolor='black', alpha=0.7))
        
        ax.set_xlabel('UMAP компонента 1')
        ax.set_ylabel('UMAP компонента 2')
        ax.set_title('UMAP: Визуализация цифр MNIST')
        ax.grid(True, alpha=0.3)
        
        cbar = plt.colorbar(scatter, ax=ax, label='Цифра')
        cbar.set_ticks(range(10))
        
        return fig_to_base64(fig)
    except ImportError:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, 'UMAP library not installed\npip install umap-learn',
               ha='center', va='center', fontsize=16)
        ax.axis('off')
        return fig_to_base64(fig)

def generate_umap_vs_tsne():
    """Generate comparison of UMAP vs t-SNE."""
    np.random.seed(42)
    digits = load_digits()
    X = digits.data[:500]
    y = digits.target[:500]
    X_scaled = StandardScaler().fit_transform(X)
    
    # t-SNE
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    X_tsne = tsne.fit_transform(X_scaled)
    
    try:
        from umap import UMAP
        
        # UMAP
        umap = UMAP(n_components=2, n_neighbors=15, random_state=42)
        X_umap = umap.fit_transform(X_scaled)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
        
        # t-SNE plot
        scatter1 = ax1.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='tab10',
                              alpha=0.6, s=30, edgecolors='w', linewidth=0.3)
        ax1.set_xlabel('t-SNE 1')
        ax1.set_ylabel('t-SNE 2')
        ax1.set_title('t-SNE (медленнее, локальная структура)')
        ax1.grid(True, alpha=0.3)
        
        # UMAP plot
        scatter2 = ax2.scatter(X_umap[:, 0], X_umap[:, 1], c=y, cmap='tab10',
                              alpha=0.6, s=30, edgecolors='w', linewidth=0.3)
        ax2.set_xlabel('UMAP 1')
        ax2.set_ylabel('UMAP 2')
        ax2.set_title('UMAP (быстрее, глобальная + локальная)')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig_to_base64(fig)
    except ImportError:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, 'UMAP library not installed\npip install umap-learn',
               ha='center', va='center', fontsize=16)
        ax.axis('off')
        return fig_to_base64(fig)

# ============================================================================
# LDA ILLUSTRATIONS
# ============================================================================

def generate_lda_visualization():
    """Generate LDA visualization with class separation."""
    np.random.seed(42)
    X, y = make_classification(n_samples=300, n_features=5, n_informative=3,
                                n_redundant=0, n_classes=3, n_clusters_per_class=1,
                                random_state=42)
    X_scaled = StandardScaler().fit_transform(X)
    
    lda = LinearDiscriminantAnalysis(n_components=2)
    X_lda = lda.fit_transform(X_scaled, y)
    
    fig, ax = plt.subplots(figsize=(10, 7))
    
    colors = ['red', 'blue', 'green']
    for i, color in enumerate(colors):
        mask = y == i
        ax.scatter(X_lda[mask, 0], X_lda[mask, 1], c=color, label=f'Класс {i}',
                  alpha=0.6, s=50, edgecolors='w', linewidth=0.5)
    
    ax.set_xlabel('LD1 (первый дискриминант)')
    ax.set_ylabel('LD2 (второй дискриминант)')
    ax.set_title('LDA: Разделение классов')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)
    ax.axvline(x=0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)
    
    return fig_to_base64(fig)

def generate_lda_vs_pca():
    """Generate comparison of LDA vs PCA."""
    np.random.seed(42)
    X, y = make_classification(n_samples=300, n_features=5, n_informative=3,
                                n_redundant=0, n_classes=3, n_clusters_per_class=1,
                                random_state=42)
    X_scaled = StandardScaler().fit_transform(X)
    
    # PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    # LDA
    lda = LinearDiscriminantAnalysis(n_components=2)
    X_lda = lda.fit_transform(X_scaled, y)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    colors = ['red', 'blue', 'green']
    
    # PCA plot
    for i, color in enumerate(colors):
        mask = y == i
        ax1.scatter(X_pca[mask, 0], X_pca[mask, 1], c=color, label=f'Класс {i}',
                   alpha=0.6, s=30, edgecolors='w', linewidth=0.3)
    ax1.set_xlabel('PC1')
    ax1.set_ylabel('PC2')
    ax1.set_title('PCA (без учета классов)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # LDA plot
    for i, color in enumerate(colors):
        mask = y == i
        ax2.scatter(X_lda[mask, 0], X_lda[mask, 1], c=color, label=f'Класс {i}',
                   alpha=0.6, s=30, edgecolors='w', linewidth=0.3)
    ax2.set_xlabel('LD1')
    ax2.set_ylabel('LD2')
    ax2.set_title('LDA (максимизирует разделение классов)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig_to_base64(fig)

def generate_lda_decision_boundaries():
    """Generate LDA with decision boundaries in 2D."""
    np.random.seed(42)
    X, y = make_classification(n_samples=300, n_features=2, n_informative=2,
                                n_redundant=0, n_classes=3, n_clusters_per_class=1,
                                random_state=42)
    X_scaled = StandardScaler().fit_transform(X)
    
    lda = LinearDiscriminantAnalysis()
    lda.fit(X_scaled, y)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create mesh
    h = 0.02
    x_min, x_max = X_scaled[:, 0].min() - 1, X_scaled[:, 0].max() + 1
    y_min, y_max = X_scaled[:, 1].min() - 1, X_scaled[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    # Predict on mesh
    Z = lda.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Plot decision boundaries
    ax.contourf(xx, yy, Z, alpha=0.3, cmap='RdYlBu')
    
    # Plot data points
    colors = ['red', 'blue', 'green']
    for i, color in enumerate(colors):
        mask = y == i
        ax.scatter(X_scaled[mask, 0], X_scaled[mask, 1], c=color, label=f'Класс {i}',
                  alpha=0.7, s=50, edgecolors='w', linewidth=0.5)
    
    ax.set_xlabel('Признак 1')
    ax.set_ylabel('Признак 2')
    ax.set_title('LDA: Границы принятия решений')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return fig_to_base64(fig)

# ============================================================================
# SVD ILLUSTRATIONS
# ============================================================================

def generate_svd_singular_values():
    """Generate singular values plot."""
    np.random.seed(42)
    X, _ = make_classification(n_samples=300, n_features=20, n_informative=15,
                                n_redundant=5, random_state=42)
    X_scaled = StandardScaler().fit_transform(X)
    
    # Compute SVD
    U, s, Vt = np.linalg.svd(X_scaled, full_matrices=False)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Singular values
    ax1.bar(range(1, len(s) + 1), s, alpha=0.7, color='steelblue')
    ax1.set_xlabel('Индекс сингулярного значения')
    ax1.set_ylabel('Величина сингулярного значения')
    ax1.set_title('Сингулярные значения')
    ax1.grid(True, alpha=0.3)
    
    # Cumulative explained variance
    explained_var = s ** 2 / np.sum(s ** 2)
    cumsum = np.cumsum(explained_var)
    ax2.plot(range(1, len(cumsum) + 1), cumsum, 'ro-', linewidth=2, markersize=6)
    ax2.axhline(y=0.95, color='g', linestyle='--', linewidth=2, label='95% информации')
    ax2.set_xlabel('Число компонент')
    ax2.set_ylabel('Накопленная объясненная дисперсия')
    ax2.set_title('Накопленная дисперсия (SVD)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1.05])
    
    plt.tight_layout()
    return fig_to_base64(fig)

def generate_svd_matrix_visualization():
    """Generate matrix decomposition visualization."""
    np.random.seed(42)
    # Create a small matrix for visualization
    m, n = 6, 8
    rank = 3
    A = np.random.randn(m, rank) @ np.random.randn(rank, n)
    
    # SVD
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    # Original matrix
    im1 = axes[0].imshow(A, cmap='RdBu_r', aspect='auto')
    axes[0].set_title('A (исходная матрица)')
    axes[0].set_xlabel(f'{n} столбцов')
    axes[0].set_ylabel(f'{m} строк')
    plt.colorbar(im1, ax=axes[0])
    
    # U matrix
    im2 = axes[1].imshow(U, cmap='RdBu_r', aspect='auto')
    axes[1].set_title('U (левые сингулярные векторы)')
    axes[1].set_xlabel(f'{rank} компонент')
    axes[1].set_ylabel(f'{m} строк')
    plt.colorbar(im2, ax=axes[1])
    
    # Sigma (diagonal)
    S = np.diag(s)
    im3 = axes[2].imshow(S, cmap='RdBu_r', aspect='auto')
    axes[2].set_title('Σ (сингулярные значения)')
    axes[2].set_xlabel(f'{rank} компонент')
    axes[2].set_ylabel(f'{rank} компонент')
    plt.colorbar(im3, ax=axes[2])
    
    # V^T matrix
    im4 = axes[3].imshow(Vt, cmap='RdBu_r', aspect='auto')
    axes[3].set_title('V^T (правые сингулярные векторы)')
    axes[3].set_xlabel(f'{n} столбцов')
    axes[3].set_ylabel(f'{rank} компонент')
    plt.colorbar(im4, ax=axes[3])
    
    plt.suptitle('SVD: A = U Σ V^T', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    return fig_to_base64(fig)

def generate_svd_reconstruction():
    """Generate SVD reconstruction with different ranks."""
    np.random.seed(42)
    # Create a small sample matrix
    m, n = 50, 50
    X = np.random.randn(m, n)
    for i in range(m):
        X[i, :] += 0.5 * np.sin(i / 5)
    
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    ranks = [1, 3, 5, 10, 20, 50]
    
    for idx, r in enumerate(ranks):
        # Reconstruct with r components
        X_r = U[:, :r] @ np.diag(s[:r]) @ Vt[:r, :]
        error = np.linalg.norm(X - X_r, 'fro') / np.linalg.norm(X, 'fro')
        
        im = axes[idx].imshow(X_r, cmap='viridis', aspect='auto')
        axes[idx].set_title(f'Ранг {r}, Ошибка: {error:.3f}')
        axes[idx].axis('off')
        plt.colorbar(im, ax=axes[idx])
    
    plt.suptitle('SVD реконструкция при разных рангах', fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig_to_base64(fig)

# ============================================================================
# ISOMAP ILLUSTRATIONS
# ============================================================================

def generate_isomap_visualization():
    """Generate Isomap visualization on Swiss roll."""
    np.random.seed(42)
    X, color = make_swiss_roll(n_samples=1000, random_state=42)
    
    # Apply Isomap
    isomap = Isomap(n_components=2, n_neighbors=10)
    X_isomap = isomap.fit_transform(X)
    
    fig = plt.figure(figsize=(16, 6))
    
    # 3D original data
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.scatter(X[:, 0], X[:, 1], X[:, 2], c=color, cmap='viridis', s=10)
    ax1.set_title('Исходные данные (Swiss Roll)')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    
    # 2D Isomap projection
    ax2 = fig.add_subplot(122)
    scatter = ax2.scatter(X_isomap[:, 0], X_isomap[:, 1], c=color, cmap='viridis', s=10)
    ax2.set_title('Isomap проекция в 2D')
    ax2.set_xlabel('Isomap компонента 1')
    ax2.set_ylabel('Isomap компонента 2')
    ax2.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax2, label='Позиция')
    
    plt.tight_layout()
    return fig_to_base64(fig)

def generate_isomap_neighbors_comparison():
    """Generate Isomap with different n_neighbors."""
    np.random.seed(42)
    X, color = make_swiss_roll(n_samples=800, random_state=42)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    n_neighbors_values = [5, 10, 20, 50]
    
    for idx, n_neighbors in enumerate(n_neighbors_values):
        ax = axes[idx // 2, idx % 2]
        isomap = Isomap(n_components=2, n_neighbors=n_neighbors)
        X_isomap = isomap.fit_transform(X)
        
        scatter = ax.scatter(X_isomap[:, 0], X_isomap[:, 1], c=color, cmap='viridis',
                           alpha=0.6, s=10)
        ax.set_title(f'n_neighbors = {n_neighbors}')
        ax.set_xlabel('Isomap 1')
        ax.set_ylabel('Isomap 2')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig_to_base64(fig)

# ============================================================================
# LLE ILLUSTRATIONS
# ============================================================================

def generate_lle_visualization():
    """Generate LLE visualization on Swiss roll."""
    np.random.seed(42)
    X, color = make_swiss_roll(n_samples=1000, random_state=42)
    
    # Apply LLE
    lle = LocallyLinearEmbedding(n_components=2, n_neighbors=10, random_state=42)
    X_lle = lle.fit_transform(X)
    
    fig = plt.figure(figsize=(16, 6))
    
    # 3D original data
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.scatter(X[:, 0], X[:, 1], X[:, 2], c=color, cmap='viridis', s=10)
    ax1.set_title('Исходные данные (Swiss Roll)')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    
    # 2D LLE projection
    ax2 = fig.add_subplot(122)
    scatter = ax2.scatter(X_lle[:, 0], X_lle[:, 1], c=color, cmap='viridis', s=10)
    ax2.set_title('LLE проекция в 2D')
    ax2.set_xlabel('LLE компонента 1')
    ax2.set_ylabel('LLE компонента 2')
    ax2.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax2, label='Позиция')
    
    plt.tight_layout()
    return fig_to_base64(fig)

def generate_lle_neighbors_comparison():
    """Generate LLE with different n_neighbors."""
    np.random.seed(42)
    X, color = make_swiss_roll(n_samples=800, random_state=42)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    n_neighbors_values = [5, 10, 20, 30]
    
    for idx, n_neighbors in enumerate(n_neighbors_values):
        ax = axes[idx // 2, idx % 2]
        lle = LocallyLinearEmbedding(n_components=2, n_neighbors=n_neighbors, random_state=42)
        X_lle = lle.fit_transform(X)
        
        scatter = ax.scatter(X_lle[:, 0], X_lle[:, 1], c=color, cmap='viridis',
                           alpha=0.6, s=10)
        ax.set_title(f'n_neighbors = {n_neighbors}')
        ax.set_xlabel('LLE 1')
        ax.set_ylabel('LLE 2')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig_to_base64(fig)

# ============================================================================
# MDS ILLUSTRATIONS
# ============================================================================

def generate_mds_visualization():
    """Generate MDS visualization."""
    np.random.seed(42)
    X, y = make_classification(n_samples=300, n_features=10, n_informative=5,
                                n_redundant=2, n_classes=3, n_clusters_per_class=1,
                                random_state=42)
    X_scaled = StandardScaler().fit_transform(X)
    
    mds = MDS(n_components=2, random_state=42)
    X_mds = mds.fit_transform(X_scaled)
    
    fig, ax = plt.subplots(figsize=(10, 7))
    
    scatter = ax.scatter(X_mds[:, 0], X_mds[:, 1], c=y, cmap='viridis',
                        alpha=0.6, s=50, edgecolors='w', linewidth=0.5)
    
    ax.set_xlabel('MDS компонента 1')
    ax.set_ylabel('MDS компонента 2')
    ax.set_title(f'MDS: Визуализация в 2D\nСтресс: {mds.stress_:.2f}')
    ax.grid(True, alpha=0.3)
    
    plt.colorbar(scatter, ax=ax, label='Класс')
    
    return fig_to_base64(fig)

def generate_mds_stress_plot():
    """Generate MDS stress plot for different dimensions."""
    np.random.seed(42)
    X, _ = make_classification(n_samples=200, n_features=10, n_informative=5,
                                random_state=42)
    X_scaled = StandardScaler().fit_transform(X)
    
    dimensions = range(1, 8)
    stresses = []
    
    for n_dim in dimensions:
        mds = MDS(n_components=n_dim, random_state=42)
        mds.fit_transform(X_scaled)
        stresses.append(mds.stress_)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(dimensions, stresses, 'bo-', linewidth=2, markersize=8)
    ax.set_xlabel('Размерность')
    ax.set_ylabel('Стресс')
    ax.set_title('MDS: Стресс vs Размерность\n(меньше = лучше)')
    ax.grid(True, alpha=0.3)
    
    return fig_to_base64(fig)

def generate_mds_vs_pca():
    """Generate comparison of MDS vs PCA."""
    np.random.seed(42)
    X, y = make_classification(n_samples=300, n_features=10, n_informative=5,
                                n_redundant=2, n_classes=3, n_clusters_per_class=1,
                                random_state=42)
    X_scaled = StandardScaler().fit_transform(X)
    
    # PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    # MDS
    mds = MDS(n_components=2, random_state=42)
    X_mds = mds.fit_transform(X_scaled)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # PCA plot
    scatter1 = ax1.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis',
                          alpha=0.6, s=30, edgecolors='w', linewidth=0.3)
    ax1.set_xlabel('PC1')
    ax1.set_ylabel('PC2')
    ax1.set_title('PCA (быстрее, линейное)')
    ax1.grid(True, alpha=0.3)
    
    # MDS plot
    scatter2 = ax2.scatter(X_mds[:, 0], X_mds[:, 1], c=y, cmap='viridis',
                          alpha=0.6, s=30, edgecolors='w', linewidth=0.3)
    ax2.set_xlabel('MDS 1')
    ax2.set_ylabel('MDS 2')
    ax2.set_title(f'MDS (сохраняет расстояния, стресс={mds.stress_:.2f})')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig_to_base64(fig)

# ============================================================================
# MANIFOLD LEARNING COMPARISON
# ============================================================================

def generate_manifold_comparison():
    """Generate comparison of different manifold learning methods."""
    np.random.seed(42)
    X, color = make_swiss_roll(n_samples=800, random_state=42)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Original data
    ax = axes[0, 0]
    ax_3d = fig.add_subplot(2, 3, 1, projection='3d')
    ax_3d.scatter(X[:, 0], X[:, 1], X[:, 2], c=color, cmap='viridis', s=5)
    ax_3d.set_title('Исходные данные (Swiss Roll)')
    ax_3d.set_xlabel('X')
    ax_3d.set_ylabel('Y')
    ax_3d.set_zlabel('Z')
    axes[0, 0].remove()
    
    # PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    ax = axes[0, 1]
    ax.scatter(X_pca[:, 0], X_pca[:, 1], c=color, cmap='viridis', s=5)
    ax.set_title('PCA (линейное)')
    ax.grid(True, alpha=0.3)
    
    # MDS
    mds = MDS(n_components=2, random_state=42, max_iter=100)
    X_mds = mds.fit_transform(X)
    ax = axes[0, 2]
    ax.scatter(X_mds[:, 0], X_mds[:, 1], c=color, cmap='viridis', s=5)
    ax.set_title('MDS (сохраняет расстояния)')
    ax.grid(True, alpha=0.3)
    
    # Isomap
    isomap = Isomap(n_components=2, n_neighbors=10)
    X_isomap = isomap.fit_transform(X)
    ax = axes[1, 0]
    ax.scatter(X_isomap[:, 0], X_isomap[:, 1], c=color, cmap='viridis', s=5)
    ax.set_title('Isomap (геодезические расстояния)')
    ax.grid(True, alpha=0.3)
    
    # LLE
    lle = LocallyLinearEmbedding(n_components=2, n_neighbors=10, random_state=42)
    X_lle = lle.fit_transform(X)
    ax = axes[1, 1]
    ax.scatter(X_lle[:, 0], X_lle[:, 1], c=color, cmap='viridis', s=5)
    ax.set_title('LLE (локальная геометрия)')
    ax.grid(True, alpha=0.3)
    
    # t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    X_tsne = tsne.fit_transform(X)
    ax = axes[1, 2]
    ax.scatter(X_tsne[:, 0], X_tsne[:, 1], c=color, cmap='viridis', s=5)
    ax.set_title('t-SNE (нелинейное, локальная структура)')
    ax.grid(True, alpha=0.3)
    
    plt.suptitle('Сравнение методов снижения размерности', fontsize=16, fontweight='bold')
    plt.tight_layout()
    return fig_to_base64(fig)

def generate_manifold_datasets_comparison():
    """Generate comparison on different datasets."""
    np.random.seed(42)
    
    # Create datasets
    datasets = []
    # Swiss roll
    X_swiss, color_swiss = make_swiss_roll(n_samples=600, random_state=42)
    datasets.append(('Swiss Roll', X_swiss, color_swiss))
    
    # S-curve
    t = np.linspace(0, 3 * np.pi, 600)
    X_scurve = np.zeros((600, 3))
    X_scurve[:, 0] = np.sin(t)
    X_scurve[:, 1] = np.sign(t) * (np.cos(t) - 1)
    X_scurve[:, 2] = np.random.rand(600) * 0.5
    datasets.append(('S-Curve', X_scurve, t))
    
    fig, axes = plt.subplots(len(datasets), 4, figsize=(18, 10))
    
    for row_idx, (name, X, color) in enumerate(datasets):
        # 3D original
        ax_3d = fig.add_subplot(len(datasets), 4, row_idx * 4 + 1, projection='3d')
        ax_3d.scatter(X[:, 0], X[:, 1], X[:, 2], c=color, cmap='viridis', s=5)
        ax_3d.set_title(f'{name} (3D)')
        
        # PCA
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)
        ax = axes[row_idx, 1]
        ax.scatter(X_pca[:, 0], X_pca[:, 1], c=color, cmap='viridis', s=5)
        ax.set_title('PCA')
        ax.axis('off')
        
        # Isomap
        isomap = Isomap(n_components=2, n_neighbors=10)
        X_isomap = isomap.fit_transform(X)
        ax = axes[row_idx, 2]
        ax.scatter(X_isomap[:, 0], X_isomap[:, 1], c=color, cmap='viridis', s=5)
        ax.set_title('Isomap')
        ax.axis('off')
        
        # LLE
        lle = LocallyLinearEmbedding(n_components=2, n_neighbors=10, random_state=42)
        X_lle = lle.fit_transform(X)
        ax = axes[row_idx, 3]
        ax.scatter(X_lle[:, 0], X_lle[:, 1], c=color, cmap='viridis', s=5)
        ax.set_title('LLE')
        ax.axis('off')
        
        # Remove the first 2D axis
        axes[row_idx, 0].remove()
    
    plt.suptitle('Методы снижения размерности на разных датасетах', fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig_to_base64(fig)

# ============================================================================
# MAIN GENERATION FUNCTION
# ============================================================================

def generate_all_illustrations():
    """Generate all dimensionality reduction illustrations and return dictionary."""
    print("Generating dimensionality reduction illustrations...")
    
    illustrations = {}
    
    # PCA
    print("  PCA...")
    illustrations['pca_scree_plot'] = generate_pca_scree_plot()
    illustrations['pca_projection_2d'] = generate_pca_projection_2d()
    illustrations['pca_3d_projection'] = generate_pca_3d_projection()
    illustrations['pca_components'] = generate_pca_components_visualization()
    
    # t-SNE
    print("  t-SNE...")
    illustrations['tsne_perplexity'] = generate_tsne_perplexity_comparison()
    illustrations['tsne_visualization'] = generate_tsne_visualization()
    illustrations['tsne_vs_pca'] = generate_tsne_vs_pca()
    
    # UMAP
    print("  UMAP...")
    illustrations['umap_comparison'] = generate_umap_comparison()
    illustrations['umap_visualization'] = generate_umap_visualization()
    illustrations['umap_vs_tsne'] = generate_umap_vs_tsne()
    
    # LDA
    print("  LDA...")
    illustrations['lda_visualization'] = generate_lda_visualization()
    illustrations['lda_vs_pca'] = generate_lda_vs_pca()
    illustrations['lda_decision_boundaries'] = generate_lda_decision_boundaries()
    
    # SVD
    print("  SVD...")
    illustrations['svd_singular_values'] = generate_svd_singular_values()
    illustrations['svd_matrix'] = generate_svd_matrix_visualization()
    illustrations['svd_reconstruction'] = generate_svd_reconstruction()
    
    # Isomap
    print("  Isomap...")
    illustrations['isomap_visualization'] = generate_isomap_visualization()
    illustrations['isomap_neighbors'] = generate_isomap_neighbors_comparison()
    
    # LLE
    print("  LLE...")
    illustrations['lle_visualization'] = generate_lle_visualization()
    illustrations['lle_neighbors'] = generate_lle_neighbors_comparison()
    
    # MDS
    print("  MDS...")
    illustrations['mds_visualization'] = generate_mds_visualization()
    illustrations['mds_stress'] = generate_mds_stress_plot()
    illustrations['mds_vs_pca'] = generate_mds_vs_pca()
    
    # Manifold Learning
    print("  Manifold Learning comparison...")
    illustrations['manifold_comparison'] = generate_manifold_comparison()
    illustrations['manifold_datasets'] = generate_manifold_datasets_comparison()
    
    print(f"Generated {len(illustrations)} illustrations!")
    return illustrations

if __name__ == '__main__':
    illustrations = generate_all_illustrations()
    
    # Save to file for inspection
    import json
    output = {k: v[:100] + '...' for k, v in illustrations.items()}  # Truncate for readability
    print("\nGenerated illustrations:")
    for key in illustrations.keys():
        print(f"  - {key}")
