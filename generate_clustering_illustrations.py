#!/usr/bin/env python3
"""
Generate matplotlib illustrations for clustering cheatsheets.
This script creates high-quality visualizations and encodes them as base64 for inline HTML embedding.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, SpectralClustering, MeanShift, AffinityPropagation, estimate_bandwidth
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_blobs, make_moons, make_circles
from sklearn.metrics import silhouette_score, silhouette_samples
from scipy.cluster.hierarchy import dendrogram, linkage
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

def generate_sample_data():
    """Generate sample datasets for clustering demonstrations."""
    np.random.seed(42)
    
    # Blobs for K-means, GMM, etc.
    X_blobs, y_blobs = make_blobs(n_samples=300, centers=4, n_features=2, 
                                   cluster_std=0.6, random_state=42)
    
    # Moons for DBSCAN, Spectral
    X_moons, y_moons = make_moons(n_samples=300, noise=0.1, random_state=42)
    
    # Circles for Spectral
    X_circles, y_circles = make_circles(n_samples=300, noise=0.05, factor=0.5, random_state=42)
    
    return X_blobs, X_moons, X_circles

# ============================================================================
# K-MEANS ILLUSTRATIONS
# ============================================================================

def generate_kmeans_elbow():
    """Generate elbow method plot for K-means."""
    X, _ = make_blobs(n_samples=300, centers=4, random_state=42)
    X_scaled = StandardScaler().fit_transform(X)
    
    inertias = []
    K_range = range(1, 11)
    
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X_scaled)
        inertias.append(kmeans.inertia_)
    
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(K_range, inertias, 'bx-', linewidth=2, markersize=8)
    ax.set_xlabel('Количество кластеров K')
    ax.set_ylabel('Инерция (WCSS)')
    ax.set_title('Метод локтя для K-means')
    ax.grid(True, alpha=0.3)
    
    # Highlight the "elbow" at k=4
    ax.axvline(x=4, color='red', linestyle='--', alpha=0.5, label='Оптимальное K=4')
    ax.legend()
    
    return fig_to_base64(fig)

def generate_kmeans_clustering():
    """Generate K-means clustering visualization."""
    X, _ = make_blobs(n_samples=300, centers=4, random_state=42)
    X_scaled = StandardScaler().fit_transform(X)
    
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    centroids = kmeans.cluster_centers_
    
    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(X_scaled[:, 0], X_scaled[:, 1], 
                        c=labels, cmap='viridis', alpha=0.6, s=50, edgecolors='w', linewidth=0.5)
    ax.scatter(centroids[:, 0], centroids[:, 1],
              c='red', marker='X', s=300, 
              edgecolors='black', linewidths=2, label='Центроиды')
    
    ax.set_xlabel('Признак 1')
    ax.set_ylabel('Признак 2')
    ax.set_title('K-means кластеризация (K=4)')
    ax.legend()
    plt.colorbar(scatter, ax=ax, label='Кластер')
    
    return fig_to_base64(fig)

def generate_kmeans_silhouette():
    """Generate silhouette plot for K-means."""
    X, _ = make_blobs(n_samples=300, centers=4, random_state=42)
    X_scaled = StandardScaler().fit_transform(X)
    
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    
    silhouette_vals = silhouette_samples(X_scaled, labels)
    silhouette_avg = silhouette_score(X_scaled, labels)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    y_lower = 10
    for i in range(4):
        cluster_silhouette_vals = silhouette_vals[labels == i]
        cluster_silhouette_vals.sort()
        
        size_cluster_i = cluster_silhouette_vals.shape[0]
        y_upper = y_lower + size_cluster_i
        
        color = plt.cm.viridis(i / 4)
        ax.fill_betweenx(np.arange(y_lower, y_upper),
                        0, cluster_silhouette_vals,
                        facecolor=color, edgecolor=color, alpha=0.7)
        
        ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
        y_lower = y_upper + 10
    
    ax.set_xlabel('Коэффициент силуэта')
    ax.set_ylabel('Кластер')
    ax.set_title(f'Silhouette анализ (средний коэф. = {silhouette_avg:.3f})')
    ax.axvline(x=silhouette_avg, color="red", linestyle="--", linewidth=2, label='Среднее')
    ax.set_yticks([])
    ax.legend()
    
    return fig_to_base64(fig)

# ============================================================================
# HIERARCHICAL CLUSTERING ILLUSTRATIONS
# ============================================================================

def generate_hierarchical_dendrogram():
    """Generate dendrogram for hierarchical clustering."""
    X, _ = make_blobs(n_samples=50, centers=3, random_state=42)
    X_scaled = StandardScaler().fit_transform(X)
    
    linkage_matrix = linkage(X_scaled, method='ward')
    
    fig, ax = plt.subplots(figsize=(10, 6))
    dendrogram(linkage_matrix, ax=ax, color_threshold=7)
    ax.set_xlabel('Индекс образца')
    ax.set_ylabel('Расстояние')
    ax.set_title('Дендрограмма иерархической кластеризации (метод Ward)')
    ax.axhline(y=7, color='red', linestyle='--', linewidth=2, label='Порог обрезки')
    ax.legend()
    
    return fig_to_base64(fig)

def generate_hierarchical_clustering():
    """Generate hierarchical clustering visualization."""
    X, _ = make_blobs(n_samples=300, centers=3, random_state=42)
    X_scaled = StandardScaler().fit_transform(X)
    
    hierarchical = AgglomerativeClustering(n_clusters=3, linkage='ward')
    labels = hierarchical.fit_predict(X_scaled)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(X_scaled[:, 0], X_scaled[:, 1], 
                        c=labels, cmap='viridis', alpha=0.6, s=50, edgecolors='w', linewidth=0.5)
    ax.set_xlabel('Признак 1')
    ax.set_ylabel('Признак 2')
    ax.set_title('Иерархическая кластеризация (3 кластера)')
    plt.colorbar(scatter, ax=ax, label='Кластер')
    
    return fig_to_base64(fig)

# ============================================================================
# DBSCAN ILLUSTRATIONS
# ============================================================================

def generate_dbscan_clustering():
    """Generate DBSCAN clustering visualization."""
    X, _ = make_moons(n_samples=300, noise=0.1, random_state=42)
    X_scaled = StandardScaler().fit_transform(X)
    
    dbscan = DBSCAN(eps=0.3, min_samples=5)
    labels = dbscan.fit_predict(X_scaled)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot noise points separately
    noise_mask = labels == -1
    ax.scatter(X_scaled[noise_mask, 0], X_scaled[noise_mask, 1], 
              c='black', marker='x', s=50, alpha=0.5, label='Шум')
    
    # Plot clusters
    unique_labels = set(labels)
    unique_labels.discard(-1)
    
    for label in unique_labels:
        mask = labels == label
        ax.scatter(X_scaled[mask, 0], X_scaled[mask, 1], 
                  s=50, alpha=0.6, edgecolors='w', linewidth=0.5,
                  label=f'Кластер {label}')
    
    ax.set_xlabel('Признак 1')
    ax.set_ylabel('Признак 2')
    ax.set_title('DBSCAN кластеризация (eps=0.3, min_samples=5)')
    ax.legend()
    
    return fig_to_base64(fig)

def generate_dbscan_parameters():
    """Generate visualization showing effect of different DBSCAN parameters."""
    X, _ = make_moons(n_samples=300, noise=0.1, random_state=42)
    X_scaled = StandardScaler().fit_transform(X)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    params = [(0.2, 5), (0.3, 5), (0.5, 5), (0.3, 10)]
    
    for idx, (eps, min_samples) in enumerate(params):
        ax = axes[idx // 2, idx % 2]
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels = dbscan.fit_predict(X_scaled)
        
        noise_mask = labels == -1
        ax.scatter(X_scaled[noise_mask, 0], X_scaled[noise_mask, 1], 
                  c='black', marker='x', s=30, alpha=0.5)
        
        unique_labels = set(labels)
        unique_labels.discard(-1)
        
        for label in unique_labels:
            mask = labels == label
            ax.scatter(X_scaled[mask, 0], X_scaled[mask, 1], 
                      s=30, alpha=0.6, edgecolors='w', linewidth=0.3)
        
        n_clusters = len(unique_labels)
        n_noise = np.sum(noise_mask)
        ax.set_title(f'eps={eps}, min_samples={min_samples}\n'
                    f'Кластеров: {n_clusters}, Шум: {n_noise}')
        ax.set_xlabel('Признак 1')
        ax.set_ylabel('Признак 2')
    
    plt.tight_layout()
    return fig_to_base64(fig)

# ============================================================================
# GMM ILLUSTRATIONS
# ============================================================================

def generate_gmm_clustering():
    """Generate GMM clustering with probability contours."""
    X, _ = make_blobs(n_samples=300, centers=3, random_state=42, cluster_std=0.8)
    X_scaled = StandardScaler().fit_transform(X)
    
    gmm = GaussianMixture(n_components=3, covariance_type='full', random_state=42)
    gmm.fit(X_scaled)
    labels = gmm.predict(X_scaled)
    probs = gmm.predict_proba(X_scaled)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot points with transparency based on confidence
    for i in range(3):
        mask = labels == i
        confidence = probs[mask, i]
        ax.scatter(X_scaled[mask, 0], X_scaled[mask, 1], 
                  alpha=confidence, s=50, edgecolors='w', linewidth=0.5)
    
    # Plot Gaussian contours
    x_min, x_max = X_scaled[:, 0].min() - 1, X_scaled[:, 0].max() + 1
    y_min, y_max = X_scaled[:, 1].min() - 1, X_scaled[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))
    Z = -gmm.score_samples(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    ax.contour(xx, yy, Z, levels=10, linewidths=1, alpha=0.3)
    
    ax.set_xlabel('Признак 1')
    ax.set_ylabel('Признак 2')
    ax.set_title('Gaussian Mixture Model (3 компоненты)')
    
    return fig_to_base64(fig)

def generate_gmm_comparison():
    """Generate comparison of different covariance types."""
    X, _ = make_blobs(n_samples=300, centers=3, random_state=42, cluster_std=[1.0, 0.5, 1.5])
    X_scaled = StandardScaler().fit_transform(X)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    cov_types = ['full', 'tied', 'diag', 'spherical']
    
    for idx, cov_type in enumerate(cov_types):
        ax = axes[idx // 2, idx % 2]
        gmm = GaussianMixture(n_components=3, covariance_type=cov_type, random_state=42)
        gmm.fit(X_scaled)
        labels = gmm.predict(X_scaled)
        
        scatter = ax.scatter(X_scaled[:, 0], X_scaled[:, 1], 
                           c=labels, cmap='viridis', alpha=0.6, s=30,
                           edgecolors='w', linewidth=0.3)
        
        # Plot contours
        x_min, x_max = X_scaled[:, 0].min() - 1, X_scaled[:, 0].max() + 1
        y_min, y_max = X_scaled[:, 1].min() - 1, X_scaled[:, 1].max() + 1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                            np.linspace(y_min, y_max, 100))
        Z = -gmm.score_samples(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        ax.contour(xx, yy, Z, levels=10, linewidths=1, alpha=0.3)
        
        ax.set_title(f'Covariance: {cov_type}')
        ax.set_xlabel('Признак 1')
        ax.set_ylabel('Признак 2')
    
    plt.tight_layout()
    return fig_to_base64(fig)

# ============================================================================
# SPECTRAL CLUSTERING ILLUSTRATIONS
# ============================================================================

def generate_spectral_clustering():
    """Generate spectral clustering on non-convex shapes."""
    X, _ = make_moons(n_samples=300, noise=0.1, random_state=42)
    X_scaled = StandardScaler().fit_transform(X)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # K-means (fails on non-convex)
    kmeans = KMeans(n_clusters=2, random_state=42)
    labels_kmeans = kmeans.fit_predict(X_scaled)
    
    ax1.scatter(X_scaled[:, 0], X_scaled[:, 1], 
               c=labels_kmeans, cmap='viridis', alpha=0.6, s=50,
               edgecolors='w', linewidth=0.5)
    ax1.set_title('K-means (не справляется)')
    ax1.set_xlabel('Признак 1')
    ax1.set_ylabel('Признак 2')
    
    # Spectral clustering (succeeds)
    spectral = SpectralClustering(n_clusters=2, affinity='nearest_neighbors', 
                                   n_neighbors=10, random_state=42)
    labels_spectral = spectral.fit_predict(X_scaled)
    
    ax2.scatter(X_scaled[:, 0], X_scaled[:, 1], 
               c=labels_spectral, cmap='viridis', alpha=0.6, s=50,
               edgecolors='w', linewidth=0.5)
    ax2.set_title('Spectral Clustering (справляется)')
    ax2.set_xlabel('Признак 1')
    ax2.set_ylabel('Признак 2')
    
    plt.tight_layout()
    return fig_to_base64(fig)

def generate_spectral_circles():
    """Generate spectral clustering on circles."""
    X, _ = make_circles(n_samples=300, noise=0.05, factor=0.5, random_state=42)
    X_scaled = StandardScaler().fit_transform(X)
    
    spectral = SpectralClustering(n_clusters=2, affinity='nearest_neighbors', 
                                   n_neighbors=10, random_state=42)
    labels = spectral.fit_predict(X_scaled)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(X_scaled[:, 0], X_scaled[:, 1], 
                        c=labels, cmap='viridis', alpha=0.6, s=50,
                        edgecolors='w', linewidth=0.5)
    ax.set_xlabel('Признак 1')
    ax.set_ylabel('Признак 2')
    ax.set_title('Spectral Clustering на концентрических кругах')
    plt.colorbar(scatter, ax=ax, label='Кластер')
    
    return fig_to_base64(fig)

# ============================================================================
# MEAN SHIFT ILLUSTRATIONS
# ============================================================================

def generate_mean_shift_clustering():
    """Generate Mean Shift clustering visualization."""
    X, _ = make_blobs(n_samples=300, centers=3, random_state=42, cluster_std=0.6)
    X_scaled = StandardScaler().fit_transform(X)
    
    bandwidth = estimate_bandwidth(X_scaled, quantile=0.2, n_samples=100)
    ms = MeanShift(bandwidth=bandwidth)
    labels = ms.fit_predict(X_scaled)
    cluster_centers = ms.cluster_centers_
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    n_clusters = len(np.unique(labels))
    scatter = ax.scatter(X_scaled[:, 0], X_scaled[:, 1], 
                        c=labels, cmap='viridis', alpha=0.6, s=50,
                        edgecolors='w', linewidth=0.5)
    
    ax.scatter(cluster_centers[:, 0], cluster_centers[:, 1],
              c='red', marker='*', s=300, 
              edgecolors='black', linewidths=2, label='Центры')
    
    ax.set_xlabel('Признак 1')
    ax.set_ylabel('Признак 2')
    ax.set_title(f'Mean Shift кластеризация ({n_clusters} кластеров)')
    ax.legend()
    plt.colorbar(scatter, ax=ax, label='Кластер')
    
    return fig_to_base64(fig)

def generate_mean_shift_bandwidth():
    """Generate visualization showing effect of bandwidth."""
    X, _ = make_blobs(n_samples=300, centers=3, random_state=42, cluster_std=0.8)
    X_scaled = StandardScaler().fit_transform(X)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    bandwidths = [0.5, 1.0, 2.0]
    
    for idx, bw in enumerate(bandwidths):
        ax = axes[idx]
        ms = MeanShift(bandwidth=bw)
        labels = ms.fit_predict(X_scaled)
        cluster_centers = ms.cluster_centers_
        n_clusters = len(cluster_centers)
        
        ax.scatter(X_scaled[:, 0], X_scaled[:, 1], 
                  c=labels, cmap='viridis', alpha=0.6, s=30,
                  edgecolors='w', linewidth=0.3)
        ax.scatter(cluster_centers[:, 0], cluster_centers[:, 1],
                  c='red', marker='*', s=200, 
                  edgecolors='black', linewidths=1.5)
        
        ax.set_title(f'Bandwidth={bw}\n({n_clusters} кластеров)')
        ax.set_xlabel('Признак 1')
        ax.set_ylabel('Признак 2')
    
    plt.tight_layout()
    return fig_to_base64(fig)

# ============================================================================
# AFFINITY PROPAGATION ILLUSTRATIONS
# ============================================================================

def generate_affinity_propagation_clustering():
    """Generate Affinity Propagation clustering visualization."""
    X, _ = make_blobs(n_samples=150, centers=3, random_state=42, cluster_std=0.5)
    X_scaled = StandardScaler().fit_transform(X)
    
    ap = AffinityPropagation(random_state=42, damping=0.9)
    labels = ap.fit_predict(X_scaled)
    cluster_centers_indices = ap.cluster_centers_indices_
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    n_clusters = len(cluster_centers_indices)
    scatter = ax.scatter(X_scaled[:, 0], X_scaled[:, 1], 
                        c=labels, cmap='viridis', alpha=0.6, s=50,
                        edgecolors='w', linewidth=0.5)
    
    # Mark exemplars
    ax.scatter(X_scaled[cluster_centers_indices, 0], 
              X_scaled[cluster_centers_indices, 1],
              c='red', marker='*', s=300, 
              edgecolors='black', linewidths=2, label='Exemplars')
    
    ax.set_xlabel('Признак 1')
    ax.set_ylabel('Признак 2')
    ax.set_title(f'Affinity Propagation ({n_clusters} кластеров)')
    ax.legend()
    plt.colorbar(scatter, ax=ax, label='Кластер')
    
    return fig_to_base64(fig)

def generate_affinity_propagation_preference():
    """Generate visualization showing effect of preference parameter."""
    X, _ = make_blobs(n_samples=150, centers=3, random_state=42, cluster_std=0.6)
    X_scaled = StandardScaler().fit_transform(X)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    preferences = [None, -50, -5]
    titles = ['По умолчанию', 'Мало кластеров', 'Много кластеров']
    
    for idx, (pref, title) in enumerate(zip(preferences, titles)):
        ax = axes[idx]
        ap = AffinityPropagation(random_state=42, preference=pref, damping=0.9)
        try:
            labels = ap.fit_predict(X_scaled)
            cluster_centers_indices = ap.cluster_centers_indices_
            n_clusters = len(cluster_centers_indices)
            
            ax.scatter(X_scaled[:, 0], X_scaled[:, 1], 
                      c=labels, cmap='viridis', alpha=0.6, s=30,
                      edgecolors='w', linewidth=0.3)
            ax.scatter(X_scaled[cluster_centers_indices, 0], 
                      X_scaled[cluster_centers_indices, 1],
                      c='red', marker='*', s=200, 
                      edgecolors='black', linewidths=1.5)
            
            ax.set_title(f'{title}\n({n_clusters} кластеров)')
        except:
            ax.text(0.5, 0.5, 'Не сошлось', ha='center', va='center',
                   transform=ax.transAxes)
            ax.set_title(f'{title}\n(ошибка)')
        
        ax.set_xlabel('Признак 1')
        ax.set_ylabel('Признак 2')
    
    plt.tight_layout()
    return fig_to_base64(fig)

# ============================================================================
# MAIN GENERATION FUNCTION
# ============================================================================

def generate_all_illustrations():
    """Generate all clustering illustrations and return dictionary."""
    print("Generating clustering illustrations...")
    
    illustrations = {}
    
    # K-means
    print("  K-means...")
    illustrations['kmeans_elbow'] = generate_kmeans_elbow()
    illustrations['kmeans_clustering'] = generate_kmeans_clustering()
    illustrations['kmeans_silhouette'] = generate_kmeans_silhouette()
    
    # Hierarchical
    print("  Hierarchical clustering...")
    illustrations['hierarchical_dendrogram'] = generate_hierarchical_dendrogram()
    illustrations['hierarchical_clustering'] = generate_hierarchical_clustering()
    
    # DBSCAN
    print("  DBSCAN...")
    illustrations['dbscan_clustering'] = generate_dbscan_clustering()
    illustrations['dbscan_parameters'] = generate_dbscan_parameters()
    
    # GMM
    print("  GMM...")
    illustrations['gmm_clustering'] = generate_gmm_clustering()
    illustrations['gmm_comparison'] = generate_gmm_comparison()
    
    # Spectral
    print("  Spectral clustering...")
    illustrations['spectral_clustering'] = generate_spectral_clustering()
    illustrations['spectral_circles'] = generate_spectral_circles()
    
    # Mean Shift
    print("  Mean Shift...")
    illustrations['mean_shift_clustering'] = generate_mean_shift_clustering()
    illustrations['mean_shift_bandwidth'] = generate_mean_shift_bandwidth()
    
    # Affinity Propagation
    print("  Affinity Propagation...")
    illustrations['affinity_propagation_clustering'] = generate_affinity_propagation_clustering()
    illustrations['affinity_propagation_preference'] = generate_affinity_propagation_preference()
    
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
