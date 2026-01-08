#!/usr/bin/env python3
"""
Generate matplotlib illustrations for anomaly detection cheatsheets.
This script creates high-quality visualizations and encodes them as base64 for inline HTML embedding.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.covariance import EllipticEnvelope
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_blobs, make_moons
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

def generate_sample_data_with_outliers():
    """Generate sample datasets with outliers for anomaly detection."""
    np.random.seed(42)
    
    # Normal data
    X_normal, _ = make_blobs(n_samples=280, centers=1, n_features=2, 
                             cluster_std=0.5, center_box=(-2.0, 2.0), random_state=42)
    
    # Add outliers
    X_outliers = np.random.uniform(low=-6, high=6, size=(20, 2))
    X = np.vstack([X_normal, X_outliers])
    
    # True labels (for visualization, not used in training)
    y_true = np.ones(len(X), dtype=int)
    y_true[-20:] = -1
    
    return X, y_true

# ============================================================================
# ISOLATION FOREST ILLUSTRATIONS
# ============================================================================

def generate_isolation_forest_basic():
    """Generate basic Isolation Forest visualization."""
    X, y_true = generate_sample_data_with_outliers()
    X_scaled = StandardScaler().fit_transform(X)
    
    # Train Isolation Forest
    iso = IsolationForest(contamination=0.1, random_state=42)
    predictions = iso.fit_predict(X_scaled)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot normal points
    mask_normal = predictions == 1
    ax.scatter(X_scaled[mask_normal, 0], X_scaled[mask_normal, 1], 
              c='blue', alpha=0.6, s=50, edgecolors='w', linewidth=0.5,
              label='Нормальные')
    
    # Plot anomalies
    mask_anomaly = predictions == -1
    ax.scatter(X_scaled[mask_anomaly, 0], X_scaled[mask_anomaly, 1], 
              c='red', marker='x', s=100, linewidth=2,
              label='Аномалии')
    
    ax.set_xlabel('Признак 1')
    ax.set_ylabel('Признак 2')
    ax.set_title('Isolation Forest: Обнаружение аномалий')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return fig_to_base64(fig)

def generate_isolation_forest_scores():
    """Generate Isolation Forest scores distribution."""
    X, y_true = generate_sample_data_with_outliers()
    X_scaled = StandardScaler().fit_transform(X)
    
    iso = IsolationForest(contamination=0.1, random_state=42)
    predictions = iso.fit_predict(X_scaled)
    scores = iso.score_samples(X_scaled)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Histogram of scores
    ax.hist(scores[predictions == 1], bins=30, alpha=0.6, 
           color='blue', label='Нормальные', edgecolor='black')
    ax.hist(scores[predictions == -1], bins=30, alpha=0.6, 
           color='red', label='Аномалии', edgecolor='black')
    
    ax.set_xlabel('Anomaly Score (чем меньше, тем аномальнее)')
    ax.set_ylabel('Количество')
    ax.set_title('Распределение Anomaly Scores')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return fig_to_base64(fig)

def generate_isolation_forest_contamination():
    """Generate visualization showing effect of contamination parameter."""
    X, y_true = generate_sample_data_with_outliers()
    X_scaled = StandardScaler().fit_transform(X)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    contaminations = [0.05, 0.1, 0.15, 0.2]
    
    for idx, cont in enumerate(contaminations):
        ax = axes[idx // 2, idx % 2]
        
        iso = IsolationForest(contamination=cont, random_state=42)
        predictions = iso.fit_predict(X_scaled)
        
        # Count anomalies
        n_anomalies = np.sum(predictions == -1)
        
        # Plot
        mask_normal = predictions == 1
        mask_anomaly = predictions == -1
        
        ax.scatter(X_scaled[mask_normal, 0], X_scaled[mask_normal, 1], 
                  c='blue', alpha=0.6, s=30, edgecolors='w', linewidth=0.3,
                  label='Нормальные')
        ax.scatter(X_scaled[mask_anomaly, 0], X_scaled[mask_anomaly, 1], 
                  c='red', marker='x', s=80, linewidth=2,
                  label='Аномалии')
        
        ax.set_title(f'contamination={cont}\nАномалий: {n_anomalies}')
        ax.set_xlabel('Признак 1')
        ax.set_ylabel('Признак 2')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig_to_base64(fig)

# ============================================================================
# LOCAL OUTLIER FACTOR ILLUSTRATIONS
# ============================================================================

def generate_lof_basic():
    """Generate basic LOF visualization."""
    X, y_true = generate_sample_data_with_outliers()
    X_scaled = StandardScaler().fit_transform(X)
    
    # Train LOF
    lof = LocalOutlierFactor(n_neighbors=20, contamination=0.1)
    predictions = lof.fit_predict(X_scaled)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot normal points
    mask_normal = predictions == 1
    ax.scatter(X_scaled[mask_normal, 0], X_scaled[mask_normal, 1], 
              c='blue', alpha=0.6, s=50, edgecolors='w', linewidth=0.5,
              label='Нормальные')
    
    # Plot anomalies
    mask_anomaly = predictions == -1
    ax.scatter(X_scaled[mask_anomaly, 0], X_scaled[mask_anomaly, 1], 
              c='red', marker='x', s=100, linewidth=2,
              label='Локальные выбросы')
    
    ax.set_xlabel('Признак 1')
    ax.set_ylabel('Признак 2')
    ax.set_title('Local Outlier Factor: Обнаружение локальных аномалий')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return fig_to_base64(fig)

def generate_lof_scores():
    """Generate LOF scores visualization."""
    X, y_true = generate_sample_data_with_outliers()
    X_scaled = StandardScaler().fit_transform(X)
    
    lof = LocalOutlierFactor(n_neighbors=20, contamination=0.1)
    predictions = lof.fit_predict(X_scaled)
    lof_scores = lof.negative_outlier_factor_
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left: scatter plot with color by LOF score
    scatter = ax1.scatter(X_scaled[:, 0], X_scaled[:, 1], 
                         c=lof_scores, cmap='coolwarm', 
                         s=50, edgecolors='w', linewidth=0.5)
    plt.colorbar(scatter, ax=ax1, label='LOF Score (negative_outlier_factor)')
    ax1.set_xlabel('Признак 1')
    ax1.set_ylabel('Признак 2')
    ax1.set_title('LOF Scores по точкам')
    ax1.grid(True, alpha=0.3)
    
    # Right: histogram of LOF scores
    ax2.hist(lof_scores[predictions == 1], bins=30, alpha=0.6, 
            color='blue', label='Нормальные', edgecolor='black')
    ax2.hist(lof_scores[predictions == -1], bins=30, alpha=0.6, 
            color='red', label='Аномалии', edgecolor='black')
    ax2.set_xlabel('LOF Score (чем меньше, тем аномальнее)')
    ax2.set_ylabel('Количество')
    ax2.set_title('Распределение LOF Scores')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig_to_base64(fig)

def generate_lof_neighbors():
    """Generate visualization showing effect of n_neighbors parameter."""
    X, y_true = generate_sample_data_with_outliers()
    X_scaled = StandardScaler().fit_transform(X)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    n_neighbors_list = [5, 10, 20, 35]
    
    for idx, n_neighbors in enumerate(n_neighbors_list):
        ax = axes[idx // 2, idx % 2]
        
        lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=0.1)
        predictions = lof.fit_predict(X_scaled)
        
        # Count anomalies
        n_anomalies = np.sum(predictions == -1)
        
        # Plot
        mask_normal = predictions == 1
        mask_anomaly = predictions == -1
        
        ax.scatter(X_scaled[mask_normal, 0], X_scaled[mask_normal, 1], 
                  c='blue', alpha=0.6, s=30, edgecolors='w', linewidth=0.3,
                  label='Нормальные')
        ax.scatter(X_scaled[mask_anomaly, 0], X_scaled[mask_anomaly, 1], 
                  c='red', marker='x', s=80, linewidth=2,
                  label='Аномалии')
        
        ax.set_title(f'n_neighbors={n_neighbors}\nАномалий: {n_anomalies}')
        ax.set_xlabel('Признак 1')
        ax.set_ylabel('Признак 2')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig_to_base64(fig)

# ============================================================================
# ONE-CLASS SVM ILLUSTRATIONS
# ============================================================================

def generate_one_class_svm_basic():
    """Generate basic One-Class SVM visualization with decision boundary."""
    X, y_true = generate_sample_data_with_outliers()
    X_scaled = StandardScaler().fit_transform(X)
    
    # Train One-Class SVM
    ocsvm = OneClassSVM(kernel='rbf', gamma='auto', nu=0.1)
    ocsvm.fit(X_scaled)
    predictions = ocsvm.predict(X_scaled)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Create mesh for decision boundary
    x_min, x_max = X_scaled[:, 0].min() - 1, X_scaled[:, 0].max() + 1
    y_min, y_max = X_scaled[:, 1].min() - 1, X_scaled[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))
    
    # Decision function
    Z = ocsvm.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Plot decision boundary and margins
    ax.contour(xx, yy, Z, levels=[0], linewidths=2, colors='black', linestyles='solid')
    ax.contourf(xx, yy, Z, levels=np.linspace(Z.min(), 0, 7), cmap='Blues', alpha=0.3)
    
    # Plot points
    mask_normal = predictions == 1
    mask_anomaly = predictions == -1
    
    ax.scatter(X_scaled[mask_normal, 0], X_scaled[mask_normal, 1], 
              c='blue', alpha=0.7, s=50, edgecolors='w', linewidth=0.5,
              label='Нормальные')
    ax.scatter(X_scaled[mask_anomaly, 0], X_scaled[mask_anomaly, 1], 
              c='red', marker='x', s=100, linewidth=2,
              label='Аномалии')
    
    ax.set_xlabel('Признак 1')
    ax.set_ylabel('Признак 2')
    ax.set_title('One-Class SVM: Граница решения (RBF kernel)')
    ax.legend()
    
    return fig_to_base64(fig)

def generate_one_class_svm_kernels():
    """Generate comparison of different kernel types."""
    X, y_true = generate_sample_data_with_outliers()
    X_scaled = StandardScaler().fit_transform(X)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    kernels = ['rbf', 'linear']
    
    for idx, kernel in enumerate(kernels):
        ax = axes[idx]
        
        ocsvm = OneClassSVM(kernel=kernel, gamma='auto', nu=0.1)
        ocsvm.fit(X_scaled)
        predictions = ocsvm.predict(X_scaled)
        
        # Create mesh
        x_min, x_max = X_scaled[:, 0].min() - 1, X_scaled[:, 0].max() + 1
        y_min, y_max = X_scaled[:, 1].min() - 1, X_scaled[:, 1].max() + 1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                            np.linspace(y_min, y_max, 200))
        
        Z = ocsvm.decision_function(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        
        # Plot
        ax.contour(xx, yy, Z, levels=[0], linewidths=2, colors='black')
        ax.contourf(xx, yy, Z, levels=np.linspace(Z.min(), 0, 7), cmap='Blues', alpha=0.3)
        
        mask_normal = predictions == 1
        mask_anomaly = predictions == -1
        
        ax.scatter(X_scaled[mask_normal, 0], X_scaled[mask_normal, 1], 
                  c='blue', alpha=0.7, s=40, edgecolors='w', linewidth=0.3)
        ax.scatter(X_scaled[mask_anomaly, 0], X_scaled[mask_anomaly, 1], 
                  c='red', marker='x', s=80, linewidth=2)
        
        n_anomalies = np.sum(predictions == -1)
        ax.set_title(f'Kernel: {kernel}\nАномалий: {n_anomalies}')
        ax.set_xlabel('Признак 1')
        ax.set_ylabel('Признак 2')
    
    plt.tight_layout()
    return fig_to_base64(fig)

def generate_one_class_svm_nu():
    """Generate visualization showing effect of nu parameter."""
    X, y_true = generate_sample_data_with_outliers()
    X_scaled = StandardScaler().fit_transform(X)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    nu_values = [0.05, 0.1, 0.15, 0.25]
    
    for idx, nu in enumerate(nu_values):
        ax = axes[idx // 2, idx % 2]
        
        ocsvm = OneClassSVM(kernel='rbf', gamma='auto', nu=nu)
        ocsvm.fit(X_scaled)
        predictions = ocsvm.predict(X_scaled)
        
        # Create mesh
        x_min, x_max = X_scaled[:, 0].min() - 1, X_scaled[:, 0].max() + 1
        y_min, y_max = X_scaled[:, 1].min() - 1, X_scaled[:, 1].max() + 1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 150),
                            np.linspace(y_min, y_max, 150))
        
        Z = ocsvm.decision_function(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        
        ax.contour(xx, yy, Z, levels=[0], linewidths=2, colors='black')
        ax.contourf(xx, yy, Z, levels=np.linspace(Z.min(), 0, 7), cmap='Blues', alpha=0.3)
        
        mask_normal = predictions == 1
        mask_anomaly = predictions == -1
        
        ax.scatter(X_scaled[mask_normal, 0], X_scaled[mask_normal, 1], 
                  c='blue', alpha=0.7, s=30, edgecolors='w', linewidth=0.3)
        ax.scatter(X_scaled[mask_anomaly, 0], X_scaled[mask_anomaly, 1], 
                  c='red', marker='x', s=60, linewidth=2)
        
        n_anomalies = np.sum(predictions == -1)
        ax.set_title(f'nu={nu}\nАномалий: {n_anomalies}')
        ax.set_xlabel('Признак 1')
        ax.set_ylabel('Признак 2')
    
    plt.tight_layout()
    return fig_to_base64(fig)

# ============================================================================
# ELLIPTIC ENVELOPE ILLUSTRATIONS
# ============================================================================

def generate_elliptic_envelope_basic():
    """Generate basic Elliptic Envelope visualization."""
    X, y_true = generate_sample_data_with_outliers()
    X_scaled = StandardScaler().fit_transform(X)
    
    # Train Elliptic Envelope
    ee = EllipticEnvelope(contamination=0.1, random_state=42)
    ee.fit(X_scaled)
    predictions = ee.predict(X_scaled)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Create mesh for decision boundary
    x_min, x_max = X_scaled[:, 0].min() - 1, X_scaled[:, 0].max() + 1
    y_min, y_max = X_scaled[:, 1].min() - 1, X_scaled[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))
    
    # Decision function
    Z = ee.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Plot ellipse boundary
    ax.contour(xx, yy, Z, levels=[0], linewidths=2, colors='darkgreen', linestyles='solid')
    ax.contourf(xx, yy, Z, levels=np.linspace(Z.min(), 0, 7), cmap='Greens', alpha=0.3)
    
    # Plot points
    mask_normal = predictions == 1
    mask_anomaly = predictions == -1
    
    ax.scatter(X_scaled[mask_normal, 0], X_scaled[mask_normal, 1], 
              c='blue', alpha=0.7, s=50, edgecolors='w', linewidth=0.5,
              label='Нормальные')
    ax.scatter(X_scaled[mask_anomaly, 0], X_scaled[mask_anomaly, 1], 
              c='red', marker='x', s=100, linewidth=2,
              label='Аномалии')
    
    ax.set_xlabel('Признак 1')
    ax.set_ylabel('Признак 2')
    ax.set_title('Elliptic Envelope: Обнаружение аномалий (Gaussian assumption)')
    ax.legend()
    
    return fig_to_base64(fig)

def generate_elliptic_envelope_contamination():
    """Generate visualization showing effect of contamination parameter."""
    X, y_true = generate_sample_data_with_outliers()
    X_scaled = StandardScaler().fit_transform(X)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    contaminations = [0.05, 0.1, 0.15, 0.2]
    
    for idx, cont in enumerate(contaminations):
        ax = axes[idx // 2, idx % 2]
        
        ee = EllipticEnvelope(contamination=cont, random_state=42)
        ee.fit(X_scaled)
        predictions = ee.predict(X_scaled)
        
        # Create mesh
        x_min, x_max = X_scaled[:, 0].min() - 1, X_scaled[:, 0].max() + 1
        y_min, y_max = X_scaled[:, 1].min() - 1, X_scaled[:, 1].max() + 1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 150),
                            np.linspace(y_min, y_max, 150))
        
        Z = ee.decision_function(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        
        ax.contour(xx, yy, Z, levels=[0], linewidths=2, colors='darkgreen')
        ax.contourf(xx, yy, Z, levels=np.linspace(Z.min(), 0, 7), cmap='Greens', alpha=0.3)
        
        mask_normal = predictions == 1
        mask_anomaly = predictions == -1
        
        ax.scatter(X_scaled[mask_normal, 0], X_scaled[mask_normal, 1], 
                  c='blue', alpha=0.7, s=30, edgecolors='w', linewidth=0.3)
        ax.scatter(X_scaled[mask_anomaly, 0], X_scaled[mask_anomaly, 1], 
                  c='red', marker='x', s=60, linewidth=2)
        
        n_anomalies = np.sum(predictions == -1)
        ax.set_title(f'contamination={cont}\nАномалий: {n_anomalies}')
        ax.set_xlabel('Признак 1')
        ax.set_ylabel('Признак 2')
    
    plt.tight_layout()
    return fig_to_base64(fig)

def generate_elliptic_envelope_mahalanobis():
    """Generate visualization showing Mahalanobis distance."""
    X, y_true = generate_sample_data_with_outliers()
    X_scaled = StandardScaler().fit_transform(X)
    
    ee = EllipticEnvelope(contamination=0.1, random_state=42)
    ee.fit(X_scaled)
    predictions = ee.predict(X_scaled)
    
    # Calculate Mahalanobis distances
    mahal_dist = ee.mahalanobis(X_scaled)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left: scatter plot with color by Mahalanobis distance
    scatter = ax1.scatter(X_scaled[:, 0], X_scaled[:, 1], 
                         c=mahal_dist, cmap='YlOrRd', 
                         s=50, edgecolors='w', linewidth=0.5)
    plt.colorbar(scatter, ax=ax1, label='Mahalanobis Distance')
    ax1.set_xlabel('Признак 1')
    ax1.set_ylabel('Признак 2')
    ax1.set_title('Расстояние Махаланобиса по точкам')
    ax1.grid(True, alpha=0.3)
    
    # Right: histogram of Mahalanobis distances
    ax2.hist(mahal_dist[predictions == 1], bins=30, alpha=0.6, 
            color='blue', label='Нормальные', edgecolor='black')
    ax2.hist(mahal_dist[predictions == -1], bins=30, alpha=0.6, 
            color='red', label='Аномалии', edgecolor='black')
    ax2.set_xlabel('Mahalanobis Distance (чем больше, тем аномальнее)')
    ax2.set_ylabel('Количество')
    ax2.set_title('Распределение Mahalanobis Distances')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig_to_base64(fig)

# ============================================================================
# MAIN GENERATION FUNCTION
# ============================================================================

def generate_all_illustrations():
    """Generate all anomaly detection illustrations and return dictionary."""
    print("Generating anomaly detection illustrations...")
    
    illustrations = {}
    
    # Isolation Forest
    print("  Isolation Forest...")
    illustrations['isolation_forest_basic'] = generate_isolation_forest_basic()
    illustrations['isolation_forest_scores'] = generate_isolation_forest_scores()
    illustrations['isolation_forest_contamination'] = generate_isolation_forest_contamination()
    
    # Local Outlier Factor
    print("  Local Outlier Factor...")
    illustrations['lof_basic'] = generate_lof_basic()
    illustrations['lof_scores'] = generate_lof_scores()
    illustrations['lof_neighbors'] = generate_lof_neighbors()
    
    # One-Class SVM
    print("  One-Class SVM...")
    illustrations['one_class_svm_basic'] = generate_one_class_svm_basic()
    illustrations['one_class_svm_kernels'] = generate_one_class_svm_kernels()
    illustrations['one_class_svm_nu'] = generate_one_class_svm_nu()
    
    # Elliptic Envelope
    print("  Elliptic Envelope...")
    illustrations['elliptic_envelope_basic'] = generate_elliptic_envelope_basic()
    illustrations['elliptic_envelope_contamination'] = generate_elliptic_envelope_contamination()
    illustrations['elliptic_envelope_mahalanobis'] = generate_elliptic_envelope_mahalanobis()
    
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
