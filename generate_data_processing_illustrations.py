#!/usr/bin/env python3
"""
Generate matplotlib illustrations for data processing cheatsheets.
This script creates high-quality visualizations and encodes them as base64 for inline HTML embedding.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.datasets import make_classification, make_blobs
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
import pandas as pd
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
# PREPROCESSING ILLUSTRATIONS
# ============================================================================

def generate_missing_data_strategies():
    """Visualize different missing data imputation strategies."""
    np.random.seed(42)
    
    # Generate data with missing values
    X = np.random.randn(100)
    # Introduce missing values
    missing_indices = np.random.choice(100, 20, replace=False)
    X_missing = X.copy()
    X_missing[missing_indices] = np.nan
    
    # Different imputation strategies
    mean_val = np.nanmean(X_missing)
    median_val = np.nanmedian(X_missing)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Original data
    ax = axes[0, 0]
    ax.hist(X, bins=20, alpha=0.7, color='blue', edgecolor='black')
    ax.set_title('Исходные данные (без пропусков)')
    ax.set_xlabel('Значение')
    ax.set_ylabel('Частота')
    
    # Data with missing values
    ax = axes[0, 1]
    ax.hist(X_missing[~np.isnan(X_missing)], bins=20, alpha=0.7, color='orange', edgecolor='black')
    ax.set_title(f'Данные с пропусками ({len(missing_indices)} пропусков)')
    ax.set_xlabel('Значение')
    ax.set_ylabel('Частота')
    
    # Mean imputation
    ax = axes[1, 0]
    X_mean = X_missing.copy()
    X_mean[np.isnan(X_mean)] = mean_val
    ax.hist(X_mean, bins=20, alpha=0.7, color='green', edgecolor='black')
    ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Среднее={mean_val:.2f}')
    ax.set_title('Заполнение средним значением')
    ax.set_xlabel('Значение')
    ax.set_ylabel('Частота')
    ax.legend()
    
    # Median imputation
    ax = axes[1, 1]
    X_median = X_missing.copy()
    X_median[np.isnan(X_median)] = median_val
    ax.hist(X_median, bins=20, alpha=0.7, color='purple', edgecolor='black')
    ax.axvline(median_val, color='red', linestyle='--', linewidth=2, label=f'Медиана={median_val:.2f}')
    ax.set_title('Заполнение медианой')
    ax.set_xlabel('Значение')
    ax.set_ylabel('Частота')
    ax.legend()
    
    plt.tight_layout()
    return fig_to_base64(fig)

def generate_outlier_detection():
    """Visualize outlier detection using IQR and Z-score methods."""
    np.random.seed(42)
    
    # Generate data with outliers
    X = np.random.randn(100)
    X = np.append(X, [5, 5.5, -4.5, -5])  # Add outliers
    
    # IQR method
    Q1 = np.percentile(X, 25)
    Q3 = np.percentile(X, 75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Z-score method
    mean = np.mean(X)
    std = np.std(X)
    z_scores = np.abs((X - mean) / std)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # IQR method
    ax = axes[0]
    colors = ['red' if (x < lower_bound or x > upper_bound) else 'blue' for x in X]
    ax.scatter(range(len(X)), X, c=colors, alpha=0.6, s=50)
    ax.axhline(upper_bound, color='green', linestyle='--', linewidth=2, label=f'Верхняя граница={upper_bound:.2f}')
    ax.axhline(lower_bound, color='green', linestyle='--', linewidth=2, label=f'Нижняя граница={lower_bound:.2f}')
    ax.axhline(Q1, color='orange', linestyle=':', linewidth=1.5, alpha=0.7, label=f'Q1={Q1:.2f}')
    ax.axhline(Q3, color='orange', linestyle=':', linewidth=1.5, alpha=0.7, label=f'Q3={Q3:.2f}')
    ax.set_title('IQR метод (1.5 * IQR)')
    ax.set_xlabel('Индекс')
    ax.set_ylabel('Значение')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Z-score method
    ax = axes[1]
    colors = ['red' if z > 3 else 'blue' for z in z_scores]
    ax.scatter(range(len(X)), X, c=colors, alpha=0.6, s=50)
    ax.axhline(mean + 3*std, color='green', linestyle='--', linewidth=2, label='μ + 3σ')
    ax.axhline(mean - 3*std, color='green', linestyle='--', linewidth=2, label='μ - 3σ')
    ax.axhline(mean, color='orange', linestyle=':', linewidth=1.5, alpha=0.7, label=f'μ={mean:.2f}')
    ax.set_title('Z-score метод (|z| > 3)')
    ax.set_xlabel('Индекс')
    ax.set_ylabel('Значение')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig_to_base64(fig)

def generate_encoding_comparison():
    """Visualize different encoding strategies for categorical variables."""
    categories = ['A', 'B', 'C']
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Label Encoding
    ax = axes[0]
    label_encoded = [0, 1, 2]
    bars = ax.bar(categories, label_encoded, color=['red', 'green', 'blue'], alpha=0.7, edgecolor='black')
    ax.set_title('Label Encoding')
    ax.set_ylabel('Закодированное значение')
    ax.set_ylim([0, 3])
    for i, (cat, val) in enumerate(zip(categories, label_encoded)):
        ax.text(i, val + 0.1, str(val), ha='center', fontsize=12, fontweight='bold')
    
    # One-Hot Encoding
    ax = axes[1]
    one_hot = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    im = ax.imshow(one_hot, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    ax.set_xticks(range(3))
    ax.set_xticklabels(['A_enc', 'B_enc', 'C_enc'])
    ax.set_yticks(range(3))
    ax.set_yticklabels(['A', 'B', 'C'])
    ax.set_title('One-Hot Encoding')
    for i in range(3):
        for j in range(3):
            ax.text(j, i, str(one_hot[i, j]), ha='center', va='center', 
                   color='black', fontweight='bold', fontsize=12)
    
    # Ordinal Encoding (with order)
    ax = axes[2]
    ordinal_encoded = [0, 1, 2]
    bars = ax.bar(categories, ordinal_encoded, color=['lightcoral', 'gold', 'lightgreen'], 
                  alpha=0.7, edgecolor='black')
    ax.arrow(0.5, 0.5, 1.8, 0, head_width=0.15, head_length=0.1, fc='black', ec='black')
    ax.set_title('Ordinal Encoding (с порядком)')
    ax.set_ylabel('Закодированное значение')
    ax.set_ylim([0, 3])
    ax.text(1.5, 2.5, 'Возрастающий порядок', ha='center', fontsize=10)
    for i, (cat, val) in enumerate(zip(categories, ordinal_encoded)):
        ax.text(i, val + 0.1, str(val), ha='center', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    return fig_to_base64(fig)

# ============================================================================
# SCALING & NORMALIZATION ILLUSTRATIONS
# ============================================================================

def generate_scaler_comparison():
    """Compare different scaling methods."""
    np.random.seed(42)
    
    # Generate data with different scales
    X = np.random.randn(100, 1) * 10 + 50
    # Add outliers
    X = np.vstack([X, [[100]], [[5]]])
    
    # Apply different scalers
    standard_scaler = StandardScaler()
    minmax_scaler = MinMaxScaler()
    robust_scaler = RobustScaler()
    
    X_standard = standard_scaler.fit_transform(X)
    X_minmax = minmax_scaler.fit_transform(X)
    X_robust = robust_scaler.fit_transform(X)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Original data
    ax = axes[0, 0]
    ax.hist(X, bins=20, alpha=0.7, color='blue', edgecolor='black')
    ax.set_title(f'Исходные данные\n(μ={X.mean():.2f}, σ={X.std():.2f})')
    ax.set_xlabel('Значение')
    ax.set_ylabel('Частота')
    ax.axvline(X.mean(), color='red', linestyle='--', linewidth=2, label='Среднее')
    ax.legend()
    
    # StandardScaler
    ax = axes[0, 1]
    ax.hist(X_standard, bins=20, alpha=0.7, color='green', edgecolor='black')
    ax.set_title(f'StandardScaler\n(μ={X_standard.mean():.2f}, σ={X_standard.std():.2f})')
    ax.set_xlabel('Значение')
    ax.set_ylabel('Частота')
    ax.axvline(X_standard.mean(), color='red', linestyle='--', linewidth=2, label='Среднее')
    ax.legend()
    
    # MinMaxScaler
    ax = axes[1, 0]
    ax.hist(X_minmax, bins=20, alpha=0.7, color='orange', edgecolor='black')
    ax.set_title(f'MinMaxScaler\n(min={X_minmax.min():.2f}, max={X_minmax.max():.2f})')
    ax.set_xlabel('Значение')
    ax.set_ylabel('Частота')
    ax.axvline(0, color='blue', linestyle='--', linewidth=2, label='0')
    ax.axvline(1, color='red', linestyle='--', linewidth=2, label='1')
    ax.legend()
    
    # RobustScaler
    ax = axes[1, 1]
    ax.hist(X_robust, bins=20, alpha=0.7, color='purple', edgecolor='black')
    ax.set_title(f'RobustScaler\n(медиана={np.median(X_robust):.2f})')
    ax.set_xlabel('Значение')
    ax.set_ylabel('Частота')
    ax.axvline(np.median(X_robust), color='red', linestyle='--', linewidth=2, label='Медиана')
    ax.legend()
    
    plt.tight_layout()
    return fig_to_base64(fig)

def generate_normalization_effect():
    """Visualize the effect of normalization on data distribution."""
    np.random.seed(42)
    
    # Create two features with different scales
    feature1 = np.random.randn(100) * 100 + 500
    feature2 = np.random.randn(100) * 10 + 50
    
    X = np.column_stack([feature1, feature2])
    X_normalized = StandardScaler().fit_transform(X)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Before normalization
    ax = axes[0]
    ax.scatter(X[:, 0], X[:, 1], alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
    ax.set_xlabel('Признак 1 (масштаб: 100)')
    ax.set_ylabel('Признак 2 (масштаб: 10)')
    ax.set_title('До нормализации\n(разные масштабы)')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('auto')
    
    # After normalization
    ax = axes[1]
    ax.scatter(X_normalized[:, 0], X_normalized[:, 1], alpha=0.6, s=50, 
              edgecolors='black', linewidth=0.5, color='green')
    ax.set_xlabel('Признак 1 (нормализован)')
    ax.set_ylabel('Признак 2 (нормализован)')
    ax.set_title('После нормализации\n(одинаковые масштабы)')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    ax.axhline(0, color='red', linestyle='--', linewidth=1, alpha=0.5)
    ax.axvline(0, color='red', linestyle='--', linewidth=1, alpha=0.5)
    
    plt.tight_layout()
    return fig_to_base64(fig)

# ============================================================================
# FEATURE ENGINEERING ILLUSTRATIONS
# ============================================================================

def generate_polynomial_features():
    """Visualize polynomial feature transformation."""
    np.random.seed(42)
    X = np.linspace(-3, 3, 100)
    y = 0.5 * X**2 + X + np.random.randn(100) * 0.5
    
    # Fit models
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import PolynomialFeatures
    
    # Linear
    model_linear = LinearRegression()
    model_linear.fit(X.reshape(-1, 1), y)
    y_pred_linear = model_linear.predict(X.reshape(-1, 1))
    
    # Polynomial degree 2
    poly = PolynomialFeatures(degree=2)
    X_poly = poly.fit_transform(X.reshape(-1, 1))
    model_poly = LinearRegression()
    model_poly.fit(X_poly, y)
    y_pred_poly = model_poly.predict(X_poly)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Linear fit
    ax = axes[0]
    ax.scatter(X, y, alpha=0.5, s=30, label='Данные')
    ax.plot(X, y_pred_linear, 'r-', linewidth=2, label='Линейная регрессия')
    ax.set_xlabel('X')
    ax.set_ylabel('y')
    ax.set_title('Линейные признаки\n(недообучение)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Polynomial fit
    ax = axes[1]
    ax.scatter(X, y, alpha=0.5, s=30, label='Данные')
    ax.plot(X, y_pred_poly, 'g-', linewidth=2, label='Полиномиальная регрессия (степень 2)')
    ax.set_xlabel('X')
    ax.set_ylabel('y')
    ax.set_title('Полиномиальные признаки\n(хорошее обучение)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig_to_base64(fig)

def generate_feature_interaction():
    """Visualize feature interaction effects."""
    np.random.seed(42)
    
    # Create synthetic data where interaction matters
    n_samples = 200
    X1 = np.random.uniform(0, 10, n_samples)
    X2 = np.random.uniform(0, 10, n_samples)
    # Target depends on interaction
    y = X1 * X2 + np.random.randn(n_samples) * 5
    
    fig = plt.figure(figsize=(12, 5))
    
    # 3D scatter plot
    ax1 = fig.add_subplot(121, projection='3d')
    scatter = ax1.scatter(X1, X2, y, c=y, cmap='viridis', alpha=0.6, s=30)
    ax1.set_xlabel('Признак X1')
    ax1.set_ylabel('Признак X2')
    ax1.set_zlabel('Целевая переменная y')
    ax1.set_title('Взаимодействие признаков\ny = X1 × X2')
    plt.colorbar(scatter, ax=ax1, label='y')
    
    # Heatmap
    ax2 = fig.add_subplot(122)
    # Create grid for heatmap
    x1_grid = np.linspace(0, 10, 20)
    x2_grid = np.linspace(0, 10, 20)
    X1_grid, X2_grid = np.meshgrid(x1_grid, x2_grid)
    Y_grid = X1_grid * X2_grid
    
    im = ax2.contourf(X1_grid, X2_grid, Y_grid, levels=20, cmap='viridis', alpha=0.8)
    ax2.set_xlabel('Признак X1')
    ax2.set_ylabel('Признак X2')
    ax2.set_title('Карта взаимодействия\n(X1 × X2)')
    plt.colorbar(im, ax=ax2, label='y = X1 × X2')
    
    plt.tight_layout()
    return fig_to_base64(fig)

# ============================================================================
# FEATURE SELECTION ILLUSTRATIONS
# ============================================================================

def generate_feature_importance():
    """Visualize feature importance from Random Forest."""
    np.random.seed(42)
    
    # Generate classification dataset
    X, y = make_classification(n_samples=300, n_features=10, n_informative=5, 
                               n_redundant=3, n_clusters_per_class=2, random_state=42)
    
    # Train Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X, y)
    
    # Get feature importances
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    feature_names = [f'Feature {i+1}' for i in range(10)]
    colors = ['green' if importances[i] > 0.1 else 'orange' for i in indices]
    
    bars = ax.barh(range(10), importances[indices], color=colors, alpha=0.7, edgecolor='black')
    ax.set_yticks(range(10))
    ax.set_yticklabels([feature_names[i] for i in indices])
    ax.set_xlabel('Важность признака')
    ax.set_title('Важность признаков (Random Forest)')
    ax.axvline(0.1, color='red', linestyle='--', linewidth=2, label='Порог важности (0.1)')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='x')
    
    # Add values on bars
    for i, (idx, imp) in enumerate(zip(indices, importances[indices])):
        ax.text(imp + 0.005, i, f'{imp:.3f}', va='center', fontsize=9)
    
    plt.tight_layout()
    return fig_to_base64(fig)

def generate_correlation_matrix():
    """Generate correlation matrix for feature selection."""
    np.random.seed(42)
    
    # Create correlated features
    n_samples = 200
    X1 = np.random.randn(n_samples)
    X2 = X1 + np.random.randn(n_samples) * 0.3  # Highly correlated with X1
    X3 = np.random.randn(n_samples)  # Independent
    X4 = X3 + np.random.randn(n_samples) * 0.5  # Moderately correlated with X3
    X5 = np.random.randn(n_samples)  # Independent
    
    X = np.column_stack([X1, X2, X3, X4, X5])
    df = pd.DataFrame(X, columns=[f'Feature {i+1}' for i in range(5)])
    
    # Calculate correlation matrix
    corr_matrix = df.corr()
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    im = ax.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1, aspect='auto')
    
    # Set ticks
    ax.set_xticks(range(5))
    ax.set_yticks(range(5))
    ax.set_xticklabels(df.columns, rotation=45, ha='right')
    ax.set_yticklabels(df.columns)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Корреляция', rotation=270, labelpad=20)
    
    # Add correlation values
    for i in range(5):
        for j in range(5):
            text = ax.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                          ha='center', va='center', color='black', fontsize=10, fontweight='bold')
    
    ax.set_title('Матрица корреляции признаков')
    plt.tight_layout()
    return fig_to_base64(fig)

def generate_univariate_selection():
    """Visualize univariate feature selection."""
    np.random.seed(42)
    
    # Generate data
    X, y = make_classification(n_samples=300, n_features=10, n_informative=5, 
                               n_redundant=3, random_state=42)
    
    # Univariate feature selection
    selector = SelectKBest(f_classif, k=5)
    selector.fit(X, y)
    scores = selector.scores_
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    feature_names = [f'Feature {i+1}' for i in range(10)]
    colors = ['green' if selector.get_support()[i] else 'red' for i in range(10)]
    
    bars = ax.bar(range(10), scores, color=colors, alpha=0.7, edgecolor='black')
    ax.set_xticks(range(10))
    ax.set_xticklabels(feature_names, rotation=45, ha='right')
    ax.set_ylabel('F-статистика')
    ax.set_title('Univariate Feature Selection (ANOVA F-test)')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='green', alpha=0.7, label='Выбрано (Top 5)'),
                      Patch(facecolor='red', alpha=0.7, label='Отброшено')]
    ax.legend(handles=legend_elements)
    
    # Add values on bars
    for i, score in enumerate(scores):
        ax.text(i, score + max(scores)*0.02, f'{score:.1f}', ha='center', fontsize=9)
    
    plt.tight_layout()
    return fig_to_base64(fig)

# ============================================================================
# DIMENSIONALITY REDUCTION ILLUSTRATIONS
# ============================================================================

def generate_pca_visualization():
    """Visualize PCA dimensionality reduction."""
    np.random.seed(42)
    
    # Generate 3D data
    n_samples = 200
    X = np.random.randn(n_samples, 3)
    X[:, 1] = X[:, 0] * 0.5 + np.random.randn(n_samples) * 0.3
    X[:, 2] = X[:, 0] * 0.3 + X[:, 1] * 0.3 + np.random.randn(n_samples) * 0.2
    
    # Apply PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    
    fig = plt.figure(figsize=(14, 5))
    
    # Original 3D data
    ax1 = fig.add_subplot(121, projection='3d')
    scatter = ax1.scatter(X[:, 0], X[:, 1], X[:, 2], c=X[:, 0], cmap='viridis', alpha=0.6, s=30)
    ax1.set_xlabel('X1')
    ax1.set_ylabel('X2')
    ax1.set_zlabel('X3')
    ax1.set_title('Исходные данные (3D)')
    
    # PCA reduced data
    ax2 = fig.add_subplot(122)
    scatter = ax2.scatter(X_pca[:, 0], X_pca[:, 1], c=X[:, 0], cmap='viridis', alpha=0.6, s=30)
    ax2.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
    ax2.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
    ax2.set_title('После PCA (2D)')
    ax2.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax2, label='Значение X1')
    
    plt.tight_layout()
    return fig_to_base64(fig)

def generate_pca_variance():
    """Visualize explained variance by principal components."""
    np.random.seed(42)
    
    # Generate high-dimensional data
    X = np.random.randn(200, 20)
    
    # Apply PCA
    pca = PCA()
    pca.fit(X)
    
    explained_variance = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Individual explained variance
    ax = axes[0]
    ax.bar(range(1, 21), explained_variance, alpha=0.7, color='blue', edgecolor='black')
    ax.set_xlabel('Главная компонента')
    ax.set_ylabel('Объясненная дисперсия')
    ax.set_title('Объясненная дисперсия по компонентам')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Cumulative explained variance
    ax = axes[1]
    ax.plot(range(1, 21), cumulative_variance, 'bo-', linewidth=2, markersize=6)
    ax.axhline(0.95, color='red', linestyle='--', linewidth=2, label='95% дисперсии')
    ax.axhline(0.90, color='orange', linestyle='--', linewidth=2, label='90% дисперсии')
    ax.set_xlabel('Количество компонент')
    ax.set_ylabel('Кумулятивная объясненная дисперсия')
    ax.set_title('Кумулятивная объясненная дисперсия')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig_to_base64(fig)

def generate_tsne_visualization():
    """Visualize t-SNE dimensionality reduction."""
    np.random.seed(42)
    
    # Generate clustered high-dimensional data
    X, y = make_blobs(n_samples=300, n_features=10, centers=3, 
                     cluster_std=1.0, random_state=42)
    
    # Apply PCA first (for comparison)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    
    # Apply t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    X_tsne = tsne.fit_transform(X)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # PCA
    ax = axes[0]
    scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', 
                        alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_title('PCA (линейное снижение размерности)')
    ax.grid(True, alpha=0.3)
    
    # t-SNE
    ax = axes[1]
    scatter = ax.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='viridis', 
                        alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    ax.set_title('t-SNE (нелинейное снижение размерности)')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig_to_base64(fig)

# ============================================================================
# IMBALANCED DATA ILLUSTRATIONS
# ============================================================================

def generate_class_imbalance():
    """Visualize class imbalance problem."""
    np.random.seed(42)
    
    # Create imbalanced dataset
    X, y = make_classification(n_samples=1000, n_features=2, n_informative=2,
                               n_redundant=0, n_clusters_per_class=1,
                               weights=[0.9, 0.1], flip_y=0, random_state=42)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Class distribution
    ax = axes[0]
    class_counts = np.bincount(y)
    bars = ax.bar(['Класс 0\n(Majority)', 'Класс 1\n(Minority)'], class_counts, 
                  color=['blue', 'red'], alpha=0.7, edgecolor='black')
    ax.set_ylabel('Количество образцов')
    ax.set_title(f'Дисбаланс классов\nСоотношение: {class_counts[0]/class_counts[1]:.1f}:1')
    
    # Add values on bars
    for bar, count in zip(bars, class_counts):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{count}\n({count/len(y)*100:.1f}%)',
               ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Data distribution
    ax = axes[1]
    for label, color, name in zip([0, 1], ['blue', 'red'], ['Класс 0', 'Класс 1']):
        mask = y == label
        ax.scatter(X[mask, 0], X[mask, 1], c=color, label=name, 
                  alpha=0.6, s=30, edgecolors='black', linewidth=0.5)
    ax.set_xlabel('Признак 1')
    ax.set_ylabel('Признак 2')
    ax.set_title('Распределение классов в пространстве признаков')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig_to_base64(fig)

def generate_smote_visualization():
    """Visualize SMOTE oversampling."""
    np.random.seed(42)
    
    # Create imbalanced dataset
    X, y = make_classification(n_samples=200, n_features=2, n_informative=2,
                               n_redundant=0, n_clusters_per_class=1,
                               weights=[0.8, 0.2], flip_y=0, random_state=42)
    
    # Apply SMOTE
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Before SMOTE
    ax = axes[0]
    for label, color, name in zip([0, 1], ['blue', 'red'], ['Класс 0', 'Класс 1']):
        mask = y == label
        ax.scatter(X[mask, 0], X[mask, 1], c=color, label=f'{name} ({np.sum(mask)})', 
                  alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
    ax.set_xlabel('Признак 1')
    ax.set_ylabel('Признак 2')
    ax.set_title('До SMOTE (дисбаланс)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # After SMOTE
    ax = axes[1]
    for label, color, name in zip([0, 1], ['blue', 'red'], ['Класс 0', 'Класс 1']):
        mask = y_resampled == label
        # Distinguish original from synthetic for minority class
        if label == 1:
            original_mask = np.zeros(len(y_resampled), dtype=bool)
            original_mask[:len(y)] = y == 1
            synthetic_mask = mask & ~original_mask
            ax.scatter(X_resampled[original_mask, 0], X_resampled[original_mask, 1], 
                      c=color, label=f'{name} оригинал ({np.sum(original_mask)})', 
                      alpha=0.8, s=50, edgecolors='black', linewidth=0.5, marker='o')
            ax.scatter(X_resampled[synthetic_mask, 0], X_resampled[synthetic_mask, 1], 
                      c=color, label=f'{name} синтетич. ({np.sum(synthetic_mask)})', 
                      alpha=0.4, s=50, edgecolors='red', linewidth=1, marker='^')
        else:
            ax.scatter(X_resampled[mask, 0], X_resampled[mask, 1], 
                      c=color, label=f'{name} ({np.sum(mask)})', 
                      alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
    ax.set_xlabel('Признак 1')
    ax.set_ylabel('Признак 2')
    ax.set_title('После SMOTE (сбалансировано)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig_to_base64(fig)

def generate_resampling_strategies():
    """Compare different resampling strategies."""
    np.random.seed(42)
    
    # Create imbalanced dataset
    X, y = make_classification(n_samples=200, n_features=2, n_informative=2,
                               n_redundant=0, n_clusters_per_class=1,
                               weights=[0.8, 0.2], flip_y=0, random_state=42)
    
    class_counts_original = np.bincount(y)
    
    # Apply SMOTE
    smote = SMOTE(random_state=42)
    X_smote, y_smote = smote.fit_resample(X, y)
    class_counts_smote = np.bincount(y_smote)
    
    # Undersample majority class
    from imblearn.under_sampling import RandomUnderSampler
    rus = RandomUnderSampler(random_state=42)
    X_under, y_under = rus.fit_resample(X, y)
    class_counts_under = np.bincount(y_under)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(2)
    width = 0.25
    
    bars1 = ax.bar(x - width, class_counts_original, width, label='Исходные', 
                   color='blue', alpha=0.7, edgecolor='black')
    bars2 = ax.bar(x, class_counts_smote, width, label='SMOTE (oversampling)', 
                   color='green', alpha=0.7, edgecolor='black')
    bars3 = ax.bar(x + width, class_counts_under, width, label='Undersampling', 
                   color='orange', alpha=0.7, edgecolor='black')
    
    ax.set_ylabel('Количество образцов')
    ax.set_title('Сравнение стратегий ресемплинга')
    ax.set_xticks(x)
    ax.set_xticklabels(['Класс 0', 'Класс 1'])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add values on bars
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height)}',
                   ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    return fig_to_base64(fig)

# ============================================================================
# DATA AUGMENTATION ILLUSTRATIONS
# ============================================================================

def generate_augmentation_examples():
    """Visualize different data augmentation techniques."""
    np.random.seed(42)
    
    # Create a simple 2D image-like data
    X = np.zeros((28, 28))
    # Draw a simple shape
    X[10:18, 10:18] = 1.0
    X[12:16, 12:16] = 0.5
    
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    
    # Original
    ax = axes[0, 0]
    ax.imshow(X, cmap='gray')
    ax.set_title('Исходное изображение')
    ax.axis('off')
    
    # Rotation
    ax = axes[0, 1]
    from scipy.ndimage import rotate
    X_rotated = rotate(X, 15, reshape=False, mode='constant', cval=0)
    ax.imshow(X_rotated, cmap='gray')
    ax.set_title('Поворот (15°)')
    ax.axis('off')
    
    # Flip
    ax = axes[0, 2]
    X_flipped = np.fliplr(X)
    ax.imshow(X_flipped, cmap='gray')
    ax.set_title('Отражение')
    ax.axis('off')
    
    # Zoom
    ax = axes[1, 0]
    from scipy.ndimage import zoom
    X_zoomed = zoom(X, 1.2, order=1)
    # Crop to original size
    start = (X_zoomed.shape[0] - 28) // 2
    X_zoomed = X_zoomed[start:start+28, start:start+28]
    ax.imshow(X_zoomed, cmap='gray')
    ax.set_title('Увеличение')
    ax.axis('off')
    
    # Noise
    ax = axes[1, 1]
    X_noisy = X + np.random.randn(28, 28) * 0.1
    X_noisy = np.clip(X_noisy, 0, 1)
    ax.imshow(X_noisy, cmap='gray')
    ax.set_title('Добавление шума')
    ax.axis('off')
    
    # Brightness
    ax = axes[1, 2]
    X_bright = X * 1.3
    X_bright = np.clip(X_bright, 0, 1)
    ax.imshow(X_bright, cmap='gray')
    ax.set_title('Изменение яркости')
    ax.axis('off')
    
    plt.tight_layout()
    return fig_to_base64(fig)

def generate_augmentation_distribution():
    """Show how augmentation increases dataset size."""
    np.random.seed(42)
    
    # Simulate original and augmented dataset sizes
    categories = ['Класс A', 'Класс B', 'Класс C']
    original_sizes = [100, 150, 80]
    augmented_sizes = [500, 750, 400]  # 5x augmentation
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Bar chart comparison
    ax = axes[0]
    x = np.arange(len(categories))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, original_sizes, width, label='Исходный датасет', 
                   color='blue', alpha=0.7, edgecolor='black')
    bars2 = ax.bar(x + width/2, augmented_sizes, width, label='После аугментации', 
                   color='green', alpha=0.7, edgecolor='black')
    
    ax.set_ylabel('Количество образцов')
    ax.set_title('Увеличение размера датасета')
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add values on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height)}',
                   ha='center', va='bottom', fontsize=10)
    
    # Pie chart showing augmentation factor
    ax = axes[1]
    total_original = sum(original_sizes)
    total_augmented = sum(augmented_sizes)
    sizes = [total_original, total_augmented - total_original]
    labels = ['Исходные данные', 'Аугментированные данные']
    colors = ['blue', 'green']
    explode = (0, 0.1)
    
    wedges, texts, autotexts = ax.pie(sizes, explode=explode, labels=labels, colors=colors,
                                       autopct='%1.1f%%', startangle=90, 
                                       textprops={'fontsize': 11})
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
    
    ax.set_title(f'Соотношение данных\n(Увеличение в {total_augmented/total_original:.1f}x)')
    
    plt.tight_layout()
    return fig_to_base64(fig)

# ============================================================================
# MAIN GENERATION FUNCTION
# ============================================================================

def generate_all_illustrations():
    """Generate all data processing illustrations and return dictionary."""
    print("Generating data processing illustrations...")
    
    illustrations = {}
    
    # Preprocessing
    print("  Preprocessing...")
    illustrations['missing_data_strategies'] = generate_missing_data_strategies()
    illustrations['outlier_detection'] = generate_outlier_detection()
    illustrations['encoding_comparison'] = generate_encoding_comparison()
    
    # Scaling & Normalization
    print("  Scaling & Normalization...")
    illustrations['scaler_comparison'] = generate_scaler_comparison()
    illustrations['normalization_effect'] = generate_normalization_effect()
    
    # Feature Engineering
    print("  Feature Engineering...")
    illustrations['polynomial_features'] = generate_polynomial_features()
    illustrations['feature_interaction'] = generate_feature_interaction()
    
    # Feature Selection
    print("  Feature Selection...")
    illustrations['feature_importance'] = generate_feature_importance()
    illustrations['correlation_matrix'] = generate_correlation_matrix()
    illustrations['univariate_selection'] = generate_univariate_selection()
    
    # Dimensionality Reduction
    print("  Dimensionality Reduction...")
    illustrations['pca_visualization'] = generate_pca_visualization()
    illustrations['pca_variance'] = generate_pca_variance()
    illustrations['tsne_visualization'] = generate_tsne_visualization()
    
    # Imbalanced Data
    print("  Imbalanced Data...")
    illustrations['class_imbalance'] = generate_class_imbalance()
    illustrations['smote_visualization'] = generate_smote_visualization()
    illustrations['resampling_strategies'] = generate_resampling_strategies()
    
    # Data Augmentation
    print("  Data Augmentation...")
    illustrations['augmentation_examples'] = generate_augmentation_examples()
    illustrations['augmentation_distribution'] = generate_augmentation_distribution()
    
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
