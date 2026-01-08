#!/usr/bin/env python3
"""
Generate matplotlib illustrations for classification cheatsheets.
This script creates high-quality visualizations and encodes them as base64 for inline HTML embedding.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.datasets import make_classification, make_moons, make_circles
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
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

def plot_decision_boundary(X, y, clf, title, ax=None):
    """Plot decision boundary for a classifier."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    
    h = 0.02  # step size in the mesh
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    ax.contourf(xx, yy, Z, alpha=0.3, cmap='RdYlBu')
    scatter = ax.scatter(X[:, 0], X[:, 1], c=y, cmap='RdYlBu', 
                        edgecolors='black', linewidth=0.5, s=50, alpha=0.8)
    ax.set_xlabel('Признак 1')
    ax.set_ylabel('Признак 2')
    ax.set_title(title)
    
    return ax

# ============================================================================
# DECISION TREES ILLUSTRATIONS
# ============================================================================

def generate_decision_tree_visualization():
    """Generate decision tree structure visualization."""
    X, y = make_classification(n_samples=100, n_features=2, n_informative=2,
                               n_redundant=0, n_clusters_per_class=1, 
                               random_state=42)
    
    clf = DecisionTreeClassifier(max_depth=3, random_state=42)
    clf.fit(X, y)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    plot_tree(clf, filled=True, feature_names=['X1', 'X2'], 
             class_names=['Класс 0', 'Класс 1'], 
             rounded=True, fontsize=9, ax=ax)
    ax.set_title('Структура дерева решений (max_depth=3)')
    
    return fig_to_base64(fig)

def generate_decision_tree_depth_comparison():
    """Generate comparison of trees with different depths."""
    X, y = make_classification(n_samples=200, n_features=2, n_informative=2,
                               n_redundant=0, n_clusters_per_class=1, 
                               random_state=42)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    depths = [2, 5, 20]
    
    for idx, depth in enumerate(depths):
        clf = DecisionTreeClassifier(max_depth=depth, random_state=42)
        clf.fit(X, y)
        train_score = clf.score(X, y)
        
        plot_decision_boundary(X, y, clf, 
                             f'Глубина = {depth}\nТочность: {train_score:.2f}',
                             ax=axes[idx])
    
    plt.tight_layout()
    return fig_to_base64(fig)

def generate_decision_tree_feature_importance():
    """Generate feature importance visualization."""
    np.random.seed(42)
    n_features = 8
    X = np.random.randn(200, n_features)
    # Make features 0, 2, 4 more important
    y = (X[:, 0] + 2 * X[:, 2] + 1.5 * X[:, 4] + 0.5 * np.random.randn(200)) > 0
    y = y.astype(int)
    
    clf = DecisionTreeClassifier(max_depth=5, random_state=42)
    clf.fit(X, y)
    
    feature_names = [f'Признак {i+1}' for i in range(n_features)]
    importances = clf.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(range(n_features), importances[indices], color='steelblue', alpha=0.8)
    ax.set_xticks(range(n_features))
    ax.set_xticklabels([feature_names[i] for i in indices], rotation=45)
    ax.set_ylabel('Важность')
    ax.set_xlabel('Признак')
    ax.set_title('Важность признаков в дереве решений')
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    return fig_to_base64(fig)

# ============================================================================
# RANDOM FOREST ILLUSTRATIONS
# ============================================================================

def generate_random_forest_ensemble():
    """Generate visualization showing ensemble nature of Random Forest."""
    X, y = make_classification(n_samples=150, n_features=2, n_informative=2,
                               n_redundant=0, n_clusters_per_class=1, 
                               random_state=42)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Single decision tree
    tree = DecisionTreeClassifier(max_depth=5, random_state=42)
    tree.fit(X, y)
    plot_decision_boundary(X, y, tree, 'Одно дерево решений', ax=axes[0, 0])
    
    # Individual trees from random forest
    rf = RandomForestClassifier(n_estimators=5, max_depth=5, random_state=42)
    rf.fit(X, y)
    
    for i in range(min(4, len(rf.estimators_))):
        row = (i + 1) // 3
        col = (i + 1) % 3
        plot_decision_boundary(X, y, rf.estimators_[i], 
                             f'Дерево {i+1} из ансамбля', 
                             ax=axes[row, col])
    
    # Combined Random Forest
    plot_decision_boundary(X, y, rf, 
                         'Random Forest (усреднение всех деревьев)', 
                         ax=axes[1, 2])
    
    plt.tight_layout()
    return fig_to_base64(fig)

def generate_random_forest_oob_error():
    """Generate OOB error plot for different number of trees."""
    X, y = make_classification(n_samples=300, n_features=10, n_informative=8,
                               n_redundant=0, random_state=42)
    
    n_estimators_range = range(1, 101, 5)
    oob_errors = []
    
    for n_est in n_estimators_range:
        rf = RandomForestClassifier(n_estimators=n_est, oob_score=True, 
                                    random_state=42, max_features='sqrt')
        rf.fit(X, y)
        oob_errors.append(1 - rf.oob_score_)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(n_estimators_range, oob_errors, 'b-', linewidth=2, marker='o', markersize=4)
    ax.set_xlabel('Количество деревьев')
    ax.set_ylabel('OOB ошибка')
    ax.set_title('Out-of-Bag ошибка vs количество деревьев')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    return fig_to_base64(fig)

def generate_random_forest_feature_importance():
    """Generate feature importance comparison between single tree and RF."""
    np.random.seed(42)
    n_features = 8
    X = np.random.randn(300, n_features)
    y = (X[:, 0] + 2 * X[:, 2] + 1.5 * X[:, 4] + 0.3 * np.random.randn(300)) > 0
    y = y.astype(int)
    
    # Single tree
    tree = DecisionTreeClassifier(max_depth=5, random_state=42)
    tree.fit(X, y)
    
    # Random Forest
    rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    rf.fit(X, y)
    
    feature_names = [f'Признак {i+1}' for i in range(n_features)]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Tree importance
    importances_tree = tree.feature_importances_
    axes[0].bar(range(n_features), importances_tree, color='steelblue', alpha=0.8)
    axes[0].set_xticks(range(n_features))
    axes[0].set_xticklabels(feature_names, rotation=45)
    axes[0].set_ylabel('Важность')
    axes[0].set_title('Важность признаков: Одно дерево')
    axes[0].grid(axis='y', alpha=0.3)
    
    # RF importance
    importances_rf = rf.feature_importances_
    axes[1].bar(range(n_features), importances_rf, color='forestgreen', alpha=0.8)
    axes[1].set_xticks(range(n_features))
    axes[1].set_xticklabels(feature_names, rotation=45)
    axes[1].set_ylabel('Важность')
    axes[1].set_title('Важность признаков: Random Forest')
    axes[1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    return fig_to_base64(fig)

# ============================================================================
# SVM ILLUSTRATIONS
# ============================================================================

def generate_svm_kernels_comparison():
    """Generate comparison of SVM with different kernels."""
    X, y = make_classification(n_samples=200, n_features=2, n_informative=2,
                               n_redundant=0, n_clusters_per_class=1, 
                               class_sep=1.5, random_state=42)
    
    kernels = ['linear', 'poly', 'rbf', 'sigmoid']
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    axes = axes.ravel()
    
    for idx, kernel in enumerate(kernels):
        clf = SVC(kernel=kernel, gamma='auto', random_state=42)
        clf.fit(X, y)
        
        # Plot decision boundary
        h = 0.02
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                            np.arange(y_min, y_max, h))
        
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        
        axes[idx].contourf(xx, yy, Z, alpha=0.3, cmap='RdYlBu')
        axes[idx].scatter(X[:, 0], X[:, 1], c=y, cmap='RdYlBu',
                         edgecolors='black', linewidth=0.5, s=50)
        
        # Plot support vectors
        axes[idx].scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1],
                         s=100, linewidth=1, facecolors='none', 
                         edgecolors='green', label='Опорные векторы')
        
        axes[idx].set_xlabel('Признак 1')
        axes[idx].set_ylabel('Признак 2')
        axes[idx].set_title(f'SVM с ядром: {kernel}')
        axes[idx].legend()
    
    plt.tight_layout()
    return fig_to_base64(fig)

def generate_svm_margin_visualization():
    """Generate visualization of SVM margin and support vectors."""
    np.random.seed(42)
    X = np.random.randn(40, 2)
    y = np.logical_xor(X[:, 0] > 0, X[:, 1] > 0)
    y = np.where(y, 1, -1)
    
    # Make linearly separable by adjusting
    X[y == 1] += 2
    
    clf = SVC(kernel='linear', C=1.0)
    clf.fit(X, y)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot decision boundary and margins
    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    
    # Create grid
    xx = np.linspace(xlim[0], xlim[1], 30)
    yy = np.linspace(ylim[0], ylim[1], 30)
    YY, XX = np.meshgrid(yy, xx)
    xy = np.vstack([XX.ravel(), YY.ravel()]).T
    Z = clf.decision_function(xy).reshape(XX.shape)
    
    # Plot decision boundary and margins
    ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5,
              linestyles=['--', '-', '--'])
    
    # Plot support vectors
    ax.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1],
              s=200, linewidth=2, facecolors='none', 
              edgecolors='green', label='Опорные векторы')
    
    # Plot samples
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap='RdYlBu', s=50,
              edgecolors='black', linewidth=0.5)
    
    ax.set_xlabel('Признак 1')
    ax.set_ylabel('Признак 2')
    ax.set_title('SVM: Разделяющая гиперплоскость и margin')
    ax.legend()
    
    return fig_to_base64(fig)

def generate_svm_c_parameter():
    """Generate visualization showing effect of C parameter."""
    X, y = make_classification(n_samples=100, n_features=2, n_informative=2,
                               n_redundant=0, n_clusters_per_class=1, 
                               class_sep=1.0, random_state=42)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    C_values = [0.1, 1.0, 100.0]
    
    for idx, C in enumerate(C_values):
        clf = SVC(kernel='rbf', C=C, gamma='auto', random_state=42)
        clf.fit(X, y)
        
        plot_decision_boundary(X, y, clf, 
                             f'C = {C}',
                             ax=axes[idx])
        
        # Add support vectors
        axes[idx].scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1],
                         s=100, linewidth=1.5, facecolors='none', 
                         edgecolors='green', label='Опорные векторы')
        axes[idx].legend()
    
    plt.suptitle('Влияние параметра C на SVM', fontsize=14, y=1.02)
    plt.tight_layout()
    return fig_to_base64(fig)

# ============================================================================
# LOGISTIC REGRESSION ILLUSTRATIONS
# ============================================================================

def generate_logreg_sigmoid():
    """Generate sigmoid function visualization."""
    z = np.linspace(-10, 10, 200)
    sigmoid = 1 / (1 + np.exp(-z))
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(z, sigmoid, 'b-', linewidth=2.5, label='σ(z) = 1/(1+e^(-z))')
    ax.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='Порог 0.5')
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.3)
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('z = w^T·x + b')
    ax.set_ylabel('σ(z)')
    ax.set_title('Сигмоидная функция активации')
    ax.legend()
    ax.set_ylim([-0.1, 1.1])
    
    return fig_to_base64(fig)

def generate_logreg_decision_boundary():
    """Generate logistic regression decision boundary."""
    X, y = make_classification(n_samples=200, n_features=2, n_informative=2,
                               n_redundant=0, n_clusters_per_class=1, 
                               random_state=42)
    
    clf = LogisticRegression(random_state=42)
    clf.fit(X, y)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create mesh
    h = 0.02
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    # Get probability predictions
    Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
    Z = Z.reshape(xx.shape)
    
    # Plot probability contours
    contour = ax.contourf(xx, yy, Z, alpha=0.4, cmap='RdYlBu', levels=20)
    plt.colorbar(contour, ax=ax, label='P(y=1)')
    
    # Plot decision boundary
    ax.contour(xx, yy, Z, levels=[0.5], colors='black', linewidths=2)
    
    # Plot samples
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap='RdYlBu',
              edgecolors='black', linewidth=0.5, s=50)
    
    ax.set_xlabel('Признак 1')
    ax.set_ylabel('Признак 2')
    ax.set_title('Логистическая регрессия: вероятностные контуры')
    
    return fig_to_base64(fig)

def generate_logreg_regularization():
    """Generate comparison of different regularization strengths."""
    X, y = make_classification(n_samples=100, n_features=2, n_informative=2,
                               n_redundant=0, n_clusters_per_class=1, 
                               random_state=42)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    C_values = [0.1, 1.0, 10.0]
    
    for idx, C in enumerate(C_values):
        clf = LogisticRegression(C=C, random_state=42)
        clf.fit(X, y)
        
        plot_decision_boundary(X, y, clf, 
                             f'C = {C} (регуляризация)',
                             ax=axes[idx])
    
    plt.suptitle('Влияние параметра регуляризации C', fontsize=14, y=1.02)
    plt.tight_layout()
    return fig_to_base64(fig)

# ============================================================================
# KNN ILLUSTRATIONS
# ============================================================================

def generate_knn_decision_boundary():
    """Generate KNN decision boundaries for different K values."""
    X, y = make_classification(n_samples=100, n_features=2, n_informative=2,
                               n_redundant=0, n_clusters_per_class=1, 
                               random_state=42)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.ravel()
    k_values = [1, 3, 5, 10, 20, 50]
    
    for idx, k in enumerate(k_values):
        clf = KNeighborsClassifier(n_neighbors=k)
        clf.fit(X, y)
        
        train_score = clf.score(X, y)
        plot_decision_boundary(X, y, clf, 
                             f'K = {k}\nТочность: {train_score:.2f}',
                             ax=axes[idx])
    
    plt.suptitle('KNN: влияние параметра K на границу решения', fontsize=14, y=1.00)
    plt.tight_layout()
    return fig_to_base64(fig)

def generate_knn_distance_metrics():
    """Generate comparison of different distance metrics."""
    X, y = make_classification(n_samples=100, n_features=2, n_informative=2,
                               n_redundant=0, n_clusters_per_class=1, 
                               random_state=42)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    metrics = ['euclidean', 'manhattan', 'minkowski']
    metric_names = ['Евклидово', 'Манхэттенское', 'Минковского (p=3)']
    
    for idx, (metric, name) in enumerate(zip(metrics, metric_names)):
        if metric == 'minkowski':
            clf = KNeighborsClassifier(n_neighbors=5, metric=metric, p=3)
        else:
            clf = KNeighborsClassifier(n_neighbors=5, metric=metric)
        clf.fit(X, y)
        
        plot_decision_boundary(X, y, clf, 
                             f'Метрика: {name}',
                             ax=axes[idx])
    
    plt.suptitle('KNN: различные метрики расстояния (K=5)', fontsize=14, y=1.02)
    plt.tight_layout()
    return fig_to_base64(fig)

def generate_knn_illustration():
    """Generate illustration showing how KNN works."""
    np.random.seed(42)
    # Create simple 2D dataset
    X_class0 = np.random.randn(30, 2) + np.array([2, 2])
    X_class1 = np.random.randn(30, 2) + np.array([-2, -2])
    X = np.vstack([X_class0, X_class1])
    y = np.array([0] * 30 + [1] * 30)
    
    # Test point
    X_test = np.array([[0, 0]])
    
    clf = KNeighborsClassifier(n_neighbors=5)
    clf.fit(X, y)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot training data
    ax.scatter(X[y==0, 0], X[y==0, 1], c='blue', label='Класс 0', 
              s=60, alpha=0.6, edgecolors='black', linewidth=0.5)
    ax.scatter(X[y==1, 0], X[y==1, 1], c='red', label='Класс 1', 
              s=60, alpha=0.6, edgecolors='black', linewidth=0.5)
    
    # Plot test point
    ax.scatter(X_test[0, 0], X_test[0, 1], c='green', marker='*', 
              s=400, label='Новая точка', edgecolors='black', linewidth=2)
    
    # Find K nearest neighbors
    distances, indices = clf.kneighbors(X_test)
    
    # Draw circles to K nearest neighbors
    for i, idx in enumerate(indices[0]):
        ax.plot([X_test[0, 0], X[idx, 0]], [X_test[0, 1], X[idx, 1]], 
               'g--', alpha=0.5, linewidth=1.5)
        # Highlight nearest neighbors
        color = 'blue' if y[idx] == 0 else 'red'
        ax.scatter(X[idx, 0], X[idx, 1], c=color, s=150, 
                  edgecolors='green', linewidth=2.5)
    
    ax.set_xlabel('Признак 1')
    ax.set_ylabel('Признак 2')
    ax.set_title('KNN (K=5): поиск ближайших соседей')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return fig_to_base64(fig)

# ============================================================================
# NAIVE BAYES ILLUSTRATIONS
# ============================================================================

def generate_naive_bayes_distributions():
    """Generate visualization of class distributions in Naive Bayes."""
    np.random.seed(42)
    
    # Generate data from two Gaussian distributions
    X_class0 = np.random.randn(100, 2) + np.array([2, 2])
    X_class1 = np.random.randn(100, 2) + np.array([-2, -2])
    X = np.vstack([X_class0, X_class1])
    y = np.array([0] * 100 + [1] * 100)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Feature 1 distributions
    axes[0].hist(X_class0[:, 0], bins=20, alpha=0.6, label='Класс 0', color='blue', density=True)
    axes[0].hist(X_class1[:, 0], bins=20, alpha=0.6, label='Класс 1', color='red', density=True)
    
    # Overlay Gaussian fits
    from scipy.stats import norm
    axes[0].plot(x_range, norm.pdf(x_range, X_class0[:, 0].mean(), X_class0[:, 0].std()),
                'b-', linewidth=2, label='N(μ₀, σ₀)')
    axes[0].plot(x_range, norm.pdf(x_range, X_class1[:, 0].mean(), X_class1[:, 0].std()),
                'r-', linewidth=2, label='N(μ₁, σ₁)')
    
    axes[0].set_xlabel('Признак 1')
    axes[0].set_ylabel('Плотность вероятности')
    axes[0].set_title('Распределения по признаку 1')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Feature 2 distributions
    axes[1].hist(X_class0[:, 1], bins=20, alpha=0.6, label='Класс 0', color='blue', density=True)
    axes[1].hist(X_class1[:, 1], bins=20, alpha=0.6, label='Класс 1', color='red', density=True)
    
    y_range = np.linspace(X[:, 1].min(), X[:, 1].max(), 100)
    axes[1].plot(y_range, norm.pdf(y_range, X_class0[:, 1].mean(), X_class0[:, 1].std()),
                'b-', linewidth=2, label='N(μ₀, σ₀)')
    axes[1].plot(y_range, norm.pdf(y_range, X_class1[:, 1].mean(), X_class1[:, 1].std()),
                'r-', linewidth=2, label='N(μ₁, σ₁)')
    
    axes[1].set_xlabel('Признак 2')
    axes[1].set_ylabel('Плотность вероятности')
    axes[1].set_title('Распределения по признаку 2')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.suptitle('Naive Bayes: Гауссовские распределения признаков', fontsize=14, y=1.02)
    plt.tight_layout()
    return fig_to_base64(fig)

def generate_naive_bayes_decision_boundary():
    """Generate Naive Bayes decision boundary."""
    X, y = make_classification(n_samples=200, n_features=2, n_informative=2,
                               n_redundant=0, n_clusters_per_class=1, 
                               random_state=42)
    
    clf = GaussianNB()
    clf.fit(X, y)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create mesh
    h = 0.02
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    # Get probability predictions
    Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
    Z = Z.reshape(xx.shape)
    
    # Plot probability contours
    contour = ax.contourf(xx, yy, Z, alpha=0.4, cmap='RdYlBu', levels=20)
    plt.colorbar(contour, ax=ax, label='P(y=1)')
    
    # Plot decision boundary
    ax.contour(xx, yy, Z, levels=[0.5], colors='black', linewidths=2)
    
    # Plot samples
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap='RdYlBu',
              edgecolors='black', linewidth=0.5, s=50)
    
    ax.set_xlabel('Признак 1')
    ax.set_ylabel('Признак 2')
    ax.set_title('Naive Bayes: вероятностные контуры и граница решения')
    
    return fig_to_base64(fig)

def generate_naive_bayes_comparison():
    """Generate comparison of Naive Bayes with other classifiers."""
    X, y = make_moons(n_samples=200, noise=0.2, random_state=42)
    
    classifiers = [
        ('Naive Bayes', GaussianNB()),
        ('Logistic Regression', LogisticRegression(random_state=42)),
        ('Decision Tree', DecisionTreeClassifier(max_depth=5, random_state=42))
    ]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for idx, (name, clf) in enumerate(classifiers):
        clf.fit(X, y)
        score = clf.score(X, y)
        
        plot_decision_boundary(X, y, clf, 
                             f'{name}\nТочность: {score:.2f}',
                             ax=axes[idx])
    
    plt.suptitle('Сравнение классификаторов на полулуниях', fontsize=14, y=1.02)
    plt.tight_layout()
    return fig_to_base64(fig)

# ============================================================================
# MAIN GENERATION FUNCTION
# ============================================================================

def generate_all_illustrations():
    """Generate all classification illustrations and return as dictionary."""
    print("Generating classification illustrations...")
    
    illustrations = {}
    
    print("  - Decision Trees...")
    illustrations['dt_structure'] = generate_decision_tree_visualization()
    illustrations['dt_depth_comparison'] = generate_decision_tree_depth_comparison()
    illustrations['dt_feature_importance'] = generate_decision_tree_feature_importance()
    
    print("  - Random Forest...")
    illustrations['rf_ensemble'] = generate_random_forest_ensemble()
    illustrations['rf_oob_error'] = generate_random_forest_oob_error()
    illustrations['rf_feature_importance'] = generate_random_forest_feature_importance()
    
    print("  - SVM...")
    illustrations['svm_kernels'] = generate_svm_kernels_comparison()
    illustrations['svm_margin'] = generate_svm_margin_visualization()
    illustrations['svm_c_parameter'] = generate_svm_c_parameter()
    
    print("  - Logistic Regression...")
    illustrations['logreg_sigmoid'] = generate_logreg_sigmoid()
    illustrations['logreg_boundary'] = generate_logreg_decision_boundary()
    illustrations['logreg_regularization'] = generate_logreg_regularization()
    
    print("  - KNN...")
    illustrations['knn_decision_boundary'] = generate_knn_decision_boundary()
    illustrations['knn_distance_metrics'] = generate_knn_distance_metrics()
    illustrations['knn_illustration'] = generate_knn_illustration()
    
    print("  - Naive Bayes...")
    illustrations['nb_distributions'] = generate_naive_bayes_distributions()
    illustrations['nb_boundary'] = generate_naive_bayes_decision_boundary()
    illustrations['nb_comparison'] = generate_naive_bayes_comparison()
    
    print("✓ All illustrations generated successfully!")
    return illustrations

if __name__ == '__main__':
    # Test generation
    illustrations = generate_all_illustrations()
    print(f"\nGenerated {len(illustrations)} illustrations")
    for key in illustrations.keys():
        print(f"  - {key}")
