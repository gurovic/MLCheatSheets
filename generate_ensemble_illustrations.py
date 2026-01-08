#!/usr/bin/env python3
"""
Generate matplotlib illustrations for ensemble method cheatsheets.
This script creates high-quality visualizations and encodes them as base64 for inline HTML embedding.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import (
    RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier,
    BaggingClassifier, VotingClassifier
)
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.datasets import make_classification, make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle
from matplotlib.collections import PatchCollection
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
# ENSEMBLE METHODS ILLUSTRATIONS
# ============================================================================

def generate_voting_diagram():
    """Generate voting ensemble architecture diagram."""
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Title
    ax.text(5, 9.5, 'Voting Ensemble Architecture', 
            ha='center', va='top', fontsize=14, fontweight='bold')
    
    # Input data box
    input_box = FancyBboxPatch((4, 7.5), 2, 0.6, 
                               boxstyle="round,pad=0.1", 
                               facecolor='lightblue', edgecolor='blue', linewidth=2)
    ax.add_patch(input_box)
    ax.text(5, 7.8, 'Входные данные', ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Three models
    models = ['Модель 1\n(Logistic)', 'Модель 2\n(Decision Tree)', 'Модель 3\n(SVM)']
    colors = ['lightcoral', 'lightgreen', 'lightyellow']
    x_positions = [1.5, 5, 8.5]
    
    for i, (model, color, x_pos) in enumerate(zip(models, colors, x_positions)):
        # Model box
        model_box = FancyBboxPatch((x_pos - 0.8, 5), 1.6, 1.2,
                                   boxstyle="round,pad=0.1",
                                   facecolor=color, edgecolor='black', linewidth=1.5)
        ax.add_patch(model_box)
        ax.text(x_pos, 5.6, model, ha='center', va='center', fontsize=9)
        
        # Arrow from input to model
        arrow = FancyArrowPatch((5, 7.5), (x_pos, 6.2),
                               arrowstyle='->', lw=2, color='gray')
        ax.add_patch(arrow)
        
        # Predictions
        pred_text = f'Предсказание {i+1}'
        ax.text(x_pos, 4.5, pred_text, ha='center', va='center', fontsize=8)
        
        # Arrow from model to voting
        arrow2 = FancyArrowPatch((x_pos, 5), (5, 3.2),
                                arrowstyle='->', lw=2, color='gray')
        ax.add_patch(arrow2)
    
    # Voting box
    voting_box = FancyBboxPatch((3.5, 2), 3, 1.2,
                                boxstyle="round,pad=0.1",
                                facecolor='lavender', edgecolor='purple', linewidth=2)
    ax.add_patch(voting_box)
    ax.text(5, 2.8, 'Голосование', ha='center', va='center', fontsize=11, fontweight='bold')
    ax.text(5, 2.4, '(Hard/Soft)', ha='center', va='center', fontsize=9, style='italic')
    
    # Final prediction
    arrow3 = FancyArrowPatch((5, 2), (5, 1),
                            arrowstyle='->', lw=3, color='darkgreen')
    ax.add_patch(arrow3)
    
    final_box = FancyBboxPatch((3.8, 0.2), 2.4, 0.7,
                               boxstyle="round,pad=0.1",
                               facecolor='lightgreen', edgecolor='darkgreen', linewidth=2)
    ax.add_patch(final_box)
    ax.text(5, 0.55, 'Итоговое предсказание', ha='center', va='center', 
            fontsize=10, fontweight='bold')
    
    return fig_to_base64(fig)

def generate_ensemble_comparison():
    """Generate comparison of ensemble predictions vs individual models."""
    np.random.seed(42)
    X, y = make_moons(n_samples=200, noise=0.3, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Train models
    lr = LogisticRegression(random_state=42)
    dt = DecisionTreeClassifier(max_depth=3, random_state=42)
    svm = SVC(kernel='rbf', probability=True, random_state=42)
    
    voting = VotingClassifier(
        estimators=[('lr', lr), ('dt', dt), ('svm', svm)],
        voting='soft'
    )
    
    models = [lr, dt, svm, voting]
    titles = ['Logistic Regression', 'Decision Tree', 'SVM', 'Voting Ensemble']
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.ravel()
    
    # Create mesh for decision boundaries
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))
    
    for idx, (model, title) in enumerate(zip(models, titles)):
        model.fit(X_train, y_train)
        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        
        axes[idx].contourf(xx, yy, Z, alpha=0.4, cmap='RdYlBu')
        axes[idx].scatter(X_test[:, 0], X_test[:, 1], c=y_test, 
                         cmap='RdYlBu', edgecolors='black', linewidth=0.5, s=50)
        
        score = model.score(X_test, y_test)
        axes[idx].set_title(f'{title}\nAccuracy: {score:.3f}', fontweight='bold')
        axes[idx].set_xlabel('Признак 1')
        axes[idx].set_ylabel('Признак 2')
    
    plt.suptitle('Сравнение моделей: Ансамбль vs Отдельные модели', 
                 fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()
    
    return fig_to_base64(fig)

# ============================================================================
# BAGGING ILLUSTRATIONS
# ============================================================================

def generate_bagging_diagram():
    """Generate bagging bootstrap sampling diagram."""
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Title
    ax.text(6, 9.5, 'Bagging (Bootstrap Aggregating)', 
            ha='center', va='top', fontsize=14, fontweight='bold')
    
    # Original dataset
    orig_box = FancyBboxPatch((4.5, 7.8), 3, 0.8,
                             boxstyle="round,pad=0.1",
                             facecolor='lightblue', edgecolor='blue', linewidth=2)
    ax.add_patch(orig_box)
    ax.text(6, 8.2, 'Исходный датасет', ha='center', va='center', 
            fontsize=11, fontweight='bold')
    
    # Bootstrap samples
    n_bags = 4
    x_positions = np.linspace(1.5, 10.5, n_bags)
    
    for i, x_pos in enumerate(x_positions):
        # Arrow from original to bootstrap
        arrow = FancyArrowPatch((6, 7.8), (x_pos, 6.8),
                               arrowstyle='->', lw=1.5, color='gray')
        ax.add_patch(arrow)
        
        # Bootstrap sample
        boot_box = FancyBboxPatch((x_pos - 0.7, 6), 1.4, 0.8,
                                 boxstyle="round,pad=0.05",
                                 facecolor='wheat', edgecolor='orange', linewidth=1.5)
        ax.add_patch(boot_box)
        ax.text(x_pos, 6.5, f'Bootstrap {i+1}', ha='center', va='center', fontsize=9)
        ax.text(x_pos, 6.1, '(с возвращ.)', ha='center', va='center', fontsize=7, style='italic')
        
        # Model trained on bootstrap
        model_box = FancyBboxPatch((x_pos - 0.7, 4.2), 1.4, 1.2,
                                  boxstyle="round,pad=0.05",
                                  facecolor='lightgreen', edgecolor='green', linewidth=1.5)
        ax.add_patch(model_box)
        ax.text(x_pos, 4.9, f'Модель {i+1}', ha='center', va='center', fontsize=9, fontweight='bold')
        ax.text(x_pos, 4.5, '(дерево)', ha='center', va='center', fontsize=8)
        
        # Arrow from bootstrap to model
        arrow2 = FancyArrowPatch((x_pos, 6), (x_pos, 5.4),
                                arrowstyle='->', lw=1.5, color='gray')
        ax.add_patch(arrow2)
        
        # Arrow from model to aggregation
        arrow3 = FancyArrowPatch((x_pos, 4.2), (6, 2.8),
                                arrowstyle='->', lw=1.5, color='gray')
        ax.add_patch(arrow3)
    
    # Aggregation
    agg_box = FancyBboxPatch((4.5, 1.8), 3, 1,
                            boxstyle="round,pad=0.1",
                            facecolor='lavender', edgecolor='purple', linewidth=2)
    ax.add_patch(agg_box)
    ax.text(6, 2.5, 'Агрегация', ha='center', va='center', fontsize=11, fontweight='bold')
    ax.text(6, 2.1, '(усреднение/голосование)', ha='center', va='center', fontsize=8)
    
    # Final prediction
    arrow4 = FancyArrowPatch((6, 1.8), (6, 1),
                            arrowstyle='->', lw=2.5, color='darkgreen')
    ax.add_patch(arrow4)
    
    final_box = FancyBboxPatch((4.8, 0.2), 2.4, 0.7,
                              boxstyle="round,pad=0.1",
                              facecolor='lightgreen', edgecolor='darkgreen', linewidth=2)
    ax.add_patch(final_box)
    ax.text(6, 0.55, 'Итоговое предсказание', ha='center', va='center', 
            fontsize=10, fontweight='bold')
    
    return fig_to_base64(fig)

def generate_bagging_variance_reduction():
    """Generate visualization showing variance reduction with bagging."""
    np.random.seed(42)
    X, y = make_classification(n_samples=200, n_features=2, n_informative=2,
                               n_redundant=0, n_clusters_per_class=1,
                               random_state=42, flip_y=0.1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Single tree vs bagging
    single_tree = DecisionTreeClassifier(random_state=42)
    bagging = BaggingClassifier(DecisionTreeClassifier(), 
                                n_estimators=50, random_state=42)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    models = [single_tree, bagging]
    titles = ['Одно дерево решений', 'Bagging (50 деревьев)']
    
    # Create mesh
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))
    
    for idx, (model, title) in enumerate(zip(models, titles)):
        model.fit(X_train, y_train)
        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        
        axes[idx].contourf(xx, yy, Z, alpha=0.4, cmap='RdYlBu')
        axes[idx].scatter(X_test[:, 0], X_test[:, 1], c=y_test, 
                         cmap='RdYlBu', edgecolors='black', linewidth=0.5, s=50)
        
        score = model.score(X_test, y_test)
        axes[idx].set_title(f'{title}\nAccuracy: {score:.3f}', fontweight='bold')
        axes[idx].set_xlabel('Признак 1')
        axes[idx].set_ylabel('Признак 2')
    
    plt.suptitle('Bagging: Снижение переобучения через усреднение', 
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    
    return fig_to_base64(fig)

# ============================================================================
# BOOSTING ILLUSTRATIONS
# ============================================================================

def generate_boosting_diagram():
    """Generate boosting sequential learning diagram."""
    fig, ax = plt.subplots(figsize=(12, 9))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 11)
    ax.axis('off')
    
    # Title
    ax.text(6, 10.5, 'Boosting: Последовательное обучение', 
            ha='center', va='top', fontsize=14, fontweight='bold')
    
    # Initial data
    data_box = FancyBboxPatch((4.5, 9), 3, 0.7,
                             boxstyle="round,pad=0.1",
                             facecolor='lightblue', edgecolor='blue', linewidth=2)
    ax.add_patch(data_box)
    ax.text(6, 9.35, 'Исходные данные (равные веса)', ha='center', va='center', 
            fontsize=10, fontweight='bold')
    
    # Iterations
    n_iterations = 3
    y_positions = [7.5, 5, 2.5]
    
    for i, y_pos in enumerate(y_positions):
        iteration_num = i + 1
        
        # Weighted data
        weight_box = FancyBboxPatch((0.5, y_pos + 0.5), 2.5, 1.2,
                                   boxstyle="round,pad=0.05",
                                   facecolor='lightyellow', edgecolor='orange', linewidth=1.5)
        ax.add_patch(weight_box)
        ax.text(1.75, y_pos + 1.3, f'Взвешенные\nданные {iteration_num}', 
                ha='center', va='center', fontsize=9)
        
        if i == 0:
            arrow_from_data = FancyArrowPatch((6, 9), (1.75, y_pos + 1.7),
                                             arrowstyle='->', lw=2, color='gray')
            ax.add_patch(arrow_from_data)
        else:
            arrow_from_prev = FancyArrowPatch((10.5, y_positions[i-1] + 0.5), 
                                             (1.75, y_pos + 1.7),
                                             arrowstyle='->', lw=2, color='gray',
                                             connectionstyle="arc3,rad=0.3")
            ax.add_patch(arrow_from_prev)
        
        # Train model
        arrow1 = FancyArrowPatch((3, y_pos + 1.1), (4.5, y_pos + 1.1),
                                arrowstyle='->', lw=2, color='gray')
        ax.add_patch(arrow1)
        
        # Weak learner
        model_box = FancyBboxPatch((4.5, y_pos + 0.5), 2.5, 1.2,
                                  boxstyle="round,pad=0.05",
                                  facecolor='lightgreen', edgecolor='green', linewidth=1.5)
        ax.add_patch(model_box)
        ax.text(5.75, y_pos + 1.3, f'Слабая модель {iteration_num}', 
                ha='center', va='center', fontsize=9, fontweight='bold')
        ax.text(5.75, y_pos + 0.8, f'(α{iteration_num})', 
                ha='center', va='center', fontsize=8, style='italic')
        
        # Evaluate errors
        arrow2 = FancyArrowPatch((7, y_pos + 1.1), (8.5, y_pos + 1.1),
                                arrowstyle='->', lw=2, color='gray')
        ax.add_patch(arrow2)
        
        error_box = FancyBboxPatch((8.5, y_pos + 0.5), 2.5, 1.2,
                                  boxstyle="round,pad=0.05",
                                  facecolor='lightcoral', edgecolor='red', linewidth=1.5)
        ax.add_patch(error_box)
        ax.text(9.75, y_pos + 1.3, f'Оценка ошибок', 
                ha='center', va='center', fontsize=9)
        ax.text(9.75, y_pos + 0.8, 'Обновить веса', 
                ha='center', va='center', fontsize=8)
    
    # Final combination
    ax.text(6, 0.8, '↓', ha='center', va='center', fontsize=30, color='darkgreen')
    
    final_box = FancyBboxPatch((3.5, 0), 5, 0.6,
                              boxstyle="round,pad=0.1",
                              facecolor='lightgreen', edgecolor='darkgreen', linewidth=2)
    ax.add_patch(final_box)
    ax.text(6, 0.3, 'F(x) = α₁·f₁(x) + α₂·f₂(x) + α₃·f₃(x)', 
            ha='center', va='center', fontsize=11, fontweight='bold')
    
    return fig_to_base64(fig)

def generate_boosting_vs_bagging():
    """Generate comparison of boosting vs bagging."""
    np.random.seed(42)
    X, y = make_moons(n_samples=200, noise=0.25, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Models
    single_tree = DecisionTreeClassifier(max_depth=1, random_state=42)
    bagging = BaggingClassifier(DecisionTreeClassifier(max_depth=3), 
                                n_estimators=50, random_state=42)
    boosting = GradientBoostingClassifier(n_estimators=50, learning_rate=0.1, 
                                         max_depth=3, random_state=42)
    
    models = [single_tree, bagging, boosting]
    titles = ['Одно слабое дерево', 'Bagging', 'Gradient Boosting']
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Create mesh
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))
    
    for idx, (model, title) in enumerate(zip(models, titles)):
        model.fit(X_train, y_train)
        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        
        axes[idx].contourf(xx, yy, Z, alpha=0.4, cmap='RdYlBu')
        axes[idx].scatter(X_test[:, 0], X_test[:, 1], c=y_test, 
                         cmap='RdYlBu', edgecolors='black', linewidth=0.5, s=50)
        
        score = model.score(X_test, y_test)
        axes[idx].set_title(f'{title}\nAccuracy: {score:.3f}', fontweight='bold')
        axes[idx].set_xlabel('Признак 1')
        axes[idx].set_ylabel('Признак 2')
    
    plt.suptitle('Сравнение: Bagging vs Boosting', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    return fig_to_base64(fig)

# ============================================================================
# ADABOOST ILLUSTRATIONS
# ============================================================================

def generate_adaboost_weight_updates():
    """Generate AdaBoost weight update visualization."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.ravel()
    
    np.random.seed(42)
    n_samples = 40
    X = np.random.randn(n_samples, 2)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    
    # Simulate weight updates over iterations
    weights = np.ones(n_samples)
    
    for iteration in range(4):
        ax = axes[iteration]
        
        # Normalize weights for visualization
        sizes = 100 + 400 * (weights / weights.max())
        
        scatter = ax.scatter(X[:, 0], X[:, 1], c=y, s=sizes, 
                           cmap='RdYlBu', alpha=0.6, edgecolors='black', linewidth=1)
        
        ax.set_title(f'Итерация {iteration + 1}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Признак 1')
        ax.set_ylabel('Признак 2')
        ax.grid(True, alpha=0.3)
        
        # Simulate misclassification and weight update
        if iteration < 3:
            # Randomly increase some weights (simulate errors)
            error_indices = np.random.choice(n_samples, size=n_samples//5, replace=False)
            weights[error_indices] *= 2
            weights = weights / weights.sum() * n_samples
    
    plt.suptitle('AdaBoost: Обновление весов образцов\n(размер точки = вес)', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    return fig_to_base64(fig)

def generate_adaboost_performance():
    """Generate AdaBoost performance over iterations."""
    np.random.seed(42)
    X, y = make_classification(n_samples=300, n_features=2, n_informative=2,
                               n_redundant=0, n_clusters_per_class=1,
                               random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Train AdaBoost and track performance
    n_estimators_range = range(1, 101, 5)
    train_scores = []
    test_scores = []
    
    for n_est in n_estimators_range:
        ada = AdaBoostClassifier(n_estimators=n_est, random_state=42)
        ada.fit(X_train, y_train)
        train_scores.append(ada.score(X_train, y_train))
        test_scores.append(ada.score(X_test, y_test))
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(n_estimators_range, train_scores, 'b-o', label='Train', linewidth=2, markersize=4)
    ax.plot(n_estimators_range, test_scores, 'r-s', label='Test', linewidth=2, markersize=4)
    
    ax.set_xlabel('Количество слабых моделей', fontsize=11)
    ax.set_ylabel('Accuracy', fontsize=11)
    ax.set_title('AdaBoost: Качество от числа итераций', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    return fig_to_base64(fig)

# ============================================================================
# GRADIENT BOOSTING (XGBoost/LightGBM/CatBoost) ILLUSTRATIONS
# ============================================================================

def generate_gradient_boosting_residuals():
    """Visualize gradient boosting fitting residuals."""
    np.random.seed(42)
    X = np.linspace(0, 10, 100).reshape(-1, 1)
    y = np.sin(X).ravel() + np.random.normal(0, 0.3, X.shape[0])
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.ravel()
    
    from sklearn.tree import DecisionTreeRegressor
    
    # Initial prediction (mean)
    y_pred = np.full_like(y, y.mean())
    
    for iteration in range(4):
        ax = axes[iteration]
        
        # Calculate residuals
        residuals = y - y_pred
        
        # Plot data and predictions
        ax.scatter(X, y, alpha=0.5, s=30, label='Данные', color='blue')
        ax.plot(X, y_pred, 'r-', linewidth=2, label='Текущее предсказание')
        
        # Show residuals as vertical lines
        for i in range(0, len(X), 5):  # Show every 5th residual for clarity
            ax.plot([X[i], X[i]], [y_pred[i], y[i]], 'g--', alpha=0.3, linewidth=1)
        
        ax.set_title(f'Итерация {iteration + 1}\nОстаточная ошибка: {np.mean(residuals**2):.3f}', 
                    fontweight='bold')
        ax.set_xlabel('X')
        ax.set_ylabel('y')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        
        # Fit tree to residuals and update prediction
        if iteration < 3:
            tree = DecisionTreeRegressor(max_depth=2, random_state=42)
            tree.fit(X, residuals)
            y_pred += 0.3 * tree.predict(X)  # Learning rate = 0.3
    
    plt.suptitle('Gradient Boosting: Подгонка остатков (residuals)', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    return fig_to_base64(fig)

def generate_tree_growth_comparison():
    """Compare level-wise vs leaf-wise tree growth."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Level-wise (XGBoost)
    ax1 = axes[0]
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 10)
    ax1.axis('off')
    ax1.set_title('Level-wise (XGBoost, LightGBM balanced)', 
                  fontsize=12, fontweight='bold')
    
    # Draw tree structure
    # Root
    root = Circle((5, 8), 0.3, facecolor='lightblue', edgecolor='black', linewidth=2)
    ax1.add_patch(root)
    ax1.text(5, 8, '1', ha='center', va='center', fontweight='bold')
    
    # Level 1
    for i, x in enumerate([2.5, 7.5]):
        node = Circle((x, 5.5), 0.3, facecolor='lightgreen', edgecolor='black', linewidth=2)
        ax1.add_patch(node)
        ax1.text(x, 5.5, str(i+2), ha='center', va='center', fontweight='bold')
        ax1.plot([5, x], [7.7, 5.8], 'k-', linewidth=2)
    
    # Level 2
    for i, x in enumerate([1, 2.5, 4, 6.5, 7.5, 9]):
        node = Circle((x, 3), 0.3, facecolor='lightyellow', edgecolor='black', linewidth=1.5)
        ax1.add_patch(node)
        ax1.text(x, 3, str(i+4), ha='center', va='center', fontsize=9)
        parent_x = 2.5 if x < 4 else 7.5
        ax1.plot([parent_x, x], [5.2, 3.3], 'k-', linewidth=1.5)
    
    ax1.text(5, 0.5, 'Растёт по уровням (сбалансированное дерево)', 
            ha='center', fontsize=10, style='italic')
    
    # Leaf-wise (LightGBM)
    ax2 = axes[1]
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 10)
    ax2.axis('off')
    ax2.set_title('Leaf-wise (LightGBM default)', 
                  fontsize=12, fontweight='bold')
    
    # Root
    root2 = Circle((5, 8), 0.3, facecolor='lightblue', edgecolor='black', linewidth=2)
    ax2.add_patch(root2)
    ax2.text(5, 8, '1', ha='center', va='center', fontweight='bold')
    
    # Asymmetric growth - focus on best leaf
    node2 = Circle((3, 6), 0.3, facecolor='lightgreen', edgecolor='black', linewidth=2)
    ax2.add_patch(node2)
    ax2.text(3, 6, '2', ha='center', va='center', fontweight='bold')
    ax2.plot([5, 3], [7.7, 6.3], 'k-', linewidth=2)
    
    node3 = Circle((7, 6), 0.3, facecolor='lightgray', edgecolor='gray', linewidth=1)
    ax2.add_patch(node3)
    ax2.text(7, 6, 'leaf', ha='center', va='center', fontsize=8)
    ax2.plot([5, 7], [7.7, 6.3], 'k-', linewidth=1)
    
    # Continue growing best leaf
    node4 = Circle((1.5, 4), 0.3, facecolor='lightyellow', edgecolor='black', linewidth=2)
    ax2.add_patch(node4)
    ax2.text(1.5, 4, '3', ha='center', va='center', fontweight='bold')
    ax2.plot([3, 1.5], [5.7, 4.3], 'k-', linewidth=2)
    
    node5 = Circle((4.5, 4), 0.3, facecolor='lightgray', edgecolor='gray', linewidth=1)
    ax2.add_patch(node5)
    ax2.text(4.5, 4, 'leaf', ha='center', va='center', fontsize=8)
    ax2.plot([3, 4.5], [5.7, 4.3], 'k-', linewidth=1)
    
    # Continue further
    node6 = Circle((0.5, 2), 0.3, facecolor='lightcoral', edgecolor='black', linewidth=2)
    ax2.add_patch(node6)
    ax2.text(0.5, 2, '4', ha='center', va='center', fontweight='bold')
    ax2.plot([1.5, 0.5], [3.7, 2.3], 'k-', linewidth=2)
    
    node7 = Circle((2.5, 2), 0.3, facecolor='lightgray', edgecolor='gray', linewidth=1)
    ax2.add_patch(node7)
    ax2.text(2.5, 2, 'leaf', ha='center', va='center', fontsize=8)
    ax2.plot([1.5, 2.5], [3.7, 2.3], 'k-', linewidth=1)
    
    ax2.text(5, 0.5, 'Растёт по лучшему листу (может быть глубже)', 
            ha='center', fontsize=10, style='italic')
    
    plt.suptitle('Стратегии роста дерева', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    return fig_to_base64(fig)

def generate_learning_rate_effect():
    """Visualize effect of learning rate in gradient boosting."""
    np.random.seed(42)
    X = np.linspace(0, 10, 100).reshape(-1, 1)
    y = np.sin(X).ravel() + np.random.normal(0, 0.2, X.shape[0])
    
    learning_rates = [0.01, 0.1, 0.5, 1.0]
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.ravel()
    
    for idx, lr in enumerate(learning_rates):
        gb = GradientBoostingClassifier(n_estimators=50, learning_rate=lr, 
                                        max_depth=3, random_state=42)
        
        # For visualization, use regression
        from sklearn.ensemble import GradientBoostingRegressor
        gb_reg = GradientBoostingRegressor(n_estimators=50, learning_rate=lr,
                                          max_depth=2, random_state=42)
        gb_reg.fit(X, y)
        y_pred = gb_reg.predict(X)
        
        axes[idx].scatter(X, y, alpha=0.5, s=30, label='Данные')
        axes[idx].plot(X, y_pred, 'r-', linewidth=2, label='Предсказание')
        axes[idx].set_title(f'Learning Rate = {lr}', fontweight='bold')
        axes[idx].set_xlabel('X')
        axes[idx].set_ylabel('y')
        axes[idx].legend()
        axes[idx].grid(True, alpha=0.3)
    
    plt.suptitle('Gradient Boosting: Влияние Learning Rate', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    return fig_to_base64(fig)

# ============================================================================
# STACKING ILLUSTRATIONS
# ============================================================================

def generate_stacking_diagram():
    """Generate stacking architecture diagram."""
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 12)
    ax.axis('off')
    
    # Title
    ax.text(6, 11.5, 'Stacking: Многоуровневое обучение', 
            ha='center', va='top', fontsize=14, fontweight='bold')
    
    # Training data
    train_box = FancyBboxPatch((1, 9.5), 4, 0.8,
                              boxstyle="round,pad=0.1",
                              facecolor='lightblue', edgecolor='blue', linewidth=2)
    ax.add_patch(train_box)
    ax.text(3, 9.9, 'Обучающая выборка', ha='center', va='center', 
            fontsize=10, fontweight='bold')
    
    # Test data
    test_box = FancyBboxPatch((7, 9.5), 4, 0.8,
                             boxstyle="round,pad=0.1",
                             facecolor='lightcyan', edgecolor='blue', linewidth=2)
    ax.add_patch(test_box)
    ax.text(9, 9.9, 'Тестовая выборка', ha='center', va='center', 
            fontsize=10, fontweight='bold')
    
    # Level 0 - Base models
    ax.text(6, 8.5, 'Уровень 0: Базовые модели', ha='center', va='center', 
            fontsize=11, fontweight='bold', bbox=dict(boxstyle='round', facecolor='wheat'))
    
    base_models = ['Модель 1\n(Random Forest)', 'Модель 2\n(XGBoost)', 'Модель 3\n(SVM)']
    colors = ['lightcoral', 'lightgreen', 'lightyellow']
    x_positions = [1.5, 4.5, 7.5]
    
    for i, (model, color, x_pos) in enumerate(zip(base_models, colors, x_positions)):
        # Train section - base model
        model_box_train = FancyBboxPatch((x_pos - 0.9, 6.5), 1.8, 1,
                                        boxstyle="round,pad=0.05",
                                        facecolor=color, edgecolor='black', linewidth=1.5)
        ax.add_patch(model_box_train)
        ax.text(x_pos, 7, model, ha='center', va='center', fontsize=8)
        
        # Arrow from train data to model
        arrow1 = FancyArrowPatch((3, 9.5), (x_pos, 7.5),
                                arrowstyle='->', lw=1.5, color='gray')
        ax.add_patch(arrow1)
        
        # CV predictions for train
        ax.text(x_pos, 6, f'CV pred {i+1}', ha='center', va='center', fontsize=7)
        arrow2 = FancyArrowPatch((x_pos, 6.5), (x_pos, 5.2),
                                arrowstyle='->', lw=1.5, color='gray')
        ax.add_patch(arrow2)
        
        # Test section - same model
        model_box_test = FancyBboxPatch((x_pos + 4, 6.5), 1.8, 1,
                                       boxstyle="round,pad=0.05",
                                       facecolor=color, edgecolor='black', linewidth=1.5)
        ax.add_patch(model_box_test)
        ax.text(x_pos + 4, 7, model, ha='center', va='center', fontsize=8)
        
        # Arrow from test data to model
        arrow3 = FancyArrowPatch((9, 9.5), (x_pos + 4, 7.5),
                                arrowstyle='->', lw=1.5, color='gray')
        ax.add_patch(arrow3)
        
        # Test predictions
        ax.text(x_pos + 4, 6, f'Test pred {i+1}', ha='center', va='center', fontsize=7)
        arrow4 = FancyArrowPatch((x_pos + 4, 6.5), (6, 3.8),
                                arrowstyle='->', lw=1.5, color='gray')
        ax.add_patch(arrow4)
    
    # Meta-features (train)
    meta_train = FancyBboxPatch((2, 4.2), 3, 1,
                               boxstyle="round,pad=0.1",
                               facecolor='lavender', edgecolor='purple', linewidth=2)
    ax.add_patch(meta_train)
    ax.text(3.5, 4.7, 'Мета-признаки\n(из CV предсказаний)', ha='center', va='center', fontsize=9)
    
    # Level 1 - Meta model
    ax.text(6, 3, 'Уровень 1: Мета-модель', ha='center', va='center', 
            fontsize=11, fontweight='bold', bbox=dict(boxstyle='round', facecolor='wheat'))
    
    meta_model_train = FancyBboxPatch((2.5, 1.2), 3, 1,
                                     boxstyle="round,pad=0.1",
                                     facecolor='lightgreen', edgecolor='darkgreen', linewidth=2)
    ax.add_patch(meta_model_train)
    ax.text(4, 1.7, 'Мета-модель\n(Logistic Regression)', 
            ha='center', va='center', fontsize=9, fontweight='bold')
    
    arrow5 = FancyArrowPatch((3.5, 4.2), (4, 2.2),
                            arrowstyle='->', lw=2, color='darkblue')
    ax.add_patch(arrow5)
    
    # Meta-features (test)
    meta_test = FancyBboxPatch((7, 2.5), 3, 1,
                              boxstyle="round,pad=0.1",
                              facecolor='lavender', edgecolor='purple', linewidth=2)
    ax.add_patch(meta_test)
    ax.text(8.5, 3, 'Мета-признаки\n(из test предсказаний)', ha='center', va='center', fontsize=9)
    
    # Final prediction
    final_box = FancyBboxPatch((7, 0.5), 3, 0.7,
                              boxstyle="round,pad=0.1",
                              facecolor='gold', edgecolor='darkorange', linewidth=2)
    ax.add_patch(final_box)
    ax.text(8.5, 0.85, 'Итоговое предсказание', ha='center', va='center', 
            fontsize=10, fontweight='bold')
    
    arrow6 = FancyArrowPatch((8.5, 2.5), (8.5, 1.2),
                            arrowstyle='->', lw=2.5, color='darkgreen')
    ax.add_patch(arrow6)
    
    return fig_to_base64(fig)

# ============================================================================
# MAIN FUNCTION
# ============================================================================

def generate_all_illustrations():
    """Generate all ensemble illustrations and return as dictionary."""
    print("Generating ensemble method illustrations...")
    
    illustrations = {}
    
    # Ensemble Methods
    print("  - Voting diagram...")
    illustrations['voting_diagram'] = generate_voting_diagram()
    
    print("  - Ensemble comparison...")
    illustrations['ensemble_comparison'] = generate_ensemble_comparison()
    
    # Bagging
    print("  - Bagging diagram...")
    illustrations['bagging_diagram'] = generate_bagging_diagram()
    
    print("  - Bagging variance reduction...")
    illustrations['bagging_variance_reduction'] = generate_bagging_variance_reduction()
    
    # Boosting
    print("  - Boosting diagram...")
    illustrations['boosting_diagram'] = generate_boosting_diagram()
    
    print("  - Boosting vs bagging...")
    illustrations['boosting_vs_bagging'] = generate_boosting_vs_bagging()
    
    # AdaBoost
    print("  - AdaBoost weight updates...")
    illustrations['adaboost_weight_updates'] = generate_adaboost_weight_updates()
    
    print("  - AdaBoost performance...")
    illustrations['adaboost_performance'] = generate_adaboost_performance()
    
    # Gradient Boosting (XGBoost/LightGBM/CatBoost)
    print("  - Gradient boosting residuals...")
    illustrations['gradient_boosting_residuals'] = generate_gradient_boosting_residuals()
    
    print("  - Tree growth comparison...")
    illustrations['tree_growth_comparison'] = generate_tree_growth_comparison()
    
    print("  - Learning rate effect...")
    illustrations['learning_rate_effect'] = generate_learning_rate_effect()
    
    # Stacking
    print("  - Stacking diagram...")
    illustrations['stacking_diagram'] = generate_stacking_diagram()
    
    print(f"✓ Generated {len(illustrations)} illustrations")
    
    return illustrations

if __name__ == '__main__':
    print("=" * 70)
    print("Generating Ensemble Method Illustrations")
    print("=" * 70)
    illustrations = generate_all_illustrations()
    print("\n" + "=" * 70)
    print("All illustrations generated successfully!")
    print("=" * 70)
