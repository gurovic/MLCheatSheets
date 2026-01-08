#!/usr/bin/env python3
"""
Generate matplotlib illustrations for interpretability cheatsheets.
This script creates high-quality visualizations and encodes them as base64 for inline HTML embedding.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance, PartialDependenceDisplay
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
# EXPLAINABLE AI (XAI) ILLUSTRATIONS
# ============================================================================

def generate_xai_feature_importance():
    """Generate feature importance bar chart for XAI."""
    np.random.seed(42)
    X, y = make_classification(n_samples=1000, n_features=10, n_informative=6, 
                                n_redundant=2, random_state=42)
    
    # Train model
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X, y)
    
    # Get feature importances
    importances = rf.feature_importances_
    feature_names = [f'Feature {i+1}' for i in range(10)]
    
    # Sort by importance
    indices = np.argsort(importances)[::-1]
    
    fig, ax = plt.subplots(figsize=(9, 6))
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(indices)))
    ax.barh(range(len(indices)), importances[indices], color=colors, edgecolor='black', linewidth=0.5)
    ax.set_yticks(range(len(indices)))
    ax.set_yticklabels([feature_names[i] for i in indices])
    ax.set_xlabel('Важность признака')
    ax.set_title('Feature Importance: Random Forest')
    ax.grid(True, alpha=0.3, axis='x')
    
    # Add values on bars
    for i, v in enumerate(importances[indices]):
        ax.text(v + 0.002, i, f'{v:.3f}', va='center', fontsize=8)
    
    return fig_to_base64(fig)

def generate_xai_model_comparison():
    """Generate comparison of interpretable vs complex models."""
    # Simulated accuracy vs interpretability trade-off
    models = ['Linear\nRegression', 'Decision\nTree', 'Random\nForest', 
              'XGBoost', 'Neural\nNetwork', 'Deep\nNeural Net']
    interpretability = [9.5, 8.5, 6.0, 4.5, 3.0, 2.0]
    accuracy = [70, 75, 85, 88, 90, 92]
    
    fig, ax = plt.subplots(figsize=(9, 6))
    colors = ['#2ecc71', '#27ae60', '#f39c12', '#e67e22', '#e74c3c', '#c0392b']
    
    scatter = ax.scatter(interpretability, accuracy, s=300, c=colors, 
                        alpha=0.7, edgecolors='black', linewidth=1.5)
    
    # Add labels
    for i, model in enumerate(models):
        ax.annotate(model, (interpretability[i], accuracy[i]), 
                   ha='center', va='center', fontsize=9, fontweight='bold')
    
    ax.set_xlabel('Интерпретируемость (выше → лучше)', fontsize=11)
    ax.set_ylabel('Точность (%)', fontsize=11)
    ax.set_title('Trade-off: Интерпретируемость vs Точность', fontsize=12, fontweight='bold')
    ax.set_xlim(0, 11)
    ax.set_ylim(65, 95)
    ax.grid(True, alpha=0.3)
    
    # Add arrows and text
    ax.annotate('', xy=(8, 72), xytext=(3, 90),
                arrowprops=dict(arrowstyle='->', lw=2, color='gray', alpha=0.5))
    ax.text(5.5, 82, 'Компромисс', fontsize=10, color='gray', 
           rotation=-35, ha='center', style='italic')
    
    return fig_to_base64(fig)

# ============================================================================
# SHAP ILLUSTRATIONS
# ============================================================================

def generate_shap_waterfall():
    """Generate SHAP waterfall plot visualization."""
    np.random.seed(42)
    
    # Simulated SHAP values for one prediction
    feature_names = ['Age', 'Income', 'Credit\nScore', 'Debt', 'Employment\nYears']
    shap_values = np.array([0.15, 0.32, -0.18, -0.25, 0.08])
    base_value = 0.5
    
    # Calculate cumulative values
    cumsum = np.concatenate([[base_value], base_value + np.cumsum(shap_values)])
    prediction = cumsum[-1]
    
    fig, ax = plt.subplots(figsize=(9, 7))
    
    # Plot waterfall
    colors = ['#ff6b6b' if v < 0 else '#51cf66' for v in shap_values]
    
    positions = range(len(shap_values) + 2)
    ax.bar(0, base_value, color='lightgray', edgecolor='black', linewidth=1)
    
    for i, (val, color) in enumerate(zip(shap_values, colors)):
        start = cumsum[i]
        ax.bar(i + 1, abs(val), bottom=min(start, start + val), 
               color=color, edgecolor='black', linewidth=0.8, alpha=0.8)
        
        # Add connector lines
        if i < len(shap_values) - 1:
            next_start = cumsum[i + 1]
            ax.plot([i + 1.4, i + 1.6], [start + val, next_start], 
                   'k--', linewidth=1, alpha=0.5)
    
    ax.bar(len(shap_values) + 1, prediction, color='steelblue', 
           edgecolor='black', linewidth=1.5, alpha=0.9)
    
    # Labels
    labels = ['Base\nValue'] + feature_names + ['Final\nPrediction']
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=0, ha='center')
    ax.set_ylabel('Значение предсказания')
    ax.set_title('SHAP Waterfall Plot: Объяснение предсказания', fontweight='bold')
    ax.axhline(y=base_value, color='gray', linestyle='--', alpha=0.3)
    ax.grid(True, alpha=0.2, axis='y')
    
    # Add value annotations
    for i, val in enumerate([base_value] + list(shap_values) + [prediction]):
        if i == 0 or i == len(positions) - 1:
            ax.text(positions[i], val + 0.02, f'{val:.2f}', 
                   ha='center', va='bottom', fontsize=9, fontweight='bold')
        else:
            actual_shap = shap_values[i - 1]
            ax.text(positions[i], cumsum[i] + actual_shap/2, 
                   f'{actual_shap:+.2f}', ha='center', va='center', 
                   fontsize=8, fontweight='bold', color='white')
    
    return fig_to_base64(fig)

def generate_shap_beeswarm():
    """Generate SHAP beeswarm plot visualization."""
    np.random.seed(42)
    
    # Simulated SHAP values for multiple predictions
    n_samples = 200
    feature_names = ['Age', 'Income', 'Credit Score', 'Debt Ratio', 'Employment Years']
    
    fig, ax = plt.subplots(figsize=(9, 6))
    
    for i, feature in enumerate(feature_names):
        # Generate SHAP values with different distributions
        if i == 1:  # Income - strongest positive effect
            shap_vals = np.random.normal(0.2, 0.15, n_samples)
            feature_vals = np.random.normal(60, 20, n_samples)
        elif i == 3:  # Debt - strong negative effect
            shap_vals = np.random.normal(-0.15, 0.12, n_samples)
            feature_vals = np.random.normal(40, 15, n_samples)
        else:
            shap_vals = np.random.normal(0, 0.1, n_samples)
            feature_vals = np.random.normal(50, 20, n_samples)
        
        # Normalize feature values for coloring
        feature_vals_norm = (feature_vals - feature_vals.min()) / (feature_vals.max() - feature_vals.min())
        
        # Add jitter to y-axis
        y_pos = np.ones(n_samples) * i + np.random.normal(0, 0.15, n_samples)
        
        scatter = ax.scatter(shap_vals, y_pos, c=feature_vals_norm, 
                           cmap='coolwarm', alpha=0.6, s=20, 
                           edgecolors='none')
    
    ax.axvline(x=0, color='gray', linestyle='--', linewidth=1.5, alpha=0.5)
    ax.set_yticks(range(len(feature_names)))
    ax.set_yticklabels(feature_names)
    ax.set_xlabel('SHAP value (влияние на предсказание)')
    ax.set_title('SHAP Beeswarm Plot: Сводка по всем признакам', fontweight='bold')
    ax.grid(True, alpha=0.2, axis='x')
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax, pad=0.02)
    cbar.set_label('Значение признака', rotation=270, labelpad=15)
    cbar.set_ticks([0, 0.5, 1])
    cbar.set_ticklabels(['Низкое', 'Среднее', 'Высокое'])
    
    return fig_to_base64(fig)

# ============================================================================
# LIME ILLUSTRATIONS
# ============================================================================

def generate_lime_explanation():
    """Generate LIME explanation visualization."""
    np.random.seed(42)
    
    # Simulated LIME weights for features
    feature_names = ['Age > 45', 'Income < 50K', 'Credit Score\n> 700', 
                     'Debt Ratio\n< 0.3', 'Employment\n> 5 years',
                     'Education:\nCollege', 'Location:\nUrban', 'Married:\nYes']
    weights = np.array([0.25, -0.32, 0.28, 0.15, 0.12, -0.08, 0.18, -0.05])
    
    # Sort by absolute weight
    indices = np.argsort(np.abs(weights))[::-1]
    sorted_features = [feature_names[i] for i in indices]
    sorted_weights = weights[indices]
    
    fig, ax = plt.subplots(figsize=(9, 7))
    
    colors = ['#51cf66' if w > 0 else '#ff6b6b' for w in sorted_weights]
    bars = ax.barh(range(len(sorted_weights)), sorted_weights, color=colors, 
                   edgecolor='black', linewidth=0.8, alpha=0.8)
    
    ax.set_yticks(range(len(sorted_features)))
    ax.set_yticklabels(sorted_features)
    ax.set_xlabel('Вес признака в LIME объяснении')
    ax.set_title('LIME: Локальное объяснение предсказания\n(Класс: Одобрено)', 
                fontweight='bold')
    ax.axvline(x=0, color='black', linewidth=1.5)
    ax.grid(True, alpha=0.3, axis='x')
    
    # Add value labels
    for i, (weight, bar) in enumerate(zip(sorted_weights, bars)):
        x_pos = weight + (0.02 if weight > 0 else -0.02)
        ha = 'left' if weight > 0 else 'right'
        ax.text(x_pos, i, f'{weight:+.2f}', va='center', ha=ha, 
               fontsize=9, fontweight='bold')
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#51cf66', edgecolor='black', label='Поддерживает класс'),
        Patch(facecolor='#ff6b6b', edgecolor='black', label='Против класса')
    ]
    ax.legend(handles=legend_elements, loc='lower right', framealpha=0.9)
    
    return fig_to_base64(fig)

def generate_lime_local_approximation():
    """Generate LIME local linear approximation visualization."""
    np.random.seed(42)
    
    # Create non-linear decision boundary
    x = np.linspace(-3, 3, 100)
    y_true = 1 / (1 + np.exp(-x**2 + 2))  # Non-linear function
    
    # Point to explain
    x_explain = 1.0
    y_explain = 1 / (1 + np.exp(-x_explain**2 + 2))
    
    # Local linear approximation (LIME)
    slope = 2 * x_explain * np.exp(-x_explain**2 + 2) / (1 + np.exp(-x_explain**2 + 2))**2
    y_lime = y_explain + slope * (x - x_explain)
    
    fig, ax = plt.subplots(figsize=(9, 6))
    
    # Plot true model
    ax.plot(x, y_true, 'b-', linewidth=2.5, label='Истинная модель (сложная)', alpha=0.8)
    
    # Plot LIME approximation
    mask = (x > x_explain - 0.8) & (x < x_explain + 0.8)
    ax.plot(x[mask], y_lime[mask], 'r--', linewidth=2.5, 
           label='LIME аппроксимация (линейная)', alpha=0.8)
    
    # Plot point being explained
    ax.scatter([x_explain], [y_explain], s=200, c='gold', 
              edgecolors='black', linewidth=2, zorder=5, 
              label='Объясняемая точка')
    
    # Shade local region
    ax.axvspan(x_explain - 0.8, x_explain + 0.8, alpha=0.15, color='yellow', 
              label='Локальная область')
    
    ax.set_xlabel('Значение признака')
    ax.set_ylabel('Предсказание модели')
    ax.set_title('LIME: Локальная линейная аппроксимация сложной модели', 
                fontweight='bold')
    ax.legend(loc='upper left', framealpha=0.95)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-3, 3)
    ax.set_ylim(-0.1, 1.1)
    
    return fig_to_base64(fig)

# ============================================================================
# FEATURE IMPORTANCE ILLUSTRATIONS
# ============================================================================

def generate_feature_importance_comparison():
    """Generate comparison of different feature importance methods."""
    np.random.seed(42)
    X, y = make_classification(n_samples=500, n_features=8, n_informative=5, 
                                n_redundant=2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Train model
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    
    # Get different importance measures
    feature_names = [f'Feature {i+1}' for i in range(8)]
    mdi_importance = rf.feature_importances_
    
    # Permutation importance
    perm_importance = permutation_importance(rf, X_test, y_test, n_repeats=10, random_state=42)
    perm_imp = perm_importance.importances_mean
    
    # Normalize for comparison
    mdi_norm = mdi_importance / mdi_importance.max()
    perm_norm = perm_imp / perm_imp.max()
    
    fig, ax = plt.subplots(figsize=(10, 7))
    
    x = np.arange(len(feature_names))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, mdi_norm, width, label='MDI (встроенная)', 
                   color='#3498db', edgecolor='black', linewidth=0.8, alpha=0.8)
    bars2 = ax.bar(x + width/2, perm_norm, width, label='Permutation', 
                   color='#e74c3c', edgecolor='black', linewidth=0.8, alpha=0.8)
    
    ax.set_xlabel('Признаки')
    ax.set_ylabel('Нормализованная важность')
    ax.set_title('Сравнение методов Feature Importance', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(feature_names, rotation=45, ha='right')
    ax.legend(loc='upper right', framealpha=0.95)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on top of bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                   f'{height:.2f}', ha='center', va='bottom', fontsize=7)
    
    plt.tight_layout()
    return fig_to_base64(fig)

def generate_feature_importance_drop():
    """Generate illustration showing importance via performance drop."""
    np.random.seed(42)
    
    feature_names = ['Feature A', 'Feature B', 'Feature C', 'Feature D', 'Feature E']
    baseline_score = 0.85
    
    # Simulated score drops when feature is removed
    score_drops = np.array([0.12, 0.18, 0.05, 0.09, 0.03])
    scores_without = baseline_score - score_drops
    
    fig, ax = plt.subplots(figsize=(9, 6))
    
    x = np.arange(len(feature_names))
    
    # Plot baseline
    ax.axhline(y=baseline_score, color='green', linestyle='--', linewidth=2, 
              label=f'Baseline (все признаки): {baseline_score:.2f}', alpha=0.7)
    
    # Plot scores without each feature
    colors = plt.cm.Reds(np.linspace(0.4, 0.9, len(feature_names)))
    bars = ax.bar(x, scores_without, color=colors, edgecolor='black', 
                  linewidth=0.8, alpha=0.8)
    
    # Add arrows showing drop
    for i, (score, drop) in enumerate(zip(scores_without, score_drops)):
        ax.annotate('', xy=(i, score), xytext=(i, baseline_score),
                   arrowprops=dict(arrowstyle='->', lw=2, color='red', alpha=0.6))
        # Add drop value
        ax.text(i, score + drop/2, f'-{drop:.2f}', ha='center', va='center',
               fontsize=9, fontweight='bold', color='darkred')
    
    ax.set_xlabel('Признак удалён')
    ax.set_ylabel('Точность модели')
    ax.set_title('Feature Importance: Падение качества при удалении признака', 
                fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(feature_names)
    ax.set_ylim(0.6, 0.9)
    ax.legend(loc='lower right', framealpha=0.95)
    ax.grid(True, alpha=0.3, axis='y')
    
    return fig_to_base64(fig)

# ============================================================================
# PARTIAL DEPENDENCE ILLUSTRATIONS
# ============================================================================

def generate_pdp_1d():
    """Generate 1D Partial Dependence Plot."""
    np.random.seed(42)
    X, y = make_classification(n_samples=1000, n_features=5, n_informative=3,
                                n_redundant=1, random_state=42)
    
    # Train model
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X, y)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot PDP for first two features
    for idx, ax in enumerate(axes):
        display = PartialDependenceDisplay.from_estimator(
            rf, X, [idx], ax=ax, grid_resolution=50
        )
        ax.set_title(f'PDP: Feature {idx+1}', fontweight='bold')
        ax.set_xlabel(f'Feature {idx+1}')
        ax.set_ylabel('Частичная зависимость')
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Partial Dependence Plots (1D)', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    return fig_to_base64(fig)

def generate_pdp_2d():
    """Generate 2D Partial Dependence Plot."""
    np.random.seed(42)
    X, y = make_classification(n_samples=1000, n_features=5, n_informative=3,
                                n_redundant=1, random_state=42)
    
    # Train model
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X, y)
    
    fig, ax = plt.subplots(figsize=(9, 7))
    
    # 2D PDP
    display = PartialDependenceDisplay.from_estimator(
        rf, X, [(0, 1)], ax=ax, grid_resolution=30
    )
    
    ax.set_title('2D Partial Dependence Plot: Взаимодействие признаков', 
                fontweight='bold', fontsize=12)
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    
    return fig_to_base64(fig)

def generate_ice_plots():
    """Generate Individual Conditional Expectation (ICE) plots."""
    np.random.seed(42)
    
    # Generate synthetic data with clear patterns
    n_samples = 100
    x = np.linspace(-3, 3, n_samples)
    
    fig, ax = plt.subplots(figsize=(9, 6))
    
    # Generate ICE curves - individual predictions as function of feature
    for i in range(30):
        # Each curve represents one instance
        offset = np.random.normal(0, 0.3)
        noise = np.random.normal(0, 0.05, n_samples)
        y_ice = 1 / (1 + np.exp(-1.5 * x + offset)) + noise
        ax.plot(x, y_ice, alpha=0.3, linewidth=1, color='steelblue')
    
    # Add PDP (average)
    y_pdp = 1 / (1 + np.exp(-1.5 * x))
    ax.plot(x, y_pdp, color='red', linewidth=3, label='PDP (среднее)', alpha=0.9)
    
    ax.set_xlabel('Значение признака')
    ax.set_ylabel('Предсказание модели')
    ax.set_title('ICE Plots: Индивидуальные условные ожидания\n(каждая линия = один объект)', 
                fontweight='bold')
    ax.legend(loc='upper left', framealpha=0.95)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.1, 1.1)
    
    # Add annotation
    ax.text(0.5, 0.95, 'ICE показывает гетерогенность эффектов', 
           transform=ax.transAxes, ha='center', va='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
           fontsize=9, style='italic')
    
    return fig_to_base64(fig)

# ============================================================================
# INTEGRATED GRADIENTS ILLUSTRATIONS
# ============================================================================

def generate_integrated_gradients_path():
    """Generate illustration of integrated gradients interpolation path."""
    np.random.seed(42)
    
    # Simulated baseline and input images (represented as 1D)
    n_features = 20
    baseline = np.zeros(n_features)
    input_image = np.random.rand(n_features)
    
    # Create interpolation path
    n_steps = 5
    alphas = np.linspace(0, 1, n_steps)
    
    fig, axes = plt.subplots(1, n_steps, figsize=(12, 3))
    
    for i, (alpha, ax) in enumerate(zip(alphas, axes)):
        # Interpolated image
        interpolated = baseline + alpha * (input_image - baseline)
        
        # Display as image
        ax.imshow(interpolated.reshape(4, 5), cmap='viridis', aspect='auto', 
                 vmin=0, vmax=1)
        ax.set_title(f'α = {alpha:.2f}', fontsize=10, fontweight='bold')
        ax.axis('off')
        
        # Add arrow between steps
        if i < n_steps - 1:
            # This is just for visual connection
            pass
    
    # Add arrows between subplots
    fig.text(0.19, 0.5, '→', ha='center', va='center', fontsize=20, color='red')
    fig.text(0.37, 0.5, '→', ha='center', va='center', fontsize=20, color='red')
    fig.text(0.55, 0.5, '→', ha='center', va='center', fontsize=20, color='red')
    fig.text(0.73, 0.5, '→', ha='center', va='center', fontsize=20, color='red')
    
    fig.suptitle('Integrated Gradients: Путь интерполяции от baseline к input', 
                fontsize=12, fontweight='bold', y=0.98)
    
    # Add labels
    fig.text(0.1, 0.05, 'Baseline\n(нулевое изображение)', ha='center', fontsize=9)
    fig.text(0.9, 0.05, 'Input\n(исходное изображение)', ha='center', fontsize=9)
    
    plt.tight_layout(rect=[0, 0.08, 1, 0.95])
    
    return fig_to_base64(fig)

def generate_integrated_gradients_attribution():
    """Generate attribution heatmap for integrated gradients."""
    np.random.seed(42)
    
    # Create a simulated attribution map (e.g., for an image)
    height, width = 8, 8
    
    # Simulate attribution with some pattern (center more important)
    y, x = np.ogrid[-height/2:height/2, -width/2:width/2]
    mask = np.exp(-(x**2 + y**2) / (2 * 2**2))
    
    # Add some noise and features
    attribution = mask + np.random.randn(height, width) * 0.1
    attribution = np.clip(attribution, -1, 1)
    
    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    
    # Original "image" (random pattern)
    original = np.random.rand(height, width)
    axes[0].imshow(original, cmap='gray', aspect='auto')
    axes[0].set_title('Исходное изображение', fontweight='bold')
    axes[0].axis('off')
    
    # Attribution heatmap
    im = axes[1].imshow(attribution, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
    axes[1].set_title('Integrated Gradients Attribution', fontweight='bold')
    axes[1].axis('off')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    cbar.set_label('Вклад в предсказание', rotation=270, labelpad=15)
    cbar.set_ticks([-1, 0, 1])
    cbar.set_ticklabels(['Отрицательный', 'Нейтральный', 'Положительный'])
    
    plt.suptitle('Integrated Gradients: Визуализация важности пикселей', 
                fontsize=13, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    
    return fig_to_base64(fig)

# ============================================================================
# MAIN FUNCTION
# ============================================================================

def generate_all_illustrations():
    """Generate all interpretability illustrations and return as dictionary."""
    print("Generating interpretability illustrations...")
    
    illustrations = {}
    
    # XAI illustrations
    print("  - XAI feature importance...")
    illustrations['xai_feature_importance'] = generate_xai_feature_importance()
    
    print("  - XAI model comparison...")
    illustrations['xai_model_comparison'] = generate_xai_model_comparison()
    
    # SHAP illustrations
    print("  - SHAP waterfall plot...")
    illustrations['shap_waterfall'] = generate_shap_waterfall()
    
    print("  - SHAP beeswarm plot...")
    illustrations['shap_beeswarm'] = generate_shap_beeswarm()
    
    # LIME illustrations
    print("  - LIME explanation...")
    illustrations['lime_explanation'] = generate_lime_explanation()
    
    print("  - LIME local approximation...")
    illustrations['lime_local_approximation'] = generate_lime_local_approximation()
    
    # Feature Importance illustrations
    print("  - Feature importance comparison...")
    illustrations['feature_importance_comparison'] = generate_feature_importance_comparison()
    
    print("  - Feature importance drop...")
    illustrations['feature_importance_drop'] = generate_feature_importance_drop()
    
    # Partial Dependence illustrations
    print("  - PDP 1D...")
    illustrations['pdp_1d'] = generate_pdp_1d()
    
    print("  - PDP 2D...")
    illustrations['pdp_2d'] = generate_pdp_2d()
    
    print("  - ICE plots...")
    illustrations['ice_plots'] = generate_ice_plots()
    
    # Integrated Gradients illustrations
    print("  - Integrated gradients path...")
    illustrations['integrated_gradients_path'] = generate_integrated_gradients_path()
    
    print("  - Integrated gradients attribution...")
    illustrations['integrated_gradients_attribution'] = generate_integrated_gradients_attribution()
    
    print(f"✓ Generated {len(illustrations)} illustrations")
    
    return illustrations

if __name__ == '__main__':
    illustrations = generate_all_illustrations()
    print("\nIllustrations ready for embedding in HTML files.")
