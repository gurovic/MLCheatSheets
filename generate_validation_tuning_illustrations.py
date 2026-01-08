#!/usr/bin/env python3
"""
Generate matplotlib illustrations for validation and tuning cheatsheets.
This script creates high-quality visualizations and encodes them as base64 for inline HTML embedding.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import (
    KFold, StratifiedKFold, TimeSeriesSplit, 
    GridSearchCV, RandomizedSearchCV, cross_val_score
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from sklearn.svm import SVC
from sklearn.metrics import brier_score_loss
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
# CROSS VALIDATION ILLUSTRATIONS
# ============================================================================

def generate_kfold_diagram():
    """Generate K-Fold cross-validation visualization."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    n_samples = 50
    n_splits = 5
    
    kf = KFold(n_splits=n_splits, shuffle=False)
    
    for i, (train_idx, val_idx) in enumerate(kf.split(range(n_samples))):
        # Plot training data
        ax.barh(i, len(train_idx), left=0, height=0.8, 
                color='#3498db', alpha=0.7, label='Train' if i == 0 else '')
        
        # Plot validation data
        val_start = train_idx[-1] + 1 if len(train_idx) > 0 else 0
        ax.barh(i, len(val_idx), left=val_start, height=0.8,
                color='#e74c3c', alpha=0.7, label='Validation' if i == 0 else '')
    
    ax.set_yticks(range(n_splits))
    ax.set_yticklabels([f'Fold {i+1}' for i in range(n_splits)])
    ax.set_xlabel('Индекс образца')
    ax.set_title('K-Fold Cross-Validation (K=5)')
    ax.legend(loc='upper right')
    ax.set_xlim(0, n_samples)
    
    return fig_to_base64(fig)

def generate_timeseries_split_diagram():
    """Generate Time Series Split visualization."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    n_samples = 50
    n_splits = 5
    
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    for i, (train_idx, test_idx) in enumerate(tscv.split(range(n_samples))):
        # Plot training data
        ax.barh(i, len(train_idx), left=0, height=0.8,
                color='#2ecc71', alpha=0.7, label='Train' if i == 0 else '')
        
        # Plot test data
        test_start = train_idx[-1] + 1 if len(train_idx) > 0 else 0
        ax.barh(i, len(test_idx), left=test_start, height=0.8,
                color='#f39c12', alpha=0.7, label='Test' if i == 0 else '')
    
    ax.set_yticks(range(n_splits))
    ax.set_yticklabels([f'Split {i+1}' for i in range(n_splits)])
    ax.set_xlabel('Временная метка (индекс)')
    ax.set_title('Time Series Split - Сохранение временного порядка')
    ax.legend(loc='upper left')
    ax.set_xlim(0, n_samples)
    ax.annotate('Время →', xy=(0.5, -0.15), xycoords='axes fraction',
                ha='center', fontsize=11, style='italic')
    
    return fig_to_base64(fig)

def generate_nested_cv_diagram():
    """Generate Nested Cross-Validation diagram."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Outer CV visualization
    n_outer = 5
    for i in range(n_outer):
        ax.add_patch(plt.Rectangle((0, i*2), 10, 0.8, 
                                   facecolor='#3498db', alpha=0.5, 
                                   edgecolor='black', linewidth=2))
        ax.text(5, i*2 + 0.4, f'Outer Fold {i+1}', 
                ha='center', va='center', fontsize=10, weight='bold')
        
        # Inner CV visualization (smaller)
        inner_width = 7
        for j in range(3):
            ax.add_patch(plt.Rectangle((0.5 + j*2.3, i*2 + 1), inner_width/3, 0.6,
                                       facecolor='#e74c3c', alpha=0.6,
                                       edgecolor='gray', linewidth=1))
            if i == 0:
                ax.text(0.5 + j*2.3 + inner_width/6, i*2 + 1.3, 
                       f'Inner\nCV {j+1}', ha='center', va='center', 
                       fontsize=8)
    
    ax.set_xlim(0, 10)
    ax.set_ylim(0, n_outer*2)
    ax.set_title('Nested Cross-Validation\nВнешний CV для оценки, Внутренний CV для подбора гиперпараметров', 
                fontsize=12, weight='bold')
    ax.axis('off')
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#3498db', alpha=0.5, edgecolor='black', label='Outer CV (оценка модели)'),
        Patch(facecolor='#e74c3c', alpha=0.6, edgecolor='gray', label='Inner CV (подбор гиперпараметров)')
    ]
    ax.legend(handles=legend_elements, loc='upper center', 
             bbox_to_anchor=(0.5, -0.05), ncol=2)
    
    return fig_to_base64(fig)

def generate_cv_scores_comparison():
    """Generate comparison of CV scores across different folds."""
    np.random.seed(42)
    X, y = make_classification(n_samples=500, n_features=20, n_informative=15,
                              n_redundant=5, random_state=42)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    
    # K-Fold CV
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    scores_kf = cross_val_score(model, X, y, cv=kf, scoring='accuracy')
    
    # Stratified K-Fold CV
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores_skf = cross_val_score(model, X, y, cv=skf, scoring='accuracy')
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    folds = np.arange(1, 6)
    width = 0.35
    
    ax.bar(folds - width/2, scores_kf, width, label='K-Fold', 
           color='#3498db', alpha=0.8)
    ax.bar(folds + width/2, scores_skf, width, label='Stratified K-Fold',
           color='#e74c3c', alpha=0.8)
    
    # Add mean lines
    ax.axhline(scores_kf.mean(), color='#3498db', linestyle='--', 
              linewidth=2, alpha=0.7, label=f'K-Fold mean: {scores_kf.mean():.3f}')
    ax.axhline(scores_skf.mean(), color='#e74c3c', linestyle='--',
              linewidth=2, alpha=0.7, label=f'Stratified mean: {scores_skf.mean():.3f}')
    
    ax.set_xlabel('Номер фолда')
    ax.set_ylabel('Accuracy')
    ax.set_title('Сравнение K-Fold и Stratified K-Fold Cross-Validation')
    ax.set_xticks(folds)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    return fig_to_base64(fig)

# ============================================================================
# HYPERPARAMETER TUNING ILLUSTRATIONS
# ============================================================================

def generate_grid_search_heatmap():
    """Generate Grid Search results heatmap."""
    np.random.seed(42)
    X, y = make_classification(n_samples=300, n_features=20, n_informative=15,
                              random_state=42)
    
    param_grid = {
        'n_estimators': [50, 100, 150, 200],
        'max_depth': [5, 10, 15, 20, None]
    }
    
    model = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(model, param_grid, cv=3, scoring='accuracy')
    grid_search.fit(X, y)
    
    # Create heatmap data
    results = grid_search.cv_results_
    scores = results['mean_test_score'].reshape(5, 4)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    im = ax.imshow(scores, cmap='YlOrRd', aspect='auto')
    
    # Set ticks
    ax.set_xticks(np.arange(len(param_grid['n_estimators'])))
    ax.set_yticks(np.arange(len(param_grid['max_depth'])))
    ax.set_xticklabels(param_grid['n_estimators'])
    ax.set_yticklabels([str(d) if d is not None else 'None' 
                       for d in param_grid['max_depth']])
    
    # Labels
    ax.set_xlabel('n_estimators', fontsize=12)
    ax.set_ylabel('max_depth', fontsize=12)
    ax.set_title('Grid Search: Mean CV Score (Accuracy)', fontsize=13, weight='bold')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Accuracy', rotation=270, labelpad=15)
    
    # Add text annotations
    for i in range(len(param_grid['max_depth'])):
        for j in range(len(param_grid['n_estimators'])):
            text = ax.text(j, i, f'{scores[i, j]:.3f}',
                         ha="center", va="center", color="black", fontsize=9)
    
    # Highlight best
    best_idx = np.unravel_index(np.argmax(scores), scores.shape)
    ax.add_patch(plt.Rectangle((best_idx[1]-0.5, best_idx[0]-0.5), 1, 1,
                               fill=False, edgecolor='lime', linewidth=3))
    
    return fig_to_base64(fig)

def generate_random_vs_grid_search():
    """Generate comparison of Random Search vs Grid Search."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Grid Search visualization
    n_params = 10
    x = np.linspace(0, 1, n_params)
    y = np.linspace(0, 1, n_params)
    xx, yy = np.meshgrid(x, y)
    
    ax1.scatter(xx, yy, c='blue', s=100, alpha=0.6, edgecolors='black', linewidth=1)
    ax1.set_xlim(-0.1, 1.1)
    ax1.set_ylim(-0.1, 1.1)
    ax1.set_xlabel('Параметр 1', fontsize=11)
    ax1.set_ylabel('Параметр 2', fontsize=11)
    ax1.set_title(f'Grid Search\n{n_params*n_params} комбинаций', fontsize=12, weight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Random Search visualization
    np.random.seed(42)
    n_random = 50
    x_random = np.random.uniform(0, 1, n_random)
    y_random = np.random.uniform(0, 1, n_random)
    
    ax2.scatter(x_random, y_random, c='red', s=100, alpha=0.6, 
               edgecolors='black', linewidth=1)
    ax2.set_xlim(-0.1, 1.1)
    ax2.set_ylim(-0.1, 1.1)
    ax2.set_xlabel('Параметр 1', fontsize=11)
    ax2.set_ylabel('Параметр 2', fontsize=11)
    ax2.set_title(f'Random Search\n{n_random} комбинаций', fontsize=12, weight='bold')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    return fig_to_base64(fig)

def generate_optimization_history():
    """Generate Bayesian Optimization history."""
    np.random.seed(42)
    
    # Simulate optimization history
    n_trials = 50
    trials = np.arange(1, n_trials + 1)
    
    # Random search baseline
    random_scores = np.random.uniform(0.7, 0.85, n_trials)
    random_best = np.maximum.accumulate(random_scores)
    
    # Bayesian optimization (improving over time)
    bayesian_scores = 0.7 + 0.15 * (1 - np.exp(-trials / 15)) + np.random.normal(0, 0.02, n_trials)
    bayesian_scores = np.clip(bayesian_scores, 0.7, 0.88)
    bayesian_best = np.maximum.accumulate(bayesian_scores)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    
    # Trial scores
    ax1.scatter(trials, random_scores, alpha=0.5, s=30, c='blue', label='Random Search')
    ax1.scatter(trials, bayesian_scores, alpha=0.5, s=30, c='red', label='Bayesian Optimization')
    ax1.set_xlabel('Номер итерации')
    ax1.set_ylabel('Score')
    ax1.set_title('Scores каждой итерации')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Best score over time
    ax2.plot(trials, random_best, 'b-', linewidth=2, label='Random Search (лучший)')
    ax2.plot(trials, bayesian_best, 'r-', linewidth=2, label='Bayesian Optimization (лучший)')
    ax2.fill_between(trials, random_best, alpha=0.2, color='blue')
    ax2.fill_between(trials, bayesian_best, alpha=0.2, color='red')
    ax2.set_xlabel('Номер итерации')
    ax2.set_ylabel('Best Score')
    ax2.set_title('Эволюция лучшего score')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    return fig_to_base64(fig)

def generate_hyperparameter_importance():
    """Generate hyperparameter importance visualization."""
    # Simulated importance scores
    params = ['max_depth', 'n_estimators', 'min_samples_split', 
             'min_samples_leaf', 'max_features']
    importance = np.array([0.35, 0.28, 0.18, 0.12, 0.07])
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = plt.cm.RdYlGn_r(importance / importance.max())
    bars = ax.barh(params, importance, color=colors, alpha=0.8, edgecolor='black')
    
    # Add values
    for i, (bar, val) in enumerate(zip(bars, importance)):
        ax.text(val + 0.01, i, f'{val:.2f}', va='center', fontsize=10, weight='bold')
    
    ax.set_xlabel('Важность гиперпараметра', fontsize=12)
    ax.set_title('Влияние гиперпараметров на качество модели', fontsize=13, weight='bold')
    ax.set_xlim(0, max(importance) * 1.15)
    ax.grid(axis='x', alpha=0.3)
    
    return fig_to_base64(fig)

# ============================================================================
# MODEL CALIBRATION ILLUSTRATIONS
# ============================================================================

def generate_calibration_curve():
    """Generate calibration curve comparison."""
    np.random.seed(42)
    X, y = make_classification(n_samples=1000, n_features=20, n_informative=15,
                              n_classes=2, random_state=42)
    
    X_train, X_test = X[:700], X[700:]
    y_train, y_test = y[:700], y[700:]
    
    # Uncalibrated model (SVM)
    svm = SVC(probability=True, random_state=42)
    svm.fit(X_train, y_train)
    
    # Calibrated model
    calibrated = CalibratedClassifierCV(svm, method='isotonic', cv=5)
    calibrated.fit(X_train, y_train)
    
    # Get probabilities
    prob_svm = svm.predict_proba(X_test)[:, 1]
    prob_calibrated = calibrated.predict_proba(X_test)[:, 1]
    
    # Calibration curves
    fraction_of_positives_svm, mean_predicted_value_svm = \
        calibration_curve(y_test, prob_svm, n_bins=10)
    fraction_of_positives_cal, mean_predicted_value_cal = \
        calibration_curve(y_test, prob_calibrated, n_bins=10)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Perfect calibration line
    ax.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Идеальная калибровка')
    
    # Before calibration
    ax.plot(mean_predicted_value_svm, fraction_of_positives_svm, 
           's-', linewidth=2, markersize=8, color='#e74c3c',
           label=f'До калибровки (Brier={brier_score_loss(y_test, prob_svm):.3f})')
    
    # After calibration
    ax.plot(mean_predicted_value_cal, fraction_of_positives_cal,
           'o-', linewidth=2, markersize=8, color='#2ecc71',
           label=f'После калибровки (Brier={brier_score_loss(y_test, prob_calibrated):.3f})')
    
    ax.set_xlabel('Предсказанная вероятность', fontsize=12)
    ax.set_ylabel('Истинная доля положительных', fontsize=12)
    ax.set_title('Calibration Curve - Надёжность вероятностей', fontsize=13, weight='bold')
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    
    return fig_to_base64(fig)

def generate_reliability_diagram():
    """Generate reliability diagram with histogram."""
    np.random.seed(42)
    X, y = make_classification(n_samples=1000, n_features=20, n_informative=15,
                              n_classes=2, random_state=42)
    
    X_train, X_test = X[:700], X[700:]
    y_train, y_test = y[:700], y[700:]
    
    # Train uncalibrated and calibrated models
    svm = SVC(probability=True, random_state=42)
    svm.fit(X_train, y_train)
    
    calibrated = CalibratedClassifierCV(svm, method='sigmoid', cv=5)
    calibrated.fit(X_train, y_train)
    
    prob_svm = svm.predict_proba(X_test)[:, 1]
    prob_calibrated = calibrated.predict_proba(X_test)[:, 1]
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Calibration curves
    for idx, (probs, name, color) in enumerate([
        (prob_svm, 'До калибровки (SVM)', '#e74c3c'),
        (prob_calibrated, 'После калибровки', '#2ecc71')
    ]):
        ax_calib = axes[idx, 0]
        ax_hist = axes[idx, 1]
        
        # Calibration curve
        fraction_of_positives, mean_predicted_value = \
            calibration_curve(y_test, probs, n_bins=10)
        
        ax_calib.plot([0, 1], [0, 1], 'k--', linewidth=2, alpha=0.5)
        ax_calib.plot(mean_predicted_value, fraction_of_positives,
                     's-', linewidth=2, markersize=8, color=color)
        ax_calib.set_xlabel('Предсказанная вероятность', fontsize=11)
        ax_calib.set_ylabel('Истинная доля', fontsize=11)
        ax_calib.set_title(f'{name}\nBrier Score: {brier_score_loss(y_test, probs):.4f}',
                          fontsize=11, weight='bold')
        ax_calib.grid(True, alpha=0.3)
        ax_calib.set_xlim([0, 1])
        ax_calib.set_ylim([0, 1])
        
        # Histogram of predicted probabilities
        ax_hist.hist(probs[y_test == 0], bins=20, alpha=0.5, color='blue',
                    label='Класс 0', range=(0, 1))
        ax_hist.hist(probs[y_test == 1], bins=20, alpha=0.5, color='red',
                    label='Класс 1', range=(0, 1))
        ax_hist.set_xlabel('Предсказанная вероятность', fontsize=11)
        ax_hist.set_ylabel('Частота', fontsize=11)
        ax_hist.set_title('Распределение вероятностей', fontsize=11, weight='bold')
        ax_hist.legend()
        ax_hist.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    return fig_to_base64(fig)

def generate_calibration_methods_comparison():
    """Generate comparison of calibration methods."""
    np.random.seed(42)
    X, y = make_classification(n_samples=1000, n_features=20, n_informative=15,
                              n_classes=2, random_state=42)
    
    X_train, X_test = X[:700], X[700:]
    y_train, y_test = y[:700], y[700:]
    
    # Base model
    svm = SVC(probability=True, random_state=42)
    svm.fit(X_train, y_train)
    
    # Different calibration methods
    calibrated_sigmoid = CalibratedClassifierCV(svm, method='sigmoid', cv=5)
    calibrated_sigmoid.fit(X_train, y_train)
    
    calibrated_isotonic = CalibratedClassifierCV(svm, method='isotonic', cv=5)
    calibrated_isotonic.fit(X_train, y_train)
    
    # Get probabilities
    prob_base = svm.predict_proba(X_test)[:, 1]
    prob_sigmoid = calibrated_sigmoid.predict_proba(X_test)[:, 1]
    prob_isotonic = calibrated_isotonic.predict_proba(X_test)[:, 1]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Perfect calibration
    ax.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Идеальная калибровка', alpha=0.7)
    
    # Plot all methods
    for probs, name, color, marker in [
        (prob_base, 'Без калибровки', '#e74c3c', 's'),
        (prob_sigmoid, 'Sigmoid (Platt Scaling)', '#f39c12', 'o'),
        (prob_isotonic, 'Isotonic Regression', '#2ecc71', '^')
    ]:
        fraction_of_positives, mean_predicted_value = \
            calibration_curve(y_test, probs, n_bins=10)
        brier = brier_score_loss(y_test, probs)
        
        ax.plot(mean_predicted_value, fraction_of_positives,
               marker=marker, linestyle='-', linewidth=2, markersize=8,
               color=color, label=f'{name} (Brier={brier:.3f})')
    
    ax.set_xlabel('Предсказанная вероятность', fontsize=12)
    ax.set_ylabel('Истинная доля положительных', fontsize=12)
    ax.set_title('Сравнение методов калибровки', fontsize=13, weight='bold')
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    
    return fig_to_base64(fig)

def generate_brier_score_comparison():
    """Generate Brier Score comparison across models."""
    # Simulated Brier scores for different models
    models = ['Logistic\nRegression', 'Random\nForest', 'SVM', 'Gradient\nBoosting', 
             'Neural\nNetwork']
    
    brier_before = np.array([0.15, 0.22, 0.28, 0.20, 0.25])
    brier_after = np.array([0.14, 0.16, 0.17, 0.16, 0.18])
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(models))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, brier_before, width, label='До калибровки',
                   color='#e74c3c', alpha=0.8, edgecolor='black')
    bars2 = ax.bar(x + width/2, brier_after, width, label='После калибровки',
                   color='#2ecc71', alpha=0.8, edgecolor='black')
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    ax.set_xlabel('Модель', fontsize=12)
    ax.set_ylabel('Brier Score (меньше = лучше)', fontsize=12)
    ax.set_title('Влияние калибровки на Brier Score', fontsize=13, weight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, max(brier_before) * 1.2)
    
    return fig_to_base64(fig)

# ============================================================================
# MAIN GENERATION FUNCTION
# ============================================================================

def generate_all_illustrations():
    """Generate all validation and tuning illustrations."""
    print("Generating validation and tuning illustrations...")
    
    illustrations = {}
    
    # Cross Validation
    print("  - K-Fold diagram...")
    illustrations['cv_kfold_diagram'] = generate_kfold_diagram()
    
    print("  - Time Series Split diagram...")
    illustrations['cv_timeseries_split'] = generate_timeseries_split_diagram()
    
    print("  - Nested CV diagram...")
    illustrations['cv_nested_diagram'] = generate_nested_cv_diagram()
    
    print("  - CV scores comparison...")
    illustrations['cv_scores_comparison'] = generate_cv_scores_comparison()
    
    # Hyperparameter Tuning
    print("  - Grid Search heatmap...")
    illustrations['tuning_grid_heatmap'] = generate_grid_search_heatmap()
    
    print("  - Random vs Grid Search...")
    illustrations['tuning_random_vs_grid'] = generate_random_vs_grid_search()
    
    print("  - Optimization history...")
    illustrations['tuning_optimization_history'] = generate_optimization_history()
    
    print("  - Hyperparameter importance...")
    illustrations['tuning_param_importance'] = generate_hyperparameter_importance()
    
    # Model Calibration
    print("  - Calibration curve...")
    illustrations['calibration_curve'] = generate_calibration_curve()
    
    print("  - Reliability diagram...")
    illustrations['calibration_reliability'] = generate_reliability_diagram()
    
    print("  - Calibration methods comparison...")
    illustrations['calibration_methods_comparison'] = generate_calibration_methods_comparison()
    
    print("  - Brier Score comparison...")
    illustrations['calibration_brier_comparison'] = generate_brier_score_comparison()
    
    print(f"Generated {len(illustrations)} illustrations!")
    
    return illustrations

if __name__ == "__main__":
    illustrations = generate_all_illustrations()
    print("\nIllustrations generated successfully!")
    print("Use add_validation_tuning_illustrations.py to embed them into HTML files.")
