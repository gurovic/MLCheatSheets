#!/usr/bin/env python3
"""
Generate matplotlib illustrations for metrics and evaluation cheatsheets.
This script creates high-quality visualizations and encodes them as base64 for inline HTML embedding.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split, learning_curve, validation_curve
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc, roc_auc_score,
    precision_recall_curve, classification_report,
    mean_squared_error, mean_absolute_error, r2_score
)
from sklearn.preprocessing import label_binarize
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
# CONFUSION MATRIX ILLUSTRATIONS
# ============================================================================

def generate_confusion_matrix_binary():
    """Generate binary confusion matrix visualization."""
    # Create sample data
    y_true = np.array([0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1])
    y_pred = np.array([0, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1])
    
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Negative (0)', 'Positive (1)'],
                yticklabels=['Negative (0)', 'Positive (1)'],
                cbar_kws={'label': 'Количество'}, ax=ax)
    ax.set_ylabel('Actual (Истинный класс)')
    ax.set_xlabel('Predicted (Предсказанный класс)')
    ax.set_title('Матрица ошибок (Confusion Matrix)\nБинарная классификация')
    
    # Add TN, FP, FN, TP labels
    tn, fp, fn, tp = cm.ravel()
    ax.text(0.5, 0.25, f'TN={tn}', ha='center', va='center', 
            fontsize=11, color='darkblue', weight='bold')
    ax.text(1.5, 0.25, f'FP={fp}', ha='center', va='center', 
            fontsize=11, color='darkred', weight='bold')
    ax.text(0.5, 1.25, f'FN={fn}', ha='center', va='center', 
            fontsize=11, color='darkred', weight='bold')
    ax.text(1.5, 1.25, f'TP={tp}', ha='center', va='center', 
            fontsize=11, color='darkblue', weight='bold')
    
    plt.tight_layout()
    return fig_to_base64(fig)

def generate_confusion_matrix_multiclass():
    """Generate multiclass confusion matrix visualization."""
    # Create sample multiclass data
    np.random.seed(42)
    y_true = np.random.randint(0, 3, 50)
    y_pred = y_true.copy()
    # Add some errors
    error_indices = np.random.choice(50, 15, replace=False)
    y_pred[error_indices] = np.random.randint(0, 3, 15)
    
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Класс 0', 'Класс 1', 'Класс 2'],
                yticklabels=['Класс 0', 'Класс 1', 'Класс 2'],
                cbar_kws={'label': 'Количество'}, ax=ax)
    ax.set_ylabel('Actual (Истинный класс)')
    ax.set_xlabel('Predicted (Предсказанный класс)')
    ax.set_title('Матрица ошибок: Мультиклассовая классификация')
    plt.tight_layout()
    return fig_to_base64(fig)

def generate_confusion_matrix_normalized():
    """Generate normalized confusion matrix."""
    y_true = np.array([0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1])
    y_pred = np.array([0, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1])
    
    cm = confusion_matrix(y_true, y_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues',
                xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive'],
                cbar_kws={'label': 'Доля'}, ax=ax)
    ax.set_ylabel('Actual')
    ax.set_xlabel('Predicted')
    ax.set_title('Нормализованная матрица ошибок\n(по строкам)')
    plt.tight_layout()
    return fig_to_base64(fig)

# ============================================================================
# ROC AUC ILLUSTRATIONS
# ============================================================================

def generate_roc_curve_single():
    """Generate single ROC curve."""
    # Generate sample data
    np.random.seed(42)
    X, y = make_classification(n_samples=200, n_features=10, n_informative=8,
                                n_redundant=0, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    clf = LogisticRegression(random_state=42)
    clf.fit(X_train, y_train)
    y_proba = clf.predict_proba(X_test)[:, 1]
    
    fpr, tpr, thresholds = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.plot(fpr, tpr, color='darkorange', lw=2.5, 
            label=f'ROC кривая (AUC = {roc_auc:.3f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
            label='Случайный классификатор (AUC = 0.5)')
    ax.fill_between(fpr, tpr, alpha=0.2, color='darkorange', label='Area Under Curve')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate (FPR)')
    ax.set_ylabel('True Positive Rate (TPR, Recall)')
    ax.set_title('ROC Кривая (Receiver Operating Characteristic)')
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig_to_base64(fig)

def generate_roc_curve_comparison():
    """Generate ROC curves comparison for multiple models."""
    np.random.seed(42)
    X, y = make_classification(n_samples=200, n_features=10, n_informative=8,
                                n_redundant=0, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    models = {
        'Логистическая регрессия': LogisticRegression(random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=50, random_state=42),
        'Decision Tree': DecisionTreeClassifier(max_depth=5, random_state=42)
    }
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    colors = ['darkorange', 'green', 'blue']
    for (name, clf), color in zip(models.items(), colors):
        clf.fit(X_train, y_train)
        y_proba = clf.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=color, lw=2.5, label=f'{name} (AUC = {roc_auc:.3f})')
    
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Случайный (AUC = 0.5)')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate (FPR)')
    ax.set_ylabel('True Positive Rate (TPR)')
    ax.set_title('Сравнение ROC кривых для разных моделей')
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig_to_base64(fig)

def generate_roc_multiclass():
    """Generate ROC curve for multiclass classification."""
    np.random.seed(42)
    X, y = make_classification(n_samples=200, n_features=10, n_classes=3,
                                n_informative=8, n_redundant=0, n_clusters_per_class=1,
                                random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    clf = LogisticRegression(random_state=42, max_iter=1000)
    clf.fit(X_train, y_train)
    y_proba = clf.predict_proba(X_test)
    
    # Binarize the output
    y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
    n_classes = 3
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    colors = ['darkorange', 'green', 'blue']
    for i, color in zip(range(n_classes), colors):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_proba[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=color, lw=2.5,
                label=f'Класс {i} (AUC = {roc_auc:.3f})')
    
    ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Случайный (AUC = 0.5)')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Кривые: Мультиклассовая классификация\n(One-vs-Rest)')
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig_to_base64(fig)

# ============================================================================
# CLASSIFICATION METRICS ILLUSTRATIONS
# ============================================================================

def generate_precision_recall_curve():
    """Generate precision-recall curve."""
    np.random.seed(42)
    X, y = make_classification(n_samples=200, n_features=10, n_informative=8,
                                n_redundant=0, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    clf = LogisticRegression(random_state=42)
    clf.fit(X_train, y_train)
    y_proba = clf.predict_proba(X_test)[:, 1]
    
    precision, recall, thresholds = precision_recall_curve(y_test, y_proba)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.plot(recall, precision, color='darkorange', lw=2.5, label='Precision-Recall кривая')
    ax.set_xlabel('Recall (Полнота)')
    ax.set_ylabel('Precision (Точность)')
    ax.set_title('Precision-Recall Кривая')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.legend(loc="lower left")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig_to_base64(fig)

def generate_metrics_visualization():
    """Generate visualization of classification metrics."""
    # Sample confusion matrix
    cm = np.array([[50, 10], [5, 35]])
    tn, fp, fn, tp = cm.ravel()
    
    # Calculate metrics
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    specificity = tn / (tn + fp)
    f1 = 2 * (precision * recall) / (precision + recall)
    
    metrics = {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'Specificity': specificity,
        'F1-Score': f1
    }
    
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    bars = ax.bar(metrics.keys(), metrics.values(), color=colors, alpha=0.8, edgecolor='black')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontsize=11, weight='bold')
    
    ax.set_ylabel('Значение метрики')
    ax.set_title('Метрики классификации\n(TP=35, TN=50, FP=10, FN=5)')
    ax.set_ylim([0, 1.1])
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    return fig_to_base64(fig)

def generate_threshold_impact():
    """Generate visualization showing impact of threshold on metrics."""
    np.random.seed(42)
    X, y = make_classification(n_samples=200, n_features=10, n_informative=8,
                                n_redundant=0, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    clf = LogisticRegression(random_state=42)
    clf.fit(X_train, y_train)
    y_proba = clf.predict_proba(X_test)[:, 1]
    
    thresholds = np.linspace(0, 1, 100)
    precisions = []
    recalls = []
    f1_scores = []
    
    for threshold in thresholds:
        y_pred_threshold = (y_proba >= threshold).astype(int)
        if len(np.unique(y_pred_threshold)) > 1:
            from sklearn.metrics import precision_score, recall_score, f1_score
            precisions.append(precision_score(y_test, y_pred_threshold, zero_division=0))
            recalls.append(recall_score(y_test, y_pred_threshold, zero_division=0))
            f1_scores.append(f1_score(y_test, y_pred_threshold, zero_division=0))
        else:
            precisions.append(0)
            recalls.append(0)
            f1_scores.append(0)
    
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.plot(thresholds, precisions, label='Precision', linewidth=2.5, color='blue')
    ax.plot(thresholds, recalls, label='Recall', linewidth=2.5, color='green')
    ax.plot(thresholds, f1_scores, label='F1-Score', linewidth=2.5, color='red')
    ax.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5, label='Порог = 0.5')
    ax.set_xlabel('Порог классификации (Threshold)')
    ax.set_ylabel('Значение метрики')
    ax.set_title('Влияние порога на метрики классификации')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig_to_base64(fig)

# ============================================================================
# REGRESSION METRICS ILLUSTRATIONS
# ============================================================================

def generate_regression_residuals():
    """Generate residual plot for regression."""
    np.random.seed(42)
    X, y = make_regression(n_samples=100, n_features=1, noise=10, random_state=42)
    
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)
    residuals = y - y_pred
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Scatter plot with predictions
    ax1.scatter(X, y, alpha=0.6, label='Истинные значения', color='blue')
    ax1.plot(X, y_pred, color='red', linewidth=2.5, label='Предсказания')
    # Draw residual lines
    for i in range(len(X)):
        ax1.plot([X[i], X[i]], [y[i], y_pred[i]], 'gray', linestyle='--', alpha=0.5, linewidth=1)
    ax1.set_xlabel('X')
    ax1.set_ylabel('y')
    ax1.set_title('Линейная регрессия и остатки')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Residual plot
    ax2.scatter(y_pred, residuals, alpha=0.6, color='purple')
    ax2.axhline(y=0, color='red', linestyle='--', linewidth=2)
    ax2.set_xlabel('Предсказанные значения')
    ax2.set_ylabel('Остатки (Residuals)')
    ax2.set_title('График остатков')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig_to_base64(fig)

def generate_regression_metrics_comparison():
    """Generate comparison of regression metrics."""
    np.random.seed(42)
    X, y = make_regression(n_samples=100, n_features=1, noise=15, random_state=42)
    
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)
    
    mse = mean_squared_error(y, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Predictions vs True
    axes[0, 0].scatter(y, y_pred, alpha=0.6, color='blue')
    axes[0, 0].plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
    axes[0, 0].set_xlabel('Истинные значения')
    axes[0, 0].set_ylabel('Предсказания')
    axes[0, 0].set_title(f'Предсказания vs Истина\nR² = {r2:.3f}')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Residuals distribution
    residuals = y - y_pred
    axes[0, 1].hist(residuals, bins=20, alpha=0.7, color='purple', edgecolor='black')
    axes[0, 1].axvline(x=0, color='red', linestyle='--', linewidth=2)
    axes[0, 1].set_xlabel('Остатки')
    axes[0, 1].set_ylabel('Частота')
    axes[0, 1].set_title('Распределение остатков')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Metrics bar chart
    metrics = {'MAE': mae, 'RMSE': rmse, 'MSE': mse}
    bars = axes[1, 0].bar(metrics.keys(), metrics.values(), 
                          color=['green', 'orange', 'red'], alpha=0.7, edgecolor='black')
    for bar in bars:
        height = bar.get_height()
        axes[1, 0].text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.2f}', ha='center', va='bottom', fontsize=10, weight='bold')
    axes[1, 0].set_ylabel('Значение')
    axes[1, 0].set_title('Метрики ошибок регрессии')
    axes[1, 0].grid(axis='y', alpha=0.3)
    
    # Plot 4: Error types visualization
    errors = np.abs(residuals)
    axes[1, 1].scatter(y_pred, errors, alpha=0.6, color='red')
    axes[1, 1].axhline(y=mae, color='green', linestyle='--', linewidth=2, label=f'MAE = {mae:.2f}')
    axes[1, 1].axhline(y=rmse, color='orange', linestyle='--', linewidth=2, label=f'RMSE = {rmse:.2f}')
    axes[1, 1].set_xlabel('Предсказания')
    axes[1, 1].set_ylabel('Абсолютная ошибка')
    axes[1, 1].set_title('Абсолютные ошибки vs Предсказания')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig_to_base64(fig)

# ============================================================================
# LEARNING CURVES ILLUSTRATIONS
# ============================================================================

def generate_learning_curves_good():
    """Generate learning curve for well-fitted model."""
    np.random.seed(42)
    X, y = make_classification(n_samples=500, n_features=20, n_informative=15,
                                n_redundant=5, random_state=42)
    
    clf = RandomForestClassifier(n_estimators=50, max_depth=8, random_state=42)
    
    train_sizes, train_scores, val_scores = learning_curve(
        clf, X, y, train_sizes=np.linspace(0.1, 1.0, 10), 
        cv=5, scoring='accuracy', n_jobs=-1
    )
    
    train_mean = train_scores.mean(axis=1)
    train_std = train_scores.std(axis=1)
    val_mean = val_scores.mean(axis=1)
    val_std = val_scores.std(axis=1)
    
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.plot(train_sizes, train_mean, 'o-', color='blue', label='Training score', linewidth=2.5)
    ax.plot(train_sizes, val_mean, 'o-', color='green', label='Validation score', linewidth=2.5)
    ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, 
                     alpha=0.15, color='blue')
    ax.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, 
                     alpha=0.15, color='green')
    ax.set_xlabel('Размер обучающей выборки')
    ax.set_ylabel('Accuracy')
    ax.set_title('Кривые обучения: Хорошая модель\n(Train и Validation близки, обе высокие)')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0.5, 1.05])
    plt.tight_layout()
    return fig_to_base64(fig)

def generate_learning_curves_overfitting():
    """Generate learning curve showing overfitting."""
    np.random.seed(42)
    X, y = make_classification(n_samples=300, n_features=20, n_informative=5,
                                n_redundant=15, random_state=42)
    
    # Very complex model - prone to overfitting
    clf = DecisionTreeClassifier(random_state=42)  # No constraints
    
    train_sizes, train_scores, val_scores = learning_curve(
        clf, X, y, train_sizes=np.linspace(0.1, 1.0, 10), 
        cv=5, scoring='accuracy', n_jobs=-1
    )
    
    train_mean = train_scores.mean(axis=1)
    train_std = train_scores.std(axis=1)
    val_mean = val_scores.mean(axis=1)
    val_std = val_scores.std(axis=1)
    
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.plot(train_sizes, train_mean, 'o-', color='blue', label='Training score', linewidth=2.5)
    ax.plot(train_sizes, val_mean, 'o-', color='red', label='Validation score', linewidth=2.5)
    ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, 
                     alpha=0.15, color='blue')
    ax.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, 
                     alpha=0.15, color='red')
    
    # Highlight the gap
    for i, size in enumerate(train_sizes):
        ax.plot([size, size], [val_mean[i], train_mean[i]], 
                'gray', linestyle='--', alpha=0.3, linewidth=1)
    
    ax.set_xlabel('Размер обучающей выборки')
    ax.set_ylabel('Accuracy')
    ax.set_title('Кривые обучения: Переобучение (High Variance)\n(Большой разрыв между Train и Validation)')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0.4, 1.05])
    
    # Add annotation
    mid_idx = len(train_sizes) // 2
    ax.annotate('Gap (переобучение)', 
                xy=(train_sizes[mid_idx], (train_mean[mid_idx] + val_mean[mid_idx])/2),
                xytext=(train_sizes[mid_idx] + 50, 0.65),
                arrowprops=dict(arrowstyle='->', color='red', lw=2),
                fontsize=11, color='red', weight='bold')
    
    plt.tight_layout()
    return fig_to_base64(fig)

def generate_learning_curves_underfitting():
    """Generate learning curve showing underfitting."""
    np.random.seed(42)
    X, y = make_classification(n_samples=500, n_features=20, n_informative=15,
                                n_redundant=5, random_state=42)
    
    # Too simple model - underfitting
    clf = DecisionTreeClassifier(max_depth=1, random_state=42)
    
    train_sizes, train_scores, val_scores = learning_curve(
        clf, X, y, train_sizes=np.linspace(0.1, 1.0, 10), 
        cv=5, scoring='accuracy', n_jobs=-1
    )
    
    train_mean = train_scores.mean(axis=1)
    train_std = train_scores.std(axis=1)
    val_mean = val_scores.mean(axis=1)
    val_std = val_scores.std(axis=1)
    
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.plot(train_sizes, train_mean, 'o-', color='blue', label='Training score', linewidth=2.5)
    ax.plot(train_sizes, val_mean, 'o-', color='red', label='Validation score', linewidth=2.5)
    ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, 
                     alpha=0.15, color='blue')
    ax.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, 
                     alpha=0.15, color='red')
    
    ax.axhline(y=0.85, color='green', linestyle='--', alpha=0.5, 
               label='Желаемая точность', linewidth=2)
    
    ax.set_xlabel('Размер обучающей выборки')
    ax.set_ylabel('Accuracy')
    ax.set_title('Кривые обучения: Недообучение (High Bias)\n(Train и Validation близки, но обе низкие)')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0.5, 1.0])
    plt.tight_layout()
    return fig_to_base64(fig)

# ============================================================================
# VALIDATION CURVES ILLUSTRATIONS
# ============================================================================

def generate_validation_curve():
    """Generate validation curve for hyperparameter tuning."""
    np.random.seed(42)
    X, y = make_classification(n_samples=300, n_features=20, n_informative=15,
                                random_state=42)
    
    param_range = [1, 2, 3, 5, 7, 10, 15, 20, 30]
    train_scores, val_scores = validation_curve(
        DecisionTreeClassifier(random_state=42), X, y,
        param_name='max_depth', param_range=param_range,
        cv=5, scoring='accuracy', n_jobs=-1
    )
    
    train_mean = train_scores.mean(axis=1)
    train_std = train_scores.std(axis=1)
    val_mean = val_scores.mean(axis=1)
    val_std = val_scores.std(axis=1)
    
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.plot(param_range, train_mean, 'o-', color='blue', label='Training score', linewidth=2.5)
    ax.plot(param_range, val_mean, 'o-', color='green', label='Validation score', linewidth=2.5)
    ax.fill_between(param_range, train_mean - train_std, train_mean + train_std, 
                     alpha=0.15, color='blue')
    ax.fill_between(param_range, val_mean - val_std, val_mean + val_std, 
                     alpha=0.15, color='green')
    
    # Mark optimal point
    optimal_idx = np.argmax(val_mean)
    optimal_depth = param_range[optimal_idx]
    ax.axvline(x=optimal_depth, color='red', linestyle='--', alpha=0.7, linewidth=2)
    ax.scatter([optimal_depth], [val_mean[optimal_idx]], 
               color='red', s=200, zorder=5, label=f'Оптимум: max_depth={optimal_depth}')
    
    ax.set_xlabel('max_depth (сложность модели)')
    ax.set_ylabel('Accuracy')
    ax.set_title('Validation Curve: Выбор оптимального max_depth')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0.6, 1.05])
    
    # Add regions
    ax.axvspan(param_range[0], optimal_depth, alpha=0.1, color='yellow', label='Underfitting')
    ax.axvspan(optimal_depth, param_range[-1], alpha=0.1, color='red', label='Overfitting')
    
    plt.tight_layout()
    return fig_to_base64(fig)

def generate_validation_curve_regularization():
    """Generate validation curve for regularization parameter."""
    np.random.seed(42)
    X, y = make_regression(n_samples=200, n_features=20, noise=10, random_state=42)
    
    param_range = np.logspace(-3, 3, 20)
    train_scores, val_scores = validation_curve(
        Ridge(), X, y,
        param_name='alpha', param_range=param_range,
        cv=5, scoring='r2', n_jobs=-1
    )
    
    train_mean = train_scores.mean(axis=1)
    train_std = train_scores.std(axis=1)
    val_mean = val_scores.mean(axis=1)
    val_std = val_scores.std(axis=1)
    
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.semilogx(param_range, train_mean, 'o-', color='blue', label='Training score', linewidth=2.5)
    ax.semilogx(param_range, val_mean, 'o-', color='green', label='Validation score', linewidth=2.5)
    ax.fill_between(param_range, train_mean - train_std, train_mean + train_std, 
                     alpha=0.15, color='blue')
    ax.fill_between(param_range, val_mean - val_std, val_mean + val_std, 
                     alpha=0.15, color='green')
    
    # Mark optimal point
    optimal_idx = np.argmax(val_mean)
    optimal_alpha = param_range[optimal_idx]
    ax.axvline(x=optimal_alpha, color='red', linestyle='--', alpha=0.7, linewidth=2)
    ax.scatter([optimal_alpha], [val_mean[optimal_idx]], 
               color='red', s=200, zorder=5, label=f'Оптимум: α={optimal_alpha:.4f}')
    
    ax.set_xlabel('α (параметр регуляризации)')
    ax.set_ylabel('R² Score')
    ax.set_title('Validation Curve: Выбор параметра регуляризации Ridge')
    ax.legend(loc='lower left')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig_to_base64(fig)

# ============================================================================
# BIAS-VARIANCE ILLUSTRATIONS
# ============================================================================

def generate_bias_variance_tradeoff():
    """Generate bias-variance tradeoff visualization."""
    # Simulate errors
    complexity = np.linspace(0, 10, 100)
    bias_squared = 10 * np.exp(-complexity * 0.5)
    variance = 0.1 * np.exp(complexity * 0.3)
    noise = np.ones_like(complexity) * 2
    total_error = bias_squared + variance + noise
    
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(complexity, bias_squared, 'b-', linewidth=3, label='Bias²')
    ax.plot(complexity, variance, 'r-', linewidth=3, label='Variance')
    ax.plot(complexity, noise, 'gray', linestyle='--', linewidth=2, label='Irreducible Error')
    ax.plot(complexity, total_error, 'k-', linewidth=3, label='Total Error')
    
    # Mark optimal point
    optimal_idx = np.argmin(total_error)
    ax.axvline(x=complexity[optimal_idx], color='green', linestyle='--', 
               alpha=0.7, linewidth=2.5)
    ax.scatter([complexity[optimal_idx]], [total_error[optimal_idx]], 
               color='green', s=300, zorder=5, marker='*', 
               edgecolors='black', linewidth=2, label='Оптимальная сложность')
    
    # Add regions
    ax.axvspan(0, complexity[optimal_idx], alpha=0.1, color='blue')
    ax.text(complexity[optimal_idx]/2, ax.get_ylim()[1]*0.9, 
            'Underfitting\n(High Bias)', ha='center', fontsize=12, weight='bold', color='blue')
    
    ax.axvspan(complexity[optimal_idx], 10, alpha=0.1, color='red')
    ax.text((complexity[optimal_idx] + 10)/2, ax.get_ylim()[1]*0.9, 
            'Overfitting\n(High Variance)', ha='center', fontsize=12, weight='bold', color='red')
    
    ax.set_xlabel('Сложность модели', fontsize=12)
    ax.set_ylabel('Ошибка', fontsize=12)
    ax.set_title('Bias-Variance Trade-off', fontsize=14, weight='bold')
    ax.legend(loc='upper center', fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig_to_base64(fig)

def generate_bias_variance_examples():
    """Generate visual examples of bias and variance."""
    np.random.seed(42)
    
    # True function
    X = np.linspace(0, 10, 100).reshape(-1, 1)
    y_true = np.sin(X).ravel()
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # High Bias, Low Variance
    X_sample = np.random.uniform(0, 10, 30).reshape(-1, 1)
    y_sample = np.sin(X_sample).ravel() + np.random.normal(0, 0.1, 30)
    
    model_simple = LinearRegression()
    for i in range(10):
        X_train = np.random.uniform(0, 10, 30).reshape(-1, 1)
        y_train = np.sin(X_train).ravel() + np.random.normal(0, 0.1, 30)
        model_simple.fit(X_train, y_train)
        y_pred = model_simple.predict(X)
        axes[0].plot(X, y_pred, 'b-', alpha=0.3, linewidth=1)
    
    axes[0].plot(X, y_true, 'g-', linewidth=3, label='True function')
    axes[0].scatter(X_sample, y_sample, c='red', alpha=0.6, s=30)
    axes[0].set_title('High Bias, Low Variance\n(Underfitting)', fontsize=12, weight='bold')
    axes[0].set_xlabel('X')
    axes[0].set_ylabel('y')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Low Bias, Low Variance (Good!)
    from sklearn.ensemble import RandomForestRegressor
    for i in range(10):
        X_train = np.random.uniform(0, 10, 50).reshape(-1, 1)
        y_train = np.sin(X_train).ravel() + np.random.normal(0, 0.1, 50)
        model_good = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=i)
        model_good.fit(X_train, y_train)
        y_pred = model_good.predict(X)
        axes[1].plot(X, y_pred, 'purple', alpha=0.3, linewidth=1)
    
    axes[1].plot(X, y_true, 'g-', linewidth=3, label='True function')
    axes[1].scatter(X_sample, y_sample, c='red', alpha=0.6, s=30)
    axes[1].set_title('Low Bias, Low Variance\n(Good Balance!)', fontsize=12, weight='bold')
    axes[1].set_xlabel('X')
    axes[1].set_ylabel('y')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Low Bias, High Variance
    from sklearn.tree import DecisionTreeRegressor
    for i in range(10):
        X_train = np.random.uniform(0, 10, 30).reshape(-1, 1)
        y_train = np.sin(X_train).ravel() + np.random.normal(0, 0.1, 30)
        model_complex = DecisionTreeRegressor(max_depth=10, random_state=i)
        model_complex.fit(X_train, y_train)
        y_pred = model_complex.predict(X)
        axes[2].plot(X, y_pred, 'r-', alpha=0.3, linewidth=1)
    
    axes[2].plot(X, y_true, 'g-', linewidth=3, label='True function')
    axes[2].scatter(X_sample, y_sample, c='red', alpha=0.6, s=30)
    axes[2].set_title('Low Bias, High Variance\n(Overfitting)', fontsize=12, weight='bold')
    axes[2].set_xlabel('X')
    axes[2].set_ylabel('y')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig_to_base64(fig)

# ============================================================================
# OVERFITTING/UNDERFITTING ILLUSTRATIONS
# ============================================================================

def generate_overfitting_underfitting():
    """Generate visualization of underfitting, good fit, and overfitting."""
    np.random.seed(42)
    
    # Generate sample data
    X = np.sort(np.random.uniform(0, 10, 80)).reshape(-1, 1)
    y = np.sin(X).ravel() + np.random.normal(0, 0.3, X.shape[0])
    
    X_test = np.linspace(0, 10, 300).reshape(-1, 1)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Underfitting
    from sklearn.preprocessing import PolynomialFeatures
    poly_under = PolynomialFeatures(degree=1)
    X_poly_under = poly_under.fit_transform(X)
    X_test_poly_under = poly_under.transform(X_test)
    
    model_under = LinearRegression()
    model_under.fit(X_poly_under, y)
    y_pred_under = model_under.predict(X_test_poly_under)
    
    axes[0].scatter(X, y, color='blue', alpha=0.6, s=40, label='Training data')
    axes[0].plot(X_test, y_pred_under, color='red', linewidth=3, label='Model (degree=1)')
    axes[0].set_title('Underfitting (High Bias)\nМодель слишком простая', 
                     fontsize=12, weight='bold', color='red')
    axes[0].set_xlabel('X')
    axes[0].set_ylabel('y')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Good fit
    poly_good = PolynomialFeatures(degree=3)
    X_poly_good = poly_good.fit_transform(X)
    X_test_poly_good = poly_good.transform(X_test)
    
    model_good = LinearRegression()
    model_good.fit(X_poly_good, y)
    y_pred_good = model_good.predict(X_test_poly_good)
    
    axes[1].scatter(X, y, color='blue', alpha=0.6, s=40, label='Training data')
    axes[1].plot(X_test, y_pred_good, color='green', linewidth=3, label='Model (degree=3)')
    axes[1].set_title('Good Fit (Balanced)\nОптимальная сложность', 
                     fontsize=12, weight='bold', color='green')
    axes[1].set_xlabel('X')
    axes[1].set_ylabel('y')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Overfitting
    poly_over = PolynomialFeatures(degree=15)
    X_poly_over = poly_over.fit_transform(X)
    X_test_poly_over = poly_over.transform(X_test)
    
    model_over = LinearRegression()
    model_over.fit(X_poly_over, y)
    y_pred_over = model_over.predict(X_test_poly_over)
    
    axes[2].scatter(X, y, color='blue', alpha=0.6, s=40, label='Training data')
    axes[2].plot(X_test, y_pred_over, color='red', linewidth=3, label='Model (degree=15)')
    axes[2].set_title('Overfitting (High Variance)\nМодель слишком сложная', 
                     fontsize=12, weight='bold', color='red')
    axes[2].set_xlabel('X')
    axes[2].set_ylabel('y')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    axes[2].set_ylim([-3, 3])
    
    plt.tight_layout()
    return fig_to_base64(fig)

# ============================================================================
# MAIN GENERATION FUNCTION
# ============================================================================

def generate_all_illustrations():
    """Generate all metrics and evaluation illustrations and return as dictionary."""
    print("Generating metrics and evaluation illustrations...")
    
    illustrations = {}
    
    print("  - Confusion Matrix...")
    illustrations['cm_binary'] = generate_confusion_matrix_binary()
    illustrations['cm_multiclass'] = generate_confusion_matrix_multiclass()
    illustrations['cm_normalized'] = generate_confusion_matrix_normalized()
    
    print("  - ROC AUC...")
    illustrations['roc_single'] = generate_roc_curve_single()
    illustrations['roc_comparison'] = generate_roc_curve_comparison()
    illustrations['roc_multiclass'] = generate_roc_multiclass()
    
    print("  - Classification Metrics...")
    illustrations['precision_recall_curve'] = generate_precision_recall_curve()
    illustrations['metrics_viz'] = generate_metrics_visualization()
    illustrations['threshold_impact'] = generate_threshold_impact()
    
    print("  - Regression Metrics...")
    illustrations['regression_residuals'] = generate_regression_residuals()
    illustrations['regression_metrics'] = generate_regression_metrics_comparison()
    
    print("  - Learning Curves...")
    illustrations['learning_good'] = generate_learning_curves_good()
    illustrations['learning_overfit'] = generate_learning_curves_overfitting()
    illustrations['learning_underfit'] = generate_learning_curves_underfitting()
    
    print("  - Validation Curves...")
    illustrations['validation_curve'] = generate_validation_curve()
    illustrations['validation_regularization'] = generate_validation_curve_regularization()
    
    print("  - Bias-Variance...")
    illustrations['bias_variance_tradeoff'] = generate_bias_variance_tradeoff()
    illustrations['bias_variance_examples'] = generate_bias_variance_examples()
    
    print("  - Overfitting/Underfitting...")
    illustrations['overfitting_underfitting'] = generate_overfitting_underfitting()
    
    print("✓ All illustrations generated successfully!")
    return illustrations

if __name__ == '__main__':
    # Test generation
    illustrations = generate_all_illustrations()
    print(f"\nGenerated {len(illustrations)} illustrations")
    for key in illustrations.keys():
        print(f"  - {key}")
