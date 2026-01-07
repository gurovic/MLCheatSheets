#!/usr/bin/env python3
"""
Generate matplotlib illustrations for regression cheatsheets.
This script creates high-quality visualizations and encodes them as base64 for inline HTML embedding.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
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
# LINEAR REGRESSION ILLUSTRATIONS
# ============================================================================

def generate_linreg_basic():
    """Generate basic linear regression visualization."""
    np.random.seed(42)
    X = np.linspace(0, 10, 100).reshape(-1, 1)
    y = 2 * X.ravel() + 1 + np.random.normal(0, 1, 100)
    
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(X, y, alpha=0.5, s=30, label='Данные')
    ax.plot(X, y_pred, 'r-', linewidth=2, label=f'Регрессия: y = {model.coef_[0]:.2f}x + {model.intercept_:.2f}')
    ax.set_xlabel('X')
    ax.set_ylabel('y')
    ax.set_title('Линейная регрессия')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return fig_to_base64(fig)

def generate_linreg_residuals():
    """Generate residuals visualization."""
    np.random.seed(42)
    X = np.linspace(0, 10, 50).reshape(-1, 1)
    y = 2 * X.ravel() + 1 + np.random.normal(0, 1.5, 50)
    
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)
    residuals = y - y_pred
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Residuals on regression plot
    ax1.scatter(X, y, alpha=0.5, s=30, label='Данные')
    ax1.plot(X, y_pred, 'r-', linewidth=2, label='Регрессия')
    
    # Draw residuals
    for i in range(len(X)):
        ax1.plot([X[i], X[i]], [y[i], y_pred[i]], 'g--', alpha=0.3, linewidth=1)
    
    ax1.set_xlabel('X')
    ax1.set_ylabel('y')
    ax1.set_title('Остатки (residuals)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Residual plot
    ax2.scatter(y_pred, residuals, alpha=0.5, s=30)
    ax2.axhline(y=0, color='r', linestyle='--', linewidth=2)
    ax2.set_xlabel('Предсказанные значения')
    ax2.set_ylabel('Остатки')
    ax2.set_title('График остатков')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig_to_base64(fig)

def generate_linreg_r2():
    """Generate R² visualization with different fits."""
    np.random.seed(42)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Good fit
    X1 = np.linspace(0, 10, 50).reshape(-1, 1)
    y1 = 2 * X1.ravel() + 1 + np.random.normal(0, 0.5, 50)
    model1 = LinearRegression()
    model1.fit(X1, y1)
    r2_1 = r2_score(y1, model1.predict(X1))
    
    axes[0].scatter(X1, y1, alpha=0.5, s=30)
    axes[0].plot(X1, model1.predict(X1), 'r-', linewidth=2)
    axes[0].set_title(f'Хорошая модель\nR² = {r2_1:.3f}')
    axes[0].set_xlabel('X')
    axes[0].set_ylabel('y')
    axes[0].grid(True, alpha=0.3)
    
    # Medium fit
    X2 = np.linspace(0, 10, 50).reshape(-1, 1)
    y2 = 2 * X2.ravel() + 1 + np.random.normal(0, 2, 50)
    model2 = LinearRegression()
    model2.fit(X2, y2)
    r2_2 = r2_score(y2, model2.predict(X2))
    
    axes[1].scatter(X2, y2, alpha=0.5, s=30)
    axes[1].plot(X2, model2.predict(X2), 'r-', linewidth=2)
    axes[1].set_title(f'Средняя модель\nR² = {r2_2:.3f}')
    axes[1].set_xlabel('X')
    axes[1].set_ylabel('y')
    axes[1].grid(True, alpha=0.3)
    
    # Poor fit
    X3 = np.linspace(0, 10, 50).reshape(-1, 1)
    y3 = np.random.normal(5, 3, 50)  # No correlation
    model3 = LinearRegression()
    model3.fit(X3, y3)
    r2_3 = r2_score(y3, model3.predict(X3))
    
    axes[2].scatter(X3, y3, alpha=0.5, s=30)
    axes[2].plot(X3, model3.predict(X3), 'r-', linewidth=2)
    axes[2].set_title(f'Плохая модель\nR² = {r2_3:.3f}')
    axes[2].set_xlabel('X')
    axes[2].set_ylabel('y')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig_to_base64(fig)

# ============================================================================
# POLYNOMIAL REGRESSION ILLUSTRATIONS
# ============================================================================

def generate_polynomial_degrees():
    """Generate polynomial regression with different degrees."""
    np.random.seed(42)
    X = np.linspace(0, 10, 50).reshape(-1, 1)
    y = 0.5 * X.ravel()**2 - 3 * X.ravel() + 10 + np.random.normal(0, 5, 50)
    
    X_test = np.linspace(0, 10, 200).reshape(-1, 1)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    degrees = [1, 2, 3, 5, 10, 20]
    
    for idx, degree in enumerate(degrees):
        ax = axes[idx // 3, idx % 3]
        
        poly = PolynomialFeatures(degree=degree)
        X_poly = poly.fit_transform(X)
        X_test_poly = poly.transform(X_test)
        
        model = LinearRegression()
        model.fit(X_poly, y)
        y_pred = model.predict(X_test_poly)
        
        train_r2 = r2_score(y, model.predict(X_poly))
        
        ax.scatter(X, y, alpha=0.5, s=20, label='Данные')
        ax.plot(X_test, y_pred, 'r-', linewidth=2, label=f'Полином степени {degree}')
        ax.set_title(f'Степень = {degree}, R² = {train_r2:.3f}')
        ax.set_xlabel('X')
        ax.set_ylabel('y')
        ax.set_ylim([y.min() - 10, y.max() + 10])
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig_to_base64(fig)

def generate_polynomial_overfitting():
    """Generate polynomial overfitting visualization."""
    np.random.seed(42)
    X_train = np.linspace(0, 10, 15).reshape(-1, 1)
    y_train = 0.5 * X_train.ravel()**2 - 3 * X_train.ravel() + 10 + np.random.normal(0, 5, 15)
    
    X_test = np.linspace(0, 10, 100).reshape(-1, 1)
    y_test_true = 0.5 * X_test.ravel()**2 - 3 * X_test.ravel() + 10
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Good fit - degree 2
    poly2 = PolynomialFeatures(degree=2)
    X_train_poly2 = poly2.fit_transform(X_train)
    X_test_poly2 = poly2.transform(X_test)
    
    model2 = LinearRegression()
    model2.fit(X_train_poly2, y_train)
    y_pred2 = model2.predict(X_test_poly2)
    
    ax1.scatter(X_train, y_train, alpha=0.7, s=50, label='Обучающие данные', color='blue')
    ax1.plot(X_test, y_test_true, 'g--', linewidth=2, label='Истинная функция', alpha=0.5)
    ax1.plot(X_test, y_pred2, 'r-', linewidth=2, label='Полином степени 2')
    ax1.set_title('Оптимальная модель (степень 2)')
    ax1.set_xlabel('X')
    ax1.set_ylabel('y')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Overfit - degree 14
    poly14 = PolynomialFeatures(degree=14)
    X_train_poly14 = poly14.fit_transform(X_train)
    X_test_poly14 = poly14.transform(X_test)
    
    model14 = LinearRegression()
    model14.fit(X_train_poly14, y_train)
    y_pred14 = model14.predict(X_test_poly14)
    
    ax2.scatter(X_train, y_train, alpha=0.7, s=50, label='Обучающие данные', color='blue')
    ax2.plot(X_test, y_test_true, 'g--', linewidth=2, label='Истинная функция', alpha=0.5)
    ax2.plot(X_test, y_pred14, 'r-', linewidth=2, label='Полином степени 14')
    ax2.set_title('Переобучение (степень 14)')
    ax2.set_xlabel('X')
    ax2.set_ylabel('y')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([y_train.min() - 20, y_train.max() + 20])
    
    plt.tight_layout()
    return fig_to_base64(fig)

# ============================================================================
# RIDGE/LASSO/ELASTICNET ILLUSTRATIONS
# ============================================================================

def generate_regularization_comparison():
    """Generate comparison of Ridge, Lasso, and ElasticNet."""
    np.random.seed(42)
    X = np.linspace(0, 10, 15).reshape(-1, 1)
    y = 0.5 * X.ravel()**2 - 3 * X.ravel() + 10 + np.random.normal(0, 5, 15)
    
    X_test = np.linspace(0, 10, 200).reshape(-1, 1)
    
    # Create polynomial features
    poly = PolynomialFeatures(degree=10)
    X_poly = poly.fit_transform(X)
    X_test_poly = poly.transform(X_test)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    models = [
        ('Без регуляризации', LinearRegression(), axes[0, 0]),
        ('Ridge (L2)', Ridge(alpha=10), axes[0, 1]),
        ('Lasso (L1)', Lasso(alpha=1), axes[1, 0]),
        ('ElasticNet (L1+L2)', ElasticNet(alpha=1, l1_ratio=0.5), axes[1, 1])
    ]
    
    for name, model, ax in models:
        model.fit(X_poly, y)
        y_pred = model.predict(X_test_poly)
        
        ax.scatter(X, y, alpha=0.7, s=50, label='Данные', color='blue')
        ax.plot(X_test, y_pred, 'r-', linewidth=2, label=name)
        ax.set_title(name)
        ax.set_xlabel('X')
        ax.set_ylabel('y')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim([y.min() - 20, y.max() + 20])
    
    plt.tight_layout()
    return fig_to_base64(fig)

def generate_regularization_coefficients():
    """Generate coefficient comparison for different regularization methods."""
    np.random.seed(42)
    n_samples, n_features = 100, 20
    X = np.random.randn(n_samples, n_features)
    
    # True coefficients: only first 5 are non-zero
    true_coef = np.zeros(n_features)
    true_coef[:5] = [3, -2, 1.5, -1, 0.5]
    
    y = X @ true_coef + np.random.randn(n_samples) * 0.5
    
    # Fit models
    ridge = Ridge(alpha=1).fit(X, y)
    lasso = Lasso(alpha=0.1).fit(X, y)
    elasticnet = ElasticNet(alpha=0.1, l1_ratio=0.5).fit(X, y)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(n_features)
    width = 0.2
    
    ax.bar(x - width*1.5, true_coef, width, label='Истинные коэффициенты', alpha=0.8)
    ax.bar(x - width/2, ridge.coef_, width, label='Ridge', alpha=0.8)
    ax.bar(x + width/2, lasso.coef_, width, label='Lasso', alpha=0.8)
    ax.bar(x + width*1.5, elasticnet.coef_, width, label='ElasticNet', alpha=0.8)
    
    ax.set_xlabel('Номер признака')
    ax.set_ylabel('Значение коэффициента')
    ax.set_title('Сравнение коэффициентов: Ridge vs Lasso vs ElasticNet')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(y=0, color='black', linewidth=0.5)
    
    return fig_to_base64(fig)

def generate_regularization_path():
    """Generate regularization path for Ridge and Lasso."""
    np.random.seed(42)
    n_samples, n_features = 100, 10
    X = np.random.randn(n_samples, n_features)
    
    true_coef = np.array([3, -2, 1.5, -1, 0.5, 0, 0, 0, 0, 0])
    y = X @ true_coef + np.random.randn(n_samples) * 0.5
    
    alphas = np.logspace(-3, 3, 100)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Ridge path
    coefs_ridge = []
    for alpha in alphas:
        ridge = Ridge(alpha=alpha)
        ridge.fit(X, y)
        coefs_ridge.append(ridge.coef_)
    coefs_ridge = np.array(coefs_ridge)
    
    for i in range(n_features):
        ax1.plot(alphas, coefs_ridge[:, i], linewidth=2)
    ax1.set_xscale('log')
    ax1.set_xlabel('Alpha (λ)')
    ax1.set_ylabel('Коэффициенты')
    ax1.set_title('Ridge: путь регуляризации')
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='black', linewidth=0.5)
    
    # Lasso path
    coefs_lasso = []
    for alpha in alphas:
        lasso = Lasso(alpha=alpha, max_iter=10000)
        lasso.fit(X, y)
        coefs_lasso.append(lasso.coef_)
    coefs_lasso = np.array(coefs_lasso)
    
    for i in range(n_features):
        ax2.plot(alphas, coefs_lasso[:, i], linewidth=2)
    ax2.set_xscale('log')
    ax2.set_xlabel('Alpha (λ)')
    ax2.set_ylabel('Коэффициенты')
    ax2.set_title('Lasso: путь регуляризации (отбор признаков)')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='black', linewidth=0.5)
    
    plt.tight_layout()
    return fig_to_base64(fig)

# ============================================================================
# DECISION TREES REGRESSION ILLUSTRATIONS
# ============================================================================

def generate_decision_tree_basic():
    """Generate basic decision tree regression visualization."""
    np.random.seed(42)
    X = np.linspace(0, 10, 100).reshape(-1, 1)
    y = np.sin(X.ravel()) * 3 + np.random.normal(0, 0.3, 100)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    depths = [2, 5, 20]
    
    for ax, depth in zip(axes, depths):
        model = DecisionTreeRegressor(max_depth=depth, random_state=42)
        model.fit(X, y)
        
        X_test = np.linspace(0, 10, 300).reshape(-1, 1)
        y_pred = model.predict(X_test)
        
        ax.scatter(X, y, alpha=0.5, s=20, label='Данные')
        ax.plot(X_test, y_pred, 'r-', linewidth=2, label=f'max_depth={depth}')
        ax.set_title(f'Дерево решений (глубина={depth})')
        ax.set_xlabel('X')
        ax.set_ylabel('y')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig_to_base64(fig)

def generate_decision_tree_overfitting():
    """Generate decision tree overfitting visualization."""
    np.random.seed(42)
    X_train = np.sort(np.random.rand(30, 1) * 10, axis=0)
    y_train = np.sin(X_train.ravel()) * 3 + np.random.normal(0, 0.5, 30)
    
    X_test = np.linspace(0, 10, 200).reshape(-1, 1)
    y_test_true = np.sin(X_test.ravel()) * 3
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Good model
    model_good = DecisionTreeRegressor(max_depth=3, random_state=42)
    model_good.fit(X_train, y_train)
    y_pred_good = model_good.predict(X_test)
    train_score_good = model_good.score(X_train, y_train)
    
    ax1.scatter(X_train, y_train, alpha=0.7, s=50, label='Обучающие данные', color='blue')
    ax1.plot(X_test, y_test_true, 'g--', linewidth=2, label='Истинная функция', alpha=0.5)
    ax1.plot(X_test, y_pred_good, 'r-', linewidth=2, label='Предсказание')
    ax1.set_title(f'Хорошая модель (max_depth=3)\nTrain R²={train_score_good:.3f}')
    ax1.set_xlabel('X')
    ax1.set_ylabel('y')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Overfit model
    model_overfit = DecisionTreeRegressor(max_depth=None, random_state=42)
    model_overfit.fit(X_train, y_train)
    y_pred_overfit = model_overfit.predict(X_test)
    train_score_overfit = model_overfit.score(X_train, y_train)
    
    ax2.scatter(X_train, y_train, alpha=0.7, s=50, label='Обучающие данные', color='blue')
    ax2.plot(X_test, y_test_true, 'g--', linewidth=2, label='Истинная функция', alpha=0.5)
    ax2.plot(X_test, y_pred_overfit, 'r-', linewidth=2, label='Предсказание')
    ax2.set_title(f'Переобучение (max_depth=None)\nTrain R²={train_score_overfit:.3f}')
    ax2.set_xlabel('X')
    ax2.set_ylabel('y')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig_to_base64(fig)

def generate_decision_tree_2d():
    """Generate 2D decision tree regression visualization."""
    np.random.seed(42)
    n = 200
    X = np.random.rand(n, 2) * 10
    y = np.sin(X[:, 0]) + np.cos(X[:, 1]) + np.random.normal(0, 0.3, n)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    depths = [2, 5, 10]
    
    for ax, depth in zip(axes, depths):
        model = DecisionTreeRegressor(max_depth=depth, random_state=42)
        model.fit(X, y)
        
        # Create grid
        x_min, x_max = 0, 10
        y_min, y_max = 0, 10
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                             np.linspace(y_min, y_max, 100))
        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        
        # Plot
        contour = ax.contourf(xx, yy, Z, levels=20, cmap='viridis', alpha=0.6)
        ax.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', 
                  edgecolors='white', linewidth=0.5, s=20)
        ax.set_title(f'max_depth={depth}')
        ax.set_xlabel('X1')
        ax.set_ylabel('X2')
        plt.colorbar(contour, ax=ax, label='Предсказание')
    
    plt.tight_layout()
    return fig_to_base64(fig)

# ============================================================================
# MAIN GENERATION FUNCTION
# ============================================================================

def generate_all_illustrations():
    """Generate all regression illustrations and return dictionary."""
    print("Generating regression illustrations...")
    
    illustrations = {}
    
    # Linear Regression
    print("  Linear Regression...")
    illustrations['linreg_basic'] = generate_linreg_basic()
    illustrations['linreg_residuals'] = generate_linreg_residuals()
    illustrations['linreg_r2'] = generate_linreg_r2()
    
    # Polynomial Regression
    print("  Polynomial Regression...")
    illustrations['polynomial_degrees'] = generate_polynomial_degrees()
    illustrations['polynomial_overfitting'] = generate_polynomial_overfitting()
    
    # Ridge/Lasso/ElasticNet
    print("  Ridge/Lasso/ElasticNet...")
    illustrations['regularization_comparison'] = generate_regularization_comparison()
    illustrations['regularization_coefficients'] = generate_regularization_coefficients()
    illustrations['regularization_path'] = generate_regularization_path()
    
    # Decision Trees Regression
    print("  Decision Trees Regression...")
    illustrations['decision_tree_basic'] = generate_decision_tree_basic()
    illustrations['decision_tree_overfitting'] = generate_decision_tree_overfitting()
    illustrations['decision_tree_2d'] = generate_decision_tree_2d()
    
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
