#!/usr/bin/env python3
"""
Generate matplotlib illustrations for Bayesian methods cheatsheets.
This script creates high-quality visualizations and encodes them as base64 for inline HTML embedding.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm, multivariate_normal, beta
from scipy.optimize import minimize
import base64
from io import BytesIO
import warnings
warnings.filterwarnings('ignore')

# Set consistent style
sns.set_style("whitegrid")
DPI_VALUE = 300
plt.rcParams['figure.dpi'] = DPI_VALUE
plt.rcParams['savefig.dpi'] = DPI_VALUE
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9

def fig_to_base64(fig):
    """Convert matplotlib figure to base64 string."""
    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=DPI_VALUE)
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()
    plt.close(fig)
    return img_base64

# ============================================================================
# BAYESIAN OPTIMIZATION ILLUSTRATIONS
# ============================================================================

def generate_bayesian_optimization_process():
    """Visualize Bayesian Optimization process with acquisition function."""
    np.random.seed(42)
    
    # Define true function (expensive to evaluate)
    def true_function(x):
        return -np.sin(3*x) - x**2 + 0.7*x
    
    # Generate true function values
    x_true = np.linspace(0, 2, 200)
    y_true = true_function(x_true)
    
    # Simulated observations (iterations)
    x_obs = np.array([0.3, 0.8, 1.5])
    y_obs = true_function(x_obs)
    
    # Simple GP mean and std (simplified for visualization)
    x_pred = np.linspace(0, 2, 100)
    y_pred_mean = np.interp(x_pred, x_obs, y_obs)
    
    # Create uncertainty that decreases near observations
    uncertainty = np.ones_like(x_pred) * 0.3
    for x_o in x_obs:
        uncertainty *= (1 + 0.5 * np.abs(x_pred - x_o))
    uncertainty = np.clip(uncertainty, 0.05, 0.5)
    
    y_pred_std = uncertainty
    
    # Acquisition function (EI-like)
    best_y = np.max(y_obs)
    acquisition = uncertainty * (y_pred_mean > best_y - 0.2)
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    # Plot 1: True function and observations
    ax1 = axes[0]
    ax1.plot(x_true, y_true, 'b-', linewidth=2, label='Истинная функция (неизвестна)')
    ax1.plot(x_obs, y_obs, 'ro', markersize=10, label='Наблюдения', zorder=5)
    ax1.set_xlabel('x')
    ax1.set_ylabel('f(x)')
    ax1.set_title('Истинная функция и наблюдаемые точки', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: GP surrogate model
    ax2 = axes[1]
    ax2.plot(x_true, y_true, 'b--', linewidth=1, alpha=0.3, label='Истинная функция')
    ax2.plot(x_pred, y_pred_mean, 'g-', linewidth=2, label='GP предсказание (μ)')
    ax2.fill_between(x_pred, 
                      y_pred_mean - 1.96*y_pred_std, 
                      y_pred_mean + 1.96*y_pred_std,
                      alpha=0.3, color='green', label='95% доверительный интервал')
    ax2.plot(x_obs, y_obs, 'ro', markersize=10, label='Наблюдения', zorder=5)
    ax2.set_xlabel('x')
    ax2.set_ylabel('f(x)')
    ax2.set_title('Суррогатная модель (Gaussian Process)', fontweight='bold')
    ax2.legend(loc='upper right', fontsize=8)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Acquisition function
    ax3 = axes[2]
    ax3.plot(x_pred, acquisition, 'r-', linewidth=2, label='Acquisition Function (EI)')
    ax3.fill_between(x_pred, 0, acquisition, alpha=0.3, color='red')
    next_x = x_pred[np.argmax(acquisition)]
    ax3.axvline(next_x, color='orange', linestyle='--', linewidth=2, 
                label=f'Следующая точка: x={next_x:.2f}')
    ax3.set_xlabel('x')
    ax3.set_ylabel('Acquisition value')
    ax3.set_title('Acquisition Function (баланс exploration/exploitation)', fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    plt.suptitle('Процесс Bayesian Optimization', fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    return fig_to_base64(fig)

def generate_acquisition_functions_comparison():
    """Compare different acquisition functions."""
    np.random.seed(42)
    
    x = np.linspace(0, 10, 100)
    
    # Simulated GP predictions
    mu = -0.5 * (x - 5)**2 + 2
    sigma = 0.5 + 0.3 * np.sin(x)
    
    best_f = np.max(mu)
    
    # Expected Improvement (EI)
    Z = (mu - best_f) / (sigma + 1e-9)
    ei = (mu - best_f) * norm.cdf(Z) + sigma * norm.pdf(Z)
    ei = np.maximum(ei, 0)
    
    # Upper Confidence Bound (UCB)
    kappa = 2.0
    ucb = mu + kappa * sigma
    
    # Probability of Improvement (PI)
    pi = norm.cdf(Z)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot GP
    ax1 = axes[0, 0]
    ax1.plot(x, mu, 'b-', linewidth=2, label='μ(x) - среднее')
    ax1.fill_between(x, mu - 1.96*sigma, mu + 1.96*sigma, 
                      alpha=0.3, color='blue', label='95% CI')
    ax1.axhline(best_f, color='red', linestyle='--', alpha=0.5, label=f'Лучшее f = {best_f:.2f}')
    ax1.set_xlabel('x')
    ax1.set_ylabel('f(x)')
    ax1.set_title('Gaussian Process (суррогатная модель)', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot EI
    ax2 = axes[0, 1]
    ax2.plot(x, ei, 'g-', linewidth=2)
    ax2.fill_between(x, 0, ei, alpha=0.3, color='green')
    max_ei_idx = np.argmax(ei)
    ax2.axvline(x[max_ei_idx], color='red', linestyle='--', 
                label=f'Max EI at x={x[max_ei_idx]:.2f}')
    ax2.set_xlabel('x')
    ax2.set_ylabel('EI(x)')
    ax2.set_title('Expected Improvement (EI)', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot UCB
    ax3 = axes[1, 0]
    ax3.plot(x, mu, 'b--', linewidth=1, alpha=0.5, label='μ(x)')
    ax3.plot(x, ucb, 'r-', linewidth=2, label=f'UCB (κ={kappa})')
    max_ucb_idx = np.argmax(ucb)
    ax3.axvline(x[max_ucb_idx], color='orange', linestyle='--',
                label=f'Max UCB at x={x[max_ucb_idx]:.2f}')
    ax3.set_xlabel('x')
    ax3.set_ylabel('UCB(x)')
    ax3.set_title('Upper Confidence Bound (UCB)', fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot PI
    ax4 = axes[1, 1]
    ax4.plot(x, pi, 'purple', linewidth=2)
    ax4.fill_between(x, 0, pi, alpha=0.3, color='purple')
    max_pi_idx = np.argmax(pi)
    ax4.axvline(x[max_pi_idx], color='red', linestyle='--',
                label=f'Max PI at x={x[max_pi_idx]:.2f}')
    ax4.set_xlabel('x')
    ax4.set_ylabel('PI(x)')
    ax4.set_title('Probability of Improvement (PI)', fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.suptitle('Сравнение Acquisition Functions', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    return fig_to_base64(fig)

def generate_bayesian_optimization_iterations():
    """Show iterations of Bayesian Optimization."""
    np.random.seed(42)
    
    # True function
    def true_func(x):
        return -(x - 2)**2 * np.sin(5*x) + 0.5*x
    
    x_range = np.linspace(0, 5, 200)
    y_true = true_func(x_range)
    
    # Simulate iterations
    iterations = [
        {'x': np.array([0.5, 4.5]), 'label': 'Итерация 1: случайная инициализация'},
        {'x': np.array([0.5, 4.5, 2.3]), 'label': 'Итерация 2: добавлена точка'},
        {'x': np.array([0.5, 4.5, 2.3, 3.8]), 'label': 'Итерация 3'},
        {'x': np.array([0.5, 4.5, 2.3, 3.8, 1.7]), 'label': 'Итерация 4'},
    ]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for idx, (ax, iter_data) in enumerate(zip(axes, iterations)):
        x_obs = iter_data['x']
        y_obs = true_func(x_obs)
        
        # Plot true function
        ax.plot(x_range, y_true, 'b-', linewidth=2, alpha=0.5, label='Истинная функция')
        
        # Plot observations
        ax.plot(x_obs, y_obs, 'ro', markersize=10, label='Наблюдения', zorder=5)
        
        # Highlight best so far
        best_idx = np.argmax(y_obs)
        ax.plot(x_obs[best_idx], y_obs[best_idx], 'g*', markersize=20, 
                label=f'Лучшее: f={y_obs[best_idx]:.2f}', zorder=6)
        
        # Highlight newest point
        if idx > 0:
            ax.plot(x_obs[-1], y_obs[-1], 'mo', markersize=12, 
                    label='Новая точка', zorder=6)
        
        ax.set_xlabel('x')
        ax.set_ylabel('f(x)')
        ax.set_title(iter_data['label'], fontweight='bold')
        ax.legend(loc='upper left', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 5])
        ax.set_ylim([y_true.min() - 0.5, y_true.max() + 0.5])
    
    plt.suptitle('Итерации Bayesian Optimization', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    return fig_to_base64(fig)

# ============================================================================
# BAYESIAN NEURAL NETWORKS ILLUSTRATIONS
# ============================================================================

def generate_bnn_uncertainty_comparison():
    """Compare regular NN vs Bayesian NN predictions with uncertainty."""
    np.random.seed(42)
    
    # Generate data
    X_train = np.linspace(-3, 3, 20)
    y_train = X_train + np.sin(2*X_train) + np.random.randn(20) * 0.3
    
    X_test = np.linspace(-4, 4, 100)
    
    # Regular NN prediction (deterministic)
    y_pred_regular = X_test + np.sin(2*X_test)
    
    # Bayesian NN predictions (with uncertainty)
    y_pred_bayes = X_test + np.sin(2*X_test)
    
    # Uncertainty increases outside training data
    uncertainty = 0.3 * np.ones_like(X_test)
    uncertainty += 0.5 * np.abs(X_test)**2 * (np.abs(X_test) > 3)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Regular NN
    ax1 = axes[0]
    ax1.scatter(X_train, y_train, c='blue', s=50, alpha=0.6, label='Обучающие данные', zorder=5)
    ax1.plot(X_test, y_pred_regular, 'r-', linewidth=2, label='NN предсказание')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_title('Обычная нейронная сеть\n(нет uncertainty)', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([-4, 4])
    ax1.axvspan(-4, -3, alpha=0.1, color='red', label='Вне обучающих данных')
    ax1.axvspan(3, 4, alpha=0.1, color='red')
    
    # Bayesian NN
    ax2 = axes[1]
    ax2.scatter(X_train, y_train, c='blue', s=50, alpha=0.6, label='Обучающие данные', zorder=5)
    ax2.plot(X_test, y_pred_bayes, 'g-', linewidth=2, label='BNN среднее предсказание')
    ax2.fill_between(X_test, 
                      y_pred_bayes - 1.96*uncertainty, 
                      y_pred_bayes + 1.96*uncertainty,
                      alpha=0.3, color='green', label='95% доверительный интервал')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_title('Байесовская нейронная сеть\n(с uncertainty)', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([-4, 4])
    ax2.axvspan(-4, -3, alpha=0.1, color='red')
    ax2.axvspan(3, 4, alpha=0.1, color='red')
    
    plt.suptitle('Сравнение: обычная NN vs Байесовская NN', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    return fig_to_base64(fig)

def generate_weight_distributions():
    """Visualize weight distributions in Bayesian NN."""
    np.random.seed(42)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Prior distribution
    ax1 = axes[0, 0]
    x = np.linspace(-3, 3, 100)
    prior = norm.pdf(x, 0, 1)
    ax1.plot(x, prior, 'b-', linewidth=2)
    ax1.fill_between(x, 0, prior, alpha=0.3, color='blue')
    ax1.set_xlabel('Значение веса')
    ax1.set_ylabel('Плотность вероятности')
    ax1.set_title('Prior P(w): N(0, 1)', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.text(0, np.max(prior)*0.8, 'До обучения:\nшироко распределенные веса', 
             ha='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
    # Posterior distribution (after training)
    ax2 = axes[0, 1]
    posterior = norm.pdf(x, 0.5, 0.3)
    ax2.plot(x, posterior, 'g-', linewidth=2)
    ax2.fill_between(x, 0, posterior, alpha=0.3, color='green')
    ax2.set_xlabel('Значение веса')
    ax2.set_ylabel('Плотность вероятности')
    ax2.set_title('Posterior P(w|D): N(0.5, 0.3²)', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.text(0.5, np.max(posterior)*0.8, 'После обучения:\nболее узкое распределение', 
             ha='center', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
    
    # Multiple weight samples
    ax3 = axes[1, 0]
    samples = np.random.normal(0.5, 0.3, 1000)
    ax3.hist(samples, bins=30, density=True, alpha=0.6, color='green', edgecolor='black')
    ax3.plot(x, posterior, 'r-', linewidth=2, label='Теоретическое распределение')
    ax3.set_xlabel('Значение веса')
    ax3.set_ylabel('Плотность')
    ax3.set_title('Семплы из Posterior (1000 весов)', fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Regular NN vs BNN weights
    ax4 = axes[1, 1]
    
    # Single weight value for regular NN
    regular_weight = 0.45
    ax4.axvline(regular_weight, color='red', linewidth=3, label='Обычная NN (одно значение)')
    
    # Distribution for BNN
    ax4.plot(x, posterior, 'g-', linewidth=2, label='Байесовская NN (распределение)')
    ax4.fill_between(x, 0, posterior, alpha=0.3, color='green')
    
    ax4.set_xlabel('Значение веса')
    ax4.set_ylabel('Плотность вероятности')
    ax4.set_title('Обычная NN vs Байесовская NN', fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.suptitle('Распределения весов в Байесовских нейронных сетях', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    return fig_to_base64(fig)

def generate_bnn_prediction_samples():
    """Show multiple prediction samples from BNN."""
    np.random.seed(42)
    
    # Training data
    X_train = np.array([-2, -1, 0, 1, 2])
    y_train = np.array([1, 0.5, 0, 0.5, 1])
    
    # Test range
    X_test = np.linspace(-3, 3, 100)
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Plot multiple prediction samples (different weight samples)
    n_samples = 20
    for i in range(n_samples):
        # Each sample represents predictions with different weight values
        noise = np.random.randn() * 0.1
        y_sample = X_test**2 * 0.25 + noise * np.abs(X_test) * 0.1
        alpha = 0.3 if i < n_samples - 1 else 0.8
        color = 'gray' if i < n_samples - 1 else 'red'
        lw = 1 if i < n_samples - 1 else 2
        label = 'Семплы предсказаний' if i == 0 else None
        if i == n_samples - 1:
            label = 'Последний семпл'
        ax.plot(X_test, y_sample, color=color, alpha=alpha, linewidth=lw, label=label)
    
    # Mean prediction
    y_mean = X_test**2 * 0.25
    ax.plot(X_test, y_mean, 'b-', linewidth=3, label='Среднее предсказание', zorder=10)
    
    # Training data
    ax.scatter(X_train, y_train, c='green', s=100, edgecolors='black', 
               linewidth=2, label='Обучающие данные', zorder=11)
    
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('y', fontsize=12)
    ax.set_title('Множественные предсказания Байесовской NN\n(различные семплы весов)', 
                 fontsize=13, fontweight='bold')
    ax.legend(loc='upper center', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    return fig_to_base64(fig)

# ============================================================================
# BAYESIAN INFERENCE ILLUSTRATIONS
# ============================================================================

def generate_prior_likelihood_posterior():
    """Visualize Prior, Likelihood, and Posterior."""
    np.random.seed(42)
    
    # Parameter space (e.g., coin bias)
    theta = np.linspace(0, 1, 100)
    
    # Prior (uniform or slightly informed)
    prior = beta.pdf(theta, 2, 2)
    prior = prior / np.trapezoid(prior, theta)  # normalize
    
    # Likelihood (observed 7 heads out of 10 flips)
    n_heads, n_total = 7, 10
    likelihood = theta**n_heads * (1-theta)**(n_total - n_heads)
    likelihood = likelihood / np.trapezoid(likelihood, theta)  # normalize for visualization
    
    # Posterior (Beta distribution)
    posterior = beta.pdf(theta, 2 + n_heads, 2 + n_total - n_heads)
    posterior = posterior / np.trapezoid(posterior, theta)  # normalize
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Prior
    ax1 = axes[0, 0]
    ax1.plot(theta, prior, 'b-', linewidth=2)
    ax1.fill_between(theta, 0, prior, alpha=0.3, color='blue')
    ax1.set_xlabel('θ (вероятность выпадения орла)', fontsize=10)
    ax1.set_ylabel('Плотность вероятности', fontsize=10)
    ax1.set_title('Prior P(θ): наше начальное знание', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.axvline(0.5, color='red', linestyle='--', alpha=0.5, label='θ = 0.5 (честная монета)')
    ax1.legend()
    
    # Likelihood
    ax2 = axes[0, 1]
    ax2.plot(theta, likelihood, 'g-', linewidth=2)
    ax2.fill_between(theta, 0, likelihood, alpha=0.3, color='green')
    ax2.set_xlabel('θ (вероятность выпадения орла)', fontsize=10)
    ax2.set_ylabel('Likelihood', fontsize=10)
    ax2.set_title(f'Likelihood P(D|θ): {n_heads} орлов из {n_total} бросков', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.axvline(n_heads/n_total, color='red', linestyle='--', alpha=0.5, 
                label=f'MLE: θ = {n_heads/n_total:.2f}')
    ax2.legend()
    
    # Posterior
    ax3 = axes[1, 0]
    ax3.plot(theta, posterior, 'r-', linewidth=2)
    ax3.fill_between(theta, 0, posterior, alpha=0.3, color='red')
    ax3.set_xlabel('θ (вероятность выпадения орла)', fontsize=10)
    ax3.set_ylabel('Плотность вероятности', fontsize=10)
    ax3.set_title('Posterior P(θ|D): обновленное знание', fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # MAP estimate
    map_estimate = theta[np.argmax(posterior)]
    ax3.axvline(map_estimate, color='blue', linestyle='--', linewidth=2, 
                label=f'MAP: θ = {map_estimate:.2f}')
    ax3.legend()
    
    # All together
    ax4 = axes[1, 1]
    ax4.plot(theta, prior, 'b-', linewidth=2, label='Prior', alpha=0.7)
    ax4.plot(theta, likelihood / likelihood.max() * posterior.max(), 'g--', 
             linewidth=2, label='Likelihood (scaled)', alpha=0.7)
    ax4.plot(theta, posterior, 'r-', linewidth=3, label='Posterior', alpha=0.8)
    ax4.set_xlabel('θ (вероятность выпадения орла)', fontsize=10)
    ax4.set_ylabel('Плотность (нормализованная)', fontsize=10)
    ax4.set_title('Байесовское обновление: Prior × Likelihood = Posterior', fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.suptitle('Байесовский вывод: Prior → Likelihood → Posterior', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    return fig_to_base64(fig)

def generate_conjugate_priors():
    """Illustrate conjugate priors."""
    theta = np.linspace(0, 1, 100)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    
    # Different prior strengths
    priors = [
        (1, 1, 'Uniform Prior: Beta(1, 1)'),
        (2, 2, 'Weak Prior: Beta(2, 2)'),
        (10, 10, 'Strong Prior: Beta(10, 10)'),
    ]
    
    # Observed data: 7 heads out of 10
    n_heads, n_flips = 7, 10
    
    for idx, (alpha, beta_param, title) in enumerate(priors):
        # Prior
        ax_prior = axes[0, idx]
        prior = beta.pdf(theta, alpha, beta_param)
        ax_prior.plot(theta, prior, 'b-', linewidth=2)
        ax_prior.fill_between(theta, 0, prior, alpha=0.3, color='blue')
        ax_prior.set_title(title, fontweight='bold', fontsize=10)
        ax_prior.set_xlabel('θ')
        ax_prior.set_ylabel('P(θ)')
        ax_prior.grid(True, alpha=0.3)
        ax_prior.set_ylim([0, max(prior) * 1.2])
        
        # Posterior
        ax_post = axes[1, idx]
        alpha_post = alpha + n_heads
        beta_post = beta_param + (n_flips - n_heads)
        posterior = beta.pdf(theta, alpha_post, beta_post)
        ax_post.plot(theta, posterior, 'r-', linewidth=2)
        ax_post.fill_between(theta, 0, posterior, alpha=0.3, color='red')
        ax_post.set_title(f'Posterior: Beta({alpha_post}, {beta_post})', 
                         fontweight='bold', fontsize=10)
        ax_post.set_xlabel('θ')
        ax_post.set_ylabel('P(θ|D)')
        ax_post.axvline(n_heads/n_flips, color='green', linestyle='--', 
                       label=f'Observed: {n_heads}/{n_flips}')
        ax_post.grid(True, alpha=0.3)
        ax_post.set_ylim([0, max(posterior) * 1.2])
        ax_post.legend(fontsize=8)
    
    plt.suptitle(f'Conjugate Priors: Beta-Binomial (данные: {n_heads} орлов из {n_flips})', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    return fig_to_base64(fig)

def generate_mcmc_sampling():
    """Visualize MCMC sampling process."""
    np.random.seed(42)
    
    # Target distribution (2D Gaussian mixture)
    def target_distribution(x, y):
        g1 = multivariate_normal([0, 0], [[1, 0.5], [0.5, 1]])
        g2 = multivariate_normal([3, 3], [[1, -0.3], [-0.3, 1]])
        return 0.6 * g1.pdf(np.dstack([x, y])) + 0.4 * g2.pdf(np.dstack([x, y]))
    
    # Create grid
    x = np.linspace(-3, 6, 100)
    y = np.linspace(-3, 6, 100)
    X, Y = np.meshgrid(x, y)
    Z = target_distribution(X, Y)
    
    # Simulate MCMC samples (simplified random walk)
    n_samples = 1000
    samples = np.zeros((n_samples, 2))
    samples[0] = [0, 0]
    
    for i in range(1, n_samples):
        proposal = samples[i-1] + np.random.randn(2) * 0.5
        # Accept/reject (simplified)
        current_prob = target_distribution(samples[i-1, 0], samples[i-1, 1])
        proposal_prob = target_distribution(proposal[0], proposal[1])
        
        if proposal_prob > current_prob * np.random.rand():
            samples[i] = proposal
        else:
            samples[i] = samples[i-1]
    
    fig = plt.figure(figsize=(14, 6))
    
    # Plot 1: Target distribution
    ax1 = plt.subplot(1, 2, 1)
    contour = ax1.contourf(X, Y, Z, levels=20, cmap='viridis', alpha=0.7)
    ax1.contour(X, Y, Z, levels=10, colors='black', alpha=0.3, linewidths=0.5)
    plt.colorbar(contour, ax=ax1, label='Плотность вероятности')
    ax1.set_xlabel('θ₁')
    ax1.set_ylabel('θ₂')
    ax1.set_title('Целевое распределение P(θ|D)', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: MCMC samples
    ax2 = plt.subplot(1, 2, 2)
    contour2 = ax2.contourf(X, Y, Z, levels=20, cmap='viridis', alpha=0.3)
    
    # Plot MCMC chain
    burn_in = 100
    ax2.plot(samples[:burn_in, 0], samples[:burn_in, 1], 'r-', alpha=0.3, 
             linewidth=0.5, label='Burn-in (первые 100)')
    ax2.plot(samples[burn_in:, 0], samples[burn_in:, 1], 'b-', alpha=0.2, linewidth=0.5)
    ax2.scatter(samples[burn_in::10, 0], samples[burn_in::10, 1], 
                c='blue', s=10, alpha=0.6, label='MCMC семплы')
    ax2.scatter(samples[0, 0], samples[0, 1], c='red', s=100, 
                marker='*', label='Старт', zorder=5)
    
    ax2.set_xlabel('θ₁')
    ax2.set_ylabel('θ₂')
    ax2.set_title('MCMC Sampling (Metropolis-Hastings)', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle('MCMC: семплирование из сложного распределения', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    return fig_to_base64(fig)

# ============================================================================
# GAUSSIAN PROCESSES ILLUSTRATIONS
# ============================================================================

def generate_gp_regression():
    """Visualize Gaussian Process regression."""
    np.random.seed(42)
    
    # Training data
    X_train = np.array([-4, -3, -1, 0, 2, 3, 5]).reshape(-1, 1)
    y_train = np.sin(X_train).ravel() + np.random.randn(7) * 0.1
    
    # Test points
    X_test = np.linspace(-5, 6, 100).reshape(-1, 1)
    
    # Simple GP prediction (using RBF kernel approximation)
    from scipy.spatial.distance import cdist
    
    def rbf_kernel(X1, X2, length_scale=1.0, variance=1.0):
        dists = cdist(X1, X2, 'euclidean')
        return variance * np.exp(-0.5 * (dists / length_scale)**2)
    
    K_train = rbf_kernel(X_train, X_train, length_scale=1.0) + 1e-8 * np.eye(len(X_train))
    K_test = rbf_kernel(X_test, X_train, length_scale=1.0)
    K_test_test = rbf_kernel(X_test, X_test, length_scale=1.0)
    
    # GP mean and variance
    K_inv = np.linalg.inv(K_train)
    mu = K_test @ K_inv @ y_train
    cov = K_test_test - K_test @ K_inv @ K_test.T
    std = np.sqrt(np.diag(cov))
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot 1: GP with confidence intervals
    ax1 = axes[0]
    ax1.plot(X_test, mu, 'b-', linewidth=2, label='GP среднее предсказание')
    ax1.fill_between(X_test.ravel(), mu - 1.96*std, mu + 1.96*std, 
                      alpha=0.3, color='blue', label='95% доверительный интервал')
    ax1.fill_between(X_test.ravel(), mu - std, mu + std, 
                      alpha=0.5, color='lightblue', label='68% доверительный интервал')
    ax1.scatter(X_train, y_train, c='red', s=100, edgecolors='black', 
                linewidth=2, label='Обучающие данные', zorder=5)
    ax1.set_xlabel('x')
    ax1.set_ylabel('f(x)')
    ax1.set_title('Gaussian Process Regression', fontweight='bold', fontsize=13)
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Sample functions from GP
    ax2 = axes[1]
    ax2.scatter(X_train, y_train, c='red', s=100, edgecolors='black', 
                linewidth=2, label='Обучающие данные', zorder=5)
    
    # Draw multiple samples from GP
    n_samples = 5
    samples = np.random.multivariate_normal(mu, cov + 1e-6*np.eye(len(mu)), n_samples)
    
    for i, sample in enumerate(samples):
        ax2.plot(X_test, sample, alpha=0.6, linewidth=1.5, 
                 label=f'Семпл {i+1}' if i < 3 else None)
    
    ax2.plot(X_test, mu, 'b-', linewidth=3, label='Среднее', zorder=4)
    ax2.set_xlabel('x')
    ax2.set_ylabel('f(x)')
    ax2.set_title('Семплы функций из GP Posterior', fontweight='bold', fontsize=13)
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle('Gaussian Process: предсказание с неопределенностью', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    return fig_to_base64(fig)

def generate_gp_kernels():
    """Compare different GP kernels."""
    np.random.seed(42)
    
    X_train = np.array([[-2], [0], [2]])
    y_train = np.array([0, 1, 0])
    
    X_test = np.linspace(-4, 4, 100).reshape(-1, 1)
    
    from scipy.spatial.distance import cdist
    
    def rbf_kernel(X1, X2, length_scale=1.0):
        dists = cdist(X1, X2, 'euclidean')
        return np.exp(-0.5 * (dists / length_scale)**2)
    
    def periodic_kernel(X1, X2, period=2.0, length_scale=1.0):
        dists = cdist(X1, X2, 'euclidean')
        return np.exp(-2 * (np.sin(np.pi * dists / period) / length_scale)**2)
    
    def linear_kernel(X1, X2):
        return X1 @ X2.T
    
    def matern_kernel(X1, X2, length_scale=1.0):
        dists = cdist(X1, X2, 'euclidean') / length_scale
        return (1 + dists) * np.exp(-dists)
    
    kernels = [
        (rbf_kernel, 'RBF (Radial Basis Function)'),
        (periodic_kernel, 'Periodic'),
        (linear_kernel, 'Linear'),
        (matern_kernel, 'Matérn'),
    ]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for ax, (kernel_func, kernel_name) in zip(axes, kernels):
        # Compute GP
        K_train = kernel_func(X_train, X_train) + 1e-8 * np.eye(len(X_train))
        K_test = kernel_func(X_test, X_train)
        
        K_inv = np.linalg.inv(K_train)
        mu = K_test @ K_inv @ y_train
        
        # Approximate variance for visualization
        K_test_test = kernel_func(X_test, X_test)
        cov = K_test_test - K_test @ K_inv @ K_test.T
        std = np.sqrt(np.abs(np.diag(cov)))
        
        # Plot
        ax.plot(X_test, mu, 'b-', linewidth=2, label='GP предсказание')
        ax.fill_between(X_test.ravel(), mu - 2*std, mu + 2*std, 
                         alpha=0.3, color='blue', label='95% CI')
        ax.scatter(X_train, y_train, c='red', s=100, edgecolors='black', 
                   linewidth=2, label='Данные', zorder=5)
        ax.set_xlabel('x')
        ax.set_ylabel('f(x)')
        ax.set_title(f'Kernel: {kernel_name}', fontweight='bold')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([-2, 2])
    
    plt.suptitle('Различные Kernel Functions в Gaussian Processes', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    return fig_to_base64(fig)

def generate_gp_hyperparameters():
    """Show effect of GP hyperparameters."""
    np.random.seed(42)
    
    X_train = np.array([[-2], [0], [2]])
    y_train = np.array([0, 1, 0])
    X_test = np.linspace(-4, 4, 100).reshape(-1, 1)
    
    from scipy.spatial.distance import cdist
    
    def rbf_kernel(X1, X2, length_scale=1.0, variance=1.0):
        dists = cdist(X1, X2, 'euclidean')
        return variance * np.exp(-0.5 * (dists / length_scale)**2)
    
    # Different length scales
    length_scales = [0.5, 1.0, 2.0]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for ax, ls in zip(axes, length_scales):
        K_train = rbf_kernel(X_train, X_train, length_scale=ls) + 1e-8 * np.eye(len(X_train))
        K_test = rbf_kernel(X_test, X_train, length_scale=ls)
        K_test_test = rbf_kernel(X_test, X_test, length_scale=ls)
        
        K_inv = np.linalg.inv(K_train)
        mu = K_test @ K_inv @ y_train
        cov = K_test_test - K_test @ K_inv @ K_test.T
        std = np.sqrt(np.diag(cov))
        
        ax.plot(X_test, mu, 'b-', linewidth=2, label='GP предсказание')
        ax.fill_between(X_test.ravel(), mu - 1.96*std, mu + 1.96*std, 
                         alpha=0.3, color='blue', label='95% CI')
        ax.scatter(X_train, y_train, c='red', s=100, edgecolors='black', 
                   linewidth=2, label='Данные', zorder=5)
        ax.set_xlabel('x')
        ax.set_ylabel('f(x)')
        ax.set_title(f'Length scale = {ls}', fontweight='bold', fontsize=12)
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([-2, 2.5])
        
        # Add description
        if ls == 0.5:
            desc = 'Короткий length scale:\nбыстрые изменения'
        elif ls == 1.0:
            desc = 'Средний length scale:\nумеренная гладкость'
        else:
            desc = 'Длинный length scale:\nплавные изменения'
        
        ax.text(0.5, 0.95, desc, transform=ax.transAxes, 
                ha='center', va='top', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
    plt.suptitle('Влияние Length Scale на Gaussian Process', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    return fig_to_base64(fig)

# ============================================================================
# MAIN FUNCTION
# ============================================================================

def generate_all_illustrations():
    """Generate all Bayesian methods illustrations."""
    print("Generating Bayesian methods illustrations...")
    
    illustrations = {}
    
    print("  - Generating Bayesian Optimization illustrations...")
    illustrations['bo_process'] = generate_bayesian_optimization_process()
    print("    ✓ Bayesian Optimization process")
    
    illustrations['bo_acquisition'] = generate_acquisition_functions_comparison()
    print("    ✓ Acquisition functions comparison")
    
    illustrations['bo_iterations'] = generate_bayesian_optimization_iterations()
    print("    ✓ Optimization iterations")
    
    print("  - Generating Bayesian Neural Networks illustrations...")
    illustrations['bnn_uncertainty'] = generate_bnn_uncertainty_comparison()
    print("    ✓ Uncertainty comparison")
    
    illustrations['bnn_weights'] = generate_weight_distributions()
    print("    ✓ Weight distributions")
    
    illustrations['bnn_samples'] = generate_bnn_prediction_samples()
    print("    ✓ Prediction samples")
    
    print("  - Generating Bayesian Inference illustrations...")
    illustrations['bi_posterior'] = generate_prior_likelihood_posterior()
    print("    ✓ Prior, Likelihood, Posterior")
    
    illustrations['bi_conjugate'] = generate_conjugate_priors()
    print("    ✓ Conjugate priors")
    
    illustrations['bi_mcmc'] = generate_mcmc_sampling()
    print("    ✓ MCMC sampling")
    
    print("  - Generating Gaussian Processes illustrations...")
    illustrations['gp_regression'] = generate_gp_regression()
    print("    ✓ GP regression")
    
    illustrations['gp_kernels'] = generate_gp_kernels()
    print("    ✓ Different kernels")
    
    illustrations['gp_hyperparams'] = generate_gp_hyperparameters()
    print("    ✓ Hyperparameter effects")
    
    print("All illustrations generated successfully!")
    return illustrations

if __name__ == '__main__':
    illustrations = generate_all_illustrations()
    print(f"\nGenerated {len(illustrations)} illustrations")
    print("Illustrations are base64 encoded and ready for HTML embedding")
