#!/usr/bin/env python3
"""
Generate matplotlib illustrations for Additional (Дополнительно) section cheatsheets.
This script creates high-quality visualizations and encodes them as base64 for inline HTML embedding.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.special import rel_entr
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
# NEURAL NETWORK TRAINING DYNAMICS ILLUSTRATIONS
# ============================================================================

def generate_loss_landscape():
    """Generate 2D loss landscape visualization."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create loss landscape
    x = np.linspace(-3, 3, 100)
    y = np.linspace(-3, 3, 100)
    X, Y = np.meshgrid(x, y)
    
    # Complex landscape with local minima and saddle points
    Z = (X**2 + Y**2) * 0.5 + np.sin(X*2) * 0.5 + np.cos(Y*2) * 0.5
    
    # Plot contours
    contour = ax.contourf(X, Y, Z, levels=20, cmap='viridis', alpha=0.8)
    ax.contour(X, Y, Z, levels=10, colors='white', alpha=0.3, linewidths=0.5)
    
    # Simulate gradient descent trajectory
    trajectory_x = [-2.5, -2.0, -1.5, -1.0, -0.5, -0.2, 0.0, 0.1]
    trajectory_y = [-2.0, -1.5, -1.0, -0.7, -0.4, -0.2, 0.0, 0.05]
    ax.plot(trajectory_x, trajectory_y, 'r-o', linewidth=2, markersize=6, 
            label='Траектория градиентного спуска', alpha=0.9)
    
    ax.set_xlabel('Параметр θ₁')
    ax.set_ylabel('Параметр θ₂')
    ax.set_title('Ландшафт функции потерь с траекторией обучения')
    ax.legend(loc='upper right')
    plt.colorbar(contour, ax=ax, label='Loss')
    
    return fig_to_base64(fig)

def generate_learning_rate_schedules():
    """Generate learning rate scheduling strategies comparison."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    epochs = np.arange(0, 100)
    
    # Step decay
    ax = axes[0, 0]
    lr_step = 0.1 * (0.1 ** (epochs // 30))
    ax.plot(epochs, lr_step, 'b-', linewidth=2)
    ax.set_title('Step Decay')
    ax.set_xlabel('Эпоха')
    ax.set_ylabel('Learning Rate')
    ax.grid(True, alpha=0.3)
    
    # Cosine annealing
    ax = axes[0, 1]
    lr_cosine = 0.1 * (1 + np.cos(np.pi * epochs / 100)) / 2
    ax.plot(epochs, lr_cosine, 'g-', linewidth=2)
    ax.set_title('Cosine Annealing')
    ax.set_xlabel('Эпоха')
    ax.set_ylabel('Learning Rate')
    ax.grid(True, alpha=0.3)
    
    # Exponential decay
    ax = axes[1, 0]
    lr_exp = 0.1 * np.exp(-0.03 * epochs)
    ax.plot(epochs, lr_exp, 'r-', linewidth=2)
    ax.set_title('Exponential Decay')
    ax.set_xlabel('Эпоха')
    ax.set_ylabel('Learning Rate')
    ax.grid(True, alpha=0.3)
    
    # Cyclic LR
    ax = axes[1, 1]
    lr_cyclic = 0.001 + (0.01 - 0.001) * (1 + np.cos(np.pi * (epochs % 20) / 20)) / 2
    ax.plot(epochs, lr_cyclic, 'm-', linewidth=2)
    ax.set_title('Cyclic Learning Rate')
    ax.set_xlabel('Эпоха')
    ax.set_ylabel('Learning Rate')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig_to_base64(fig)

def generate_training_phases():
    """Generate training phases illustration."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    epochs = np.arange(0, 100)
    
    # Training loss with phases
    train_loss = 2.5 * np.exp(-epochs / 15) + 0.3 + np.random.normal(0, 0.05, len(epochs))
    val_loss = 2.5 * np.exp(-epochs / 15) + 0.5 + np.random.normal(0, 0.07, len(epochs))
    
    ax1.plot(epochs, train_loss, 'b-', linewidth=2, label='Training Loss', alpha=0.7)
    ax1.plot(epochs, val_loss, 'r-', linewidth=2, label='Validation Loss', alpha=0.7)
    
    # Mark phases
    ax1.axvspan(0, 10, alpha=0.2, color='yellow', label='Warm-up')
    ax1.axvspan(10, 70, alpha=0.2, color='green', label='Linear regime')
    ax1.axvspan(70, 100, alpha=0.2, color='blue', label='Fine-tuning')
    
    ax1.set_xlabel('Эпоха')
    ax1.set_ylabel('Loss')
    ax1.set_title('Фазы обучения нейросети')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # Learning rate warmup
    lr = np.zeros(100)
    lr[0:10] = np.linspace(0, 0.01, 10)  # warmup
    lr[10:70] = 0.01  # constant
    lr[70:] = 0.01 * np.exp(-0.1 * (epochs[70:] - 70))  # decay
    
    ax2.plot(epochs, lr, 'g-', linewidth=2)
    ax2.axvspan(0, 10, alpha=0.2, color='yellow')
    ax2.axvspan(10, 70, alpha=0.2, color='green')
    ax2.axvspan(70, 100, alpha=0.2, color='blue')
    ax2.set_xlabel('Эпоха')
    ax2.set_ylabel('Learning Rate')
    ax2.set_title('Расписание Learning Rate')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig_to_base64(fig)

def generate_sharp_vs_flat_minima():
    """Generate sharp vs flat minima comparison."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    x = np.linspace(-2, 2, 200)
    
    # Sharp minimum
    sharp = 5 * x**2
    ax1.plot(x, sharp, 'r-', linewidth=2)
    ax1.scatter([0], [0], c='red', s=100, zorder=5)
    ax1.set_xlabel('Параметр')
    ax1.set_ylabel('Loss')
    ax1.set_title('Острый минимум (плохая обобщающая способность)')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(-1, 20)
    
    # Flat minimum
    flat = 0.3 * x**2
    ax2.plot(x, flat, 'g-', linewidth=2)
    ax2.scatter([0], [0], c='green', s=100, zorder=5)
    ax2.set_xlabel('Параметр')
    ax2.set_ylabel('Loss')
    ax2.set_title('Плоский минимум (хорошая обобщающая способность)')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(-1, 20)
    
    plt.tight_layout()
    return fig_to_base64(fig)

# ============================================================================
# INFORMATION THEORY ML ILLUSTRATIONS
# ============================================================================

def generate_entropy_visualization():
    """Generate entropy visualization for different distributions."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Low entropy (peaked distribution)
    ax = axes[0, 0]
    probs_low = np.array([0.9, 0.05, 0.03, 0.02])
    categories = ['A', 'B', 'C', 'D']
    ax.bar(categories, probs_low, color='blue', alpha=0.7)
    entropy_low = -np.sum(probs_low * np.log2(probs_low + 1e-10))
    ax.set_title(f'Низкая энтропия (H = {entropy_low:.2f} bits)')
    ax.set_ylabel('Вероятность')
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    
    # High entropy (uniform distribution)
    ax = axes[0, 1]
    probs_high = np.array([0.25, 0.25, 0.25, 0.25])
    ax.bar(categories, probs_high, color='red', alpha=0.7)
    entropy_high = -np.sum(probs_high * np.log2(probs_high + 1e-10))
    ax.set_title(f'Высокая энтропия (H = {entropy_high:.2f} bits)')
    ax.set_ylabel('Вероятность')
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    
    # Continuous distribution - Gaussian
    ax = axes[1, 0]
    x = np.linspace(-5, 5, 1000)
    for sigma in [0.5, 1.0, 2.0]:
        y = stats.norm.pdf(x, 0, sigma)
        ax.plot(x, y, linewidth=2, label=f'σ={sigma}')
    ax.set_title('Дифференциальная энтропия (Гаусс)')
    ax.set_xlabel('x')
    ax.set_ylabel('Плотность вероятности')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Entropy as function of probability
    ax = axes[1, 1]
    p = np.linspace(0.01, 0.99, 100)
    H = -p * np.log2(p) - (1-p) * np.log2(1-p)
    ax.plot(p, H, 'purple', linewidth=2)
    ax.set_title('Бинарная энтропия H(p)')
    ax.set_xlabel('Вероятность p')
    ax.set_ylabel('Энтропия (bits)')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig_to_base64(fig)

def generate_kl_divergence():
    """Generate KL divergence visualization."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    x = np.linspace(-5, 5, 1000)
    
    # Two Gaussians with different means
    p = stats.norm.pdf(x, 0, 1)
    q = stats.norm.pdf(x, 1.5, 1)
    
    ax1.plot(x, p, 'b-', linewidth=2, label='P (истинное распределение)')
    ax1.plot(x, q, 'r--', linewidth=2, label='Q (приближение)')
    ax1.fill_between(x, p, alpha=0.3, color='blue')
    ax1.fill_between(x, q, alpha=0.3, color='red')
    ax1.set_xlabel('x')
    ax1.set_ylabel('Плотность')
    ax1.set_title('KL дивергенция: D_KL(P || Q)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # KL divergence as function of mean difference
    ax2_data = []
    means = np.linspace(0, 3, 50)
    for mu in means:
        p_disc = stats.norm.pdf(np.linspace(-5, 5, 100), 0, 1)
        q_disc = stats.norm.pdf(np.linspace(-5, 5, 100), mu, 1)
        p_disc = p_disc / p_disc.sum()
        q_disc = q_disc / q_disc.sum()
        kl = np.sum(rel_entr(p_disc, q_disc))
        ax2_data.append(kl)
    
    ax2.plot(means, ax2_data, 'g-', linewidth=2)
    ax2.set_xlabel('Разница средних μ_Q - μ_P')
    ax2.set_ylabel('D_KL(P || Q)')
    ax2.set_title('KL дивергенция vs разница средних')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig_to_base64(fig)

def generate_mutual_information():
    """Generate mutual information visualization."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    np.random.seed(42)
    n = 500
    
    # High mutual information (strong correlation)
    ax = axes[0, 0]
    x1 = np.random.randn(n)
    y1 = x1 + np.random.randn(n) * 0.3
    ax.scatter(x1, y1, alpha=0.5, s=10)
    ax.set_title('Высокая взаимная информация I(X;Y)')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.grid(True, alpha=0.3)
    
    # Low mutual information (independence)
    ax = axes[0, 1]
    x2 = np.random.randn(n)
    y2 = np.random.randn(n)
    ax.scatter(x2, y2, alpha=0.5, s=10)
    ax.set_title('Низкая взаимная информация I(X;Y) ≈ 0')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.grid(True, alpha=0.3)
    
    # Venn diagram of information
    ax = axes[1, 0]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    circle1 = plt.Circle((3, 5), 2, color='blue', alpha=0.3, label='H(X)')
    circle2 = plt.Circle((7, 5), 2, color='red', alpha=0.3, label='H(Y)')
    ax.add_patch(circle1)
    ax.add_patch(circle2)
    ax.text(3, 5, 'H(X|Y)', ha='center', va='center', fontsize=10)
    ax.text(7, 5, 'H(Y|X)', ha='center', va='center', fontsize=10)
    ax.text(5, 5, 'I(X;Y)', ha='center', va='center', fontsize=10, fontweight='bold')
    ax.set_title('Декомпозиция информации')
    ax.set_aspect('equal')
    ax.axis('off')
    ax.legend(loc='upper right')
    
    # Cross-entropy vs KL divergence
    ax = axes[1, 1]
    p_true = 0.7
    q_range = np.linspace(0.01, 0.99, 100)
    cross_entropy = -p_true * np.log(q_range) - (1-p_true) * np.log(1-q_range)
    entropy_p = -p_true * np.log(p_true) - (1-p_true) * np.log(1-p_true)
    kl_div = cross_entropy - entropy_p
    
    ax.plot(q_range, cross_entropy, 'b-', linewidth=2, label='Cross-Entropy H(P,Q)')
    ax.plot(q_range, kl_div, 'r-', linewidth=2, label='KL Divergence D_KL(P||Q)')
    ax.axhline(entropy_p, color='g', linestyle='--', linewidth=2, label='Entropy H(P)')
    ax.axvline(p_true, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel('q (предсказанная вероятность)')
    ax.set_ylabel('Значение')
    ax.set_title(f'Cross-Entropy = Entropy + KL (p_true={p_true})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig_to_base64(fig)

# ============================================================================
# PROBABILITY THEORY ILLUSTRATIONS
# ============================================================================

def generate_probability_distributions():
    """Generate common probability distributions."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Bernoulli
    ax = axes[0, 0]
    p = 0.7
    x_bern = [0, 1]
    y_bern = [1-p, p]
    ax.bar(x_bern, y_bern, color='blue', alpha=0.7, width=0.3)
    ax.set_title(f'Бернулли (p={p})')
    ax.set_xlabel('x')
    ax.set_ylabel('P(X=x)')
    ax.set_xticks([0, 1])
    ax.grid(True, alpha=0.3)
    
    # Binomial
    ax = axes[0, 1]
    n, p = 10, 0.5
    x_binom = np.arange(0, n+1)
    y_binom = stats.binom.pmf(x_binom, n, p)
    ax.bar(x_binom, y_binom, color='green', alpha=0.7)
    ax.set_title(f'Биномиальное (n={n}, p={p})')
    ax.set_xlabel('x')
    ax.set_ylabel('P(X=x)')
    ax.grid(True, alpha=0.3)
    
    # Poisson
    ax = axes[0, 2]
    lam = 3
    x_pois = np.arange(0, 15)
    y_pois = stats.poisson.pmf(x_pois, lam)
    ax.bar(x_pois, y_pois, color='red', alpha=0.7)
    ax.set_title(f'Пуассона (λ={lam})')
    ax.set_xlabel('x')
    ax.set_ylabel('P(X=x)')
    ax.grid(True, alpha=0.3)
    
    # Gaussian (Normal)
    ax = axes[1, 0]
    x_norm = np.linspace(-4, 4, 200)
    for mu, sigma in [(0, 1), (0, 0.5), (1, 1)]:
        y_norm = stats.norm.pdf(x_norm, mu, sigma)
        ax.plot(x_norm, y_norm, linewidth=2, label=f'μ={mu}, σ={sigma}')
    ax.set_title('Нормальное (Гаусс)')
    ax.set_xlabel('x')
    ax.set_ylabel('f(x)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Exponential
    ax = axes[1, 1]
    x_exp = np.linspace(0, 5, 200)
    for lam in [0.5, 1.0, 2.0]:
        y_exp = stats.expon.pdf(x_exp, scale=1/lam)
        ax.plot(x_exp, y_exp, linewidth=2, label=f'λ={lam}')
    ax.set_title('Экспоненциальное')
    ax.set_xlabel('x')
    ax.set_ylabel('f(x)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Beta
    ax = axes[1, 2]
    x_beta = np.linspace(0, 1, 200)
    for a, b in [(2, 2), (5, 2), (2, 5)]:
        y_beta = stats.beta.pdf(x_beta, a, b)
        ax.plot(x_beta, y_beta, linewidth=2, label=f'α={a}, β={b}')
    ax.set_title('Бета-распределение')
    ax.set_xlabel('x')
    ax.set_ylabel('f(x)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig_to_base64(fig)

def generate_bayes_theorem():
    """Generate Bayes theorem visualization."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Bayes theorem with tree diagram
    ax = axes[0]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    
    # Prior probabilities
    ax.text(1, 8, 'Prior', fontsize=12, fontweight='bold')
    ax.text(1, 7, 'P(H) = 0.3', fontsize=10, bbox=dict(boxstyle='round', facecolor='lightblue'))
    ax.text(1, 6, 'P(¬H) = 0.7', fontsize=10, bbox=dict(boxstyle='round', facecolor='lightcoral'))
    
    # Likelihoods
    ax.text(5, 8.5, 'P(E|H) = 0.9', fontsize=10, bbox=dict(boxstyle='round', facecolor='lightgreen'))
    ax.text(5, 7, 'P(E|¬H) = 0.2', fontsize=10, bbox=dict(boxstyle='round', facecolor='lightyellow'))
    
    # Posterior
    prior_h = 0.3
    prior_not_h = 0.7
    like_e_h = 0.9
    like_e_not_h = 0.2
    evidence = prior_h * like_e_h + prior_not_h * like_e_not_h
    posterior = (like_e_h * prior_h) / evidence
    
    ax.text(8, 7.5, 'Posterior', fontsize=12, fontweight='bold')
    ax.text(8, 6.5, f'P(H|E) = {posterior:.3f}', fontsize=11, 
            bbox=dict(boxstyle='round', facecolor='lightgreen', linewidth=2))
    
    # Arrows
    ax.annotate('', xy=(4.5, 8.3), xytext=(2.5, 7.2),
                arrowprops=dict(arrowstyle='->', lw=1.5))
    ax.annotate('', xy=(4.5, 7), xytext=(2.5, 6.2),
                arrowprops=dict(arrowstyle='->', lw=1.5))
    ax.annotate('', xy=(7, 7), xytext=(6, 7.5),
                arrowprops=dict(arrowstyle='->', lw=1.5, color='red'))
    
    ax.set_title('Теорема Байеса: P(H|E) = P(E|H)·P(H) / P(E)')
    ax.axis('off')
    
    # Prior to posterior update
    ax = axes[1]
    x = np.linspace(0, 1, 200)
    prior = stats.beta.pdf(x, 2, 2)
    posterior_dist = stats.beta.pdf(x, 12, 4)
    
    ax.plot(x, prior, 'b-', linewidth=2, label='Prior P(θ)')
    ax.plot(x, posterior_dist, 'r-', linewidth=2, label='Posterior P(θ|D)')
    ax.fill_between(x, prior, alpha=0.3, color='blue')
    ax.fill_between(x, posterior_dist, alpha=0.3, color='red')
    ax.set_xlabel('Параметр θ')
    ax.set_ylabel('Плотность')
    ax.set_title('Обновление убеждений через теорему Байеса')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig_to_base64(fig)

def generate_central_limit_theorem():
    """Generate Central Limit Theorem visualization."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    np.random.seed(42)
    n_samples = 10000
    
    # Original uniform distribution
    ax = axes[0, 0]
    data = np.random.uniform(0, 1, n_samples)
    ax.hist(data, bins=50, density=True, alpha=0.7, color='blue', edgecolor='black')
    ax.set_title('Исходное распределение (равномерное)')
    ax.set_xlabel('x')
    ax.set_ylabel('Плотность')
    ax.grid(True, alpha=0.3)
    
    # Means of samples (n=5)
    ax = axes[0, 1]
    sample_means_5 = [np.mean(np.random.uniform(0, 1, 5)) for _ in range(n_samples)]
    ax.hist(sample_means_5, bins=50, density=True, alpha=0.7, color='green', edgecolor='black')
    x_norm = np.linspace(0, 1, 100)
    ax.plot(x_norm, stats.norm.pdf(x_norm, 0.5, np.std(sample_means_5)), 'r-', linewidth=2, label='Нормальное')
    ax.set_title('Среднее выборок (n=5)')
    ax.set_xlabel('x̄')
    ax.set_ylabel('Плотность')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Means of samples (n=30)
    ax = axes[1, 0]
    sample_means_30 = [np.mean(np.random.uniform(0, 1, 30)) for _ in range(n_samples)]
    ax.hist(sample_means_30, bins=50, density=True, alpha=0.7, color='orange', edgecolor='black')
    ax.plot(x_norm, stats.norm.pdf(x_norm, 0.5, np.std(sample_means_30)), 'r-', linewidth=2, label='Нормальное')
    ax.set_title('Среднее выборок (n=30)')
    ax.set_xlabel('x̄')
    ax.set_ylabel('Плотность')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Q-Q plot
    ax = axes[1, 1]
    stats.probplot(sample_means_30, dist="norm", plot=ax)
    ax.set_title('Q-Q график (проверка нормальности)')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig_to_base64(fig)

# ============================================================================
# GRADIENT CHECKPOINTING ILLUSTRATIONS
# ============================================================================

def generate_gradient_checkpointing_memory():
    """Generate memory usage comparison for gradient checkpointing."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    layers = np.arange(1, 11)
    
    # Standard backpropagation memory
    memory_standard = layers ** 2 * 10  # Quadratic growth
    ax1.bar(layers, memory_standard, color='red', alpha=0.7, label='Активации в памяти')
    ax1.set_xlabel('Номер слоя')
    ax1.set_ylabel('Использование памяти (отн. ед.)')
    ax1.set_title('Стандартный Backprop (высокое использование памяти)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Gradient checkpointing memory
    memory_checkpoint = layers * 10 + np.where(layers % 3 == 0, 50, 0)  # Linear with checkpoints
    ax2.bar(layers, memory_checkpoint, color='green', alpha=0.7, label='Активации (с checkpoint)')
    checkpoint_layers = layers[layers % 3 == 0]
    checkpoint_vals = memory_checkpoint[layers % 3 == 0]
    ax2.scatter(checkpoint_layers, checkpoint_vals, color='blue', s=100, 
                marker='*', label='Checkpoint точки', zorder=5)
    ax2.set_xlabel('Номер слоя')
    ax2.set_ylabel('Использование памяти (отн. ед.)')
    ax2.set_title('Gradient Checkpointing (экономия памяти)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig_to_base64(fig)

def generate_checkpointing_computation_trade():
    """Generate computation vs memory trade-off."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Trade-off curve
    memory_saved = np.linspace(0, 90, 100)
    computation_overhead = 1 + (memory_saved / 50) ** 1.5  # Non-linear increase
    
    ax.plot(memory_saved, computation_overhead, 'b-', linewidth=3)
    ax.fill_between(memory_saved, 1, computation_overhead, alpha=0.3, color='blue')
    
    # Mark sweet spot
    sweet_spot_mem = 50
    sweet_spot_comp = 1 + (sweet_spot_mem / 50) ** 1.5
    ax.scatter([sweet_spot_mem], [sweet_spot_comp], color='red', s=200, 
               marker='*', zorder=5, label='Оптимальная точка')
    
    ax.set_xlabel('Экономия памяти (%)')
    ax.set_ylabel('Накладные расходы на вычисления (×)')
    ax.set_title('Компромисс: память vs вычисления в Gradient Checkpointing')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(y=1, color='gray', linestyle='--', alpha=0.5, label='Базовая линия')
    
    return fig_to_base64(fig)

def generate_checkpointing_layers():
    """Generate layer-wise checkpointing visualization."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    layers = ['Input', 'Layer 1', 'Layer 2', 'Layer 3*', 'Layer 4', 
              'Layer 5', 'Layer 6*', 'Layer 7', 'Layer 8', 'Layer 9*', 'Output']
    y_pos = np.arange(len(layers))
    
    colors = ['lightblue' if '*' not in layer else 'lightgreen' for layer in layers]
    
    ax.barh(y_pos, [1]*len(layers), color=colors, alpha=0.7, edgecolor='black')
    
    # Mark checkpoints
    checkpoint_indices = [i for i, layer in enumerate(layers) if '*' in layer]
    for idx in checkpoint_indices:
        ax.scatter([0.5], [idx], marker='*', s=500, color='red', zorder=5)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(layers)
    ax.set_xlabel('Forward Pass')
    ax.set_title('Gradient Checkpointing: сохранение активаций (* = checkpoint)')
    ax.set_xlim(0, 1)
    ax.grid(True, alpha=0.3, axis='x')
    
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='lightblue', label='Обычный слой'),
        Patch(facecolor='lightgreen', label='Checkpoint слой'),
    ]
    ax.legend(handles=legend_elements, loc='lower right')
    
    return fig_to_base64(fig)

# ============================================================================
# MIXED PRECISION TRAINING ILLUSTRATIONS
# ============================================================================

def generate_precision_formats():
    """Generate precision format comparison."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    formats = ['FP32', 'FP16', 'BF16', 'INT8']
    memory = [32, 16, 16, 8]
    colors_mem = ['red', 'orange', 'yellow', 'green']
    
    ax1.bar(formats, memory, color=colors_mem, alpha=0.7, edgecolor='black')
    ax1.set_ylabel('Биты')
    ax1.set_title('Размер представления чисел')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Range and precision comparison
    range_vals = [1e38, 65504, 3.4e38, 128]  # Approximate max values
    ax2.bar(formats, np.log10(range_vals), color=colors_mem, alpha=0.7, edgecolor='black')
    ax2.set_ylabel('log₁₀(Максимальное значение)')
    ax2.set_title('Динамический диапазон')
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    return fig_to_base64(fig)

def generate_mixed_precision_speedup():
    """Generate speedup comparison."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    models = ['ResNet-50', 'BERT-Base', 'GPT-2', 'ViT-B/16', 'T5-Large']
    speedup_fp16 = [1.8, 2.1, 2.3, 1.9, 2.5]
    speedup_mixed = [2.5, 2.9, 3.1, 2.7, 3.3]
    
    x = np.arange(len(models))
    width = 0.35
    
    ax.bar(x - width/2, speedup_fp16, width, label='FP16', color='orange', alpha=0.7)
    ax.bar(x + width/2, speedup_mixed, width, label='Mixed Precision', color='green', alpha=0.7)
    
    ax.set_xlabel('Модель')
    ax.set_ylabel('Ускорение (×)')
    ax.set_title('Ускорение обучения с Mixed Precision Training')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    return fig_to_base64(fig)

def generate_loss_scaling():
    """Generate loss scaling visualization."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Without loss scaling - gradient underflow
    ax1.set_xlim(0, 10)
    ax1.set_ylim(-5, 5)
    
    epochs = np.arange(0, 100)
    gradients_no_scale = 0.0001 * np.random.randn(100)  # Very small gradients
    gradients_no_scale[gradients_no_scale < 0.00005] = 0  # Underflow
    
    ax1.hist(gradients_no_scale, bins=30, alpha=0.7, color='red', edgecolor='black')
    ax1.set_title('Без Loss Scaling (gradient underflow)')
    ax1.set_xlabel('Значение градиента')
    ax1.set_ylabel('Частота')
    ax1.axvline(x=0, color='black', linestyle='--', alpha=0.5)
    ax1.grid(True, alpha=0.3)
    
    # With loss scaling
    gradients_scaled = 0.0001 * 1000 * np.random.randn(100)  # Scaled up
    ax2.hist(gradients_scaled, bins=30, alpha=0.7, color='green', edgecolor='black')
    ax2.set_title('С Loss Scaling (preserved gradients)')
    ax2.set_xlabel('Значение градиента (scaled)')
    ax2.set_ylabel('Частота')
    ax2.axvline(x=0, color='black', linestyle='--', alpha=0.5)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig_to_base64(fig)

# ============================================================================
# MODEL COMPRESSION ILLUSTRATIONS
# ============================================================================

def generate_compression_techniques():
    """Generate model compression techniques comparison."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    techniques = ['Baseline', 'Pruning', 'Quantization', 'Knowledge\nDistillation', 
                  'Low-Rank\nFactorization', 'Combined']
    model_size = [100, 60, 25, 80, 70, 20]
    accuracy = [95, 94, 93, 94.5, 94.2, 92]
    
    x = np.arange(len(techniques))
    width = 0.35
    
    ax.bar(x - width/2, model_size, width, label='Размер модели (%)', color='blue', alpha=0.7)
    ax2 = ax.twinx()
    ax2.bar(x + width/2, accuracy, width, label='Точность (%)', color='green', alpha=0.7)
    
    ax.set_xlabel('Техника сжатия')
    ax.set_ylabel('Размер модели (%)', color='blue')
    ax2.set_ylabel('Точность (%)', color='green')
    ax.set_title('Сравнение методов сжатия моделей')
    ax.set_xticks(x)
    ax.set_xticklabels(techniques)
    ax.tick_params(axis='y', labelcolor='blue')
    ax2.tick_params(axis='y', labelcolor='green')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Legends
    ax.legend(loc='upper left')
    ax2.legend(loc='upper right')
    
    plt.tight_layout()
    return fig_to_base64(fig)

def generate_pruning_visualization():
    """Generate pruning visualization."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Weight distribution before pruning
    np.random.seed(42)
    weights_before = np.random.randn(10000) * 0.1
    ax1.hist(weights_before, bins=50, alpha=0.7, color='blue', edgecolor='black')
    ax1.axvline(x=-0.05, color='red', linestyle='--', linewidth=2, label='Порог pruning')
    ax1.axvline(x=0.05, color='red', linestyle='--', linewidth=2)
    ax1.set_xlabel('Значение веса')
    ax1.set_ylabel('Частота')
    ax1.set_title('До Pruning (100% весов)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Weight distribution after pruning
    weights_after = weights_before.copy()
    weights_after[(weights_after > -0.05) & (weights_after < 0.05)] = 0
    
    # Split into zero and non-zero for visualization
    weights_nonzero = weights_after[weights_after != 0]
    weights_zero_count = len(weights_after[weights_after == 0])
    
    ax2.hist(weights_nonzero, bins=50, alpha=0.7, color='green', edgecolor='black', label='Ненулевые веса')
    ax2.bar([0], [weights_zero_count], width=0.01, alpha=0.7, color='red', label=f'Обнулено: {weights_zero_count}')
    ax2.set_xlabel('Значение веса')
    ax2.set_ylabel('Частота')
    ax2.set_title(f'После Pruning ({len(weights_nonzero)/len(weights_before)*100:.1f}% весов)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig_to_base64(fig)

def generate_quantization_visualization():
    """Generate quantization visualization."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Signal before quantization (FP32)
    x = np.linspace(0, 4*np.pi, 1000)
    signal_fp32 = np.sin(x)
    
    ax1.plot(x, signal_fp32, 'b-', linewidth=2, label='FP32 (full precision)')
    ax1.set_xlabel('x')
    ax1.set_ylabel('Значение активации')
    ax1.set_title('Полная точность (FP32)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Signal after quantization (INT8)
    levels = 256  # 8-bit
    signal_quantized = np.round(signal_fp32 * (levels/2)) / (levels/2)
    signal_quantized = np.clip(signal_quantized, -1, 1)
    
    ax2.plot(x, signal_fp32, 'b-', linewidth=1, alpha=0.5, label='Оригинал (FP32)')
    ax2.plot(x, signal_quantized, 'r-', linewidth=2, label='Квантованный (INT8)')
    ax2.set_xlabel('x')
    ax2.set_ylabel('Значение активации')
    ax2.set_title('После квантизации (INT8)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig_to_base64(fig)

def generate_knowledge_distillation():
    """Generate knowledge distillation visualization."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Teacher vs Student softmax outputs
    classes = ['Cat', 'Dog', 'Bird', 'Fish', 'Horse']
    teacher_hard = np.array([0.9, 0.05, 0.02, 0.02, 0.01])
    teacher_soft = np.array([0.4, 0.25, 0.15, 0.12, 0.08])  # With temperature
    student_output = np.array([0.45, 0.22, 0.14, 0.11, 0.08])
    
    x = np.arange(len(classes))
    width = 0.25
    
    ax1.bar(x - width, teacher_hard, width, label='Teacher (T=1)', color='blue', alpha=0.7)
    ax1.bar(x, teacher_soft, width, label='Teacher (T=3)', color='orange', alpha=0.7)
    ax1.bar(x + width, student_output, width, label='Student', color='green', alpha=0.7)
    
    ax1.set_xlabel('Класс')
    ax1.set_ylabel('Вероятность')
    ax1.set_title('Knowledge Distillation: Teacher → Student')
    ax1.set_xticks(x)
    ax1.set_xticklabels(classes)
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Model size vs accuracy trade-off
    sizes = [100, 80, 60, 40, 20, 10]
    acc_scratch = [95, 93, 89, 83, 75, 65]
    acc_distill = [95, 94, 92, 88, 82, 74]
    
    ax2.plot(sizes, acc_scratch, 'r-o', linewidth=2, markersize=8, label='Обучение с нуля')
    ax2.plot(sizes, acc_distill, 'g-o', linewidth=2, markersize=8, label='Knowledge Distillation')
    ax2.set_xlabel('Размер модели (% от teacher)')
    ax2.set_ylabel('Точность (%)')
    ax2.set_title('Преимущество Knowledge Distillation')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.invert_xaxis()
    
    plt.tight_layout()
    return fig_to_base64(fig)

# ============================================================================
# MAIN FUNCTION
# ============================================================================

def generate_all_illustrations():
    """Generate all illustrations and return as dictionary."""
    print("Generating Neural Network Training Dynamics illustrations...")
    illustrations = {
        # Neural Network Training Dynamics
        'nn_loss_landscape': generate_loss_landscape(),
        'nn_lr_schedules': generate_learning_rate_schedules(),
        'nn_training_phases': generate_training_phases(),
        'nn_sharp_flat_minima': generate_sharp_vs_flat_minima(),
        
        # Information Theory ML
        'it_entropy': generate_entropy_visualization(),
        'it_kl_divergence': generate_kl_divergence(),
        'it_mutual_info': generate_mutual_information(),
        
        # Probability Theory
        'prob_distributions': generate_probability_distributions(),
        'prob_bayes': generate_bayes_theorem(),
        'prob_clt': generate_central_limit_theorem(),
        
        # Gradient Checkpointing
        'gc_memory': generate_gradient_checkpointing_memory(),
        'gc_trade_off': generate_checkpointing_computation_trade(),
        'gc_layers': generate_checkpointing_layers(),
        
        # Mixed Precision Training
        'mp_formats': generate_precision_formats(),
        'mp_speedup': generate_mixed_precision_speedup(),
        'mp_loss_scaling': generate_loss_scaling(),
        
        # Model Compression
        'mc_techniques': generate_compression_techniques(),
        'mc_pruning': generate_pruning_visualization(),
        'mc_quantization': generate_quantization_visualization(),
        'mc_distillation': generate_knowledge_distillation(),
    }
    
    print(f"Generated {len(illustrations)} illustrations successfully!")
    return illustrations

if __name__ == '__main__':
    illustrations = generate_all_illustrations()
    print("\nIllustrations ready for embedding in HTML files.")
    print("Keys:", list(illustrations.keys()))
