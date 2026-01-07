#!/usr/bin/env python3
"""
Generate matplotlib illustrations for optimization and learning cheatsheets.
This script creates high-quality visualizations and encodes them as base64 for inline HTML embedding.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch
from matplotlib.collections import LineCollection
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
# GRADIENT DESCENT ILLUSTRATIONS
# ============================================================================

def generate_gradient_descent_convergence():
    """Generate gradient descent convergence visualization."""
    # Simple quadratic function
    def f(x, y):
        return x**2 + 0.5*y**2
    
    # Gradient
    def grad_f(x, y):
        return np.array([2*x, y])
    
    # Create mesh for contour
    x = np.linspace(-5, 5, 100)
    y = np.linspace(-5, 5, 100)
    X, Y = np.meshgrid(x, y)
    Z = f(X, Y)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    learning_rates = [0.1, 0.5, 0.9]
    titles = ['Маленький LR (0.1)', 'Оптимальный LR (0.5)', 'Большой LR (0.9)']
    
    for idx, (lr, title) in enumerate(zip(learning_rates, titles)):
        ax = axes[idx]
        
        # Plot contour
        contour = ax.contour(X, Y, Z, levels=20, alpha=0.3, cmap='viridis')
        
        # Gradient descent trajectory
        pos = np.array([4.5, 4.5])
        trajectory = [pos.copy()]
        
        for _ in range(30):
            grad = grad_f(pos[0], pos[1])
            pos = pos - lr * grad
            trajectory.append(pos.copy())
            
            # Stop if converged or diverged
            if np.linalg.norm(grad) < 0.01 or np.linalg.norm(pos) > 10:
                break
        
        trajectory = np.array(trajectory)
        ax.plot(trajectory[:, 0], trajectory[:, 1], 'r.-', linewidth=2, markersize=8, alpha=0.7)
        ax.plot(trajectory[0, 0], trajectory[0, 1], 'go', markersize=12, label='Старт')
        ax.plot(0, 0, 'r*', markersize=15, label='Минимум')
        
        ax.set_xlim(-5, 5)
        ax.set_ylim(-5, 5)
        ax.set_xlabel('θ₁')
        ax.set_ylabel('θ₂')
        ax.set_title(title)
        ax.legend()
        ax.set_aspect('equal')
    
    plt.tight_layout()
    return fig_to_base64(fig)

def generate_gradient_descent_types():
    """Generate comparison of batch, stochastic, and mini-batch GD."""
    np.random.seed(42)
    
    # Simulated loss over iterations
    iterations = 100
    
    # Batch GD - smooth convergence
    batch_loss = 10 * np.exp(-np.linspace(0, 4, iterations)) + 0.5
    
    # SGD - noisy convergence
    sgd_loss = 10 * np.exp(-np.linspace(0, 4, iterations)) + np.random.normal(0, 0.3, iterations) + 0.5
    sgd_loss = np.maximum(sgd_loss, 0.3)  # Keep positive
    
    # Mini-batch - moderate noise
    minibatch_loss = 10 * np.exp(-np.linspace(0, 4, iterations)) + np.random.normal(0, 0.1, iterations) + 0.5
    minibatch_loss = np.maximum(minibatch_loss, 0.3)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(batch_loss, label='Batch GD', linewidth=2, color='blue')
    ax.plot(sgd_loss, label='Stochastic GD', linewidth=1.5, alpha=0.7, color='red')
    ax.plot(minibatch_loss, label='Mini-Batch GD', linewidth=2, alpha=0.8, color='green')
    
    ax.set_xlabel('Итерации')
    ax.set_ylabel('Функция потерь')
    ax.set_title('Сравнение типов градиентного спуска')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    return fig_to_base64(fig)

# ============================================================================
# BACKPROPAGATION ILLUSTRATIONS
# ============================================================================

def generate_backpropagation_graph():
    """Generate computational graph for backpropagation."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis('off')
    
    # Network layers
    layer_positions = [1, 3.5, 6, 8.5]
    nodes_per_layer = [2, 3, 3, 1]
    
    # Draw nodes
    node_positions = []
    for layer_idx, (x_pos, n_nodes) in enumerate(zip(layer_positions, nodes_per_layer)):
        y_positions = np.linspace(2, 6, n_nodes)
        layer_nodes = []
        for y_pos in y_positions:
            circle = Circle((x_pos, y_pos), 0.3, facecolor='lightblue', 
                          edgecolor='blue', linewidth=2, zorder=3)
            ax.add_patch(circle)
            layer_nodes.append((x_pos, y_pos))
        node_positions.append(layer_nodes)
    
    # Draw forward pass connections
    for layer_idx in range(len(node_positions) - 1):
        for node1 in node_positions[layer_idx]:
            for node2 in node_positions[layer_idx + 1]:
                ax.plot([node1[0], node2[0]], [node1[1], node2[1]], 
                       'b-', alpha=0.3, linewidth=1, zorder=1)
    
    # Draw backward pass arrows (gradient flow)
    for layer_idx in range(len(node_positions) - 1, 0, -1):
        node1 = node_positions[layer_idx][0]
        node2 = node_positions[layer_idx - 1][1]
        arrow = FancyArrowPatch((node1[0] - 0.3, node1[1]), 
                               (node2[0] + 0.3, node2[1]),
                               arrowstyle='->', mutation_scale=20, linewidth=2,
                               color='red', alpha=0.7, zorder=2)
        ax.add_patch(arrow)
    
    # Labels
    ax.text(1, 0.5, 'Вход\n(x)', ha='center', fontsize=11, fontweight='bold')
    ax.text(3.5, 0.5, 'Скрытый\nслой 1', ha='center', fontsize=11, fontweight='bold')
    ax.text(6, 0.5, 'Скрытый\nслой 2', ha='center', fontsize=11, fontweight='bold')
    ax.text(8.5, 0.5, 'Выход\n(ŷ)', ha='center', fontsize=11, fontweight='bold')
    
    # Forward and backward labels
    ax.text(5, 7.5, 'Forward Pass →', fontsize=12, color='blue', fontweight='bold')
    ax.text(5, 7, '← Backpropagation (градиенты)', fontsize=12, color='red', fontweight='bold')
    
    ax.set_title('Вычислительный граф: Forward Pass и Backpropagation', 
                fontsize=13, fontweight='bold', pad=20)
    
    return fig_to_base64(fig)

def generate_gradient_flow():
    """Generate gradient flow visualization showing vanishing/exploding gradients."""
    layers = np.arange(1, 11)
    
    # Normal gradient flow
    normal_gradients = np.ones(10) * 0.5
    
    # Vanishing gradients
    vanishing_gradients = 0.5 * (0.5 ** np.arange(10))
    
    # Exploding gradients
    exploding_gradients = 0.5 * (1.5 ** np.arange(10))
    exploding_gradients = np.minimum(exploding_gradients, 10)  # Cap for visualization
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(layers, normal_gradients, 'go-', linewidth=2, markersize=8, 
           label='Нормальный поток градиента', alpha=0.7)
    ax.plot(layers, vanishing_gradients, 'ro-', linewidth=2, markersize=8,
           label='Затухающий градиент', alpha=0.7)
    ax.plot(layers, exploding_gradients, 'bo-', linewidth=2, markersize=8,
           label='Взрывающийся градиент', alpha=0.7)
    
    ax.set_xlabel('Номер слоя (от выхода к входу)')
    ax.set_ylabel('Величина градиента')
    ax.set_title('Поток градиента через слои сети')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    return fig_to_base64(fig)

# ============================================================================
# OPTIMIZERS ILLUSTRATIONS
# ============================================================================

def generate_optimizers_comparison():
    """Generate comparison of different optimizers."""
    def f(x, y):
        return x**2 + 4*y**2
    
    def grad_f(x, y):
        return np.array([2*x, 8*y])
    
    # Create mesh
    x = np.linspace(-5, 5, 100)
    y = np.linspace(-5, 5, 100)
    X, Y = np.meshgrid(x, y)
    Z = f(X, Y)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    start_pos = np.array([4.5, 4.5])
    
    # SGD
    ax = axes[0, 0]
    ax.contour(X, Y, Z, levels=20, alpha=0.3)
    pos = start_pos.copy()
    trajectory = [pos.copy()]
    lr = 0.1
    for _ in range(50):
        grad = grad_f(pos[0], pos[1])
        pos = pos - lr * grad
        trajectory.append(pos.copy())
    trajectory = np.array(trajectory)
    ax.plot(trajectory[:, 0], trajectory[:, 1], 'r.-', linewidth=2, alpha=0.7)
    ax.plot(start_pos[0], start_pos[1], 'go', markersize=10)
    ax.plot(0, 0, 'r*', markersize=15)
    ax.set_title('SGD')
    ax.set_xlabel('θ₁')
    ax.set_ylabel('θ₂')
    ax.set_aspect('equal')
    
    # Momentum
    ax = axes[0, 1]
    ax.contour(X, Y, Z, levels=20, alpha=0.3)
    pos = start_pos.copy()
    velocity = np.zeros(2)
    trajectory = [pos.copy()]
    lr = 0.1
    momentum = 0.9
    for _ in range(50):
        grad = grad_f(pos[0], pos[1])
        velocity = momentum * velocity - lr * grad
        pos = pos + velocity
        trajectory.append(pos.copy())
    trajectory = np.array(trajectory)
    ax.plot(trajectory[:, 0], trajectory[:, 1], 'b.-', linewidth=2, alpha=0.7)
    ax.plot(start_pos[0], start_pos[1], 'go', markersize=10)
    ax.plot(0, 0, 'r*', markersize=15)
    ax.set_title('SGD with Momentum')
    ax.set_xlabel('θ₁')
    ax.set_ylabel('θ₂')
    ax.set_aspect('equal')
    
    # RMSprop
    ax = axes[1, 0]
    ax.contour(X, Y, Z, levels=20, alpha=0.3)
    pos = start_pos.copy()
    eg2 = np.zeros(2)
    trajectory = [pos.copy()]
    lr = 0.5
    beta = 0.9
    eps = 1e-8
    for _ in range(50):
        grad = grad_f(pos[0], pos[1])
        eg2 = beta * eg2 + (1 - beta) * grad**2
        pos = pos - lr * grad / (np.sqrt(eg2) + eps)
        trajectory.append(pos.copy())
    trajectory = np.array(trajectory)
    ax.plot(trajectory[:, 0], trajectory[:, 1], 'g.-', linewidth=2, alpha=0.7)
    ax.plot(start_pos[0], start_pos[1], 'go', markersize=10)
    ax.plot(0, 0, 'r*', markersize=15)
    ax.set_title('RMSprop')
    ax.set_xlabel('θ₁')
    ax.set_ylabel('θ₂')
    ax.set_aspect('equal')
    
    # Adam
    ax = axes[1, 1]
    ax.contour(X, Y, Z, levels=20, alpha=0.3)
    pos = start_pos.copy()
    m = np.zeros(2)
    v = np.zeros(2)
    trajectory = [pos.copy()]
    lr = 0.5
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8
    for t in range(1, 51):
        grad = grad_f(pos[0], pos[1])
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * grad**2
        m_hat = m / (1 - beta1**t)
        v_hat = v / (1 - beta2**t)
        pos = pos - lr * m_hat / (np.sqrt(v_hat) + eps)
        trajectory.append(pos.copy())
    trajectory = np.array(trajectory)
    ax.plot(trajectory[:, 0], trajectory[:, 1], 'm.-', linewidth=2, alpha=0.7)
    ax.plot(start_pos[0], start_pos[1], 'go', markersize=10)
    ax.plot(0, 0, 'r*', markersize=15)
    ax.set_title('Adam')
    ax.set_xlabel('θ₁')
    ax.set_ylabel('θ₂')
    ax.set_aspect('equal')
    
    plt.tight_layout()
    return fig_to_base64(fig)

def generate_momentum_visualization():
    """Generate momentum effect visualization."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Simulated convergence
    iterations = 100
    
    # Without momentum
    loss_no_momentum = 10 * np.exp(-np.linspace(0, 3, iterations)) + \
                       0.5 * np.sin(np.linspace(0, 20, iterations)) + 1
    
    # With momentum
    loss_momentum = 10 * np.exp(-np.linspace(0, 4, iterations)) + \
                    0.1 * np.sin(np.linspace(0, 20, iterations)) + 0.5
    
    ax1.plot(loss_no_momentum, label='Без momentum', linewidth=2, color='red')
    ax1.plot(loss_momentum, label='С momentum', linewidth=2, color='blue')
    ax1.set_xlabel('Итерации')
    ax1.set_ylabel('Функция потерь')
    ax1.set_title('Эффект Momentum на сходимость')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Velocity visualization
    velocity_vals = np.zeros(iterations)
    gradient = -0.1 * np.ones(iterations) + 0.05 * np.sin(np.linspace(0, 10, iterations))
    momentum = 0.9
    
    for i in range(1, iterations):
        velocity_vals[i] = momentum * velocity_vals[i-1] + gradient[i]
    
    ax2.plot(gradient, label='Градиент', linewidth=2, alpha=0.7, color='orange')
    ax2.plot(velocity_vals, label='Velocity (накопленный импульс)', linewidth=2, color='blue')
    ax2.set_xlabel('Итерации')
    ax2.set_ylabel('Величина')
    ax2.set_title('Накопление импульса (momentum)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='black', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    return fig_to_base64(fig)

# ============================================================================
# LEARNING RATE SCHEDULING ILLUSTRATIONS
# ============================================================================

def generate_lr_schedules():
    """Generate different learning rate scheduling strategies."""
    epochs = 100
    initial_lr = 0.1
    
    # Constant
    constant_lr = np.ones(epochs) * initial_lr
    
    # Step decay
    step_lr = initial_lr * (0.5 ** (np.arange(epochs) // 20))
    
    # Exponential decay
    exp_lr = initial_lr * np.exp(-0.05 * np.arange(epochs))
    
    # Cosine annealing
    cosine_lr = initial_lr * 0.5 * (1 + np.cos(np.pi * np.arange(epochs) / epochs))
    
    # Polynomial decay
    poly_lr = initial_lr * (1 - np.arange(epochs) / epochs) ** 2
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(constant_lr, label='Constant', linewidth=2)
    ax.plot(step_lr, label='Step Decay', linewidth=2)
    ax.plot(exp_lr, label='Exponential Decay', linewidth=2)
    ax.plot(cosine_lr, label='Cosine Annealing', linewidth=2)
    ax.plot(poly_lr, label='Polynomial Decay', linewidth=2)
    
    ax.set_xlabel('Эпохи')
    ax.set_ylabel('Learning Rate')
    ax.set_title('Стратегии изменения Learning Rate')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return fig_to_base64(fig)

def generate_warmup_schedule():
    """Generate warmup with decay schedule."""
    epochs = 100
    warmup_epochs = 10
    max_lr = 0.1
    
    # Linear warmup
    warmup = np.linspace(0, max_lr, warmup_epochs)
    
    # Cosine decay after warmup
    decay_epochs = epochs - warmup_epochs
    decay = max_lr * 0.5 * (1 + np.cos(np.pi * np.arange(decay_epochs) / decay_epochs))
    
    schedule = np.concatenate([warmup, decay])
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(schedule, linewidth=3, color='blue')
    ax.axvline(x=warmup_epochs, color='red', linestyle='--', 
              linewidth=2, label=f'Конец warmup (эпоха {warmup_epochs})')
    ax.fill_between(range(warmup_epochs), 0, schedule[:warmup_epochs], 
                    alpha=0.3, color='green', label='Warmup фаза')
    ax.fill_between(range(warmup_epochs, epochs), 0, schedule[warmup_epochs:], 
                    alpha=0.3, color='orange', label='Decay фаза')
    
    ax.set_xlabel('Эпохи')
    ax.set_ylabel('Learning Rate')
    ax.set_title('Warmup + Cosine Decay Schedule')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return fig_to_base64(fig)

# ============================================================================
# BATCH NORMALIZATION ILLUSTRATIONS
# ============================================================================

def generate_batchnorm_distribution():
    """Generate batch normalization effect on distributions."""
    np.random.seed(42)
    
    # Simulate activations before and after batch norm
    before_bn = np.random.normal(5, 3, 1000)
    after_bn = (before_bn - before_bn.mean()) / before_bn.std()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Before BN
    ax1.hist(before_bn, bins=30, alpha=0.7, color='red', edgecolor='black')
    ax1.axvline(before_bn.mean(), color='blue', linestyle='--', linewidth=2, 
               label=f'μ = {before_bn.mean():.2f}')
    ax1.axvline(before_bn.mean() + before_bn.std(), color='green', linestyle='--', 
               linewidth=2, label=f'σ = {before_bn.std():.2f}')
    ax1.axvline(before_bn.mean() - before_bn.std(), color='green', linestyle='--', linewidth=2)
    ax1.set_xlabel('Значение активации')
    ax1.set_ylabel('Частота')
    ax1.set_title('До Batch Normalization')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # After BN
    ax2.hist(after_bn, bins=30, alpha=0.7, color='blue', edgecolor='black')
    ax2.axvline(after_bn.mean(), color='red', linestyle='--', linewidth=2, 
               label=f'μ ≈ {after_bn.mean():.2f}')
    ax2.axvline(after_bn.mean() + after_bn.std(), color='green', linestyle='--', 
               linewidth=2, label=f'σ ≈ {after_bn.std():.2f}')
    ax2.axvline(after_bn.mean() - after_bn.std(), color='green', linestyle='--', linewidth=2)
    ax2.set_xlabel('Значение активации')
    ax2.set_ylabel('Частота')
    ax2.set_title('После Batch Normalization')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig_to_base64(fig)

def generate_internal_covariate_shift():
    """Generate internal covariate shift visualization."""
    np.random.seed(42)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Without batch norm - distributions shift
    epochs = [0, 20, 40, 60, 80]
    colors = plt.cm.Reds(np.linspace(0.3, 0.9, len(epochs)))
    
    for i, (epoch, color) in enumerate(zip(epochs, colors)):
        shift = i * 1.5
        scale = 1 + i * 0.3
        data = np.random.normal(shift, scale, 500)
        ax1.hist(data, bins=30, alpha=0.4, color=color, label=f'Эпоха {epoch}')
    
    ax1.set_xlabel('Значение активации')
    ax1.set_ylabel('Частота')
    ax1.set_title('Без Batch Normalization\n(Internal Covariate Shift)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # With batch norm - distributions stable
    colors = plt.cm.Blues(np.linspace(0.3, 0.9, len(epochs)))
    
    for i, (epoch, color) in enumerate(zip(epochs, colors)):
        data = np.random.normal(0, 1, 500)
        ax2.hist(data, bins=30, alpha=0.4, color=color, label=f'Эпоха {epoch}')
    
    ax2.set_xlabel('Значение активации')
    ax2.set_ylabel('Частота')
    ax2.set_title('С Batch Normalization\n(Стабильные распределения)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig_to_base64(fig)

# ============================================================================
# DROPOUT ILLUSTRATIONS
# ============================================================================

def generate_dropout_visualization():
    """Generate dropout visualization."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Network without dropout
    ax1.set_xlim(0, 4)
    ax1.set_ylim(0, 5)
    ax1.axis('off')
    ax1.set_title('Без Dropout\n(все нейроны активны)', fontsize=12, fontweight='bold')
    
    layer_x = [0.5, 2, 3.5]
    nodes = [3, 4, 2]
    
    for layer_idx, (x, n) in enumerate(zip(layer_x, nodes)):
        y_pos = np.linspace(1, 4, n)
        for y in y_pos:
            circle = Circle((x, y), 0.2, facecolor='lightblue', 
                          edgecolor='blue', linewidth=2)
            ax1.add_patch(circle)
    
    # Draw all connections
    for i in range(len(layer_x) - 1):
        y1 = np.linspace(1, 4, nodes[i])
        y2 = np.linspace(1, 4, nodes[i+1])
        for y_from in y1:
            for y_to in y2:
                ax1.plot([layer_x[i], layer_x[i+1]], [y_from, y_to], 
                        'b-', alpha=0.3, linewidth=1)
    
    # Network with dropout
    ax2.set_xlim(0, 4)
    ax2.set_ylim(0, 5)
    ax2.axis('off')
    ax2.set_title('С Dropout (p=0.5)\n(50% нейронов отключены)', fontsize=12, fontweight='bold')
    
    np.random.seed(42)
    for layer_idx, (x, n) in enumerate(zip(layer_x, nodes)):
        y_pos = np.linspace(1, 4, n)
        for y_idx, y in enumerate(y_pos):
            if layer_idx > 0 and np.random.random() < 0.5:  # 50% dropout
                circle = Circle((x, y), 0.2, facecolor='gray', 
                              edgecolor='gray', linewidth=2, alpha=0.3)
                ax2.add_patch(circle)
                # Add X mark
                ax2.plot([x-0.1, x+0.1], [y-0.1, y+0.1], 'r-', linewidth=2)
                ax2.plot([x-0.1, x+0.1], [y+0.1, y-0.1], 'r-', linewidth=2)
            else:
                circle = Circle((x, y), 0.2, facecolor='lightblue', 
                              edgecolor='blue', linewidth=2)
                ax2.add_patch(circle)
    
    plt.tight_layout()
    return fig_to_base64(fig)

def generate_dropout_effect():
    """Generate dropout effect on training."""
    np.random.seed(42)
    epochs = 100
    
    # Without dropout - overfitting
    train_loss_no_dropout = 2 * np.exp(-np.linspace(0, 5, epochs)) + 0.1
    val_loss_no_dropout = 2 * np.exp(-np.linspace(0, 3, epochs)) + \
                           0.3 * np.linspace(0, 1, epochs) + 0.3
    
    # With dropout - better generalization
    train_loss_dropout = 2 * np.exp(-np.linspace(0, 4.5, epochs)) + 0.15
    val_loss_dropout = 2 * np.exp(-np.linspace(0, 4, epochs)) + 0.2
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(train_loss_no_dropout, label='Train (без dropout)', 
           linewidth=2, color='blue', linestyle='--')
    ax.plot(val_loss_no_dropout, label='Validation (без dropout)', 
           linewidth=2, color='red', linestyle='--')
    ax.plot(train_loss_dropout, label='Train (с dropout)', 
           linewidth=2, color='blue')
    ax.plot(val_loss_dropout, label='Validation (с dropout)', 
           linewidth=2, color='green')
    
    ax.set_xlabel('Эпохи')
    ax.set_ylabel('Loss')
    ax.set_title('Эффект Dropout на обобщающую способность')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return fig_to_base64(fig)

# ============================================================================
# REGULARIZATION ILLUSTRATIONS
# ============================================================================

def generate_regularization_effect():
    """Generate L1 and L2 regularization effects."""
    theta = np.linspace(-3, 3, 300)
    
    # Loss without regularization
    loss = theta**2
    
    # L1 regularization
    lambda_l1 = 0.5
    l1_penalty = lambda_l1 * np.abs(theta)
    loss_l1 = loss + l1_penalty
    
    # L2 regularization
    lambda_l2 = 0.5
    l2_penalty = lambda_l2 * theta**2
    loss_l2 = loss + l2_penalty
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss comparison
    ax1.plot(theta, loss, label='Loss без регуляризации', linewidth=2, color='blue')
    ax1.plot(theta, loss_l1, label='Loss + L1 регуляризация', linewidth=2, color='red')
    ax1.plot(theta, loss_l2, label='Loss + L2 регуляризация', linewidth=2, color='green')
    ax1.set_xlabel('θ (вес)')
    ax1.set_ylabel('Функция потерь')
    ax1.set_title('Эффект регуляризации на функцию потерь')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.axvline(x=0, color='black', linestyle='--', alpha=0.3)
    
    # Weight distribution
    np.random.seed(42)
    weights_no_reg = np.random.normal(0, 2, 1000)
    weights_l1 = np.random.laplace(0, 0.5, 1000)
    weights_l2 = np.random.normal(0, 0.7, 1000)
    
    ax2.hist(weights_no_reg, bins=50, alpha=0.5, label='Без регуляризации', color='blue')
    ax2.hist(weights_l1, bins=50, alpha=0.5, label='L1 (Lasso)', color='red')
    ax2.hist(weights_l2, bins=50, alpha=0.5, label='L2 (Ridge)', color='green')
    ax2.set_xlabel('Значение веса')
    ax2.set_ylabel('Частота')
    ax2.set_title('Распределение весов с разной регуляризацией')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.axvline(x=0, color='black', linestyle='--', linewidth=2)
    
    plt.tight_layout()
    return fig_to_base64(fig)

def generate_l1_l2_comparison():
    """Generate L1 vs L2 contour plots."""
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4))
    
    # Create mesh
    w1 = np.linspace(-2, 2, 100)
    w2 = np.linspace(-2, 2, 100)
    W1, W2 = np.meshgrid(w1, w2)
    
    # Loss contours (elliptical)
    Loss = 0.5 * (W1 - 1)**2 + 2 * (W2 - 0.5)**2
    
    # L1 regularization
    ax1.contour(W1, W2, Loss, levels=15, alpha=0.6, cmap='viridis')
    # L1 constraint (diamond)
    ax1.plot([1, 0, -1, 0, 1], [0, 1, 0, -1, 0], 'r-', linewidth=3, label='L1 constraint')
    ax1.plot(0.5, 0.5, 'r*', markersize=20, label='Оптимум')
    ax1.set_xlabel('w₁')
    ax1.set_ylabel('w₂')
    ax1.set_title('L1 Регуляризация (Lasso)\nСпарсификация весов')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')
    
    # L2 regularization
    ax2.contour(W1, W2, Loss, levels=15, alpha=0.6, cmap='viridis')
    # L2 constraint (circle)
    circle = plt.Circle((0, 0), 1, fill=False, edgecolor='r', linewidth=3, label='L2 constraint')
    ax2.add_patch(circle)
    ax2.plot(0.8, 0.6, 'r*', markersize=20, label='Оптимум')
    ax2.set_xlabel('w₁')
    ax2.set_ylabel('w₂')
    ax2.set_title('L2 Регуляризация (Ridge)\nУменьшение весов')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(-2, 2)
    ax2.set_ylim(-2, 2)
    ax2.set_aspect('equal')
    
    # Elastic Net
    ax3.contour(W1, W2, Loss, levels=15, alpha=0.6, cmap='viridis')
    # Elastic Net constraint (rounded diamond)
    theta = np.linspace(0, 2*np.pi, 100)
    r = 0.7 * (1 + 0.3 * np.abs(np.cos(2*theta)))
    ax3.plot(r * np.cos(theta), r * np.sin(theta), 'r-', linewidth=3, label='Elastic Net constraint')
    ax3.plot(0.6, 0.6, 'r*', markersize=20, label='Оптимум')
    ax3.set_xlabel('w₁')
    ax3.set_ylabel('w₂')
    ax3.set_title('Elastic Net\nКомбинация L1 и L2')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_aspect('equal')
    
    plt.tight_layout()
    return fig_to_base64(fig)

# ============================================================================
# EARLY STOPPING ILLUSTRATIONS
# ============================================================================

def generate_early_stopping():
    """Generate early stopping visualization."""
    np.random.seed(42)
    epochs = 100
    
    # Training loss - keeps decreasing
    train_loss = 2 * np.exp(-np.linspace(0, 5, epochs)) + \
                 0.05 * np.random.randn(epochs) + 0.1
    train_loss = np.maximum(train_loss, 0.05)
    
    # Validation loss - U-shape (overfitting after epoch 40)
    val_loss = 2 * np.exp(-np.linspace(0, 3, epochs)) + \
               0.1 * np.maximum(0, np.linspace(-20, 80, epochs))**2 / 1000 + \
               0.1 * np.random.randn(epochs) + 0.2
    val_loss = np.maximum(val_loss, 0.1)
    
    # Find best epoch
    best_epoch = np.argmin(val_loss)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(train_loss, label='Training Loss', linewidth=2, color='blue')
    ax.plot(val_loss, label='Validation Loss', linewidth=2, color='orange')
    
    # Mark best epoch
    ax.axvline(x=best_epoch, color='green', linestyle='--', linewidth=2,
              label=f'Лучшая эпоха: {best_epoch}')
    ax.plot(best_epoch, val_loss[best_epoch], 'g*', markersize=20, 
           label='Early stopping point')
    
    # Mark overfitting region
    ax.axvspan(best_epoch + 10, epochs, alpha=0.2, color='red', 
              label='Зона переобучения')
    
    ax.set_xlabel('Эпохи')
    ax.set_ylabel('Loss')
    ax.set_title('Early Stopping: остановка при ухудшении validation loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return fig_to_base64(fig)

def generate_patience_illustration():
    """Generate patience parameter illustration."""
    np.random.seed(42)
    epochs = 80
    
    val_loss = 1.5 * np.exp(-np.linspace(0, 2.5, epochs)) + \
               0.1 * np.sin(np.linspace(0, 15, epochs)) + 0.3
    
    # Find local minima
    best_loss = float('inf')
    best_epoch = 0
    patience = 10
    wait = 0
    stop_epoch = None
    
    for epoch in range(epochs):
        if val_loss[epoch] < best_loss:
            best_loss = val_loss[epoch]
            best_epoch = epoch
            wait = 0
        else:
            wait += 1
            if wait >= patience and stop_epoch is None:
                stop_epoch = epoch
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(val_loss, linewidth=2, color='orange', label='Validation Loss')
    ax.axvline(x=best_epoch, color='green', linestyle='--', linewidth=2,
              label=f'Лучшая эпоха: {best_epoch}')
    ax.plot(best_epoch, val_loss[best_epoch], 'g*', markersize=20)
    
    if stop_epoch:
        ax.axvline(x=stop_epoch, color='red', linestyle='--', linewidth=2,
                  label=f'Остановка: эпоха {stop_epoch}')
        ax.axvspan(best_epoch, stop_epoch, alpha=0.2, color='yellow',
                  label=f'Patience = {patience} эпох')
    
    ax.set_xlabel('Эпохи')
    ax.set_ylabel('Validation Loss')
    ax.set_title('Early Stopping с параметром Patience')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return fig_to_base64(fig)

# ============================================================================
# VANISHING GRADIENT ILLUSTRATIONS
# ============================================================================

def generate_vanishing_gradient_problem():
    """Generate vanishing gradient visualization."""
    layers = np.arange(1, 11)
    
    # Sigmoid activation - vanishing gradients
    sigmoid_grads = 0.25 * (0.5 ** np.arange(10))
    
    # ReLU activation - better gradient flow
    relu_grads = 0.8 * np.ones(10) + 0.05 * np.random.randn(10)
    relu_grads = np.maximum(relu_grads, 0.1)
    
    # Exploding gradients (without proper initialization)
    exploding_grads = 0.1 * (1.8 ** np.arange(10))
    exploding_grads = np.minimum(exploding_grads, 50)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Gradient magnitude
    ax1.plot(layers, sigmoid_grads, 'ro-', linewidth=2, markersize=8,
            label='Sigmoid (затухающий)', alpha=0.7)
    ax1.plot(layers, relu_grads, 'go-', linewidth=2, markersize=8,
            label='ReLU (стабильный)', alpha=0.7)
    ax1.plot(layers, exploding_grads, 'bo-', linewidth=2, markersize=8,
            label='Плохая инициализация (взрывающийся)', alpha=0.7)
    ax1.set_xlabel('Номер слоя (от выхода к входу)')
    ax1.set_ylabel('Величина градиента')
    ax1.set_title('Проблема затухающего/взрывающегося градиента')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    # Sigmoid saturation
    x = np.linspace(-6, 6, 200)
    sigmoid = 1 / (1 + np.exp(-x))
    sigmoid_grad = sigmoid * (1 - sigmoid)
    
    ax2.plot(x, sigmoid, label='Sigmoid(x)', linewidth=2, color='blue')
    ax2_twin = ax2.twinx()
    ax2_twin.plot(x, sigmoid_grad, label='Производная', linewidth=2, 
                 color='red', linestyle='--')
    
    # Mark saturation regions
    ax2.axvspan(-6, -2, alpha=0.2, color='red', label='Зона насыщения')
    ax2.axvspan(2, 6, alpha=0.2, color='red')
    
    ax2.set_xlabel('x')
    ax2.set_ylabel('Sigmoid(x)', color='blue')
    ax2_twin.set_ylabel('Производная', color='red')
    ax2.set_title('Sigmoid: насыщение → малые градиенты')
    ax2.legend(loc='upper left')
    ax2_twin.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig_to_base64(fig)

def generate_activation_comparison():
    """Generate activation functions comparison for gradient flow."""
    x = np.linspace(-3, 3, 200)
    
    # Activations
    sigmoid = 1 / (1 + np.exp(-x))
    tanh = np.tanh(x)
    relu = np.maximum(0, x)
    leaky_relu = np.where(x > 0, x, 0.1 * x)
    
    # Derivatives
    sigmoid_grad = sigmoid * (1 - sigmoid)
    tanh_grad = 1 - tanh**2
    relu_grad = np.where(x > 0, 1, 0)
    leaky_relu_grad = np.where(x > 0, 1, 0.1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Activation functions
    ax1.plot(x, sigmoid, label='Sigmoid', linewidth=2)
    ax1.plot(x, tanh, label='Tanh', linewidth=2)
    ax1.plot(x, relu, label='ReLU', linewidth=2)
    ax1.plot(x, leaky_relu, label='Leaky ReLU', linewidth=2)
    ax1.set_xlabel('x')
    ax1.set_ylabel('f(x)')
    ax1.set_title('Функции активации')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax1.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    
    # Derivatives
    ax2.plot(x, sigmoid_grad, label="Sigmoid'", linewidth=2)
    ax2.plot(x, tanh_grad, label="Tanh'", linewidth=2)
    ax2.plot(x, relu_grad, label="ReLU'", linewidth=2)
    ax2.plot(x, leaky_relu_grad, label="Leaky ReLU'", linewidth=2)
    ax2.set_xlabel('x')
    ax2.set_ylabel("f'(x)")
    ax2.set_title('Производные (градиенты)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax2.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    ax2.set_ylim(-0.1, 1.2)
    
    plt.tight_layout()
    return fig_to_base64(fig)

# ============================================================================
# WEIGHT INITIALIZATION ILLUSTRATIONS
# ============================================================================

def generate_weight_initialization():
    """Generate weight initialization comparison."""
    np.random.seed(42)
    n_features = 100
    
    # Different initializations
    zeros = np.zeros(n_features)
    random_normal = np.random.randn(n_features)
    xavier = np.random.randn(n_features) * np.sqrt(1 / n_features)
    he = np.random.randn(n_features) * np.sqrt(2 / n_features)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Zeros
    ax = axes[0, 0]
    ax.hist(zeros, bins=30, edgecolor='black', alpha=0.7, color='gray')
    ax.set_title('Инициализация нулями\n(симметрия - плохо!)')
    ax.set_xlabel('Значение веса')
    ax.set_ylabel('Частота')
    ax.set_xlim(-0.5, 0.5)
    
    # Random Normal
    ax = axes[0, 1]
    ax.hist(random_normal, bins=30, edgecolor='black', alpha=0.7, color='red')
    ax.set_title('Случайная N(0,1)\n(слишком большие значения)')
    ax.set_xlabel('Значение веса')
    ax.set_ylabel('Частота')
    ax.axvline(x=0, color='black', linestyle='--', linewidth=2)
    
    # Xavier/Glorot
    ax = axes[1, 0]
    ax.hist(xavier, bins=30, edgecolor='black', alpha=0.7, color='blue')
    ax.set_title(f'Xavier/Glorot: N(0, √(1/n))\nn={n_features}')
    ax.set_xlabel('Значение веса')
    ax.set_ylabel('Частота')
    ax.axvline(x=0, color='black', linestyle='--', linewidth=2)
    
    # He
    ax = axes[1, 1]
    ax.hist(he, bins=30, edgecolor='black', alpha=0.7, color='green')
    ax.set_title(f'He: N(0, √(2/n))\nn={n_features}')
    ax.set_xlabel('Значение веса')
    ax.set_ylabel('Частота')
    ax.axvline(x=0, color='black', linestyle='--', linewidth=2)
    
    plt.tight_layout()
    return fig_to_base64(fig)

def generate_initialization_impact():
    """Generate impact of initialization on training."""
    np.random.seed(42)
    epochs = 50
    
    # Poor initialization - slow convergence
    poor_init = 5 * np.exp(-np.linspace(0, 1.5, epochs)) + 1
    
    # Xavier - good convergence
    xavier_init = 5 * np.exp(-np.linspace(0, 3, epochs)) + 0.3
    
    # He - best for ReLU
    he_init = 5 * np.exp(-np.linspace(0, 3.5, epochs)) + 0.2
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(poor_init, label='Плохая инициализация', linewidth=2, 
           color='red', linestyle='--')
    ax.plot(xavier_init, label='Xavier (для Sigmoid/Tanh)', linewidth=2, color='blue')
    ax.plot(he_init, label='He (для ReLU)', linewidth=2, color='green')
    
    ax.set_xlabel('Эпохи')
    ax.set_ylabel('Training Loss')
    ax.set_title('Влияние инициализации весов на обучение')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    return fig_to_base64(fig)

# ============================================================================
# LOSS FUNCTIONS ILLUSTRATIONS
# ============================================================================

def generate_loss_functions_regression():
    """Generate regression loss functions."""
    y_true = 0
    y_pred = np.linspace(-3, 3, 200)
    
    # MSE
    mse = (y_pred - y_true)**2
    
    # MAE
    mae = np.abs(y_pred - y_true)
    
    # Huber
    delta = 1.0
    huber = np.where(np.abs(y_pred - y_true) <= delta,
                     0.5 * (y_pred - y_true)**2,
                     delta * (np.abs(y_pred - y_true) - 0.5 * delta))
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss functions
    ax1.plot(y_pred, mse, label='MSE (L2)', linewidth=2, color='blue')
    ax1.plot(y_pred, mae, label='MAE (L1)', linewidth=2, color='red')
    ax1.plot(y_pred, huber, label='Huber', linewidth=2, color='green')
    ax1.set_xlabel('Ошибка предсказания (y_pred - y_true)')
    ax1.set_ylabel('Loss')
    ax1.set_title('Функции потерь для регрессии')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.axvline(x=0, color='black', linestyle='--', linewidth=1)
    
    # Derivatives
    mse_grad = 2 * (y_pred - y_true)
    mae_grad = np.sign(y_pred - y_true)
    
    ax2.plot(y_pred, mse_grad, label='MSE gradient', linewidth=2, color='blue')
    ax2.plot(y_pred, mae_grad, label='MAE gradient', linewidth=2, color='red')
    ax2.set_xlabel('Ошибка предсказания')
    ax2.set_ylabel('Градиент')
    ax2.set_title('Градиенты функций потерь')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.axvline(x=0, color='black', linestyle='--', linewidth=1)
    ax2.axhline(y=0, color='black', linestyle='--', linewidth=1)
    
    plt.tight_layout()
    return fig_to_base64(fig)

def generate_loss_functions_classification():
    """Generate classification loss functions."""
    z = np.linspace(-4, 4, 200)
    
    # Binary cross-entropy (assuming y=1)
    # BCE = -log(sigmoid(z)) for y=1
    sigmoid_z = 1 / (1 + np.exp(-z))
    bce = -np.log(np.clip(sigmoid_z, 1e-10, 1))
    
    # Hinge loss (SVM)
    hinge = np.maximum(0, 1 - z)
    
    # Exponential loss
    exp_loss = np.exp(-z)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(z, bce, label='Binary Cross-Entropy', linewidth=2, color='blue')
    ax.plot(z, hinge, label='Hinge (SVM)', linewidth=2, color='red')
    ax.plot(z, exp_loss, label='Exponential', linewidth=2, color='green')
    
    ax.set_xlabel('z = y_true × y_pred (margin)')
    ax.set_ylabel('Loss')
    ax.set_title('Функции потерь для бинарной классификации')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axvline(x=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax.set_ylim(0, 4)
    
    # Add annotations
    ax.annotate('Правильная классификация', xy=(2, 0.5), 
               xytext=(2.5, 1.5), fontsize=10,
               arrowprops=dict(arrowstyle='->', color='green', lw=1.5))
    ax.annotate('Неправильная классификация', xy=(-2, 2), 
               xytext=(-3, 3), fontsize=10,
               arrowprops=dict(arrowstyle='->', color='red', lw=1.5))
    
    return fig_to_base64(fig)

# ============================================================================
# MAIN GENERATION FUNCTION
# ============================================================================

def generate_all_illustrations():
    """Generate all optimization & learning illustrations and return dictionary."""
    print("Generating optimization & learning illustrations...")
    
    illustrations = {}
    
    # Gradient Descent
    print("  Gradient Descent...")
    illustrations['gd_convergence'] = generate_gradient_descent_convergence()
    illustrations['gd_types'] = generate_gradient_descent_types()
    
    # Backpropagation
    print("  Backpropagation...")
    illustrations['backprop_graph'] = generate_backpropagation_graph()
    illustrations['gradient_flow'] = generate_gradient_flow()
    
    # Optimizers
    print("  Optimizers...")
    illustrations['optimizers_comparison'] = generate_optimizers_comparison()
    illustrations['momentum_effect'] = generate_momentum_visualization()
    
    # Learning Rate Scheduling
    print("  Learning Rate Scheduling...")
    illustrations['lr_schedules'] = generate_lr_schedules()
    illustrations['warmup_schedule'] = generate_warmup_schedule()
    
    # Batch Normalization
    print("  Batch Normalization...")
    illustrations['batchnorm_distribution'] = generate_batchnorm_distribution()
    illustrations['covariate_shift'] = generate_internal_covariate_shift()
    
    # Dropout
    print("  Dropout...")
    illustrations['dropout_visualization'] = generate_dropout_visualization()
    illustrations['dropout_effect'] = generate_dropout_effect()
    
    # Regularization
    print("  Regularization...")
    illustrations['regularization_effect'] = generate_regularization_effect()
    illustrations['l1_l2_comparison'] = generate_l1_l2_comparison()
    
    # Early Stopping
    print("  Early Stopping...")
    illustrations['early_stopping'] = generate_early_stopping()
    illustrations['patience_illustration'] = generate_patience_illustration()
    
    # Vanishing Gradient
    print("  Vanishing Gradient...")
    illustrations['vanishing_gradient'] = generate_vanishing_gradient_problem()
    illustrations['activation_comparison'] = generate_activation_comparison()
    
    # Weight Initialization
    print("  Weight Initialization...")
    illustrations['weight_init'] = generate_weight_initialization()
    illustrations['init_impact'] = generate_initialization_impact()
    
    # Loss Functions
    print("  Loss Functions...")
    illustrations['loss_regression'] = generate_loss_functions_regression()
    illustrations['loss_classification'] = generate_loss_functions_classification()
    
    print(f"Generated {len(illustrations)} illustrations!")
    return illustrations

if __name__ == '__main__':
    illustrations = generate_all_illustrations()
    
    # Print list of generated illustrations
    print("\nGenerated illustrations:")
    for key in illustrations.keys():
        print(f"  - {key}")
