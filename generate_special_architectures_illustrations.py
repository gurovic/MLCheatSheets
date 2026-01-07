#!/usr/bin/env python3
"""
Generate matplotlib illustrations for special architectures cheatsheets.
This script creates high-quality visualizations and encodes them as base64 for inline HTML embedding.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch, Rectangle
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
# MEMORY NETWORKS ILLUSTRATIONS
# ============================================================================

def generate_memory_networks_attention():
    """Generate memory networks attention mechanism visualization."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Memory slots
    n_slots = 5
    memory_colors = plt.cm.Blues(np.linspace(0.3, 0.9, n_slots))
    
    # Query
    query_x, query_y = 2, 5
    query = Circle((query_x, query_y), 0.3, color='red', alpha=0.7, zorder=10)
    ax.add_patch(query)
    ax.text(query_x, query_y, 'Q', ha='center', va='center', fontsize=12, fontweight='bold', color='white')
    
    # Memory slots
    memory_positions = []
    for i in range(n_slots):
        mem_x, mem_y = 7, 6.5 - i * 1.5
        memory_positions.append((mem_x, mem_y))
        mem = Rectangle((mem_x - 0.4, mem_y - 0.3), 0.8, 0.6, 
                        color=memory_colors[i], alpha=0.7, zorder=5)
        ax.add_patch(mem)
        ax.text(mem_x, mem_y, f'M{i+1}', ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Attention weights
    attention_weights = np.array([0.05, 0.1, 0.6, 0.2, 0.05])
    
    # Draw attention arrows
    for i, (mem_x, mem_y) in enumerate(memory_positions):
        arrow = FancyArrowPatch((query_x + 0.3, query_y), (mem_x - 0.5, mem_y),
                               connectionstyle="arc3,rad=.2", 
                               arrowstyle='->', mutation_scale=20,
                               linewidth=attention_weights[i] * 10,
                               color='purple', alpha=0.6, zorder=3)
        ax.add_patch(arrow)
        
        # Attention weight labels
        mid_x = (query_x + mem_x) / 2
        mid_y = (query_y + mem_y) / 2 + 0.3
        ax.text(mid_x, mid_y, f'{attention_weights[i]:.2f}', 
               fontsize=9, ha='center', 
               bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.5))
    
    # Output
    output_x, output_y = 12, 3.5
    output = Circle((output_x, output_y), 0.3, color='green', alpha=0.7, zorder=10)
    ax.add_patch(output)
    ax.text(output_x, output_y, 'O', ha='center', va='center', fontsize=12, fontweight='bold', color='white')
    
    # Arrow from memories to output
    arrow_out = FancyArrowPatch((9, 4), (output_x - 0.4, output_y),
                               arrowstyle='->', mutation_scale=20,
                               linewidth=3, color='green', alpha=0.6, zorder=3)
    ax.add_patch(arrow_out)
    
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 7)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('Memory Networks: Attention Mechanism\n(Query reads from memory with learned weights)', 
                fontsize=13, fontweight='bold', pad=20)
    
    return fig_to_base64(fig)

def generate_memory_networks_multihop():
    """Generate multi-hop reasoning visualization."""
    fig, ax = plt.subplots(figsize=(12, 5))
    
    n_hops = 3
    colors = ['#ff9999', '#99ccff', '#99ff99']
    
    for hop in range(n_hops):
        x_base = hop * 4
        
        # Query representation
        query = Circle((x_base + 1, 3), 0.4, color=colors[hop], alpha=0.7, zorder=10)
        ax.add_patch(query)
        ax.text(x_base + 1, 3, f'u{hop}', ha='center', va='center', 
               fontsize=11, fontweight='bold')
        
        # Memory reading
        mem_box = Rectangle((x_base + 2.5, 2.3), 1, 1.4, 
                           color=colors[hop], alpha=0.3, zorder=5)
        ax.add_patch(mem_box)
        ax.text(x_base + 3, 3, 'Memory\nRead', ha='center', va='center', fontsize=9)
        
        # Arrow from query to memory
        arrow1 = FancyArrowPatch((x_base + 1.4, 3), (x_base + 2.5, 3),
                                arrowstyle='->', mutation_scale=15,
                                linewidth=2, color=colors[hop], alpha=0.8)
        ax.add_patch(arrow1)
        
        if hop < n_hops - 1:
            # Arrow to next hop
            arrow2 = FancyArrowPatch((x_base + 3.5, 3), (x_base + 5, 3),
                                    arrowstyle='->', mutation_scale=15,
                                    linewidth=2, color='gray', alpha=0.8)
            ax.add_patch(arrow2)
            ax.text(x_base + 4.25, 3.5, 'u = u + o', ha='center', fontsize=8,
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    # Final answer
    answer = Circle((11.5, 3), 0.4, color='gold', alpha=0.8, zorder=10)
    ax.add_patch(answer)
    ax.text(11.5, 3, 'A', ha='center', va='center', 
           fontsize=12, fontweight='bold')
    
    ax.text(11.5, 1.5, 'Answer', ha='center', fontsize=10, fontweight='bold')
    
    ax.set_xlim(-0.5, 12.5)
    ax.set_ylim(1, 5)
    ax.axis('off')
    ax.set_title('Multi-Hop Reasoning in Memory Networks\n(Query is updated at each hop)', 
                fontsize=13, fontweight='bold', pad=20)
    
    return fig_to_base64(fig)

# ============================================================================
# NEURAL ODES ILLUSTRATIONS
# ============================================================================

def generate_neural_odes_dynamics():
    """Generate Neural ODE dynamics visualization."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Left: ResNet discrete jumps
    t_discrete = np.array([0, 1, 2, 3, 4, 5])
    h_discrete = np.array([0, 0.5, 1.2, 2.0, 2.5, 2.8])
    
    ax1.plot(t_discrete, h_discrete, 'o-', markersize=10, linewidth=2, 
            color='blue', label='Hidden states')
    for i in range(len(t_discrete) - 1):
        ax1.arrow(t_discrete[i], h_discrete[i], 
                 t_discrete[i+1] - t_discrete[i], 
                 h_discrete[i+1] - h_discrete[i],
                 head_width=0.15, head_length=0.1, fc='blue', ec='blue', alpha=0.3)
    
    ax1.set_xlabel('Layer depth', fontsize=11)
    ax1.set_ylabel('Hidden state h', fontsize=11)
    ax1.set_title('ResNet: Discrete Transformations\nh(t+1) = h(t) + f(h(t), θ)', 
                 fontsize=11, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Right: Neural ODE continuous flow
    t_continuous = np.linspace(0, 5, 100)
    h_continuous = 3 * (1 - np.exp(-0.6 * t_continuous))
    
    ax2.plot(t_continuous, h_continuous, linewidth=3, color='red', label='Continuous trajectory')
    
    # Add tangent vectors
    for t in [0.5, 1.5, 2.5, 3.5, 4.5]:
        idx = int(t * 20)
        if idx < len(t_continuous) - 1:
            dh = h_continuous[idx+1] - h_continuous[idx]
            dt = t_continuous[idx+1] - t_continuous[idx]
            ax2.arrow(t_continuous[idx], h_continuous[idx], 
                     dt * 5, dh * 5,
                     head_width=0.1, head_length=0.08, fc='orange', ec='orange', alpha=0.6)
    
    ax2.set_xlabel('Time t', fontsize=11)
    ax2.set_ylabel('Hidden state h', fontsize=11)
    ax2.set_title('Neural ODE: Continuous Dynamics\ndh/dt = f(h(t), t, θ)', 
                 fontsize=11, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    return fig_to_base64(fig)

def generate_neural_odes_trajectories():
    """Generate Neural ODE trajectory visualization in 2D phase space."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Generate spiral trajectories
    n_trajectories = 5
    t = np.linspace(0, 3 * np.pi, 200)
    
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, n_trajectories))
    
    for i, color in enumerate(colors):
        r = 0.5 + i * 0.3
        x = r * np.cos(t) * np.exp(-0.1 * t) + np.random.randn(len(t)) * 0.02
        y = r * np.sin(t) * np.exp(-0.1 * t) + np.random.randn(len(t)) * 0.02
        
        # Plot trajectory
        ax.plot(x, y, linewidth=2, color=color, alpha=0.7, label=f'Class {i+1}')
        
        # Start and end points
        ax.plot(x[0], y[0], 'o', markersize=10, color=color, zorder=10)
        ax.plot(x[-1], y[-1], 's', markersize=10, color=color, zorder=10)
        
        # Add arrows to show direction
        n_arrows = 5
        for j in range(n_arrows):
            idx = (len(t) // n_arrows) * j
            if idx < len(t) - 10:
                dx = x[idx + 10] - x[idx]
                dy = y[idx + 10] - y[idx]
                ax.arrow(x[idx], y[idx], dx, dy, 
                        head_width=0.05, head_length=0.03, 
                        fc=color, ec=color, alpha=0.5, zorder=5)
    
    ax.set_xlabel('h₁(t)', fontsize=12)
    ax.set_ylabel('h₂(t)', fontsize=12)
    ax.set_title('Neural ODE: Phase Space Trajectories\n(Each class follows a learned continuous path)', 
                fontsize=13, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right')
    ax.set_aspect('equal')
    
    return fig_to_base64(fig)

# ============================================================================
# NORMALIZING FLOWS ILLUSTRATIONS
# ============================================================================

def generate_normalizing_flows_transformation():
    """Generate normalizing flows transformation visualization."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    np.random.seed(42)
    
    # 1. Simple Gaussian
    z = np.random.randn(1000, 2)
    
    axes[0].scatter(z[:, 0], z[:, 1], alpha=0.5, s=20, c='blue')
    axes[0].set_xlim(-4, 4)
    axes[0].set_ylim(-4, 4)
    axes[0].set_aspect('equal')
    axes[0].set_title('z ~ N(0, I)\n(Simple distribution)', fontsize=11, fontweight='bold')
    axes[0].set_xlabel('z₁')
    axes[0].set_ylabel('z₂')
    axes[0].grid(True, alpha=0.3)
    
    # 2. After first flow
    # Apply affine transformation
    x1 = z.copy()
    x1[:, 0] = z[:, 0] * 1.5 + z[:, 1] * 0.5
    x1[:, 1] = z[:, 1] * 0.8
    
    axes[1].scatter(x1[:, 0], x1[:, 1], alpha=0.5, s=20, c='green')
    axes[1].set_xlim(-6, 6)
    axes[1].set_ylim(-4, 4)
    axes[1].set_aspect('equal')
    axes[1].set_title('x₁ = f₁(z)\n(After flow 1)', fontsize=11, fontweight='bold')
    axes[1].set_xlabel('x₁₁')
    axes[1].set_ylabel('x₁₂')
    axes[1].grid(True, alpha=0.3)
    axes[1].arrow(-3, 2.5, 1, 0, head_width=0.3, head_length=0.2, fc='black', ec='black')
    
    # 3. After second flow (non-linear)
    x2 = x1.copy()
    x2[:, 0] = x1[:, 0] + 0.3 * np.sin(x1[:, 1] * 2)
    x2[:, 1] = x1[:, 1] + 0.3 * np.sin(x1[:, 0] * 2)
    
    axes[2].scatter(x2[:, 0], x2[:, 1], alpha=0.5, s=20, c='red')
    axes[2].set_xlim(-6, 6)
    axes[2].set_ylim(-4, 4)
    axes[2].set_aspect('equal')
    axes[2].set_title('x₂ = f₂(x₁)\n(Complex distribution)', fontsize=11, fontweight='bold')
    axes[2].set_xlabel('x₂₁')
    axes[2].set_ylabel('x₂₂')
    axes[2].grid(True, alpha=0.3)
    axes[2].arrow(-4, 2.5, 1, 0, head_width=0.3, head_length=0.2, fc='black', ec='black')
    
    plt.suptitle('Normalizing Flows: Sequential Transformations\nx = f_K ∘ ... ∘ f₂ ∘ f₁(z)', 
                fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    return fig_to_base64(fig)

def generate_normalizing_flows_density():
    """Generate density mapping visualization for normalizing flows."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Create grid
    x = np.linspace(-3, 3, 100)
    y = np.linspace(-3, 3, 100)
    X, Y = np.meshgrid(x, y)
    
    # Simple Gaussian density
    Z_simple = np.exp(-(X**2 + Y**2) / 2) / (2 * np.pi)
    
    im1 = ax1.contourf(X, Y, Z_simple, levels=20, cmap='Blues')
    ax1.contour(X, Y, Z_simple, levels=10, colors='black', alpha=0.3, linewidths=0.5)
    ax1.set_title('p(z): Simple Gaussian\nlog p(z) = -½(z₁² + z₂²) - log(2π)', 
                 fontsize=11, fontweight='bold')
    ax1.set_xlabel('z₁')
    ax1.set_ylabel('z₂')
    ax1.set_aspect('equal')
    plt.colorbar(im1, ax=ax1, label='Density')
    
    # Complex distribution (two modes)
    Z_complex = (np.exp(-((X-1.5)**2 + (Y-1.5)**2) / 0.5) + 
                 np.exp(-((X+1.5)**2 + (Y+1.5)**2) / 0.5) +
                 0.3 * np.exp(-((X-1.5)**2 + (Y+1.5)**2) / 0.8)) / 10
    
    im2 = ax2.contourf(X, Y, Z_complex, levels=20, cmap='Reds')
    ax2.contour(X, Y, Z_complex, levels=10, colors='black', alpha=0.3, linewidths=0.5)
    ax2.set_title('p(x): Complex Distribution\nlog p(x) = log p(z) + log|det J|', 
                 fontsize=11, fontweight='bold')
    ax2.set_xlabel('x₁')
    ax2.set_ylabel('x₂')
    ax2.set_aspect('equal')
    plt.colorbar(im2, ax=ax2, label='Density')
    
    plt.suptitle('Normalizing Flows: Density Transformation\n(Change of variables formula)', 
                fontsize=13, fontweight='bold', y=1.00)
    plt.tight_layout()
    
    return fig_to_base64(fig)

# ============================================================================
# HYPERNETWORKS ILLUSTRATIONS
# ============================================================================

def generate_hypernetworks_architecture():
    """Generate hypernetworks architecture visualization."""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Task embedding
    task_box = FancyBboxPatch((1, 6), 1.5, 1, boxstyle="round,pad=0.1", 
                             facecolor='lightblue', edgecolor='blue', linewidth=2)
    ax.add_patch(task_box)
    ax.text(1.75, 6.5, 'Task\nEmbedding', ha='center', va='center', 
           fontsize=10, fontweight='bold')
    
    # Hypernetwork
    hyper_box = FancyBboxPatch((4, 5), 3, 3, boxstyle="round,pad=0.1", 
                              facecolor='lightyellow', edgecolor='orange', linewidth=3)
    ax.add_patch(hyper_box)
    ax.text(5.5, 6.5, 'Hypernetwork\n(Weight Generator)', ha='center', va='center', 
           fontsize=11, fontweight='bold')
    
    # Add some internal structure to hypernetwork
    for i in range(3):
        circle = Circle((4.5 + i * 0.8, 6.5), 0.25, color='orange', alpha=0.5)
        ax.add_patch(circle)
    
    # Arrow from task to hypernetwork
    arrow1 = FancyArrowPatch((2.5, 6.5), (4, 6.5),
                            arrowstyle='->', mutation_scale=20,
                            linewidth=2.5, color='blue', alpha=0.7)
    ax.add_patch(arrow1)
    
    # Generated weights
    weights_positions = [(8.5, 7.5), (8.5, 6.5), (8.5, 5.5)]
    weight_labels = ['W₁', 'W₂', 'W₃']
    
    for (wx, wy), label in zip(weights_positions, weight_labels):
        weight_box = Rectangle((wx - 0.3, wy - 0.25), 0.6, 0.5, 
                              facecolor='lightgreen', edgecolor='green', linewidth=2)
        ax.add_patch(weight_box)
        ax.text(wx, wy, label, ha='center', va='center', fontsize=10, fontweight='bold')
        
        # Arrow from hypernetwork to weights
        arrow = FancyArrowPatch((7, 6.5), (wx - 0.3, wy),
                               connectionstyle="arc3,rad=.2",
                               arrowstyle='->', mutation_scale=15,
                               linewidth=2, color='orange', alpha=0.6)
        ax.add_patch(arrow)
    
    # Main network (target network)
    main_box = FancyBboxPatch((10, 4.5), 2.5, 4, boxstyle="round,pad=0.1", 
                             facecolor='lightcoral', edgecolor='red', linewidth=3)
    ax.add_patch(main_box)
    ax.text(11.25, 6.5, 'Main\nNetwork', ha='center', va='center', 
           fontsize=11, fontweight='bold')
    
    # Arrows from weights to main network
    for wx, wy in weights_positions:
        arrow = FancyArrowPatch((wx + 0.3, wy), (10, 6.5),
                               connectionstyle="arc3,rad=.3",
                               arrowstyle='->', mutation_scale=15,
                               linewidth=2, color='green', alpha=0.6)
        ax.add_patch(arrow)
    
    # Input and output
    input_circle = Circle((11.25, 2.5), 0.4, color='purple', alpha=0.7)
    ax.add_patch(input_circle)
    ax.text(11.25, 2.5, 'x', ha='center', va='center', 
           fontsize=12, fontweight='bold', color='white')
    ax.text(11.25, 1.8, 'Input', ha='center', fontsize=9)
    
    output_circle = Circle((11.25, 9.5), 0.4, color='darkgreen', alpha=0.7)
    ax.add_patch(output_circle)
    ax.text(11.25, 9.5, 'y', ha='center', va='center', 
           fontsize=12, fontweight='bold', color='white')
    ax.text(11.25, 10.2, 'Output', ha='center', fontsize=9)
    
    # Arrows
    arrow_in = FancyArrowPatch((11.25, 3), (11.25, 4.5),
                              arrowstyle='->', mutation_scale=20,
                              linewidth=2, color='purple', alpha=0.7)
    ax.add_patch(arrow_in)
    
    arrow_out = FancyArrowPatch((11.25, 8.5), (11.25, 9.1),
                               arrowstyle='->', mutation_scale=20,
                               linewidth=2, color='darkgreen', alpha=0.7)
    ax.add_patch(arrow_out)
    
    ax.set_xlim(0, 13)
    ax.set_ylim(1, 11)
    ax.axis('off')
    ax.set_title('Hypernetworks: Generating Weights for Main Network\n(One network generates weights for another)', 
                fontsize=13, fontweight='bold', pad=20)
    
    return fig_to_base64(fig)

def generate_hypernetworks_comparison():
    """Generate comparison between standard and hypernetwork training."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Standard network
    ax1.text(5, 9, 'Standard Network', ha='center', fontsize=12, fontweight='bold')
    
    # Multiple tasks with separate networks
    tasks = ['Task 1', 'Task 2', 'Task 3']
    colors = ['#ff9999', '#99ccff', '#99ff99']
    
    for i, (task, color) in enumerate(zip(tasks, colors)):
        y_base = 7 - i * 2.5
        
        # Task box
        task_box = Rectangle((1, y_base - 0.3), 1.5, 0.6, 
                            facecolor=color, edgecolor='black', linewidth=1.5)
        ax1.add_patch(task_box)
        ax1.text(1.75, y_base, task, ha='center', va='center', fontsize=9)
        
        # Network box
        net_box = Rectangle((4, y_base - 0.5), 2, 1, 
                           facecolor=color, edgecolor='black', linewidth=2, alpha=0.5)
        ax1.add_patch(net_box)
        ax1.text(5, y_base, f'Network {i+1}\n(θ_{i+1})', ha='center', va='center', 
                fontsize=9, fontweight='bold')
        
        # Arrow
        arrow = FancyArrowPatch((2.5, y_base), (4, y_base),
                               arrowstyle='->', mutation_scale=15,
                               linewidth=2, color=color)
        ax1.add_patch(arrow)
    
    ax1.text(5, 0.5, '⚠ No weight sharing\n⚠ More parameters', ha='center', 
            fontsize=9, bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.5))
    
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 10)
    ax1.axis('off')
    
    # Hypernetwork
    ax2.text(5, 9, 'Hypernetwork Approach', ha='center', fontsize=12, fontweight='bold')
    
    # Tasks
    for i, (task, color) in enumerate(zip(tasks, colors)):
        y_base = 7 - i * 2.5
        
        # Task embedding
        task_box = Rectangle((1, y_base - 0.3), 1.5, 0.6, 
                            facecolor=color, edgecolor='black', linewidth=1.5)
        ax2.add_patch(task_box)
        ax2.text(1.75, y_base, task, ha='center', va='center', fontsize=9)
        
        # Arrow to hypernetwork
        arrow = FancyArrowPatch((2.5, y_base), (3.5, 5),
                               connectionstyle="arc3,rad=.3",
                               arrowstyle='->', mutation_scale=12,
                               linewidth=1.5, color=color, alpha=0.7)
        ax2.add_patch(arrow)
    
    # Single hypernetwork
    hyper_box = FancyBboxPatch((3.5, 4), 3, 2, boxstyle="round,pad=0.1", 
                              facecolor='gold', edgecolor='orange', linewidth=3)
    ax2.add_patch(hyper_box)
    ax2.text(5, 5, 'Hypernetwork\n(Shared θ_H)', ha='center', va='center', 
            fontsize=10, fontweight='bold')
    
    # Generated weights
    for i, color in enumerate(colors):
        y_base = 7 - i * 2.5
        
        # Arrow to generated weights
        arrow = FancyArrowPatch((6.5, 5), (7.5, y_base),
                               connectionstyle="arc3,rad=.3",
                               arrowstyle='->', mutation_scale=12,
                               linewidth=2, color=color, alpha=0.7)
        ax2.add_patch(arrow)
        
        # Generated weights
        weight_box = Rectangle((7.5, y_base - 0.35), 1.2, 0.7, 
                              facecolor=color, edgecolor='green', linewidth=2, alpha=0.6)
        ax2.add_patch(weight_box)
        ax2.text(8.1, y_base, f'θ_{i+1}', ha='center', va='center', 
                fontsize=9, fontweight='bold')
    
    ax2.text(5, 0.5, '✅ Weight sharing\n✅ Fewer parameters', ha='center', 
            fontsize=9, bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.5))
    
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 10)
    ax2.axis('off')
    
    plt.tight_layout()
    return fig_to_base64(fig)

# ============================================================================
# SPIKING NEURAL NETWORKS ILLUSTRATIONS
# ============================================================================

def generate_snn_spike_trains():
    """Generate spike train visualization for multiple neurons."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    n_neurons = 5
    t_max = 100
    
    np.random.seed(42)
    
    colors = plt.cm.Set2(np.linspace(0, 1, n_neurons))
    
    for neuron_id in range(n_neurons):
        # Generate random spike times
        n_spikes = np.random.randint(8, 15)
        spike_times = np.sort(np.random.uniform(5, t_max - 5, n_spikes))
        
        # Plot baseline
        ax.hlines(neuron_id, 0, t_max, colors='gray', linewidth=1, alpha=0.3)
        
        # Plot spikes
        for spike_time in spike_times:
            ax.plot([spike_time, spike_time], [neuron_id - 0.4, neuron_id + 0.4], 
                   color=colors[neuron_id], linewidth=2.5)
            ax.plot(spike_time, neuron_id + 0.4, 'v', 
                   color=colors[neuron_id], markersize=8)
        
        # Label
        ax.text(-3, neuron_id, f'N{neuron_id + 1}', va='center', ha='right', 
               fontsize=10, fontweight='bold', color=colors[neuron_id])
    
    ax.set_xlim(-5, t_max + 5)
    ax.set_ylim(-1, n_neurons)
    ax.set_xlabel('Time (ms)', fontsize=12)
    ax.set_ylabel('Neuron ID', fontsize=12)
    ax.set_title('Spiking Neural Network: Spike Trains\n(Temporal sparse events)', 
                fontsize=13, fontweight='bold', pad=20)
    ax.set_yticks(range(n_neurons))
    ax.set_yticklabels([f'N{i+1}' for i in range(n_neurons)])
    ax.grid(True, alpha=0.3, axis='x')
    
    return fig_to_base64(fig)

def generate_snn_lif_dynamics():
    """Generate LIF neuron dynamics visualization."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    # Simulation parameters
    dt = 0.1  # ms
    t_max = 100  # ms
    t = np.arange(0, t_max, dt)
    
    # LIF parameters
    tau = 10.0  # membrane time constant
    V_rest = -70.0  # resting potential
    V_thresh = -50.0  # threshold
    V_reset = -75.0  # reset potential
    
    # Input current (with some pulses)
    I = np.zeros_like(t)
    I[100:200] = 2.5
    I[300:350] = 3.0
    I[500:600] = 4.0
    I[700:800] = 2.0
    
    # Simulate LIF neuron
    V = np.zeros_like(t)
    V[0] = V_rest
    spikes = []
    
    for i in range(1, len(t)):
        # LIF dynamics: dV/dt = (-(V - V_rest) + R*I) / tau
        dV = (-(V[i-1] - V_rest) + I[i-1]) / tau
        V[i] = V[i-1] + dV * dt
        
        # Check for spike
        if V[i] >= V_thresh:
            V[i] = V_reset
            spikes.append(i)
    
    # Plot membrane potential
    ax1.plot(t, V, linewidth=2, color='blue', label='Membrane potential V(t)')
    ax1.axhline(V_thresh, color='red', linestyle='--', linewidth=2, 
               label=f'Threshold ({V_thresh} mV)')
    ax1.axhline(V_rest, color='green', linestyle='--', linewidth=1.5, 
               label=f'Rest ({V_rest} mV)')
    
    # Mark spikes
    for spike_idx in spikes:
        ax1.plot([t[spike_idx], t[spike_idx]], [V_thresh, V_thresh + 10], 
                'r-', linewidth=2)
        ax1.plot(t[spike_idx], V_thresh + 10, 'rv', markersize=10)
    
    ax1.set_ylabel('Membrane Potential (mV)', fontsize=11)
    ax1.set_title('Leaky Integrate-and-Fire (LIF) Neuron Dynamics', 
                 fontsize=13, fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(-80, -40)
    
    # Plot input current
    ax2.plot(t, I, linewidth=2, color='orange', label='Input current I(t)')
    ax2.fill_between(t, 0, I, alpha=0.3, color='orange')
    ax2.set_xlabel('Time (ms)', fontsize=11)
    ax2.set_ylabel('Input Current', fontsize=11)
    ax2.set_title('Input Current Pulses', fontsize=11, fontweight='bold')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 5)
    
    plt.tight_layout()
    return fig_to_base64(fig)

def generate_snn_encoding():
    """Generate input encoding schemes visualization."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Input signal
    t = np.linspace(0, 100, 1000)
    signal = 0.5 + 0.3 * np.sin(0.1 * t) + 0.1 * np.sin(0.3 * t)
    
    # 1. Rate Coding
    ax = axes[0, 0]
    # Higher intensity = higher spike rate
    for intensity in [0.3, 0.5, 0.7, 0.9]:
        y_pos = intensity
        n_spikes = int(intensity * 20)
        spike_times = np.sort(np.random.uniform(0, 100, n_spikes))
        ax.eventplot([spike_times], lineoffsets=y_pos, colors='blue', linewidths=2)
        ax.text(-5, y_pos, f'{intensity:.1f}', va='center', ha='right', fontsize=9)
    
    ax.set_xlim(-8, 105)
    ax.set_ylim(0.2, 1.0)
    ax.set_xlabel('Time (ms)', fontsize=10)
    ax.set_ylabel('Input Intensity', fontsize=10)
    ax.set_title('Rate Coding\n(Spike frequency ∝ intensity)', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # 2. Temporal Coding
    ax = axes[0, 1]
    # Time of first spike encodes value
    values = [0.2, 0.4, 0.6, 0.8]
    for i, val in enumerate(values):
        spike_time = 10 + (1 - val) * 80  # Earlier spike = higher value
        ax.plot([spike_time], [i], 'ro', markersize=12)
        ax.plot([spike_time, spike_time], [i - 0.2, i + 0.2], 'r-', linewidth=2)
        ax.text(5, i, f'{val:.1f}', va='center', ha='right', fontsize=9)
    
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.5, 3.5)
    ax.set_xlabel('Time (ms)', fontsize=10)
    ax.set_ylabel('Neuron', fontsize=10)
    ax.set_title('Temporal Coding\n(Spike time encodes value)', fontsize=11, fontweight='bold')
    ax.set_yticks(range(4))
    ax.grid(True, alpha=0.3)
    
    # 3. Population Coding
    ax = axes[1, 0]
    n_neurons = 10
    input_val = 0.65
    
    # Neurons with different preferred values
    preferred_vals = np.linspace(0, 1, n_neurons)
    
    for i, pref_val in enumerate(preferred_vals):
        # Firing rate based on distance from preferred value
        activity = np.exp(-((input_val - pref_val) ** 2) / 0.05)
        n_spikes = int(activity * 15)
        
        if n_spikes > 0:
            spike_times = np.sort(np.random.uniform(0, 100, n_spikes))
            ax.eventplot([spike_times], lineoffsets=i, colors='green', linewidths=2)
    
    ax.axvline(input_val * 100, color='red', linestyle='--', linewidth=2, 
              label=f'Input = {input_val:.2f}')
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.5, n_neurons - 0.5)
    ax.set_xlabel('Time (ms)', fontsize=10)
    ax.set_ylabel('Neuron ID', fontsize=10)
    ax.set_title('Population Coding\n(Distributed representation)', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # 4. Latency Coding
    ax = axes[1, 1]
    intensities = np.linspace(0.2, 1.0, 8)
    
    for i, intensity in enumerate(intensities):
        # Higher intensity = shorter latency
        latency = 10 + (1 - intensity) * 50
        ax.plot([latency], [i], 'mo', markersize=10)
        ax.plot([latency, latency], [i - 0.2, i + 0.2], 'm-', linewidth=2)
        ax.text(5, i, f'{intensity:.1f}', va='center', ha='right', fontsize=9)
    
    ax.set_xlim(0, 70)
    ax.set_ylim(-0.5, 7.5)
    ax.set_xlabel('Latency (ms)', fontsize=10)
    ax.set_ylabel('Stimulus', fontsize=10)
    ax.set_title('Latency Coding\n(First-spike time matters)', fontsize=11, fontweight='bold')
    ax.set_yticks(range(8))
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig_to_base64(fig)

# ============================================================================
# MAIN FUNCTION
# ============================================================================

def generate_all_illustrations():
    """Generate all special architectures illustrations."""
    print("Generating illustrations for special architectures...")
    
    illustrations = {}
    
    print("\n1. Memory Networks illustrations...")
    illustrations['memory_networks_attention'] = generate_memory_networks_attention()
    illustrations['memory_networks_multihop'] = generate_memory_networks_multihop()
    
    print("2. Neural ODEs illustrations...")
    illustrations['neural_odes_dynamics'] = generate_neural_odes_dynamics()
    illustrations['neural_odes_trajectories'] = generate_neural_odes_trajectories()
    
    print("3. Normalizing Flows illustrations...")
    illustrations['normalizing_flows_transformation'] = generate_normalizing_flows_transformation()
    illustrations['normalizing_flows_density'] = generate_normalizing_flows_density()
    
    print("4. Hypernetworks illustrations...")
    illustrations['hypernetworks_architecture'] = generate_hypernetworks_architecture()
    illustrations['hypernetworks_comparison'] = generate_hypernetworks_comparison()
    
    print("5. Spiking Neural Networks illustrations...")
    illustrations['snn_spike_trains'] = generate_snn_spike_trains()
    illustrations['snn_lif_dynamics'] = generate_snn_lif_dynamics()
    illustrations['snn_encoding'] = generate_snn_encoding()
    
    print("\n✓ All illustrations generated successfully!")
    return illustrations

if __name__ == '__main__':
    illustrations = generate_all_illustrations()
    print(f"\nGenerated {len(illustrations)} illustrations")
    for name in illustrations.keys():
        print(f"  - {name}")
