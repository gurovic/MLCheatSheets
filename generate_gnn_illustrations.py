#!/usr/bin/env python3
"""
Generate matplotlib illustrations for Graph Neural Networks cheatsheets.
This script creates high-quality visualizations and encodes them as base64 for inline HTML embedding.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle
from matplotlib.collections import LineCollection
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
# GNN BASICS ILLUSTRATIONS
# ============================================================================

def generate_graph_structure():
    """Visualize basic graph structure with nodes and edges."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    
    # Example 1: Social Network
    ax1 = axes[0]
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 10)
    ax1.axis('off')
    ax1.set_title('Социальная сеть', fontsize=12, fontweight='bold')
    
    # Node positions
    nodes_social = {
        'A': (2, 7), 'B': (5, 8), 'C': (8, 7),
        'D': (2, 4), 'E': (5, 5), 'F': (8, 4),
        'G': (5, 2)
    }
    
    # Edges
    edges_social = [('A', 'B'), ('B', 'C'), ('A', 'D'), ('B', 'E'), 
                   ('C', 'F'), ('D', 'E'), ('E', 'F'), ('E', 'G')]
    
    # Draw edges
    for n1, n2 in edges_social:
        x1, y1 = nodes_social[n1]
        x2, y2 = nodes_social[n2]
        ax1.plot([x1, x2], [y1, y2], 'gray', linewidth=2, alpha=0.6, zorder=1)
    
    # Draw nodes
    for name, (x, y) in nodes_social.items():
        circle = Circle((x, y), 0.4, color='lightblue', ec='darkblue', linewidth=2, zorder=2)
        ax1.add_patch(circle)
        ax1.text(x, y, name, ha='center', va='center', fontweight='bold', fontsize=10)
    
    # Example 2: Molecular Structure
    ax2 = axes[1]
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 10)
    ax2.axis('off')
    ax2.set_title('Молекула (граф)', fontsize=12, fontweight='bold')
    
    # Node positions
    nodes_mol = {
        'C': (3, 6), 'O': (6, 7), 'C2': (7, 4),
        'N': (5, 3), 'C3': (3, 3)
    }
    
    # Edges with types
    edges_mol = [
        ('C', 'O', 'double'), ('O', 'C2', 'single'),
        ('C2', 'N', 'single'), ('N', 'C3', 'single'),
        ('C3', 'C', 'single')
    ]
    
    # Draw edges
    for n1, n2, bond_type in edges_mol:
        x1, y1 = nodes_mol[n1]
        x2, y2 = nodes_mol[n2]
        if bond_type == 'double':
            ax2.plot([x1, x2], [y1, y2], 'gray', linewidth=4, alpha=0.6, zorder=1)
        else:
            ax2.plot([x1, x2], [y1, y2], 'gray', linewidth=2, alpha=0.6, zorder=1)
    
    # Draw nodes with different colors
    node_colors = {'C': 'lightgray', 'O': 'lightcoral', 'N': 'lightblue'}
    for name, (x, y) in nodes_mol.items():
        atom = name[0]  # Get first character
        color = node_colors.get(atom, 'lightgray')
        circle = Circle((x, y), 0.5, color=color, ec='black', linewidth=2, zorder=2)
        ax2.add_patch(circle)
        ax2.text(x, y, atom, ha='center', va='center', fontweight='bold', fontsize=11)
    
    # Example 3: Citation Network
    ax3 = axes[2]
    ax3.set_xlim(0, 10)
    ax3.set_ylim(0, 10)
    ax3.axis('off')
    ax3.set_title('Citation Network', fontsize=12, fontweight='bold')
    
    # Node positions
    nodes_cite = {
        'P1': (2, 8), 'P2': (5, 8), 'P3': (8, 8),
        'P4': (2, 5), 'P5': (5, 5), 'P6': (8, 5),
        'P7': (5, 2)
    }
    
    # Directed edges
    edges_cite = [('P1', 'P4'), ('P2', 'P4'), ('P2', 'P5'), 
                  ('P3', 'P5'), ('P3', 'P6'), ('P4', 'P7'), 
                  ('P5', 'P7'), ('P6', 'P7')]
    
    # Draw edges with arrows
    for n1, n2 in edges_cite:
        x1, y1 = nodes_cite[n1]
        x2, y2 = nodes_cite[n2]
        arrow = FancyArrowPatch((x1, y1), (x2, y2), 
                               arrowstyle='->', mutation_scale=15,
                               linewidth=2, color='gray', alpha=0.6, zorder=1)
        ax3.add_patch(arrow)
    
    # Draw nodes
    for name, (x, y) in nodes_cite.items():
        circle = Circle((x, y), 0.4, color='lightyellow', ec='orange', linewidth=2, zorder=2)
        ax3.add_patch(circle)
        ax3.text(x, y, name, ha='center', va='center', fontweight='bold', fontsize=9)
    
    plt.suptitle('Примеры графовых структур', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    return fig_to_base64(fig)

def generate_message_passing():
    """Visualize message passing mechanism."""
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis('off')
    
    # Central node
    center = (7, 4)
    
    # Neighbor nodes
    neighbors = [
        (3, 6, 'v1'), (3, 2, 'v2'), 
        (11, 6, 'v3'), (11, 2, 'v4')
    ]
    
    # Step 1: Initial state
    ax.text(7, 7.5, 'Message Passing: h_v^(t+1) = UPDATE(h_v^(t), AGGREGATE({h_u^(t) : u ∈ N(v)}))', 
           ha='center', fontsize=11, fontweight='bold',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Draw neighbors
    for x, y, label in neighbors:
        circle = Circle((x, y), 0.5, color='lightblue', ec='darkblue', linewidth=2, zorder=2)
        ax.add_patch(circle)
        ax.text(x, y, label, ha='center', va='center', fontweight='bold', fontsize=10)
        
        # Messages
        if x < center[0]:
            arrow = FancyArrowPatch((x + 0.5, y), (center[0] - 0.6, center[1]),
                                   arrowstyle='->', mutation_scale=20,
                                   linewidth=2.5, color='green', alpha=0.7, zorder=1)
        else:
            arrow = FancyArrowPatch((x - 0.5, y), (center[0] + 0.6, center[1]),
                                   arrowstyle='->', mutation_scale=20,
                                   linewidth=2.5, color='green', alpha=0.7, zorder=1)
        ax.add_patch(arrow)
        
        # Message label
        mid_x = (x + center[0]) / 2
        mid_y = (y + center[1]) / 2
        ax.text(mid_x, mid_y + 0.3, f'm_{label}', fontsize=9, 
               color='green', fontweight='bold',
               bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.6))
    
    # Draw center node
    circle = Circle(center, 0.6, color='lightcoral', ec='darkred', linewidth=3, zorder=2)
    ax.add_patch(circle)
    ax.text(center[0], center[1], 'v', ha='center', va='center', 
           fontweight='bold', fontsize=12)
    
    # Aggregation box
    ax.text(center[0], center[1] - 1.5, 'AGGREGATE\n(sum/mean/max)', 
           ha='center', va='center', fontsize=9,
           bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    
    # Legend
    ax.text(1, 0.5, '1. Соседи отправляют сообщения', fontsize=9)
    ax.text(1, 0.2, '2. Центральный узел агрегирует', fontsize=9)
    
    plt.title('Message Passing механизм', fontsize=14, fontweight='bold', pad=20)
    
    return fig_to_base64(fig)

def generate_aggregation_functions():
    """Compare different aggregation functions."""
    np.random.seed(42)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Sample neighbor features
    neighbor_features = np.random.rand(5, 3) * 10
    
    # SUM aggregation
    ax1 = axes[0, 0]
    sum_result = np.sum(neighbor_features, axis=0)
    
    x = np.arange(neighbor_features.shape[0])
    for i in range(3):
        ax1.bar(x + i*0.25, neighbor_features[:, i], width=0.25, 
               label=f'Feature {i+1}', alpha=0.7)
    
    ax1.axhline(y=sum_result[0], color='red', linestyle='--', linewidth=2, 
               label=f'Sum Feat 1: {sum_result[0]:.1f}')
    ax1.set_title('SUM Aggregation', fontsize=11, fontweight='bold')
    ax1.set_xlabel('Neighbor Index')
    ax1.set_ylabel('Feature Value')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    # MEAN aggregation
    ax2 = axes[0, 1]
    mean_result = np.mean(neighbor_features, axis=0)
    
    for i in range(3):
        ax2.bar(x + i*0.25, neighbor_features[:, i], width=0.25, 
               label=f'Feature {i+1}', alpha=0.7)
    
    ax2.axhline(y=mean_result[0], color='red', linestyle='--', linewidth=2,
               label=f'Mean Feat 1: {mean_result[0]:.1f}')
    ax2.set_title('MEAN Aggregation', fontsize=11, fontweight='bold')
    ax2.set_xlabel('Neighbor Index')
    ax2.set_ylabel('Feature Value')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    
    # MAX aggregation
    ax3 = axes[1, 0]
    max_result = np.max(neighbor_features, axis=0)
    
    for i in range(3):
        bars = ax3.bar(x + i*0.25, neighbor_features[:, i], width=0.25, 
               label=f'Feature {i+1}', alpha=0.7)
        # Highlight max
        max_idx = np.argmax(neighbor_features[:, i])
        bars[max_idx].set_edgecolor('red')
        bars[max_idx].set_linewidth(3)
    
    ax3.axhline(y=max_result[0], color='red', linestyle='--', linewidth=2,
               label=f'Max Feat 1: {max_result[0]:.1f}')
    ax3.set_title('MAX Aggregation', fontsize=11, fontweight='bold')
    ax3.set_xlabel('Neighbor Index')
    ax3.set_ylabel('Feature Value')
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)
    
    # Comparison
    ax4 = axes[1, 1]
    agg_types = ['SUM', 'MEAN', 'MAX']
    results = [sum_result, mean_result, max_result]
    
    x_pos = np.arange(len(agg_types))
    for i in range(3):
        values = [r[i] for r in results]
        ax4.bar(x_pos + i*0.25, values, width=0.25, label=f'Feature {i+1}', alpha=0.7)
    
    ax4.set_title('Сравнение методов агрегации', fontsize=11, fontweight='bold')
    ax4.set_xlabel('Aggregation Type')
    ax4.set_ylabel('Aggregated Value')
    ax4.set_xticks(x_pos + 0.25)
    ax4.set_xticklabels(agg_types)
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('Функции агрегации в GNN', fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()
    
    return fig_to_base64(fig)

# ============================================================================
# GCN ILLUSTRATIONS
# ============================================================================

def generate_gcn_layer_operation():
    """Visualize GCN layer operation with normalization."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    
    # Simple 4-node graph
    nodes = {0: (1, 3), 1: (3, 4), 2: (3, 2), 3: (5, 3)}
    edges = [(0, 1), (0, 2), (1, 2), (1, 3), (2, 3)]
    
    # Adjacency matrix
    ax1 = axes[0]
    A = np.array([
        [0, 1, 1, 0],
        [1, 0, 1, 1],
        [1, 1, 0, 1],
        [0, 1, 1, 0]
    ])
    
    im1 = ax1.imshow(A, cmap='Blues', vmin=0, vmax=1)
    ax1.set_title('Матрица смежности A', fontsize=11, fontweight='bold')
    ax1.set_xlabel('Node j')
    ax1.set_ylabel('Node i')
    
    for i in range(4):
        for j in range(4):
            text = ax1.text(j, i, f'{A[i, j]}',
                          ha="center", va="center", color="black", fontsize=11, fontweight='bold')
    
    ax1.set_xticks(range(4))
    ax1.set_yticks(range(4))
    plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    
    # Normalized adjacency
    ax2 = axes[1]
    # Add self-loops
    A_tilde = A + np.eye(4)
    # Degree matrix
    D_tilde = np.diag(np.sum(A_tilde, axis=1))
    D_tilde_inv_sqrt = np.diag(1.0 / np.sqrt(np.diag(D_tilde)))
    # Normalized adjacency
    A_norm = D_tilde_inv_sqrt @ A_tilde @ D_tilde_inv_sqrt
    
    im2 = ax2.imshow(A_norm, cmap='RdYlGn', vmin=0, vmax=1)
    ax2.set_title('Нормализованная Ã\n(D̃^(-½) Ã D̃^(-½))', fontsize=11, fontweight='bold')
    ax2.set_xlabel('Node j')
    ax2.set_ylabel('Node i')
    
    for i in range(4):
        for j in range(4):
            text = ax2.text(j, i, f'{A_norm[i, j]:.2f}',
                          ha="center", va="center", color="black", fontsize=9)
    
    ax2.set_xticks(range(4))
    ax2.set_yticks(range(4))
    plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    
    # Graph visualization
    ax3 = axes[2]
    ax3.set_xlim(0, 6)
    ax3.set_ylim(0, 6)
    ax3.axis('off')
    ax3.set_title('Граф', fontsize=11, fontweight='bold')
    
    # Draw edges
    for n1, n2 in edges:
        x1, y1 = nodes[n1]
        x2, y2 = nodes[n2]
        ax3.plot([x1, x2], [y1, y2], 'gray', linewidth=2, alpha=0.6, zorder=1)
    
    # Draw nodes
    for node_id, (x, y) in nodes.items():
        degree = np.sum(A[node_id])
        circle = Circle((x, y), 0.3, color='lightblue', ec='darkblue', linewidth=2, zorder=2)
        ax3.add_patch(circle)
        ax3.text(x, y, str(node_id), ha='center', va='center', fontweight='bold', fontsize=11)
        ax3.text(x, y - 0.6, f'd={int(degree)}', ha='center', va='center', fontsize=8, style='italic')
    
    plt.suptitle('GCN: Нормализация по степеням узлов', fontsize=13, fontweight='bold', y=1.00)
    plt.tight_layout()
    
    return fig_to_base64(fig)

def generate_gcn_vs_simple():
    """Compare simple aggregation vs GCN normalization."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Without normalization
    ax1 = axes[0]
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 10)
    ax1.axis('off')
    ax1.set_title('Без нормализации (проблема)', fontsize=12, fontweight='bold')
    
    # Center node with many neighbors
    center1 = (5, 5)
    neighbors1 = [(3, 7), (7, 7), (3, 3), (7, 3), (5, 8), (5, 2), (2, 5), (8, 5)]
    
    # Draw edges
    for nx, ny in neighbors1:
        ax1.plot([center1[0], nx], [center1[1], ny], 'gray', linewidth=2, alpha=0.4, zorder=1)
    
    # Draw neighbors
    for nx, ny in neighbors1:
        circle = Circle((nx, ny), 0.3, color='lightblue', ec='blue', linewidth=1, zorder=2)
        ax1.add_patch(circle)
        ax1.text(nx, ny, '1', ha='center', va='center', fontsize=9)
    
    # Draw center
    circle = Circle(center1, 0.5, color='red', ec='darkred', linewidth=2, zorder=2, alpha=0.7)
    ax1.add_patch(circle)
    ax1.text(center1[0], center1[1], '8', ha='center', va='center', 
            fontweight='bold', fontsize=12, color='white')
    ax1.text(center1[0], center1[1] - 1.2, 'Сумма = 8\n(большая степень)', 
            ha='center', fontsize=9, bbox=dict(boxstyle='round', facecolor='pink', alpha=0.7))
    
    # With GCN normalization
    ax2 = axes[1]
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 10)
    ax2.axis('off')
    ax2.set_title('С GCN нормализацией (решение)', fontsize=12, fontweight='bold')
    
    # Draw edges
    for nx, ny in neighbors1:
        ax2.plot([center1[0], nx], [center1[1], ny], 'gray', linewidth=2, alpha=0.4, zorder=1)
    
    # Draw neighbors
    for nx, ny in neighbors1:
        circle = Circle((nx, ny), 0.3, color='lightblue', ec='blue', linewidth=1, zorder=2)
        ax2.add_patch(circle)
        ax2.text(nx, ny, '0.3', ha='center', va='center', fontsize=8)
    
    # Draw center
    circle = Circle(center1, 0.5, color='green', ec='darkgreen', linewidth=2, zorder=2, alpha=0.7)
    ax2.add_patch(circle)
    ax2.text(center1[0], center1[1], '~1', ha='center', va='center', 
            fontweight='bold', fontsize=12, color='white')
    ax2.text(center1[0], center1[1] - 1.2, 'Нормализовано\n(стабильно)', 
            ha='center', fontsize=9, bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
    
    plt.suptitle('Зачем нужна нормализация в GCN', fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    return fig_to_base64(fig)

# ============================================================================
# GAT ILLUSTRATIONS
# ============================================================================

def generate_attention_mechanism():
    """Visualize attention mechanism in GAT."""
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Central node
    center = (7, 5)
    
    # Neighbor nodes with different attention weights
    neighbors = [
        (3, 7, 'v1', 0.1, 'Низкая релевантность'),
        (3, 3, 'v2', 0.5, 'Средняя релевантность'),
        (11, 7, 'v3', 0.3, 'Средняя релевантность'),
        (11, 3, 'v4', 0.1, 'Низкая релевантность')
    ]
    
    # Title
    ax.text(7, 9.5, 'GAT Attention: α_ij = softmax(LeakyReLU(a^T[Wh_i || Wh_j]))', 
           ha='center', fontsize=11, fontweight='bold',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Draw neighbors and attention
    for x, y, label, attention, desc in neighbors:
        # Edge with varying thickness based on attention
        linewidth = 1 + attention * 8  # Scale between 1 and 5
        alpha = 0.3 + attention * 0.7  # Scale between 0.3 and 1.0
        
        if x < center[0]:
            ax.plot([x, center[0]], [y, center[1]], 
                   color='blue', linewidth=linewidth, alpha=alpha, zorder=1)
        else:
            ax.plot([x, center[0]], [y, center[1]], 
                   color='blue', linewidth=linewidth, alpha=alpha, zorder=1)
        
        # Neighbor node
        circle = Circle((x, y), 0.4, color='lightblue', ec='darkblue', linewidth=2, zorder=2)
        ax.add_patch(circle)
        ax.text(x, y, label, ha='center', va='center', fontweight='bold', fontsize=10)
        
        # Attention weight
        mid_x = (x + center[0]) / 2
        mid_y = (y + center[1]) / 2
        ax.text(mid_x, mid_y, f'α={attention:.1f}', fontsize=10, 
               color='red', fontweight='bold',
               bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))
    
    # Draw center node
    circle = Circle(center, 0.6, color='lightcoral', ec='darkred', linewidth=3, zorder=2)
    ax.add_patch(circle)
    ax.text(center[0], center[1], 'v_i', ha='center', va='center', 
           fontweight='bold', fontsize=12)
    
    # Output
    ax.text(center[0], center[1] - 1.5, 
           "h'_i = σ(Σ α_ij · Wh_j)", 
           ha='center', va='center', fontsize=10,
           bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
    
    # Legend
    ax.text(1, 1.5, 'Толщина линии = вес attention', fontsize=9,
           bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))
    ax.text(1, 0.8, 'Больший вес → больше влияния', fontsize=9,
           bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))
    
    plt.title('GAT Attention Mechanism', fontsize=14, fontweight='bold', pad=20)
    
    return fig_to_base64(fig)

def generate_multihead_attention():
    """Visualize multi-head attention in GAT."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    
    np.random.seed(42)
    
    # Same graph structure, different attention patterns
    num_nodes = 5
    
    for idx, ax in enumerate(axes):
        ax.set_xlim(0, 6)
        ax.set_ylim(0, 6)
        ax.axis('off')
        ax.set_title(f'Attention Head {idx+1}', fontsize=11, fontweight='bold')
        
        # Node positions
        center = np.array([3, 3])
        angles = np.linspace(0, 2*np.pi, num_nodes, endpoint=False)
        radius = 2
        
        positions = []
        for angle in angles:
            x = center[0] + radius * np.cos(angle)
            y = center[1] + radius * np.sin(angle)
            positions.append((x, y))
        
        # Generate different attention patterns for each head
        np.random.seed(42 + idx * 10)
        attention_weights = np.random.dirichlet(np.ones(num_nodes))
        
        # Draw edges with attention weights
        for i, (x, y) in enumerate(positions):
            linewidth = 1 + attention_weights[i] * 6
            alpha = 0.3 + attention_weights[i] * 0.7
            
            ax.plot([center[0], x], [center[1], y], 
                   color='blue', linewidth=linewidth, alpha=alpha, zorder=1)
            
            # Attention label
            mid_x = (center[0] + x) / 2
            mid_y = (center[1] + y) / 2
            ax.text(mid_x, mid_y, f'{attention_weights[i]:.2f}', 
                   fontsize=7, color='red', fontweight='bold',
                   bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.6, pad=0.2))
        
        # Draw neighbor nodes
        for i, (x, y) in enumerate(positions):
            circle = Circle((x, y), 0.25, color='lightblue', ec='darkblue', linewidth=1.5, zorder=2)
            ax.add_patch(circle)
            ax.text(x, y, str(i+1), ha='center', va='center', fontsize=9)
        
        # Draw center node
        circle = Circle((center[0], center[1]), 0.35, color='lightcoral', 
                       ec='darkred', linewidth=2, zorder=2)
        ax.add_patch(circle)
        ax.text(center[0], center[1], 'v', ha='center', va='center', 
               fontweight='bold', fontsize=11)
    
    plt.suptitle('Multi-head Attention: разные головы учат разные паттерны', 
                fontsize=13, fontweight='bold', y=1.00)
    plt.tight_layout()
    
    return fig_to_base64(fig)

# ============================================================================
# MESSAGE PASSING NETWORKS ILLUSTRATIONS
# ============================================================================

def generate_mpnn_framework():
    """Visualize MPNN framework components."""
    fig = plt.figure(figsize=(14, 8))
    
    # Create grid for subplots
    gs = fig.add_gridspec(3, 3, hspace=0.4, wspace=0.3)
    
    # Title
    fig.suptitle('Message Passing Neural Network Framework', fontsize=14, fontweight='bold')
    
    # 1. Message function
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.axis('off')
    ax1.text(0.5, 0.7, 'MESSAGE\nFUNCTION', ha='center', va='center', 
            fontsize=11, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    ax1.text(0.5, 0.3, 'M_t(h_v, h_w, e_vw)', ha='center', va='center', fontsize=9)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    
    # 2. Aggregation
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.axis('off')
    ax2.text(0.5, 0.7, 'AGGREGATE', ha='center', va='center', 
            fontsize=11, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))
    ax2.text(0.5, 0.3, 'Σ or mean or max', ha='center', va='center', fontsize=9)
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    
    # 3. Update function
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.axis('off')
    ax3.text(0.5, 0.7, 'UPDATE\nFUNCTION', ha='center', va='center', 
            fontsize=11, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    ax3.text(0.5, 0.3, 'U_t(h_v, m_v)', ha='center', va='center', fontsize=9)
    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1)
    
    # 4. Full process visualization
    ax4 = fig.add_subplot(gs[1:, :])
    ax4.set_xlim(0, 14)
    ax4.set_ylim(0, 6)
    ax4.axis('off')
    
    # Time steps
    time_steps = 3
    x_positions = [2, 6, 10]
    
    for t, x_pos in enumerate(x_positions):
        # Draw graph at each time step
        center = (x_pos, 3)
        neighbors = [(x_pos - 1, 4.5), (x_pos + 1, 4.5), 
                    (x_pos - 1, 1.5), (x_pos + 1, 1.5)]
        
        # Title for time step
        ax4.text(x_pos, 5.5, f't = {t}', ha='center', fontsize=10, fontweight='bold')
        
        # Draw edges
        for nx, ny in neighbors:
            ax4.plot([center[0], nx], [center[1], ny], 'gray', linewidth=1.5, alpha=0.5, zorder=1)
        
        # Draw neighbor nodes
        for nx, ny in neighbors:
            circle = Circle((nx, ny), 0.2, color='lightblue', ec='blue', linewidth=1, zorder=2)
            ax4.add_patch(circle)
        
        # Draw center node with color changing over time
        colors = ['lightyellow', 'lightgreen', 'lightcoral']
        circle = Circle(center, 0.35, color=colors[t], ec='darkred', linewidth=2, zorder=2)
        ax4.add_patch(circle)
        ax4.text(center[0], center[1], f'h^({t})', ha='center', va='center', fontsize=9)
        
        # Arrow to next step
        if t < time_steps - 1:
            arrow = FancyArrowPatch((x_pos + 1.5, 3), (x_positions[t+1] - 1.5, 3),
                                   arrowstyle='->', mutation_scale=20,
                                   linewidth=2, color='black', zorder=1)
            ax4.add_patch(arrow)
            ax4.text((x_pos + x_positions[t+1]) / 2, 3.5, 'Update', 
                    ha='center', fontsize=9, style='italic')
    
    plt.tight_layout()
    
    return fig_to_base64(fig)

# ============================================================================
# GRAPH EMBEDDINGS ILLUSTRATIONS
# ============================================================================

def generate_node_embeddings():
    """Visualize node embeddings projection."""
    np.random.seed(42)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Original graph
    ax1 = axes[0]
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 10)
    ax1.axis('off')
    ax1.set_title('Исходный граф', fontsize=12, fontweight='bold')
    
    # Node positions
    nodes = {
        0: (2, 7, 'A'), 1: (5, 8, 'A'), 2: (8, 7, 'A'),
        3: (2, 4, 'B'), 4: (5, 5, 'B'), 5: (8, 4, 'B'),
        6: (5, 2, 'C')
    }
    
    # Edges
    edges = [(0, 1), (1, 2), (0, 3), (1, 4), (2, 5), (3, 4), (4, 5), (4, 6)]
    
    # Draw edges
    for n1, n2 in edges:
        x1, y1, _ = nodes[n1]
        x2, y2, _ = nodes[n2]
        ax1.plot([x1, x2], [y1, y2], 'gray', linewidth=2, alpha=0.5, zorder=1)
    
    # Draw nodes with colors by class
    colors_map = {'A': 'lightcoral', 'B': 'lightblue', 'C': 'lightgreen'}
    for node_id, (x, y, class_label) in nodes.items():
        circle = Circle((x, y), 0.4, color=colors_map[class_label], 
                       ec='black', linewidth=2, zorder=2)
        ax1.add_patch(circle)
        ax1.text(x, y, str(node_id), ha='center', va='center', 
                fontweight='bold', fontsize=10)
    
    # Legend
    for i, (class_label, color) in enumerate(colors_map.items()):
        ax1.text(1, 9 - i*0.7, f'Class {class_label}', fontsize=9,
                bbox=dict(boxstyle='round', facecolor=color, alpha=0.7))
    
    # Embedding space (2D projection)
    ax2 = axes[1]
    ax2.set_title('Пространство эмбеддингов (2D проекция)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Embedding dim 1')
    ax2.set_ylabel('Embedding dim 2')
    ax2.grid(True, alpha=0.3)
    
    # Generate embeddings (simulate clustering by class)
    embeddings = {
        'A': np.random.randn(3, 2) * 0.5 + np.array([2, 3]),
        'B': np.random.randn(3, 2) * 0.5 + np.array([-2, -1]),
        'C': np.random.randn(1, 2) * 0.3 + np.array([1, -3])
    }
    
    for class_label, color in colors_map.items():
        emb = embeddings[class_label]
        ax2.scatter(emb[:, 0], emb[:, 1], c=color, s=200, 
                   edgecolors='black', linewidths=2, alpha=0.7, 
                   label=f'Class {class_label}', zorder=2)
        
        # Add node IDs
        for i, (x, y) in enumerate(emb):
            node_ids = [nid for nid, (_, _, cl) in nodes.items() if cl == class_label]
            if i < len(node_ids):
                ax2.text(x, y, str(node_ids[i]), ha='center', va='center', 
                        fontweight='bold', fontsize=9)
    
    ax2.legend(loc='upper right')
    
    plt.suptitle('Node Embeddings: граф → векторное пространство', 
                fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()
    
    return fig_to_base64(fig)

def generate_graph_classification():
    """Visualize graph classification pipeline."""
    fig = plt.figure(figsize=(14, 6))
    ax = fig.add_subplot(111)
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis('off')
    
    # Stage 1: Input graphs
    stage1_x = 2
    ax.text(stage1_x, 7, 'Input Graphs', ha='center', fontsize=10, fontweight='bold')
    
    # Draw two small graphs
    for graph_idx, y_offset in enumerate([5, 3]):
        nodes_pos = [(stage1_x - 0.4, y_offset), (stage1_x + 0.4, y_offset), 
                    (stage1_x, y_offset + 0.5)]
        edges = [(0, 1), (1, 2), (2, 0)]
        
        # Edges
        for e1, e2 in edges:
            x1, y1 = nodes_pos[e1]
            x2, y2 = nodes_pos[e2]
            ax.plot([x1, x2], [y1, y2], 'gray', linewidth=1.5, alpha=0.5, zorder=1)
        
        # Nodes
        for x, y in nodes_pos:
            circle = Circle((x, y), 0.15, color='lightblue', ec='blue', linewidth=1, zorder=2)
            ax.add_patch(circle)
    
    # Arrow
    arrow1 = FancyArrowPatch((3, 4), (4.5, 4), arrowstyle='->', mutation_scale=20,
                            linewidth=2, color='black')
    ax.add_patch(arrow1)
    
    # Stage 2: GNN layers
    stage2_x = 6
    ax.text(stage2_x, 7, 'GNN Layers', ha='center', fontsize=10, fontweight='bold')
    
    # Stack of layers
    for i in range(3):
        rect = FancyBboxPatch((stage2_x - 0.8, 2.5 + i*0.6), 1.6, 0.5,
                             boxstyle="round,pad=0.05", 
                             facecolor=['lightblue', 'lightgreen', 'lightyellow'][i],
                             edgecolor='black', linewidth=1.5)
        ax.add_patch(rect)
        ax.text(stage2_x, 2.75 + i*0.6, f'GNN Layer {i+1}', 
               ha='center', va='center', fontsize=8)
    
    # Arrow
    arrow2 = FancyArrowPatch((7.5, 4), (9, 4), arrowstyle='->', mutation_scale=20,
                            linewidth=2, color='black')
    ax.add_patch(arrow2)
    
    # Stage 3: Readout
    stage3_x = 10
    ax.text(stage3_x, 7, 'Graph Pooling', ha='center', fontsize=10, fontweight='bold')
    
    rect = FancyBboxPatch((stage3_x - 0.6, 3), 1.2, 2,
                         boxstyle="round,pad=0.05", 
                         facecolor='orange', edgecolor='black', linewidth=2)
    ax.add_patch(rect)
    ax.text(stage3_x, 4, 'Readout\n(sum/mean)', ha='center', va='center', fontsize=9)
    
    # Arrow
    arrow3 = FancyArrowPatch((11, 4), (11.5, 4), arrowstyle='->', mutation_scale=20,
                            linewidth=2, color='black')
    ax.add_patch(arrow3)
    
    # Stage 4: Classification
    stage4_x = 12.5
    ax.text(stage4_x, 7, 'Prediction', ha='center', fontsize=10, fontweight='bold')
    
    rect = FancyBboxPatch((stage4_x - 0.4, 3), 0.8, 2,
                         boxstyle="round,pad=0.05", 
                         facecolor='lightcoral', edgecolor='black', linewidth=2)
    ax.add_patch(rect)
    ax.text(stage4_x, 4.5, 'Class A', ha='center', va='center', fontsize=8)
    ax.text(stage4_x, 4, 'Class B', ha='center', va='center', fontsize=8)
    ax.text(stage4_x, 3.5, 'Class C', ha='center', va='center', fontsize=8)
    
    # Bottom annotation
    ax.text(7, 1, 'Node features → Node embeddings → Graph embedding → Prediction', 
           ha='center', fontsize=10, style='italic',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
    plt.title('Graph Classification Pipeline', fontsize=14, fontweight='bold', pad=20)
    
    return fig_to_base64(fig)

# ============================================================================
# MAIN FUNCTION
# ============================================================================

def generate_all_illustrations():
    """Generate all GNN illustrations."""
    print("Generating GNN illustrations...")
    
    illustrations = {}
    
    print("  - Generating GNN basics illustrations...")
    illustrations['gnn_graph_structure'] = generate_graph_structure()
    print("    ✓ Graph structure examples")
    
    illustrations['gnn_message_passing'] = generate_message_passing()
    print("    ✓ Message passing")
    
    illustrations['gnn_aggregation'] = generate_aggregation_functions()
    print("    ✓ Aggregation functions")
    
    print("  - Generating GCN illustrations...")
    illustrations['gcn_layer_operation'] = generate_gcn_layer_operation()
    print("    ✓ GCN layer operation")
    
    illustrations['gcn_normalization'] = generate_gcn_vs_simple()
    print("    ✓ GCN normalization")
    
    print("  - Generating GAT illustrations...")
    illustrations['gat_attention'] = generate_attention_mechanism()
    print("    ✓ GAT attention mechanism")
    
    illustrations['gat_multihead'] = generate_multihead_attention()
    print("    ✓ Multi-head attention")
    
    print("  - Generating MPNN illustrations...")
    illustrations['mpnn_framework'] = generate_mpnn_framework()
    print("    ✓ MPNN framework")
    
    print("  - Generating graph embeddings illustrations...")
    illustrations['embeddings_node'] = generate_node_embeddings()
    print("    ✓ Node embeddings")
    
    illustrations['embeddings_graph_classification'] = generate_graph_classification()
    print("    ✓ Graph classification")
    
    print("All illustrations generated successfully!")
    return illustrations

if __name__ == '__main__':
    illustrations = generate_all_illustrations()
    print(f"\nGenerated {len(illustrations)} illustrations")
    print("Illustrations are base64 encoded and ready for HTML embedding")
