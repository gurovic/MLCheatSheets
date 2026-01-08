#!/usr/bin/env python3
"""
Generate matplotlib illustrations for graphical models cheatsheets.
This script creates high-quality visualizations for Bayesian Networks, HMM, and CRF.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
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
# BAYESIAN NETWORKS ILLUSTRATIONS
# ============================================================================

def generate_bayesian_network_example():
    """Generate a Bayesian network example (Rain -> Sprinkler, Grass_Wet)."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create directed graph
    G = nx.DiGraph()
    G.add_edges_from([
        ('Дождь', 'Спринклер'),
        ('Дождь', 'Мокрая\nтрава'),
        ('Спринклер', 'Мокрая\nтрава')
    ])
    
    # Set positions
    pos = {
        'Дождь': (0.5, 1),
        'Спринклер': (0.2, 0.5),
        'Мокрая\nтрава': (0.8, 0.5)
    }
    
    # Draw network
    nx.draw_networkx_nodes(G, pos, node_color='lightblue', 
                          node_size=3000, ax=ax, alpha=0.9)
    nx.draw_networkx_labels(G, pos, font_size=11, font_weight='bold', ax=ax)
    nx.draw_networkx_edges(G, pos, edge_color='gray', 
                          arrows=True, arrowsize=20, arrowstyle='->', 
                          width=2, ax=ax)
    
    ax.set_title('Байесовская сеть: Классический пример с дождём', 
                fontsize=14, fontweight='bold', pad=20)
    ax.axis('off')
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(0.3, 1.2)
    
    return fig_to_base64(fig)

def generate_markov_network_example():
    """Generate a Markov network example (undirected)."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create undirected graph
    G = nx.Graph()
    G.add_edges_from([
        ('A', 'B'),
        ('B', 'C'),
        ('A', 'C'),
        ('C', 'D')
    ])
    
    # Set positions
    pos = {
        'A': (0, 0.5),
        'B': (0.33, 1),
        'C': (0.67, 0.5),
        'D': (1, 1)
    }
    
    # Draw network
    nx.draw_networkx_nodes(G, pos, node_color='lightcoral', 
                          node_size=2500, ax=ax, alpha=0.9)
    nx.draw_networkx_labels(G, pos, font_size=12, font_weight='bold', ax=ax)
    nx.draw_networkx_edges(G, pos, edge_color='gray', width=3, ax=ax)
    
    ax.set_title('Марковская сеть (ненаправленный граф)', 
                fontsize=14, fontweight='bold', pad=20)
    ax.axis('off')
    ax.set_xlim(-0.2, 1.2)
    ax.set_ylim(0.2, 1.2)
    
    return fig_to_base64(fig)

def generate_d_separation_examples():
    """Generate d-separation examples (Chain, Fork, Collider)."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Chain: A -> B -> C
    G1 = nx.DiGraph()
    G1.add_edges_from([('A', 'B'), ('B', 'C')])
    pos1 = {'A': (0, 0.5), 'B': (0.5, 0.5), 'C': (1, 0.5)}
    nx.draw_networkx_nodes(G1, pos1, node_color='lightgreen', 
                          node_size=2000, ax=axes[0], alpha=0.9)
    nx.draw_networkx_labels(G1, pos1, font_size=11, font_weight='bold', ax=axes[0])
    nx.draw_networkx_edges(G1, pos1, edge_color='gray', 
                          arrows=True, arrowsize=15, width=2, ax=axes[0])
    axes[0].set_title('Chain: A → B → C\nA ⊥ C | B', fontsize=12, fontweight='bold')
    axes[0].axis('off')
    
    # Fork: A <- B -> C
    G2 = nx.DiGraph()
    G2.add_edges_from([('B', 'A'), ('B', 'C')])
    pos2 = {'A': (0, 0.3), 'B': (0.5, 0.7), 'C': (1, 0.3)}
    nx.draw_networkx_nodes(G2, pos2, node_color='lightyellow', 
                          node_size=2000, ax=axes[1], alpha=0.9)
    nx.draw_networkx_labels(G2, pos2, font_size=11, font_weight='bold', ax=axes[1])
    nx.draw_networkx_edges(G2, pos2, edge_color='gray', 
                          arrows=True, arrowsize=15, width=2, ax=axes[1])
    axes[1].set_title('Fork: A ← B → C\nA ⊥ C | B', fontsize=12, fontweight='bold')
    axes[1].axis('off')
    
    # Collider: A -> B <- C
    G3 = nx.DiGraph()
    G3.add_edges_from([('A', 'B'), ('C', 'B')])
    pos3 = {'A': (0, 0.7), 'B': (0.5, 0.3), 'C': (1, 0.7)}
    nx.draw_networkx_nodes(G3, pos3, node_color='lightpink', 
                          node_size=2000, ax=axes[2], alpha=0.9)
    nx.draw_networkx_labels(G3, pos3, font_size=11, font_weight='bold', ax=axes[2])
    nx.draw_networkx_edges(G3, pos3, edge_color='gray', 
                          arrows=True, arrowsize=15, width=2, ax=axes[2])
    axes[2].set_title('Collider: A → B ← C\nA ⊥ C (без B)', fontsize=12, fontweight='bold')
    axes[2].axis('off')
    
    plt.suptitle('Условная независимость в Байесовских сетях', 
                fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    return fig_to_base64(fig)

# ============================================================================
# HMM ILLUSTRATIONS
# ============================================================================

def generate_hmm_structure():
    """Generate HMM structure diagram."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Hidden states
    hidden_states = ['S₁', 'S₂', 'S₃', 'S₄']
    # Observations
    observations = ['O₁', 'O₂', 'O₃', 'O₄']
    
    # Positions for hidden states (top row)
    h_pos = [(i, 0.7) for i in range(len(hidden_states))]
    # Positions for observations (bottom row)
    o_pos = [(i, 0.3) for i in range(len(observations))]
    
    # Draw hidden states
    for i, (state, pos) in enumerate(zip(hidden_states, h_pos)):
        circle = plt.Circle(pos, 0.12, color='lightblue', alpha=0.7, zorder=2)
        ax.add_patch(circle)
        ax.text(pos[0], pos[1], state, ha='center', va='center', 
               fontsize=12, fontweight='bold', zorder=3)
    
    # Draw observations
    for i, (obs, pos) in enumerate(zip(observations, o_pos)):
        circle = plt.Circle(pos, 0.12, color='lightgreen', alpha=0.7, zorder=2)
        ax.add_patch(circle)
        ax.text(pos[0], pos[1], obs, ha='center', va='center', 
               fontsize=12, fontweight='bold', zorder=3)
    
    # Draw transition arrows (between hidden states)
    for i in range(len(hidden_states) - 1):
        arrow = FancyArrowPatch(
            (h_pos[i][0] + 0.13, h_pos[i][1]),
            (h_pos[i+1][0] - 0.13, h_pos[i+1][1]),
            arrowstyle='->', mutation_scale=20, lw=2,
            color='gray', zorder=1
        )
        ax.add_patch(arrow)
    
    # Draw emission arrows (hidden to observed)
    for i in range(len(hidden_states)):
        arrow = FancyArrowPatch(
            (h_pos[i][0], h_pos[i][1] - 0.13),
            (o_pos[i][0], o_pos[i][1] + 0.13),
            arrowstyle='->', mutation_scale=20, lw=2,
            color='orange', linestyle='--', zorder=1
        )
        ax.add_patch(arrow)
    
    # Add legend
    ax.plot([], [], 'o', color='lightblue', markersize=15, 
           label='Скрытые состояния', alpha=0.7)
    ax.plot([], [], 'o', color='lightgreen', markersize=15, 
           label='Наблюдения', alpha=0.7)
    ax.plot([], [], '->', color='gray', linewidth=2, label='Переходы')
    ax.plot([], [], '-->', color='orange', linewidth=2, label='Эмиссии')
    ax.legend(loc='upper right', fontsize=10)
    
    ax.set_title('Структура скрытой марковской модели (HMM)', 
                fontsize=14, fontweight='bold', pad=20)
    ax.set_xlim(-0.5, 3.5)
    ax.set_ylim(0.1, 0.9)
    ax.axis('off')
    
    return fig_to_base64(fig)

def generate_hmm_transition_matrix():
    """Generate transition and emission matrices heatmaps."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Transition matrix
    trans_matrix = np.array([
        [0.7, 0.2, 0.1],
        [0.1, 0.8, 0.1],
        [0.1, 0.1, 0.8]
    ])
    
    im1 = axes[0].imshow(trans_matrix, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)
    axes[0].set_xticks(range(3))
    axes[0].set_yticks(range(3))
    axes[0].set_xticklabels(['S₁', 'S₂', 'S₃'])
    axes[0].set_yticklabels(['S₁', 'S₂', 'S₃'])
    axes[0].set_xlabel('Следующее состояние', fontsize=11)
    axes[0].set_ylabel('Текущее состояние', fontsize=11)
    axes[0].set_title('Матрица переходов A', fontsize=12, fontweight='bold')
    
    # Add values
    for i in range(3):
        for j in range(3):
            text = axes[0].text(j, i, f'{trans_matrix[i, j]:.1f}',
                              ha="center", va="center", color="black", fontsize=10)
    
    fig.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)
    
    # Emission matrix
    emis_matrix = np.array([
        [0.7, 0.3],
        [0.4, 0.6],
        [0.1, 0.9]
    ])
    
    im2 = axes[1].imshow(emis_matrix, cmap='YlGnBu', aspect='auto', vmin=0, vmax=1)
    axes[1].set_xticks(range(2))
    axes[1].set_yticks(range(3))
    axes[1].set_xticklabels(['O₁', 'O₂'])
    axes[1].set_yticklabels(['S₁', 'S₂', 'S₃'])
    axes[1].set_xlabel('Наблюдение', fontsize=11)
    axes[1].set_ylabel('Скрытое состояние', fontsize=11)
    axes[1].set_title('Матрица эмиссий B', fontsize=12, fontweight='bold')
    
    # Add values
    for i in range(3):
        for j in range(2):
            text = axes[1].text(j, i, f'{emis_matrix[i, j]:.1f}',
                              ha="center", va="center", color="black", fontsize=10)
    
    fig.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)
    
    plt.suptitle('Параметры HMM', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    return fig_to_base64(fig)

def generate_viterbi_algorithm():
    """Generate Viterbi algorithm visualization."""
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Trellis diagram
    n_states = 3
    n_time = 5
    
    # Positions
    for t in range(n_time):
        for s in range(n_states):
            x, y = t, s
            circle = plt.Circle((x, y), 0.2, color='lightblue', alpha=0.7, zorder=2)
            ax.add_patch(circle)
            ax.text(x, y, f't={t}\ns={s}', ha='center', va='center', 
                   fontsize=9, zorder=3)
    
    # Draw some example paths
    # Best path
    best_path = [(0, 0), (1, 0), (2, 1), (3, 2), (4, 2)]
    for i in range(len(best_path) - 1):
        arrow = FancyArrowPatch(
            (best_path[i][0] + 0.2, best_path[i][1]),
            (best_path[i+1][0] - 0.2, best_path[i+1][1]),
            arrowstyle='->', mutation_scale=15, lw=3,
            color='red', zorder=1, label='Лучший путь' if i == 0 else ''
        )
        ax.add_patch(arrow)
    
    # Other paths
    for t in range(n_time - 1):
        for s1 in range(n_states):
            for s2 in range(n_states):
                if (t, s1) not in best_path[:-1] or (t+1, s2) not in best_path[1:]:
                    arrow = FancyArrowPatch(
                        (t + 0.2, s1), (t + 1 - 0.2, s2),
                        arrowstyle='->', mutation_scale=10, lw=1,
                        color='gray', alpha=0.3, zorder=0
                    )
                    ax.add_patch(arrow)
    
    ax.set_xlim(-0.5, n_time - 0.5)
    ax.set_ylim(-0.5, n_states - 0.5)
    ax.set_xlabel('Время (t)', fontsize=12)
    ax.set_ylabel('Состояние (s)', fontsize=12)
    ax.set_title('Алгоритм Витерби: Поиск наиболее вероятной последовательности', 
                fontsize=14, fontweight='bold', pad=20)
    
    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='red', lw=3, label='Оптимальный путь'),
        Line2D([0], [0], color='gray', lw=1, alpha=0.3, label='Возможные пути')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
    
    ax.grid(True, alpha=0.3)
    
    return fig_to_base64(fig)

def generate_forward_backward():
    """Generate Forward-Backward algorithm visualization."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # Forward algorithm
    T = 5
    N = 3
    
    # Generate synthetic forward probabilities
    forward_probs = np.random.rand(N, T)
    forward_probs = forward_probs / forward_probs.sum(axis=0, keepdims=True)
    
    im1 = axes[0].imshow(forward_probs, cmap='Blues', aspect='auto', vmin=0, vmax=1)
    axes[0].set_xlabel('Время', fontsize=11)
    axes[0].set_ylabel('Состояние', fontsize=11)
    axes[0].set_title('Forward алгоритм: α_t(i) = P(o₁,...,o_t, s_t=i | λ)', 
                     fontsize=12, fontweight='bold')
    axes[0].set_xticks(range(T))
    axes[0].set_yticks(range(N))
    axes[0].set_xticklabels([f't={i}' for i in range(T)])
    axes[0].set_yticklabels([f's={i}' for i in range(N)])
    fig.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04, label='Вероятность')
    
    # Backward algorithm
    backward_probs = np.random.rand(N, T)
    backward_probs = backward_probs / backward_probs.sum(axis=0, keepdims=True)
    
    im2 = axes[1].imshow(backward_probs, cmap='Oranges', aspect='auto', vmin=0, vmax=1)
    axes[1].set_xlabel('Время', fontsize=11)
    axes[1].set_ylabel('Состояние', fontsize=11)
    axes[1].set_title('Backward алгоритм: β_t(i) = P(o_t+1,...,o_T | s_t=i, λ)', 
                     fontsize=12, fontweight='bold')
    axes[1].set_xticks(range(T))
    axes[1].set_yticks(range(N))
    axes[1].set_xticklabels([f't={i}' for i in range(T)])
    axes[1].set_yticklabels([f's={i}' for i in range(N)])
    fig.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04, label='Вероятность')
    
    plt.suptitle('Forward-Backward алгоритм для HMM', 
                fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()
    
    return fig_to_base64(fig)

# ============================================================================
# CRF ILLUSTRATIONS
# ============================================================================

def generate_crf_structure():
    """Generate CRF structure diagram."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Labels (like HMM hidden states)
    labels = ['y₁', 'y₂', 'y₃', 'y₄']
    # Input features
    features = ['x₁', 'x₂', 'x₃', 'x₄']
    
    # Positions for labels (top row)
    l_pos = [(i, 0.7) for i in range(len(labels))]
    # Positions for features (bottom row)
    f_pos = [(i, 0.3) for i in range(len(features))]
    
    # Draw labels
    for i, (label, pos) in enumerate(zip(labels, l_pos)):
        square = FancyBboxPatch((pos[0]-0.12, pos[1]-0.12), 0.24, 0.24,
                               boxstyle="round,pad=0.02", 
                               facecolor='lightcoral', edgecolor='black',
                               alpha=0.7, zorder=2)
        ax.add_patch(square)
        ax.text(pos[0], pos[1], label, ha='center', va='center', 
               fontsize=12, fontweight='bold', zorder=3)
    
    # Draw features
    for i, (feat, pos) in enumerate(zip(features, f_pos)):
        circle = plt.Circle(pos, 0.12, color='lightgreen', alpha=0.7, zorder=2)
        ax.add_patch(circle)
        ax.text(pos[0], pos[1], feat, ha='center', va='center', 
               fontsize=12, fontweight='bold', zorder=3)
    
    # Draw undirected edges between labels
    for i in range(len(labels) - 1):
        ax.plot([l_pos[i][0] + 0.12, l_pos[i+1][0] - 0.12],
               [l_pos[i][1], l_pos[i+1][1]],
               'k-', lw=2, zorder=1)
    
    # Draw dependencies from features to labels
    for i in range(len(features)):
        arrow = FancyArrowPatch(
            (f_pos[i][0], f_pos[i][1] + 0.13),
            (l_pos[i][0], l_pos[i][1] - 0.13),
            arrowstyle='->', mutation_scale=20, lw=2,
            color='blue', linestyle='--', zorder=1
        )
        ax.add_patch(arrow)
    
    # Add legend
    ax.plot([], [], 's', color='lightcoral', markersize=15, 
           label='Метки (labels)', alpha=0.7)
    ax.plot([], [], 'o', color='lightgreen', markersize=15, 
           label='Признаки (features)', alpha=0.7)
    ax.plot([], [], 'k-', linewidth=2, label='Зависимости между метками')
    ax.plot([], [], '-->', color='blue', linewidth=2, label='Признаковые функции')
    ax.legend(loc='upper right', fontsize=10)
    
    ax.set_title('Структура условного случайного поля (CRF)', 
                fontsize=14, fontweight='bold', pad=20)
    ax.set_xlim(-0.5, 3.5)
    ax.set_ylim(0.1, 0.9)
    ax.axis('off')
    
    return fig_to_base64(fig)

def generate_hmm_vs_crf():
    """Generate comparison between HMM and CRF."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # HMM (generative)
    ax = axes[0]
    # Simple HMM structure
    h_states = ['h₁', 'h₂', 'h₃']
    o_states = ['o₁', 'o₂', 'o₃']
    h_pos = [(i, 0.7) for i in range(3)]
    o_pos = [(i, 0.3) for i in range(3)]
    
    for i, (state, pos) in enumerate(zip(h_states, h_pos)):
        circle = plt.Circle(pos, 0.12, color='lightblue', alpha=0.7, zorder=2)
        ax.add_patch(circle)
        ax.text(pos[0], pos[1], state, ha='center', va='center', 
               fontsize=11, fontweight='bold', zorder=3)
    
    for i, (obs, pos) in enumerate(zip(o_states, o_pos)):
        circle = plt.Circle(pos, 0.12, color='lightgreen', alpha=0.7, zorder=2)
        ax.add_patch(circle)
        ax.text(pos[0], pos[1], obs, ha='center', va='center', 
               fontsize=11, fontweight='bold', zorder=3)
    
    for i in range(2):
        arrow = FancyArrowPatch(
            (h_pos[i][0] + 0.13, h_pos[i][1]),
            (h_pos[i+1][0] - 0.13, h_pos[i+1][1]),
            arrowstyle='->', mutation_scale=15, lw=2,
            color='gray', zorder=1
        )
        ax.add_patch(arrow)
    
    for i in range(3):
        arrow = FancyArrowPatch(
            (h_pos[i][0], h_pos[i][1] - 0.13),
            (o_pos[i][0], o_pos[i][1] + 0.13),
            arrowstyle='->', mutation_scale=15, lw=2,
            color='orange', linestyle='--', zorder=1
        )
        ax.add_patch(arrow)
    
    ax.set_title('HMM (Генеративная модель)\nP(X,Y)', 
                fontsize=12, fontweight='bold')
    ax.set_xlim(-0.3, 2.3)
    ax.set_ylim(0.1, 0.9)
    ax.axis('off')
    
    # CRF (discriminative)
    ax = axes[1]
    # Simple CRF structure
    y_states = ['y₁', 'y₂', 'y₃']
    x_states = ['x₁', 'x₂', 'x₃']
    y_pos = [(i, 0.7) for i in range(3)]
    x_pos = [(i, 0.3) for i in range(3)]
    
    for i, (label, pos) in enumerate(zip(y_states, y_pos)):
        square = FancyBboxPatch((pos[0]-0.12, pos[1]-0.12), 0.24, 0.24,
                               boxstyle="round,pad=0.02", 
                               facecolor='lightcoral', edgecolor='black',
                               alpha=0.7, zorder=2)
        ax.add_patch(square)
        ax.text(pos[0], pos[1], label, ha='center', va='center', 
               fontsize=11, fontweight='bold', zorder=3)
    
    for i, (feat, pos) in enumerate(zip(x_states, x_pos)):
        circle = plt.Circle(pos, 0.12, color='lightgreen', alpha=0.7, zorder=2)
        ax.add_patch(circle)
        ax.text(pos[0], pos[1], feat, ha='center', va='center', 
               fontsize=11, fontweight='bold', zorder=3)
    
    for i in range(2):
        ax.plot([y_pos[i][0] + 0.12, y_pos[i+1][0] - 0.12],
               [y_pos[i][1], y_pos[i+1][1]],
               'k-', lw=2, zorder=1)
    
    for i in range(3):
        arrow = FancyArrowPatch(
            (x_pos[i][0], x_pos[i][1] + 0.13),
            (y_pos[i][0], y_pos[i][1] - 0.13),
            arrowstyle='->', mutation_scale=15, lw=2,
            color='blue', linestyle='--', zorder=1
        )
        ax.add_patch(arrow)
    
    ax.set_title('CRF (Дискриминативная модель)\nP(Y|X)', 
                fontsize=12, fontweight='bold')
    ax.set_xlim(-0.3, 2.3)
    ax.set_ylim(0.1, 0.9)
    ax.axis('off')
    
    plt.suptitle('Сравнение HMM и CRF', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    return fig_to_base64(fig)

def generate_crf_feature_functions():
    """Generate CRF feature functions visualization."""
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Example feature functions
    features = [
        'word.lower()',
        'word.isupper()',
        'word.istitle()',
        'word[-3:]',
        '-1:word.lower()',
        '+1:word.lower()',
        'postag',
        'BOS/EOS'
    ]
    
    # Example weights
    weights = np.array([0.8, 0.3, 0.9, 0.6, 0.7, 0.5, 1.2, 0.4])
    
    colors = ['green' if w > 0.6 else 'orange' if w > 0.4 else 'red' for w in weights]
    
    bars = ax.barh(features, weights, color=colors, alpha=0.7, edgecolor='black')
    
    ax.set_xlabel('Вес признака', fontsize=12)
    ax.set_title('Примеры признаковых функций в CRF для NER', 
                fontsize=14, fontweight='bold', pad=20)
    ax.set_xlim(0, 1.4)
    ax.grid(axis='x', alpha=0.3)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='green', alpha=0.7, label='Высокий вес (>0.6)'),
        Patch(facecolor='orange', alpha=0.7, label='Средний вес (0.4-0.6)'),
        Patch(facecolor='red', alpha=0.7, label='Низкий вес (<0.4)')
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=10)
    
    return fig_to_base64(fig)

# ============================================================================
# MAIN GENERATION FUNCTION
# ============================================================================

def generate_all_illustrations():
    """Generate all graphical models illustrations."""
    print("Generating Graphical Models illustrations...")
    
    illustrations = {}
    
    # Bayesian Networks
    print("  - Generating Bayesian network example...")
    illustrations['bayesian_network'] = generate_bayesian_network_example()
    
    print("  - Generating Markov network example...")
    illustrations['markov_network'] = generate_markov_network_example()
    
    print("  - Generating d-separation examples...")
    illustrations['d_separation'] = generate_d_separation_examples()
    
    # HMM
    print("  - Generating HMM structure...")
    illustrations['hmm_structure'] = generate_hmm_structure()
    
    print("  - Generating HMM matrices...")
    illustrations['hmm_matrices'] = generate_hmm_transition_matrix()
    
    print("  - Generating Viterbi algorithm...")
    illustrations['viterbi'] = generate_viterbi_algorithm()
    
    print("  - Generating Forward-Backward algorithm...")
    illustrations['forward_backward'] = generate_forward_backward()
    
    # CRF
    print("  - Generating CRF structure...")
    illustrations['crf_structure'] = generate_crf_structure()
    
    print("  - Generating HMM vs CRF comparison...")
    illustrations['hmm_vs_crf'] = generate_hmm_vs_crf()
    
    print("  - Generating CRF feature functions...")
    illustrations['crf_features'] = generate_crf_feature_functions()
    
    print("All illustrations generated successfully!")
    return illustrations

if __name__ == '__main__':
    illustrations = generate_all_illustrations()
    print(f"\nGenerated {len(illustrations)} illustrations:")
    for key in illustrations.keys():
        print(f"  - {key}")
