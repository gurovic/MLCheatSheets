#!/usr/bin/env python3
"""
Generate matplotlib illustrations for reinforcement learning cheatsheets.
This script creates high-quality visualizations and encodes them as base64 for inline HTML embedding.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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
# REINFORCEMENT LEARNING BASICS
# ============================================================================

def generate_rl_cycle():
    """Generate RL agent-environment interaction cycle diagram."""
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Agent box
    agent_rect = plt.Rectangle((1, 6), 3, 2, facecolor='#e3f2fd', edgecolor='#1976d2', linewidth=2)
    ax.add_patch(agent_rect)
    ax.text(2.5, 7, 'Agent', ha='center', va='center', fontsize=14, fontweight='bold')
    
    # Environment box
    env_rect = plt.Rectangle((6, 6), 3, 2, facecolor='#f3e5f5', edgecolor='#7b1fa2', linewidth=2)
    ax.add_patch(env_rect)
    ax.text(7.5, 7, 'Environment', ha='center', va='center', fontsize=14, fontweight='bold')
    
    # Action arrow (Agent -> Environment)
    ax.annotate('', xy=(6, 7.5), xytext=(4, 7.5),
                arrowprops=dict(arrowstyle='->', lw=2, color='#2e7d32'))
    ax.text(5, 7.8, 'Action (a)', ha='center', fontsize=11, color='#2e7d32')
    
    # Reward arrow (Environment -> Agent)
    ax.annotate('', xy=(4, 6.8), xytext=(6, 6.8),
                arrowprops=dict(arrowstyle='->', lw=2, color='#c62828'))
    ax.text(5, 6.5, 'Reward (r)', ha='center', fontsize=11, color='#c62828')
    
    # State arrow (Environment -> Agent)
    ax.annotate('', xy=(4, 6.2), xytext=(6, 6.2),
                arrowprops=dict(arrowstyle='->', lw=2, color='#f57c00'))
    ax.text(5, 5.9, 'State (s)', ha='center', fontsize=11, color='#f57c00')
    
    ax.set_title('Reinforcement Learning: Agent-Environment Interaction', fontsize=14, fontweight='bold', pad=15)
    
    return fig_to_base64(fig)

def generate_q_values_grid():
    """Generate Q-values visualization for a simple grid world."""
    np.random.seed(42)
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Create a 5x5 grid
    grid_size = 5
    q_values = np.random.rand(grid_size, grid_size) * 10
    q_values[0, 0] = 0  # Start state
    q_values[4, 4] = 15  # Goal state
    
    im = ax.imshow(q_values, cmap='YlGnBu', aspect='auto')
    
    # Add values to cells
    for i in range(grid_size):
        for j in range(grid_size):
            text = ax.text(j, i, f'{q_values[i, j]:.1f}',
                          ha="center", va="center", color="black", fontsize=10)
    
    # Mark special states
    ax.text(0, 0, 'START', ha='center', va='bottom', fontsize=8, color='red', fontweight='bold')
    ax.text(4, 4, 'GOAL', ha='center', va='bottom', fontsize=8, color='green', fontweight='bold')
    
    ax.set_xticks(range(grid_size))
    ax.set_yticks(range(grid_size))
    ax.set_title('Q-Values in Grid World', fontsize=14, fontweight='bold')
    plt.colorbar(im, ax=ax, label='Q-value')
    
    return fig_to_base64(fig)

def generate_exploration_exploitation():
    """Generate exploration vs exploitation comparison."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    episodes = np.arange(1000)
    
    # Epsilon-greedy with decay
    epsilon = 1.0 * np.exp(-episodes / 200)
    ax1.plot(episodes, epsilon, 'b-', linewidth=2, label='ε (exploration rate)')
    ax1.plot(episodes, 1 - epsilon, 'r-', linewidth=2, label='1-ε (exploitation rate)')
    ax1.set_xlabel('Episodes')
    ax1.set_ylabel('Probability')
    ax1.set_title('ε-greedy Strategy: Exploration Decay', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Cumulative reward comparison
    np.random.seed(42)
    reward_explore = np.cumsum(np.random.randn(1000) * 0.5 + 0.8)
    reward_exploit = np.cumsum(np.random.randn(1000) * 0.3 + 0.6)
    reward_balanced = np.cumsum(np.random.randn(1000) * 0.4 + 1.0)
    
    ax2.plot(episodes, reward_explore, 'g-', alpha=0.7, label='High exploration', linewidth=2)
    ax2.plot(episodes, reward_exploit, 'r-', alpha=0.7, label='High exploitation', linewidth=2)
    ax2.plot(episodes, reward_balanced, 'b-', alpha=0.7, label='Balanced', linewidth=2.5)
    ax2.set_xlabel('Episodes')
    ax2.set_ylabel('Cumulative Reward')
    ax2.set_title('Cumulative Rewards: Different Strategies', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig_to_base64(fig)

# ============================================================================
# Q-LEARNING AND SARSA
# ============================================================================

def generate_q_learning_convergence():
    """Generate Q-learning convergence plot."""
    np.random.seed(42)
    episodes = np.arange(500)
    
    # Simulate Q-learning convergence
    optimal_value = 100
    q_values = optimal_value * (1 - np.exp(-episodes / 100)) + np.random.randn(500) * 5
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(episodes, q_values, 'b-', alpha=0.6, linewidth=1.5, label='Q-value estimate')
    ax.axhline(y=optimal_value, color='r', linestyle='--', linewidth=2, label='Optimal Q-value')
    ax.fill_between(episodes, q_values - 10, q_values + 10, alpha=0.2, color='blue')
    
    ax.set_xlabel('Episodes')
    ax.set_ylabel('Q-value')
    ax.set_title('Q-Learning Convergence', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return fig_to_base64(fig)

def generate_sarsa_vs_qlearning():
    """Generate SARSA vs Q-Learning comparison."""
    np.random.seed(42)
    episodes = np.arange(300)
    
    # Simulate learning curves
    q_learning = 50 * (1 - np.exp(-episodes / 60)) + np.random.randn(300) * 3
    sarsa = 45 * (1 - np.exp(-episodes / 80)) + np.random.randn(300) * 2
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(episodes, q_learning, 'b-', linewidth=2, label='Q-Learning (off-policy)', alpha=0.8)
    ax.plot(episodes, sarsa, 'g-', linewidth=2, label='SARSA (on-policy)', alpha=0.8)
    
    ax.set_xlabel('Episodes')
    ax.set_ylabel('Average Reward')
    ax.set_title('Q-Learning vs SARSA: Learning Curves', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return fig_to_base64(fig)

def generate_backup_diagram():
    """Generate backup diagram for Q-learning."""
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Current state
    state_circle = plt.Circle((2, 7), 0.5, facecolor='#64b5f6', edgecolor='#1976d2', linewidth=2)
    ax.add_patch(state_circle)
    ax.text(2, 7, 's', ha='center', va='center', fontsize=12, fontweight='bold')
    ax.text(2, 6, 'Current State', ha='center', fontsize=10)
    
    # Actions
    actions = [(5, 8.5), (5, 7), (5, 5.5)]
    action_labels = ['a₁', 'a₂', 'a₃']
    
    for i, (x, y) in enumerate(actions):
        ax.annotate('', xy=(x-0.3, y), xytext=(2.5, 7),
                   arrowprops=dict(arrowstyle='->', lw=1.5, color='gray'))
        next_circle = plt.Circle((x, y), 0.4, facecolor='#81c784', edgecolor='#388e3c', linewidth=2)
        ax.add_patch(next_circle)
        ax.text(x, y, f"s'", ha='center', va='center', fontsize=10, fontweight='bold')
        ax.text(x+1, y, f'Q(s,{action_labels[i]})', ha='left', fontsize=9)
    
    # Max Q-value indicator
    ax.text(5, 4.5, 'max Q(s\', a\')', ha='center', fontsize=11, 
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
    
    ax.set_title('Q-Learning Backup Diagram', fontsize=14, fontweight='bold', pad=15)
    
    return fig_to_base64(fig)

# ============================================================================
# DQN
# ============================================================================

def generate_dqn_architecture():
    """Generate DQN architecture diagram."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Input layer
    input_rect = plt.Rectangle((0.5, 4), 1.5, 2, facecolor='#e1f5fe', edgecolor='#01579b', linewidth=2)
    ax.add_patch(input_rect)
    ax.text(1.25, 5, 'State\nInput', ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Hidden layers
    hidden_positions = [(3, 3.5), (5, 3.5), (7, 3.5)]
    for x, y in hidden_positions:
        hidden_rect = plt.Rectangle((x, y), 1, 3, facecolor='#fff3e0', edgecolor='#e65100', linewidth=2)
        ax.add_patch(hidden_rect)
        ax.text(x+0.5, y+1.5, 'Hidden\nLayer', ha='center', va='center', fontsize=9)
    
    # Output layer
    output_rect = plt.Rectangle((8.5, 4), 1, 2, facecolor='#f3e5f5', edgecolor='#4a148c', linewidth=2)
    ax.add_patch(output_rect)
    ax.text(9, 5, 'Q-values\n(actions)', ha='center', va='center', fontsize=9, fontweight='bold')
    
    # Arrows
    arrow_positions = [(2, 5, 3, 5), (4, 5, 5, 5), (6, 5, 7, 5), (8, 5, 8.5, 5)]
    for x1, y1, x2, y2 in arrow_positions:
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                   arrowprops=dict(arrowstyle='->', lw=2, color='#424242'))
    
    ax.set_title('Deep Q-Network (DQN) Architecture', fontsize=14, fontweight='bold', pad=15)
    
    return fig_to_base64(fig)

def generate_experience_replay():
    """Generate experience replay visualization."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Timeline
    episodes = np.arange(100)
    
    # Simulate replay buffer usage
    np.random.seed(42)
    experiences_collected = np.cumsum(np.random.randint(1, 10, 100))
    experiences_used = np.cumsum(np.random.randint(5, 15, 100))
    
    ax.plot(episodes, experiences_collected, 'b-', linewidth=2, label='Experiences Collected', alpha=0.8)
    ax.plot(episodes, experiences_used, 'r--', linewidth=2, label='Experiences Sampled', alpha=0.8)
    ax.fill_between(episodes, 0, experiences_collected, alpha=0.2, color='blue')
    
    ax.set_xlabel('Training Steps')
    ax.set_ylabel('Number of Experiences')
    ax.set_title('Experience Replay: Collection vs Sampling', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return fig_to_base64(fig)

def generate_target_network():
    """Generate target network concept visualization."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    episodes = np.arange(200)
    
    # Simulate network updates
    np.random.seed(42)
    online_values = np.cumsum(np.random.randn(200) * 0.5 + 0.1)
    
    # Target network updated periodically
    target_values = np.zeros(200)
    update_interval = 20
    for i in range(0, 200, update_interval):
        target_values[i:min(i+update_interval, 200)] = online_values[i]
    
    ax.plot(episodes, online_values, 'b-', linewidth=2, label='Online Network (updated continuously)', alpha=0.8)
    ax.plot(episodes, target_values, 'r--', linewidth=2, label='Target Network (updated periodically)', alpha=0.8)
    
    # Mark update points
    for i in range(0, 200, update_interval):
        ax.axvline(x=i, color='gray', linestyle=':', alpha=0.5)
    
    ax.set_xlabel('Training Steps')
    ax.set_ylabel('Network Parameters')
    ax.set_title('DQN: Online vs Target Network Updates', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return fig_to_base64(fig)

# ============================================================================
# POLICY GRADIENT
# ============================================================================

def generate_policy_gradient_concept():
    """Generate policy gradient concept visualization."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create action probability distribution
    actions = np.arange(5)
    probs_initial = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    probs_trained = np.array([0.05, 0.1, 0.6, 0.15, 0.1])
    
    x = np.arange(len(actions))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, probs_initial, width, label='Initial Policy', alpha=0.8, color='#90caf9')
    bars2 = ax.bar(x + width/2, probs_trained, width, label='Trained Policy', alpha=0.8, color='#66bb6a')
    
    ax.set_xlabel('Actions')
    ax.set_ylabel('Probability')
    ax.set_title('Policy Gradient: Action Probability Distribution', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'a{i}' for i in actions])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    return fig_to_base64(fig)

def generate_trajectory_returns():
    """Generate trajectory returns visualization."""
    np.random.seed(42)
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Simulate different trajectories
    time_steps = np.arange(50)
    trajectories = []
    
    for i in range(5):
        reward = np.random.randn(50) * 2 + (i - 2)
        cumulative_return = np.cumsum(reward)
        trajectories.append(cumulative_return)
        ax.plot(time_steps, cumulative_return, alpha=0.7, linewidth=2, label=f'Trajectory {i+1}')
    
    # Highlight best trajectory
    best_idx = np.argmax([t[-1] for t in trajectories])
    ax.plot(time_steps, trajectories[best_idx], 'r-', linewidth=3, alpha=0.9, label='Best Trajectory')
    
    ax.set_xlabel('Time Steps')
    ax.set_ylabel('Cumulative Return')
    ax.set_title('Policy Gradient: Trajectory Returns', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return fig_to_base64(fig)

# ============================================================================
# ACTOR-CRITIC
# ============================================================================

def generate_actor_critic_architecture():
    """Generate actor-critic architecture diagram."""
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # State input
    state_rect = plt.Rectangle((0.5, 4.5), 1.5, 1, facecolor='#e1f5fe', edgecolor='#01579b', linewidth=2)
    ax.add_patch(state_rect)
    ax.text(1.25, 5, 'State', ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Shared layers
    shared_rect = plt.Rectangle((3, 4), 2, 2, facecolor='#fff3e0', edgecolor='#e65100', linewidth=2)
    ax.add_patch(shared_rect)
    ax.text(4, 5, 'Shared\nLayers', ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Actor
    actor_rect = plt.Rectangle((6.5, 6), 2, 1.5, facecolor='#f3e5f5', edgecolor='#4a148c', linewidth=2)
    ax.add_patch(actor_rect)
    ax.text(7.5, 6.75, 'Actor\nπ(a|s)', ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Critic
    critic_rect = plt.Rectangle((6.5, 2.5), 2, 1.5, facecolor='#e8f5e9', edgecolor='#1b5e20', linewidth=2)
    ax.add_patch(critic_rect)
    ax.text(7.5, 3.25, 'Critic\nV(s)', ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Arrows
    ax.annotate('', xy=(3, 5), xytext=(2, 5),
               arrowprops=dict(arrowstyle='->', lw=2, color='#424242'))
    ax.annotate('', xy=(6.5, 6.75), xytext=(5, 5.5),
               arrowprops=dict(arrowstyle='->', lw=2, color='#424242'))
    ax.annotate('', xy=(6.5, 3.25), xytext=(5, 4.5),
               arrowprops=dict(arrowstyle='->', lw=2, color='#424242'))
    
    # Output labels
    ax.text(9, 6.75, 'Action\nProbabilities', ha='left', fontsize=9)
    ax.text(9, 3.25, 'Value\nEstimate', ha='left', fontsize=9)
    
    ax.set_title('Actor-Critic Architecture', fontsize=14, fontweight='bold', pad=15)
    
    return fig_to_base64(fig)

def generate_advantage_function():
    """Generate advantage function visualization."""
    np.random.seed(42)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Q-values and V-values
    states = np.arange(20)
    v_values = np.sin(states * 0.3) * 5 + 10
    q_values = v_values + np.random.randn(20) * 2
    advantage = q_values - v_values
    
    ax1.plot(states, q_values, 'b-', linewidth=2, label='Q(s,a)', marker='o')
    ax1.plot(states, v_values, 'r--', linewidth=2, label='V(s)')
    ax1.fill_between(states, v_values, q_values, alpha=0.3, color='green', label='Advantage A(s,a)')
    ax1.set_xlabel('States')
    ax1.set_ylabel('Value')
    ax1.set_title('Advantage Function: A(s,a) = Q(s,a) - V(s)', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Advantage distribution
    ax2.bar(states, advantage, color=['green' if a > 0 else 'red' for a in advantage], alpha=0.7)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax2.set_xlabel('States')
    ax2.set_ylabel('Advantage A(s,a)')
    ax2.set_title('Advantage Values (Positive = Better than Average)', fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    return fig_to_base64(fig)

# ============================================================================
# PPO
# ============================================================================

def generate_ppo_clipping():
    """Generate PPO clipping visualization."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Probability ratio
    ratio = np.linspace(0, 2.5, 100)
    epsilon = 0.2
    
    # Unclipped objective
    unclipped = ratio
    
    # Clipped objective
    clipped = np.clip(ratio, 1 - epsilon, 1 + epsilon)
    
    ax.plot(ratio, unclipped, 'b--', linewidth=2, label='Unclipped objective', alpha=0.7)
    ax.plot(ratio, clipped, 'r-', linewidth=2.5, label=f'Clipped objective (ε={epsilon})')
    ax.axvline(x=1-epsilon, color='gray', linestyle=':', alpha=0.5)
    ax.axvline(x=1+epsilon, color='gray', linestyle=':', alpha=0.5)
    ax.axvline(x=1, color='green', linestyle='--', alpha=0.5, label='Current policy')
    
    ax.set_xlabel('Probability Ratio (π_new / π_old)')
    ax.set_ylabel('Objective Value')
    ax.set_title('PPO Clipping Mechanism', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return fig_to_base64(fig)

def generate_ppo_training_curves():
    """Generate PPO training curves."""
    np.random.seed(42)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    episodes = np.arange(200)
    
    # Reward curve
    rewards = 100 * (1 - np.exp(-episodes / 50)) + np.random.randn(200) * 5
    ax1.plot(episodes, rewards, 'b-', linewidth=1, alpha=0.6)
    # Moving average
    window = 10
    moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
    ax1.plot(episodes[window-1:], moving_avg, 'r-', linewidth=2, label='Moving Average')
    ax1.set_xlabel('Episodes')
    ax1.set_ylabel('Episode Reward')
    ax1.set_title('PPO Training: Episode Rewards', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Loss curves
    policy_loss = 2 * np.exp(-episodes / 40) + np.random.randn(200) * 0.2
    value_loss = 5 * np.exp(-episodes / 50) + np.random.randn(200) * 0.3
    ax2.plot(episodes, policy_loss, 'g-', linewidth=1.5, label='Policy Loss', alpha=0.8)
    ax2.plot(episodes, value_loss, 'orange', linewidth=1.5, label='Value Loss', alpha=0.8)
    ax2.set_xlabel('Episodes')
    ax2.set_ylabel('Loss')
    ax2.set_title('PPO Training: Loss Curves', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig_to_base64(fig)

# ============================================================================
# MARKOV DECISION PROCESSES
# ============================================================================

def generate_mdp_graph():
    """Generate MDP state transition graph."""
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # States
    states = [
        (2, 7, 'S₁'),
        (5, 8, 'S₂'),
        (8, 7, 'S₃'),
        (5, 5, 'S₄'),
        (8, 3, 'Goal')
    ]
    
    for x, y, label in states:
        if label == 'Goal':
            circle = plt.Circle((x, y), 0.5, facecolor='#81c784', edgecolor='#2e7d32', linewidth=3)
        else:
            circle = plt.Circle((x, y), 0.5, facecolor='#64b5f6', edgecolor='#1976d2', linewidth=2)
        ax.add_patch(circle)
        ax.text(x, y, label, ha='center', va='center', fontsize=11, fontweight='bold')
    
    # Transitions with probabilities
    transitions = [
        ((2, 7), (5, 8), '0.7'),
        ((2, 7), (5, 5), '0.3'),
        ((5, 8), (8, 7), '0.8'),
        ((5, 8), (5, 5), '0.2'),
        ((8, 7), (8, 3), '1.0'),
        ((5, 5), (8, 3), '0.6'),
        ((5, 5), (2, 7), '0.4'),
    ]
    
    for (x1, y1), (x2, y2), prob in transitions:
        dx = x2 - x1
        dy = y2 - y1
        length = np.sqrt(dx**2 + dy**2)
        dx_norm = dx / length * 0.5
        dy_norm = dy / length * 0.5
        
        ax.annotate('', xy=(x2-dx_norm, y2-dy_norm), xytext=(x1+dx_norm, y1+dy_norm),
                   arrowprops=dict(arrowstyle='->', lw=1.5, color='#424242'))
        mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(mid_x, mid_y, f'P={prob}', fontsize=8, 
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax.set_title('Markov Decision Process: State Transitions', fontsize=14, fontweight='bold', pad=15)
    
    return fig_to_base64(fig)

def generate_value_iteration():
    """Generate value iteration convergence."""
    np.random.seed(42)
    fig, ax = plt.subplots(figsize=(10, 6))
    
    iterations = np.arange(30)
    n_states = 5
    
    # Simulate value iteration for different states
    for i in range(n_states):
        optimal_value = 10 + i * 2
        values = optimal_value * (1 - np.exp(-iterations / 5)) + np.random.randn(30) * 0.3
        ax.plot(iterations, values, linewidth=2, label=f'State {i+1}', alpha=0.8)
    
    ax.set_xlabel('Iterations')
    ax.set_ylabel('State Value V(s)')
    ax.set_title('Value Iteration: Convergence of State Values', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return fig_to_base64(fig)

def generate_policy_iteration():
    """Generate policy iteration steps."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    iterations = [0, 1, 2, 3, 4, 5]
    policy_changes = [100, 45, 20, 8, 2, 0]
    value_error = [10, 5, 2, 0.8, 0.3, 0.1]
    
    ax2 = ax.twinx()
    
    bars = ax.bar(iterations, policy_changes, alpha=0.7, color='#64b5f6', label='Policy Changes')
    line = ax2.plot(iterations, value_error, 'r-', linewidth=2.5, marker='o', 
                    markersize=8, label='Value Error')
    
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Number of Policy Changes', color='#1976d2')
    ax2.set_ylabel('Value Function Error', color='#c62828')
    ax.set_title('Policy Iteration: Convergence', fontsize=14, fontweight='bold')
    
    # Combine legends
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, loc='upper right')
    
    ax.grid(True, alpha=0.3, axis='x')
    
    return fig_to_base64(fig)

# ============================================================================
# MULTI-ARMED BANDITS
# ============================================================================

def generate_bandit_regret():
    """Generate regret curves for different strategies."""
    np.random.seed(42)
    fig, ax = plt.subplots(figsize=(10, 6))
    
    steps = np.arange(1000)
    
    # Simulate cumulative regret for different strategies
    strategies = {
        'Random': np.cumsum(np.random.rand(1000) * 0.5 + 0.4),
        'ε-greedy (ε=0.1)': np.cumsum(np.random.rand(1000) * 0.2 + 0.15),
        'ε-greedy (ε=0.01)': np.cumsum(np.random.rand(1000) * 0.15 + 0.08),
        'UCB': np.cumsum(np.random.rand(1000) * 0.1 + 0.05),
        'Thompson Sampling': np.cumsum(np.random.rand(1000) * 0.08 + 0.03),
    }
    
    colors = ['gray', 'blue', 'green', 'orange', 'red']
    for (name, regret), color in zip(strategies.items(), colors):
        ax.plot(steps, regret, linewidth=2, label=name, alpha=0.8, color=color)
    
    ax.set_xlabel('Steps')
    ax.set_ylabel('Cumulative Regret')
    ax.set_title('Multi-Armed Bandits: Cumulative Regret Comparison', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return fig_to_base64(fig)

def generate_exploration_strategies():
    """Generate exploration strategy comparison."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Arm selection frequency
    arms = ['Arm 1', 'Arm 2', 'Arm 3', 'Arm 4']
    epsilon_greedy = [15, 70, 10, 5]
    ucb = [25, 50, 15, 10]
    thompson = [20, 55, 15, 10]
    
    x = np.arange(len(arms))
    width = 0.25
    
    ax1.bar(x - width, epsilon_greedy, width, label='ε-greedy', alpha=0.8, color='#64b5f6')
    ax1.bar(x, ucb, width, label='UCB', alpha=0.8, color='#81c784')
    ax1.bar(x + width, thompson, width, label='Thompson', alpha=0.8, color='#ffb74d')
    
    ax1.set_xlabel('Arms')
    ax1.set_ylabel('Selection Frequency (%)')
    ax1.set_title('Arm Selection Distribution', fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(arms)
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Confidence bounds (UCB)
    np.random.seed(42)
    steps = np.arange(100)
    mean_reward = np.ones(100) * 0.7
    ucb_bound = 2 / np.sqrt(steps + 1)
    
    ax2.plot(steps, mean_reward, 'b-', linewidth=2, label='Estimated Mean')
    ax2.fill_between(steps, mean_reward - ucb_bound, mean_reward + ucb_bound, 
                     alpha=0.3, color='blue', label='UCB Confidence Bound')
    ax2.set_xlabel('Steps')
    ax2.set_ylabel('Reward Estimate')
    ax2.set_title('UCB: Confidence Bounds Over Time', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig_to_base64(fig)

def generate_optimal_arm_convergence():
    """Generate convergence to optimal arm."""
    np.random.seed(42)
    fig, ax = plt.subplots(figsize=(10, 6))
    
    steps = np.arange(500)
    
    # True arm values
    true_values = [0.3, 0.5, 0.7, 0.4]
    
    # Simulate learning of arm values
    for i, true_val in enumerate(true_values):
        learned = true_val * (1 - np.exp(-steps / 80)) + np.random.randn(500) * 0.05
        linestyle = '-' if i == 2 else '--'  # Optimal arm is arm 3
        linewidth = 3 if i == 2 else 1.5
        ax.plot(steps, learned, linestyle=linestyle, linewidth=linewidth, 
               label=f'Arm {i+1} (true: {true_val:.1f})', alpha=0.8)
        ax.axhline(y=true_val, color=f'C{i}', linestyle=':', alpha=0.5)
    
    ax.set_xlabel('Steps')
    ax.set_ylabel('Estimated Reward')
    ax.set_title('Multi-Armed Bandits: Convergence to True Values', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return fig_to_base64(fig)

# ============================================================================
# GENERATE ALL ILLUSTRATIONS
# ============================================================================

def generate_all_illustrations():
    """Generate all reinforcement learning illustrations."""
    print("Generating all reinforcement learning illustrations...")
    illustrations = {}
    
    # Reinforcement Learning Basics
    print("  - RL cycle diagram")
    illustrations['rl_cycle'] = generate_rl_cycle()
    print("  - Q-values grid")
    illustrations['q_values_grid'] = generate_q_values_grid()
    print("  - Exploration vs Exploitation")
    illustrations['exploration_exploitation'] = generate_exploration_exploitation()
    
    # Q-Learning and SARSA
    print("  - Q-learning convergence")
    illustrations['q_learning_convergence'] = generate_q_learning_convergence()
    print("  - SARSA vs Q-learning")
    illustrations['sarsa_vs_qlearning'] = generate_sarsa_vs_qlearning()
    print("  - Backup diagram")
    illustrations['backup_diagram'] = generate_backup_diagram()
    
    # DQN
    print("  - DQN architecture")
    illustrations['dqn_architecture'] = generate_dqn_architecture()
    print("  - Experience replay")
    illustrations['experience_replay'] = generate_experience_replay()
    print("  - Target network")
    illustrations['target_network'] = generate_target_network()
    
    # Policy Gradient
    print("  - Policy gradient concept")
    illustrations['policy_gradient_concept'] = generate_policy_gradient_concept()
    print("  - Trajectory returns")
    illustrations['trajectory_returns'] = generate_trajectory_returns()
    
    # Actor-Critic
    print("  - Actor-critic architecture")
    illustrations['actor_critic_architecture'] = generate_actor_critic_architecture()
    print("  - Advantage function")
    illustrations['advantage_function'] = generate_advantage_function()
    
    # PPO
    print("  - PPO clipping")
    illustrations['ppo_clipping'] = generate_ppo_clipping()
    print("  - PPO training curves")
    illustrations['ppo_training_curves'] = generate_ppo_training_curves()
    
    # Markov Decision Processes
    print("  - MDP graph")
    illustrations['mdp_graph'] = generate_mdp_graph()
    print("  - Value iteration")
    illustrations['value_iteration'] = generate_value_iteration()
    print("  - Policy iteration")
    illustrations['policy_iteration'] = generate_policy_iteration()
    
    # Multi-Armed Bandits
    print("  - Bandit regret")
    illustrations['bandit_regret'] = generate_bandit_regret()
    print("  - Exploration strategies")
    illustrations['exploration_strategies'] = generate_exploration_strategies()
    print("  - Optimal arm convergence")
    illustrations['optimal_arm_convergence'] = generate_optimal_arm_convergence()
    
    print(f"Generated {len(illustrations)} illustrations!")
    return illustrations

if __name__ == '__main__':
    illustrations = generate_all_illustrations()
    print("\nAll illustrations generated successfully!")
