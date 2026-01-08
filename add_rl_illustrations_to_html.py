#!/usr/bin/env python3
"""
Add matplotlib illustrations to reinforcement learning cheatsheet HTML files.
"""

import re
from generate_reinforcement_learning_illustrations import generate_all_illustrations

def create_img_tag(base64_data, alt_text, width="100%"):
    """Create HTML img tag with base64 encoded image."""
    return f'''
    <div style="text-align: center; margin: 10px 0;">
      <img src="data:image/png;base64,{base64_data}" 
           alt="{alt_text}" 
           style="max-width: {width}; height: auto; border-radius: 4px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
    </div>
'''

def add_illustrations_to_rl_basics(html_content, illustrations):
    """Add illustrations to RL basics cheatsheet."""
    # Add RL cycle after the first block about what RL is
    pattern1 = r'(<h2>🔷 1\. Что такое RL</h2>.*?</ul></div>)'
    img1 = create_img_tag(illustrations['rl_cycle'], 'RL Agent-Environment Interaction', '90%')
    html_content = re.sub(pattern1, r'\1' + img1, html_content, count=1, flags=re.DOTALL)
    
    # Add Q-values grid after MDP section
    pattern2 = r'(V\(s\) = max_a \[R\(s,a\) \+ γ \* Σ P\(s\'|s,a\) \* V\(s\'\)\]</code></pre></div>)'
    img2 = create_img_tag(illustrations['q_values_grid'], 'Q-Values Visualization', '85%')
    html_content = re.sub(pattern2, r'\1' + img2, html_content, count=1)
    
    # Add exploration vs exploitation after section 7
    pattern3 = r'(<h2>🔷 7\. Exploration vs Exploitation</h2>.*?</ul></div>)'
    img3 = create_img_tag(illustrations['exploration_exploitation'], 'Exploration vs Exploitation', '95%')
    html_content = re.sub(pattern3, r'\1' + img3, html_content, count=1, flags=re.DOTALL)
    
    return html_content

def add_illustrations_to_q_learning_sarsa(html_content, illustrations):
    """Add illustrations to Q-learning and SARSA cheatsheet."""
    # Add Q-learning convergence after Q-learning algorithm section
    pattern1 = r'(Q\[state, action\] \+= alpha.*?state = next_state</code></pre></div>)'
    img1 = create_img_tag(illustrations['q_learning_convergence'], 'Q-Learning Convergence', '90%')
    html_content = re.sub(pattern1, r'\1' + img1, html_content, count=1, flags=re.DOTALL)
    
    # Add SARSA vs Q-learning comparison
    pattern2 = r'(# SARSA использует фактическое следующее действие</code></pre></div>)'
    img2 = create_img_tag(illustrations['sarsa_vs_qlearning'], 'SARSA vs Q-Learning', '90%')
    html_content = re.sub(pattern2, r'\1' + img2, html_content, count=1)
    
    # Add backup diagram
    pattern3 = r'(<h2>🔷.*?Q-Learning vs SARSA.*?</h2>)'
    img3 = create_img_tag(illustrations['backup_diagram'], 'Q-Learning Backup Diagram', '85%')
    match = re.search(pattern3, html_content, re.DOTALL | re.IGNORECASE)
    if match:
        insert_pos = match.end(1)
        html_content = html_content[:insert_pos] + img3 + html_content[insert_pos:]
    
    return html_content

def add_illustrations_to_dqn(html_content, illustrations):
    """Add illustrations to DQN cheatsheet."""
    # Add DQN architecture after architecture section
    pattern1 = r'(<h2>🔷 3\. DQN архитектура</h2>.*?</ul></div>)'
    img1 = create_img_tag(illustrations['dqn_architecture'], 'DQN Architecture', '95%')
    html_content = re.sub(pattern1, r'\1' + img1, html_content, count=1, flags=re.DOTALL)
    
    # Add experience replay after replay section
    pattern2 = r'(<h2>🔷 4\. Experience Replay</h2>.*?</ul></div>)'
    img2 = create_img_tag(illustrations['experience_replay'], 'Experience Replay', '90%')
    html_content = re.sub(pattern2, r'\1' + img2, html_content, count=1, flags=re.DOTALL)
    
    # Add target network after target network section
    pattern3 = r'(<h2>🔷 5\. Target Network</h2>.*?</ul></div>)'
    img3 = create_img_tag(illustrations['target_network'], 'Target Network', '90%')
    html_content = re.sub(pattern3, r'\1' + img3, html_content, count=1, flags=re.DOTALL)
    
    return html_content

def add_illustrations_to_policy_gradient(html_content, illustrations):
    """Add illustrations to Policy Gradient cheatsheet."""
    # Add policy gradient concept early in the file
    pattern1 = r'(<h2>🔷 1\..*?Policy Gradient.*?</h2>)'
    img1 = create_img_tag(illustrations['policy_gradient_concept'], 'Policy Gradient Concept', '85%')
    match = re.search(pattern1, html_content, re.DOTALL | re.IGNORECASE)
    if match:
        insert_pos = match.end(1)
        html_content = html_content[:insert_pos] + img1 + html_content[insert_pos:]
    
    # Add trajectory returns after REINFORCE algorithm
    pattern2 = r'(optimizer\.step\(\)</code></pre></div>)'
    img2 = create_img_tag(illustrations['trajectory_returns'], 'Trajectory Returns', '90%')
    match = re.search(pattern2, html_content)
    if match:
        insert_pos = match.end(1)
        html_content = html_content[:insert_pos] + img2 + html_content[insert_pos:]
    
    return html_content

def add_illustrations_to_actor_critic(html_content, illustrations):
    """Add illustrations to Actor-Critic cheatsheet."""
    # Add actor-critic architecture early in file
    pattern1 = r'(<h2>🔷 1\..*?Actor.*?Critic.*?</h2>)'
    img1 = create_img_tag(illustrations['actor_critic_architecture'], 'Actor-Critic Architecture', '90%')
    match = re.search(pattern1, html_content, re.DOTALL | re.IGNORECASE)
    if match:
        insert_pos = match.end(1)
        html_content = html_content[:insert_pos] + img1 + html_content[insert_pos:]
    
    # Add advantage function
    pattern2 = r'(<h2>🔷.*?Advantage.*?</h2>)'
    img2 = create_img_tag(illustrations['advantage_function'], 'Advantage Function', '95%')
    match = re.search(pattern2, html_content, re.DOTALL | re.IGNORECASE)
    if match:
        insert_pos = match.end(1)
        html_content = html_content[:insert_pos] + img2 + html_content[insert_pos:]
    
    return html_content

def add_illustrations_to_ppo(html_content, illustrations):
    """Add illustrations to PPO cheatsheet."""
    # Add PPO clipping after PPO objective section
    pattern1 = r'(<h2>🔷 9\. PPO objective</h2>.*?</div>)'
    img1 = create_img_tag(illustrations['ppo_clipping'], 'PPO Clipping Mechanism', '90%')
    match = re.search(pattern1, html_content, re.DOTALL)
    if match:
        insert_pos = match.end(1)
        html_content = html_content[:insert_pos] + img1 + html_content[insert_pos:]
    
    # Add training curves after PPO algorithm section
    pattern2 = r'(<h2>🔷 8\. PPO \(Proximal Policy Optimization\)</h2>.*?</div>)'
    img2 = create_img_tag(illustrations['ppo_training_curves'], 'PPO Training Curves', '95%')
    match = re.search(pattern2, html_content, re.DOTALL)
    if match:
        insert_pos = match.end(1)
        html_content = html_content[:insert_pos] + img2 + html_content[insert_pos:]
    
    return html_content

def add_illustrations_to_mdp(html_content, illustrations):
    """Add illustrations to MDP cheatsheet."""
    # Add MDP graph after definition
    pattern1 = r'(<h2>1\. Определение MDP</h2>.*?</div>)'
    img1 = create_img_tag(illustrations['mdp_graph'], 'Markov Decision Process Graph', '90%')
    match = re.search(pattern1, html_content, re.DOTALL)
    if match:
        insert_pos = match.end(1)
        html_content = html_content[:insert_pos] + img1 + html_content[insert_pos:]
    
    # Add value iteration after methods section
    pattern2 = r'(<h2>9\. Методы решения</h2>.*?</div>)'
    img2 = create_img_tag(illustrations['value_iteration'], 'Value Iteration Convergence', '90%')
    match = re.search(pattern2, html_content, re.DOTALL)
    if match:
        insert_pos = match.end(1)
        html_content = html_content[:insert_pos] + img2 + html_content[insert_pos:]
    
    # Add policy iteration after value iteration
    img3_combined = create_img_tag(illustrations['policy_iteration'], 'Policy Iteration Convergence', '90%')
    if img2 in html_content:
        # Insert after value iteration image
        pattern3 = re.escape(img2)
        html_content = re.sub(pattern3, img2 + img3_combined, html_content, count=1)
    
    return html_content

def add_illustrations_to_bandits(html_content, illustrations):
    """Add illustrations to Multi-Armed Bandits cheatsheet."""
    # Add regret curves after metrics section
    pattern1 = r'(<h2>2\. Ключевые метрики</h2>.*?</div>)'
    img1 = create_img_tag(illustrations['bandit_regret'], 'Multi-Armed Bandits Regret', '90%')
    match = re.search(pattern1, html_content, re.DOTALL)
    if match:
        insert_pos = match.end(1)
        html_content = html_content[:insert_pos] + img1 + html_content[insert_pos:]
    
    # Add exploration strategies after epsilon-greedy section
    pattern2 = r'(<h2>3\. ε-Greedy алгоритм</h2>.*?</div>)'
    img2 = create_img_tag(illustrations['exploration_strategies'], 'Exploration Strategies', '95%')
    match = re.search(pattern2, html_content, re.DOTALL)
    if match:
        insert_pos = match.end(1)
        html_content = html_content[:insert_pos] + img2 + html_content[insert_pos:]
    
    # Add optimal arm convergence after UCB section
    pattern3 = r'(<h2>4\. Upper Confidence Bound \(UCB\)</h2>.*?</div>)'
    img3 = create_img_tag(illustrations['optimal_arm_convergence'], 'Convergence to Optimal Arm', '90%')
    match = re.search(pattern3, html_content, re.DOTALL)
    if match:
        insert_pos = match.end(1)
        html_content = html_content[:insert_pos] + img3 + html_content[insert_pos:]
    
    return html_content

def process_html_file(filepath, add_illustrations_func, illustrations):
    """Process a single HTML file to add illustrations."""
    print(f"Processing {filepath}...")
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        # Add illustrations
        modified_content = add_illustrations_func(html_content, illustrations)
        
        # Check if content was actually modified
        if modified_content != html_content:
            # Write back
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(modified_content)
            print(f"  ✓ Successfully updated {filepath}")
            return True
        else:
            print(f"  ⚠ No changes made to {filepath} (patterns not found)")
            return False
    except Exception as e:
        print(f"  ✗ Error processing {filepath}: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function to add illustrations to all RL cheatsheets."""
    print("=" * 70)
    print("Adding matplotlib illustrations to reinforcement learning cheatsheets")
    print("=" * 70)
    
    # Generate all illustrations
    print("\n1. Generating illustrations...")
    illustrations = generate_all_illustrations()
    
    # Process each HTML file
    print("\n2. Adding illustrations to HTML files...")
    
    files_to_process = [
        ('cheatsheets/reinforcement_learning_basics_cheatsheet.html', add_illustrations_to_rl_basics),
        ('cheatsheets/q_learning_sarsa_cheatsheet.html', add_illustrations_to_q_learning_sarsa),
        ('cheatsheets/dqn_cheatsheet.html', add_illustrations_to_dqn),
        ('cheatsheets/policy_gradient_cheatsheet.html', add_illustrations_to_policy_gradient),
        ('cheatsheets/actor_critic_cheatsheet.html', add_illustrations_to_actor_critic),
        ('cheatsheets/ppo_reinforcement_learning_cheatsheet.html', add_illustrations_to_ppo),
        ('cheatsheets/markov_decision_processes_cheatsheet.html', add_illustrations_to_mdp),
        ('cheatsheets/multi_armed_bandits_cheatsheet.html', add_illustrations_to_bandits),
    ]
    
    success_count = 0
    for filepath, add_func in files_to_process:
        if process_html_file(filepath, add_func, illustrations):
            success_count += 1
    
    print("\n" + "=" * 70)
    print(f"Completed: {success_count}/{len(files_to_process)} files successfully updated")
    print("=" * 70)

if __name__ == '__main__':
    main()
