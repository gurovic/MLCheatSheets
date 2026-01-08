#!/usr/bin/env python3
"""
Add matplotlib illustrations to special architectures cheatsheet HTML files.
"""

import re
from generate_special_architectures_illustrations import generate_all_illustrations

def create_img_tag(base64_data, alt_text, width="100%"):
    """Create HTML img tag with base64 encoded image."""
    return f'''
    <div style="text-align: center; margin: 10px 0;">
      <img src="data:image/png;base64,{base64_data}" 
           alt="{alt_text}" 
           style="max-width: {width}; height: auto; border-radius: 4px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
    </div>
'''

def add_illustrations_to_memory_networks(html_content, illustrations):
    """Add illustrations to Memory Networks cheatsheet."""
    # Add attention mechanism after the attention section
    attention_pattern = r'(# Attention weights:.*?p_i = softmax\(u\^T m_i\).*?</code></pre>\s*</div>)'
    attention_img = create_img_tag(illustrations['memory_networks_attention'], 
                                   'Memory Networks Attention Mechanism', '90%')
    match = re.search(attention_pattern, html_content, re.DOTALL)
    if match:
        insert_pos = match.end(1)
        html_content = html_content[:insert_pos] + attention_img + html_content[insert_pos:]
    
    # Add multi-hop reasoning after End-to-End Memory Networks section
    multihop_pattern = r'(# Multi-hop: повторить шаги 3-5.*?</code></pre>\s*</div>)'
    multihop_img = create_img_tag(illustrations['memory_networks_multihop'], 
                                  'Multi-Hop Reasoning', '95%')
    match = re.search(multihop_pattern, html_content, re.DOTALL)
    if match:
        insert_pos = match.end(1)
        html_content = html_content[:insert_pos] + multihop_img + html_content[insert_pos:]
    
    return html_content

def add_illustrations_to_neural_odes(html_content, illustrations):
    """Add illustrations to Neural ODEs cheatsheet."""
    # Add dynamics visualization after main concept explanation
    # Look for section about ODE dynamics
    dynamics_pattern = r'(<h2>🔷.*?ODE.*?</h2>)'
    dynamics_img = create_img_tag(illustrations['neural_odes_dynamics'], 
                                  'Neural ODE vs ResNet Dynamics', '95%')
    match = re.search(dynamics_pattern, html_content, re.DOTALL | re.IGNORECASE)
    if match:
        insert_pos = match.end(1)
        html_content = html_content[:insert_pos] + dynamics_img + html_content[insert_pos:]
    
    # Add trajectories after implementation or training section
    traj_pattern = r'(adjoint.*?backward.*?</code></pre>\s*</div>)'
    traj_img = create_img_tag(illustrations['neural_odes_trajectories'], 
                             'Neural ODE Trajectories in Phase Space', '90%')
    match = re.search(traj_pattern, html_content, re.DOTALL | re.IGNORECASE)
    if match:
        insert_pos = match.end(1)
        html_content = html_content[:insert_pos] + traj_img + html_content[insert_pos:]
    
    return html_content

def add_illustrations_to_normalizing_flows(html_content, illustrations):
    """Add illustrations to Normalizing Flows cheatsheet."""
    # Add transformation visualization early in the document
    transform_pattern = r'(<h2>🔷.*?[Пп]реобразован.*?</h2>)'
    transform_img = create_img_tag(illustrations['normalizing_flows_transformation'], 
                                   'Normalizing Flows Transformations', '95%')
    match = re.search(transform_pattern, html_content, re.DOTALL)
    if match:
        insert_pos = match.end(1)
        html_content = html_content[:insert_pos] + transform_img + html_content[insert_pos:]
    
    # Add density mapping after mathematical formulation
    density_pattern = r'(log.*?det.*?J.*?</code></pre>\s*</div>)'
    density_img = create_img_tag(illustrations['normalizing_flows_density'], 
                                 'Density Transformation', '90%')
    match = re.search(density_pattern, html_content, re.DOTALL)
    if match:
        insert_pos = match.end(1)
        html_content = html_content[:insert_pos] + density_img + html_content[insert_pos:]
    
    return html_content

def add_illustrations_to_hypernetworks(html_content, illustrations):
    """Add illustrations to Hypernetworks cheatsheet."""
    # Add architecture diagram after basic concept
    arch_pattern = r'(<h2>🔷.*?[Аа]рхитектур.*?</h2>)'
    arch_img = create_img_tag(illustrations['hypernetworks_architecture'], 
                              'Hypernetworks Architecture', '95%')
    match = re.search(arch_pattern, html_content, re.DOTALL)
    if match:
        insert_pos = match.end(1)
        html_content = html_content[:insert_pos] + arch_img + html_content[insert_pos:]
    
    # Add comparison diagram
    comp_pattern = r'(<h2>🔷.*?[Пп]реимущества.*?</h2>)'
    comp_img = create_img_tag(illustrations['hypernetworks_comparison'], 
                              'Standard vs Hypernetwork Comparison', '95%')
    match = re.search(comp_pattern, html_content, re.DOTALL)
    if match:
        insert_pos = match.end(1)
        html_content = html_content[:insert_pos] + comp_img + html_content[insert_pos:]
    
    return html_content

def add_illustrations_to_spiking_neural_networks(html_content, illustrations):
    """Add illustrations to Spiking Neural Networks cheatsheet."""
    # Add spike trains visualization early
    spike_pattern = r'(<h2>🔷.*?[Сс]пайк.*?</h2>)'
    spike_img = create_img_tag(illustrations['snn_spike_trains'], 
                               'Spike Trains Visualization', '90%')
    match = re.search(spike_pattern, html_content, re.DOTALL)
    if match:
        insert_pos = match.end(1)
        html_content = html_content[:insert_pos] + spike_img + html_content[insert_pos:]
    
    # Add LIF dynamics after LIF neuron explanation
    lif_pattern = r'(LIF.*?integrate.*?fire.*?</code></pre>\s*</div>)'
    lif_img = create_img_tag(illustrations['snn_lif_dynamics'], 
                            'LIF Neuron Dynamics', '95%')
    match = re.search(lif_pattern, html_content, re.DOTALL | re.IGNORECASE)
    if match:
        insert_pos = match.end(1)
        html_content = html_content[:insert_pos] + lif_img + html_content[insert_pos:]
    
    # Add encoding schemes after coding explanation
    encoding_pattern = r'(<h2>🔷.*?[Кк]одирован.*?</h2>)'
    encoding_img = create_img_tag(illustrations['snn_encoding'], 
                                  'Input Encoding Schemes', '95%')
    match = re.search(encoding_pattern, html_content, re.DOTALL)
    if match:
        insert_pos = match.end(1)
        html_content = html_content[:insert_pos] + encoding_img + html_content[insert_pos:]
    
    return html_content

def process_html_file(filepath, add_illustrations_func, illustrations):
    """Process a single HTML file to add illustrations."""
    print(f"Processing {filepath}...")
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        # Add illustrations
        modified_content = add_illustrations_func(html_content, illustrations)
        
        # Write back
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(modified_content)
        
        print(f"  ✓ Successfully updated {filepath}")
        return True
    except Exception as e:
        print(f"  ✗ Error processing {filepath}: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function to add illustrations to all special architectures cheatsheets."""
    print("=" * 70)
    print("Adding matplotlib illustrations to special architectures cheatsheets")
    print("=" * 70)
    
    # Generate all illustrations
    print("\n1. Generating illustrations...")
    illustrations = generate_all_illustrations()
    
    # Process each HTML file
    print("\n2. Adding illustrations to HTML files...")
    
    files_to_process = [
        ('cheatsheets/memory_networks_cheatsheet.html', add_illustrations_to_memory_networks),
        ('cheatsheets/neural_odes_cheatsheet.html', add_illustrations_to_neural_odes),
        ('cheatsheets/normalizing_flows_cheatsheet.html', add_illustrations_to_normalizing_flows),
        ('cheatsheets/hypernetworks_cheatsheet.html', add_illustrations_to_hypernetworks),
        ('cheatsheets/spiking_neural_networks_cheatsheet.html', add_illustrations_to_spiking_neural_networks),
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
