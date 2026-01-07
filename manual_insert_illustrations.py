#!/usr/bin/env python3
"""
Manual insertion of illustrations into the remaining cheatsheet files.
"""

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

def add_to_memory_networks(html_content, illustrations):
    """Add illustrations to memory networks cheatsheet."""
    # Add attention illustration after section 3
    insertion_point = html_content.find('7. Ответ: answer = softmax(W(u_final))</code></pre>\n  </div>')
    if insertion_point != -1:
        # Find the end of the div
        end_div = html_content.find('</div>', insertion_point)
        if end_div != -1:
            attention_img = create_img_tag(
                illustrations['memory_networks_attention'],
                'Memory Networks Attention Mechanism',
                '90%'
            )
            html_content = html_content[:end_div] + attention_img + html_content[end_div:]
    
    # Add multi-hop illustration after section 3 or beginning of section 4
    insertion_point = html_content.find('<h2>🔷 4. Практическая реализация</h2>')
    if insertion_point != -1:
        multihop_img = create_img_tag(
            illustrations['memory_networks_multihop'],
            'Multi-Hop Reasoning',
            '95%'
        )
        # Insert after the h2 tag
        next_line = html_content.find('\n', insertion_point)
        if next_line != -1:
            html_content = html_content[:next_line] + '\n' + multihop_img + html_content[next_line:]
    
    return html_content

def add_to_normalizing_flows(html_content, illustrations):
    """Add illustrations to normalizing flows cheatsheet."""
    # Add transformation visualization after section 2 or 3
    insertion_point = html_content.find('<h2>🔷 3. Обратный процесс (Reverse)</h2>')
    if insertion_point != -1:
        transform_img = create_img_tag(
            illustrations['normalizing_flows_transformation'],
            'Normalizing Flows Transformations',
            '95%'
        )
        # Insert after the h2 tag
        next_line = html_content.find('\n', insertion_point)
        if next_line != -1:
            html_content = html_content[:next_line] + '\n' + transform_img + html_content[next_line:]
    
    # Add density mapping after section 5
    insertion_point = html_content.find('Минимизировать MSE</li></ol>\n  </div>')
    if insertion_point != -1:
        end_div = html_content.find('</div>', insertion_point)
        if end_div != -1:
            density_img = create_img_tag(
                illustrations['normalizing_flows_density'],
                'Density Transformation',
                '90%'
            )
            html_content = html_content[:end_div] + density_img + html_content[end_div:]
    
    return html_content

def add_to_hypernetworks(html_content, illustrations):
    """Add illustrations to hypernetworks cheatsheet."""
    # Check if this file actually contains hypernetworks content
    if 'Hypernetworks' not in html_content and 'hypernetwork' not in html_content.lower():
        print("WARNING: This file does not appear to contain Hypernetworks content!")
        return html_content
    
    # Add architecture diagram
    insertion_point = html_content.find('<h2>🔷 2.')
    if insertion_point != -1:
        arch_img = create_img_tag(
            illustrations['hypernetworks_architecture'],
            'Hypernetworks Architecture',
            '95%'
        )
        next_line = html_content.find('\n', insertion_point)
        if next_line != -1:
            html_content = html_content[:next_line] + '\n' + arch_img + html_content[next_line:]
    
    # Add comparison diagram
    insertion_point = html_content.find('<h2>🔷 3.')
    if insertion_point != -1:
        comp_img = create_img_tag(
            illustrations['hypernetworks_comparison'],
            'Standard vs Hypernetwork Comparison',
            '95%'
        )
        next_line = html_content.find('\n', insertion_point)
        if next_line != -1:
            html_content = html_content[:next_line] + '\n' + comp_img + html_content[next_line:]
    
    return html_content

def main():
    """Main function to add illustrations to remaining files."""
    print("="  * 70)
    print("Manual insertion of illustrations to remaining cheatsheets")
    print("=" * 70)
    
    # Generate illustrations
    print("\n1. Generating illustrations...")
    illustrations = generate_all_illustrations()
    
    # Process each file
    print("\n2. Processing files...")
    
    files = [
        ('cheatsheets/memory_networks_cheatsheet.html', add_to_memory_networks),
        ('cheatsheets/normalizing_flows_cheatsheet.html', add_to_normalizing_flows),
        ('cheatsheets/hypernetworks_cheatsheet.html', add_to_hypernetworks),
    ]
    
    for filepath, add_func in files:
        print(f"\nProcessing {filepath}...")
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            modified = add_func(content, illustrations)
            
            if modified != content:
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(modified)
                print(f"  ✓ Successfully updated {filepath}")
            else:
                print(f"  ! No changes made to {filepath}")
        except Exception as e:
            print(f"  ✗ Error: {e}")
    
    print("\n" + "=" * 70)
    print("Completed")
    print("=" * 70)

if __name__ == '__main__':
    main()
