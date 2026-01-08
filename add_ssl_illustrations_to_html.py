#!/usr/bin/env python3
"""
Add matplotlib illustrations to Self-Supervised and Semi-Supervised Learning cheatsheet HTML files.
"""

import re
from generate_ssl_illustrations import generate_all_illustrations

def create_img_tag(base64_data, alt_text, width="100%"):
    """Create HTML img tag with base64 encoded image."""
    return f'''
    <div style="text-align: center; margin: 10px 0;">
      <img src="data:image/png;base64,{base64_data}" 
           alt="{alt_text}" 
           style="max-width: {width}; height: auto; border-radius: 4px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
    </div>
'''

def add_illustrations_to_self_supervised(html_content, illustrations):
    """Add illustrations to self-supervised learning cheatsheet."""
    
    # Add rotation prediction illustration after rotation prediction code
    rotation_pattern = r'(print\(f\'Epoch \{epoch\}, Loss: \{loss\.item\(\):.4f\}\'\)</code></pre>\s*</div>)'
    rotation_img = create_img_tag(illustrations['rotation_prediction'], 
                                  'Rotation Prediction: Pretext Task', '85%')
    match = re.search(rotation_pattern, html_content, re.DOTALL)
    if match:
        insert_pos = match.end(1)
        html_content = html_content[:insert_pos] + rotation_img + html_content[insert_pos:]
        print("  ✓ Added rotation prediction illustration")
    
    # Add jigsaw puzzles illustration
    jigsaw_pattern = r'(# Обучить классификатор восстанавливать порядок.*?</code></pre>\s*</div>)'
    jigsaw_img = create_img_tag(illustrations['jigsaw_puzzles'], 
                                'Jigsaw Puzzles: Pretext Task', '85%')
    match = re.search(jigsaw_pattern, html_content, re.DOTALL)
    if match:
        insert_pos = match.end(1)
        html_content = html_content[:insert_pos] + jigsaw_img + html_content[insert_pos:]
        print("  ✓ Added jigsaw puzzles illustration")
    
    # Add colorization illustration
    colorization_pattern = r'(loss\.backward\(\)\s+optimizer\.step\(\)</code></pre>\s*</div>\s*<div class="block">\s*<h2>🔷 6\.)'
    colorization_img = create_img_tag(illustrations['colorization'], 
                                      'Colorization: Pretext Task', '85%')
    match = re.search(colorization_pattern, html_content, re.DOTALL)
    if match:
        insert_pos = match.end(1)
        # Insert before the next block
        insert_text = colorization_img + '\n  <div class="block">\n    <h2>🔷 6.'
        html_content = html_content[:match.start(1)] + match.group(1)[:match.group(1).rfind('</div>')+6] + colorization_img + html_content[match.end(1):]
        print("  ✓ Added colorization illustration")
    
    # Add MAE illustration after MAE section
    mae_pattern = r'(<h2>🔷 7\. Masked Autoencoder \(MAE\)</h2>)'
    mae_img = create_img_tag(illustrations['mae_masking'], 
                            'Masked Autoencoder (MAE)', '90%')
    match = re.search(mae_pattern, html_content, re.DOTALL)
    if match:
        insert_pos = match.end(1)
        html_content = html_content[:insert_pos] + mae_img + html_content[insert_pos:]
        print("  ✓ Added MAE illustration")
    
    # Add comparison chart after comparison section header
    comparison_pattern = r'(<h2>🔷 8\. Сравнение методов</h2>)'
    comparison_img = create_img_tag(illustrations['ssl_comparison'], 
                                    'Self-Supervised Methods Comparison', '90%')
    match = re.search(comparison_pattern, html_content, re.DOTALL)
    if match:
        insert_pos = match.end(1)
        html_content = html_content[:insert_pos] + comparison_img + html_content[insert_pos:]
        print("  ✓ Added SSL comparison illustration")
    
    return html_content

def add_illustrations_to_semi_supervised(html_content, illustrations):
    """Add illustrations to semi-supervised learning cheatsheet."""
    
    # Add self-training process after self-training code
    self_training_pattern = r'(return model</code></pre>\s*</div>)'
    self_training_img = create_img_tag(illustrations['self_training'], 
                                       'Self-Training Process', '95%')
    match = re.search(self_training_pattern, html_content, re.DOTALL)
    if match:
        insert_pos = match.end(1)
        html_content = html_content[:insert_pos] + self_training_img + html_content[insert_pos:]
        print("  ✓ Added self-training illustration")
    
    # Add label propagation illustration
    label_prop_pattern = r'(<h2>🔷 4\. Label Propagation</h2>)'
    label_prop_img = create_img_tag(illustrations['label_propagation'], 
                                    'Label Propagation', '90%')
    match = re.search(label_prop_pattern, html_content, re.DOTALL)
    if match:
        insert_pos = match.end(1)
        html_content = html_content[:insert_pos] + label_prop_img + html_content[insert_pos:]
        print("  ✓ Added label propagation illustration")
    
    # Add performance comparison after "When to use SSL" section
    performance_pattern = r'(<h2>🔷 7\. Когда использовать SSL</h2>)'
    performance_img = create_img_tag(illustrations['ssl_performance'], 
                                     'Semi-Supervised vs Supervised Performance', '90%')
    match = re.search(performance_pattern, html_content, re.DOTALL)
    if match:
        insert_pos = match.end(1)
        html_content = html_content[:insert_pos] + performance_img + html_content[insert_pos:]
        print("  ✓ Added SSL performance illustration")
    
    return html_content

def add_illustrations_to_contrastive(html_content, illustrations):
    """Add illustrations to contrastive learning cheatsheet."""
    
    # Add contrastive pairs after principle section
    pairs_pattern = r'(<h2>🔷 2\. Принцип работы</h2>)'
    pairs_img = create_img_tag(illustrations['contrastive_pairs'], 
                               'Positive vs Negative Pairs', '90%')
    match = re.search(pairs_pattern, html_content, re.DOTALL)
    if match:
        insert_pos = match.end(1)
        html_content = html_content[:insert_pos] + pairs_img + html_content[insert_pos:]
        print("  ✓ Added contrastive pairs illustration")
    
    # Add SimCLR architecture after SimCLR augmentation code
    simclr_pattern = r'(transforms\.Normalize\(\s+mean=\[0\.485.*?\]\s+\)\s+\]\)</code></pre>\s*</div>)'
    simclr_img = create_img_tag(illustrations['simclr_architecture'], 
                                'SimCLR Architecture', '95%')
    match = re.search(simclr_pattern, html_content, re.DOTALL)
    if match:
        insert_pos = match.end(1)
        html_content = html_content[:insert_pos] + simclr_img + html_content[insert_pos:]
        print("  ✓ Added SimCLR architecture illustration")
    
    # Add NT-Xent loss visualization after NT-Xent loss code
    ntxent_pattern = r'(<h2>🔷 4\. NT-Xent Loss</h2>)'
    ntxent_img = create_img_tag(illustrations['ntxent_loss'], 
                                'NT-Xent Loss: Temperature Effect', '95%')
    match = re.search(ntxent_pattern, html_content, re.DOTALL)
    if match:
        insert_pos = match.end(1)
        html_content = html_content[:insert_pos] + ntxent_img + html_content[insert_pos:]
        print("  ✓ Added NT-Xent loss illustration")
    
    # Add embedding space visualization after comparison section
    embedding_pattern = r'(<h2>🔷 8\. Сравнение методов</h2>)'
    embedding_img = create_img_tag(illustrations['embedding_space'], 
                                   'Effect of Contrastive Learning on Embeddings', '90%')
    match = re.search(embedding_pattern, html_content, re.DOTALL)
    if match:
        insert_pos = match.end(1)
        html_content = html_content[:insert_pos] + embedding_img + html_content[insert_pos:]
        print("  ✓ Added embedding space illustration")
    
    # Add methods comparison
    methods_pattern = r'(<h2>🔷 8\. Сравнение методов</h2>)'
    methods_img = create_img_tag(illustrations['contrastive_comparison'], 
                                 'Contrastive Learning Methods Comparison', '85%')
    # This should go after embedding space, so we search from a later position
    # Let's add it after the embedding illustration
    comparison_section = re.search(r'(<h2>🔷 8\. Сравнение методов</h2>.*?)<ul>', 
                                   html_content, re.DOTALL)
    if comparison_section:
        insert_pos = comparison_section.end(1)
        html_content = html_content[:insert_pos] + methods_img + html_content[insert_pos:]
        print("  ✓ Added contrastive methods comparison")
    
    return html_content

def process_html_file(filepath, add_illustrations_func, illustrations):
    """Process a single HTML file to add illustrations."""
    print(f"\nProcessing {filepath}...")
    
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
    """Main function to add illustrations to all SSL cheatsheets."""
    print("=" * 80)
    print("Adding matplotlib illustrations to Self/Semi-Supervised Learning cheatsheets")
    print("=" * 80)
    
    # Generate all illustrations
    print("\n1. Generating illustrations...")
    illustrations = generate_all_illustrations()
    
    # Process each HTML file
    print("\n2. Adding illustrations to HTML files...")
    
    files_to_process = [
        ('cheatsheets/self_supervised_learning_cheatsheet.html', add_illustrations_to_self_supervised),
        ('cheatsheets/semi_supervised_learning_cheatsheet.html', add_illustrations_to_semi_supervised),
        ('cheatsheets/contrastive_learning_cheatsheet.html', add_illustrations_to_contrastive),
    ]
    
    success_count = 0
    for filepath, add_func in files_to_process:
        if process_html_file(filepath, add_func, illustrations):
            success_count += 1
    
    print("\n" + "=" * 80)
    print(f"Completed: {success_count}/{len(files_to_process)} files successfully updated")
    print("=" * 80)

if __name__ == '__main__':
    main()
