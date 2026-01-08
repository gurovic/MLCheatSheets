#!/usr/bin/env python3
"""
Add matplotlib illustrations to meta-learning and few-shot learning cheatsheet HTML files.
"""

import re
from generate_meta_learning_illustrations import generate_all_illustrations

def create_img_tag(base64_data, alt_text, width="100%"):
    """Create HTML img tag with base64 encoded image."""
    return f'''
    <div style="text-align: center; margin: 10px 0;">
      <img src="data:image/png;base64,{base64_data}" 
           alt="{alt_text}" 
           style="max-width: {width}; height: auto; border-radius: 4px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
    </div>
'''

def add_illustrations_to_meta_learning_maml(html_content, illustrations):
    """Add illustrations to meta_learning_maml_cheatsheet.html."""
    
    # Add MAML inner/outer loop illustration after section 3 (MAML algorithm)
    pattern1 = r'(meta_optimizer\.step\(\)</code></pre>\s*</div>)'
    img1 = create_img_tag(illustrations['maml_inner_outer_loop'], 
                         'MAML Inner и Outer Loops', '95%')
    match = re.search(pattern1, html_content, re.DOTALL)
    if match:
        insert_pos = match.end(1)
        html_content = html_content[:insert_pos] + img1 + html_content[insert_pos:]
    
    # Add prototypical network visualization after section 4
    pattern2 = r'(loss = F\.cross_entropy\(logits, query_y\)</code></pre>\s*</div>)'
    img2 = create_img_tag(illustrations['prototypical_network'], 
                         'Prototypical Networks: Классификация по прототипам', '95%')
    match = re.search(pattern2, html_content, re.DOTALL)
    if match:
        insert_pos = match.end(1)
        html_content = html_content[:insert_pos] + img2 + html_content[insert_pos:]
    
    # Add episodic training illustration after section 7
    pattern3 = r'(return support_x, support_y, query_x, query_y</code></pre>\s*</div>)'
    img3 = create_img_tag(illustrations['episodic_training'], 
                         'Episodic Training: N-way K-shot Episodes', '95%')
    match = re.search(pattern3, html_content, re.DOTALL)
    if match:
        insert_pos = match.end(1)
        html_content = html_content[:insert_pos] + img3 + html_content[insert_pos:]
    
    # Add meta-learning performance comparison after section 9 (Meta-training vs Meta-testing)
    pattern4 = r'(print\(f"Meta-test accuracy: \{np\.mean\(test_accuracies\)\}.*?"\)</code></pre>\s*</div>)'
    img4 = create_img_tag(illustrations['meta_learning_performance'], 
                         'Сравнение скорости адаптации: Meta-Learning vs Standard Learning', '90%')
    match = re.search(pattern4, html_content, re.DOTALL)
    if match:
        insert_pos = match.end(1)
        html_content = html_content[:insert_pos] + img4 + html_content[insert_pos:]
    
    return html_content

def add_illustrations_to_few_shot_learning(html_content, illustrations):
    """Add illustrations to few_shot_learning_cheatsheet.html."""
    
    # Add Siamese network architecture after section 2
    pattern1 = r'(return out1, out2</code></pre>\s*</div>)'
    img1 = create_img_tag(illustrations['siamese_network'], 
                         'Siamese Network Architecture', '95%')
    match = re.search(pattern1, html_content, re.DOTALL)
    if match:
        insert_pos = match.end(1)
        html_content = html_content[:insert_pos] + img1 + html_content[insert_pos:]
    
    # Add contrastive/triplet loss illustration after section 4
    pattern2 = r'(loss = criterion\(emb_a, emb_p, emb_n\)</code></pre>\s*</div>)'
    img2 = create_img_tag(illustrations['contrastive_triplet_loss'], 
                         'Contrastive и Triplet Loss', '95%')
    match = re.search(pattern2, html_content, re.DOTALL)
    if match:
        insert_pos = match.end(1)
        html_content = html_content[:insert_pos] + img2 + html_content[insert_pos:]
    
    # Add prototypical network visualization after section 5
    pattern3 = r'(predictions = \(-distances\)\.softmax\(dim=1\)</code></pre>\s*</div>)'
    img3 = create_img_tag(illustrations['prototypical_network'], 
                         'Prototypical Networks: Классификация по прототипам', '90%')
    match = re.search(pattern3, html_content, re.DOTALL)
    if match:
        insert_pos = match.end(1)
        html_content = html_content[:insert_pos] + img3 + html_content[insert_pos:]
    
    # Add few-shot accuracy comparison after section 9 (Evaluation Protocol)
    pattern4 = r'(print\(f"Accuracy: \{np\.mean\(accuracies\)\}.*?"\)</code></pre>\s*</div>)'
    img4 = create_img_tag(illustrations['few_shot_accuracy'], 
                         'Сравнение методов Few-Shot Learning', '90%')
    match = re.search(pattern4, html_content, re.DOTALL)
    if match:
        insert_pos = match.end(1)
        html_content = html_content[:insert_pos] + img4 + html_content[insert_pos:]
    
    return html_content

def add_illustrations_to_maml_meta_learning(html_content, illustrations):
    """Add illustrations to maml_meta_learning_cheatsheet.html."""
    
    # Add N-way K-shot illustration after section 6
    pattern1 = r'(<h2>🔷 6\. N-way K-shot классификация</h2>)'
    img1 = create_img_tag(illustrations['nway_kshot'], 
                         'N-way K-shot Learning Examples', '95%')
    match = re.search(pattern1, html_content, re.DOTALL)
    if match:
        insert_pos = match.end(1)
        html_content = html_content[:insert_pos] + img1 + html_content[insert_pos:]
    
    # Add MAML inner/outer loop illustration after section 3 or 5
    pattern2 = r'(meta_optimizer\.step\(\)</code></pre>\s*</div>)'
    img2 = create_img_tag(illustrations['maml_inner_outer_loop'], 
                         'MAML Inner и Outer Loops', '95%')
    match = re.search(pattern2, html_content, re.DOTALL)
    if match:
        insert_pos = match.end(1)
        html_content = html_content[:insert_pos] + img2 + html_content[insert_pos:]
    
    # Add episodic training illustration after section 10 (MAML for RL) or near end
    pattern3 = r'(<h2>🔷 12\. Практические рекомендации</h2>)'
    img3 = create_img_tag(illustrations['episodic_training'], 
                         'Episodic Training: N-way K-shot Episodes', '95%')
    match = re.search(pattern3, html_content, re.DOTALL)
    if match:
        insert_pos = match.end(1)
        html_content = html_content[:insert_pos] + img3 + html_content[insert_pos:]
    
    # Add meta-learning performance comparison after section 14 (comparison with other approaches)
    pattern4 = r'(</table>\s*</div>\s*<div class="block">\s*<h2>🔷 15\. Современные направления</h2>)'
    img4 = create_img_tag(illustrations['meta_learning_performance'], 
                         'Сравнение скорости адаптации: Meta-Learning vs Standard Learning', '90%')
    match = re.search(pattern4, html_content, re.DOTALL)
    if match:
        insert_pos = match.end(0) - len('<div class="block">\n    <h2>🔷 15. Современные направления</h2>')
        html_content = html_content[:insert_pos] + img4 + html_content[insert_pos:]
    
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
        return False

def main():
    """Main function to add illustrations to all meta-learning and few-shot cheatsheets."""
    print("=" * 70)
    print("Adding matplotlib illustrations to meta-learning cheatsheets")
    print("=" * 70)
    
    # Generate all illustrations
    print("\n1. Generating illustrations...")
    illustrations = generate_all_illustrations()
    
    # Process each HTML file
    print("\n2. Adding illustrations to HTML files...")
    
    files_to_process = [
        ('cheatsheets/meta_learning_maml_cheatsheet.html', add_illustrations_to_meta_learning_maml),
        ('cheatsheets/few_shot_learning_cheatsheet.html', add_illustrations_to_few_shot_learning),
        ('cheatsheets/maml_meta_learning_cheatsheet.html', add_illustrations_to_maml_meta_learning),
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
