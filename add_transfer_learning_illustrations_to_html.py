#!/usr/bin/env python3
"""
Add matplotlib illustrations to transfer learning cheatsheet HTML files.
"""

import re
import traceback
from generate_transfer_learning_illustrations import generate_all_illustrations

def create_img_tag(base64_data, alt_text, width="100%"):
    """Create HTML img tag with base64 encoded image."""
    return f'''
    <div style="text-align: center; margin: 10px 0;">
      <img src="data:image/png;base64,{base64_data}" 
           alt="{alt_text}" 
           style="max-width: {width}; height: auto; border-radius: 4px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
    </div>
'''

def add_illustrations_to_transfer_learning(html_content, illustrations):
    """Add illustrations to transfer_learning_cheatsheet.html."""
    
    # Add transfer learning concept after "Что такое Transfer Learning?" section
    concept_pattern = r'(<h2>🔷 1\. Что такое Transfer Learning\?</h2>.*?</div>)'
    concept_img = create_img_tag(illustrations['tl_concept'], 
                                'Концепция Transfer Learning: Source → Target', '95%')
    match = re.search(concept_pattern, html_content, re.DOTALL)
    if match:
        insert_pos = match.end(1)
        html_content = html_content[:insert_pos] + concept_img + html_content[insert_pos:]
    
    # Add types visualization after "Типы Transfer Learning" section
    types_pattern = r'(<h2>🔷 2\. Типы Transfer Learning</h2>.*?</table>\s*</div>)'
    types_img = create_img_tag(illustrations['tl_types'], 
                               'Типы Transfer Learning', '95%')
    match = re.search(types_pattern, html_content, re.DOTALL)
    if match:
        insert_pos = match.end(1)
        html_content = html_content[:insert_pos] + types_img + html_content[insert_pos:]
    
    # Add domain shift after "Domain Adaptation" section
    domain_pattern = r'(<h2>🔷 6\. Domain Adaptation</h2>)'
    domain_img = create_img_tag(illustrations['domain_shift'], 
                                'Domain Shift: различие между source и target', '95%')
    match = re.search(domain_pattern, html_content, re.DOTALL)
    if match:
        insert_pos = match.end(1)
        html_content = html_content[:insert_pos] + domain_img + html_content[insert_pos:]
    
    # Add TrAdaBoost visualization after TrAdaBoost section
    tradaboost_pattern = r'(<h2>🔷 7\. TrAdaBoost</h2>)'
    tradaboost_img = create_img_tag(illustrations['tradaboost'], 
                                    'Процесс TrAdaBoost: эволюция весов', '95%')
    match = re.search(tradaboost_pattern, html_content, re.DOTALL)
    if match:
        insert_pos = match.end(1)
        html_content = html_content[:insert_pos] + tradaboost_img + html_content[insert_pos:]
    
    # Add self-training after self-training section
    self_training_pattern = r'(<h2>🔷 8\. Self-training для Transfer</h2>)'
    self_training_img = create_img_tag(illustrations['self_training'], 
                                       'Self-training: постепенное добавление псевдометок', '95%')
    match = re.search(self_training_pattern, html_content, re.DOTALL)
    if match:
        insert_pos = match.end(1)
        html_content = html_content[:insert_pos] + self_training_img + html_content[insert_pos:]
    
    return html_content

def add_illustrations_to_transfer_learning_cnn(html_content, illustrations):
    """Add illustrations to transfer_learning_cnn_cheatsheet.html."""
    
    # Add CNN architecture visualization after first major section
    # Look for a section about transfer learning or fine-tuning
    patterns_to_try = [
        r'(<h2>🔷 \d+\. .*?[Ff]ine-?tuning.*?</h2>)',
        r'(<h2>🔷 \d+\. .*?[Tt]ransfer.*?</h2>)',
        r'(<div class="block">\s*<h2>🔷 2\.)',  # After second section
    ]
    
    cnn_img = create_img_tag(illustrations['cnn_architecture'], 
                            'CNN Transfer Learning: Feature Extraction vs Fine-tuning', '95%')
    
    for pattern in patterns_to_try:
        match = re.search(pattern, html_content, re.DOTALL | re.IGNORECASE)
        if match:
            insert_pos = match.end(1) if len(match.groups()) > 0 else match.start()
            html_content = html_content[:insert_pos] + cnn_img + html_content[insert_pos:]
            break
    
    # Add fine-tuning strategies
    # Try to find a relevant section
    strategy_patterns = [
        r'(<h2>🔷 \d+\. .*?[Сс]тратеги.*?</h2>)',
        r'(<h2>🔷 \d+\. .*?подход.*?</h2>)',
    ]
    
    strategy_img = create_img_tag(illustrations['finetuning_strategies'], 
                                  'Стратегии fine-tuning', '95%')
    
    for pattern in strategy_patterns:
        match = re.search(pattern, html_content, re.DOTALL | re.IGNORECASE)
        if match:
            insert_pos = match.end(1)
            html_content = html_content[:insert_pos] + strategy_img + html_content[insert_pos:]
            break
    
    # Add learning rate schedule
    lr_patterns = [
        r'(<h2>🔷 \d+\. .*?[Ll]earning [Rr]ate.*?</h2>)',
        r'(<h2>🔷 \d+\. .*?скорост.*?обучени.*?</h2>)',
    ]
    
    lr_img = create_img_tag(illustrations['learning_rate_schedule'], 
                           'Discriminative Learning Rates для разных слоев', '90%')
    
    for pattern in lr_patterns:
        match = re.search(pattern, html_content, re.DOTALL | re.IGNORECASE)
        if match:
            insert_pos = match.end(1)
            html_content = html_content[:insert_pos] + lr_img + html_content[insert_pos:]
            break
    
    return html_content

def add_illustrations_to_transfer_learning_deep(html_content, illustrations):
    """Add illustrations to transfer_learning_deep_cheatsheet.html."""
    
    # Add concept visualization early in the document
    concept_pattern = r'(<h2>🔷 1\. Суть</h2>.*?</div>)'
    concept_img = create_img_tag(illustrations['tl_concept'], 
                                'Transfer Learning: перенос знаний', '95%')
    match = re.search(concept_pattern, html_content, re.DOTALL)
    if match:
        insert_pos = match.end(1)
        html_content = html_content[:insert_pos] + concept_img + html_content[insert_pos:]
    
    # Add architecture visualization after Feature Extraction section
    feature_pattern = r'(<h2>🔷 3\. Feature Extraction</h2>)'
    arch_img = create_img_tag(illustrations['cnn_architecture'], 
                             'Feature Extraction vs Fine-tuning архитектуры', '95%')
    match = re.search(feature_pattern, html_content, re.DOTALL)
    if match:
        insert_pos = match.end(1)
        html_content = html_content[:insert_pos] + arch_img + html_content[insert_pos:]
    
    # Add discriminative LR after that section
    disc_lr_pattern = r'(<h2>🔷 5\. Discriminative Learning Rates</h2>)'
    lr_img = create_img_tag(illustrations['learning_rate_schedule'], 
                           'Разные learning rates для разных слоев', '90%')
    match = re.search(disc_lr_pattern, html_content, re.DOTALL)
    if match:
        insert_pos = match.end(1)
        html_content = html_content[:insert_pos] + lr_img + html_content[insert_pos:]
    
    # Add comparison visualization
    comp_patterns = [
        r'(<h2>🔷 \d+\. .*?[Сс]равнени.*?</h2>)',
        r'(<h2>🔷 \d+\. .*?vs.*?</h2>)',
        r'(<h2>🔷 \d+\. .*?[Пп]реимуществ.*?</h2>)',
    ]
    
    comp_img = create_img_tag(illustrations['transfer_vs_scratch'], 
                             'Transfer Learning vs обучение с нуля', '95%')
    
    for pattern in comp_patterns:
        match = re.search(pattern, html_content, re.DOTALL | re.IGNORECASE)
        if match:
            insert_pos = match.end(1)
            html_content = html_content[:insert_pos] + comp_img + html_content[insert_pos:]
            break
    
    # Add negative transfer illustration
    negative_patterns = [
        r'(<h2>🔷 \d+\. .*?[Nn]egative.*?[Tt]ransfer.*?</h2>)',
        r'(<h2>🔷 \d+\. .*?[Оо]граничени.*?</h2>)',
        r'(<h2>🔷 \d+\. .*?[Кк]огда.*?использовать.*?</h2>)',
    ]
    
    neg_img = create_img_tag(illustrations['negative_transfer'], 
                            'Negative Transfer: когда transfer learning вредит', '95%')
    
    for pattern in negative_patterns:
        match = re.search(pattern, html_content, re.DOTALL | re.IGNORECASE)
        if match:
            insert_pos = match.end(1)
            html_content = html_content[:insert_pos] + neg_img + html_content[insert_pos:]
            break
    
    return html_content

def add_illustrations_to_domain_adaptation(html_content, illustrations):
    """Add illustrations to domain_adaptation_cheatsheet.html."""
    
    # Add domain shift after "Суть" section
    essence_pattern = r'(<h2>🔷 1\. Суть</h2>.*?</div>)'
    shift_img = create_img_tag(illustrations['domain_shift'], 
                               'Domain Shift Problem', '95%')
    match = re.search(essence_pattern, html_content, re.DOTALL)
    if match:
        insert_pos = match.end(1)
        html_content = html_content[:insert_pos] + shift_img + html_content[insert_pos:]
    
    # Add domain adaptation methods comparison
    methods_patterns = [
        r'(<h2>🔷 \d+\. .*?метод.*?</h2>)',
        r'(<h2>🔷 \d+\. .*?подход.*?</h2>)',
        r'(<h2>🔷 2\. Типы задач</h2>.*?</div>)',
    ]
    
    methods_img = create_img_tag(illustrations['da_methods'], 
                                 'Методы Domain Adaptation', '95%')
    
    for pattern in methods_patterns:
        match = re.search(pattern, html_content, re.DOTALL | re.IGNORECASE)
        if match:
            # Use the end of the first captured group if it exists, otherwise use match end
            insert_pos = match.end(1) if match.lastindex and match.lastindex >= 1 else match.end()
            html_content = html_content[:insert_pos] + methods_img + html_content[insert_pos:]
            break
    
    # Add MMD visualization after DANN or MMD section
    mmd_patterns = [
        r'(<h2>🔷 \d+\. .*?MMD.*?</h2>)',
        r'(<h2>🔷 \d+\. .*?Maximum Mean Discrepancy.*?</h2>)',
        r'(<h2>🔷 \d+\. CORAL.*?</h2>)',
    ]
    
    mmd_img = create_img_tag(illustrations['mmd'], 
                            'Maximum Mean Discrepancy (MMD)', '95%')
    
    for pattern in mmd_patterns:
        match = re.search(pattern, html_content, re.DOTALL | re.IGNORECASE)
        if match:
            insert_pos = match.end(1)
            html_content = html_content[:insert_pos] + mmd_img + html_content[insert_pos:]
            break
    
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
        traceback.print_exc()
        return False

def main():
    """Main function to add illustrations to all transfer learning cheatsheets."""
    print("=" * 70)
    print("Adding matplotlib illustrations to transfer learning cheatsheets")
    print("=" * 70)
    
    # Generate all illustrations
    print("\n1. Generating illustrations...")
    illustrations = generate_all_illustrations()
    
    # Process each HTML file
    print("\n2. Adding illustrations to HTML files...")
    
    files_to_process = [
        ('cheatsheets/transfer_learning_cheatsheet.html', add_illustrations_to_transfer_learning),
        ('cheatsheets/transfer_learning_cnn_cheatsheet.html', add_illustrations_to_transfer_learning_cnn),
        ('cheatsheets/transfer_learning_deep_cheatsheet.html', add_illustrations_to_transfer_learning_deep),
        ('cheatsheets/domain_adaptation_cheatsheet.html', add_illustrations_to_domain_adaptation),
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
