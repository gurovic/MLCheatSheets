#!/usr/bin/env python3
"""
Add matplotlib illustrations to graphical models cheatsheet HTML files.
"""

import re
from generate_graphical_models_illustrations import generate_all_illustrations

def create_img_tag(base64_data, alt_text, width="100%"):
    """Create HTML img tag with base64 encoded image."""
    return f'''
    <div style="text-align: center; margin: 10px 0;">
      <img src="data:image/png;base64,{base64_data}" 
           alt="{alt_text}" 
           style="max-width: {width}; height: auto; border-radius: 4px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
    </div>
'''

def add_illustrations_to_graphical_models(html_content, illustrations):
    """Add illustrations to graphical models cheatsheet."""
    
    # Add Bayesian network after Bayesian networks section
    bayesian_pattern = r'(<h2>🔷 2\. Байесовские сети</h2>.*?model\.check_model\(\)</code></pre>\s*</div>)'
    bayesian_img = create_img_tag(illustrations['bayesian_network'], 
                                  'Байесовская сеть: пример с дождём', '90%')
    match = re.search(bayesian_pattern, html_content, re.DOTALL)
    if match:
        insert_pos = match.end(1)
        html_content = html_content[:insert_pos] + bayesian_img + html_content[insert_pos:]
    
    # Add Markov network after Markov networks section
    markov_pattern = r'(<h2>🔷 3\. Марковские сети</h2>.*?model\.add_factors.*?</code></pre>\s*</div>)'
    markov_img = create_img_tag(illustrations['markov_network'], 
                                'Марковская сеть (ненаправленный граф)', '90%')
    match = re.search(markov_pattern, html_content, re.DOTALL)
    if match:
        insert_pos = match.end(1)
        html_content = html_content[:insert_pos] + markov_img + html_content[insert_pos:]
    
    # Add d-separation after conditional independence section
    dsep_pattern = r'(<h2>🔷 4\. Условная независимость</h2>.*?</table>)'
    dsep_img = create_img_tag(illustrations['d_separation'], 
                              'Примеры условной независимости', '95%')
    match = re.search(dsep_pattern, html_content, re.DOTALL)
    if match:
        insert_pos = match.end(1)
        html_content = html_content[:insert_pos] + dsep_img + html_content[insert_pos:]
    
    return html_content

def add_illustrations_to_hmm(html_content, illustrations):
    """Add illustrations to HMM cheatsheet."""
    
    # Add HMM structure after basic components section
    structure_pattern = r'(<h2>🔷 2\. Основные компоненты</h2>.*?</blockquote>\s*</div>)'
    structure_img = create_img_tag(illustrations['hmm_structure'], 
                                   'Структура скрытой марковской модели', '95%')
    match = re.search(structure_pattern, html_content, re.DOTALL)
    if match:
        insert_pos = match.end(1)
        html_content = html_content[:insert_pos] + structure_img + html_content[insert_pos:]
    
    # Add matrices after basic code section
    matrices_pattern = r'(<h2>🔷 4\. Базовый код на Python</h2>.*?model\.score\(X\)</code></pre>\s*</div>)'
    matrices_img = create_img_tag(illustrations['hmm_matrices'], 
                                  'Матрицы переходов и эмиссий HMM', '95%')
    match = re.search(matrices_pattern, html_content, re.DOTALL)
    if match:
        insert_pos = match.end(1)
        html_content = html_content[:insert_pos] + matrices_img + html_content[insert_pos:]
    
    # Add Viterbi visualization after Viterbi algorithm section
    viterbi_pattern = r'(<h2>🔷 6\. Viterbi алгоритм</h2>.*?return best_path</code></pre>\s*</div>)'
    viterbi_img = create_img_tag(illustrations['viterbi'], 
                                 'Визуализация алгоритма Витерби', '95%')
    match = re.search(viterbi_pattern, html_content, re.DOTALL)
    if match:
        insert_pos = match.end(1)
        html_content = html_content[:insert_pos] + viterbi_img + html_content[insert_pos:]
    
    # Add Forward-Backward after Baum-Welch section
    fb_pattern = r'(<h2>🔷 7\. Baum-Welch алгоритм</h2>.*?</blockquote>\s*</div>)'
    fb_img = create_img_tag(illustrations['forward_backward'], 
                           'Forward-Backward алгоритм', '95%')
    match = re.search(fb_pattern, html_content, re.DOTALL)
    if match:
        insert_pos = match.end(1)
        html_content = html_content[:insert_pos] + fb_img + html_content[insert_pos:]
    
    return html_content

def add_illustrations_to_crf(html_content, illustrations):
    """Add illustrations to CRF cheatsheet."""
    
    # Add CRF structure after the main formula section
    structure_pattern = r'(<h2>🔷 2\. Формула</h2>.*?</ul>\s*</div>)'
    structure_img = create_img_tag(illustrations['crf_structure'], 
                                   'Структура условного случайного поля', '95%')
    match = re.search(structure_pattern, html_content, re.DOTALL)
    if match:
        insert_pos = match.end(1)
        html_content = html_content[:insert_pos] + structure_img + html_content[insert_pos:]
    
    # Add feature functions visualization after features section
    features_pattern = r'(<h2>🔷 4\. Признаки</h2>.*?}</code></pre>\s*</div>)'
    features_img = create_img_tag(illustrations['crf_features'], 
                                  'Примеры признаковых функций в CRF', '90%')
    match = re.search(features_pattern, html_content, re.DOTALL)
    if match:
        insert_pos = match.end(1)
        html_content = html_content[:insert_pos] + features_img + html_content[insert_pos:]
    
    # Add HMM vs CRF comparison after comparison section
    comparison_pattern = r'(<h2>🔷 7\. CRF vs HMM</h2>.*?</table>)'
    comparison_img = create_img_tag(illustrations['hmm_vs_crf'], 
                                    'Сравнение HMM и CRF', '95%')
    match = re.search(comparison_pattern, html_content, re.DOTALL)
    if match:
        insert_pos = match.end(1)
        html_content = html_content[:insert_pos] + comparison_img + html_content[insert_pos:]
    
    return html_content

def main():
    """Main function to add illustrations to all graphical models cheatsheets."""
    print("Generating illustrations...")
    illustrations = generate_all_illustrations()
    
    # Process graphical_models_cheatsheet.html
    print("\nProcessing graphical_models_cheatsheet.html...")
    with open('cheatsheets/graphical_models_cheatsheet.html', 'r', encoding='utf-8') as f:
        html_content = f.read()
    
    html_content = add_illustrations_to_graphical_models(html_content, illustrations)
    
    with open('cheatsheets/graphical_models_cheatsheet.html', 'w', encoding='utf-8') as f:
        f.write(html_content)
    print("✓ Illustrations added to graphical_models_cheatsheet.html")
    
    # Process hmm_cheatsheet.html
    print("\nProcessing hmm_cheatsheet.html...")
    with open('cheatsheets/hmm_cheatsheet.html', 'r', encoding='utf-8') as f:
        html_content = f.read()
    
    html_content = add_illustrations_to_hmm(html_content, illustrations)
    
    with open('cheatsheets/hmm_cheatsheet.html', 'w', encoding='utf-8') as f:
        f.write(html_content)
    print("✓ Illustrations added to hmm_cheatsheet.html")
    
    # Process crf_conditional_random_fields_cheatsheet.html
    print("\nProcessing crf_conditional_random_fields_cheatsheet.html...")
    with open('cheatsheets/crf_conditional_random_fields_cheatsheet.html', 'r', encoding='utf-8') as f:
        html_content = f.read()
    
    html_content = add_illustrations_to_crf(html_content, illustrations)
    
    with open('cheatsheets/crf_conditional_random_fields_cheatsheet.html', 'w', encoding='utf-8') as f:
        f.write(html_content)
    print("✓ Illustrations added to crf_conditional_random_fields_cheatsheet.html")
    
    print("\n✅ All illustrations added successfully!")

if __name__ == '__main__':
    main()
