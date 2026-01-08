#!/usr/bin/env python3
"""
Add matplotlib illustrations to anomaly detection cheatsheet HTML files.
"""

import re
from generate_anomaly_detection_illustrations import generate_all_illustrations

def create_img_tag(base64_data, alt_text, width="100%"):
    """Create HTML img tag with base64 encoded image."""
    return f'''
    <div style="text-align: center; margin: 10px 0;">
      <img src="data:image/png;base64,{base64_data}" 
           alt="{alt_text}" 
           style="max-width: {width}; height: auto; border-radius: 4px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
    </div>
'''

def add_illustrations_to_isolation_forest(html_content, illustrations):
    """Add illustrations to Isolation Forest cheatsheet."""
    
    # Add basic visualization after the visualization code block (section 5)
    viz_pattern = r'(plt\.show\(\)</code></pre>\s*</div>\s*<div class="block">\s*<h2>🔷 6\.)'
    viz_img = create_img_tag(illustrations['isolation_forest_basic'], 
                            'Isolation Forest обнаружение аномалий', '90%')
    if re.search(viz_pattern, html_content):
        html_content = re.sub(viz_pattern, viz_img + r'\n  <div class="block">\n    <h2>🔷 6.', 
                            html_content, count=1)
    
    # Add scores distribution after scores histogram code
    scores_pattern = r'(plt\.title\(\'Isolation Forest Scores\'\)\s*plt\.show\(\)</code></pre>)'
    scores_img = create_img_tag(illustrations['isolation_forest_scores'], 
                               'Распределение Anomaly Scores', '90%')
    if re.search(scores_pattern, html_content):
        match = re.search(scores_pattern, html_content)
        if match:
            insert_pos = match.end(1)
            html_content = html_content[:insert_pos] + scores_img + html_content[insert_pos:]
    
    # Add contamination comparison after contamination parameter section
    contam_pattern = r'(<h2>🔷 4\. Выбор contamination</h2>)'
    contam_img = create_img_tag(illustrations['isolation_forest_contamination'], 
                               'Влияние параметра contamination', '95%')
    if re.search(contam_pattern, html_content):
        match = re.search(contam_pattern, html_content)
        if match:
            insert_pos = match.end(1)
            html_content = html_content[:insert_pos] + contam_img + html_content[insert_pos:]
    
    return html_content

def add_illustrations_to_lof(html_content, illustrations):
    """Add illustrations to Local Outlier Factor cheatsheet."""
    
    # Add basic visualization after the basic code block
    basic_pattern = r'(print\(f"Найдено аномалий: \{len\(anomalies\)\}"\)</code></pre>\s*</div>)'
    basic_img = create_img_tag(illustrations['lof_basic'], 
                              'LOF обнаружение локальных аномалий', '90%')
    if re.search(basic_pattern, html_content):
        match = re.search(basic_pattern, html_content)
        if match:
            insert_pos = match.end(1)
            html_content = html_content[:insert_pos] + basic_img + html_content[insert_pos:]
    
    # Add LOF scores visualization after the section about scores
    scores_pattern = r'(# Чем меньше \(более отрицательный\), тем более аномальный</code></pre>\s*</div>)'
    scores_img = create_img_tag(illustrations['lof_scores'], 
                               'LOF Scores и их распределение', '95%')
    if re.search(scores_pattern, html_content):
        match = re.search(scores_pattern, html_content)
        if match:
            insert_pos = match.end(1)
            html_content = html_content[:insert_pos] + scores_img + html_content[insert_pos:]
    
    # Add n_neighbors comparison after parameters section
    neighbors_pattern = r'(<h2>🔷.*?Параметры.*?</h2>.*?</table>)'
    neighbors_img = create_img_tag(illustrations['lof_neighbors'], 
                                  'Влияние параметра n_neighbors', '95%')
    if re.search(neighbors_pattern, html_content, re.DOTALL):
        match = re.search(neighbors_pattern, html_content, re.DOTALL)
        if match:
            insert_pos = match.end(1)
            html_content = html_content[:insert_pos] + neighbors_img + html_content[insert_pos:]
    
    return html_content

def add_illustrations_to_one_class_svm(html_content, illustrations):
    """Add illustrations to One-Class SVM cheatsheet."""
    
    # Add basic visualization after the basic code block
    basic_pattern = r'(print\(f"Найдено аномалий: \{len\(anomalies\)\}"\)</code></pre>\s*</div>)'
    basic_img = create_img_tag(illustrations['one_class_svm_basic'], 
                              'One-Class SVM граница решения', '90%')
    if re.search(basic_pattern, html_content):
        match = re.search(basic_pattern, html_content)
        if match:
            insert_pos = match.end(1)
            html_content = html_content[:insert_pos] + basic_img + html_content[insert_pos:]
    
    # Add kernel comparison after parameters section
    kernel_pattern = r'(<h2>🔷 3\. Ключевые параметры</h2>.*?</table>)'
    kernel_img = create_img_tag(illustrations['one_class_svm_kernels'], 
                               'Сравнение типов ядер', '95%')
    if re.search(kernel_pattern, html_content, re.DOTALL):
        match = re.search(kernel_pattern, html_content, re.DOTALL)
        if match:
            insert_pos = match.end(1)
            html_content = html_content[:insert_pos] + kernel_img + html_content[insert_pos:]
    
    # Add nu parameter comparison after kernel or parameters discussion
    nu_pattern = r'(<h2>🔷.*?[Пп]араметр.*?nu.*?</h2>)'
    nu_img = create_img_tag(illustrations['one_class_svm_nu'], 
                          'Влияние параметра nu', '95%')
    if re.search(nu_pattern, html_content, re.DOTALL | re.IGNORECASE):
        match = re.search(nu_pattern, html_content, re.DOTALL | re.IGNORECASE)
        if match:
            insert_pos = match.end(1)
            html_content = html_content[:insert_pos] + nu_img + html_content[insert_pos:]
    
    return html_content

def add_illustrations_to_elliptic_envelope(html_content, illustrations):
    """Add illustrations to Elliptic Envelope cheatsheet."""
    
    # Add basic visualization after the basic code block
    basic_pattern = r'(from sklearn\.covariance import EllipticEnvelope.*?ee\.fit\(X_scaled\).*?</code></pre>\s*</div>)'
    basic_img = create_img_tag(illustrations['elliptic_envelope_basic'], 
                              'Elliptic Envelope обнаружение аномалий', '90%')
    if re.search(basic_pattern, html_content, re.DOTALL):
        match = re.search(basic_pattern, html_content, re.DOTALL)
        if match:
            insert_pos = match.end(1)
            html_content = html_content[:insert_pos] + basic_img + html_content[insert_pos:]
    
    # Add contamination comparison after parameters section
    contam_pattern = r'(<h2>🔷.*?[Пп]араметр.*?contamination.*?</h2>)'
    contam_img = create_img_tag(illustrations['elliptic_envelope_contamination'], 
                               'Влияние параметра contamination', '95%')
    if re.search(contam_pattern, html_content, re.DOTALL | re.IGNORECASE):
        match = re.search(contam_pattern, html_content, re.DOTALL | re.IGNORECASE)
        if match:
            insert_pos = match.end(1)
            html_content = html_content[:insert_pos] + contam_img + html_content[insert_pos:]
    
    # Add Mahalanobis distance visualization after Mahalanobis discussion
    mahal_pattern = r'(<h2>🔷.*?[Мм]ахаланобис.*?</h2>)'
    mahal_img = create_img_tag(illustrations['elliptic_envelope_mahalanobis'], 
                              'Mahalanobis Distance визуализация', '95%')
    if re.search(mahal_pattern, html_content, re.DOTALL | re.IGNORECASE):
        match = re.search(mahal_pattern, html_content, re.DOTALL | re.IGNORECASE)
        if match:
            insert_pos = match.end(1)
            html_content = html_content[:insert_pos] + mahal_img + html_content[insert_pos:]
    
    return html_content

def process_html_file(filepath, add_illustrations_func, illustrations):
    """Process a single HTML file to add illustrations."""
    print(f"Processing {filepath}...")
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        # Add illustrations
        modified_content = add_illustrations_func(html_content, illustrations)
        
        # Check if any changes were made
        if html_content == modified_content:
            print(f"  ⚠ Warning: No modifications made to {filepath}")
        
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
    """Main function to add illustrations to all anomaly detection cheatsheets."""
    print("=" * 70)
    print("Adding matplotlib illustrations to anomaly detection cheatsheets")
    print("=" * 70)
    
    # Generate all illustrations
    print("\n1. Generating illustrations...")
    illustrations = generate_all_illustrations()
    
    # Process each HTML file
    print("\n2. Adding illustrations to HTML files...")
    
    files_to_process = [
        ('cheatsheets/isolation_forest_cheatsheet.html', add_illustrations_to_isolation_forest),
        ('cheatsheets/local_outlier_factor_cheatsheet.html', add_illustrations_to_lof),
        ('cheatsheets/one_class_svm_cheatsheet.html', add_illustrations_to_one_class_svm),
        ('cheatsheets/elliptic_envelope_cheatsheet.html', add_illustrations_to_elliptic_envelope),
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
