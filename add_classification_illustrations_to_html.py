#!/usr/bin/env python3
"""
Add matplotlib illustrations to classification cheatsheet HTML files.
"""

import re
from generate_classification_illustrations import generate_all_illustrations

def create_img_tag(base64_data, alt_text, width="100%"):
    """Create HTML img tag with base64 encoded image."""
    return f'''
    <div style="text-align: center; margin: 10px 0;">
      <img src="data:image/png;base64,{base64_data}" 
           alt="{alt_text}" 
           style="max-width: {width}; height: auto; border-radius: 4px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
    </div>
'''

def add_illustrations_to_decision_trees(html_content, illustrations):
    """Add illustrations to decision trees cheatsheet."""
    # Add tree structure visualization after visualization section
    viz_pattern = r'(plot_tree.*?plt\.show\(\)</code></pre>\s*</div>)'
    viz_img = create_img_tag(illustrations['dt_structure'], 'Структура дерева решений', '95%')
    match = re.search(viz_pattern, html_content, re.DOTALL)
    if match:
        insert_pos = match.end(1)
        html_content = html_content[:insert_pos] + viz_img + html_content[insert_pos:]
    
    # Add depth comparison after overfitting section or parameters section
    depth_pattern = r'(<h2>🔷.*?Борьба с переобучением.*?</h2>)'
    depth_img = create_img_tag(illustrations['dt_depth_comparison'], 
                               'Влияние глубины дерева на переобучение', '95%')
    match = re.search(depth_pattern, html_content, re.DOTALL)
    if match:
        insert_pos = match.end(1)
        html_content = html_content[:insert_pos] + depth_img + html_content[insert_pos:]
    
    # Add feature importance after importance section
    importance_pattern = r'(feature_importances_.*?</code></pre>\s*</div>)'
    importance_img = create_img_tag(illustrations['dt_feature_importance'], 
                                   'Важность признаков', '90%')
    match = re.search(importance_pattern, html_content, re.DOTALL)
    if match:
        insert_pos = match.end(1)
        html_content = html_content[:insert_pos] + importance_img + html_content[insert_pos:]
    
    return html_content

def add_illustrations_to_random_forest(html_content, illustrations):
    """Add illustrations to random forest cheatsheet."""
    # Add ensemble visualization after basic code section
    ensemble_pattern = r'(<h2>🔷.*?Пример.*?</h2>|<h2>🔷.*?Базовый код.*?</h2>.*?</div>)'
    ensemble_img = create_img_tag(illustrations['rf_ensemble'], 
                                  'Random Forest как ансамбль деревьев', '95%')
    match = re.search(ensemble_pattern, html_content, re.DOTALL)
    if match:
        insert_pos = match.end(1)
        html_content = html_content[:insert_pos] + ensemble_img + html_content[insert_pos:]
    
    # Add OOB error plot
    oob_pattern = r'(<h2>🔷.*?OOB.*?</h2>|<h2>🔷.*?out-of-bag.*?</h2>)'
    oob_img = create_img_tag(illustrations['rf_oob_error'], 
                            'Out-of-Bag ошибка', '90%')
    match = re.search(oob_pattern, html_content, re.DOTALL | re.IGNORECASE)
    if match:
        insert_pos = match.end(1)
        html_content = html_content[:insert_pos] + oob_img + html_content[insert_pos:]
    
    # Add feature importance comparison
    importance_pattern = r'(feature_importances_.*?</code></pre>\s*</div>)'
    importance_img = create_img_tag(illustrations['rf_feature_importance'], 
                                   'Сравнение важности признаков', '95%')
    match = re.search(importance_pattern, html_content, re.DOTALL)
    if match:
        insert_pos = match.end(1)
        html_content = html_content[:insert_pos] + importance_img + html_content[insert_pos:]
    
    return html_content

def add_illustrations_to_svm(html_content, illustrations):
    """Add illustrations to SVM cheatsheet."""
    # Add kernels comparison after kernels section
    kernels_pattern = r'(<h2>🔷.*?[Яя]дра.*?</h2>|<h2>🔷.*?Kernel.*?</h2>)'
    kernels_img = create_img_tag(illustrations['svm_kernels'], 
                                 'Сравнение различных ядер SVM', '95%')
    match = re.search(kernels_pattern, html_content, re.DOTALL | re.IGNORECASE)
    if match:
        insert_pos = match.end(1)
        html_content = html_content[:insert_pos] + kernels_img + html_content[insert_pos:]
    
    # Add margin visualization
    margin_pattern = r'(<h2>🔷.*?[Мм]арж.*?</h2>|<h2>🔷.*?Margin.*?</h2>|<h2>🔷.*?Суть.*?</h2>)'
    margin_img = create_img_tag(illustrations['svm_margin'], 
                                'Разделяющая гиперплоскость и margin', '90%')
    match = re.search(margin_pattern, html_content, re.DOTALL | re.IGNORECASE)
    if match:
        insert_pos = match.end(1)
        html_content = html_content[:insert_pos] + margin_img + html_content[insert_pos:]
    
    # Add C parameter comparison
    c_pattern = r'(<h2>🔷.*?[Пп]араметр.*?C.*?</h2>|<h2>🔷.*?[Рр]егуляризаци.*?</h2>)'
    c_img = create_img_tag(illustrations['svm_c_parameter'], 
                          'Влияние параметра C', '95%')
    match = re.search(c_pattern, html_content, re.DOTALL | re.IGNORECASE)
    if match:
        insert_pos = match.end(1)
        html_content = html_content[:insert_pos] + c_img + html_content[insert_pos:]
    
    return html_content

def add_illustrations_to_logreg(html_content, illustrations):
    """Add illustrations to logistic regression cheatsheet."""
    # Add sigmoid function after basic section
    sigmoid_pattern = r'(<h2>🔷.*?Суть.*?</h2>|<h2>🔷.*?Базовый код.*?</h2>)'
    sigmoid_img = create_img_tag(illustrations['logreg_sigmoid'], 
                                 'Сигмоидная функция активации', '90%')
    match = re.search(sigmoid_pattern, html_content, re.DOTALL)
    if match:
        insert_pos = match.end(1)
        html_content = html_content[:insert_pos] + sigmoid_img + html_content[insert_pos:]
    
    # Add decision boundary with probabilities
    boundary_pattern = r'(predict_proba.*?</code></pre>\s*</div>|<h2>🔷.*?Пример.*?</h2>)'
    boundary_img = create_img_tag(illustrations['logreg_boundary'], 
                                  'Вероятностные контуры и граница решения', '95%')
    match = re.search(boundary_pattern, html_content, re.DOTALL)
    if match:
        insert_pos = match.end(1)
        html_content = html_content[:insert_pos] + boundary_img + html_content[insert_pos:]
    
    # Add regularization comparison
    reg_pattern = r'(<h2>🔷.*?[Рр]егуляризаци.*?</h2>)'
    reg_img = create_img_tag(illustrations['logreg_regularization'], 
                            'Влияние регуляризации', '95%')
    match = re.search(reg_pattern, html_content, re.DOTALL | re.IGNORECASE)
    if match:
        insert_pos = match.end(1)
        html_content = html_content[:insert_pos] + reg_img + html_content[insert_pos:]
    
    return html_content

def add_illustrations_to_knn(html_content, illustrations):
    """Add illustrations to KNN cheatsheet."""
    # Add KNN illustration showing how it works
    sutt_pattern = r'(<h2>🔷.*?Суть.*?</h2>)'
    sutt_img = create_img_tag(illustrations['knn_illustration'], 
                             'Принцип работы KNN', '90%')
    match = re.search(sutt_pattern, html_content, re.DOTALL)
    if match:
        insert_pos = match.end(1)
        html_content = html_content[:insert_pos] + sutt_img + html_content[insert_pos:]
    
    # Add decision boundary comparison for different K values
    k_pattern = r'(<h2>🔷.*?[Вв]ыбор.*?K.*?</h2>|<h2>🔷.*?[Пп]араметр.*?K.*?</h2>)'
    k_img = create_img_tag(illustrations['knn_decision_boundary'], 
                          'Влияние параметра K на границу решения', '95%')
    match = re.search(k_pattern, html_content, re.DOTALL | re.IGNORECASE)
    if match:
        insert_pos = match.end(1)
        html_content = html_content[:insert_pos] + k_img + html_content[insert_pos:]
    
    # Add distance metrics comparison
    dist_pattern = r'(<h2>🔷.*?[Мм]етрик.*?расстояни.*?</h2>|<h2>🔷.*?Distance.*?</h2>)'
    dist_img = create_img_tag(illustrations['knn_distance_metrics'], 
                             'Сравнение метрик расстояния', '95%')
    match = re.search(dist_pattern, html_content, re.DOTALL | re.IGNORECASE)
    if match:
        insert_pos = match.end(1)
        html_content = html_content[:insert_pos] + dist_img + html_content[insert_pos:]
    
    return html_content

def add_illustrations_to_naive_bayes(html_content, illustrations):
    """Add illustrations to Naive Bayes cheatsheet."""
    # Add distributions visualization
    dist_pattern = r'(<h2>🔷.*?Суть.*?</h2>|<h2>🔷.*?[Вв]ероятност.*?</h2>)'
    dist_img = create_img_tag(illustrations['nb_distributions'], 
                             'Гауссовские распределения признаков', '95%')
    match = re.search(dist_pattern, html_content, re.DOTALL | re.IGNORECASE)
    if match:
        insert_pos = match.end(1)
        html_content = html_content[:insert_pos] + dist_img + html_content[insert_pos:]
    
    # Add decision boundary
    boundary_pattern = r'(predict_proba.*?</code></pre>\s*</div>|<h2>🔷.*?Пример.*?</h2>)'
    boundary_img = create_img_tag(illustrations['nb_boundary'], 
                                  'Граница решения Naive Bayes', '90%')
    match = re.search(boundary_pattern, html_content, re.DOTALL)
    if match:
        insert_pos = match.end(1)
        html_content = html_content[:insert_pos] + boundary_img + html_content[insert_pos:]
    
    # Add comparison with other classifiers
    comp_pattern = r'(<h2>🔷.*?Когда использовать.*?</h2>|<h2>🔷.*?[Пп]реимуществ.*?</h2>)'
    comp_img = create_img_tag(illustrations['nb_comparison'], 
                             'Сравнение с другими классификаторами', '95%')
    match = re.search(comp_pattern, html_content, re.DOTALL | re.IGNORECASE)
    if match:
        insert_pos = match.end(1)
        html_content = html_content[:insert_pos] + comp_img + html_content[insert_pos:]
    
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
    """Main function to add illustrations to all classification cheatsheets."""
    print("=" * 70)
    print("Adding matplotlib illustrations to classification cheatsheets")
    print("=" * 70)
    
    # Generate all illustrations
    print("\n1. Generating illustrations...")
    illustrations = generate_all_illustrations()
    
    # Process each HTML file
    print("\n2. Adding illustrations to HTML files...")
    
    files_to_process = [
        ('cheatsheets/decision_trees_cheatsheet.html', add_illustrations_to_decision_trees),
        ('cheatsheets/random_forest_cheatsheet.html', add_illustrations_to_random_forest),
        ('cheatsheets/svm_cheatsheet.html', add_illustrations_to_svm),
        ('cheatsheets/logreg_cheatsheet.html', add_illustrations_to_logreg),
        ('cheatsheets/knn_cheatsheet.html', add_illustrations_to_knn),
        ('cheatsheets/naive_bayes_cheatsheet.html', add_illustrations_to_naive_bayes),
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
