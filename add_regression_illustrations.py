#!/usr/bin/env python3
"""
Add matplotlib illustrations to regression cheatsheet HTML files.
"""

import re
from generate_regression_illustrations import generate_all_illustrations

def create_img_tag(base64_data, alt_text, width="100%"):
    """Create HTML img tag with base64 encoded image."""
    return f'''
    <div style="text-align: center; margin: 10px 0;">
      <img src="data:image/png;base64,{base64_data}" 
           alt="{alt_text}" 
           style="max-width: {width}; height: auto; border-radius: 4px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
    </div>
'''

def add_illustrations_to_linreg(html_content, illustrations):
    """Add illustrations to linear regression cheatsheet."""
    # Add basic regression plot after the basic code section
    basic_pattern = r'(<div class="block">\s*<h2>🔷 2\. Базовый код</h2>.*?</code></pre>\s*</div>)'
    basic_img = create_img_tag(illustrations['linreg_basic'], 'Линейная регрессия', '85%')
    match = re.search(basic_pattern, html_content, re.DOTALL)
    if match:
        insert_pos = match.end(1)
        html_content = html_content[:insert_pos] + basic_img + html_content[insert_pos:]
    
    # Add residuals visualization after section 5
    residuals_pattern = r'(<h2>🔷 5\. Метрики качества.*?</div>\s*<div class="block">)'
    residuals_img = create_img_tag(illustrations['linreg_residuals'], 'Остатки регрессии', '95%')
    match = re.search(residuals_pattern, html_content, re.DOTALL)
    if match:
        insert_pos = match.end(0) - len('<div class="block">')
        html_content = html_content[:insert_pos] + residuals_img + '\n  ' + html_content[insert_pos:]
    
    # Add R² comparison after section 6
    r2_pattern = r'(<h2>🔷 6\. Интерпретация.*?</div>\s*<div class="block">)'
    r2_img = create_img_tag(illustrations['linreg_r2'], 'Сравнение моделей по R²', '95%')
    match = re.search(r2_pattern, html_content, re.DOTALL)
    if match:
        insert_pos = match.end(0) - len('<div class="block">')
        html_content = html_content[:insert_pos] + r2_img + '\n  ' + html_content[insert_pos:]
    
    return html_content

def add_illustrations_to_polynomial(html_content, illustrations):
    """Add illustrations to polynomial regression cheatsheet."""
    # Add polynomial degrees comparison after section 3
    degrees_pattern = r'(<h2>🔷 3\. Степень полинома.*?</div>\s*<div class="block">)'
    degrees_img = create_img_tag(illustrations['polynomial_degrees'], 'Полиномы разных степеней', '95%')
    match = re.search(degrees_pattern, html_content, re.DOTALL)
    if match:
        insert_pos = match.end(0) - len('<div class="block">')
        html_content = html_content[:insert_pos] + degrees_img + '\n  ' + html_content[insert_pos:]
    
    # Add overfitting illustration after visualization example (section 7)
    overfitting_pattern = r'(plt\.show\(\)</code></pre>\s*</div>\s*<div class="block">\s*<h2>🔷 8\.)'
    overfitting_img = create_img_tag(illustrations['polynomial_overfitting'], 'Переобучение полиномиальной регрессии', '95%')
    match = re.search(overfitting_pattern, html_content, re.DOTALL)
    if match:
        insert_pos = match.end(1)
        html_content = html_content[:insert_pos] + overfitting_img + '\n  ' + html_content[insert_pos:]
    
    return html_content

def add_illustrations_to_ridge_lasso(html_content, illustrations):
    """Add illustrations to Ridge/Lasso/ElasticNet cheatsheet."""
    # Add comparison of regularization methods after section 5 (ElasticNet)
    comparison_pattern = r'(<h2>🔷 5\. ElasticNet.*?</div>\s*<div class="block">)'
    comparison_img = create_img_tag(illustrations['regularization_comparison'], 
                                    'Сравнение методов регуляризации', '95%')
    match = re.search(comparison_pattern, html_content, re.DOTALL)
    if match:
        insert_pos = match.end(0) - len('<div class="block">')
        html_content = html_content[:insert_pos] + comparison_img + '\n  ' + html_content[insert_pos:]
    
    # Add coefficients comparison after section 6
    coef_pattern = r'(<h2>🔷 6\. Сравнение Ridge vs Lasso.*?</div>\s*<div class="block">)'
    coef_img = create_img_tag(illustrations['regularization_coefficients'], 
                              'Влияние регуляризации на коэффициенты', '90%')
    match = re.search(coef_pattern, html_content, re.DOTALL)
    if match:
        insert_pos = match.end(0) - len('<div class="block">')
        html_content = html_content[:insert_pos] + coef_img + '\n  ' + html_content[insert_pos:]
    
    # Add regularization path after section 7
    path_pattern = r'(<h2>🔷 7\. Подбор alpha.*?</div>\s*<div class="block">)'
    path_img = create_img_tag(illustrations['regularization_path'], 
                              'Путь регуляризации: Ridge vs Lasso', '95%')
    match = re.search(path_pattern, html_content, re.DOTALL)
    if match:
        insert_pos = match.end(0) - len('<div class="block">')
        html_content = html_content[:insert_pos] + path_img + '\n  ' + html_content[insert_pos:]
    
    return html_content

def add_illustrations_to_decision_trees(html_content, illustrations):
    """Add illustrations to decision trees regression cheatsheet."""
    # Add basic decision tree visualization after basic code
    basic_pattern = r'(<div class="block">\s*<h2>🔷 2\. Базовый код</h2>.*?</code></pre>\s*</div>)'
    basic_img = create_img_tag(illustrations['decision_tree_basic'], 
                               'Деревья решений разной глубины', '95%')
    match = re.search(basic_pattern, html_content, re.DOTALL)
    if match:
        insert_pos = match.end(1)
        html_content = html_content[:insert_pos] + basic_img + html_content[insert_pos:]
    
    # Add overfitting illustration
    overfitting_pattern = r'(<h2>🔷 7\. Переобучение и регуляризация.*?</code></pre>\s*</div>)'
    overfitting_img = create_img_tag(illustrations['decision_tree_overfitting'], 
                                     'Переобучение дерева решений', '95%')
    match = re.search(overfitting_pattern, html_content, re.DOTALL)
    if match:
        insert_pos = match.end(1)
        html_content = html_content[:insert_pos] + overfitting_img + html_content[insert_pos:]
    
    # Add 2D visualization if there's a section about feature importance
    viz_pattern = r'(<h2>🔷 6\. Важность признаков.*?plt\.show\(\)</code></pre>\s*</div>)'
    viz_img = create_img_tag(illustrations['decision_tree_2d'], 
                            'Дерево решений в 2D пространстве', '95%')
    match = re.search(viz_pattern, html_content, re.DOTALL)
    if match:
        insert_pos = match.end(1)
        html_content = html_content[:insert_pos] + viz_img + html_content[insert_pos:]
    
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
    """Main function to add illustrations to all regression cheatsheets."""
    print("=" * 70)
    print("Adding matplotlib illustrations to regression cheatsheets")
    print("=" * 70)
    
    # Generate all illustrations
    print("\n1. Generating illustrations...")
    illustrations = generate_all_illustrations()
    
    # Process each HTML file
    print("\n2. Adding illustrations to HTML files...")
    
    files_to_process = [
        ('cheatsheets/linreg_cheatsheet.html', add_illustrations_to_linreg),
        ('cheatsheets/polynomial_regression_cheatsheet.html', add_illustrations_to_polynomial),
        ('cheatsheets/ridge_lasso_elasticnet_cheatsheet.html', add_illustrations_to_ridge_lasso),
        ('cheatsheets/decision_trees_regression_cheatsheet.html', add_illustrations_to_decision_trees),
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
