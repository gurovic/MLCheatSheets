#!/usr/bin/env python3
"""
Add matplotlib illustrations to ensemble cheatsheet HTML files.
"""

import re
import html
import traceback
from generate_ensemble_illustrations import generate_all_illustrations

def create_img_tag(base64_data, alt_text, width="100%"):
    """Create HTML img tag with base64 encoded image."""
    # Sanitize alt_text to prevent XSS
    safe_alt_text = html.escape(alt_text)
    return f'''
    <div style="text-align: center; margin: 10px 0;">
      <img src="data:image/png;base64,{base64_data}" 
           alt="{safe_alt_text}" 
           style="max-width: {width}; height: auto; border-radius: 4px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
    </div>
'''

def add_illustrations_to_ensemble_methods(html_content, illustrations):
    """Add illustrations to ensemble methods cheatsheet."""
    # Add voting diagram after voting classifier section
    voting_pattern = r'(voting_soft\.fit\(X_train, y_train\).*?y_pred = voting_soft\.predict\(X_test\)</code></pre>\s*</div>)'
    voting_img = create_img_tag(illustrations['voting_diagram'], 'Архитектура Voting Ensemble', '95%')
    match = re.search(voting_pattern, html_content, re.DOTALL)
    if match:
        insert_pos = match.end(1)
        html_content = html_content[:insert_pos] + voting_img + html_content[insert_pos:]
    
    # Add ensemble comparison after stacking section or near the end
    stacking_pattern = r'(stacking\.fit\(X_train, y_train\).*?y_pred = stacking\.predict\(X_test\)</code></pre>\s*</div>)'
    comparison_img = create_img_tag(illustrations['ensemble_comparison'], 
                                    'Сравнение ансамбля с отдельными моделями', '95%')
    match = re.search(stacking_pattern, html_content, re.DOTALL)
    if match:
        insert_pos = match.end(1)
        html_content = html_content[:insert_pos] + comparison_img + html_content[insert_pos:]
    
    return html_content

def add_illustrations_to_bagging(html_content, illustrations):
    """Add illustrations to bagging cheatsheet."""
    # Add bagging diagram at the beginning or after concept explanation
    concept_pattern = r'(<h2>🔷.*?[Сс]уть.*?</h2>.*?</ul>\s*</div>)'
    bagging_diagram_img = create_img_tag(illustrations['bagging_diagram'], 
                                         'Схема работы Bagging', '95%')
    match = re.search(concept_pattern, html_content, re.DOTALL)
    if match:
        insert_pos = match.end(1)
        html_content = html_content[:insert_pos] + bagging_diagram_img + html_content[insert_pos:]
    
    # Add variance reduction visualization after code example
    bagging_pattern = r'(BaggingClassifier.*?bagging\.fit\(X_train, y_train\).*?</code></pre>\s*</div>)'
    variance_img = create_img_tag(illustrations['bagging_variance_reduction'], 
                                  'Bagging: снижение переобучения', '90%')
    match = re.search(bagging_pattern, html_content, re.DOTALL)
    if match:
        insert_pos = match.end(1)
        html_content = html_content[:insert_pos] + variance_img + html_content[insert_pos:]
    
    return html_content

def add_illustrations_to_boosting(html_content, illustrations):
    """Add illustrations to boosting cheatsheet."""
    # Add boosting diagram after concept section
    concept_pattern = r'(<h2>🔷.*?[Сс]уть.*?</h2>.*?</ul>\s*</div>)'
    boosting_diagram_img = create_img_tag(illustrations['boosting_diagram'], 
                                          'Схема работы Boosting', '95%')
    match = re.search(concept_pattern, html_content, re.DOTALL)
    if match:
        insert_pos = match.end(1)
        html_content = html_content[:insert_pos] + boosting_diagram_img + html_content[insert_pos:]
    
    # Add comparison with bagging
    comparison_pattern = r'(<h2>🔷.*?Gradient Boosting.*?</h2>)'
    comparison_img = create_img_tag(illustrations['boosting_vs_bagging'], 
                                    'Сравнение Boosting и Bagging', '90%')
    match = re.search(comparison_pattern, html_content, re.DOTALL)
    if match:
        insert_pos = match.end(1)
        html_content = html_content[:insert_pos] + comparison_img + html_content[insert_pos:]
    
    return html_content

def add_illustrations_to_adaboost(html_content, illustrations):
    """Add illustrations to AdaBoost cheatsheet."""
    # Add weight updates visualization after AdaBoost code
    adaboost_pattern = r'(AdaBoostClassifier.*?ada\.fit\(X_train, y_train\).*?</code></pre>\s*</div>)'
    weight_img = create_img_tag(illustrations['adaboost_weight_updates'], 
                                'AdaBoost: обновление весов образцов', '90%')
    match = re.search(adaboost_pattern, html_content, re.DOTALL)
    if match:
        insert_pos = match.end(1)
        html_content = html_content[:insert_pos] + weight_img + html_content[insert_pos:]
    
    # Add performance over iterations
    performance_pattern = r'(<h2>🔷.*?[Пп]араметры.*?</h2>)'
    performance_img = create_img_tag(illustrations['adaboost_performance'], 
                                     'Качество AdaBoost от числа итераций', '85%')
    match = re.search(performance_pattern, html_content, re.DOTALL)
    if match:
        insert_pos = match.end(1)
        html_content = html_content[:insert_pos] + performance_img + html_content[insert_pos:]
    
    return html_content

def add_illustrations_to_xgboost(html_content, illustrations):
    """Add illustrations to XGBoost cheatsheet."""
    # Add gradient boosting residuals visualization after first example
    xgb_pattern = r'(model\.fit\(X_train, y_train\).*?y_pred = model\.predict\(X_test\).*?</code></pre>\s*</div>)'
    residuals_img = create_img_tag(illustrations['gradient_boosting_residuals'], 
                                   'Gradient Boosting: подгонка остатков', '90%')
    match = re.search(xgb_pattern, html_content, re.DOTALL)
    if match:
        insert_pos = match.end(1)
        html_content = html_content[:insert_pos] + residuals_img + html_content[insert_pos:]
    
    # Add learning rate effect
    lr_pattern = r'(<h2>🔷.*?[Пп]араметры.*?</h2>)'
    lr_img = create_img_tag(illustrations['learning_rate_effect'], 
                           'Влияние learning rate', '90%')
    match = re.search(lr_pattern, html_content, re.DOTALL)
    if match:
        insert_pos = match.end(1)
        html_content = html_content[:insert_pos] + lr_img + html_content[insert_pos:]
    
    return html_content

def add_illustrations_to_lightgbm(html_content, illustrations):
    """Add illustrations to LightGBM cheatsheet."""
    # Add tree growth comparison after sklearn API section
    sklearn_pattern = r'(y_pred = clf\.predict\(X_test\).*?y_proba = clf\.predict_proba\(X_test\).*?</code></pre>\s*</div>)'
    tree_growth_img = create_img_tag(illustrations['tree_growth_comparison'], 
                                     'Сравнение стратегий роста дерева (LightGBM leaf-wise)', '95%')
    match = re.search(sklearn_pattern, html_content, re.DOTALL)
    if match:
        insert_pos = match.end(1)
        html_content = html_content[:insert_pos] + tree_growth_img + html_content[insert_pos:]
    
    # Add gradient boosting residuals after concept section
    concept_pattern = r'(<h2>🔷 1\. Суть</h2>.*?</ul>\s*</div>)'
    residuals_img = create_img_tag(illustrations['gradient_boosting_residuals'], 
                                   'Gradient Boosting: подгонка остатков', '85%')
    match = re.search(concept_pattern, html_content, re.DOTALL)
    if match:
        insert_pos = match.end(1)
        html_content = html_content[:insert_pos] + residuals_img + html_content[insert_pos:]
    
    return html_content

def add_illustrations_to_catboost(html_content, illustrations):
    """Add illustrations to CatBoost cheatsheet."""
    # Add tree growth comparison after first code example
    catboost_pattern = r'(model\.fit\(X_train, y_train.*?\).*?</code></pre>\s*</div>)'
    tree_growth_img = create_img_tag(illustrations['tree_growth_comparison'], 
                                     'Стратегии роста дерева', '90%')
    match = re.search(catboost_pattern, html_content, re.DOTALL)
    if match:
        insert_pos = match.end(1)
        html_content = html_content[:insert_pos] + tree_growth_img + html_content[insert_pos:]
    
    # Add learning rate effect after parameters section
    params_pattern = r'(<h2>🔷.*?[Пп]араметры.*?</h2>)'
    lr_img = create_img_tag(illustrations['learning_rate_effect'], 
                           'Влияние learning rate', '85%')
    match = re.search(params_pattern, html_content, re.DOTALL)
    if match:
        insert_pos = match.end(1)
        html_content = html_content[:insert_pos] + lr_img + html_content[insert_pos:]
    
    return html_content

def add_illustrations_to_voting_stacking(html_content, illustrations):
    """Add illustrations to voting/stacking cheatsheet."""
    # Add voting diagram
    voting_pattern = r'(VotingClassifier.*?voting.*?\.fit\(X_train, y_train\).*?</code></pre>\s*</div>)'
    voting_img = create_img_tag(illustrations['voting_diagram'], 
                                'Архитектура Voting Ensemble', '90%')
    match = re.search(voting_pattern, html_content, re.DOTALL)
    if match:
        insert_pos = match.end(1)
        html_content = html_content[:insert_pos] + voting_img + html_content[insert_pos:]
    
    # Add stacking diagram
    stacking_pattern = r'(StackingClassifier.*?stacking.*?\.fit\(X_train, y_train\).*?</code></pre>\s*</div>)'
    stacking_img = create_img_tag(illustrations['stacking_diagram'], 
                                  'Архитектура Stacking', '95%')
    match = re.search(stacking_pattern, html_content, re.DOTALL)
    if match:
        insert_pos = match.end(1)
        html_content = html_content[:insert_pos] + stacking_img + html_content[insert_pos:]
    
    # Add ensemble comparison
    comparison_pattern = r'(<h2>🔷.*?[Сс]равнение.*?</h2>)'
    comparison_img = create_img_tag(illustrations['ensemble_comparison'], 
                                    'Сравнение ансамбля с отдельными моделями', '90%')
    match = re.search(comparison_pattern, html_content, re.DOTALL)
    if match:
        insert_pos = match.end(1)
        html_content = html_content[:insert_pos] + comparison_img + html_content[insert_pos:]
    
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
    """Main function to add illustrations to all ensemble cheatsheets."""
    print("=" * 70)
    print("Adding matplotlib illustrations to ensemble cheatsheets")
    print("=" * 70)
    
    # Generate all illustrations
    print("\n1. Generating illustrations...")
    illustrations = generate_all_illustrations()
    
    # Process each HTML file
    print("\n2. Adding illustrations to HTML files...")
    
    files_to_process = [
        ('cheatsheets/ensemble_methods_cheatsheet.html', add_illustrations_to_ensemble_methods),
        ('cheatsheets/bagging_cheatsheet.html', add_illustrations_to_bagging),
        ('cheatsheets/boosting_cheatsheet.html', add_illustrations_to_boosting),
        ('cheatsheets/adaboost_cheatsheet.html', add_illustrations_to_adaboost),
        ('cheatsheets/xgboost_cheatsheet.html', add_illustrations_to_xgboost),
        ('cheatsheets/lightgbm_cheatsheet.html', add_illustrations_to_lightgbm),
        ('cheatsheets/catboost_cheatsheet.html', add_illustrations_to_catboost),
        ('cheatsheets/voting_stacking_cheatsheet.html', add_illustrations_to_voting_stacking),
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
