#!/usr/bin/env python3
"""
Add matplotlib illustrations to metrics and evaluation cheatsheet HTML files.
"""

import re
from generate_metrics_illustrations import generate_all_illustrations

def create_img_tag(base64_data, alt_text, width="100%"):
    """Create HTML img tag with base64 encoded image."""
    return f'''
    <div style="text-align: center; margin: 10px 0;">
      <img src="data:image/png;base64,{base64_data}" 
           alt="{alt_text}" 
           style="max-width: {width}; height: auto; border-radius: 4px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
    </div>
'''

def add_illustrations_to_confusion_matrix(html_content, illustrations):
    """Add illustrations to confusion matrix cheatsheet."""
    # Add binary confusion matrix after visualization section
    viz_pattern = r'(<h2>🔷.*?[Вв]изуализаци.*?</h2>.*?</div>)'
    viz_img = create_img_tag(illustrations['cm_binary'], 
                            'Матрица ошибок: Бинарная классификация', '90%')
    match = re.search(viz_pattern, html_content, re.DOTALL | re.IGNORECASE)
    if match:
        insert_pos = match.end(1)
        html_content = html_content[:insert_pos] + viz_img + html_content[insert_pos:]
    
    # Add normalized confusion matrix
    norm_pattern = r'(cm_normalized.*?plt\.show\(\)</code></pre>\s*</div>)'
    norm_img = create_img_tag(illustrations['cm_normalized'], 
                             'Нормализованная матрица ошибок', '85%')
    match = re.search(norm_pattern, html_content, re.DOTALL)
    if match:
        insert_pos = match.end(1)
        html_content = html_content[:insert_pos] + norm_img + html_content[insert_pos:]
    
    # Add multiclass confusion matrix after multiclass section
    multi_pattern = r'(<h2>🔷.*?[Мм]ультикласс.*?</h2>.*?</div>)'
    multi_img = create_img_tag(illustrations['cm_multiclass'], 
                               'Матрица ошибок: Мультиклассовая классификация', '90%')
    match = re.search(multi_pattern, html_content, re.DOTALL | re.IGNORECASE)
    if match:
        insert_pos = match.end(1)
        html_content = html_content[:insert_pos] + multi_img + html_content[insert_pos:]
    
    return html_content

def add_illustrations_to_roc_auc(html_content, illustrations):
    """Add illustrations to ROC AUC cheatsheet."""
    # Add single ROC curve after basic code or visualization section
    basic_pattern = r'(<h2>🔷.*?[Бб]азовый.*?код.*?</h2>.*?</div>|<h2>🔷.*?[Вв]изуализаци.*?</h2>.*?</div>)'
    basic_img = create_img_tag(illustrations['roc_single'], 
                               'ROC кривая', '95%')
    match = re.search(basic_pattern, html_content, re.DOTALL | re.IGNORECASE)
    if match:
        insert_pos = match.end(1)
        html_content = html_content[:insert_pos] + basic_img + html_content[insert_pos:]
    
    # Add ROC comparison
    comp_pattern = r'(<h2>🔷.*?[Сс]равнени.*?</h2>|<h2>🔷.*?[Нн]есколько.*?модел.*?</h2>)'
    comp_img = create_img_tag(illustrations['roc_comparison'], 
                              'Сравнение ROC кривых разных моделей', '95%')
    match = re.search(comp_pattern, html_content, re.DOTALL | re.IGNORECASE)
    if match:
        insert_pos = match.end(1)
        html_content = html_content[:insert_pos] + comp_img + html_content[insert_pos:]
    
    # Add multiclass ROC
    multi_pattern = r'(<h2>🔷.*?[Мм]ультикласс.*?</h2>)'
    multi_img = create_img_tag(illustrations['roc_multiclass'], 
                               'ROC кривые для мультиклассовой классификации', '95%')
    match = re.search(multi_pattern, html_content, re.DOTALL | re.IGNORECASE)
    if match:
        insert_pos = match.end(1)
        html_content = html_content[:insert_pos] + multi_img + html_content[insert_pos:]
    
    return html_content

def add_illustrations_to_classification_metrics(html_content, illustrations):
    """Add illustrations to classification metrics cheatsheet."""
    # Add metrics visualization after metrics section
    metrics_pattern = r'(<h2>🔷.*?[Мм]етрик.*?</h2>.*?</div>)'
    metrics_img = create_img_tag(illustrations['metrics_viz'], 
                                 'Визуализация метрик классификации', '90%')
    match = re.search(metrics_pattern, html_content, re.DOTALL | re.IGNORECASE)
    if match:
        insert_pos = match.end(1)
        html_content = html_content[:insert_pos] + metrics_img + html_content[insert_pos:]
    
    # Add precision-recall curve
    pr_pattern = r'(<h2>🔷.*?Precision.*?Recall.*?</h2>|<h2>🔷.*?[Тт]очност.*?полнот.*?</h2>)'
    pr_img = create_img_tag(illustrations['precision_recall_curve'], 
                           'Precision-Recall кривая', '90%')
    match = re.search(pr_pattern, html_content, re.DOTALL | re.IGNORECASE)
    if match:
        insert_pos = match.end(1)
        html_content = html_content[:insert_pos] + pr_img + html_content[insert_pos:]
    
    # Add threshold impact
    thresh_pattern = r'(<h2>🔷.*?[Пп]орог.*?</h2>|<h2>🔷.*?[Тт]hreshold.*?</h2>)'
    thresh_img = create_img_tag(illustrations['threshold_impact'], 
                                'Влияние порога на метрики', '95%')
    match = re.search(thresh_pattern, html_content, re.DOTALL | re.IGNORECASE)
    if match:
        insert_pos = match.end(1)
        html_content = html_content[:insert_pos] + thresh_img + html_content[insert_pos:]
    
    return html_content

def add_illustrations_to_regression_metrics(html_content, illustrations):
    """Add illustrations to regression metrics cheatsheet."""
    # Add residuals plot
    residual_pattern = r'(<h2>🔷.*?[Оо]статк.*?</h2>|<h2>🔷.*?Residual.*?</h2>)'
    residual_img = create_img_tag(illustrations['regression_residuals'], 
                                  'График остатков регрессии', '95%')
    match = re.search(residual_pattern, html_content, re.DOTALL | re.IGNORECASE)
    if match:
        insert_pos = match.end(1)
        html_content = html_content[:insert_pos] + residual_img + html_content[insert_pos:]
    
    # Add metrics comparison
    metrics_pattern = r'(<h2>🔷.*?[Мм]етрик.*?</h2>.*?</div>)'
    metrics_img = create_img_tag(illustrations['regression_metrics'], 
                                 'Сравнение метрик регрессии', '95%')
    match = re.search(metrics_pattern, html_content, re.DOTALL | re.IGNORECASE)
    if match:
        insert_pos = match.end(1)
        html_content = html_content[:insert_pos] + metrics_img + html_content[insert_pos:]
    
    return html_content

def add_illustrations_to_learning_curves(html_content, illustrations):
    """Add illustrations to learning curves cheatsheet."""
    # Add good learning curve after ideal/good section
    good_pattern = r'(<h2>🔷.*?[Ии]деальн.*?</h2>|<h2>🔷.*?[Хх]орош.*?</h2>)'
    good_img = create_img_tag(illustrations['learning_good'], 
                             'Кривые обучения: Хорошая модель', '95%')
    match = re.search(good_pattern, html_content, re.DOTALL | re.IGNORECASE)
    if match:
        insert_pos = match.end(1)
        html_content = html_content[:insert_pos] + good_img + html_content[insert_pos:]
    
    # Add overfitting curve
    overfit_pattern = r'(<h2>🔷.*?[Пп]ереобучени.*?</h2>|<h2>🔷.*?High.*?Variance.*?</h2>)'
    overfit_img = create_img_tag(illustrations['learning_overfit'], 
                                 'Кривые обучения: Переобучение', '95%')
    match = re.search(overfit_pattern, html_content, re.DOTALL | re.IGNORECASE)
    if match:
        insert_pos = match.end(1)
        html_content = html_content[:insert_pos] + overfit_img + html_content[insert_pos:]
    
    # Add underfitting curve
    underfit_pattern = r'(<h2>🔷.*?[Нн]едообучени.*?</h2>|<h2>🔷.*?High.*?Bias.*?</h2>)'
    underfit_img = create_img_tag(illustrations['learning_underfit'], 
                                  'Кривые обучения: Недообучение', '95%')
    match = re.search(underfit_pattern, html_content, re.DOTALL | re.IGNORECASE)
    if match:
        insert_pos = match.end(1)
        html_content = html_content[:insert_pos] + underfit_img + html_content[insert_pos:]
    
    return html_content

def add_illustrations_to_validation_curves(html_content, illustrations):
    """Add illustrations to validation curves cheatsheet."""
    # Add validation curve after basic section
    basic_pattern = r'(<h2>🔷.*?[Бб]азовый.*?</h2>|<h2>🔷.*?[Сс]уть.*?</h2>)'
    basic_img = create_img_tag(illustrations['validation_curve'], 
                               'Validation Curve: Выбор гиперпараметра', '95%')
    match = re.search(basic_pattern, html_content, re.DOTALL | re.IGNORECASE)
    if match:
        insert_pos = match.end(1)
        html_content = html_content[:insert_pos] + basic_img + html_content[insert_pos:]
    
    # Add regularization curve
    reg_pattern = r'(<h2>🔷.*?[Рр]егуляризаци.*?</h2>)'
    reg_img = create_img_tag(illustrations['validation_regularization'], 
                            'Validation Curve: Параметр регуляризации', '95%')
    match = re.search(reg_pattern, html_content, re.DOTALL | re.IGNORECASE)
    if match:
        insert_pos = match.end(1)
        html_content = html_content[:insert_pos] + reg_img + html_content[insert_pos:]
    
    return html_content

def add_illustrations_to_bias_variance(html_content, illustrations):
    """Add illustrations to bias-variance cheatsheet."""
    # Add trade-off visualization after basic section
    basic_pattern = r'(<h2>🔷.*?[Сс]уть.*?</h2>|<h2>🔷.*?Trade.*?off.*?</h2>)'
    basic_img = create_img_tag(illustrations['bias_variance_tradeoff'], 
                               'Bias-Variance Trade-off', '95%')
    match = re.search(basic_pattern, html_content, re.DOTALL | re.IGNORECASE)
    if match:
        insert_pos = match.end(1)
        html_content = html_content[:insert_pos] + basic_img + html_content[insert_pos:]
    
    # Add visual examples
    visual_pattern = r'(<h2>🔷.*?[Вв]изуальн.*?</h2>|<h2>🔷.*?[Пп]римеры.*?</h2>)'
    visual_img = create_img_tag(illustrations['bias_variance_examples'], 
                                'Визуальные примеры Bias и Variance', '95%')
    match = re.search(visual_pattern, html_content, re.DOTALL | re.IGNORECASE)
    if match:
        insert_pos = match.end(1)
        html_content = html_content[:insert_pos] + visual_img + html_content[insert_pos:]
    
    return html_content

def add_illustrations_to_bias_variance_decomposition(html_content, illustrations):
    """Add illustrations to bias-variance decomposition cheatsheet."""
    # Add trade-off diagram
    pattern = r'(<h2>🔷.*?[Сс]уть.*?</h2>|<h2>🔷.*?[Дд]екомпозиц.*?</h2>)'
    img = create_img_tag(illustrations['bias_variance_tradeoff'], 
                        'Декомпозиция Bias-Variance', '95%')
    match = re.search(pattern, html_content, re.DOTALL | re.IGNORECASE)
    if match:
        insert_pos = match.end(1)
        html_content = html_content[:insert_pos] + img + html_content[insert_pos:]
    
    # Add visual examples
    example_pattern = r'(<h2>🔷.*?[Пп]римеры.*?</h2>)'
    example_img = create_img_tag(illustrations['bias_variance_examples'], 
                                 'Примеры разложения ошибки', '95%')
    match = re.search(example_pattern, html_content, re.DOTALL | re.IGNORECASE)
    if match:
        insert_pos = match.end(1)
        html_content = html_content[:insert_pos] + example_img + html_content[insert_pos:]
    
    return html_content

def add_illustrations_to_overfitting_underfitting(html_content, illustrations):
    """Add illustrations to overfitting/underfitting cheatsheet."""
    # Add main visualization after basic section
    basic_pattern = r'(<h2>🔷.*?[Сс]уть.*?</h2>|<h2>🔷.*?[Оо]пределени.*?</h2>)'
    basic_img = create_img_tag(illustrations['overfitting_underfitting'], 
                               'Underfitting, Good Fit, Overfitting', '95%')
    match = re.search(basic_pattern, html_content, re.DOTALL | re.IGNORECASE)
    if match:
        insert_pos = match.end(1)
        html_content = html_content[:insert_pos] + basic_img + html_content[insert_pos:]
    
    # Add learning curves showing overfitting
    overfit_pattern = r'(<h2>🔷.*?[Пп]ереобучени.*?</h2>)'
    overfit_img = create_img_tag(illustrations['learning_overfit'], 
                                 'Кривые обучения при переобучении', '90%')
    match = re.search(overfit_pattern, html_content, re.DOTALL | re.IGNORECASE)
    if match:
        insert_pos = match.end(1)
        html_content = html_content[:insert_pos] + overfit_img + html_content[insert_pos:]
    
    # Add learning curves showing underfitting
    underfit_pattern = r'(<h2>🔷.*?[Нн]едообучени.*?</h2>)'
    underfit_img = create_img_tag(illustrations['learning_underfit'], 
                                  'Кривые обучения при недообучении', '90%')
    match = re.search(underfit_pattern, html_content, re.DOTALL | re.IGNORECASE)
    if match:
        insert_pos = match.end(1)
        html_content = html_content[:insert_pos] + underfit_img + html_content[insert_pos:]
    
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
    """Main function to add illustrations to all metrics/evaluation cheatsheets."""
    print("=" * 70)
    print("Adding matplotlib illustrations to Metrics & Evaluation cheatsheets")
    print("=" * 70)
    
    # Generate all illustrations
    print("\n1. Generating illustrations...")
    illustrations = generate_all_illustrations()
    
    # Process each HTML file
    print("\n2. Adding illustrations to HTML files...")
    
    files_to_process = [
        ('cheatsheets/confusion_matrix_cheatsheet.html', add_illustrations_to_confusion_matrix),
        ('cheatsheets/roc_auc_cheatsheet.html', add_illustrations_to_roc_auc),
        ('cheatsheets/classification_metrics_cheatsheet.html', add_illustrations_to_classification_metrics),
        ('cheatsheets/regression_metrics_cheatsheet.html', add_illustrations_to_regression_metrics),
        ('cheatsheets/learning_curves_cheatsheet.html', add_illustrations_to_learning_curves),
        ('cheatsheets/learning_validation_curves_cheatsheet.html', add_illustrations_to_validation_curves),
        ('cheatsheets/bias_variance_cheatsheet.html', add_illustrations_to_bias_variance),
        ('cheatsheets/bias_variance_decomposition_cheatsheet.html', add_illustrations_to_bias_variance_decomposition),
        ('cheatsheets/overfitting_underfitting_cheatsheet.html', add_illustrations_to_overfitting_underfitting),
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
