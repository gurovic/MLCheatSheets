#!/usr/bin/env python3
"""
Add matplotlib illustrations to validation and tuning cheatsheet HTML files.
"""

import re
import os
from generate_validation_tuning_illustrations import generate_all_illustrations

def create_img_tag(base64_data, alt_text, width="100%"):
    """Create HTML img tag with base64 encoded image."""
    return f'''
    <div style="text-align: center; margin: 10px 0;">
      <img src="data:image/png;base64,{base64_data}" 
           alt="{alt_text}" 
           style="max-width: {width}; height: auto; border-radius: 4px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
    </div>
'''

def add_illustrations_to_cross_validation(html_content, illustrations):
    """Add illustrations to cross_validation_cheatsheet.html."""
    print("  Adding illustrations to cross_validation_cheatsheet.html...")
    
    # Add K-Fold diagram after the K-Fold section
    kfold_pattern = r'(<h2>🔷 2\. K-Fold Cross-Validation</h2>.*?</code></pre>\s*</div>)'
    kfold_img = create_img_tag(illustrations['cv_kfold_diagram'], 
                               'K-Fold Cross-Validation схема', '90%')
    match = re.search(kfold_pattern, html_content, re.DOTALL)
    if match:
        insert_pos = match.end(1)
        html_content = html_content[:insert_pos] + kfold_img + html_content[insert_pos:]
        print("    ✓ Added K-Fold diagram")
    else:
        print("    ✗ K-Fold pattern not found")
    
    # Add Time Series Split diagram
    timeseries_pattern = r'(<h2>🔷 6\. Time Series Split</h2>.*?</code></pre>\s*</div>)'
    timeseries_img = create_img_tag(illustrations['cv_timeseries_split'],
                                    'Time Series Split схема', '90%')
    match = re.search(timeseries_pattern, html_content, re.DOTALL)
    if match:
        insert_pos = match.end(1)
        html_content = html_content[:insert_pos] + timeseries_img + html_content[insert_pos:]
        print("    ✓ Added Time Series Split diagram")
    else:
        print("    ✗ Time Series Split pattern not found")
    
    # Add Nested CV diagram
    nested_pattern = r'(<h2>🔷 9\. Nested Cross-Validation</h2>.*?</code></pre>\s*</div>)'
    nested_img = create_img_tag(illustrations['cv_nested_diagram'],
                                'Nested Cross-Validation схема', '85%')
    match = re.search(nested_pattern, html_content, re.DOTALL)
    if match:
        insert_pos = match.end(1)
        html_content = html_content[:insert_pos] + nested_img + html_content[insert_pos:]
        print("    ✓ Added Nested CV diagram")
    else:
        print("    ✗ Nested CV pattern not found")
    
    # Add CV scores comparison after Stratified K-Fold
    scores_pattern = r'(<h2>🔷 4\. Stratified K-Fold</h2>.*?</code></pre>\s*</div>)'
    scores_img = create_img_tag(illustrations['cv_scores_comparison'],
                                'Сравнение K-Fold и Stratified K-Fold', '90%')
    match = re.search(scores_pattern, html_content, re.DOTALL)
    if match:
        insert_pos = match.end(1)
        html_content = html_content[:insert_pos] + scores_img + html_content[insert_pos:]
        print("    ✓ Added CV scores comparison")
    else:
        print("    ✗ CV scores comparison pattern not found")
    
    return html_content

def add_illustrations_to_hyperparameter_tuning(html_content, illustrations):
    """Add illustrations to hyperparameter_tuning_cheatsheet.html."""
    print("  Adding illustrations to hyperparameter_tuning_cheatsheet.html...")
    
    # Add Grid Search heatmap
    grid_pattern = r'(<h2>🔷 2\. Grid Search</h2>.*?best_model = grid_search\.best_estimator_</code></pre>\s*</div>)'
    grid_img = create_img_tag(illustrations['tuning_grid_heatmap'],
                             'Grid Search результаты', '90%')
    match = re.search(grid_pattern, html_content, re.DOTALL)
    if match:
        insert_pos = match.end(1)
        html_content = html_content[:insert_pos] + grid_img + html_content[insert_pos:]
        print("    ✓ Added Grid Search heatmap")
    else:
        print("    ✗ Grid Search pattern not found")
    
    # Add Random vs Grid comparison after comparison section
    comparison_pattern = r'(<h2>🔷 4\. Сравнение методов</h2>.*?</table>\s*</div>)'
    comparison_img = create_img_tag(illustrations['tuning_random_vs_grid'],
                                    'Random Search vs Grid Search', '95%')
    match = re.search(comparison_pattern, html_content, re.DOTALL)
    if match:
        insert_pos = match.end(1)
        html_content = html_content[:insert_pos] + comparison_img + html_content[insert_pos:]
        print("    ✓ Added Random vs Grid comparison")
    else:
        print("    ✗ Comparison pattern not found")
    
    # Add optimization history after Bayesian Optimization section
    bayesian_pattern = r'(<h2>🔷 5\. Bayesian Optimization \(Optuna\)</h2>.*?best_model\.fit\(X_train, y_train\)</code></pre>\s*</div>)'
    optimization_img = create_img_tag(illustrations['tuning_optimization_history'],
                                      'История оптимизации', '90%')
    match = re.search(bayesian_pattern, html_content, re.DOTALL)
    if match:
        insert_pos = match.end(1)
        html_content = html_content[:insert_pos] + optimization_img + html_content[insert_pos:]
        print("    ✓ Added optimization history")
    else:
        print("    ✗ Bayesian Optimization pattern not found")
    
    # Add hyperparameter importance after Grid Search results analysis
    analysis_pattern = r'(<h2>🔷 9\. Анализ результатов Grid Search</h2>.*?plt\.show\(\)</code></pre>\s*</div>)'
    importance_img = create_img_tag(illustrations['tuning_param_importance'],
                                    'Важность гиперпараметров', '85%')
    match = re.search(analysis_pattern, html_content, re.DOTALL)
    if match:
        insert_pos = match.end(1)
        html_content = html_content[:insert_pos] + importance_img + html_content[insert_pos:]
        print("    ✓ Added hyperparameter importance")
    else:
        print("    ✗ Analysis pattern not found")
    
    return html_content

def add_illustrations_to_model_calibration(html_content, illustrations):
    """Add illustrations to model_calibration_cheatsheet.html."""
    print("  Adding illustrations to model_calibration_cheatsheet.html...")
    
    # Add calibration curve after the calibration curve section (no numbers in headers)
    curve_pattern = r'(<h2>🔷 Calibration Curve</h2>.*?plt\.show\(\)</code></pre>\s*</div>)'
    curve_img = create_img_tag(illustrations['calibration_curve'],
                               'Calibration Curve - До и после калибровки', '90%')
    match = re.search(curve_pattern, html_content, re.DOTALL)
    if match:
        insert_pos = match.end(1)
        html_content = html_content[:insert_pos] + curve_img + html_content[insert_pos:]
        print("    ✓ Added calibration curve")
    else:
        print("    ✗ Calibration curve pattern not found")
    
    # Add reliability diagram after the full evaluation section
    reliability_pattern = r'(<h2>🔷 Полная оценка</h2>.*?evaluate_calibration.*?</code></pre>\s*</div>)'
    reliability_img = create_img_tag(illustrations['calibration_reliability'],
                                     'Reliability Diagram с гистограммами', '95%')
    match = re.search(reliability_pattern, html_content, re.DOTALL)
    if match:
        insert_pos = match.end(1)
        html_content = html_content[:insert_pos] + reliability_img + html_content[insert_pos:]
        print("    ✓ Added reliability diagram")
    else:
        print("    ✗ Reliability pattern not found")
    
    # Add methods comparison after calibration methods table
    methods_pattern = r'(<h2>🔷 Методы калибровки</h2>.*?</table>\s*</div>)'
    methods_img = create_img_tag(illustrations['calibration_methods_comparison'],
                                 'Сравнение методов калибровки', '90%')
    match = re.search(methods_pattern, html_content, re.DOTALL)
    if match:
        insert_pos = match.end(1)
        html_content = html_content[:insert_pos] + methods_img + html_content[insert_pos:]
        print("    ✓ Added calibration methods comparison")
    else:
        print("    ✗ Methods comparison pattern not found")
    
    # Add Brier score comparison after the Brier Score section
    brier_pattern = r'(<h2>🔷 Brier Score</h2>.*?# Чем меньше, тем лучше</code></pre>\s*</div>)'
    brier_img = create_img_tag(illustrations['calibration_brier_comparison'],
                               'Влияние калибровки на Brier Score', '85%')
    match = re.search(brier_pattern, html_content, re.DOTALL)
    if match:
        insert_pos = match.end(1)
        html_content = html_content[:insert_pos] + brier_img + html_content[insert_pos:]
        print("    ✓ Added Brier score comparison")
    else:
        print("    ✗ Brier score pattern not found")
    
    return html_content

def main():
    """Main function to add illustrations to all cheatsheets."""
    print("Generating illustrations...")
    illustrations = generate_all_illustrations()
    print("\nEmbedding illustrations into HTML files...\n")
    
    cheatsheets_dir = "cheatsheets"
    
    # Process cross_validation_cheatsheet.html
    cv_file = os.path.join(cheatsheets_dir, "cross_validation_cheatsheet.html")
    if os.path.exists(cv_file):
        with open(cv_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        content = add_illustrations_to_cross_validation(content, illustrations)
        
        with open(cv_file, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"  ✓ Updated {cv_file}\n")
    else:
        print(f"  ✗ File not found: {cv_file}\n")
    
    # Process hyperparameter_tuning_cheatsheet.html
    tuning_file = os.path.join(cheatsheets_dir, "hyperparameter_tuning_cheatsheet.html")
    if os.path.exists(tuning_file):
        with open(tuning_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        content = add_illustrations_to_hyperparameter_tuning(content, illustrations)
        
        with open(tuning_file, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"  ✓ Updated {tuning_file}\n")
    else:
        print(f"  ✗ File not found: {tuning_file}\n")
    
    # Process model_calibration_cheatsheet.html
    calibration_file = os.path.join(cheatsheets_dir, "model_calibration_cheatsheet.html")
    if os.path.exists(calibration_file):
        with open(calibration_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        content = add_illustrations_to_model_calibration(content, illustrations)
        
        with open(calibration_file, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"  ✓ Updated {calibration_file}\n")
    else:
        print(f"  ✗ File not found: {calibration_file}\n")
    
    print("✓ All cheatsheets have been updated with illustrations!")

if __name__ == "__main__":
    main()
