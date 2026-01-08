#!/usr/bin/env python3
"""
Add matplotlib illustrations to interpretability cheatsheet HTML files.
"""

import re
from generate_interpretability_illustrations import generate_all_illustrations

def create_img_tag(base64_data, alt_text, width="100%"):
    """Create HTML img tag with base64 encoded image."""
    return f'''
    <div style="text-align: center; margin: 10px 0;">
      <img src="data:image/png;base64,{base64_data}" 
           alt="{alt_text}" 
           style="max-width: {width}; height: auto; border-radius: 4px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
    </div>
'''

def add_illustrations_to_xai(html_content, illustrations):
    """Add illustrations to Explainable AI (XAI) cheatsheet."""
    # Add feature importance after the feature importance code section
    feature_pattern = r'(plt\.show\(\)</code></pre>\s*</div>)'
    feature_img = create_img_tag(illustrations['xai_feature_importance'], 
                                  'Feature Importance визуализация', '90%')
    # Find first occurrence after feature importance code
    match = re.search(r'feature_importances_.*?' + feature_pattern, html_content, re.DOTALL | re.IGNORECASE)
    if match:
        insert_pos = match.end(1)
        html_content = html_content[:insert_pos] + feature_img + html_content[insert_pos:]
    
    # Add model comparison after types of interpretability section
    model_pattern = r'(<h2>🔷 2\. Типы интерпретируемости</h2>.*?</table>\s*</div>)'
    model_img = create_img_tag(illustrations['xai_model_comparison'], 
                               'Сравнение моделей: интерпретируемость vs точность', '95%')
    match = re.search(model_pattern, html_content, re.DOTALL)
    if match:
        insert_pos = match.end(1)
        html_content = html_content[:insert_pos] + model_img + html_content[insert_pos:]
    
    return html_content

def add_illustrations_to_shap(html_content, illustrations):
    """Add illustrations to SHAP cheatsheet."""
    # Add waterfall plot after waterfall code mention
    waterfall_pattern = r'(shap\.plots\.waterfall\(shap_values\[0\]\).*?</code></pre>\s*</div>)'
    waterfall_img = create_img_tag(illustrations['shap_waterfall'], 
                                    'SHAP Waterfall Plot', '90%')
    match = re.search(waterfall_pattern, html_content, re.DOTALL)
    if match:
        insert_pos = match.end(1)
        html_content = html_content[:insert_pos] + waterfall_img + html_content[insert_pos:]
    
    # Add beeswarm plot after beeswarm code mention
    beeswarm_pattern = r'(shap\.plots\.beeswarm\(shap_values\).*?</code></pre>\s*</div>)'
    beeswarm_img = create_img_tag(illustrations['shap_beeswarm'], 
                                   'SHAP Beeswarm Plot', '90%')
    match = re.search(beeswarm_pattern, html_content, re.DOTALL)
    if match:
        insert_pos = match.end(1)
        html_content = html_content[:insert_pos] + beeswarm_img + html_content[insert_pos:]
    
    return html_content

def add_illustrations_to_lime(html_content, illustrations):
    """Add illustrations to LIME cheatsheet."""
    # Add explanation visualization after explain_instance code
    explain_pattern = r'(explainer\.explain_instance\(.*?num_features=10.*?</code></pre>\s*</div>)'
    explain_img = create_img_tag(illustrations['lime_explanation'], 
                                  'LIME объяснение предсказания', '90%')
    match = re.search(explain_pattern, html_content, re.DOTALL)
    if match:
        insert_pos = match.end(1)
        html_content = html_content[:insert_pos] + explain_img + html_content[insert_pos:]
    
    # Add local approximation after mode parameter or intro section
    # Updated pattern to match the actual LIME file structure
    local_pattern = r'(<h2>🔷 1\. Что такое LIME\?</h2>.*?</ul>\s*</div>)'
    local_img = create_img_tag(illustrations['lime_local_approximation'], 
                               'LIME: Локальная аппроксимация', '95%')
    match = re.search(local_pattern, html_content, re.DOTALL)
    if match:
        insert_pos = match.end(1)
        html_content = html_content[:insert_pos] + local_img + html_content[insert_pos:]
    
    return html_content

def add_illustrations_to_feature_importance(html_content, illustrations):
    """Add illustrations to Feature Importance cheatsheet."""
    # Add comparison chart after introduction or permutation importance section
    comparison_pattern = r'(<h2>🔷 1\..*?</h2>.*?</div>)'
    comparison_img = create_img_tag(illustrations['feature_importance_comparison'], 
                                    'Сравнение методов Feature Importance', '95%')
    match = re.search(comparison_pattern, html_content, re.DOTALL)
    if match:
        insert_pos = match.end(1)
        html_content = html_content[:insert_pos] + comparison_img + html_content[insert_pos:]
    
    # Add performance drop visualization
    drop_pattern = r'(permutation_importance.*?importances_mean.*?</code></pre>\s*</div>)'
    drop_img = create_img_tag(illustrations['feature_importance_drop'], 
                              'Падение качества при удалении признака', '90%')
    match = re.search(drop_pattern, html_content, re.DOTALL | re.IGNORECASE)
    if match:
        insert_pos = match.end(1)
        html_content = html_content[:insert_pos] + drop_img + html_content[insert_pos:]
    
    return html_content

def add_illustrations_to_partial_dependence(html_content, illustrations):
    """Add illustrations to Partial Dependence cheatsheet."""
    # Add 1D PDP after 1D PDP section
    pdp1d_pattern = r'(<h2>🔷 3\. 1D PDP</h2>.*?</ul>\s*</div>)'
    pdp1d_img = create_img_tag(illustrations['pdp_1d'], 
                               'Partial Dependence Plots (1D)', '95%')
    match = re.search(pdp1d_pattern, html_content, re.DOTALL)
    if match:
        insert_pos = match.end(1)
        html_content = html_content[:insert_pos] + pdp1d_img + html_content[insert_pos:]
    
    # Add 2D PDP after 2D PDP section
    pdp2d_pattern = r'(<h2>🔷 4\. 2D PDP</h2>.*?</ul>\s*</div>)'
    pdp2d_img = create_img_tag(illustrations['pdp_2d'], 
                               'Partial Dependence Plot (2D)', '90%')
    match = re.search(pdp2d_pattern, html_content, re.DOTALL)
    if match:
        insert_pos = match.end(1)
        html_content = html_content[:insert_pos] + pdp2d_img + html_content[insert_pos:]
    
    # Add ICE plots after ICE section
    ice_pattern = r'(<h2>🔷 5\. ICE plots</h2>.*?</ul>\s*</div>)'
    ice_img = create_img_tag(illustrations['ice_plots'], 
                            'Individual Conditional Expectation plots', '90%')
    match = re.search(ice_pattern, html_content, re.DOTALL)
    if match:
        insert_pos = match.end(1)
        html_content = html_content[:insert_pos] + ice_img + html_content[insert_pos:]
    
    return html_content

def add_illustrations_to_integrated_gradients(html_content, illustrations):
    """Add illustrations to Integrated Gradients cheatsheet."""
    # Add path visualization after formula section
    path_pattern = r'(<h2>🔷 2\. Формула</h2>.*?</ul>\s*</div>)'
    path_img = create_img_tag(illustrations['integrated_gradients_path'], 
                              'Integrated Gradients: Путь интерполяции', '95%')
    match = re.search(path_pattern, html_content, re.DOTALL)
    if match:
        insert_pos = match.end(1)
        html_content = html_content[:insert_pos] + path_img + html_content[insert_pos:]
    
    # Add attribution heatmap after implementation section
    attribution_pattern = r'(<h2>🔷 3\. Реализация</h2>.*?</code></pre>\s*</div>)'
    attribution_img = create_img_tag(illustrations['integrated_gradients_attribution'], 
                                     'Integrated Gradients: Attribution heatmap', '90%')
    match = re.search(attribution_pattern, html_content, re.DOTALL)
    if match:
        insert_pos = match.end(1)
        html_content = html_content[:insert_pos] + attribution_img + html_content[insert_pos:]
    
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
    """Main function to add illustrations to all interpretability cheatsheets."""
    print("=" * 70)
    print("Adding matplotlib illustrations to interpretability cheatsheets")
    print("=" * 70)
    
    # Generate all illustrations
    print("\n1. Generating illustrations...")
    illustrations = generate_all_illustrations()
    
    # Process each HTML file
    print("\n2. Adding illustrations to HTML files...")
    
    files_to_process = [
        ('cheatsheets/explainable_ai_xai_cheatsheet.html', add_illustrations_to_xai),
        ('cheatsheets/shap_cheatsheet.html', add_illustrations_to_shap),
        ('cheatsheets/lime_cheatsheet.html', add_illustrations_to_lime),
        ('cheatsheets/feature_importance_cheatsheet.html', add_illustrations_to_feature_importance),
        ('cheatsheets/partial_dependence_cheatsheet.html', add_illustrations_to_partial_dependence),
        ('cheatsheets/integrated_gradients_cheatsheet.html', add_illustrations_to_integrated_gradients),
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
