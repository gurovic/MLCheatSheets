#!/usr/bin/env python3
"""
Add matplotlib illustrations to data processing cheatsheet HTML files.
"""

import re
from generate_data_processing_illustrations import generate_all_illustrations

def create_img_tag(base64_data, alt_text, width="100%"):
    """Create HTML img tag with base64 encoded image."""
    return f'''
    <div style="text-align: center; margin: 10px 0;">
      <img src="data:image/png;base64,{base64_data}" 
           alt="{alt_text}" 
           style="max-width: {width}; height: auto; border-radius: 4px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
    </div>
'''

def add_illustrations_to_preprocessing(html_content, illustrations):
    """Add illustrations to preprocessing cheatsheet."""
    # Add missing data strategies after the missing data table
    missing_pattern = r'(<h2>🔷 2\. Обработка пропусков</h2>.*?</table>\s*</div>)'
    missing_img = create_img_tag(illustrations['missing_data_strategies'], 
                                 'Стратегии заполнения пропущенных значений', '95%')
    match = re.search(missing_pattern, html_content, re.DOTALL)
    if match:
        insert_pos = match.end(1)
        html_content = html_content[:insert_pos] + missing_img + html_content[insert_pos:]
    
    # Add outlier detection after outlier code
    outlier_pattern = r'(<h2>🔷 3\. Обработка выбросов</h2>.*?</code></pre>\s*</div>)'
    outlier_img = create_img_tag(illustrations['outlier_detection'], 
                                 'Методы обнаружения выбросов', '95%')
    match = re.search(outlier_pattern, html_content, re.DOTALL)
    if match:
        insert_pos = match.end(1)
        html_content = html_content[:insert_pos] + outlier_img + html_content[insert_pos:]
    
    # Add encoding comparison after encoding section
    encoding_pattern = r'(<h2>🔷 4\. Кодирование категорий</h2>.*?y_encoded = le\.fit_transform\(y\)</code></pre>\s*</div>)'
    encoding_img = create_img_tag(illustrations['encoding_comparison'], 
                                  'Сравнение методов кодирования', '95%')
    match = re.search(encoding_pattern, html_content, re.DOTALL)
    if match:
        insert_pos = match.end(1)
        html_content = html_content[:insert_pos] + encoding_img + html_content[insert_pos:]
    
    return html_content

def add_illustrations_to_scaling(html_content, illustrations):
    """Add illustrations to scaling and normalization cheatsheet."""
    # Add scaler comparison at the beginning after the intro section
    intro_pattern = r'(<h2>🔷 1\. Суть</h2>.*?</div>)'
    scaler_img = create_img_tag(illustrations['scaler_comparison'], 
                                'Сравнение методов масштабирования', '95%')
    match = re.search(intro_pattern, html_content, re.DOTALL)
    if match:
        insert_pos = match.end(1)
        html_content = html_content[:insert_pos] + scaler_img + html_content[insert_pos:]
    
    # Add normalization effect visualization after StandardScaler section
    norm_pattern = r'(StandardScaler.*?fit_transform.*?</code></pre>\s*</div>)'
    norm_img = create_img_tag(illustrations['normalization_effect'], 
                              'Влияние нормализации на данные', '90%')
    match = re.search(norm_pattern, html_content, re.DOTALL)
    if match:
        insert_pos = match.end(1)
        html_content = html_content[:insert_pos] + norm_img + html_content[insert_pos:]
    
    return html_content

def add_illustrations_to_feature_engineering(html_content, illustrations):
    """Add illustrations to feature engineering cheatsheet."""
    # Add polynomial features visualization after intro
    intro_pattern = r'(<h2>🔷 1\. Суть</h2>.*?</div>)'
    poly_img = create_img_tag(illustrations['polynomial_features'], 
                              'Полиномиальные признаки', '95%')
    match = re.search(intro_pattern, html_content, re.DOTALL)
    if match:
        insert_pos = match.end(1)
        html_content = html_content[:insert_pos] + poly_img + html_content[insert_pos:]
    
    # Add feature interaction visualization after polynomial section
    poly_pattern = r'(PolynomialFeatures.*?fit_transform.*?</code></pre>\s*</div>)'
    interaction_img = create_img_tag(illustrations['feature_interaction'], 
                                     'Взаимодействие признаков', '95%')
    match = re.search(poly_pattern, html_content, re.DOTALL)
    if match:
        insert_pos = match.end(1)
        html_content = html_content[:insert_pos] + interaction_img + html_content[insert_pos:]
    
    return html_content

def add_illustrations_to_feature_selection(html_content, illustrations):
    """Add illustrations to feature selection cheatsheet."""
    # Add feature importance after Random Forest example
    importance_pattern = r'(from sklearn\.ensemble import RandomForestClassifier.*?feature_importances_.*?</code></pre>\s*</div>)'
    importance_img = create_img_tag(illustrations['feature_importance'], 
                                    'Важность признаков', '90%')
    match = re.search(importance_pattern, html_content, re.DOTALL)
    if match:
        insert_pos = match.end(1)
        html_content = html_content[:insert_pos] + importance_img + html_content[insert_pos:]
    
    # Add correlation matrix
    corr_pattern = r'(<h2>🔷.*?[Кк]орреляция.*?</h2>)'
    corr_img = create_img_tag(illustrations['correlation_matrix'], 
                              'Матрица корреляции', '85%')
    match = re.search(corr_pattern, html_content, re.DOTALL | re.IGNORECASE)
    if match:
        insert_pos = match.end(1)
        html_content = html_content[:insert_pos] + corr_img + html_content[insert_pos:]
    
    # Add univariate selection
    univariate_pattern = r'(from sklearn\.feature_selection import SelectKBest.*?fit.*?</code></pre>\s*</div>)'
    univariate_img = create_img_tag(illustrations['univariate_selection'], 
                                    'Univariate Feature Selection', '90%')
    match = re.search(univariate_pattern, html_content, re.DOTALL)
    if match:
        insert_pos = match.end(1)
        html_content = html_content[:insert_pos] + univariate_img + html_content[insert_pos:]
    
    return html_content

def add_illustrations_to_dimensionality_reduction(html_content, illustrations):
    """Add illustrations to dimensionality reduction cheatsheet."""
    # Add PCA visualization after PCA code
    pca_pattern = r'(from sklearn\.decomposition import PCA.*?pca\.fit_transform.*?</code></pre>\s*</div>)'
    pca_img = create_img_tag(illustrations['pca_visualization'], 
                            'PCA снижение размерности', '95%')
    match = re.search(pca_pattern, html_content, re.DOTALL)
    if match:
        insert_pos = match.end(1)
        html_content = html_content[:insert_pos] + pca_img + html_content[insert_pos:]
    
    # Add explained variance
    variance_pattern = r'(<h2>🔷.*?[Оо]бъясненная.*?дисперсия.*?</h2>)'
    variance_img = create_img_tag(illustrations['pca_variance'], 
                                  'Объясненная дисперсия компонент', '95%')
    match = re.search(variance_pattern, html_content, re.DOTALL | re.IGNORECASE)
    if match:
        insert_pos = match.end(1)
        html_content = html_content[:insert_pos] + variance_img + html_content[insert_pos:]
    
    # Add t-SNE visualization
    tsne_pattern = r'(from sklearn\.manifold import TSNE.*?fit_transform.*?</code></pre>\s*</div>)'
    tsne_img = create_img_tag(illustrations['tsne_visualization'], 
                             't-SNE vs PCA', '95%')
    match = re.search(tsne_pattern, html_content, re.DOTALL)
    if match:
        insert_pos = match.end(1)
        html_content = html_content[:insert_pos] + tsne_img + html_content[insert_pos:]
    
    return html_content

def add_illustrations_to_imbalanced_data(html_content, illustrations):
    """Add illustrations to imbalanced data cheatsheet."""
    # Add class imbalance visualization
    imbalance_pattern = r'(<h2>🔷.*?[Пп]роблема.*?дисбаланса.*?</h2>)'
    imbalance_img = create_img_tag(illustrations['class_imbalance'], 
                                   'Проблема дисбаланса классов', '95%')
    match = re.search(imbalance_pattern, html_content, re.DOTALL | re.IGNORECASE)
    if match:
        insert_pos = match.end(1)
        html_content = html_content[:insert_pos] + imbalance_img + html_content[insert_pos:]
    
    # Add SMOTE visualization
    smote_pattern = r'(from imblearn\.over_sampling import SMOTE.*?fit_resample.*?</code></pre>\s*</div>)'
    smote_img = create_img_tag(illustrations['smote_visualization'], 
                               'SMOTE oversampling', '95%')
    match = re.search(smote_pattern, html_content, re.DOTALL)
    if match:
        insert_pos = match.end(1)
        html_content = html_content[:insert_pos] + smote_img + html_content[insert_pos:]
    
    # Add resampling strategies comparison
    resampling_pattern = r'(<h2>🔷.*?[Сс]тратегии.*?ресемплинга.*?</h2>)'
    resampling_img = create_img_tag(illustrations['resampling_strategies'], 
                                    'Сравнение стратегий ресемплинга', '90%')
    match = re.search(resampling_pattern, html_content, re.DOTALL | re.IGNORECASE)
    if match:
        insert_pos = match.end(1)
        html_content = html_content[:insert_pos] + resampling_img + html_content[insert_pos:]
    
    return html_content

def add_illustrations_to_data_augmentation(html_content, illustrations):
    """Add illustrations to data augmentation cheatsheet."""
    # Add augmentation examples after intro
    intro_pattern = r'(<h2>🔷 1\. Суть</h2>.*?</div>)'
    aug_examples_img = create_img_tag(illustrations['augmentation_examples'], 
                                      'Методы аугментации данных', '95%')
    match = re.search(intro_pattern, html_content, re.DOTALL)
    if match:
        insert_pos = match.end(1)
        html_content = html_content[:insert_pos] + aug_examples_img + html_content[insert_pos:]
    
    # Add distribution comparison after the basic techniques table
    table_pattern = r'(<h2>🔷 2\. Аугментация изображений.*?</table>\s*</div>)'
    aug_dist_img = create_img_tag(illustrations['augmentation_distribution'], 
                                  'Увеличение размера датасета', '95%')
    match = re.search(table_pattern, html_content, re.DOTALL)
    if match:
        insert_pos = match.end(1)
        html_content = html_content[:insert_pos] + aug_dist_img + html_content[insert_pos:]
    
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
    """Main function to add illustrations to all data processing cheatsheets."""
    print("=" * 70)
    print("Adding matplotlib illustrations to data processing cheatsheets")
    print("=" * 70)
    
    # Generate all illustrations
    print("\n1. Generating illustrations...")
    illustrations = generate_all_illustrations()
    
    # Process each HTML file
    print("\n2. Adding illustrations to HTML files...")
    
    files_to_process = [
        ('cheatsheets/preprocessing_cheatsheet.html', add_illustrations_to_preprocessing),
        ('cheatsheets/scaling_normalization_cheatsheet.html', add_illustrations_to_scaling),
        ('cheatsheets/feature_engineering_cheatsheet.html', add_illustrations_to_feature_engineering),
        ('cheatsheets/feature_selection_cheatsheet.html', add_illustrations_to_feature_selection),
        ('cheatsheets/dimensionality_reduction_cheatsheet.html', add_illustrations_to_dimensionality_reduction),
        ('cheatsheets/imbalanced_data_cheatsheet.html', add_illustrations_to_imbalanced_data),
        ('cheatsheets/data_augmentation_cheatsheet.html', add_illustrations_to_data_augmentation),
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
