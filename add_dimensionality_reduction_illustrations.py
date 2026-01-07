#!/usr/bin/env python3
"""
Add matplotlib illustrations to dimensionality reduction cheatsheet HTML files.
"""

import re
from generate_dimensionality_reduction_illustrations import generate_all_illustrations

def create_img_tag(base64_data, alt_text, width="100%"):
    """Create HTML img tag with base64 encoded image."""
    return f'''
    <div style="text-align: center; margin: 10px 0;">
      <img src="data:image/png;base64,{base64_data}" 
           alt="{alt_text}" 
           style="max-width: {width}; height: auto; border-radius: 4px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
    </div>
'''

def add_illustrations_to_pca(html_content, illustrations):
    """Add illustrations to PCA cheatsheet."""
    # Add scree plot after section 4 (Выбор количества компонент)
    pattern = r'(<h2>🔷 4\. Выбор количества компонент</h2>)'
    img = create_img_tag(illustrations['pca_scree_plot'], 'PCA Scree Plot', '95%')
    match = re.search(pattern, html_content)
    if match:
        insert_pos = match.end()
        html_content = html_content[:insert_pos] + img + html_content[insert_pos:]
    
    # Add 2D projection after section 5 (Визуализация в 2D)
    pattern = r'(<h2>🔷 5\. Визуализация в 2D</h2>)'
    img = create_img_tag(illustrations['pca_projection_2d'], 'PCA 2D проекция', '90%')
    match = re.search(pattern, html_content)
    if match:
        insert_pos = match.end()
        html_content = html_content[:insert_pos] + img + html_content[insert_pos:]
    
    # Add component visualization after section 1 (Суть)
    pattern = r'(</ul>\s*</div>\s*<div class="block">\s*<h2>🔷 2\. Базовый код</h2>)'
    img = create_img_tag(illustrations['pca_components'], 'PCA компоненты', '90%')
    match = re.search(pattern, html_content)
    if match:
        insert_pos = match.start()
        html_content = html_content[:insert_pos] + img + html_content[insert_pos:]
    
    return html_content

def add_illustrations_to_tsne(html_content, illustrations):
    """Add illustrations to t-SNE cheatsheet."""
    # Add perplexity comparison after section 4 (Perplexity)
    pattern = r'(<h2>🔷 4\. Perplexity — главный параметр</h2>)'
    img = create_img_tag(illustrations['tsne_perplexity'], 'Сравнение perplexity', '95%')
    match = re.search(pattern, html_content)
    if match:
        insert_pos = match.end()
        html_content = html_content[:insert_pos] + img + html_content[insert_pos:]
    
    # Add visualization after section 2 (Базовый код)
    pattern = r'(</code></pre>\s*</div>\s*<div class="block">\s*<h2>🔷 3\. Ключевые параметры</h2>)'
    img = create_img_tag(illustrations['tsne_visualization'], 't-SNE визуализация', '90%')
    match = re.search(pattern, html_content)
    if match:
        insert_pos = match.start()
        html_content = html_content[:insert_pos] + img + html_content[insert_pos:]
    
    # Add comparison with PCA after section 6 (PCA перед t-SNE)
    pattern = r'(</code></pre>\s*</div>\s*<div class="block">\s*<h2>🔷 7\. Для больших данных</h2>)'
    img = create_img_tag(illustrations['tsne_vs_pca'], 't-SNE vs PCA', '95%')
    match = re.search(pattern, html_content)
    if match:
        insert_pos = match.start()
        html_content = html_content[:insert_pos] + img + html_content[insert_pos:]
    
    return html_content

def add_illustrations_to_umap(html_content, illustrations):
    """Add illustrations to UMAP cheatsheet."""
    # Add n_neighbors comparison after section 4
    pattern = r'(<h2>🔷 4\. n_neighbors — главный параметр</h2>)'
    img = create_img_tag(illustrations['umap_comparison'], 'Сравнение n_neighbors', '95%')
    match = re.search(pattern, html_content)
    if match:
        insert_pos = match.end()
        html_content = html_content[:insert_pos] + img + html_content[insert_pos:]
    
    # Add visualization after section 2
    pattern = r'(</code></pre>\s*</div>\s*<div class="block">\s*<h2>🔷 3\. Ключевые параметры</h2>)'
    img = create_img_tag(illustrations['umap_visualization'], 'UMAP визуализация', '90%')
    match = re.search(pattern, html_content)
    if match:
        insert_pos = match.start()
        html_content = html_content[:insert_pos] + img + html_content[insert_pos:]
    
    # Add comparison with t-SNE at the end of section 1
    pattern = r'(</ul>\s*</div>\s*<div class="block">\s*<h2>🔷 2\. Установка и базовый код</h2>)'
    img = create_img_tag(illustrations['umap_vs_tsne'], 'UMAP vs t-SNE', '95%')
    match = re.search(pattern, html_content)
    if match:
        insert_pos = match.start()
        html_content = html_content[:insert_pos] + img + html_content[insert_pos:]
    
    return html_content

def add_illustrations_to_lda(html_content, illustrations):
    """Add illustrations to LDA cheatsheet."""
    # Add visualization after section 2 (Базовый код снижение размерности)
    pattern = r'(</code></pre>\s*</div>\s*<div class="block">\s*<h2>🔷 3\. Базовый код \(классификация\)</h2>)'
    img = create_img_tag(illustrations['lda_visualization'], 'LDA визуализация', '90%')
    match = re.search(pattern, html_content)
    if match:
        insert_pos = match.start()
        html_content = html_content[:insert_pos] + img + html_content[insert_pos:]
    
    # Add LDA vs PCA comparison after section 5
    pattern = r'(<h2>🔷 5\. LDA vs PCA</h2>)'
    img = create_img_tag(illustrations['lda_vs_pca'], 'LDA vs PCA', '95%')
    match = re.search(pattern, html_content)
    if match:
        insert_pos = match.end()
        html_content = html_content[:insert_pos] + img + html_content[insert_pos:]
    
    # Add decision boundaries after section 3
    pattern = r'(</code></pre>\s*</div>\s*<div class="block">\s*<h2>🔷 4\. Ключевые параметры</h2>)'
    img = create_img_tag(illustrations['lda_decision_boundaries'], 'LDA границы решений', '90%')
    match = re.search(pattern, html_content)
    if match:
        insert_pos = match.start()
        html_content = html_content[:insert_pos] + img + html_content[insert_pos:]
    
    return html_content

def add_illustrations_to_svd(html_content, illustrations):
    """Add illustrations to SVD cheatsheet."""
    # Add singular values plot after finding a relevant section
    pattern = r'(<h2>🔷 [23]\.[^<]*(?:сингулярн|компонент)[^<]*</h2>)'
    img = create_img_tag(illustrations['svd_singular_values'], 'Сингулярные значения', '95%')
    match = re.search(pattern, html_content, re.IGNORECASE)
    if match:
        insert_pos = match.end()
        html_content = html_content[:insert_pos] + img + html_content[insert_pos:]
    
    # Add matrix visualization after basic code section
    pattern = r'(</code></pre>\s*</div>\s*<div class="block">\s*<h2>🔷 3\.)'
    img = create_img_tag(illustrations['svd_matrix'], 'SVD декомпозиция матрицы', '95%')
    match = re.search(pattern, html_content)
    if match:
        insert_pos = match.start()
        html_content = html_content[:insert_pos] + img + html_content[insert_pos:]
    
    # Add reconstruction visualization
    pattern = r'(<h2>🔷 [456]\.[^<]*(?:реконструкци|восстановлен)[^<]*</h2>)'
    img = create_img_tag(illustrations['svd_reconstruction'], 'SVD реконструкция', '95%')
    match = re.search(pattern, html_content, re.IGNORECASE)
    if match:
        insert_pos = match.end()
        html_content = html_content[:insert_pos] + img + html_content[insert_pos:]
    
    return html_content

def add_illustrations_to_isomap(html_content, illustrations):
    """Add illustrations to Isomap cheatsheet."""
    # Add visualization after basic code
    pattern = r'(</code></pre>\s*</div>\s*<div class="block">\s*<h2>🔷 3\.)'
    img = create_img_tag(illustrations['isomap_visualization'], 'Isomap визуализация', '95%')
    match = re.search(pattern, html_content)
    if match:
        insert_pos = match.start()
        html_content = html_content[:insert_pos] + img + html_content[insert_pos:]
    
    # Add neighbors comparison
    pattern = r'(<h2>🔷 [34]\.[^<]*(?:параметр|neighbor)[^<]*</h2>)'
    img = create_img_tag(illustrations['isomap_neighbors'], 'Сравнение n_neighbors', '95%')
    match = re.search(pattern, html_content, re.IGNORECASE)
    if match:
        insert_pos = match.end()
        html_content = html_content[:insert_pos] + img + html_content[insert_pos:]
    
    return html_content

def add_illustrations_to_lle(html_content, illustrations):
    """Add illustrations to LLE cheatsheet."""
    # Add visualization after basic code
    pattern = r'(</code></pre>\s*</div>\s*<div class="block">\s*<h2>🔷 3\.)'
    img = create_img_tag(illustrations['lle_visualization'], 'LLE визуализация', '95%')
    match = re.search(pattern, html_content)
    if match:
        insert_pos = match.start()
        html_content = html_content[:insert_pos] + img + html_content[insert_pos:]
    
    # Add neighbors comparison
    pattern = r'(<h2>🔷 [34]\.[^<]*(?:параметр|neighbor)[^<]*</h2>)'
    img = create_img_tag(illustrations['lle_neighbors'], 'Сравнение n_neighbors', '95%')
    match = re.search(pattern, html_content, re.IGNORECASE)
    if match:
        insert_pos = match.end()
        html_content = html_content[:insert_pos] + img + html_content[insert_pos:]
    
    return html_content

def add_illustrations_to_mds(html_content, illustrations):
    """Add illustrations to MDS cheatsheet."""
    # Add visualization after basic code
    pattern = r'(</code></pre>\s*</div>\s*<div class="block">\s*<h2>🔷 3\.)'
    img = create_img_tag(illustrations['mds_visualization'], 'MDS визуализация', '90%')
    match = re.search(pattern, html_content)
    if match:
        insert_pos = match.start()
        html_content = html_content[:insert_pos] + img + html_content[insert_pos:]
    
    # Add stress plot
    pattern = r'(<h2>🔷 [34]\.[^<]*(?:стресс|stress|качеств)[^<]*</h2>)'
    img = create_img_tag(illustrations['mds_stress'], 'MDS стресс', '90%')
    match = re.search(pattern, html_content, re.IGNORECASE)
    if match:
        insert_pos = match.end()
        html_content = html_content[:insert_pos] + img + html_content[insert_pos:]
    
    # Add comparison with PCA
    pattern = r'(<h2>🔷 [5678]\.[^<]*(?:сравнени|vs|против)[^<]*</h2>)'
    img = create_img_tag(illustrations['mds_vs_pca'], 'MDS vs PCA', '95%')
    match = re.search(pattern, html_content, re.IGNORECASE)
    if match:
        insert_pos = match.end()
        html_content = html_content[:insert_pos] + img + html_content[insert_pos:]
    
    return html_content

def add_illustrations_to_manifold(html_content, illustrations):
    """Add illustrations to Manifold Learning cheatsheet."""
    # Add comparison at the beginning (after section 1)
    pattern = r'(</ul>\s*</div>\s*<div class="block">\s*<h2>🔷 2\.)'
    img = create_img_tag(illustrations['manifold_comparison'], 'Сравнение методов снижения размерности', '98%')
    match = re.search(pattern, html_content)
    if match:
        insert_pos = match.start()
        html_content = html_content[:insert_pos] + img + html_content[insert_pos:]
    
    # Add datasets comparison
    pattern = r'(<h2>🔷 [89]\.[^<]*(?:сравнени|датасет|данны)[^<]*</h2>)'
    img = create_img_tag(illustrations['manifold_datasets'], 'Методы на разных датасетах', '98%')
    match = re.search(pattern, html_content, re.IGNORECASE)
    if match:
        insert_pos = match.end()
        html_content = html_content[:insert_pos] + img + html_content[insert_pos:]
    
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
    """Main function to add illustrations to all dimensionality reduction cheatsheets."""
    print("=" * 70)
    print("Adding matplotlib illustrations to dimensionality reduction cheatsheets")
    print("=" * 70)
    
    # Generate all illustrations
    print("\n1. Generating illustrations...")
    illustrations = generate_all_illustrations()
    
    # Process each HTML file
    print("\n2. Adding illustrations to HTML files...")
    
    files_to_process = [
        ('cheatsheets/pca_cheatsheet.html', add_illustrations_to_pca),
        ('cheatsheets/tsne_cheatsheet.html', add_illustrations_to_tsne),
        ('cheatsheets/umap_cheatsheet.html', add_illustrations_to_umap),
        ('cheatsheets/lda_cheatsheet.html', add_illustrations_to_lda),
        ('cheatsheets/svd_cheatsheet.html', add_illustrations_to_svd),
        ('cheatsheets/isomap_cheatsheet.html', add_illustrations_to_isomap),
        ('cheatsheets/lle_cheatsheet.html', add_illustrations_to_lle),
        ('cheatsheets/mds_cheatsheet.html', add_illustrations_to_mds),
        ('cheatsheets/manifold_learning_cheatsheet.html', add_illustrations_to_manifold),
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
