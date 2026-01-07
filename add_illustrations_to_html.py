#!/usr/bin/env python3
"""
Add matplotlib illustrations to clustering cheatsheet HTML files.
"""

import re
from generate_clustering_illustrations import generate_all_illustrations

def create_img_tag(base64_data, alt_text, width="100%"):
    """Create HTML img tag with base64 encoded image."""
    return f'''
    <div style="text-align: center; margin: 10px 0;">
      <img src="data:image/png;base64,{base64_data}" 
           alt="{alt_text}" 
           style="max-width: {width}; height: auto; border-radius: 4px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
    </div>
'''

def add_illustrations_to_kmeans(html_content, illustrations):
    """Add illustrations to K-means cheatsheet."""
    # Add elbow method illustration after the elbow method code block
    elbow_pattern = r'(# Ищем "локоть" - точку, где улучшение замедляется</code></pre>\s*</div>)'
    elbow_img = create_img_tag(illustrations['kmeans_elbow'], 'Метод локтя для K-means', '90%')
    html_content = re.sub(elbow_pattern, r'\1' + elbow_img, html_content, count=1)
    
    # Add clustering visualization after the visualization code section
    viz_pattern = r'(plt\.show\(\)</code></pre>\s*</div>\s*<div class="block">\s*<h2>🔷 8\.)'
    viz_img = create_img_tag(illustrations['kmeans_clustering'], 'K-means кластеризация', '90%')
    html_content = re.sub(viz_pattern, viz_img + r'\n  <div class="block">\n    <h2>🔷 8.', html_content, count=1)
    
    # Add silhouette plot after silhouette code
    silhouette_pattern = r'(plt\.show\(\)</code></pre>\s*</div>\s*<div class="block">\s*<h2>🔷 7\. Визуализация результатов)'
    silhouette_img = create_img_tag(illustrations['kmeans_silhouette'], 'Silhouette анализ', '90%')
    html_content = re.sub(silhouette_pattern, silhouette_img + r'\n  <div class="block">\n    <h2>🔷 7. Визуализация результатов', html_content, count=1)
    
    return html_content

def add_illustrations_to_hierarchical(html_content, illustrations):
    """Add illustrations to hierarchical clustering cheatsheet."""
    # Add dendrogram after dendrogram code
    dendro_pattern = r'(plt\.show\(\)</code></pre>\s*</div>)'
    dendro_img = create_img_tag(illustrations['hierarchical_dendrogram'], 'Дендрограмма', '95%')
    # Find first occurrence after dendrogram code
    match = re.search(r'dendrogram\([^)]+\).*?' + dendro_pattern, html_content, re.DOTALL)
    if match:
        insert_pos = match.end(1)
        html_content = html_content[:insert_pos] + dendro_img + html_content[insert_pos:]
    
    # Add clustering visualization
    viz_pattern = r'(AgglomerativeClustering.*?fit_predict.*?</code></pre>\s*</div>)'
    viz_img = create_img_tag(illustrations['hierarchical_clustering'], 'Иерархическая кластеризация', '90%')
    match = re.search(viz_pattern, html_content, re.DOTALL)
    if match:
        insert_pos = match.end(1)
        html_content = html_content[:insert_pos] + viz_img + html_content[insert_pos:]
    
    return html_content

def add_illustrations_to_dbscan(html_content, illustrations):
    """Add illustrations to DBSCAN cheatsheet."""
    # Add clustering visualization after DBSCAN fit code
    viz_pattern = r'(dbscan\.fit\(X_scaled\).*?</code></pre>\s*</div>)'
    viz_img = create_img_tag(illustrations['dbscan_clustering'], 'DBSCAN кластеризация', '90%')
    match = re.search(viz_pattern, html_content, re.DOTALL)
    if match:
        insert_pos = match.end(1)
        html_content = html_content[:insert_pos] + viz_img + html_content[insert_pos:]
    
    # Add parameter comparison
    param_pattern = r'(<h2>🔷.*?Подбор параметров.*?</h2>)'
    param_img = create_img_tag(illustrations['dbscan_parameters'], 'Влияние параметров DBSCAN', '95%')
    match = re.search(param_pattern, html_content, re.DOTALL)
    if match:
        insert_pos = match.end(1)
        html_content = html_content[:insert_pos] + param_img + html_content[insert_pos:]
    
    return html_content

def add_illustrations_to_gmm(html_content, illustrations):
    """Add illustrations to GMM cheatsheet."""
    # Add GMM clustering visualization
    viz_pattern = r'(GaussianMixture\(.*?\).*?gmm\.fit\(X\).*?</code></pre>\s*</div>)'
    viz_img = create_img_tag(illustrations['gmm_clustering'], 'GMM кластеризация с вероятностными контурами', '90%')
    match = re.search(viz_pattern, html_content, re.DOTALL)
    if match:
        insert_pos = match.end(1)
        html_content = html_content[:insert_pos] + viz_img + html_content[insert_pos:]
    
    # Add covariance type comparison
    cov_pattern = r'(<h2>🔷.*?Типы ковариационных матриц.*?</h2>)'
    cov_img = create_img_tag(illustrations['gmm_comparison'], 'Сравнение типов ковариационных матриц', '95%')
    match = re.search(cov_pattern, html_content, re.DOTALL)
    if match:
        insert_pos = match.end(1)
        html_content = html_content[:insert_pos] + cov_img + html_content[insert_pos:]
    
    return html_content

def add_illustrations_to_spectral(html_content, illustrations):
    """Add illustrations to spectral clustering cheatsheet."""
    # Add comparison with K-means
    viz_pattern = r'(<h2>🔷.*?Пример.*?</h2>)'
    viz_img = create_img_tag(illustrations['spectral_clustering'], 'Spectral vs K-means на полулуниях', '95%')
    match = re.search(viz_pattern, html_content, re.DOTALL)
    if match:
        insert_pos = match.end(1)
        html_content = html_content[:insert_pos] + viz_img + html_content[insert_pos:]
    
    # Add circles example
    circles_pattern = r'(SpectralClustering\(.*?\).*?fit_predict.*?</code></pre>\s*</div>)'
    circles_img = create_img_tag(illustrations['spectral_circles'], 'Spectral clustering на концентрических кругах', '90%')
    match = re.search(circles_pattern, html_content, re.DOTALL)
    if match:
        insert_pos = match.end(1)
        html_content = html_content[:insert_pos] + circles_img + html_content[insert_pos:]
    
    return html_content

def add_illustrations_to_mean_shift(html_content, illustrations):
    """Add illustrations to mean shift cheatsheet."""
    # Add clustering visualization
    viz_pattern = r'(MeanShift\(.*?\).*?ms\.fit\(X\).*?</code></pre>\s*</div>)'
    viz_img = create_img_tag(illustrations['mean_shift_clustering'], 'Mean Shift кластеризация', '90%')
    match = re.search(viz_pattern, html_content, re.DOTALL)
    if match:
        insert_pos = match.end(1)
        html_content = html_content[:insert_pos] + viz_img + html_content[insert_pos:]
    
    # Add bandwidth comparison
    bw_pattern = r'(<h2>🔷.*?[Вв]ыбор.*?bandwidth.*?</h2>)'
    bw_img = create_img_tag(illustrations['mean_shift_bandwidth'], 'Влияние параметра bandwidth', '95%')
    match = re.search(bw_pattern, html_content, re.DOTALL | re.IGNORECASE)
    if match:
        insert_pos = match.end(1)
        html_content = html_content[:insert_pos] + bw_img + html_content[insert_pos:]
    
    return html_content

def add_illustrations_to_affinity_propagation(html_content, illustrations):
    """Add illustrations to affinity propagation cheatsheet."""
    # Add clustering visualization
    viz_pattern = r'(AffinityPropagation\(.*?\).*?ap\.fit\(X\).*?</code></pre>\s*</div>)'
    viz_img = create_img_tag(illustrations['affinity_propagation_clustering'], 'Affinity Propagation кластеризация', '90%')
    match = re.search(viz_pattern, html_content, re.DOTALL)
    if match:
        insert_pos = match.end(1)
        html_content = html_content[:insert_pos] + viz_img + html_content[insert_pos:]
    
    # Add preference parameter comparison
    pref_pattern = r'(<h2>🔷.*?[Пп]араметр.*?preference.*?</h2>)'
    pref_img = create_img_tag(illustrations['affinity_propagation_preference'], 'Влияние параметра preference', '95%')
    match = re.search(pref_pattern, html_content, re.DOTALL | re.IGNORECASE)
    if match:
        insert_pos = match.end(1)
        html_content = html_content[:insert_pos] + pref_img + html_content[insert_pos:]
    
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
        return False

def main():
    """Main function to add illustrations to all clustering cheatsheets."""
    print("=" * 70)
    print("Adding matplotlib illustrations to clustering cheatsheets")
    print("=" * 70)
    
    # Generate all illustrations
    print("\n1. Generating illustrations...")
    illustrations = generate_all_illustrations()
    
    # Process each HTML file
    print("\n2. Adding illustrations to HTML files...")
    
    files_to_process = [
        ('cheatsheets/kmeans_cheatsheet.html', add_illustrations_to_kmeans),
        ('cheatsheets/hierarchical_clustering_cheatsheet.html', add_illustrations_to_hierarchical),
        ('cheatsheets/dbscan_cheatsheet.html', add_illustrations_to_dbscan),
        ('cheatsheets/gmm_cheatsheet.html', add_illustrations_to_gmm),
        ('cheatsheets/spectral_clustering_cheatsheet.html', add_illustrations_to_spectral),
        ('cheatsheets/mean_shift_cheatsheet.html', add_illustrations_to_mean_shift),
        ('cheatsheets/affinity_propagation_cheatsheet.html', add_illustrations_to_affinity_propagation),
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
