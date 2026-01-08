#!/usr/bin/env python3
"""
Add matplotlib illustrations to recommender systems cheatsheet HTML files.
"""

import re
from generate_recommender_illustrations import generate_all_illustrations

def create_img_tag(base64_data, alt_text, width="100%"):
    """Create HTML img tag with base64 encoded image."""
    return f'''
    <div style="text-align: center; margin: 10px 0;">
      <img src="data:image/png;base64,{base64_data}" 
           alt="{alt_text}" 
           style="max-width: {width}; height: auto; border-radius: 4px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
    </div>
'''

def add_illustrations_to_collaborative_filtering(html_content, illustrations):
    """Add illustrations to collaborative filtering cheatsheet."""
    
    # 1. Add user-item matrix after the first block explaining what CF is
    pattern1 = r'(<h2>🔷 1\. Что такое коллаборативная фильтрация\?</h2>.*?</ul>\s*</div>)'
    img1 = create_img_tag(illustrations['user_item_matrix'], 
                         'Матрица рейтингов пользователь-товар', '90%')
    html_content = re.sub(pattern1, r'\1' + img1, html_content, count=1, flags=re.DOTALL)
    
    # 2. Add similarity matrices after the types table
    pattern2 = r'(</table>\s*</div>\s*<div class="block">\s*<h2>🔷 3\. User-based CF</h2>)'
    img2 = create_img_tag(illustrations['similarity_matrices'],
                         'Матрицы схожести пользователей и товаров', '95%')
    html_content = re.sub(pattern2, img2 + r'\n  <div class="block">\n    <h2>🔷 3. User-based CF</h2>', 
                         html_content, count=1, flags=re.DOTALL)
    
    # 3. Add CF prediction example after User-based CF code
    pattern3 = r'(print\(f"Predicted rating: \{predicted_rating:.2f\}"\)</code></pre>\s*</div>)'
    img3 = create_img_tag(illustrations['cf_prediction'],
                         'Процесс предсказания в User-based CF', '90%')
    match = re.search(pattern3, html_content, re.DOTALL)
    if match:
        insert_pos = match.end(1)
        html_content = html_content[:insert_pos] + img3 + html_content[insert_pos:]
    
    return html_content

def add_illustrations_to_matrix_factorization(html_content, illustrations):
    """Add illustrations to matrix factorization SVD cheatsheet."""
    
    # 1. Add SVD decomposition diagram after the mathematics section
    pattern1 = r'(<h2>🔷 2\. Математика SVD</h2>.*?</ul>\s*</div>)'
    img1 = create_img_tag(illustrations['svd_decomposition'],
                         'SVD разложение матрицы рейтингов', '95%')
    html_content = re.sub(pattern1, r'\1' + img1, html_content, count=1, flags=re.DOTALL)
    
    # 2. Add SVD example after the basic implementation code
    pattern2 = r'(print\("Исходная матрица:"\).*?print\(R_pred\.round\(2\)\)</code></pre>\s*</div>)'
    img2 = create_img_tag(illustrations['svd_example'],
                         'Пример SVD: исходная, восстановленная и ошибка', '95%')
    match = re.search(pattern2, html_content, re.DOTALL)
    if match:
        insert_pos = match.end(1)
        html_content = html_content[:insert_pos] + img2 + html_content[insert_pos:]
    
    # 3. Add k selection plot after the optimization section
    pattern3 = r'(<h2>🔷 6\. Оптимизация</h2>.*?print\(f"k=\{k\}: RMSE=\{rmse:.4f\}"\)</code></pre>\s*</div>)'
    img3 = create_img_tag(illustrations['svd_k_selection'],
                         'Выбор оптимального количества факторов k', '90%')
    html_content = re.sub(pattern3, r'\1' + img3, html_content, count=1, flags=re.DOTALL)
    
    return html_content

def add_illustrations_to_content_based(html_content, illustrations):
    """Add illustrations to content-based filtering cheatsheet."""
    
    # 1. Add comparison chart after the comparison table
    pattern1 = r'(<h2>🔷 2\. Vs коллаборативная фильтрация</h2>.*?</table>\s*</div>)'
    img1 = create_img_tag(illustrations['collab_vs_content'],
                         'Сравнение коллаборативной и контентной фильтрации', '90%')
    html_content = re.sub(pattern1, r'\1' + img1, html_content, count=1, flags=re.DOTALL)
    
    # 2. Add TF-IDF visualization after the basic example code
    pattern2 = r'(# Рекомендации.*?print\(get_recommendations\(\'Terminator\'\)\).*?# Output: Avatar, True Lies</code></pre>\s*</div>)'
    img2 = create_img_tag(illustrations['tfidf_example'],
                         'TF-IDF матрица признаков', '90%')
    html_content = re.sub(pattern2, r'\1' + img2, html_content, count=1, flags=re.DOTALL)
    
    # 3. Add content similarity matrix after TF-IDF illustration
    img3 = create_img_tag(illustrations['content_similarity'],
                         'Матрица схожести по контенту', '85%')
    # Insert right after the TF-IDF image we just added
    pattern3 = r'(data:image/png;base64,[A-Za-z0-9+/=]+"\s+alt="TF-IDF матрица признаков".*?</div>\s*</div>)'
    match = re.search(pattern3, html_content, re.DOTALL)
    if match:
        insert_pos = match.end(1)
        html_content = html_content[:insert_pos] + img3 + html_content[insert_pos:]
    
    # 4. Add user profile visualization after the user profile creation section
    pattern4 = r'(<h2>🔷 5\. Создание профиля пользователя</h2>.*?user_profile = tfidf_matrix\[user_likes\]\.mean\(axis=0\)</code></pre>)'
    img4 = create_img_tag(illustrations['user_profile'],
                         'Построение профиля пользователя', '90%')
    match = re.search(pattern4, html_content, re.DOTALL)
    if match:
        insert_pos = match.end(0)
        html_content = html_content[:insert_pos] + img4 + html_content[insert_pos:]
    
    return html_content

def process_html_file(filepath, add_illustrations_func, illustrations):
    """Process a single HTML file to add illustrations."""
    print(f"Processing {filepath}...")
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        # Check if illustrations already exist
        if 'data:image/png;base64,' in html_content:
            print(f"  ⚠ Warning: File already contains base64 images. Skipping to avoid duplicates.")
            print(f"    If you want to re-add images, please remove existing ones first.")
            return False
        
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
    """Main function to add illustrations to all recommender systems cheatsheets."""
    print("=" * 80)
    print("Adding matplotlib illustrations to recommender systems cheatsheets")
    print("=" * 80)
    
    # Generate all illustrations
    print("\n1. Generating illustrations...")
    illustrations = generate_all_illustrations()
    
    # Process each HTML file
    print("\n2. Adding illustrations to HTML files...")
    
    files_to_process = [
        ('cheatsheets/collaborative_filtering_cheatsheet.html', 
         add_illustrations_to_collaborative_filtering),
        ('cheatsheets/matrix_factorization_svd_cheatsheet.html', 
         add_illustrations_to_matrix_factorization),
        ('cheatsheets/content_based_filtering_cheatsheet.html', 
         add_illustrations_to_content_based),
    ]
    
    success_count = 0
    for filepath, add_func in files_to_process:
        if process_html_file(filepath, add_func, illustrations):
            success_count += 1
    
    print("\n" + "=" * 80)
    print(f"Completed: {success_count}/{len(files_to_process)} files successfully updated")
    print("=" * 80)

if __name__ == '__main__':
    main()
