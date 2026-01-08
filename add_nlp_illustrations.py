#!/usr/bin/env python3
"""
Add matplotlib illustrations to NLP cheatsheet HTML files.
"""

import re
from generate_nlp_illustrations import generate_all_illustrations

def create_img_tag(base64_data, alt_text, width="100%"):
    """Create HTML img tag with base64 encoded image."""
    return f'''
    <div style="text-align: center; margin: 10px 0;">
      <img src="data:image/png;base64,{base64_data}" 
           alt="{alt_text}" 
           style="max-width: {width}; height: auto; border-radius: 4px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
    </div>
'''

def add_illustrations_to_transformer_nlp(html_content, illustrations):
    """Add illustrations to transformer_nlp_cheatsheet.html."""
    
    # Add transformer architecture after the "Базовая архитектура" section
    pattern1 = r'(<h2>🔷 2\. Базовая архитектура</h2>.*?</ul>\s*</div>)'
    img1 = create_img_tag(illustrations['transformer_architecture'], 
                         'Transformer Architecture', '95%')
    html_content = re.sub(pattern1, r'\1' + img1, html_content, count=1, flags=re.DOTALL)
    
    # Add transformer layers after the "Self-Attention механизм" section
    pattern2 = r'(<h2>🔷 3\. Self-Attention механизм</h2>.*?</ul>\s*</div>)'
    img2 = create_img_tag(illustrations['transformer_layers'], 
                         'Transformer Layer Flow', '90%')
    html_content = re.sub(pattern2, r'\1' + img2, html_content, count=1, flags=re.DOTALL)
    
    return html_content

def add_illustrations_to_attention_mechanism(html_content, illustrations):
    """Add illustrations to attention_mechanism_cheatsheet.html."""
    
    # Add attention mechanism visualization after "Суть Attention" section
    pattern1 = r'(<h2>🔷 1\. Суть Attention</h2>.*?</ul></div>)'
    img1 = create_img_tag(illustrations['attention_mechanism'], 
                         'Self-Attention Weights', '90%')
    html_content = re.sub(pattern1, r'\1' + img1, html_content, count=1, flags=re.DOTALL)
    
    # Add scaled dot product after "Математика" section
    pattern2 = r'(<h2>🔷 2\. Математика.*?</h2>.*?Attention\(Q, K, V\) = softmax\(QKᵀ / √dₖ\) · V</p></div>)'
    img2 = create_img_tag(illustrations['scaled_dot_product'], 
                         'Scaled Dot-Product Attention', '90%')
    html_content = re.sub(pattern2, r'\1' + img2, html_content, count=1, flags=re.DOTALL)
    
    # Add multi-head attention after the multi-head attention code section
    pattern3 = r'(<h2>🔷 5\. Multi-Head Attention</h2>.*?return self\.W_o\(output\)</code></pre></div>)'
    img3 = create_img_tag(illustrations['multi_head_attention'], 
                         'Multi-Head Attention', '95%')
    html_content = re.sub(pattern3, r'\1' + img3, html_content, count=1, flags=re.DOTALL)
    
    return html_content

def add_illustrations_to_tokenization(html_content, illustrations):
    """Add illustrations to tokenization_cheatsheet.html."""
    
    # Add tokenization comparison after "Виды токенизации" section
    pattern1 = r'(<h2>🔷 2\. Виды токенизации</h2>.*?</table>\s*</div>)'
    img1 = create_img_tag(illustrations['tokenization_comparison'], 
                         'Tokenization Methods Comparison', '95%')
    html_content = re.sub(pattern1, r'\1' + img1, html_content, count=1, flags=re.DOTALL)
    
    # Add BPE process after "BPE (Byte Pair Encoding)" section
    pattern2 = r'(<h2>🔷 3\. BPE \(Byte Pair Encoding\)</h2>.*?</code></pre>\s*</div>)'
    img2 = create_img_tag(illustrations['bpe_process'], 
                         'BPE Algorithm Process', '95%')
    html_content = re.sub(pattern2, r'\1' + img2, html_content, count=1, flags=re.DOTALL)
    
    # Add subword tokenization comparison after "Обработка OOV" section
    pattern3 = r'(<h2>🔷 12\. Обработка OOV \(Out-of-Vocabulary\)</h2>.*?</code></pre>\s*</div>)'
    img3 = create_img_tag(illustrations['subword_tokenization'], 
                         'Subword vs Word-level Tokenization', '95%')
    html_content = re.sub(pattern3, r'\1' + img3, html_content, count=1, flags=re.DOTALL)
    
    return html_content

def add_illustrations_to_word_embeddings(html_content, illustrations):
    """Add illustrations to word_embeddings_cheatsheet.html."""
    
    # Add word embeddings space after "Концепция" section
    pattern1 = r'(<h2>🔷 1\. Концепция</h2>.*?</ul></div>)'
    img1 = create_img_tag(illustrations['word_embeddings_space'], 
                         'Word Embeddings in 2D Space', '90%')
    html_content = re.sub(pattern1, r'\1' + img1, html_content, count=1, flags=re.DOTALL)
    
    # Add Word2Vec architecture after "Word2Vec" section
    pattern2 = r'(<h2>🔷 2\. Word2Vec</h2>.*?</ul></div>)'
    img2 = create_img_tag(illustrations['word2vec_architecture'], 
                         'Word2Vec Architecture (CBOW vs Skip-gram)', '95%')
    html_content = re.sub(pattern2, r'\1' + img2, html_content, count=1, flags=re.DOTALL)
    
    # Add word analogies after "Использование" section
    pattern3 = r'(<h2>🔷 6\. Использование</h2>.*?</ul></div>)'
    img3 = create_img_tag(illustrations['word_analogies'], 
                         'Word Analogies with Embeddings', '90%')
    html_content = re.sub(pattern3, r'\1' + img3, html_content, count=1, flags=re.DOTALL)
    
    # Add embedding similarity after "Сравнение методов" section
    pattern4 = r'(<h2>🔷 8\. Сравнение методов</h2>.*?</ul></div>)'
    img4 = create_img_tag(illustrations['embedding_similarity'], 
                         'Word Embedding Similarity Matrix', '90%')
    html_content = re.sub(pattern4, r'\1' + img4, html_content, count=1, flags=re.DOTALL)
    
    return html_content

def process_html_file(filepath, add_illustrations_func, illustrations):
    """Process a single HTML file to add illustrations."""
    print(f"Processing {filepath}...")
    
    # Check if file exists
    import os
    if not os.path.exists(filepath):
        print(f"  ✗ File not found: {filepath}")
        return False
    
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
    """Main function to add illustrations to all NLP cheatsheets."""
    print("=" * 70)
    print("Adding matplotlib illustrations to NLP cheatsheets")
    print("=" * 70)
    
    # Generate all illustrations
    print("\n1. Generating illustrations...")
    illustrations = generate_all_illustrations()
    
    # Process each HTML file
    print("\n2. Adding illustrations to HTML files...")
    
    files_to_process = [
        ('cheatsheets/transformer_nlp_cheatsheet.html', add_illustrations_to_transformer_nlp),
        ('cheatsheets/attention_mechanism_cheatsheet.html', add_illustrations_to_attention_mechanism),
        ('cheatsheets/tokenization_cheatsheet.html', add_illustrations_to_tokenization),
        ('cheatsheets/word_embeddings_cheatsheet.html', add_illustrations_to_word_embeddings),
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
