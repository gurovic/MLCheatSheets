#!/usr/bin/env python3
"""
Add matplotlib illustrations to CNN and Pooling cheatsheet HTML files.
"""

import re
from generate_cnn_illustrations import generate_all_illustrations

def create_img_tag(base64_data, alt_text, width="100%"):
    """Create HTML img tag with base64 encoded image."""
    return f'''
    <div style="text-align: center; margin: 10px 0;">
      <img src="data:image/png;base64,{base64_data}" 
           alt="{alt_text}" 
           style="max-width: {width}; height: auto; border-radius: 4px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
    </div>
'''

def add_illustrations_to_cnn_basics(html_content, illustrations):
    """Add illustrations to CNN basics cheatsheet.
    
    Note: Uses regex patterns for HTML manipulation. While BeautifulSoup would be more robust,
    regex is sufficient here since we control the HTML structure and patterns are well-defined.
    """
    
    # Add convolution operation after the main layers section
    conv_pattern = r'(</table>\s*</div>\s*<div class="block">\s*<h2>🔷 3\. Простая CNN \(PyTorch\))'
    conv_img = create_img_tag(illustrations['cnn_convolution'], 
                             'Операция свертки (Convolution)', '95%')
    html_content = re.sub(conv_pattern, 
                         r'</table>\n  </div>\n\n' + conv_img + r'\n  <div class="block">\n    <h2>🔷 3. Простая CNN (PyTorch)', 
                         html_content, count=1)
    
    # Add feature maps after the TensorFlow CNN code
    feature_pattern = r'(\)</code></pre>\s*</div>\s*<div class="block">\s*<h2>🔷 5\. Параметры свертки)'
    feature_img = create_img_tag(illustrations['cnn_feature_maps'], 
                                'Различные фильтры извлекают различные признаки', '95%')
    html_content = re.sub(feature_pattern, 
                         r')</code></pre>\n  </div>\n\n' + feature_img + r'\n  <div class="block">\n    <h2>🔷 5. Параметры свертки', 
                         html_content, count=1)
    
    # Add architecture diagram after the parameters table
    arch_pattern = r'(</table>\s*</div>\s*<div class="block">\s*<h2>🔷 6\. Типы пулинга)'
    arch_img = create_img_tag(illustrations['cnn_architecture'], 
                             'Типичная архитектура CNN', '100%')
    html_content = re.sub(arch_pattern, 
                         r'</table>\n  </div>\n\n' + arch_img + r'\n  <div class="block">\n    <h2>🔷 6. Типы пулинга', 
                         html_content, count=1)
    
    return html_content

def add_illustrations_to_pooling(html_content, illustrations):
    """Add illustrations to pooling layers cheatsheet."""
    
    # Add pooling comparison after the Average Pooling code
    comparison_pattern = r'(4  5  6  7</code></pre>\s*</div>\s*<div class="block">\s*<h2>🔷 4\. Global Pooling)'
    comparison_img = create_img_tag(illustrations['pooling_comparison'], 
                                   'Сравнение Max Pooling и Average Pooling', '95%')
    html_content = re.sub(comparison_pattern, 
                         r'4  5  6  7</code></pre>\n  </div>\n\n' + comparison_img + r'\n  <div class="block">\n    <h2>🔷 4. Global Pooling', 
                         html_content, count=1)
    
    # Add global pooling visualization after the Global Pooling code
    global_pattern = r'(# - Инвариантность к размеру входа</code></pre>\s*</div>\s*<div class="block">\s*<h2>🔷 5\. Сравнение типов)'
    global_img = create_img_tag(illustrations['pooling_global'], 
                               'Global Average Pooling', '95%')
    html_content = re.sub(global_pattern, 
                         r'# - Инвариантность к размеру входа</code></pre>\n  </div>\n\n' + global_img + r'\n  <div class="block">\n    <h2>🔷 5. Сравнение типов', 
                         html_content, count=1)
    
    return html_content

def add_illustrations_to_1d_3d_cnn(html_content, illustrations):
    """Add illustrations to 1D and 3D CNN cheatsheet."""
    
    # Add 1D convolution after the 1D CNN Keras/TensorFlow code
    conv1d_pattern = r'(model\.summary\(\)</code></pre>\s*</div>\s*<div class="block">\s*<h2>🔷 6\. Dilated \(Atrous\) Convolutions)'
    conv1d_img = create_img_tag(illustrations['1d_convolution'], 
                               '1D Convolution для временных рядов', '95%')
    html_content = re.sub(conv1d_pattern, 
                         r'model.summary()</code></pre>\n  </div>\n\n' + conv1d_img + r'\n  <div class="block">\n    <h2>🔷 6. Dilated (Atrous) Convolutions', 
                         html_content, count=1)
    
    # Add 3D convolution visualization after 3D CNN PyTorch code
    conv3d_pattern = r'(print\(output\.shape\)  # \(2, 400\)</code></pre>\s*</div>\s*<div class="block">\s*<h2>🔷 9\. 3D-CNN: Применения)'
    conv3d_img = create_img_tag(illustrations['3d_convolution'], 
                               '3D Convolution для видео', '95%')
    html_content = re.sub(conv3d_pattern, 
                         r'print(output.shape)  # (2, 400)</code></pre>\n  </div>\n\n' + conv3d_img + r'\n  <div class="block">\n    <h2>🔷 9. 3D-CNN: Применения', 
                         html_content, count=1)
    
    # Add receptive field comparison after the comparison table
    receptive_pattern = r'(</table>\s*</div>\s*<div class="block">\s*<h2>🔷 14\. Оптимизация 3D-CNN)'
    receptive_img = create_img_tag(illustrations['receptive_field'], 
                                  'Сравнение Receptive Fields', '95%')
    html_content = re.sub(receptive_pattern, 
                         r'</table>\n  </div>\n\n' + receptive_img + r'\n  <div class="block">\n    <h2>🔷 14. Оптимизация 3D-CNN', 
                         html_content, count=1)
    
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
    """Main function to add illustrations to all CNN and Pooling cheatsheets."""
    print("=" * 70)
    print("Adding matplotlib illustrations to CNN and Pooling cheatsheets")
    print("=" * 70)
    
    # Generate all illustrations
    print("\n1. Generating illustrations...")
    illustrations = generate_all_illustrations()
    
    # Process each HTML file
    print("\n2. Adding illustrations to HTML files...")
    
    files_to_process = [
        ('cheatsheets/cnn_basics_cheatsheet.html', add_illustrations_to_cnn_basics),
        ('cheatsheets/pooling_layers_cheatsheet.html', add_illustrations_to_pooling),
        ('cheatsheets/1d_3d_cnn_cheatsheet.html', add_illustrations_to_1d_3d_cnn),
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
