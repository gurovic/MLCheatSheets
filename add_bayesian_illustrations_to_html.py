#!/usr/bin/env python3
"""
Add matplotlib illustrations to Bayesian methods cheatsheet HTML files.
"""

import re
from generate_bayesian_illustrations import generate_all_illustrations

def create_img_tag(base64_data, alt_text, width="95%"):
    """Create HTML img tag with base64 encoded image."""
    return f'''
    <div style="text-align: center; margin: 20px 0;">
      <img src="data:image/png;base64,{base64_data}" 
           alt="{alt_text}" 
           style="max-width: {width}; height: auto; border-radius: 4px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
    </div>
'''

def add_illustrations_to_bayesian_optimization(html_content, illustrations):
    """Add illustrations to Bayesian Optimization cheatsheet."""
    
    # Add BO process after section 4 "Как работает"
    pattern1 = r'(<p><strong>Acquisition Function</strong>: использует оба для выбора</p></div>)'
    img1 = create_img_tag(illustrations['bo_process'], 
                         'Процесс Bayesian Optimization', '100%')
    html_content = re.sub(pattern1, 
                         r'\1' + '\n' + img1, 
                         html_content, count=1)
    
    # Add acquisition functions comparison after section 5 "Acquisition Functions"
    pattern2 = r'(</table></div><div class="block"><h2>🔷 6\. Параметры оптимизации</h2>)'
    img2 = create_img_tag(illustrations['bo_acquisition'], 
                         'Сравнение Acquisition Functions', '100%')
    html_content = re.sub(pattern2, 
                         r'</table></div>\n' + img2 + r'\n<div class="block"><h2>🔷 6. Параметры оптимизации</h2>', 
                         html_content, count=1)
    
    # Add iterations after section 9 "Визуализация результатов"
    pattern3 = r'(plt\.show\(\)</code></pre></div>)'
    img3 = create_img_tag(illustrations['bo_iterations'], 
                         'Итерации Bayesian Optimization', '95%')
    html_content = re.sub(pattern3, 
                         r'\1' + '\n' + img3, 
                         html_content, count=1)
    
    return html_content

def add_illustrations_to_bayesian_neural_networks(html_content, illustrations):
    """Add illustrations to Bayesian Neural Networks cheatsheet."""
    
    # Add uncertainty comparison after section 2 "Обычные NN vs Байесовские"
    pattern1 = r'(</table></div><div class="block"><h2>🔷 3\. Математическая основа</h2>)'
    img1 = create_img_tag(illustrations['bnn_uncertainty'], 
                         'Сравнение: обычная NN vs Байесовская NN', '100%')
    html_content = re.sub(pattern1, 
                         r'</table></div>\n' + img1 + r'\n<div class="block"><h2>🔷 3. Математическая основа</h2>', 
                         html_content, count=1)
    
    # Add weight distributions after section 4 "Variational Inference"
    pattern2 = r'(↑ data fit\s+↑ regularization</code></pre></div>)'
    img2 = create_img_tag(illustrations['bnn_weights'], 
                         'Распределения весов в Байесовских нейронных сетях', '95%')
    html_content = re.sub(pattern2, 
                         r'\1' + '\n' + img2, 
                         html_content, count=1)
    
    # Add prediction samples after section 6 "Предсказание с uncertainty"
    pattern3 = r'(# Среднее и std</code></pre></div>)'
    img3 = create_img_tag(illustrations['bnn_samples'], 
                         'Множественные предсказания Байесовской NN', '95%')
    html_content = re.sub(pattern3, 
                         r'\1' + '\n' + img3, 
                         html_content, count=1)
    
    return html_content

def add_illustrations_to_bayesian_inference(html_content, illustrations):
    """Add illustrations to Bayesian Inference cheatsheet."""
    
    # Add prior-likelihood-posterior after section 2 "Формула Байеса"
    pattern1 = r'(Posterior ∝ Likelihood × Prior</code></pre>\s*</div>)'
    img1 = create_img_tag(illustrations['bi_posterior'], 
                         'Байесовский вывод: Prior → Likelihood → Posterior', '100%')
    html_content = re.sub(pattern1, 
                         r'\1\n' + img1, 
                         html_content, count=1)
    
    # Add conjugate priors after section 4 "Сопряженные распределения"  
    pattern2 = r'(</table>\s*</div>\s*<div class="block">\s*<h2>🔷 5\. Байесовская линейная регрессия</h2>)'
    img2 = create_img_tag(illustrations['bi_conjugate'], 
                         'Conjugate Priors: Beta-Binomial', '100%')
    html_content = re.sub(pattern2, 
                         r'</table>\n  </div>\n' + img2 + r'\n  <div class="block">\n    <h2>🔷 5. Байесовская линейная регрессия</h2>', 
                         html_content, count=1)
    
    # Add MCMC sampling - look for a PyMC or Stan section
    pattern3 = r'(</ul>\s*</div>\s*<div class="block">\s*<h2>🔷 10\. Библиотеки)'
    img3 = create_img_tag(illustrations['bi_mcmc'], 
                         'MCMC: семплирование из сложного распределения', '95%')
    html_content = re.sub(pattern3, 
                         r'</ul>\n  </div>\n' + img3 + r'\n  <div class="block">\n    <h2>🔷 10. Библиотеки', 
                         html_content, count=1)
    
    return html_content

def add_illustrations_to_gaussian_processes(html_content, illustrations):
    """Add illustrations to Gaussian Processes cheatsheet."""
    
    # Add GP regression after section 2 "Базовый пример регрессии"
    pattern1 = r'(print\(f"Log-marginal-likelihood: \{gp\.log_marginal_likelihood\(\):.3f\}"\)</code></pre>\s*</div>)'
    img1 = create_img_tag(illustrations['gp_regression'], 
                         'Gaussian Process: предсказание с неопределенностью', '100%')
    html_content = re.sub(pattern1, 
                         r'\1\n' + img1, 
                         html_content, count=1)
    
    # Add kernels comparison after kernel section table
    pattern2 = r'(</table>\s*</div>\s*<div class="block">\s*<h2>🔷 4\. Композиции kernels</h2>)'
    img2 = create_img_tag(illustrations['gp_kernels'], 
                         'Различные Kernel Functions в Gaussian Processes', '100%')
    html_content = re.sub(pattern2, 
                         r'</table>\n  </div>\n' + img2 + r'\n  <div class="block">\n    <h2>🔷 4. Композиции kernels</h2>', 
                         html_content, count=1)
    
    # Add hyperparameters effect after hyperparameters section
    pattern3 = r'(print\(gp\.kernel_\)</code></pre>\s*</div>\s*<div class="block">\s*<h2>🔷 6\. GP для классификации</h2>)'
    img3 = create_img_tag(illustrations['gp_hyperparams'], 
                         'Влияние Length Scale на Gaussian Process', '95%')
    html_content = re.sub(pattern3, 
                         r'print(gp.kernel_)</code></pre>\n  </div>\n' + img3 + r'\n  <div class="block">\n    <h2>🔷 6. GP для классификации</h2>', 
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
        
        # Check if any changes were made
        if modified_content == html_content:
            print(f"  ⚠ Warning: No changes made to {filepath} (patterns might not match)")
        
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
    """Main function to add illustrations to all Bayesian methods cheatsheets."""
    print("=" * 70)
    print("Adding matplotlib illustrations to Bayesian methods cheatsheets")
    print("=" * 70)
    
    # Generate all illustrations
    print("\n1. Generating illustrations...")
    illustrations = generate_all_illustrations()
    
    # Process each HTML file
    print("\n2. Adding illustrations to HTML files...")
    
    files_to_process = [
        ('cheatsheets/bayesian_optimization_cheatsheet.html', add_illustrations_to_bayesian_optimization),
        ('cheatsheets/bayesian_neural_networks_cheatsheet.html', add_illustrations_to_bayesian_neural_networks),
        ('cheatsheets/bayesian_inference_cheatsheet.html', add_illustrations_to_bayesian_inference),
        ('cheatsheets/gaussian_processes_cheatsheet.html', add_illustrations_to_gaussian_processes),
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
