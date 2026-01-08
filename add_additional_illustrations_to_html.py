#!/usr/bin/env python3
"""
Add matplotlib illustrations to Additional section cheatsheet HTML files.
"""

import re
from generate_additional_illustrations import generate_all_illustrations

def create_img_tag(base64_data, alt_text, width="100%"):
    """Create HTML img tag with base64 encoded image."""
    return f'''
    <div style="text-align: center; margin: 10px 0;">
      <img src="data:image/png;base64,{base64_data}" 
           alt="{alt_text}" 
           style="max-width: {width}; height: auto; border-radius: 4px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
    </div>
'''

def add_illustrations_to_neural_network_training_dynamics(html_content, illustrations):
    """Add illustrations to neural_network_training_dynamics_cheatsheet.html."""
    
    # Add loss landscape after landscape section
    landscape_pattern = r'(<h2>🔷 2\. Ландшафт функции потерь</h2>.*?</ul>)'
    landscape_img = create_img_tag(illustrations['nn_loss_landscape'], 
                                   'Ландшафт функции потерь', '95%')
    match = re.search(landscape_pattern, html_content, re.DOTALL)
    if match:
        insert_pos = match.end(1)
        html_content = html_content[:insert_pos] + landscape_img + html_content[insert_pos:]
        print("  ✓ Added loss landscape")
    
    # Add learning rate schedules after scheduling section
    lr_pattern = r'(</code></pre>\s*</div>\s*<blockquote>⚡ Правильный schedule может ускорить сходимость)'
    lr_img = create_img_tag(illustrations['nn_lr_schedules'], 
                           'Стратегии Learning Rate Scheduling', '95%')
    match = re.search(lr_pattern, html_content, re.DOTALL)
    if match:
        insert_pos = match.end(1)
        html_content = html_content[:insert_pos] + lr_img + html_content[insert_pos:]
        print("  ✓ Added learning rate schedules")
    
    # Add training phases after phases section
    phases_pattern = r'(<h2>🔷 4\. Фазы обучения</h2>.*?</ul>\s*</div>)'
    phases_img = create_img_tag(illustrations['nn_training_phases'], 
                                'Фазы обучения нейросети', '95%')
    match = re.search(phases_pattern, html_content, re.DOTALL)
    if match:
        insert_pos = match.end(1)
        html_content = html_content[:insert_pos] + phases_img + html_content[insert_pos:]
        print("  ✓ Added training phases")
    
    # Add sharp vs flat minima after landscape or generalization section
    minima_pattern = r'(Плоские минимумы обычно обобщают лучше, чем острые\.</p>)'
    minima_img = create_img_tag(illustrations['nn_sharp_flat_minima'], 
                                'Острые vs плоские минимумы', '95%')
    match = re.search(minima_pattern, html_content, re.DOTALL)
    if match:
        insert_pos = match.end(1)
        html_content = html_content[:insert_pos] + minima_img + html_content[insert_pos:]
        print("  ✓ Added sharp vs flat minima")
    
    return html_content

def add_illustrations_to_information_theory(html_content, illustrations):
    """Add illustrations to information_theory_ml_cheatsheet.html."""
    
    # Add entropy visualization after entropy section
    entropy_pattern = r'(<h2>🔷 2\. Энтропия.*?</code></pre>\s*</div>)'
    entropy_img = create_img_tag(illustrations['it_entropy'], 
                                 'Визуализация энтропии', '95%')
    match = re.search(entropy_pattern, html_content, re.DOTALL)
    if match:
        insert_pos = match.end(1)
        html_content = html_content[:insert_pos] + entropy_img + html_content[insert_pos:]
        print("  ✓ Added entropy visualization")
    
    # Add KL divergence after KL section
    kl_pattern = r'(<h2>🔷 5\. Расстояние Кульбака-Лейблера.*?</code></pre>\s*</div>)'
    kl_img = create_img_tag(illustrations['it_kl_divergence'], 
                           'KL дивергенция', '95%')
    match = re.search(kl_pattern, html_content, re.DOTALL)
    if match:
        insert_pos = match.end(1)
        html_content = html_content[:insert_pos] + kl_img + html_content[insert_pos:]
        print("  ✓ Added KL divergence")
    
    # Add mutual information after mutual information section
    mi_pattern = r'(<h2>🔷 6\. Взаимная информация.*?</code></pre>\s*</div>)'
    mi_img = create_img_tag(illustrations['it_mutual_info'], 
                           'Взаимная информация', '95%')
    match = re.search(mi_pattern, html_content, re.DOTALL)
    if match:
        insert_pos = match.end(1)
        html_content = html_content[:insert_pos] + mi_img + html_content[insert_pos:]
        print("  ✓ Added mutual information")
    
    return html_content

def add_illustrations_to_probability_theory(html_content, illustrations):
    """Add illustrations to probability_theory_cheatsheet.html."""
    
    # Add probability distributions after distributions section
    dist_pattern = r'(<h2>🔷 6\. Распределения.*?</ul>\s*</div>)'
    dist_img = create_img_tag(illustrations['prob_distributions'], 
                             'Основные распределения вероятностей', '95%')
    match = re.search(dist_pattern, html_content, re.DOTALL)
    if match:
        insert_pos = match.end(1)
        html_content = html_content[:insert_pos] + dist_img + html_content[insert_pos:]
        print("  ✓ Added probability distributions")
    
    # Add Bayes theorem after Bayes section
    bayes_pattern = r'(<h2>🔷 4\. Теорема Байеса.*?</ul>\s*</div>)'
    bayes_img = create_img_tag(illustrations['prob_bayes'], 
                               'Теорема Байеса', '95%')
    match = re.search(bayes_pattern, html_content, re.DOTALL)
    if match:
        insert_pos = match.end(1)
        html_content = html_content[:insert_pos] + bayes_img + html_content[insert_pos:]
        print("  ✓ Added Bayes theorem")
    
    # Add Central Limit Theorem after ML application section
    clt_pattern = r'(<h2>🔷 8\. Применение в ML.*?</ul>\s*</div>)'
    clt_img = create_img_tag(illustrations['prob_clt'], 
                            'Центральная предельная теорема', '95%')
    match = re.search(clt_pattern, html_content, re.DOTALL)
    if match:
        insert_pos = match.end(1)
        html_content = html_content[:insert_pos] + clt_img + html_content[insert_pos:]
        print("  ✓ Added Central Limit Theorem")
    
    return html_content

def add_illustrations_to_gradient_checkpointing(html_content, illustrations):
    """Add illustrations to gradient_checkpointing_cheatsheet.html."""
    
    # Add memory comparison after intro section
    memory_pattern = r'(<h2>🔷 1\. Суть Gradient Checkpointing.*?</ul>\s*</div>)'
    memory_img = create_img_tag(illustrations['gc_memory'], 
                                'Сравнение использования памяти', '95%')
    match = re.search(memory_pattern, html_content, re.DOTALL)
    if match:
        insert_pos = match.end(1)
        html_content = html_content[:insert_pos] + memory_img + html_content[insert_pos:]
        print("  ✓ Added memory comparison")
    
    # Add trade-off visualization after basic code
    tradeoff_pattern = r'(<h2>🔷 2\. Как работает.*?</ul>\s*</div>)'
    tradeoff_img = create_img_tag(illustrations['gc_trade_off'], 
                                  'Компромисс память vs вычисления', '95%')
    match = re.search(tradeoff_pattern, html_content, re.DOTALL)
    if match:
        insert_pos = match.end(1)
        html_content = html_content[:insert_pos] + tradeoff_img + html_content[insert_pos:]
        print("  ✓ Added trade-off visualization")
    
    # Add layers visualization after implementation section
    layers_pattern = r'(</code></pre>\s*</div>\s*<blockquote>.*?памяти.*?</blockquote>)'
    layers_img = create_img_tag(illustrations['gc_layers'], 
                                'Checkpoint слои в нейросети', '95%')
    match = re.search(layers_pattern, html_content, re.DOTALL)
    if match:
        insert_pos = match.end(1)
        html_content = html_content[:insert_pos] + layers_img + html_content[insert_pos:]
        print("  ✓ Added layers visualization")
    
    return html_content

def add_illustrations_to_mixed_precision(html_content, illustrations):
    """Add illustrations to mixed_precision_training_cheatsheet.html."""
    
    # Add precision formats after formats section
    formats_pattern = r'(<h2>🔷 2\. Форматы чисел.*?</ul>\s*</div>)'
    formats_img = create_img_tag(illustrations['mp_formats'], 
                                 'Форматы представления чисел', '95%')
    match = re.search(formats_pattern, html_content, re.DOTALL)
    if match:
        insert_pos = match.end(1)
        html_content = html_content[:insert_pos] + formats_img + html_content[insert_pos:]
        print("  ✓ Added precision formats")
    
    # Add speedup comparison after practical tips section
    speedup_pattern = r'(<h2>🔷 8\. Практические советы.*?</ul>\s*</div>)'
    speedup_img = create_img_tag(illustrations['mp_speedup'], 
                                 'Ускорение с Mixed Precision', '95%')
    match = re.search(speedup_pattern, html_content, re.DOTALL)
    if match:
        insert_pos = match.end(1)
        html_content = html_content[:insert_pos] + speedup_img + html_content[insert_pos:]
        print("  ✓ Added speedup comparison")
    
    # Add loss scaling after gradient scaling section
    scaling_pattern = r'(<h2>🔷 5\. Gradient Scaling.*?</code></pre>\s*</div>)'
    scaling_img = create_img_tag(illustrations['mp_loss_scaling'], 
                                 'Loss Scaling', '95%')
    match = re.search(scaling_pattern, html_content, re.DOTALL)
    if match:
        insert_pos = match.end(1)
        html_content = html_content[:insert_pos] + scaling_img + html_content[insert_pos:]
        print("  ✓ Added loss scaling")
    
    return html_content

def add_illustrations_to_model_compression(html_content, illustrations):
    """Add illustrations to model_compression_cheatsheet.html."""
    
    # Add techniques comparison after intro section
    techniques_pattern = r'(<h2>🔷 1\. Зачем сжимать модели\?.*?</ul>\s*</div>)'
    techniques_img = create_img_tag(illustrations['mc_techniques'], 
                                    'Сравнение методов сжатия', '95%')
    match = re.search(techniques_pattern, html_content, re.DOTALL)
    if match:
        insert_pos = match.end(1)
        html_content = html_content[:insert_pos] + techniques_img + html_content[insert_pos:]
        print("  ✓ Added techniques comparison")
    
    # Add pruning visualization after pruning section
    pruning_pattern = r'(<h2>🔷 4\. Прунинг \(Pruning\).*?</code></pre>\s*</div>)'
    pruning_img = create_img_tag(illustrations['mc_pruning'], 
                                 'Визуализация Pruning', '95%')
    match = re.search(pruning_pattern, html_content, re.DOTALL)
    if match:
        insert_pos = match.end(1)
        html_content = html_content[:insert_pos] + pruning_img + html_content[insert_pos:]
        print("  ✓ Added pruning visualization")
    
    # Add quantization visualization after quantization section
    quant_pattern = r'(<h2>🔷 2\. Квантизация.*?</ul>\s*</div>)'
    quant_img = create_img_tag(illustrations['mc_quantization'], 
                               'Визуализация квантизации', '95%')
    match = re.search(quant_pattern, html_content, re.DOTALL)
    if match:
        insert_pos = match.end(1)
        html_content = html_content[:insert_pos] + quant_img + html_content[insert_pos:]
        print("  ✓ Added quantization visualization")
    
    # Add knowledge distillation after distillation section
    distill_pattern = r'(<h2>🔷 6\. Knowledge Distillation.*?</code></pre>\s*</div>)'
    distill_img = create_img_tag(illustrations['mc_distillation'], 
                                 'Knowledge Distillation', '95%')
    match = re.search(distill_pattern, html_content, re.DOTALL)
    if match:
        insert_pos = match.end(1)
        html_content = html_content[:insert_pos] + distill_img + html_content[insert_pos:]
        print("  ✓ Added knowledge distillation")
    
    return html_content

def process_html_file(filepath, add_illustrations_func, illustrations):
    """Process a single HTML file."""
    print(f"\nProcessing {filepath}...")
    
    with open(filepath, 'r', encoding='utf-8') as f:
        html_content = f.read()
    
    # Add illustrations
    modified_content = add_illustrations_func(html_content, illustrations)
    
    # Write back
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(modified_content)
    
    print(f"✓ Successfully updated {filepath}")

def main():
    """Main function to add all illustrations."""
    print("=" * 70)
    print("Adding Matplotlib Illustrations to Additional Section Cheatsheets")
    print("=" * 70)
    
    # Generate all illustrations
    print("\n📊 Generating all illustrations...")
    illustrations = generate_all_illustrations()
    print(f"✓ Generated {len(illustrations)} illustrations\n")
    
    # Define files and their processing functions
    files_to_process = [
        ('cheatsheets/neural_network_training_dynamics_cheatsheet.html', 
         add_illustrations_to_neural_network_training_dynamics),
        ('cheatsheets/information_theory_ml_cheatsheet.html', 
         add_illustrations_to_information_theory),
        ('cheatsheets/probability_theory_cheatsheet.html', 
         add_illustrations_to_probability_theory),
        ('cheatsheets/gradient_checkpointing_cheatsheet.html', 
         add_illustrations_to_gradient_checkpointing),
        ('cheatsheets/mixed_precision_training_cheatsheet.html', 
         add_illustrations_to_mixed_precision),
        ('cheatsheets/model_compression_cheatsheet.html', 
         add_illustrations_to_model_compression),
    ]
    
    # Process each file
    for filepath, process_func in files_to_process:
        try:
            process_html_file(filepath, process_func, illustrations)
        except Exception as e:
            print(f"✗ Error processing {filepath}: {e}")
    
    print("\n" + "=" * 70)
    print("✓ All illustrations have been added successfully!")
    print("=" * 70)

if __name__ == '__main__':
    main()
