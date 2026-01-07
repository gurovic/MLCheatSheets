#!/usr/bin/env python3
"""
Add matplotlib illustrations to optimization and learning cheatsheet HTML files.
"""

import re
from generate_optimization_illustrations import generate_all_illustrations

def create_img_tag(base64_data, alt_text, width="100%"):
    """Create HTML img tag with base64 encoded image."""
    return f'''
    <div style="text-align: center; margin: 10px 0;">
      <img src="data:image/png;base64,{base64_data}" 
           alt="{alt_text}" 
           style="max-width: {width}; height: auto; border-radius: 4px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
    </div>
'''

# ============================================================================
# GRADIENT DESCENT
# ============================================================================

def add_illustrations_to_gradient_descent(html_content, illustrations):
    """Add illustrations to gradient descent cheatsheet."""
    # Add convergence visualization after visualization section
    viz_pattern = r'(<h2>🔷 10\. Визуализация сходимости</h2>)'
    viz_img1 = create_img_tag(illustrations['gd_convergence'], 
                              'Сходимость градиентного спуска с разными learning rates', '95%')
    viz_img2 = create_img_tag(illustrations['gd_types'], 
                              'Сравнение типов градиентного спуска', '90%')
    html_content = re.sub(viz_pattern, r'\1' + viz_img1 + viz_img2, html_content, count=1)
    
    return html_content

# ============================================================================
# BACKPROPAGATION
# ============================================================================

def add_illustrations_to_backpropagation(html_content, illustrations):
    """Add illustrations to backpropagation cheatsheet."""
    # Add computational graph early in the document
    pattern = r'(<h2>🔷 2\.[^<]*</h2>)'
    img1 = create_img_tag(illustrations['backprop_graph'], 
                         'Вычислительный граф с forward и backward pass', '95%')
    match = re.search(pattern, html_content)
    if match:
        insert_pos = match.end(1)
        html_content = html_content[:insert_pos] + img1 + html_content[insert_pos:]
    
    # Add gradient flow visualization
    pattern = r'(<h2>🔷[^<]*Градиент[^<]*</h2>)'
    img2 = create_img_tag(illustrations['gradient_flow'], 
                         'Поток градиента через слои', '90%')
    matches = list(re.finditer(pattern, html_content, re.IGNORECASE))
    if matches:
        match = matches[0]
        insert_pos = match.end(1)
        html_content = html_content[:insert_pos] + img2 + html_content[insert_pos:]
    
    return html_content

# ============================================================================
# OPTIMIZERS
# ============================================================================

def add_illustrations_to_optimizers(html_content, illustrations):
    """Add illustrations to optimizers cheatsheet."""
    # Add optimizer comparison after main section
    pattern = r'(<h2>🔷[^<]*(Сравнение|сравнение)[^<]*</h2>)'
    img1 = create_img_tag(illustrations['optimizers_comparison'], 
                         'Сравнение оптимизаторов: SGD, Momentum, RMSprop, Adam', '95%')
    matches = list(re.finditer(pattern, html_content))
    if matches:
        match = matches[0]
        insert_pos = match.end(1)
        html_content = html_content[:insert_pos] + img1 + html_content[insert_pos:]
    else:
        # Try to find any early section
        pattern = r'(<h2>🔷 2\.[^<]*</h2>)'
        match = re.search(pattern, html_content)
        if match:
            insert_pos = match.end(1)
            html_content = html_content[:insert_pos] + img1 + html_content[insert_pos:]
    
    return html_content

def add_illustrations_to_optimizers_advanced(html_content, illustrations):
    """Add illustrations to optimizers advanced cheatsheet."""
    # Add momentum visualization
    pattern = r'(<h2>🔷[^<]*Momentum[^<]*</h2>)'
    img1 = create_img_tag(illustrations['momentum_effect'], 
                         'Эффект Momentum на сходимость', '95%')
    match = re.search(pattern, html_content, re.IGNORECASE)
    if match:
        insert_pos = match.end(1)
        html_content = html_content[:insert_pos] + img1 + html_content[insert_pos:]
    
    # Add optimizer comparison
    pattern = r'(<h2>🔷 2\.[^<]*</h2>)'
    img2 = create_img_tag(illustrations['optimizers_comparison'], 
                         'Сравнение продвинутых оптимизаторов', '95%')
    match = re.search(pattern, html_content)
    if match:
        insert_pos = match.end(1)
        html_content = html_content[:insert_pos] + img2 + html_content[insert_pos:]
    
    return html_content

# ============================================================================
# LEARNING RATE SCHEDULING
# ============================================================================

def add_illustrations_to_learning_rate_scheduling(html_content, illustrations):
    """Add illustrations to learning rate scheduling cheatsheet."""
    # Add LR schedules visualization
    pattern = r'(<h2>🔷[^<]*(Типы|типы|Стратег|стратег)[^<]*</h2>)'
    img1 = create_img_tag(illustrations['lr_schedules'], 
                         'Стратегии изменения learning rate', '95%')
    matches = list(re.finditer(pattern, html_content))
    if matches:
        match = matches[0]
        insert_pos = match.end(1)
        html_content = html_content[:insert_pos] + img1 + html_content[insert_pos:]
    else:
        # Try early section
        pattern = r'(<h2>🔷 2\.[^<]*</h2>)'
        match = re.search(pattern, html_content)
        if match:
            insert_pos = match.end(1)
            html_content = html_content[:insert_pos] + img1 + html_content[insert_pos:]
    
    # Add warmup schedule
    pattern = r'(<h2>🔷[^<]*Warmup[^<]*</h2>)'
    img2 = create_img_tag(illustrations['warmup_schedule'], 
                         'Warmup с последующим Decay', '90%')
    match = re.search(pattern, html_content, re.IGNORECASE)
    if match:
        insert_pos = match.end(1)
        html_content = html_content[:insert_pos] + img2 + html_content[insert_pos:]
    
    return html_content

# ============================================================================
# BATCH NORMALIZATION
# ============================================================================

def add_illustrations_to_batch_normalization(html_content, illustrations):
    """Add illustrations to batch normalization cheatsheet."""
    # Add distribution normalization
    pattern = r'(<h2>🔷[^<]*(Принцип|принцип|Как работ|как работ)[^<]*</h2>)'
    img1 = create_img_tag(illustrations['batchnorm_distribution'], 
                         'Эффект Batch Normalization на распределения', '95%')
    matches = list(re.finditer(pattern, html_content))
    if matches:
        match = matches[0]
        insert_pos = match.end(1)
        html_content = html_content[:insert_pos] + img1 + html_content[insert_pos:]
    else:
        # Try early section
        pattern = r'(<h2>🔷 2\.[^<]*</h2>)'
        match = re.search(pattern, html_content)
        if match:
            insert_pos = match.end(1)
            html_content = html_content[:insert_pos] + img1 + html_content[insert_pos:]
    
    # Add covariate shift
    pattern = r'(<h2>🔷[^<]*Internal Covariate Shift[^<]*</h2>)'
    img2 = create_img_tag(illustrations['covariate_shift'], 
                         'Internal Covariate Shift: с и без Batch Normalization', '95%')
    match = re.search(pattern, html_content, re.IGNORECASE)
    if match:
        insert_pos = match.end(1)
        html_content = html_content[:insert_pos] + img2 + html_content[insert_pos:]
    
    return html_content

# ============================================================================
# DROPOUT
# ============================================================================

def add_illustrations_to_dropout(html_content, illustrations):
    """Add illustrations to dropout regularization cheatsheet."""
    # Add dropout visualization
    pattern = r'(<h2>🔷[^<]*(Принцип|принцип|Как работ|как работ)[^<]*</h2>)'
    img1 = create_img_tag(illustrations['dropout_visualization'], 
                         'Dropout: отключение нейронов', '95%')
    matches = list(re.finditer(pattern, html_content))
    if matches:
        match = matches[0]
        insert_pos = match.end(1)
        html_content = html_content[:insert_pos] + img1 + html_content[insert_pos:]
    else:
        # Try early section
        pattern = r'(<h2>🔷 2\.[^<]*</h2>)'
        match = re.search(pattern, html_content)
        if match:
            insert_pos = match.end(1)
            html_content = html_content[:insert_pos] + img1 + html_content[insert_pos:]
    
    # Add dropout effect on training
    pattern = r'(<h2>🔷[^<]*(Эффект|эффект|Преимущ|преимущ)[^<]*</h2>)'
    img2 = create_img_tag(illustrations['dropout_effect'], 
                         'Эффект Dropout на обобщающую способность', '90%')
    matches = list(re.finditer(pattern, html_content))
    if matches:
        match = matches[0]
        insert_pos = match.end(1)
        html_content = html_content[:insert_pos] + img2 + html_content[insert_pos:]
    
    return html_content

# ============================================================================
# REGULARIZATION METHODS
# ============================================================================

def add_illustrations_to_regularization_methods(html_content, illustrations):
    """Add illustrations to regularization methods cheatsheet."""
    # Add regularization effect
    pattern = r'(<h2>🔷[^<]*(L1 и L2|L1.*L2)[^<]*</h2>)'
    img1 = create_img_tag(illustrations['regularization_effect'], 
                         'Эффект L1 и L2 регуляризации', '95%')
    matches = list(re.finditer(pattern, html_content, re.IGNORECASE))
    if matches:
        match = matches[0]
        insert_pos = match.end(1)
        html_content = html_content[:insert_pos] + img1 + html_content[insert_pos:]
    else:
        # Try early section
        pattern = r'(<h2>🔷 2\.[^<]*</h2>)'
        match = re.search(pattern, html_content)
        if match:
            insert_pos = match.end(1)
            html_content = html_content[:insert_pos] + img1 + html_content[insert_pos:]
    
    # Add L1 vs L2 comparison
    pattern = r'(<h2>🔷[^<]*(Сравнение|сравнение|Визуал|визуал)[^<]*</h2>)'
    img2 = create_img_tag(illustrations['l1_l2_comparison'], 
                         'Геометрическое сравнение L1, L2 и Elastic Net', '95%')
    matches = list(re.finditer(pattern, html_content))
    if matches:
        match = matches[0]
        insert_pos = match.end(1)
        html_content = html_content[:insert_pos] + img2 + html_content[insert_pos:]
    
    return html_content

# ============================================================================
# EARLY STOPPING
# ============================================================================

def add_illustrations_to_early_stopping(html_content, illustrations):
    """Add illustrations to early stopping cheatsheet."""
    # Add early stopping visualization
    pattern = r'(<h2>🔷[^<]*(Принцип|принцип|Как работ|как работ)[^<]*</h2>)'
    img1 = create_img_tag(illustrations['early_stopping'], 
                         'Early Stopping: остановка при ухудшении validation loss', '95%')
    matches = list(re.finditer(pattern, html_content))
    if matches:
        match = matches[0]
        insert_pos = match.end(1)
        html_content = html_content[:insert_pos] + img1 + html_content[insert_pos:]
    else:
        # Try early section
        pattern = r'(<h2>🔷 2\.[^<]*</h2>)'
        match = re.search(pattern, html_content)
        if match:
            insert_pos = match.end(1)
            html_content = html_content[:insert_pos] + img1 + html_content[insert_pos:]
    
    # Add patience illustration
    pattern = r'(<h2>🔷[^<]*Patience[^<]*</h2>)'
    img2 = create_img_tag(illustrations['patience_illustration'], 
                         'Patience: терпеливое ожидание улучшения', '90%')
    match = re.search(pattern, html_content, re.IGNORECASE)
    if match:
        insert_pos = match.end(1)
        html_content = html_content[:insert_pos] + img2 + html_content[insert_pos:]
    
    return html_content

# ============================================================================
# VANISHING GRADIENT
# ============================================================================

def add_illustrations_to_vanishing_gradient(html_content, illustrations):
    """Add illustrations to vanishing gradient cheatsheet."""
    # Add vanishing gradient problem
    pattern = r'(<h2>🔷[^<]*(Проблема|проблема|Суть|суть)[^<]*</h2>)'
    img1 = create_img_tag(illustrations['vanishing_gradient'], 
                         'Проблема затухающего и взрывающегося градиента', '95%')
    matches = list(re.finditer(pattern, html_content))
    if matches:
        match = matches[0]
        insert_pos = match.end(1)
        html_content = html_content[:insert_pos] + img1 + html_content[insert_pos:]
    else:
        # Try early section
        pattern = r'(<h2>🔷 2\.[^<]*</h2>)'
        match = re.search(pattern, html_content)
        if match:
            insert_pos = match.end(1)
            html_content = html_content[:insert_pos] + img1 + html_content[insert_pos:]
    
    # Add activation comparison
    pattern = r'(<h2>🔷[^<]*(Активац|активац|Функци|функци)[^<]*</h2>)'
    img2 = create_img_tag(illustrations['activation_comparison'], 
                         'Сравнение функций активации и их градиентов', '95%')
    matches = list(re.finditer(pattern, html_content))
    if matches:
        match = matches[0]
        insert_pos = match.end(1)
        html_content = html_content[:insert_pos] + img2 + html_content[insert_pos:]
    
    return html_content

# ============================================================================
# WEIGHT INITIALIZATION
# ============================================================================

def add_illustrations_to_weight_initialization(html_content, illustrations):
    """Add illustrations to weight initialization cheatsheet."""
    # Add weight initialization comparison
    pattern = r'(<h2>🔷[^<]*(Метод|метод|Стратег|стратег)[^<]*</h2>)'
    img1 = create_img_tag(illustrations['weight_init'], 
                         'Сравнение методов инициализации весов', '95%')
    matches = list(re.finditer(pattern, html_content))
    if matches:
        match = matches[0]
        insert_pos = match.end(1)
        html_content = html_content[:insert_pos] + img1 + html_content[insert_pos:]
    else:
        # Try early section
        pattern = r'(<h2>🔷 2\.[^<]*</h2>)'
        match = re.search(pattern, html_content)
        if match:
            insert_pos = match.end(1)
            html_content = html_content[:insert_pos] + img1 + html_content[insert_pos:]
    
    # Add initialization impact
    pattern = r'(<h2>🔷[^<]*(Влияние|влияние|Важност|важност)[^<]*</h2>)'
    img2 = create_img_tag(illustrations['init_impact'], 
                         'Влияние инициализации на скорость обучения', '90%')
    matches = list(re.finditer(pattern, html_content))
    if matches:
        match = matches[0]
        insert_pos = match.end(1)
        html_content = html_content[:insert_pos] + img2 + html_content[insert_pos:]
    
    return html_content

# ============================================================================
# LOSS FUNCTIONS
# ============================================================================

def add_illustrations_to_loss_functions(html_content, illustrations):
    """Add illustrations to loss functions cheatsheet."""
    # Add regression loss functions
    pattern = r'(<h2>🔷[^<]*(Регресси|регресси|MSE|MAE)[^<]*</h2>)'
    img1 = create_img_tag(illustrations['loss_regression'], 
                         'Функции потерь для регрессии', '95%')
    matches = list(re.finditer(pattern, html_content, re.IGNORECASE))
    if matches:
        match = matches[0]
        insert_pos = match.end(1)
        html_content = html_content[:insert_pos] + img1 + html_content[insert_pos:]
    else:
        # Try early section
        pattern = r'(<h2>🔷 2\.[^<]*</h2>)'
        match = re.search(pattern, html_content)
        if match:
            insert_pos = match.end(1)
            html_content = html_content[:insert_pos] + img1 + html_content[insert_pos:]
    
    # Add classification loss functions
    pattern = r'(<h2>🔷[^<]*(Классифик|классифик|Cross.*Entropy|Hinge)[^<]*</h2>)'
    img2 = create_img_tag(illustrations['loss_classification'], 
                         'Функции потерь для классификации', '95%')
    matches = list(re.finditer(pattern, html_content, re.IGNORECASE))
    if matches:
        match = matches[0]
        insert_pos = match.end(1)
        html_content = html_content[:insert_pos] + img2 + html_content[insert_pos:]
    
    return html_content

# ============================================================================
# MAIN PROCESSING FUNCTION
# ============================================================================

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
    """Main function to add illustrations to all optimization & learning cheatsheets."""
    print("=" * 70)
    print("Adding matplotlib illustrations to optimization & learning cheatsheets")
    print("=" * 70)
    
    # Generate all illustrations
    print("\n1. Generating illustrations...")
    illustrations = generate_all_illustrations()
    
    # Process each HTML file
    print("\n2. Adding illustrations to HTML files...")
    
    files_to_process = [
        ('cheatsheets/gradient_descent_cheatsheet.html', add_illustrations_to_gradient_descent),
        ('cheatsheets/backpropagation_cheatsheet.html', add_illustrations_to_backpropagation),
        ('cheatsheets/optimizers_cheatsheet.html', add_illustrations_to_optimizers),
        ('cheatsheets/optimizers_advanced_cheatsheet.html', add_illustrations_to_optimizers_advanced),
        ('cheatsheets/learning_rate_scheduling_cheatsheet.html', add_illustrations_to_learning_rate_scheduling),
        ('cheatsheets/batch_normalization_cheatsheet.html', add_illustrations_to_batch_normalization),
        ('cheatsheets/dropout_regularization_cheatsheet.html', add_illustrations_to_dropout),
        ('cheatsheets/regularization_methods_cheatsheet.html', add_illustrations_to_regularization_methods),
        ('cheatsheets/early_stopping_cheatsheet.html', add_illustrations_to_early_stopping),
        ('cheatsheets/vanishing_gradient_cheatsheet.html', add_illustrations_to_vanishing_gradient),
        ('cheatsheets/weight_initialization_cheatsheet.html', add_illustrations_to_weight_initialization),
        ('cheatsheets/loss_functions_cheatsheet.html', add_illustrations_to_loss_functions),
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
