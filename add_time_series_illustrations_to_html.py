#!/usr/bin/env python3
"""
Add matplotlib illustrations to time series cheatsheet HTML files.
"""

import re
from generate_time_series_illustrations import generate_all_illustrations

def create_img_tag(base64_data, alt_text, width="100%"):
    """Create HTML img tag with base64 encoded image."""
    return f'''
    <div style="text-align: center; margin: 10px 0;">
      <img src="data:image/png;base64,{base64_data}" 
           alt="{alt_text}" 
           style="max-width: {width}; height: auto; border-radius: 4px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
    </div>
'''

def add_illustrations_to_time_series(html_content, illustrations):
    """Add illustrations to time_series_cheatsheet.html."""
    # Add ARIMA forecast after ARIMA code
    arima_pattern = r'(forecast = fitted\.forecast\(steps=30\)\s*\n\s*print\(forecast\)</code></pre>\s*</div>)'
    arima_img = create_img_tag(illustrations['arima_forecast'], 'ARIMA прогнозирование', '90%')
    html_content = re.sub(arima_pattern, r'\1' + arima_img, html_content, count=1)
    
    # Add time series components visualization after the "Суть" section
    components_pattern = r'(<h2>🔷 2\. ARIMA базовый код</h2>)'
    components_img = create_img_tag(illustrations['time_series_components'], 
                                     'Компоненты временного ряда', '95%')
    html_content = re.sub(components_pattern, components_img + r'\n    \1', html_content, count=1)
    
    return html_content

def add_illustrations_to_exponential_smoothing(html_content, illustrations):
    """Add illustrations to exponential_smoothing_cheatsheet.html."""
    # Add smoothing comparison after "Суть" section
    comparison_pattern = r'(</ul>\s*</div>\s*<div class="block">\s*<h2>🔷 2\.)'
    comparison_img = create_img_tag(illustrations['exponential_smoothing_comparison'], 
                                     'Сравнение методов экспоненциального сглаживания', '95%')
    # Find first occurrence
    match = re.search(comparison_pattern, html_content)
    if match:
        insert_pos = match.start(1)
        html_content = html_content[:insert_pos] + comparison_img + '\n  ' + html_content[insert_pos:]
    
    # Add alpha parameter comparison
    alpha_pattern = r'(<h2>🔷.*?[Пп]араметр.*?alpha.*?</h2>)'
    alpha_img = create_img_tag(illustrations['smoothing_alpha_comparison'], 
                                'Влияние параметра alpha', '90%')
    match = re.search(alpha_pattern, html_content, re.DOTALL | re.IGNORECASE)
    if match:
        insert_pos = match.end(1)
        html_content = html_content[:insert_pos] + alpha_img + html_content[insert_pos:]
    
    return html_content

def add_illustrations_to_seasonal_decomposition(html_content, illustrations):
    """Add illustrations to seasonal_decomposition_cheatsheet.html."""
    # Add decomposition visualization after the decomposition code
    decomp_pattern = r'(result\.plot\(\)\s*\n\s*plt\.show\(\)</code></pre>\s*</div>)'
    decomp_img = create_img_tag(illustrations['seasonal_decomposition'], 
                                 'Сезонная декомпозиция', '95%')
    # Find first occurrence after seasonal_decompose
    match = re.search(r'seasonal_decompose.*?' + decomp_pattern, html_content, re.DOTALL)
    if match:
        insert_pos = match.end(1)
        html_content = html_content[:insert_pos] + decomp_img + html_content[insert_pos:]
    
    # Add additive vs multiplicative comparison after section 2 (Основные компоненты)
    component_pattern = r'(</ul>\s*</div>\s*<div class="block">\s*<h2>🔷 3\.)'
    model_img = create_img_tag(illustrations['additive_vs_multiplicative'], 
                                'Аддитивная vs Мультипликативная модель', '95%')
    match = re.search(component_pattern, html_content)
    if match:
        insert_pos = match.start(1)
        html_content = html_content[:insert_pos] + model_img + '\n  ' + html_content[insert_pos:]
    
    return html_content

def add_illustrations_to_time_series_validation(html_content, illustrations):
    """Add illustrations to time_series_validation_cheatsheet.html."""
    # Add train/test split visualization after first section
    # Pattern 1: Try with </p> closing tag
    split_pattern1 = r'(</p>\s*\n\s*</div>\s*<div class="block">\s*<h2>🔷 2\.)'
    split_img = create_img_tag(illustrations['train_test_split'], 
                                'Стратегии разделения временных рядов', '95%')
    match = re.search(split_pattern1, html_content)
    if match:
        insert_pos = match.start(1)
        html_content = html_content[:insert_pos] + split_img + '\n  ' + html_content[insert_pos:]
    else:
        # Pattern 2: Try with </ul> closing tag
        split_pattern2 = r'(</ul>\s*</div>\s*<div class="block">\s*<h2>🔷 2\.)'
        match = re.search(split_pattern2, html_content)
        if match:
            insert_pos = match.start(1)
            html_content = html_content[:insert_pos] + split_img + '\n  ' + html_content[insert_pos:]
    
    return html_content

def add_illustrations_to_time_series_feature_engineering(html_content, illustrations):
    """Add illustrations to time_series_feature_engineering_cheatsheet.html."""
    # Add lag features visualization
    lag_pattern = r'(<h2>🔷.*?[Лл]аг.*?</h2>)'
    lag_img = create_img_tag(illustrations['lag_features'], 
                             'Лаговые признаки', '90%')
    match = re.search(lag_pattern, html_content, re.DOTALL | re.IGNORECASE)
    if match:
        insert_pos = match.end(1)
        html_content = html_content[:insert_pos] + lag_img + html_content[insert_pos:]
    
    # Add rolling window features
    rolling_pattern = r'(<h2>🔷.*?[Сс]кольз.*?</h2>)'
    rolling_img = create_img_tag(illustrations['rolling_window_features'], 
                                  'Скользящие окна', '90%')
    match = re.search(rolling_pattern, html_content, re.DOTALL | re.IGNORECASE)
    if match:
        insert_pos = match.end(1)
        html_content = html_content[:insert_pos] + rolling_img + html_content[insert_pos:]
    
    return html_content

def add_illustrations_to_rnn_lstm(html_content, illustrations):
    """Add illustrations to rnn_lstm_time_series_cheatsheet.html."""
    # Add LSTM architecture after "Суть" or architecture section
    arch_pattern = r'(</ul>\s*</div>\s*<div class="block">\s*<h2>🔷 2\.)'
    arch_img = create_img_tag(illustrations['lstm_architecture'], 
                               'Архитектура LSTM', '90%')
    match = re.search(arch_pattern, html_content)
    if match:
        insert_pos = match.start(1)
        html_content = html_content[:insert_pos] + arch_img + '\n  ' + html_content[insert_pos:]
    
    # Add LSTM prediction visualization
    pred_pattern = r'(<h2>🔷.*?[Пп]рогноз.*?</h2>)'
    pred_img = create_img_tag(illustrations['lstm_prediction'], 
                               'LSTM прогнозирование', '90%')
    match = re.search(pred_pattern, html_content, re.DOTALL | re.IGNORECASE)
    if match:
        insert_pos = match.end(1)
        html_content = html_content[:insert_pos] + pred_img + html_content[insert_pos:]
    
    return html_content

def add_illustrations_to_transformers(html_content, illustrations):
    """Add illustrations to transformers_time_series_cheatsheet.html."""
    # Add transformer prediction after first section
    pred_pattern1 = r'(</ul>\s*</div>\s*<div class="block">\s*<h2>🔷 2\.)'
    pred_img = create_img_tag(illustrations['transformer_prediction'], 
                               'Transformer прогнозирование', '90%')
    match = re.search(pred_pattern1, html_content)
    if match:
        insert_pos = match.start(1)
        html_content = html_content[:insert_pos] + pred_img + '\n  ' + html_content[insert_pos:]
    
    # Add attention mechanism visualization
    attention_pattern = r'(<h2>🔷.*?[Вв]ниман.*?</h2>)'
    attention_img = create_img_tag(illustrations['attention_mechanism'], 
                                    'Механизм внимания', '85%')
    match = re.search(attention_pattern, html_content, re.DOTALL | re.IGNORECASE)
    if match:
        insert_pos = match.end(1)
        html_content = html_content[:insert_pos] + attention_img + html_content[insert_pos:]
    
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
    """Main function to add illustrations to all time series cheatsheets."""
    print("=" * 70)
    print("Adding matplotlib illustrations to time series cheatsheets")
    print("=" * 70)
    
    # Generate all illustrations
    print("\n1. Generating illustrations...")
    illustrations = generate_all_illustrations()
    
    # Process each HTML file
    print("\n2. Adding illustrations to HTML files...")
    
    files_to_process = [
        ('cheatsheets/time_series_cheatsheet.html', add_illustrations_to_time_series),
        ('cheatsheets/exponential_smoothing_cheatsheet.html', add_illustrations_to_exponential_smoothing),
        ('cheatsheets/seasonal_decomposition_cheatsheet.html', add_illustrations_to_seasonal_decomposition),
        ('cheatsheets/time_series_validation_cheatsheet.html', add_illustrations_to_time_series_validation),
        ('cheatsheets/time_series_feature_engineering_cheatsheet.html', add_illustrations_to_time_series_feature_engineering),
        ('cheatsheets/rnn_lstm_time_series_cheatsheet.html', add_illustrations_to_rnn_lstm),
        ('cheatsheets/transformers_time_series_cheatsheet.html', add_illustrations_to_transformers),
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
