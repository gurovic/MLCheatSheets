#!/usr/bin/env python3
"""
Add matplotlib illustrations to audio processing cheatsheet HTML files.
"""

import re
from generate_audio_illustrations import generate_all_illustrations

def create_img_tag(base64_data, alt_text, width="100%"):
    """Create HTML img tag with base64 encoded image."""
    return f'''
    <div style="text-align: center; margin: 10px 0;">
      <img src="data:image/png;base64,{base64_data}" 
           alt="{alt_text}" 
           style="max-width: {width}; height: auto; border-radius: 4px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
    </div>
'''

def add_illustrations_to_audio_processing(html_content, illustrations):
    """Add illustrations to audio processing spectrograms & MFCC cheatsheet."""
    
    # 1. Add waveform after waveform visualization code (section 2)
    pattern1 = r"(plt\.show\(\)</code></pre>\s*</div>\s*<div class=\"block\">\s*<h2>🔷 3\. Spectrogram)"
    img1 = create_img_tag(illustrations['waveform'], 'Waveform - форма звуковой волны', '95%')
    html_content = re.sub(pattern1, img1 + r'\n  <div class="block">\n    <h2>🔷 3. Spectrogram', html_content, count=1)
    
    # 2. Add spectrogram after spectrogram code (section 3)
    pattern2 = r"(plt\.show\(\)</code></pre>\s*</div>\s*<div class=\"block\">\s*<h2>🔷 4\. Mel-Spectrogram)"
    img2 = create_img_tag(illustrations['spectrogram'], 'Спектрограмма - частоты во времени', '95%')
    html_content = re.sub(pattern2, img2 + r'\n  <div class="block">\n    <h2>🔷 4. Mel-Spectrogram', html_content, count=1)
    
    # 3. Add mel-spectrogram after mel-spectrogram code (section 4)
    pattern3 = r"(plt\.show\(\)</code></pre>\s*</div>\s*<div class=\"block\">\s*<h2>🔷 5\. MFCC)"
    img3 = create_img_tag(illustrations['mel_spectrogram'], 'Mel-Спектрограмма - mel-шкала частот', '95%')
    html_content = re.sub(pattern3, img3 + r'\n  <div class="block">\n    <h2>🔷 5. MFCC', html_content, count=1)
    
    # 4. Add MFCC after MFCC code (section 5)
    pattern4 = r"(features = np\.concatenate\(\[mfcc_mean, mfcc_std\]\)</code></pre>\s*</div>)"
    img4 = create_img_tag(illustrations['mfcc'], 'MFCC коэффициенты', '95%')
    html_content = re.sub(pattern4, r'\1' + img4, html_content, count=1)
    
    # 5. Add spectral features after the additional features code (section 6)
    pattern5 = r"(print\(f\"Tempo: \{tempo:.2f\} BPM\"\)</code></pre>\s*</div>)"
    img5 = create_img_tag(illustrations['spectral_features'], 'Дополнительные признаки во времени', '95%')
    html_content = re.sub(pattern5, r'\1' + img5, html_content, count=1)
    
    # 6. Add mel scale comparison at the beginning of section 4 (Mel-Spectrogram)
    pattern6 = r"(<h2>🔷 4\. Mel-Spectrogram</h2>\s*<p><strong>Mel-шкала</strong>:)"
    img6 = create_img_tag(illustrations['mel_scale_comparison'], 'Сравнение линейной и mel-шкалы частот', '90%')
    html_content = re.sub(pattern6, r'<h2>🔷 4. Mel-Spectrogram</h2>\n' + img6 + r'\n    <p><strong>Mel-шкала</strong>:', html_content, count=1)
    
    return html_content

def add_illustrations_to_audio_classification(html_content, illustrations):
    """Add illustrations to audio classification cheatsheet."""
    
    # 1. Add classification pipeline after section 1 (Суть)
    pattern1 = r"(<h2>🔷 1\. Суть</h2>.*?</div>)"
    img1 = create_img_tag(illustrations['audio_classification_pipeline'], 'Пайплайн классификации аудио', '95%')
    match = re.search(pattern1, html_content, re.DOTALL)
    if match:
        insert_pos = match.end(1)
        html_content = html_content[:insert_pos] + img1 + html_content[insert_pos:]
    
    # 2. Add feature comparison after section 2 (Признаки)
    pattern2 = r"(<h2>🔷 2\. Признаки \(Features\)</h2>.*?</div>)"
    img2 = create_img_tag(illustrations['feature_comparison'], 'Сравнение различных признаков для классификации', '95%')
    match = re.search(pattern2, html_content, re.DOTALL)
    if match:
        insert_pos = match.end(1)
        html_content = html_content[:insert_pos] + img2 + html_content[insert_pos:]
    
    # 3. Add confusion matrix after section 6 (Метрики качества)
    pattern3 = r"(<h2>🔷 6\. Метрики качества</h2>.*?</div>)"
    img3 = create_img_tag(illustrations['confusion_matrix'], 'Confusion Matrix для классификации звуков', '90%')
    match = re.search(pattern3, html_content, re.DOTALL)
    if match:
        insert_pos = match.end(1)
        html_content = html_content[:insert_pos] + img3 + html_content[insert_pos:]
    
    return html_content

def add_illustrations_to_asr_models(html_content, illustrations):
    """Add illustrations to ASR models cheatsheet."""
    
    # 1. Add ASR pipeline after section 1 (Суть ASR)
    pattern1 = r"(<h2>🔷 1\. Суть ASR</h2>.*?</div>)"
    img1 = create_img_tag(illustrations['asr_pipeline'], 'Пайплайн распознавания речи (ASR)', '95%')
    match = re.search(pattern1, html_content, re.DOTALL)
    if match:
        insert_pos = match.end(1)
        html_content = html_content[:insert_pos] + img1 + html_content[insert_pos:]
    
    # 2. Add WER comparison after section 3 (Deep Learning подходы)
    pattern2 = r"(<h2>🔷 3\. Deep Learning подходы</h2>.*?</div>)"
    img2 = create_img_tag(illustrations['wer_comparison'], 'Сравнение ASR моделей по WER', '95%')
    match = re.search(pattern2, html_content, re.DOTALL)
    if match:
        insert_pos = match.end(1)
        html_content = html_content[:insert_pos] + img2 + html_content[insert_pos:]
    
    # 3. Add attention heatmap after section 6 (Метрики качества)
    pattern3 = r"(<h2>🔷 6\. Метрики качества</h2>.*?</div>)"
    img3 = create_img_tag(illustrations['attention_heatmap'], 'Attention механизм в ASR модели', '90%')
    match = re.search(pattern3, html_content, re.DOTALL)
    if match:
        insert_pos = match.end(1)
        html_content = html_content[:insert_pos] + img3 + html_content[insert_pos:]
    
    # 4. Add CTC alignment after attention
    pattern4 = r"(<h2>🔷 6\. Метрики качества</h2>.*?</div>)"
    # We need to find after the attention image we just added
    img4 = create_img_tag(illustrations['ctc_alignment'], 'CTC Alignment для распознавания речи', '90%')
    # This will be added manually after the attention image
    
    return html_content

def add_ctc_to_asr(html_content, illustrations):
    """Add CTC illustration separately to avoid conflicts."""
    # Find the position after attention heatmap
    pattern = r'(<img src="data:image/png;base64,[^"]*"\s+alt="Attention механизм в ASR модели"[^>]*>\s*</div>)'
    img = create_img_tag(illustrations['ctc_alignment'], 'CTC Alignment для распознавания речи', '90%')
    match = re.search(pattern, html_content, re.DOTALL)
    if match:
        insert_pos = match.end(1)
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
    """Main function to add illustrations to all audio processing cheatsheets."""
    print("=" * 70)
    print("Adding matplotlib illustrations to audio processing cheatsheets")
    print("=" * 70)
    
    # Generate all illustrations
    print("\n1. Generating illustrations...")
    illustrations = generate_all_illustrations()
    
    # Process each HTML file
    print("\n2. Adding illustrations to HTML files...")
    
    files_to_process = [
        ('cheatsheets/audio_processing_spectrograms_mfcc_cheatsheet.html', add_illustrations_to_audio_processing),
        ('cheatsheets/audio_classification_cheatsheet.html', add_illustrations_to_audio_classification),
        ('cheatsheets/asr_models_cheatsheet.html', add_illustrations_to_asr_models),
    ]
    
    success_count = 0
    for filepath, add_func in files_to_process:
        if process_html_file(filepath, add_func, illustrations):
            success_count += 1
    
    # Add CTC separately to ASR models to avoid conflicts
    print("\n3. Adding CTC alignment to ASR models...")
    try:
        with open('cheatsheets/asr_models_cheatsheet.html', 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        modified_content = add_ctc_to_asr(html_content, illustrations)
        
        with open('cheatsheets/asr_models_cheatsheet.html', 'w', encoding='utf-8') as f:
            f.write(modified_content)
        
        print("  ✓ Successfully added CTC alignment")
    except Exception as e:
        print(f"  ✗ Error adding CTC alignment: {e}")
    
    print("\n" + "=" * 70)
    print(f"Completed: {success_count}/{len(files_to_process)} files successfully updated")
    print("=" * 70)

if __name__ == '__main__':
    main()
