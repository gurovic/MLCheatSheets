#!/usr/bin/env python3
"""
Add matplotlib illustrations to Computer Vision cheatsheet HTML files.
"""

import re
import os
from generate_computer_vision_illustrations import generate_all_illustrations

def create_img_tag(base64_data, alt_text, width="100%"):
    """Create HTML img tag with base64 encoded image."""
    return f'''
    <div style="text-align: center; margin: 15px 0;">
      <img src="data:image/png;base64,{base64_data}" 
           alt="{alt_text}" 
           style="max-width: {width}; height: auto; border-radius: 4px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
    </div>
'''

def add_illustrations_to_object_detection(html_content, illustrations):
    """Add illustrations to object detection cheatsheet."""
    # Add bounding boxes after detection task section
    pattern1 = r'(<h2>🔷 1\. Задача детекции</h2>.*?</div>)'
    img1 = create_img_tag(illustrations['od_bounding_boxes'], 
                         'Bounding Boxes и детекция объектов', '95%')
    match = re.search(pattern1, html_content, re.DOTALL)
    if match:
        insert_pos = match.end(1)
        html_content = html_content[:insert_pos] + img1 + html_content[insert_pos:]
    
    # Add IoU visualization after IoU section
    pattern2 = r'(<h2>🔷 11\. IoU.*?</h2>.*?</div>)'
    img2 = create_img_tag(illustrations['od_iou'], 
                         'IoU (Intersection over Union)', '95%')
    match = re.search(pattern2, html_content, re.DOTALL)
    if match:
        insert_pos = match.end(1)
        html_content = html_content[:insert_pos] + img2 + html_content[insert_pos:]
    
    # Add mAP visualization after mAP section
    pattern3 = r'(<h2>🔷 12\. mAP.*?</h2>.*?</div>)'
    img3 = create_img_tag(illustrations['od_map'], 
                         'mAP (mean Average Precision) метрика', '95%')
    match = re.search(pattern3, html_content, re.DOTALL)
    if match:
        insert_pos = match.end(1)
        html_content = html_content[:insert_pos] + img3 + html_content[insert_pos:]
    
    return html_content

def add_illustrations_to_yolo(html_content, illustrations):
    """Add illustrations to YOLO cheatsheet."""
    # Add grid detection after YOLO concept section
    pattern1 = r'(<h2[^>]*>.*?1\..*?(Суть|YOLO).*?</h2>.*?</div>)'
    img1 = create_img_tag(illustrations['yolo_grid'], 
                         'YOLO grid-based detection', '95%')
    match = re.search(pattern1, html_content, re.DOTALL | re.IGNORECASE)
    if match:
        insert_pos = match.end(1)
        html_content = html_content[:insert_pos] + img1 + html_content[insert_pos:]
    
    # Add architecture after installation or basic code section
    pattern2 = r'(<h2[^>]*>.*?[23]\..*?(Базовый|Код|YOLOv).*?</h2>.*?</div>)'
    img2 = create_img_tag(illustrations['yolo_architecture'], 
                         'YOLO архитектура: Multi-Scale Detection', '95%')
    match = re.search(pattern2, html_content, re.DOTALL | re.IGNORECASE)
    if match:
        insert_pos = match.end(1)
        html_content = html_content[:insert_pos] + img2 + html_content[insert_pos:]
    
    # Add anchor boxes after models/speed section
    pattern3 = r'(<h2[^>]*>.*?6\..*?(Модели|скорость|Model).*?</h2>.*?</table>.*?</div>)'
    img3 = create_img_tag(illustrations['yolo_anchors'], 
                         'YOLO Anchor Boxes по масштабам', '90%')
    match = re.search(pattern3, html_content, re.DOTALL | re.IGNORECASE)
    if match:
        insert_pos = match.end(1)
        html_content = html_content[:insert_pos] + img3 + html_content[insert_pos:]
    
    return html_content

def add_illustrations_to_image_segmentation(html_content, illustrations):
    """Add illustrations to image segmentation cheatsheet."""
    # Add segmentation types after types section
    pattern1 = r'(<h2>🔷 1\. Типы сегментации</h2>.*?</div>)'
    img1 = create_img_tag(illustrations['seg_types'], 
                         'Типы сегментации изображений', '95%')
    match = re.search(pattern1, html_content, re.DOTALL)
    if match:
        insert_pos = match.end(1)
        html_content = html_content[:insert_pos] + img1 + html_content[insert_pos:]
    
    # Add U-Net architecture after U-Net section
    pattern2 = r'(<h2>🔷 2\. U-Net.*?</h2>.*?</div>)'
    img2 = create_img_tag(illustrations['seg_unet'], 
                         'U-Net архитектура', '95%')
    match = re.search(pattern2, html_content, re.DOTALL)
    if match:
        insert_pos = match.end(1)
        html_content = html_content[:insert_pos] + img2 + html_content[insert_pos:]
    
    # Add masks visualization after any masks or visualization section
    pattern3 = r'(<h2>🔷 [456]\. (Маски|Визуализация|Loss).*?</h2>.*?</div>)'
    img3 = create_img_tag(illustrations['seg_masks'], 
                         'Маски сегментации и визуализация', '95%')
    match = re.search(pattern3, html_content, re.DOTALL)
    if match:
        insert_pos = match.end(1)
        html_content = html_content[:insert_pos] + img3 + html_content[insert_pos:]
    
    return html_content

def add_illustrations_to_keypoint_detection(html_content, illustrations):
    """Add illustrations to keypoint detection cheatsheet."""
    # Add skeleton after main concept section
    pattern1 = r'(<h2[^>]*>.*?1\..*?(Основы|Keypoint).*?</h2>.*?</div>)'
    img1 = create_img_tag(illustrations['kp_skeleton'], 
                         'Детекция ключевых точек и скелет', '95%')
    match = re.search(pattern1, html_content, re.DOTALL | re.IGNORECASE)
    if match:
        insert_pos = match.end(1)
        html_content = html_content[:insert_pos] + img1 + html_content[insert_pos:]
    
    # Add heatmap visualization after pose estimation section
    pattern2 = r'(<h2[^>]*>.*?2\..*?(Pose|Estimation|Overview).*?</h2>.*?</div>)'
    img2 = create_img_tag(illustrations['kp_heatmap'], 
                         'Heatmap визуализация ключевых точек', '95%')
    match = re.search(pattern2, html_content, re.DOTALL | re.IGNORECASE)
    if match:
        insert_pos = match.end(1)
        html_content = html_content[:insert_pos] + img2 + html_content[insert_pos:]
    
    # Add multi-person approach after top-down section
    pattern3 = r'(<h2[^>]*>.*?3\..*?(Top-down|Bottom-up).*?</h2>.*?</div>)'
    img3 = create_img_tag(illustrations['kp_multi'], 
                         'Multi-Person Pose Estimation подходы', '95%')
    match = re.search(pattern3, html_content, re.DOTALL | re.IGNORECASE)
    if match:
        insert_pos = match.end(1)
        html_content = html_content[:insert_pos] + img3 + html_content[insert_pos:]
    
    return html_content

def add_illustrations_to_cnn_visualization(html_content, illustrations):
    """Add illustrations to CNN visualization techniques cheatsheet."""
    # Add feature maps after main section
    pattern1 = r'(<h2>🔷 1\. (Зачем|Проблема).*?</h2>.*?</div>)'
    img1 = create_img_tag(illustrations['cnn_feature_maps'], 
                         'Feature Maps на разных слоях', '95%')
    match = re.search(pattern1, html_content, re.DOTALL)
    if match:
        insert_pos = match.end(1)
        html_content = html_content[:insert_pos] + img1 + html_content[insert_pos:]
    
    # Add filter patterns
    pattern2 = r'(<h2>🔷 [234]\. (Фильтр|Filter|Visualization).*?</h2>.*?</div>)'
    img2 = create_img_tag(illustrations['cnn_filters'], 
                         'Паттерны фильтров CNN', '95%')
    match = re.search(pattern2, html_content, re.DOTALL)
    if match:
        insert_pos = match.end(1)
        html_content = html_content[:insert_pos] + img2 + html_content[insert_pos:]
    
    # Add activation visualization
    pattern3 = r'(<h2>🔷 [345]\. (Activation|Активация).*?</h2>.*?</div>)'
    img3 = create_img_tag(illustrations['cnn_activations'], 
                         'Визуализация активаций', '95%')
    match = re.search(pattern3, html_content, re.DOTALL)
    if match:
        insert_pos = match.end(1)
        html_content = html_content[:insert_pos] + img3 + html_content[insert_pos:]
    
    return html_content

def add_illustrations_to_gradcam(html_content, illustrations):
    """Add illustrations to Grad-CAM cheatsheet."""
    # Add Grad-CAM visualization after main concept
    pattern1 = r'(<h2>🔷 [45]\. Grad-CAM.*?</h2>.*?</div>)'
    img1 = create_img_tag(illustrations['gradcam_viz'], 
                         'Grad-CAM визуализация', '95%')
    match = re.search(pattern1, html_content, re.DOTALL)
    if match:
        insert_pos = match.end(1)
        html_content = html_content[:insert_pos] + img1 + html_content[insert_pos:]
    
    # Add class-specific visualizations
    pattern2 = r'(<h2>🔷 [67]\. (Применение|Использование|Код).*?</h2>.*?</div>)'
    img2 = create_img_tag(illustrations['gradcam_classes'], 
                         'Grad-CAM для разных классов', '95%')
    match = re.search(pattern2, html_content, re.DOTALL)
    if match:
        insert_pos = match.end(1)
        html_content = html_content[:insert_pos] + img2 + html_content[insert_pos:]
    
    # Add layer comparison
    pattern3 = r'(<h2>🔷 [89]\. (Слои|Layer).*?</h2>.*?</div>)'
    img3 = create_img_tag(illustrations['gradcam_layers'], 
                         'Grad-CAM на разных слоях', '90%')
    match = re.search(pattern3, html_content, re.DOTALL)
    if match:
        insert_pos = match.end(1)
        html_content = html_content[:insert_pos] + img3 + html_content[insert_pos:]
    
    return html_content

def add_illustrations_to_saliency_maps(html_content, illustrations):
    """Add illustrations to saliency maps cheatsheet."""
    # Add methods comparison after main section
    pattern1 = r'(<h2>🔷 [12]\. (Что такое|Суть).*?</h2>.*?</div>)'
    img1 = create_img_tag(illustrations['saliency_methods'], 
                         'Методы построения Saliency Maps', '95%')
    match = re.search(pattern1, html_content, re.DOTALL)
    if match:
        insert_pos = match.end(1)
        html_content = html_content[:insert_pos] + img1 + html_content[insert_pos:]
    
    # Add class comparison
    pattern2 = r'(<h2>🔷 [345]\. (Vanilla|SmoothGrad|Integrated).*?</h2>.*?</div>)'
    img2 = create_img_tag(illustrations['saliency_comparison'], 
                         'Saliency Maps для разных предсказаний', '95%')
    match = re.search(pattern2, html_content, re.DOTALL)
    if match:
        insert_pos = match.end(1)
        html_content = html_content[:insert_pos] + img2 + html_content[insert_pos:]
    
    # Add integrated gradients path
    pattern3 = r'(<h2>🔷 [56]\. Integrated Gradients.*?</h2>.*?</div>)'
    img3 = create_img_tag(illustrations['saliency_integrated'], 
                         'Integrated Gradients: путь интерполяции', '95%')
    match = re.search(pattern3, html_content, re.DOTALL)
    if match:
        insert_pos = match.end(1)
        html_content = html_content[:insert_pos] + img3 + html_content[insert_pos:]
    
    return html_content

def add_illustrations_to_neural_style_transfer(html_content, illustrations):
    """Add illustrations to neural style transfer cheatsheet."""
    # Add process visualization after main concept
    pattern1 = r'(<h2[^>]*>.*?1\..*?(Основная|идея|Idea).*?</h2>.*?</div>)'
    img1 = create_img_tag(illustrations['nst_process'], 
                         'Neural Style Transfer: процесс', '95%')
    match = re.search(pattern1, html_content, re.DOTALL | re.IGNORECASE)
    if match:
        insert_pos = match.end(1)
        html_content = html_content[:insert_pos] + img1 + html_content[insert_pos:]
    
    # Add evolution/optimization after content or style loss
    pattern2 = r'(<h2[^>]*>.*?[34]\..*?(Content|Style|Loss).*?</h2>.*?</div>)'
    img2 = create_img_tag(illustrations['nst_evolution'], 
                         'Эволюция в процессе оптимизации', '95%')
    match = re.search(pattern2, html_content, re.DOTALL | re.IGNORECASE)
    if match:
        insert_pos = match.end(1)
        html_content = html_content[:insert_pos] + img2 + html_content[insert_pos:]
    
    # Add weight balance after optimization section
    pattern3 = r'(<h2[^>]*>.*?5\..*?(Оптимизация|Optimization).*?</h2>.*?</div>)'
    img3 = create_img_tag(illustrations['nst_weights'], 
                         'Влияние весов content и style', '95%')
    match = re.search(pattern3, html_content, re.DOTALL | re.IGNORECASE)
    if match:
        insert_pos = match.end(1)
        html_content = html_content[:insert_pos] + img3 + html_content[insert_pos:]
    
    return html_content

def process_html_file(filepath, add_illustrations_func, illustrations):
    """Process a single HTML file to add illustrations."""
    print(f"Processing {filepath}...")
    
    try:
        if not os.path.exists(filepath):
            print(f"  ! File not found: {filepath}")
            return False
            
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
    """Main function to add illustrations to all Computer Vision cheatsheets."""
    print("=" * 70)
    print("Adding matplotlib illustrations to Computer Vision cheatsheets")
    print("=" * 70)
    
    # Generate all illustrations
    print("\n1. Generating illustrations...")
    illustrations = generate_all_illustrations()
    
    # Process each HTML file
    print("\n2. Adding illustrations to HTML files...")
    
    files_to_process = [
        ('cheatsheets/object_detection_cheatsheet.html', add_illustrations_to_object_detection),
        ('cheatsheets/yolo_cheatsheet.html', add_illustrations_to_yolo),
        ('cheatsheets/image_segmentation_cheatsheet.html', add_illustrations_to_image_segmentation),
        ('cheatsheets/keypoint_detection_pose_estimation_cheatsheet.html', add_illustrations_to_keypoint_detection),
        ('cheatsheets/cnn_visualization_techniques_cheatsheet.html', add_illustrations_to_cnn_visualization),
        ('cheatsheets/grad_cam_cheatsheet.html', add_illustrations_to_gradcam),
        ('cheatsheets/saliency_maps_cheatsheet.html', add_illustrations_to_saliency_maps),
        ('cheatsheets/neural_style_transfer_cheatsheet.html', add_illustrations_to_neural_style_transfer),
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
