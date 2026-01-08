#!/usr/bin/env python3
"""
Generate matplotlib illustrations for Computer Vision cheatsheets.
This script creates high-quality visualizations and encodes them as base64 for inline HTML embedding.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle, FancyBboxPatch, Circle, Polygon, FancyArrowPatch
from matplotlib.colors import LinearSegmentedColormap
import base64
from io import BytesIO
import warnings
warnings.filterwarnings('ignore')

# Set consistent style
sns.set_style("whitegrid")
DPI_VALUE = 300
plt.rcParams['figure.dpi'] = DPI_VALUE
plt.rcParams['savefig.dpi'] = DPI_VALUE
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9

def fig_to_base64(fig):
    """Convert matplotlib figure to base64 string."""
    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=DPI_VALUE)
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()
    plt.close(fig)
    return img_base64

# ============================================================================
# OBJECT DETECTION ILLUSTRATIONS
# ============================================================================

def generate_object_detection_bounding_boxes():
    """Visualize bounding boxes and detection process."""
    np.random.seed(42)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Create a simple image
    img = np.ones((100, 100, 3)) * 0.9
    
    # Subplot 1: Single object detection
    ax = axes[0]
    ax.imshow(img)
    ax.set_title('Одиночная детекция', fontweight='bold')
    ax.axis('off')
    
    # Draw bounding box
    rect = Rectangle((20, 30), 40, 50, linewidth=3, edgecolor='red', facecolor='none')
    ax.add_patch(rect)
    ax.text(40, 25, 'Cat: 0.95', fontsize=10, color='red', weight='bold',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Subplot 2: Multiple detections
    ax = axes[1]
    ax.imshow(img)
    ax.set_title('Множественная детекция', fontweight='bold')
    ax.axis('off')
    
    # Draw multiple bounding boxes
    boxes = [(10, 15, 25, 30, 'Dog', 0.89), 
             (45, 50, 35, 40, 'Cat', 0.92),
             (60, 20, 30, 35, 'Bird', 0.78)]
    colors = ['red', 'green', 'blue']
    
    for i, (x, y, w, h, label, conf) in enumerate(boxes):
        rect = Rectangle((x, y), w, h, linewidth=2, edgecolor=colors[i], facecolor='none')
        ax.add_patch(rect)
        ax.text(x+w/2, y-3, f'{label}: {conf:.2f}', fontsize=9, color=colors[i],
                weight='bold', ha='center',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Subplot 3: Detection with confidence threshold
    ax = axes[2]
    ax.imshow(img)
    ax.set_title('Фильтрация по порогу (>0.85)', fontweight='bold')
    ax.axis('off')
    
    # Only high confidence boxes
    for i, (x, y, w, h, label, conf) in enumerate(boxes):
        if conf >= 0.85:
            rect = Rectangle((x, y), w, h, linewidth=2, edgecolor=colors[i], facecolor='none')
            ax.add_patch(rect)
            ax.text(x+w/2, y-3, f'{label}: {conf:.2f}', fontsize=9, color=colors[i],
                    weight='bold', ha='center',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.suptitle('Детекция объектов: Bounding Boxes', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    return fig_to_base64(fig)

def generate_object_detection_iou():
    """Visualize IoU (Intersection over Union) concept."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Ground truth box (blue)
    gt_box = [30, 30, 40, 35]
    # Predicted boxes with different IoU values
    pred_boxes = [
        ([32, 32, 38, 33], 0.85, 'Хорошее'),  # High IoU
        ([25, 25, 35, 30], 0.45, 'Среднее'),  # Medium IoU
        ([50, 50, 30, 25], 0.05, 'Плохое')    # Low IoU
    ]
    
    for idx, (pred_box, iou_val, quality) in enumerate(pred_boxes):
        ax = axes[idx]
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        ax.set_aspect('equal')
        ax.set_title(f'{quality} перекрытие (IoU={iou_val:.2f})', fontweight='bold')
        ax.axis('off')
        
        # Draw ground truth
        gt_rect = Rectangle((gt_box[0], gt_box[1]), gt_box[2], gt_box[3],
                           linewidth=2, edgecolor='blue', facecolor='blue', alpha=0.3)
        ax.add_patch(gt_rect)
        ax.text(gt_box[0]+gt_box[2]/2, gt_box[1]-3, 'Ground Truth', 
                fontsize=9, color='blue', weight='bold', ha='center')
        
        # Draw prediction
        pred_rect = Rectangle((pred_box[0], pred_box[1]), pred_box[2], pred_box[3],
                             linewidth=2, edgecolor='red', facecolor='red', alpha=0.3)
        ax.add_patch(pred_rect)
        ax.text(pred_box[0]+pred_box[2]/2, pred_box[1]+pred_box[3]+5, 'Prediction',
                fontsize=9, color='red', weight='bold', ha='center')
    
    plt.suptitle('IoU (Intersection over Union)', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    return fig_to_base64(fig)

def generate_object_detection_map():
    """Visualize mAP (mean Average Precision) curve."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Precision-Recall curve
    recall = np.linspace(0, 1, 100)
    precision = 1 - recall * 0.7 + np.random.randn(100) * 0.05
    precision = np.clip(precision, 0, 1)
    precision = np.sort(precision)[::-1]
    
    ax1.plot(recall, precision, 'b-', linewidth=2, label='Класс 1')
    ax1.fill_between(recall, precision, alpha=0.3)
    ax1.set_xlabel('Recall', fontweight='bold')
    ax1.set_ylabel('Precision', fontweight='bold')
    ax1.set_title('Precision-Recall кривая', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    # Calculate area under curve (AP)
    from scipy import integrate
    ap = integrate.trapezoid(precision, recall)
    ax1.text(0.5, 0.2, f'AP = {ap:.3f}', fontsize=11, 
             bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    ax1.legend()
    
    # mAP across classes
    classes = ['Person', 'Car', 'Dog', 'Cat', 'Bird']
    ap_values = [0.89, 0.92, 0.76, 0.81, 0.68]
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(classes)))
    
    ax2.bar(classes, ap_values, color=colors, edgecolor='black', linewidth=1.5)
    ax2.axhline(y=np.mean(ap_values), color='red', linestyle='--', linewidth=2, 
                label=f'mAP = {np.mean(ap_values):.3f}')
    ax2.set_ylabel('Average Precision', fontweight='bold')
    ax2.set_title('AP по классам', fontweight='bold')
    ax2.set_ylim(0, 1)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add values on bars
    for i, v in enumerate(ap_values):
        ax2.text(i, v + 0.02, f'{v:.2f}', ha='center', fontweight='bold')
    
    plt.suptitle('mAP (mean Average Precision) метрика', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    return fig_to_base64(fig)


# ============================================================================
# YOLO ILLUSTRATIONS
# ============================================================================

def generate_yolo_grid_detection():
    """Visualize YOLO grid-based detection."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Image with grid
    ax = axes[0]
    img = np.ones((100, 100, 3)) * 0.95
    ax.imshow(img)
    ax.set_title('YOLO сетка (7×7)', fontweight='bold')
    ax.axis('off')
    
    # Draw grid
    grid_size = 7
    for i in range(grid_size + 1):
        pos = i * 100 / grid_size
        ax.axhline(y=pos, color='gray', linestyle='-', linewidth=0.5)
        ax.axvline(x=pos, color='gray', linestyle='-', linewidth=0.5)
    
    # Draw object with responsible cell highlighted
    obj_center_x, obj_center_y = 60, 45
    obj_w, obj_h = 30, 25
    
    # Highlight responsible cell
    cell_x = int(obj_center_x / (100 / grid_size))
    cell_y = int(obj_center_y / (100 / grid_size))
    cell_rect = Rectangle((cell_x * 100 / grid_size, cell_y * 100 / grid_size),
                          100 / grid_size, 100 / grid_size,
                          facecolor='yellow', alpha=0.5, edgecolor='orange', linewidth=2)
    ax.add_patch(cell_rect)
    
    # Draw bounding box
    rect = Rectangle((obj_center_x - obj_w/2, obj_center_y - obj_h/2), obj_w, obj_h,
                    linewidth=2, edgecolor='red', facecolor='none')
    ax.add_patch(rect)
    ax.plot(obj_center_x, obj_center_y, 'ro', markersize=8)
    ax.text(obj_center_x, obj_center_y - obj_h/2 - 5, 'Dog: 0.92', 
            fontsize=9, color='red', weight='bold', ha='center',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # YOLO output tensor
    ax = axes[1]
    ax.axis('off')
    ax.set_title('YOLO выходной тензор', fontweight='bold')
    
    # Draw tensor representation
    tensor_y_start = 20
    tensor_height = 60
    cell_height = tensor_height / 7
    
    for i in range(7):
        y_pos = tensor_y_start + i * cell_height
        rect = Rectangle((10, y_pos), 80, cell_height * 0.9,
                        facecolor='lightblue', edgecolor='black', linewidth=0.5)
        ax.add_patch(rect)
        
    # Highlight one cell output
    highlight_idx = 3
    y_pos = tensor_y_start + highlight_idx * cell_height
    rect = Rectangle((10, y_pos), 80, cell_height * 0.9,
                    facecolor='orange', edgecolor='red', linewidth=2)
    ax.add_patch(rect)
    
    ax.text(50, y_pos + cell_height * 0.45, '[x, y, w, h, conf, class...]',
            fontsize=8, ha='center', va='center', weight='bold')
    
    ax.text(50, 10, 'Каждая ячейка предсказывает:', fontsize=10, ha='center', weight='bold')
    ax.text(50, 90, '• Координаты bbox\n• Confidence\n• Вероятности классов', 
            fontsize=9, ha='center', va='top')
    
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    
    plt.suptitle('YOLO: Grid-based Detection', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    return fig_to_base64(fig)

def generate_yolo_anchors():
    """Visualize anchor boxes concept."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    anchor_shapes = [
        ([(20, 30), (30, 20), (25, 25)], 'Малые объекты'),
        ([(40, 60), (60, 40), (50, 50)], 'Средние объекты'),
        ([(70, 90), (90, 70), (80, 80)], 'Большие объекты')
    ]
    
    colors = ['red', 'green', 'blue']
    
    for idx, (anchors, title) in enumerate(anchor_shapes):
        ax = axes[idx]
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        ax.set_aspect('equal')
        ax.set_title(title, fontweight='bold')
        ax.axis('off')
        
        # Draw background
        ax.add_patch(Rectangle((0, 0), 100, 100, facecolor='lightgray', alpha=0.3))
        
        # Draw anchor boxes
        for i, (w, h) in enumerate(anchors):
            x = 50 - w/2
            y = 50 - h/2
            rect = Rectangle((x, y), w, h, linewidth=2, edgecolor=colors[i],
                           facecolor='none', linestyle='--')
            ax.add_patch(rect)
            ax.text(50, y - 3, f'{w}×{h}', fontsize=8, ha='center', color=colors[i])
        
        # Center point
        ax.plot(50, 50, 'ko', markersize=6)
    
    plt.suptitle('YOLO Anchor Boxes по масштабам', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    return fig_to_base64(fig)

def generate_yolo_architecture():
    """Visualize YOLO architecture flow."""
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis('off')
    
    # Architecture blocks
    blocks = [
        (5, 40, 10, 20, 'Входное\nизображение\n(416×416×3)', 'lightblue'),
        (20, 35, 10, 30, 'Backbone\n(Feature\nExtraction)', 'lightgreen'),
        (35, 30, 10, 40, 'Neck\n(Feature\nFusion)', 'lightyellow'),
        (50, 25, 10, 50, 'Head\n(Detection)', 'lightcoral'),
        (65, 15, 10, 30, 'Small\nObjects', 'orange'),
        (65, 47.5, 10, 15, 'Medium\nObjects', 'orange'),
        (65, 65, 10, 15, 'Large\nObjects', 'orange'),
        (80, 40, 10, 20, 'Predictions\n(Boxes +\nClasses)', 'lightpink')
    ]
    
    for x, y, w, h, label, color in blocks:
        rect = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.5",
                             facecolor=color, edgecolor='black', linewidth=2)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, label, fontsize=9, ha='center', va='center',
               weight='bold', multialignment='center')
    
    # Arrows
    arrows = [
        (15, 50, 20, 50),  # Input to Backbone
        (30, 50, 35, 50),  # Backbone to Neck
        (45, 50, 50, 50),  # Neck to Head
        (60, 40, 65, 30),  # Head to Small
        (60, 50, 65, 50),  # Head to Medium
        (60, 60, 65, 70),  # Head to Large
        (75, 30, 80, 45),  # Small to Predictions
        (75, 55, 80, 50),  # Medium to Predictions
        (75, 72.5, 80, 55),  # Large to Predictions
    ]
    
    for x1, y1, x2, y2 in arrows:
        arrow = FancyArrowPatch((x1, y1), (x2, y2),
                               arrowstyle='->', mutation_scale=20,
                               linewidth=2, color='darkblue')
        ax.add_patch(arrow)
    
    ax.text(50, 95, 'YOLO архитектура: Multi-Scale Detection', 
           fontsize=13, ha='center', weight='bold')
    ax.text(50, 5, 'Предсказание на 3 масштабах для детекции объектов разных размеров', 
           fontsize=10, ha='center', style='italic')
    
    plt.tight_layout()
    return fig_to_base64(fig)


# ============================================================================
# IMAGE SEGMENTATION ILLUSTRATIONS
# ============================================================================

def generate_segmentation_types():
    """Visualize different types of segmentation."""
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    # Create simple image
    img = np.ones((80, 80, 3)) * 0.9
    
    # Original image
    ax = axes[0]
    ax.imshow(img)
    ax.set_title('Исходное\nизображение', fontweight='bold')
    ax.axis('off')
    
    # Draw two objects
    circle1 = Circle((30, 40), 15, facecolor='darkblue', alpha=0.6)
    circle2 = Circle((55, 45), 12, facecolor='darkblue', alpha=0.6)
    ax.add_patch(circle1)
    ax.add_patch(circle2)
    
    # Semantic segmentation
    ax = axes[1]
    seg_img = np.ones((80, 80)) * 0.3  # Background
    y, x = np.ogrid[:80, :80]
    mask1 = (x - 30)**2 + (y - 40)**2 <= 15**2
    mask2 = (x - 55)**2 + (y - 45)**2 <= 12**2
    seg_img[mask1 | mask2] = 1.0  # Same class for both
    ax.imshow(seg_img, cmap='tab20', vmin=0, vmax=1)
    ax.set_title('Semantic\nSegmentation', fontweight='bold')
    ax.axis('off')
    ax.text(40, 10, 'Один класс', fontsize=9, ha='center', 
           bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    
    # Instance segmentation
    ax = axes[2]
    inst_img = np.zeros((80, 80))
    inst_img[mask1] = 1.0  # Instance 1
    inst_img[mask2] = 2.0  # Instance 2
    ax.imshow(inst_img, cmap='tab20', vmin=0, vmax=3)
    ax.set_title('Instance\nSegmentation', fontweight='bold')
    ax.axis('off')
    ax.text(40, 10, 'Разные экземпляры', fontsize=9, ha='center',
           bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    
    # Panoptic segmentation
    ax = axes[3]
    panoptic_img = np.ones((80, 80)) * 0.5  # Background class
    panoptic_img[mask1] = 1.5  # Instance 1
    panoptic_img[mask2] = 2.5  # Instance 2
    ax.imshow(panoptic_img, cmap='tab20', vmin=0, vmax=3)
    ax.set_title('Panoptic\nSegmentation', fontweight='bold')
    ax.axis('off')
    ax.text(40, 10, 'Классы + Экземпляры', fontsize=9, ha='center',
           bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    
    plt.suptitle('Типы сегментации изображений', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    return fig_to_base64(fig)

def generate_unet_architecture():
    """Visualize U-Net architecture."""
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis('off')
    
    # Encoder path (left, downward)
    encoder_blocks = [
        (10, 70, 8, 12, '64', 'lightblue'),
        (20, 60, 8, 10, '128', 'lightgreen'),
        (30, 52, 8, 8, '256', 'lightyellow'),
        (40, 46, 8, 6, '512', 'lightcoral'),
    ]
    
    # Bottleneck
    bottleneck = (45, 42, 8, 4, '1024', 'orange')
    
    # Decoder path (right, upward)
    decoder_blocks = [
        (55, 46, 8, 6, '512', 'lightcoral'),
        (65, 52, 8, 8, '256', 'lightyellow'),
        (75, 60, 8, 10, '128', 'lightgreen'),
        (85, 70, 8, 12, '64', 'lightblue'),
    ]
    
    # Draw encoder
    for x, y, w, h, channels, color in encoder_blocks:
        rect = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3",
                             facecolor=color, edgecolor='black', linewidth=1.5)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, channels, fontsize=8, ha='center', va='center', weight='bold')
    
    # Draw bottleneck
    x, y, w, h, channels, color = bottleneck
    rect = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3",
                         facecolor=color, edgecolor='black', linewidth=2)
    ax.add_patch(rect)
    ax.text(x + w/2, y + h/2, channels, fontsize=8, ha='center', va='center', weight='bold')
    
    # Draw decoder
    for x, y, w, h, channels, color in decoder_blocks:
        rect = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3",
                             facecolor=color, edgecolor='black', linewidth=1.5)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, channels, fontsize=8, ha='center', va='center', weight='bold')
    
    # Draw connections (downsampling, upsampling, skip connections)
    # Downsampling arrows
    for i in range(len(encoder_blocks) - 1):
        x1 = encoder_blocks[i][0] + encoder_blocks[i][2]/2
        y1 = encoder_blocks[i][1]
        x2 = encoder_blocks[i+1][0] + encoder_blocks[i+1][2]/2
        y2 = encoder_blocks[i+1][1] + encoder_blocks[i+1][3]
        arrow = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='->', mutation_scale=15,
                               linewidth=1.5, color='red')
        ax.add_patch(arrow)
    
    # Encoder to bottleneck
    x1 = encoder_blocks[-1][0] + encoder_blocks[-1][2]/2
    y1 = encoder_blocks[-1][1]
    x2 = bottleneck[0] + bottleneck[2]/2
    y2 = bottleneck[1] + bottleneck[3]
    arrow = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='->', mutation_scale=15,
                           linewidth=1.5, color='red')
    ax.add_patch(arrow)
    
    # Bottleneck to decoder
    x1 = bottleneck[0] + bottleneck[2]
    y1 = bottleneck[1] + bottleneck[3]/2
    x2 = decoder_blocks[0][0]
    y2 = decoder_blocks[0][1] + decoder_blocks[0][3]/2
    arrow = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='->', mutation_scale=15,
                           linewidth=1.5, color='blue')
    ax.add_patch(arrow)
    
    # Upsampling arrows
    for i in range(len(decoder_blocks) - 1):
        x1 = decoder_blocks[i][0] + decoder_blocks[i][2]/2
        y1 = decoder_blocks[i][1] + decoder_blocks[i][3]
        x2 = decoder_blocks[i+1][0] + decoder_blocks[i+1][2]/2
        y2 = decoder_blocks[i+1][1]
        arrow = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='->', mutation_scale=15,
                               linewidth=1.5, color='blue')
        ax.add_patch(arrow)
    
    # Skip connections
    for i in range(len(encoder_blocks)):
        x1 = encoder_blocks[i][0] + encoder_blocks[i][2]
        y1 = encoder_blocks[i][1] + encoder_blocks[i][3]/2
        x2 = decoder_blocks[-(i+1)][0]
        y2 = decoder_blocks[-(i+1)][1] + decoder_blocks[-(i+1)][3]/2
        arrow = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='->', mutation_scale=12,
                               linewidth=1, color='green', linestyle='--')
        ax.add_patch(arrow)
    
    # Labels
    ax.text(5, 76, 'Вход', fontsize=9, ha='right', weight='bold')
    ax.text(95, 76, 'Выход', fontsize=9, ha='left', weight='bold')
    ax.text(30, 35, 'Encoder\n(Downsampling)', fontsize=9, ha='center', weight='bold')
    ax.text(70, 35, 'Decoder\n(Upsampling)', fontsize=9, ha='center', weight='bold')
    ax.text(50, 25, 'Skip Connections', fontsize=8, ha='center', color='green', style='italic')
    
    ax.text(50, 95, 'U-Net архитектура', fontsize=13, ha='center', weight='bold')
    ax.text(50, 5, 'Encoder-Decoder со Skip Connections для точной локализации', 
           fontsize=10, ha='center', style='italic')
    
    plt.tight_layout()
    return fig_to_base64(fig)

def generate_segmentation_masks():
    """Visualize segmentation masks and overlay."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Create synthetic image
    np.random.seed(42)
    img = np.random.rand(100, 100, 3) * 0.3 + 0.5
    
    # Create masks for different objects
    y, x = np.ogrid[:100, :100]
    mask_person = ((x - 30)**2 + (y - 40)**2 <= 20**2).astype(float)
    mask_car = ((x - 70)**2 + (y - 60)**2 <= 15**2).astype(float)
    mask_tree = ((x - 50)**2 + (y - 75)**2 <= 12**2).astype(float)
    
    # Row 1: Original, Binary Masks, Combined
    axes[0, 0].imshow(img)
    axes[0, 0].set_title('Исходное изображение', fontweight='bold')
    axes[0, 0].axis('off')
    
    # Binary masks
    combined_mask = np.zeros((100, 100))
    combined_mask += mask_person * 1
    combined_mask += mask_car * 2
    combined_mask += mask_tree * 3
    axes[0, 1].imshow(combined_mask, cmap='tab20', vmin=0, vmax=4)
    axes[0, 1].set_title('Маска сегментации', fontweight='bold')
    axes[0, 1].axis('off')
    
    # Overlay
    overlay = img.copy()
    overlay[mask_person > 0] = [1, 0, 0]  # Red for person
    overlay[mask_car > 0] = [0, 1, 0]     # Green for car
    overlay[mask_tree > 0] = [0, 0, 1]    # Blue for tree
    axes[0, 2].imshow(overlay)
    axes[0, 2].set_title('Наложение маски', fontweight='bold')
    axes[0, 2].axis('off')
    
    # Row 2: Individual class masks
    axes[1, 0].imshow(mask_person, cmap='Reds')
    axes[1, 0].set_title('Класс: Человек', fontweight='bold')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(mask_car, cmap='Greens')
    axes[1, 1].set_title('Класс: Машина', fontweight='bold')
    axes[1, 1].axis('off')
    
    axes[1, 2].imshow(mask_tree, cmap='Blues')
    axes[1, 2].set_title('Класс: Дерево', fontweight='bold')
    axes[1, 2].axis('off')
    
    plt.suptitle('Маски сегментации и визуализация', fontsize=14, fontweight='bold', y=1.0)
    plt.tight_layout()
    
    return fig_to_base64(fig)


# ============================================================================
# KEYPOINT DETECTION & POSE ESTIMATION ILLUSTRATIONS
# ============================================================================

def generate_keypoint_skeleton():
    """Visualize keypoint detection and skeleton."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Define skeleton connections (simplified human pose)
    keypoints = {
        'head': (50, 20),
        'neck': (50, 30),
        'left_shoulder': (40, 32),
        'right_shoulder': (60, 32),
        'left_elbow': (35, 45),
        'right_elbow': (65, 45),
        'left_wrist': (32, 58),
        'right_wrist': (68, 58),
        'left_hip': (45, 55),
        'right_hip': (55, 55),
        'left_knee': (43, 70),
        'right_knee': (57, 70),
        'left_ankle': (42, 85),
        'right_ankle': (58, 85),
    }
    
    connections = [
        ('head', 'neck'),
        ('neck', 'left_shoulder'),
        ('neck', 'right_shoulder'),
        ('left_shoulder', 'left_elbow'),
        ('right_shoulder', 'right_elbow'),
        ('left_elbow', 'left_wrist'),
        ('right_elbow', 'right_wrist'),
        ('neck', 'left_hip'),
        ('neck', 'right_hip'),
        ('left_hip', 'left_knee'),
        ('right_hip', 'right_knee'),
        ('left_knee', 'left_ankle'),
        ('right_knee', 'right_ankle'),
    ]
    
    # Plot 1: Image with keypoints
    ax = axes[0]
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_aspect('equal')
    ax.invert_yaxis()
    ax.set_title('Ключевые точки', fontweight='bold')
    ax.axis('off')
    
    # Background
    ax.add_patch(Rectangle((0, 0), 100, 100, facecolor='lightgray', alpha=0.3))
    
    # Draw keypoints
    for name, (x, y) in keypoints.items():
        ax.plot(x, y, 'ro', markersize=8, markeredgecolor='darkred', markeredgewidth=2)
    
    # Plot 2: Skeleton
    ax = axes[1]
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_aspect('equal')
    ax.invert_yaxis()
    ax.set_title('Скелет (соединения)', fontweight='bold')
    ax.axis('off')
    
    # Background
    ax.add_patch(Rectangle((0, 0), 100, 100, facecolor='lightgray', alpha=0.3))
    
    # Draw connections
    for start, end in connections:
        x1, y1 = keypoints[start]
        x2, y2 = keypoints[end]
        ax.plot([x1, x2], [y1, y2], 'b-', linewidth=3, alpha=0.7)
    
    # Draw keypoints on top
    for name, (x, y) in keypoints.items():
        ax.plot(x, y, 'ro', markersize=8, markeredgecolor='darkred', markeredgewidth=2)
    
    # Plot 3: Heatmap visualization
    ax = axes[2]
    ax.set_title('Heatmap (вероятности)', fontweight='bold')
    ax.axis('off')
    
    # Create heatmap for one keypoint
    y, x = np.ogrid[:100, :100]
    kp_x, kp_y = keypoints['right_wrist']
    heatmap = np.exp(-((x - kp_x)**2 + (y - kp_y)**2) / (2 * 5**2))
    
    im = ax.imshow(heatmap, cmap='hot', interpolation='bilinear')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.plot(kp_x, kp_y, 'g*', markersize=15, markeredgecolor='white', markeredgewidth=2)
    
    plt.suptitle('Детекция ключевых точек и оценка позы', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    return fig_to_base64(fig)

def generate_pose_estimation_multi():
    """Visualize multi-person pose estimation."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Two simplified people
    person1_keypoints = {
        'head': (30, 25), 'neck': (30, 35),
        'left_shoulder': (25, 37), 'right_shoulder': (35, 37),
        'left_hip': (27, 55), 'right_hip': (33, 55),
    }
    
    person2_keypoints = {
        'head': (70, 30), 'neck': (70, 40),
        'left_shoulder': (65, 42), 'right_shoulder': (75, 42),
        'left_hip': (67, 60), 'right_hip': (73, 60),
    }
    
    connections = [
        ('head', 'neck'),
        ('neck', 'left_shoulder'),
        ('neck', 'right_shoulder'),
        ('neck', 'left_hip'),
        ('neck', 'right_hip'),
    ]
    
    # Plot 1: Top-down approach
    ax = axes[0]
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_aspect('equal')
    ax.invert_yaxis()
    ax.set_title('Top-Down подход\n(детекция → поза)', fontweight='bold')
    ax.axis('off')
    ax.add_patch(Rectangle((0, 0), 100, 100, facecolor='lightblue', alpha=0.2))
    
    # Draw bounding boxes first
    ax.add_patch(Rectangle((15, 15), 30, 55, linewidth=2, edgecolor='blue',
                           facecolor='none', linestyle='--'))
    ax.add_patch(Rectangle((55, 20), 30, 55, linewidth=2, edgecolor='green',
                           facecolor='none', linestyle='--'))
    
    # Draw person 1
    for start, end in connections:
        if start in person1_keypoints and end in person1_keypoints:
            x1, y1 = person1_keypoints[start]
            x2, y2 = person1_keypoints[end]
            ax.plot([x1, x2], [y1, y2], 'b-', linewidth=2)
    for (x, y) in person1_keypoints.values():
        ax.plot(x, y, 'bo', markersize=6)
    
    # Draw person 2
    for start, end in connections:
        if start in person2_keypoints and end in person2_keypoints:
            x1, y1 = person2_keypoints[start]
            x2, y2 = person2_keypoints[end]
            ax.plot([x1, x2], [y1, y2], 'g-', linewidth=2)
    for (x, y) in person2_keypoints.values():
        ax.plot(x, y, 'go', markersize=6)
    
    ax.text(50, 95, '1. Найти людей\n2. Найти ключевые точки', 
           fontsize=9, ha='center', bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    
    # Plot 2: Bottom-up approach
    ax = axes[1]
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_aspect('equal')
    ax.invert_yaxis()
    ax.set_title('Bottom-Up подход\n(точки → группировка)', fontweight='bold')
    ax.axis('off')
    ax.add_patch(Rectangle((0, 0), 100, 100, facecolor='lightyellow', alpha=0.2))
    
    # Draw all keypoints first
    all_keypoints = list(person1_keypoints.values()) + list(person2_keypoints.values())
    for (x, y) in all_keypoints:
        ax.plot(x, y, 'ko', markersize=6)
    
    # Then draw connections with different colors
    for start, end in connections:
        if start in person1_keypoints and end in person1_keypoints:
            x1, y1 = person1_keypoints[start]
            x2, y2 = person1_keypoints[end]
            ax.plot([x1, x2], [y1, y2], 'b-', linewidth=2, alpha=0.7)
    
    for start, end in connections:
        if start in person2_keypoints and end in person2_keypoints:
            x1, y1 = person2_keypoints[start]
            x2, y2 = person2_keypoints[end]
            ax.plot([x1, x2], [y1, y2], 'g-', linewidth=2, alpha=0.7)
    
    ax.text(50, 95, '1. Найти все точки\n2. Сгруппировать по людям',
           fontsize=9, ha='center', bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    
    plt.suptitle('Подходы к Multi-Person Pose Estimation', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    return fig_to_base64(fig)

def generate_heatmap_visualization():
    """Visualize heatmap-based keypoint detection."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Create heatmaps for different keypoints
    keypoint_positions = [
        (30, 25, 'Голова'),
        (30, 40, 'Плечо'),
        (35, 60, 'Локоть'),
        (70, 30, 'Бедро'),
        (72, 55, 'Колено'),
        (73, 75, 'Лодыжка'),
    ]
    
    for idx, (kp_x, kp_y, label) in enumerate(keypoint_positions):
        row = idx // 3
        col = idx % 3
        ax = axes[row, col]
        
        # Create Gaussian heatmap
        y, x = np.ogrid[:100, :100]
        heatmap = np.exp(-((x - kp_x)**2 + (y - kp_y)**2) / (2 * 6**2))
        
        im = ax.imshow(heatmap, cmap='hot', interpolation='bilinear')
        ax.plot(kp_x, kp_y, 'c*', markersize=12, markeredgecolor='white', markeredgewidth=2)
        ax.set_title(f'Heatmap: {label}', fontweight='bold')
        ax.axis('off')
        
        # Add confidence value
        max_conf = heatmap.max()
        ax.text(kp_x, 5, f'Conf: {max_conf:.2f}', fontsize=8, ha='center',
               bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    
    plt.suptitle('Heatmap визуализация для каждой ключевой точки', fontsize=14, fontweight='bold', y=1.0)
    plt.tight_layout()
    
    return fig_to_base64(fig)


# ============================================================================
# CNN VISUALIZATION TECHNIQUES ILLUSTRATIONS
# ============================================================================

def generate_feature_map_visualization():
    """Visualize CNN feature maps at different layers."""
    fig, axes = plt.subplots(3, 4, figsize=(14, 10))
    
    np.random.seed(42)
    
    # Simulate feature maps at different layers
    layers = ['Conv1 (Low-level)', 'Conv2 (Mid-level)', 'Conv3 (High-level)']
    
    for row, layer_name in enumerate(layers):
        for col in range(4):
            ax = axes[row, col]
            
            # Generate synthetic feature map
            if 'Low' in layer_name:
                # Low-level: edges, textures
                feature_map = np.random.randn(32, 32)
                feature_map = np.abs(feature_map)
            elif 'Mid' in layer_name:
                # Mid-level: patterns
                feature_map = np.random.randn(16, 16)
                x, y = np.meshgrid(np.linspace(-1, 1, 16), np.linspace(-1, 1, 16))
                feature_map += np.sin(4 * x) * np.cos(4 * y)
            else:
                # High-level: abstract features
                feature_map = np.random.randn(8, 8)
                feature_map = np.abs(feature_map) * 2
            
            im = ax.imshow(feature_map, cmap='viridis', interpolation='nearest')
            ax.set_title(f'{layer_name}\nФильтр {col+1}', fontsize=9)
            ax.axis('off')
    
    plt.suptitle('Feature Maps на разных слоях CNN', fontsize=14, fontweight='bold', y=1.0)
    plt.tight_layout()
    
    return fig_to_base64(fig)

def generate_filter_patterns():
    """Visualize learned filter patterns."""
    fig, axes = plt.subplots(2, 4, figsize=(14, 7))
    
    # Create different filter patterns
    filter_patterns = []
    
    # Vertical edge detector
    f1 = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
    filter_patterns.append((f1, 'Вертикальные\nрёбра'))
    
    # Horizontal edge detector
    f2 = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
    filter_patterns.append((f2, 'Горизонтальные\nрёбра'))
    
    # Diagonal edge
    f3 = np.array([[1, 0, -1], [0, 1, 0], [-1, 0, 1]])
    filter_patterns.append((f3, 'Диагональные\nрёбра'))
    
    # Blur
    f4 = np.ones((3, 3)) / 9
    filter_patterns.append((f4, 'Размытие'))
    
    # Sharpen
    f5 = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    filter_patterns.append((f5, 'Усиление\nрезкости'))
    
    # Emboss
    f6 = np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]])
    filter_patterns.append((f6, 'Тиснение'))
    
    # Gaussian
    f7 = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]]) / 16
    filter_patterns.append((f7, 'Гауссово\nразмытие'))
    
    # Random learned
    np.random.seed(42)
    f8 = np.random.randn(3, 3)
    filter_patterns.append((f8, 'Обученный\nфильтр'))
    
    # Plot filters
    for idx, (filt, title) in enumerate(filter_patterns):
        row = idx // 4
        col = idx % 4
        ax = axes[row, col]
        
        im = ax.imshow(filt, cmap='RdBu_r', vmin=-2, vmax=2)
        ax.set_title(title, fontsize=10, fontweight='bold')
        ax.axis('off')
        
        # Add values
        for i in range(filt.shape[0]):
            for j in range(filt.shape[1]):
                text = ax.text(j, i, f'{filt[i, j]:.2f}',
                             ha="center", va="center", color="black", fontsize=7)
    
    plt.suptitle('Паттерны фильтров CNN', fontsize=14, fontweight='bold', y=1.0)
    plt.tight_layout()
    
    return fig_to_base64(fig)

def generate_activation_visualization():
    """Visualize activation maps."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    np.random.seed(42)
    
    # Create base image
    base_img = np.zeros((64, 64))
    base_img[20:45, 25:40] = 1  # Rectangle
    base_img += np.random.randn(64, 64) * 0.1
    
    # Original image
    axes[0, 0].imshow(base_img, cmap='gray')
    axes[0, 0].set_title('Входное изображение', fontweight='bold')
    axes[0, 0].axis('off')
    
    # After Conv1
    conv1 = np.abs(base_img * 1.5 + np.random.randn(64, 64) * 0.2)
    axes[0, 1].imshow(conv1, cmap='viridis')
    axes[0, 1].set_title('После Conv1 + ReLU', fontweight='bold')
    axes[0, 1].axis('off')
    
    # After MaxPool
    pool1 = conv1[::2, ::2]
    axes[0, 2].imshow(pool1, cmap='viridis')
    axes[0, 2].set_title('После MaxPooling', fontweight='bold')
    axes[0, 2].axis('off')
    
    # After Conv2
    conv2 = np.abs(pool1 * 1.8 + np.random.randn(32, 32) * 0.3)
    axes[1, 0].imshow(conv2, cmap='plasma')
    axes[1, 0].set_title('После Conv2 + ReLU', fontweight='bold')
    axes[1, 0].axis('off')
    
    # After Conv3
    conv3 = np.abs(conv2[::2, ::2] * 2 + np.random.randn(16, 16) * 0.4)
    axes[1, 1].imshow(conv3, cmap='inferno')
    axes[1, 1].set_title('После Conv3 + ReLU', fontweight='bold')
    axes[1, 1].axis('off')
    
    # Final features
    final = np.abs(conv3 * 2.5)
    axes[1, 2].imshow(final, cmap='hot')
    axes[1, 2].set_title('Финальные признаки', fontweight='bold')
    axes[1, 2].axis('off')
    
    plt.suptitle('Визуализация активаций через слои CNN', fontsize=14, fontweight='bold', y=1.0)
    plt.tight_layout()
    
    return fig_to_base64(fig)

# ============================================================================
# GRAD-CAM ILLUSTRATIONS
# ============================================================================

def generate_gradcam_visualization():
    """Visualize Grad-CAM heatmap overlay."""
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    np.random.seed(42)
    
    # Create synthetic image
    img = np.random.rand(64, 64, 3) * 0.5 + 0.3
    
    # Original image
    axes[0].imshow(img)
    axes[0].set_title('Исходное\nизображение', fontweight='bold')
    axes[0].axis('off')
    
    # Feature map
    y, x = np.ogrid[:64, :64]
    feature_map = np.exp(-((x - 35)**2 + (y - 30)**2) / (2 * 12**2))
    axes[1].imshow(feature_map, cmap='viridis')
    axes[1].set_title('Feature Map\nпоследнего слоя', fontweight='bold')
    axes[1].axis('off')
    
    # Grad-CAM heatmap
    gradcam = feature_map * (1 + np.random.randn(64, 64) * 0.1)
    gradcam = np.clip(gradcam, 0, 1)
    axes[2].imshow(gradcam, cmap='jet')
    axes[2].set_title('Grad-CAM\nHeatmap', fontweight='bold')
    axes[2].axis('off')
    
    # Overlay
    overlay = img.copy()
    heatmap_colored = plt.cm.jet(gradcam)[:, :, :3]
    overlay = overlay * 0.5 + heatmap_colored * 0.5
    axes[3].imshow(overlay)
    axes[3].set_title('Наложение\n(Overlay)', fontweight='bold')
    axes[3].axis('off')
    
    plt.suptitle('Grad-CAM: Визуализация важных областей', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    return fig_to_base64(fig)

def generate_gradcam_classes():
    """Visualize Grad-CAM for different classes."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    np.random.seed(42)
    
    # Create base image
    img = np.random.rand(64, 64, 3) * 0.4 + 0.4
    
    classes = ['Кот', 'Собака', 'Птица', 'Рыба', 'Лошадь', 'Корова']
    
    # Different focus regions for each class
    focus_regions = [
        (35, 30, 10),  # Cat
        (25, 40, 12),  # Dog
        (50, 20, 8),   # Bird
        (30, 50, 9),   # Fish
        (45, 35, 11),  # Horse
        (20, 25, 10),  # Cow
    ]
    
    for idx, (class_name, (cx, cy, radius)) in enumerate(zip(classes, focus_regions)):
        row = idx // 3
        col = idx % 3
        ax = axes[row, col]
        
        # Create heatmap for this class
        y, x = np.ogrid[:64, :64]
        heatmap = np.exp(-((x - cx)**2 + (y - cy)**2) / (2 * radius**2))
        
        # Overlay
        overlay = img.copy()
        heatmap_colored = plt.cm.jet(heatmap)[:, :, :3]
        overlay = overlay * 0.6 + heatmap_colored * 0.4
        
        ax.imshow(overlay)
        ax.set_title(f'Класс: {class_name}', fontweight='bold')
        ax.axis('off')
    
    plt.suptitle('Grad-CAM для разных классов', fontsize=14, fontweight='bold', y=1.0)
    plt.tight_layout()
    
    return fig_to_base64(fig)

def generate_gradcam_layers():
    """Visualize Grad-CAM at different layers."""
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    np.random.seed(42)
    
    # Base image
    img = np.random.rand(64, 64, 3) * 0.4 + 0.4
    
    layers = ['Conv1 (ранний)', 'Conv2 (средний)', 'Conv3 (глубокий)', 'Conv4 (финальный)']
    resolutions = [64, 32, 16, 8]
    focus_spreads = [20, 15, 10, 8]
    
    for idx, (layer_name, resolution, spread) in enumerate(zip(layers, resolutions, focus_spreads)):
        ax = axes[idx]
        
        # Create heatmap at this resolution
        y, x = np.ogrid[:resolution, :resolution]
        center = resolution // 2
        heatmap = np.exp(-((x - center)**2 + (y - center)**2) / (2 * (spread/4)**2))
        
        # Resize to original size
        from scipy import ndimage
        heatmap_resized = ndimage.zoom(heatmap, 64/resolution, order=1)
        
        # Overlay
        overlay = img.copy()
        heatmap_colored = plt.cm.jet(heatmap_resized)[:, :, :3]
        overlay = overlay * 0.6 + heatmap_colored * 0.4
        
        ax.imshow(overlay)
        ax.set_title(layer_name, fontweight='bold')
        ax.axis('off')
    
    plt.suptitle('Grad-CAM на разных слоях CNN', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    return fig_to_base64(fig)


# ============================================================================
# SALIENCY MAPS ILLUSTRATIONS
# ============================================================================

def generate_saliency_methods():
    """Visualize different saliency map methods."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    np.random.seed(42)
    
    # Create base image
    img = np.random.rand(64, 64, 3) * 0.4 + 0.4
    
    # Original image
    axes[0, 0].imshow(img)
    axes[0, 0].set_title('Исходное\nизображение', fontweight='bold')
    axes[0, 0].axis('off')
    
    # Vanilla Gradient
    y, x = np.ogrid[:64, :64]
    vanilla = np.exp(-((x - 35)**2 + (y - 30)**2) / (2 * 10**2))
    vanilla += np.random.randn(64, 64) * 0.3
    vanilla = np.abs(vanilla)
    vanilla = vanilla / vanilla.max()
    axes[0, 1].imshow(vanilla, cmap='hot')
    axes[0, 1].set_title('Vanilla Gradient\n(шумный)', fontweight='bold')
    axes[0, 1].axis('off')
    
    # SmoothGrad
    smoothgrad = np.exp(-((x - 35)**2 + (y - 30)**2) / (2 * 10**2))
    smoothgrad += np.random.randn(64, 64) * 0.05  # Less noise
    smoothgrad = np.abs(smoothgrad)
    smoothgrad = smoothgrad / smoothgrad.max()
    axes[0, 2].imshow(smoothgrad, cmap='hot')
    axes[0, 2].set_title('SmoothGrad\n(сглаженный)', fontweight='bold')
    axes[0, 2].axis('off')
    
    # Integrated Gradients
    integrated = np.exp(-((x - 35)**2 + (y - 30)**2) / (2 * 9**2))
    integrated = integrated / integrated.max()
    axes[1, 0].imshow(integrated, cmap='hot')
    axes[1, 0].set_title('Integrated\nGradients', fontweight='bold')
    axes[1, 0].axis('off')
    
    # Guided Backprop
    guided = np.exp(-((x - 35)**2 + (y - 30)**2) / (2 * 11**2))
    guided = np.maximum(guided, 0)  # ReLU-like
    guided = guided / guided.max()
    axes[1, 1].imshow(guided, cmap='hot')
    axes[1, 1].set_title('Guided\nBackpropagation', fontweight='bold')
    axes[1, 1].axis('off')
    
    # Overlay on original
    overlay = img.copy()
    saliency_colored = plt.cm.hot(smoothgrad)[:, :, :3]
    overlay = overlay * 0.5 + saliency_colored * 0.5
    axes[1, 2].imshow(overlay)
    axes[1, 2].set_title('Наложение\nна изображение', fontweight='bold')
    axes[1, 2].axis('off')
    
    plt.suptitle('Методы построения Saliency Maps', fontsize=14, fontweight='bold', y=1.0)
    plt.tight_layout()
    
    return fig_to_base64(fig)

def generate_saliency_comparison():
    """Compare saliency maps for different predictions."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    
    np.random.seed(42)
    
    # Create image with two distinct regions
    img = np.ones((64, 64, 3)) * 0.5
    # Add two objects
    y, x = np.ogrid[:64, :64]
    obj1_mask = (x - 25)**2 + (y - 30)**2 <= 12**2
    obj2_mask = (x - 45)**2 + (y - 35)**2 <= 10**2
    img[obj1_mask] = [0.8, 0.3, 0.3]  # Reddish
    img[obj2_mask] = [0.3, 0.3, 0.8]  # Bluish
    
    # Original image (both subplots)
    axes[0, 0].imshow(img)
    axes[0, 0].set_title('Исходное изображение', fontweight='bold', fontsize=12)
    axes[0, 0].axis('off')
    axes[0, 0].text(32, 55, '← Объект 1', fontsize=10, color='red', weight='bold')
    axes[0, 0].text(45, 48, 'Объект 2 →', fontsize=10, color='blue', weight='bold')
    
    # Saliency for Class 1
    saliency1 = np.exp(-((x - 25)**2 + (y - 30)**2) / (2 * 8**2))
    axes[0, 1].imshow(saliency1, cmap='hot')
    axes[0, 1].set_title('Saliency для Класса 1\n(фокус на объекте 1)', fontweight='bold', fontsize=12)
    axes[0, 1].axis('off')
    
    # Original image again
    axes[1, 0].imshow(img)
    axes[1, 0].set_title('Исходное изображение', fontweight='bold', fontsize=12)
    axes[1, 0].axis('off')
    axes[1, 0].text(32, 55, '← Объект 1', fontsize=10, color='red', weight='bold')
    axes[1, 0].text(45, 48, 'Объект 2 →', fontsize=10, color='blue', weight='bold')
    
    # Saliency for Class 2
    saliency2 = np.exp(-((x - 45)**2 + (y - 35)**2) / (2 * 7**2))
    axes[1, 1].imshow(saliency2, cmap='hot')
    axes[1, 1].set_title('Saliency для Класса 2\n(фокус на объекте 2)', fontweight='bold', fontsize=12)
    axes[1, 1].axis('off')
    
    plt.suptitle('Saliency Maps для разных предсказаний', fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    return fig_to_base64(fig)

def generate_integrated_gradients_path():
    """Visualize integrated gradients interpolation path."""
    fig, axes = plt.subplots(2, 5, figsize=(16, 7))
    
    np.random.seed(42)
    
    # Baseline (black image)
    baseline = np.zeros((48, 48, 3))
    
    # Target image
    target = np.random.rand(48, 48, 3) * 0.6 + 0.3
    
    # Interpolation steps
    alphas = np.linspace(0, 1, 10)
    
    for idx, alpha in enumerate(alphas):
        row = idx // 5
        col = idx % 5
        ax = axes[row, col]
        
        # Interpolated image
        interpolated = baseline * (1 - alpha) + target * alpha
        
        ax.imshow(interpolated)
        ax.set_title(f'α = {alpha:.1f}', fontweight='bold')
        ax.axis('off')
    
    plt.suptitle('Integrated Gradients: Интерполяция от baseline к изображению', 
                fontsize=14, fontweight='bold', y=1.0)
    plt.tight_layout()
    
    return fig_to_base64(fig)

# ============================================================================
# NEURAL STYLE TRANSFER ILLUSTRATIONS
# ============================================================================

def generate_style_transfer_process():
    """Visualize neural style transfer process."""
    fig = plt.figure(figsize=(16, 10))
    
    np.random.seed(42)
    
    # Create synthetic images
    # Content image
    ax1 = plt.subplot(2, 3, 1)
    content = np.random.rand(80, 80, 3) * 0.3 + 0.5
    y, x = np.ogrid[:80, :80]
    content_obj = ((x - 40)**2 + (y - 40)**2 <= 20**2)
    content[content_obj] = [0.2, 0.6, 0.8]
    ax1.imshow(content)
    ax1.set_title('Content Image\n(содержание)', fontweight='bold', fontsize=12)
    ax1.axis('off')
    
    # Style image
    ax2 = plt.subplot(2, 3, 2)
    style = np.random.rand(80, 80, 3)
    # Add pattern
    for i in range(0, 80, 10):
        style[i:i+5, :] = [0.9, 0.3, 0.3]
    for j in range(0, 80, 10):
        style[:, j:j+5] = [0.3, 0.9, 0.3]
    ax2.imshow(style)
    ax2.set_title('Style Image\n(стиль)', fontweight='bold', fontsize=12)
    ax2.axis('off')
    
    # Output image
    ax3 = plt.subplot(2, 3, 3)
    output = content.copy()
    # Blend with style pattern
    output = output * 0.6 + style * 0.4
    ax3.imshow(output)
    ax3.set_title('Output Image\n(результат)', fontweight='bold', fontsize=12)
    ax3.axis('off')
    
    # Content loss visualization
    ax4 = plt.subplot(2, 3, 4)
    ax4.set_xlim(0, 100)
    ax4.set_ylim(0, 100)
    ax4.axis('off')
    ax4.text(50, 70, 'Content Loss:', fontsize=12, ha='center', weight='bold')
    ax4.text(50, 50, 'L_content = ||F^l - P^l||²', fontsize=11, ha='center',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
    ax4.text(50, 30, 'Сохранение структуры\nи объектов', fontsize=10, ha='center', style='italic')
    
    # Style loss visualization
    ax5 = plt.subplot(2, 3, 5)
    ax5.set_xlim(0, 100)
    ax5.set_ylim(0, 100)
    ax5.axis('off')
    ax5.text(50, 70, 'Style Loss:', fontsize=12, ha='center', weight='bold')
    ax5.text(50, 50, 'L_style = ||G^l - A^l||²', fontsize=11, ha='center',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
    ax5.text(50, 30, 'Перенос текстур\nи паттернов', fontsize=10, ha='center', style='italic')
    
    # Total loss
    ax6 = plt.subplot(2, 3, 6)
    ax6.set_xlim(0, 100)
    ax6.set_ylim(0, 100)
    ax6.axis('off')
    ax6.text(50, 70, 'Total Loss:', fontsize=12, ha='center', weight='bold')
    ax6.text(50, 50, 'L_total = α·L_content + β·L_style', fontsize=11, ha='center',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    ax6.text(50, 30, 'α, β — веса для балансировки', fontsize=10, ha='center', style='italic')
    
    plt.suptitle('Neural Style Transfer: Процесс переноса стиля', fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    return fig_to_base64(fig)

def generate_style_transfer_evolution():
    """Visualize style transfer optimization evolution."""
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    np.random.seed(42)
    
    # Content image
    content = np.random.rand(64, 64, 3) * 0.3 + 0.5
    y, x = np.ogrid[:64, :64]
    content_obj = ((x - 32)**2 + (y - 32)**2 <= 18**2)
    content[content_obj] = [0.2, 0.6, 0.8]
    
    # Style pattern
    style_pattern = np.random.rand(64, 64, 3)
    for i in range(0, 64, 8):
        style_pattern[i:i+4, :] = [0.9, 0.2, 0.2]
    
    # Evolution steps
    iterations = [0, 50, 100, 200, 500, 1000, 2000, 5000]
    
    for idx, iteration in enumerate(iterations):
        row = idx // 4
        col = idx % 4
        ax = axes[row, col]
        
        # Blend progressively
        alpha = min(iteration / 5000, 1.0)
        # Start with content, gradually add style
        result = content.copy()
        if iteration > 0:
            style_weight = alpha * 0.5
            result = result * (1 - style_weight) + style_pattern * style_weight
        
        ax.imshow(result)
        ax.set_title(f'Iteration: {iteration}', fontweight='bold')
        ax.axis('off')
    
    plt.suptitle('Эволюция Neural Style Transfer в процессе оптимизации', 
                fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    return fig_to_base64(fig)

def generate_style_transfer_weight_balance():
    """Visualize effect of content/style weight balance."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    np.random.seed(42)
    
    # Content image
    content = np.random.rand(64, 64, 3) * 0.3 + 0.5
    y, x = np.ogrid[:64, :64]
    content_obj = ((x - 32)**2 + (y - 32)**2 <= 18**2)
    content[content_obj] = [0.2, 0.6, 0.8]
    
    # Style pattern
    style_pattern = np.random.rand(64, 64, 3)
    for i in range(0, 64, 8):
        style_pattern[i:i+4, :] = [0.9, 0.2, 0.2]
    
    # Different weight combinations
    weight_configs = [
        (0.9, 0.1, 'α=0.9, β=0.1\n(больше content)'),
        (0.7, 0.3, 'α=0.7, β=0.3'),
        (0.5, 0.5, 'α=0.5, β=0.5\n(баланс)'),
        (0.3, 0.7, 'α=0.3, β=0.7'),
        (0.1, 0.9, 'α=0.1, β=0.9\n(больше style)'),
        (0.0, 1.0, 'α=0.0, β=1.0\n(только style)'),
    ]
    
    for idx, (content_weight, style_weight, title) in enumerate(weight_configs):
        row = idx // 3
        col = idx % 3
        ax = axes[row, col]
        
        # Blend with weights
        result = content * content_weight + style_pattern * style_weight
        result = np.clip(result, 0, 1)
        
        ax.imshow(result)
        ax.set_title(title, fontweight='bold', fontsize=11)
        ax.axis('off')
    
    plt.suptitle('Влияние весов α (content) и β (style)', fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    return fig_to_base64(fig)


# ============================================================================
# MAIN FUNCTION TO GENERATE ALL ILLUSTRATIONS
# ============================================================================

def generate_all_illustrations():
    """Generate all Computer Vision illustrations and return as a dictionary."""
    print("Generating Computer Vision illustrations...")
    
    illustrations = {}
    
    # Object Detection (3 illustrations)
    print("  - Object Detection illustrations...")
    illustrations['od_bounding_boxes'] = generate_object_detection_bounding_boxes()
    illustrations['od_iou'] = generate_object_detection_iou()
    illustrations['od_map'] = generate_object_detection_map()
    
    # YOLO (3 illustrations)
    print("  - YOLO illustrations...")
    illustrations['yolo_grid'] = generate_yolo_grid_detection()
    illustrations['yolo_anchors'] = generate_yolo_anchors()
    illustrations['yolo_architecture'] = generate_yolo_architecture()
    
    # Image Segmentation (3 illustrations)
    print("  - Image Segmentation illustrations...")
    illustrations['seg_types'] = generate_segmentation_types()
    illustrations['seg_unet'] = generate_unet_architecture()
    illustrations['seg_masks'] = generate_segmentation_masks()
    
    # Keypoint Detection & Pose Estimation (3 illustrations)
    print("  - Keypoint Detection illustrations...")
    illustrations['kp_skeleton'] = generate_keypoint_skeleton()
    illustrations['kp_multi'] = generate_pose_estimation_multi()
    illustrations['kp_heatmap'] = generate_heatmap_visualization()
    
    # CNN Visualization Techniques (3 illustrations)
    print("  - CNN Visualization illustrations...")
    illustrations['cnn_feature_maps'] = generate_feature_map_visualization()
    illustrations['cnn_filters'] = generate_filter_patterns()
    illustrations['cnn_activations'] = generate_activation_visualization()
    
    # Grad-CAM (3 illustrations)
    print("  - Grad-CAM illustrations...")
    illustrations['gradcam_viz'] = generate_gradcam_visualization()
    illustrations['gradcam_classes'] = generate_gradcam_classes()
    illustrations['gradcam_layers'] = generate_gradcam_layers()
    
    # Saliency Maps (3 illustrations)
    print("  - Saliency Maps illustrations...")
    illustrations['saliency_methods'] = generate_saliency_methods()
    illustrations['saliency_comparison'] = generate_saliency_comparison()
    illustrations['saliency_integrated'] = generate_integrated_gradients_path()
    
    # Neural Style Transfer (3 illustrations)
    print("  - Neural Style Transfer illustrations...")
    illustrations['nst_process'] = generate_style_transfer_process()
    illustrations['nst_evolution'] = generate_style_transfer_evolution()
    illustrations['nst_weights'] = generate_style_transfer_weight_balance()
    
    print(f"✓ Generated {len(illustrations)} illustrations for Computer Vision section")
    return illustrations

if __name__ == '__main__':
    illustrations = generate_all_illustrations()
    print("\nAll Computer Vision illustrations generated successfully!")
    print(f"Total illustrations: {len(illustrations)}")
    print("\nIllustrations keys:")
    for key in sorted(illustrations.keys()):
        print(f"  - {key}")
