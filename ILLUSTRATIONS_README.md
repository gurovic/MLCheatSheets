# ML Cheatsheets Illustrations

This document describes the matplotlib illustrations added to various cheatsheet sections.

## Classification Illustrations

18 high-quality matplotlib illustrations have been added to the 6 classification cheatsheets:
- Decision Trees (3 illustrations)
- Random Forest (3 illustrations)
- SVM (3 illustrations)
- Logistic Regression (3 illustrations)
- KNN (3 illustrations)
- Naive Bayes (3 illustrations)

### Scripts

**generate_classification_illustrations.py** - Generates all classification illustrations using matplotlib with consistent styling.

**add_classification_illustrations_to_html.py** - Embeds generated illustrations into HTML cheatsheet files.

### Illustrations Details

#### Decision Trees
1. **Tree Structure**: Visual representation of decision tree splits
2. **Depth Comparison**: Overfitting demonstration with different max_depth values
3. **Feature Importance**: Bar chart showing relative feature importance

#### Random Forest
1. **Ensemble Visualization**: Multiple decision trees working together
2. **OOB Error**: Out-of-bag error vs number of estimators
3. **Feature Importance**: Comparison of feature importance across ensemble

#### SVM
1. **Different Kernels**: Decision boundaries for linear, RBF, and polynomial kernels
2. **Margin Visualization**: Support vectors and decision boundary with margin
3. **C Parameter**: Effect of regularization parameter on decision boundary

#### Logistic Regression
1. **Sigmoid Function**: The logistic function curve
2. **Decision Boundary**: Linear decision boundary for binary classification
3. **Regularization**: Effect of L1 and L2 regularization on coefficients

#### KNN
1. **Decision Boundary**: Comparison of K=1, K=5, and K=15
2. **Distance Metrics**: Euclidean vs Manhattan distance
3. **Illustration**: Visual explanation of KNN classification process

#### Naive Bayes
1. **Probability Distributions**: Gaussian distributions for different classes
2. **Decision Boundaries**: Classification regions for 2D data
3. **Comparison**: Performance on different dataset types

---

## Computer Vision Illustrations

24 high-quality matplotlib illustrations have been added to the 8 Computer Vision cheatsheets:
- Object Detection (3 illustrations)
- YOLO (3 illustrations)
- Image Segmentation (3 illustrations)
- Keypoint Detection & Pose Estimation (3 illustrations)
- CNN Visualization Techniques (3 illustrations)
- Grad-CAM (3 illustrations)
- Saliency Maps (3 illustrations)
- Neural Style Transfer (3 illustrations)

### Scripts

**generate_computer_vision_illustrations.py** - Generates all Computer Vision illustrations using matplotlib with consistent styling.

**add_computer_vision_illustrations_to_html.py** - Embeds generated illustrations into HTML cheatsheet files.

### Illustrations Details

#### Object Detection
1. **Bounding Boxes**: Visualization of single and multiple object detection with confidence scores
2. **IoU (Intersection over Union)**: Visual comparison of good, medium, and poor bounding box overlaps
3. **mAP Metric**: Precision-Recall curves and Average Precision across different classes

#### YOLO
1. **Grid-based Detection**: YOLO's grid division and responsible cell concept with output tensor structure
2. **Anchor Boxes**: Visualization of anchor boxes at different scales (small, medium, large objects)
3. **Architecture Flow**: Multi-scale detection pipeline from input to predictions

#### Image Segmentation
1. **Segmentation Types**: Comparison of Semantic, Instance, and Panoptic segmentation
2. **U-Net Architecture**: Encoder-Decoder structure with skip connections visualization
3. **Segmentation Masks**: Visualization of masks, overlays, and per-class segmentation

#### Keypoint Detection & Pose Estimation
1. **Keypoint Skeleton**: Detection of body keypoints with skeleton connections and heatmap visualization
2. **Multi-Person Pose**: Comparison of Top-Down vs Bottom-Up approaches
3. **Heatmap Visualization**: Gaussian heatmaps for different keypoints (head, shoulder, elbow, etc.)

#### CNN Visualization Techniques
1. **Feature Maps**: Visualization of feature maps at different CNN layers (low, mid, high-level)
2. **Filter Patterns**: Learned filter patterns including edge detectors, blur, sharpen, and custom filters
3. **Activation Visualization**: Progressive activation through CNN layers from input to final features

#### Grad-CAM
1. **Grad-CAM Visualization**: Process from feature maps to heatmap overlay on original image
2. **Class-specific Visualizations**: Different activation patterns for different predicted classes
3. **Layer Comparison**: Grad-CAM at different CNN layers showing varying levels of abstraction

#### Saliency Maps
1. **Saliency Methods**: Comparison of Vanilla Gradient, SmoothGrad, Integrated Gradients, and Guided Backprop
2. **Class Comparison**: Different saliency maps highlighting different objects for different predictions
3. **Integrated Gradients Path**: Interpolation from baseline to target image visualization

#### Neural Style Transfer
1. **Transfer Process**: Visual flow from content image and style image to output, with loss functions
2. **Optimization Evolution**: Progressive style transfer over iterations (0 to 5000)
3. **Weight Balance**: Effect of different α (content) and β (style) weight combinations

### Usage

Generate and embed Computer Vision illustrations:

```bash
cd /path/to/MLCheatSheets
python3 generate_computer_vision_illustrations.py
python3 add_computer_vision_illustrations_to_html.py
```

This will update all 8 Computer Vision cheatsheets with fresh illustrations.

---

## Technical Details

- **Image Format**: PNG (base64-encoded)
- **Embedding Method**: Inline base64 in HTML
- **Image Sizes**: 90-95% width for responsive design
- **Style**: Consistent seaborn theme with professional appearance
- **Resolution**: 300 DPI for high quality

## Maintenance

If you need to modify illustrations:

1. Edit the corresponding function in the generation script
2. Run the generation script
3. Run the embedding script to update HTML files

## Dependencies

```python
numpy
matplotlib
seaborn
scikit-learn
scipy
```

All dependencies are standard Python scientific computing libraries.
