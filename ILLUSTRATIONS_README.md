# Classification Illustrations

This document describes the matplotlib illustrations added to the Classification section cheatsheets.

## Overview

18 high-quality matplotlib illustrations have been added to the 6 classification cheatsheets:
- Decision Trees (3 illustrations)
- Random Forest (3 illustrations)
- SVM (3 illustrations)
- Logistic Regression (3 illustrations)
- KNN (3 illustrations)
- Naive Bayes (3 illustrations)

## Scripts

### generate_classification_illustrations.py

Generates all classification illustrations using matplotlib with consistent styling.

**Features:**
- Uses seaborn style for professional appearance
- Generates 18 unique visualizations
- Returns base64-encoded PNG images
- No external file dependencies

**Usage:**
```python
python generate_classification_illustrations.py
```

### add_classification_illustrations_to_html.py

Embeds generated illustrations into HTML cheatsheet files.

**Features:**
- Automatically finds appropriate insertion points in HTML
- Creates responsive image tags with Russian captions
- Preserves existing HTML structure
- Updates all 6 classification cheatsheets

**Usage:**
```python
python add_classification_illustrations_to_html.py
```

## Illustrations Details

### Decision Trees
1. **Tree Structure**: Visual representation of decision tree splits
2. **Depth Comparison**: Overfitting demonstration with different max_depth values
3. **Feature Importance**: Bar chart showing relative feature importance

### Random Forest
1. **Ensemble Visualization**: Multiple decision trees working together
2. **OOB Error**: Out-of-bag error vs number of estimators
3. **Feature Importance**: Comparison of feature importance across ensemble

### SVM
1. **Different Kernels**: Decision boundaries for linear, RBF, and polynomial kernels
2. **Margin Visualization**: Support vectors and decision boundary with margin
3. **C Parameter**: Effect of regularization parameter on decision boundary

### Logistic Regression
1. **Sigmoid Function**: The logistic function curve
2. **Decision Boundary**: Linear decision boundary for binary classification
3. **Regularization**: Effect of L1 and L2 regularization on coefficients

### KNN
1. **Decision Boundary**: Comparison of K=1, K=5, and K=15
2. **Distance Metrics**: Euclidean vs Manhattan distance
3. **Illustration**: Visual explanation of KNN classification process

### Naive Bayes
1. **Probability Distributions**: Gaussian distributions for different classes
2. **Decision Boundaries**: Classification regions for 2D data
3. **Comparison**: Performance on different dataset types

## Technical Details

- **Image Format**: PNG (base64-encoded)
- **Embedding Method**: Inline base64 in HTML
- **Image Sizes**: 70-90% width for responsive design
- **Style**: Consistent seaborn theme with professional appearance
- **Resolution**: 300 DPI for high quality

## Regenerating Illustrations

To regenerate all illustrations:

```bash
cd /path/to/MLCheatSheets
python3 generate_classification_illustrations.py
python3 add_classification_illustrations_to_html.py
```

This will update all 6 classification cheatsheets with fresh illustrations.

## Maintenance

If you need to modify illustrations:

1. Edit the corresponding function in `generate_classification_illustrations.py`
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
