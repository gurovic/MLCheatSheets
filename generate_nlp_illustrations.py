#!/usr/bin/env python3
"""
Generate matplotlib illustrations for NLP cheatsheets.
This script creates high-quality visualizations and encodes them as base64 for inline HTML embedding.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle, FancyBboxPatch, FancyArrowPatch, Circle
from matplotlib.patches import ConnectionPatch
import base64
from io import BytesIO
import warnings
warnings.filterwarnings('ignore')

# Set consistent style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9

def fig_to_base64(fig):
    """Convert matplotlib figure to base64 string."""
    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=300)
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()
    plt.close(fig)
    return img_base64

# ============================================================================
# TRANSFORMER ARCHITECTURE ILLUSTRATIONS
# ============================================================================

def generate_transformer_architecture():
    """Generate simplified transformer architecture diagram."""
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Encoder side (left)
    encoder_x = 2
    
    # Input Embedding
    ax.add_patch(FancyBboxPatch((encoder_x - 0.5, 1), 1, 0.6, 
                                boxstyle="round,pad=0.05", 
                                facecolor='lightblue', edgecolor='blue', linewidth=2))
    ax.text(encoder_x, 1.3, 'Input\nEmbedding', ha='center', va='center', fontsize=9, weight='bold')
    
    # Positional Encoding
    ax.add_patch(FancyBboxPatch((encoder_x - 0.5, 2), 1, 0.6, 
                                boxstyle="round,pad=0.05", 
                                facecolor='lightgreen', edgecolor='green', linewidth=2))
    ax.text(encoder_x, 2.3, 'Positional\nEncoding', ha='center', va='center', fontsize=9, weight='bold')
    
    # Multi-Head Attention (Encoder)
    ax.add_patch(FancyBboxPatch((encoder_x - 0.6, 3.5), 1.2, 0.8, 
                                boxstyle="round,pad=0.05", 
                                facecolor='#FFE6E6', edgecolor='red', linewidth=2))
    ax.text(encoder_x, 3.9, 'Multi-Head\nAttention', ha='center', va='center', fontsize=9, weight='bold')
    
    # Add & Norm
    ax.add_patch(FancyBboxPatch((encoder_x - 0.5, 4.5), 1, 0.5, 
                                boxstyle="round,pad=0.05", 
                                facecolor='lightyellow', edgecolor='orange', linewidth=1.5))
    ax.text(encoder_x, 4.75, 'Add & Norm', ha='center', va='center', fontsize=8)
    
    # Feed Forward
    ax.add_patch(FancyBboxPatch((encoder_x - 0.6, 5.5), 1.2, 0.8, 
                                boxstyle="round,pad=0.05", 
                                facecolor='#E6F3FF', edgecolor='blue', linewidth=2))
    ax.text(encoder_x, 5.9, 'Feed\nForward', ha='center', va='center', fontsize=9, weight='bold')
    
    # Add & Norm
    ax.add_patch(FancyBboxPatch((encoder_x - 0.5, 6.5), 1, 0.5, 
                                boxstyle="round,pad=0.05", 
                                facecolor='lightyellow', edgecolor='orange', linewidth=1.5))
    ax.text(encoder_x, 6.75, 'Add & Norm', ha='center', va='center', fontsize=8)
    
    # Encoder label
    ax.text(encoder_x, 8, 'ENCODER', ha='center', va='center', 
            fontsize=12, weight='bold', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Decoder side (right)
    decoder_x = 8
    
    # Output Embedding
    ax.add_patch(FancyBboxPatch((decoder_x - 0.5, 1), 1, 0.6, 
                                boxstyle="round,pad=0.05", 
                                facecolor='lightblue', edgecolor='blue', linewidth=2))
    ax.text(decoder_x, 1.3, 'Output\nEmbedding', ha='center', va='center', fontsize=9, weight='bold')
    
    # Positional Encoding
    ax.add_patch(FancyBboxPatch((decoder_x - 0.5, 2), 1, 0.6, 
                                boxstyle="round,pad=0.05", 
                                facecolor='lightgreen', edgecolor='green', linewidth=2))
    ax.text(decoder_x, 2.3, 'Positional\nEncoding', ha='center', va='center', fontsize=9, weight='bold')
    
    # Masked Multi-Head Attention
    ax.add_patch(FancyBboxPatch((decoder_x - 0.6, 3.0), 1.2, 0.8, 
                                boxstyle="round,pad=0.05", 
                                facecolor='#FFE6E6', edgecolor='darkred', linewidth=2))
    ax.text(decoder_x, 3.4, 'Masked\nAttention', ha='center', va='center', fontsize=9, weight='bold')
    
    # Add & Norm
    ax.add_patch(FancyBboxPatch((decoder_x - 0.5, 4.0), 1, 0.4, 
                                boxstyle="round,pad=0.05", 
                                facecolor='lightyellow', edgecolor='orange', linewidth=1.5))
    ax.text(decoder_x, 4.2, 'Add & Norm', ha='center', va='center', fontsize=8)
    
    # Cross-Attention
    ax.add_patch(FancyBboxPatch((decoder_x - 0.6, 4.8), 1.2, 0.8, 
                                boxstyle="round,pad=0.05", 
                                facecolor='#FFD9E6', edgecolor='purple', linewidth=2))
    ax.text(decoder_x, 5.2, 'Cross-\nAttention', ha='center', va='center', fontsize=9, weight='bold')
    
    # Add & Norm
    ax.add_patch(FancyBboxPatch((decoder_x - 0.5, 5.8), 1, 0.4, 
                                boxstyle="round,pad=0.05", 
                                facecolor='lightyellow', edgecolor='orange', linewidth=1.5))
    ax.text(decoder_x, 6.0, 'Add & Norm', ha='center', va='center', fontsize=8)
    
    # Feed Forward
    ax.add_patch(FancyBboxPatch((decoder_x - 0.6, 6.5), 1.2, 0.8, 
                                boxstyle="round,pad=0.05", 
                                facecolor='#E6F3FF', edgecolor='blue', linewidth=2))
    ax.text(decoder_x, 6.9, 'Feed\nForward', ha='center', va='center', fontsize=9, weight='bold')
    
    # Add & Norm
    ax.add_patch(FancyBboxPatch((decoder_x - 0.5, 7.5), 1, 0.4, 
                                boxstyle="round,pad=0.05", 
                                facecolor='lightyellow', edgecolor='orange', linewidth=1.5))
    ax.text(decoder_x, 7.7, 'Add & Norm', ha='center', va='center', fontsize=8)
    
    # Linear + Softmax
    ax.add_patch(FancyBboxPatch((decoder_x - 0.6, 8.2), 1.2, 0.6, 
                                boxstyle="round,pad=0.05", 
                                facecolor='lightcoral', edgecolor='darkred', linewidth=2))
    ax.text(decoder_x, 8.5, 'Linear +\nSoftmax', ha='center', va='center', fontsize=9, weight='bold')
    
    # Decoder label
    ax.text(decoder_x, 9.3, 'DECODER', ha='center', va='center', 
            fontsize=12, weight='bold', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Connection from encoder to decoder
    arrow = FancyArrowPatch((encoder_x + 0.8, 6.5), (decoder_x - 0.8, 5.2),
                           arrowstyle='->', mutation_scale=20, linewidth=2,
                           color='purple', linestyle='--')
    ax.add_patch(arrow)
    ax.text(5, 6, 'Context', ha='center', va='center', fontsize=9, 
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Add N× labels
    ax.text(encoder_x - 1.2, 5, 'N×', ha='center', va='center', fontsize=11, 
            style='italic', weight='bold')
    ax.text(decoder_x - 1.2, 5.5, 'N×', ha='center', va='center', fontsize=11, 
            style='italic', weight='bold')
    
    plt.title('Transformer Architecture for NLP', fontsize=14, weight='bold', pad=20)
    
    return fig_to_base64(fig)

def generate_transformer_layers():
    """Generate visualization of transformer layer components."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create a simple flow diagram
    layers = [
        ('Input Sequence\n[tokens]', 0.5, 'lightblue'),
        ('Embedding Layer\n(d_model=512)', 1.5, 'lightgreen'),
        ('Positional Encoding\n(sin/cos)', 2.5, 'lightyellow'),
        ('Multi-Head Attention\n(8 heads)', 3.5, 'lightcoral'),
        ('Add & LayerNorm', 4.3, 'wheat'),
        ('Feed Forward\n(FFN)', 5.2, 'lightsteelblue'),
        ('Add & LayerNorm', 6.0, 'wheat'),
        ('Output\n[context vectors]', 7.0, 'lightgreen')
    ]
    
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis('off')
    
    for i, (label, y_pos, color) in enumerate(layers):
        # Draw box
        width = 4 if i in [3, 5] else 3
        x_center = 5
        ax.add_patch(FancyBboxPatch((x_center - width/2, y_pos - 0.3), width, 0.6,
                                    boxstyle="round,pad=0.1",
                                    facecolor=color, edgecolor='black', linewidth=2))
        ax.text(x_center, y_pos, label, ha='center', va='center', 
                fontsize=10, weight='bold')
        
        # Draw arrow to next layer
        if i < len(layers) - 1:
            arrow = FancyArrowPatch((x_center, y_pos + 0.3), 
                                   (x_center, layers[i+1][1] - 0.3),
                                   arrowstyle='->', mutation_scale=20,
                                   linewidth=2, color='black')
            ax.add_patch(arrow)
    
    plt.title('Transformer Encoder Layer Flow', fontsize=14, weight='bold')
    
    return fig_to_base64(fig)

# ============================================================================
# ATTENTION MECHANISM ILLUSTRATIONS
# ============================================================================

def generate_attention_mechanism():
    """Generate visualization of attention mechanism."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Sample sentence
    words = ['The', 'cat', 'sat', 'on', 'the', 'mat']
    n_words = len(words)
    
    # Create attention matrix (simplified example)
    np.random.seed(42)
    attention_weights = np.random.rand(n_words, n_words)
    
    # Make diagonal stronger (self-attention)
    for i in range(n_words):
        attention_weights[i, i] += 0.5
    
    # Make "cat" attend strongly to "sat" and "mat"
    attention_weights[1, 2] += 0.8
    attention_weights[1, 5] += 0.7
    
    # Normalize rows to sum to 1 (add small epsilon to prevent division by zero)
    attention_weights = attention_weights / (attention_weights.sum(axis=1, keepdims=True) + 1e-8)
    
    # Plot heatmap
    im = ax.imshow(attention_weights, cmap='YlOrRd', aspect='auto', vmin=0, vmax=0.5)
    
    # Set ticks
    ax.set_xticks(range(n_words))
    ax.set_yticks(range(n_words))
    ax.set_xticklabels(words, rotation=0)
    ax.set_yticklabels(words)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Attention Weight', rotation=270, labelpad=20)
    
    # Add values in cells
    for i in range(n_words):
        for j in range(n_words):
            text = ax.text(j, i, f'{attention_weights[i, j]:.2f}',
                          ha="center", va="center", color="black", fontsize=8)
    
    ax.set_xlabel('Keys (attending to)', fontsize=11, weight='bold')
    ax.set_ylabel('Queries (attending from)', fontsize=11, weight='bold')
    ax.set_title('Self-Attention Weights Matrix\n"The cat sat on the mat"', 
                fontsize=13, weight='bold', pad=15)
    
    return fig_to_base64(fig)

def generate_multi_head_attention():
    """Generate visualization of multi-head attention."""
    fig, axes = plt.subplots(2, 4, figsize=(14, 7))
    
    words = ['I', 'love', 'machine', 'learning']
    n_words = len(words)
    
    np.random.seed(42)
    
    for idx, ax in enumerate(axes.flat):
        # Generate different attention patterns for each head
        attention = np.random.rand(n_words, n_words)
        
        # Create specific patterns for different heads
        if idx == 0:  # Head focuses on syntax (nearby words)
            for i in range(n_words):
                for j in range(n_words):
                    attention[i, j] = 1.0 / (1 + abs(i - j))
        elif idx == 1:  # Head focuses on semantics (related words)
            attention[2, 3] = 1.5  # "machine" <-> "learning"
            attention[3, 2] = 1.5
        elif idx == 2:  # Head focuses on first word
            attention[:, 0] += 1.0
        
        # Normalize
        attention = attention / attention.sum(axis=1, keepdims=True)
        
        im = ax.imshow(attention, cmap='Blues', aspect='auto', vmin=0, vmax=0.6)
        ax.set_xticks(range(n_words))
        ax.set_yticks(range(n_words))
        ax.set_xticklabels(words, rotation=45, ha='right', fontsize=8)
        ax.set_yticklabels(words, fontsize=8)
        ax.set_title(f'Head {idx + 1}', fontsize=10, weight='bold')
        
        # Remove tick labels for inner plots
        if idx % 4 != 0:
            ax.set_yticklabels([])
        if idx < 4:
            ax.set_xticklabels([])
    
    plt.suptitle('Multi-Head Attention (8 Heads)\nDifferent heads learn different patterns', 
                fontsize=13, weight='bold')
    plt.tight_layout()
    
    return fig_to_base64(fig)

def generate_scaled_dot_product():
    """Generate visualization of scaled dot-product attention formula."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Title
    ax.text(5, 9, 'Scaled Dot-Product Attention', ha='center', va='center',
           fontsize=14, weight='bold', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
    # Main formula
    formula = r'$\mathrm{Attention}(Q, K, V) = \mathrm{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$'
    ax.text(5, 7.5, formula, ha='center', va='center', fontsize=16, 
           bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8, pad=0.5))
    
    # Components
    components = [
        ('Q (Query)', 1.5, 5.5, 'What am I looking for?', 'lightgreen'),
        ('K (Key)', 5, 5.5, 'What do I contain?', 'lightyellow'),
        ('V (Value)', 8.5, 5.5, 'What information do I have?', 'lightcoral'),
    ]
    
    for label, x, y, desc, color in components:
        ax.add_patch(FancyBboxPatch((x - 0.8, y - 0.4), 1.6, 0.8,
                                    boxstyle="round,pad=0.1",
                                    facecolor=color, edgecolor='black', linewidth=2))
        ax.text(x, y + 0.1, label, ha='center', va='center', fontsize=11, weight='bold')
        ax.text(x, y - 0.2, desc, ha='center', va='center', fontsize=8, style='italic')
    
    # Steps explanation
    steps = [
        '1. Compute similarity: QK^T',
        '2. Scale by √d_k (stabilize gradients)',
        '3. Apply softmax (normalize to probabilities)',
        '4. Weight values: result × V'
    ]
    
    for i, step in enumerate(steps):
        y_pos = 3.5 - i * 0.6
        ax.text(5, y_pos, step, ha='center', va='center', fontsize=9,
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Dimension note
    ax.text(5, 0.8, r'Typically: $d_k = d_v = 64$ (for 8 heads with $d_{model}=512$)',
           ha='center', va='center', fontsize=9, style='italic',
           bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))
    
    return fig_to_base64(fig)

# ============================================================================
# TOKENIZATION ILLUSTRATIONS
# ============================================================================

def generate_tokenization_comparison():
    """Generate comparison of different tokenization methods."""
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Sample text
    text = "tokenization"
    
    # Title
    ax.text(5, 9.5, 'Tokenization Methods Comparison', ha='center', va='center',
           fontsize=14, weight='bold')
    ax.text(5, 9, f'Example: "{text}"', ha='center', va='center',
           fontsize=11, style='italic')
    
    # Different tokenization methods
    methods = [
        ('Word-level', ['tokenization'], 7.5, 'lightblue'),
        ('Character-level', list(text), 6, 'lightgreen'),
        ('BPE/Subword', ['token', 'ization'], 4.5, 'lightyellow'),
        ('WordPiece', ['token', '##ization'], 3, 'lightcoral'),
    ]
    
    for method, tokens, y_pos, color in methods:
        # Method label
        ax.text(1.5, y_pos, method, ha='right', va='center', fontsize=11, weight='bold')
        
        # Draw tokens
        x_start = 2.5
        box_width = 0.4 if method == 'Character-level' else 1.2
        
        for i, token in enumerate(tokens):
            x = x_start + i * (box_width + 0.2)
            if x > 9:
                break
            ax.add_patch(FancyBboxPatch((x, y_pos - 0.25), box_width, 0.5,
                                        boxstyle="round,pad=0.05",
                                        facecolor=color, edgecolor='black', linewidth=1.5))
            font_size = 7 if method == 'Character-level' else 9
            ax.text(x + box_width/2, y_pos, token, ha='center', va='center',
                   fontsize=font_size, weight='bold')
    
    # Vocabulary size comparison
    vocab_info = [
        ('Word-level: ~50k-100k', 1.5),
        ('Character-level: ~256', 1.0),
        ('BPE: ~30k-50k', 0.5),
        ('WordPiece: ~30k', 0.0),
    ]
    
    ax.text(5, 1.8, 'Typical Vocabulary Sizes:', ha='center', va='center',
           fontsize=11, weight='bold', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.6))
    
    for info, y_offset in vocab_info:
        ax.text(5, 1.5 - y_offset * 0.35, info, ha='center', va='center', fontsize=9)
    
    return fig_to_base64(fig)

def generate_bpe_process():
    """Generate visualization of BPE algorithm."""
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Title
    ax.text(5, 9.5, 'Byte Pair Encoding (BPE) Algorithm', ha='center', va='center',
           fontsize=14, weight='bold')
    
    # Step by step process
    steps = [
        ('Step 1: Start with characters', 'l o w _ l o w _ l o w e s t', 8.5, 'lightblue'),
        ('Step 2: Find most frequent pair', 'lo w _ lo w _ lo w e s t', 7.5, 'lightgreen'),
        ('Step 3: Merge and repeat', 'low _ low _ low e s t', 6.5, 'lightyellow'),
        ('Step 4: Continue merging', 'low _ low _ lowest', 5.5, 'lightcoral'),
    ]
    
    for step_label, tokens, y_pos, color in steps:
        # Step description
        ax.text(1, y_pos + 0.3, step_label, ha='left', va='center', 
               fontsize=10, weight='bold')
        
        # Token boxes
        token_list = tokens.split()
        x_start = 1
        for i, token in enumerate(token_list):
            width = len(token) * 0.15 + 0.2
            x = x_start + i * 0.05
            ax.add_patch(FancyBboxPatch((x, y_pos - 0.2), width, 0.4,
                                        boxstyle="round,pad=0.03",
                                        facecolor=color, edgecolor='black', linewidth=1))
            ax.text(x + width/2, y_pos, token, ha='center', va='center',
                   fontsize=8, family='monospace')
            x_start += width + 0.1
    
    # Final vocabulary
    ax.text(5, 4.3, 'Final Vocabulary:', ha='center', va='center',
           fontsize=11, weight='bold', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.6))
    
    vocab = ['l', 'o', 'w', 'e', 's', 't', '_', 'lo', 'low', 'est', 'lowest']
    vocab_text = ', '.join([f'"{v}"' for v in vocab])
    ax.text(5, 3.7, vocab_text, ha='center', va='center', fontsize=9,
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Benefits
    benefits = [
        '✓ Fixed vocabulary size',
        '✓ No unknown tokens',
        '✓ Handles rare words',
        '✓ Language-agnostic',
    ]
    
    ax.text(5, 2.8, 'Benefits:', ha='center', va='center', fontsize=10, weight='bold')
    for i, benefit in enumerate(benefits):
        ax.text(5, 2.4 - i * 0.3, benefit, ha='center', va='center', fontsize=9)
    
    return fig_to_base64(fig)

def generate_subword_tokenization():
    """Generate visualization showing how subword tokenization handles unknown words."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Word-level tokenization
    ax1 = axes[0]
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 10)
    ax1.axis('off')
    
    ax1.text(5, 9, 'Word-level Tokenization', ha='center', va='center',
            fontsize=12, weight='bold', bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7))
    
    examples_word = [
        ('run', 'run', 8, '✓'),
        ('running', '[UNK]', 6.5, '✗'),
        ('runner', '[UNK]', 5, '✗'),
        ('runs', '[UNK]', 3.5, '✗'),
    ]
    
    for word, token, y_pos, status in examples_word:
        ax1.text(3, y_pos, word, ha='right', va='center', fontsize=10)
        ax1.text(4.5, y_pos, '→', ha='center', va='center', fontsize=12)
        color = 'lightgreen' if status == '✓' else 'lightcoral'
        ax1.add_patch(FancyBboxPatch((5.5, y_pos - 0.25), 2, 0.5,
                                     boxstyle="round,pad=0.05",
                                     facecolor=color, edgecolor='black', linewidth=1.5))
        ax1.text(6.5, y_pos, token, ha='center', va='center', fontsize=9, weight='bold')
        ax1.text(8, y_pos, status, ha='center', va='center', fontsize=14)
    
    ax1.text(5, 1.5, 'Problem: Many [UNK] tokens!', ha='center', va='center',
            fontsize=10, style='italic', color='red')
    
    # Subword tokenization
    ax2 = axes[1]
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 10)
    ax2.axis('off')
    
    ax2.text(5, 9, 'Subword Tokenization (BPE)', ha='center', va='center',
            fontsize=12, weight='bold', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
    
    examples_subword = [
        ('run', ['run'], 8),
        ('running', ['run', '##ning'], 6.5),
        ('runner', ['run', '##ner'], 5),
        ('runs', ['run', '##s'], 3.5),
    ]
    
    for word, tokens, y_pos in examples_subword:
        ax2.text(2, y_pos, word, ha='right', va='center', fontsize=10)
        ax2.text(3, y_pos, '→', ha='center', va='center', fontsize=12)
        
        x_start = 4
        for token in tokens:
            width = len(token) * 0.15 + 0.3
            ax2.add_patch(FancyBboxPatch((x_start, y_pos - 0.25), width, 0.5,
                                         boxstyle="round,pad=0.05",
                                         facecolor='lightgreen', edgecolor='black', linewidth=1.5))
            ax2.text(x_start + width/2, y_pos, token, ha='center', va='center',
                    fontsize=8, weight='bold', family='monospace')
            x_start += width + 0.15
        
        ax2.text(8, y_pos, '✓', ha='center', va='center', fontsize=14, color='green')
    
    ax2.text(5, 1.5, 'Solution: No [UNK] tokens!', ha='center', va='center',
            fontsize=10, style='italic', color='green', weight='bold')
    
    plt.tight_layout()
    
    return fig_to_base64(fig)

# ============================================================================
# WORD EMBEDDINGS ILLUSTRATIONS
# ============================================================================

def generate_word_embeddings_space():
    """Generate 2D visualization of word embeddings using t-SNE-like projection."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Simulate word embeddings in 2D (as if projected from high dimensions)
    np.random.seed(42)
    
    # Semantic clusters
    words_data = {
        'animals': (['cat', 'dog', 'tiger', 'lion'], [2, 2.5], 'lightcoral'),
        'fruits': (['apple', 'banana', 'orange', 'grape'], [7, 7], 'lightgreen'),
        'countries': (['USA', 'China', 'France', 'Japan'], [7, 2], 'lightblue'),
        'verbs': (['run', 'walk', 'jump', 'swim'], [2, 7], 'lightyellow'),
    }
    
    for category, (words, center, color) in words_data.items():
        # Generate cluster points
        x_coords = np.random.normal(center[0], 0.5, len(words))
        y_coords = np.random.normal(center[1], 0.5, len(words))
        
        # Plot points
        ax.scatter(x_coords, y_coords, c=color, s=200, alpha=0.7, 
                  edgecolors='black', linewidth=2, label=category.title())
        
        # Add word labels
        for word, x, y in zip(words, x_coords, y_coords):
            ax.annotate(word, (x, y), ha='center', va='center', 
                       fontsize=10, weight='bold')
    
    # Draw similarity lines (similar words)
    similarity_pairs = [
        ((2, 2.5), (2.5, 2.3), 'cat-dog'),
        ((7, 7), (7.3, 7.2), 'apple-banana'),
    ]
    
    for (x1, y1), (x2, y2), label in similarity_pairs:
        # Find closest actual points
        ax.plot([x1, x2], [y1, y2], 'k--', alpha=0.3, linewidth=1)
    
    ax.set_xlabel('Embedding Dimension 1 (t-SNE)', fontsize=11, weight='bold')
    ax.set_ylabel('Embedding Dimension 2 (t-SNE)', fontsize=11, weight='bold')
    ax.set_title('Word Embeddings in 2D Space\n(Projected from high-dimensional space)',
                fontsize=13, weight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 9)
    ax.set_ylim(0, 9)
    
    return fig_to_base64(fig)

def generate_word_analogies():
    """Generate visualization of word analogies (king - man + woman = queen)."""
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Title
    ax.text(5, 9.5, 'Word Analogies with Embeddings', ha='center', va='center',
           fontsize=14, weight='bold')
    ax.text(5, 9, 'Vector arithmetic: king - man + woman ≈ queen', ha='center', va='center',
           fontsize=11, style='italic')
    
    # Word positions (2D projection)
    words = {
        'king': (3, 7),
        'queen': (7, 7),
        'man': (3, 3),
        'woman': (7, 3),
    }
    
    colors = {
        'king': 'lightblue',
        'queen': 'lightcoral',
        'man': 'lightgreen',
        'woman': 'lightyellow',
    }
    
    # Draw vectors
    for word, (x, y) in words.items():
        # Draw point
        circle = Circle((x, y), 0.3, facecolor=colors[word], 
                       edgecolor='black', linewidth=2)
        ax.add_patch(circle)
        ax.text(x, y, word, ha='center', va='center', 
               fontsize=11, weight='bold')
    
    # Draw vector arrows
    # king -> man (subtract man from king)
    arrow1 = FancyArrowPatch(words['king'], words['man'],
                            arrowstyle='->', mutation_scale=20,
                            linewidth=2, color='red', linestyle='--')
    ax.add_patch(arrow1)
    ax.text(2.5, 5, '- man', ha='center', va='center', fontsize=10,
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # woman -> queen (add woman)
    arrow2 = FancyArrowPatch(words['woman'], words['queen'],
                            arrowstyle='->', mutation_scale=20,
                            linewidth=2, color='green', linestyle='--')
    ax.add_patch(arrow2)
    ax.text(7.5, 5, '+ woman', ha='center', va='center', fontsize=10,
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Result arrow (king -> queen)
    arrow3 = FancyArrowPatch(words['king'], words['queen'],
                            arrowstyle='->', mutation_scale=25,
                            linewidth=3, color='blue')
    ax.add_patch(arrow3)
    ax.text(5, 7.5, 'Result', ha='center', va='center', fontsize=10, weight='bold',
           bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    # Explanation
    explanation = [
        'Semantic relationships are captured as vector offsets',
        'king - man = queen - woman (royalty relationship)',
        'Similar patterns: Paris - France + Germany ≈ Berlin',
    ]
    
    for i, text in enumerate(explanation):
        y_pos = 1.8 - i * 0.4
        ax.text(5, y_pos, text, ha='center', va='center', fontsize=9,
               bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))
    
    return fig_to_base64(fig)

def generate_word2vec_architecture():
    """Generate visualization of Word2Vec architecture (CBOW vs Skip-gram)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    
    # CBOW (Continuous Bag of Words)
    ax1 = axes[0]
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 10)
    ax1.axis('off')
    
    ax1.text(5, 9.5, 'CBOW (Continuous Bag of Words)', ha='center', va='center',
            fontsize=12, weight='bold', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
    ax1.text(5, 9, 'Predict center word from context', ha='center', va='center',
            fontsize=9, style='italic')
    
    # Input words
    context_words = ['The', 'quick', 'brown', 'fox']
    y_start = 7.5
    for i, word in enumerate(context_words):
        y = y_start - i * 0.7
        ax1.add_patch(FancyBboxPatch((1, y - 0.2), 1.5, 0.4,
                                     boxstyle="round,pad=0.05",
                                     facecolor='lightgreen', edgecolor='black', linewidth=1.5))
        ax1.text(1.75, y, word, ha='center', va='center', fontsize=9, weight='bold')
        
        # Arrow to hidden layer
        arrow = FancyArrowPatch((2.5, y), (4, 5.5),
                               arrowstyle='->', mutation_scale=15,
                               linewidth=1.5, color='gray', alpha=0.5)
        ax1.add_patch(arrow)
    
    # Hidden layer (average)
    ax1.add_patch(Circle((5, 5.5), 0.8, facecolor='lightyellow', 
                         edgecolor='black', linewidth=2))
    ax1.text(5, 5.5, 'Hidden\nLayer\n(avg)', ha='center', va='center', 
            fontsize=9, weight='bold')
    
    # Output word
    arrow = FancyArrowPatch((5.8, 5.5), (7.5, 5.5),
                           arrowstyle='->', mutation_scale=20,
                           linewidth=2, color='blue')
    ax1.add_patch(arrow)
    
    ax1.add_patch(FancyBboxPatch((7.5, 5.3), 1.5, 0.4,
                                 boxstyle="round,pad=0.05",
                                 facecolor='lightcoral', edgecolor='black', linewidth=2))
    ax1.text(8.25, 5.5, 'jumps', ha='center', va='center', 
            fontsize=10, weight='bold')
    
    ax1.text(5, 3, 'Context → Target Word', ha='center', va='center',
            fontsize=10, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Skip-gram
    ax2 = axes[1]
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 10)
    ax2.axis('off')
    
    ax2.text(5, 9.5, 'Skip-gram', ha='center', va='center',
            fontsize=12, weight='bold', bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7))
    ax2.text(5, 9, 'Predict context from center word', ha='center', va='center',
            fontsize=9, style='italic')
    
    # Input word
    ax2.add_patch(FancyBboxPatch((1, 5.3), 1.5, 0.4,
                                 boxstyle="round,pad=0.05",
                                 facecolor='lightgreen', edgecolor='black', linewidth=2))
    ax2.text(1.75, 5.5, 'jumps', ha='center', va='center', 
            fontsize=10, weight='bold')
    
    # Arrow to hidden layer
    arrow = FancyArrowPatch((2.5, 5.5), (4.2, 5.5),
                           arrowstyle='->', mutation_scale=20,
                           linewidth=2, color='blue')
    ax2.add_patch(arrow)
    
    # Hidden layer
    ax2.add_patch(Circle((5, 5.5), 0.8, facecolor='lightyellow', 
                         edgecolor='black', linewidth=2))
    ax2.text(5, 5.5, 'Hidden\nLayer', ha='center', va='center', 
            fontsize=9, weight='bold')
    
    # Output words
    output_words = ['The', 'quick', 'brown', 'fox']
    y_start = 7.5
    for i, word in enumerate(output_words):
        y = y_start - i * 0.7
        
        # Arrow from hidden to output
        arrow = FancyArrowPatch((6, 5.5), (7.5, y),
                               arrowstyle='->', mutation_scale=15,
                               linewidth=1.5, color='gray', alpha=0.5)
        ax2.add_patch(arrow)
        
        ax2.add_patch(FancyBboxPatch((7.5, y - 0.2), 1.5, 0.4,
                                     boxstyle="round,pad=0.05",
                                     facecolor='lightblue', edgecolor='black', linewidth=1.5))
        ax2.text(8.25, y, word, ha='center', va='center', fontsize=9, weight='bold')
    
    ax2.text(5, 3, 'Target Word → Context', ha='center', va='center',
            fontsize=10, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    return fig_to_base64(fig)

def generate_embedding_similarity():
    """Generate visualization of cosine similarity between word embeddings."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Sample words
    words = ['king', 'queen', 'man', 'woman', 'apple', 'orange', 'car', 'train']
    n_words = len(words)
    
    # Generate similarity matrix (simulated)
    np.random.seed(42)
    similarity = np.random.rand(n_words, n_words) * 0.3
    
    # Set diagonal to 1
    for i in range(n_words):
        similarity[i, i] = 1.0
    
    # Set high similarities for related words
    # king-queen
    similarity[0, 1] = similarity[1, 0] = 0.85
    # man-woman
    similarity[2, 3] = similarity[3, 2] = 0.80
    # king-man, queen-woman (gender relationships)
    similarity[0, 2] = similarity[2, 0] = 0.70
    similarity[1, 3] = similarity[3, 1] = 0.68
    # apple-orange
    similarity[4, 5] = similarity[5, 4] = 0.82
    # car-train
    similarity[6, 7] = similarity[7, 6] = 0.75
    
    # Plot heatmap
    im = ax.imshow(similarity, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    
    # Set ticks
    ax.set_xticks(range(n_words))
    ax.set_yticks(range(n_words))
    ax.set_xticklabels(words, rotation=45, ha='right')
    ax.set_yticklabels(words)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Cosine Similarity', rotation=270, labelpad=20)
    
    # Add values in cells
    for i in range(n_words):
        for j in range(n_words):
            color = 'white' if similarity[i, j] > 0.6 else 'black'
            text = ax.text(j, i, f'{similarity[i, j]:.2f}',
                          ha="center", va="center", color=color, fontsize=9)
    
    ax.set_title('Word Embedding Similarity Matrix\nCosine Similarity between Word Vectors',
                fontsize=13, weight='bold', pad=15)
    
    return fig_to_base64(fig)

# ============================================================================
# MAIN GENERATION FUNCTION
# ============================================================================

def generate_all_illustrations():
    """Generate all NLP illustrations and return dictionary."""
    print("Generating NLP illustrations...")
    
    illustrations = {}
    
    # Transformer
    print("  Transformer architecture...")
    illustrations['transformer_architecture'] = generate_transformer_architecture()
    illustrations['transformer_layers'] = generate_transformer_layers()
    
    # Attention mechanism
    print("  Attention mechanisms...")
    illustrations['attention_mechanism'] = generate_attention_mechanism()
    illustrations['multi_head_attention'] = generate_multi_head_attention()
    illustrations['scaled_dot_product'] = generate_scaled_dot_product()
    
    # Tokenization
    print("  Tokenization...")
    illustrations['tokenization_comparison'] = generate_tokenization_comparison()
    illustrations['bpe_process'] = generate_bpe_process()
    illustrations['subword_tokenization'] = generate_subword_tokenization()
    
    # Word embeddings
    print("  Word embeddings...")
    illustrations['word_embeddings_space'] = generate_word_embeddings_space()
    illustrations['word_analogies'] = generate_word_analogies()
    illustrations['word2vec_architecture'] = generate_word2vec_architecture()
    illustrations['embedding_similarity'] = generate_embedding_similarity()
    
    print(f"Generated {len(illustrations)} illustrations!")
    return illustrations

if __name__ == '__main__':
    illustrations = generate_all_illustrations()
    
    # Save to file for inspection
    import json
    output = {k: v[:100] + '...' for k, v in illustrations.items()}  # Truncate for readability
    print("\nGenerated illustrations:")
    for key in illustrations.keys():
        print(f"  - {key}")
