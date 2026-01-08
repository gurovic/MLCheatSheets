#!/usr/bin/env python3
"""
Generate matplotlib illustrations for audio processing cheatsheets.
This script creates high-quality visualizations and encodes them as base64 for inline HTML embedding.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal
from scipy.fft import fft, fftfreq
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

def generate_synthetic_audio(duration=2.0, sample_rate=16000):
    """Generate synthetic audio signal with multiple frequency components."""
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # Create a signal with multiple frequency components
    # Simulate speech-like characteristics
    signal_data = (
        0.5 * np.sin(2 * np.pi * 200 * t) +  # Fundamental frequency
        0.3 * np.sin(2 * np.pi * 400 * t) +  # Harmonic
        0.2 * np.sin(2 * np.pi * 600 * t) +  # Harmonic
        0.15 * np.sin(2 * np.pi * 1000 * t)  # Higher frequency
    )
    
    # Add envelope to make it more speech-like
    envelope = np.exp(-t / 0.5) * (1 - np.exp(-t / 0.05))
    signal_data = signal_data * envelope
    
    # Add some noise
    noise = 0.05 * np.random.randn(len(t))
    signal_data = signal_data + noise
    
    return signal_data, sample_rate, t

# ============================================================================
# AUDIO PROCESSING SPECTROGRAMS & MFCC ILLUSTRATIONS
# ============================================================================

def generate_waveform():
    """Generate waveform visualization."""
    audio, sr, t = generate_synthetic_audio(duration=2.0)
    
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t, audio, linewidth=0.8, color='#1a5fb4')
    ax.set_xlabel('Время (с)')
    ax.set_ylabel('Амплитуда')
    ax.set_title('Звуковая волна (Waveform)')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 2.0])
    
    return fig_to_base64(fig)

def generate_spectrogram():
    """Generate spectrogram visualization."""
    audio, sr, _ = generate_synthetic_audio(duration=2.0)
    
    # Compute spectrogram
    nperseg = 512
    f, t, Sxx = signal.spectrogram(audio, sr, nperseg=nperseg, noverlap=nperseg//2)
    
    # Convert to dB scale
    Sxx_db = 10 * np.log10(Sxx + 1e-10)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.pcolormesh(t, f, Sxx_db, shading='gouraud', cmap='viridis')
    ax.set_ylabel('Частота (Hz)')
    ax.set_xlabel('Время (с)')
    ax.set_title('Спектрограмма')
    ax.set_ylim([0, 2000])  # Focus on speech frequencies
    cbar = plt.colorbar(im, ax=ax, label='Мощность (дБ)')
    
    return fig_to_base64(fig)

def mel_scale(f):
    """Convert frequency to mel scale."""
    return 2595 * np.log10(1 + f / 700)

def generate_mel_spectrogram():
    """Generate mel-spectrogram visualization."""
    audio, sr, _ = generate_synthetic_audio(duration=2.0)
    
    # Compute spectrogram
    nperseg = 512
    f, t, Sxx = signal.spectrogram(audio, sr, nperseg=nperseg, noverlap=nperseg//2)
    
    # Simulate mel-scale conversion for visualization
    # In reality, this would use mel filterbanks
    n_mels = 64
    mel_frequencies = np.linspace(mel_scale(0), mel_scale(sr/2), n_mels)
    
    # Create mel-scale spectrogram (simplified for visualization)
    Sxx_db = 10 * np.log10(Sxx + 1e-10)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # For visualization, interpolate to mel scale
    from scipy.interpolate import interp1d
    f_mel = mel_scale(f[f > 0])
    
    # Simple visualization approach
    im = ax.pcolormesh(t, np.arange(n_mels), 
                       Sxx_db[:n_mels], 
                       shading='gouraud', cmap='magma')
    ax.set_ylabel('Mel-частота')
    ax.set_xlabel('Время (с)')
    ax.set_title('Mel-Спектрограмма')
    cbar = plt.colorbar(im, ax=ax, label='Мощность (дБ)')
    
    return fig_to_base64(fig)

def generate_mfcc():
    """Generate MFCC visualization."""
    audio, sr, _ = generate_synthetic_audio(duration=2.0)
    
    # Simulate MFCC coefficients
    # In reality, this would be computed using DCT on log mel-spectrogram
    n_mfcc = 13
    n_frames = 100
    
    # Create synthetic MFCC data with realistic patterns
    np.random.seed(42)
    mfccs = np.zeros((n_mfcc, n_frames))
    
    # First coefficient (energy) - usually higher
    mfccs[0, :] = 5 + 2 * np.sin(np.linspace(0, 4*np.pi, n_frames)) + np.random.randn(n_frames) * 0.5
    
    # Other coefficients with decreasing variance
    for i in range(1, n_mfcc):
        phase = np.random.rand() * 2 * np.pi
        freq = np.random.rand() * 3 + 1
        mfccs[i, :] = (3 / (i + 1)) * np.sin(freq * np.linspace(0, 2*np.pi, n_frames) + phase) + \
                      np.random.randn(n_frames) * (1 / (i + 1))
    
    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(mfccs, aspect='auto', origin='lower', cmap='coolwarm', interpolation='bilinear')
    ax.set_ylabel('MFCC коэффициенты')
    ax.set_xlabel('Время (кадры)')
    ax.set_title('MFCC (Mel-Frequency Cepstral Coefficients)')
    ax.set_yticks(range(n_mfcc))
    ax.set_yticklabels([f'{i+1}' for i in range(n_mfcc)])
    cbar = plt.colorbar(im, ax=ax, label='Значение')
    
    return fig_to_base64(fig)

def generate_spectral_features():
    """Generate visualization of spectral features over time."""
    audio, sr, t_audio = generate_synthetic_audio(duration=2.0)
    
    # Compute features over time using sliding window
    window_size = 512
    hop_length = 256
    n_frames = (len(audio) - window_size) // hop_length + 1
    
    spectral_centroids = []
    zero_crossing_rates = []
    spectral_rolloffs = []
    
    for i in range(n_frames):
        start = i * hop_length
        end = start + window_size
        frame = audio[start:end]
        
        # Spectral centroid (simplified)
        spectrum = np.abs(fft(frame))
        freqs = fftfreq(len(frame), 1/sr)
        pos_freqs = freqs[:len(freqs)//2]
        pos_spectrum = spectrum[:len(spectrum)//2]
        
        if pos_spectrum.sum() > 0:
            centroid = np.sum(pos_freqs * pos_spectrum) / pos_spectrum.sum()
        else:
            centroid = 0
        spectral_centroids.append(centroid)
        
        # Zero crossing rate
        zcr = np.sum(np.abs(np.diff(np.sign(frame)))) / (2 * len(frame))
        zero_crossing_rates.append(zcr)
        
        # Spectral rolloff (85% of energy)
        cumsum_spectrum = np.cumsum(pos_spectrum)
        rolloff_idx = np.where(cumsum_spectrum >= 0.85 * cumsum_spectrum[-1])[0]
        if len(rolloff_idx) > 0:
            rolloff = pos_freqs[rolloff_idx[0]]
        else:
            rolloff = 0
        spectral_rolloffs.append(rolloff)
    
    t_frames = np.arange(n_frames) * hop_length / sr
    
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    
    # Spectral Centroid
    axes[0].plot(t_frames, spectral_centroids, color='#2e8b57', linewidth=2)
    axes[0].set_ylabel('Частота (Hz)')
    axes[0].set_title('Spectral Centroid (центр масс спектра)')
    axes[0].grid(True, alpha=0.3)
    
    # Zero Crossing Rate
    axes[1].plot(t_frames, zero_crossing_rates, color='#d32f2f', linewidth=2)
    axes[1].set_ylabel('ZCR')
    axes[1].set_title('Zero Crossing Rate (частота смены знака)')
    axes[1].grid(True, alpha=0.3)
    
    # Spectral Rolloff
    axes[2].plot(t_frames, spectral_rolloffs, color='#1a5fb4', linewidth=2)
    axes[2].set_ylabel('Частота (Hz)')
    axes[2].set_xlabel('Время (с)')
    axes[2].set_title('Spectral Rolloff (85% энергии)')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    return fig_to_base64(fig)

def generate_mel_scale_comparison():
    """Generate comparison between linear and mel frequency scales."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Linear scale
    freqs_linear = np.linspace(0, 8000, 100)
    ax1.plot(freqs_linear, np.ones_like(freqs_linear), 'o-', color='#1a5fb4', markersize=3)
    ax1.set_xlabel('Частота (Hz)')
    ax1.set_title('Линейная шкала частот')
    ax1.set_ylim([0.5, 1.5])
    ax1.set_yticks([])
    ax1.grid(True, alpha=0.3)
    
    # Mel scale
    freqs_mel = mel_scale(freqs_linear)
    # Convert back to show distribution
    mel_points = np.linspace(mel_scale(0), mel_scale(8000), 100)
    # Inverse mel
    freqs_from_mel = 700 * (10**(mel_points/2595) - 1)
    
    ax2.plot(freqs_from_mel, np.ones_like(freqs_from_mel), 'o-', color='#d32f2f', markersize=3)
    ax2.set_xlabel('Частота (Hz)')
    ax2.set_title('Mel-шкала частот (больше точек на низких частотах)')
    ax2.set_ylim([0.5, 1.5])
    ax2.set_yticks([])
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    return fig_to_base64(fig)

# ============================================================================
# AUDIO CLASSIFICATION ILLUSTRATIONS
# ============================================================================

def generate_audio_classification_pipeline():
    """Generate visualization of audio classification pipeline."""
    fig, axes = plt.subplots(1, 4, figsize=(14, 3))
    
    # Step 1: Raw Audio
    audio, sr, t = generate_synthetic_audio(duration=0.5)
    axes[0].plot(t, audio, linewidth=0.8, color='#1a5fb4')
    axes[0].set_title('1. Аудио сигнал')
    axes[0].set_xlabel('Время')
    axes[0].set_ylabel('Амплитуда')
    axes[0].set_xticks([])
    axes[0].set_yticks([])
    
    # Step 2: Feature Extraction (MFCC visualization)
    n_mfcc = 13
    n_frames = 20
    np.random.seed(42)
    mfccs = np.random.randn(n_mfcc, n_frames)
    axes[1].imshow(mfccs, aspect='auto', origin='lower', cmap='coolwarm')
    axes[1].set_title('2. MFCC признаки')
    axes[1].set_xlabel('Время')
    axes[1].set_ylabel('Коэфф.')
    axes[1].set_xticks([])
    axes[1].set_yticks([])
    
    # Step 3: Model (Simple representation)
    # Neural network representation
    layer_sizes = [13, 32, 32, 8]
    for i, (left_size, right_size) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
        left_positions = np.linspace(0.2, 0.8, left_size)
        right_positions = np.linspace(0.2, 0.8, right_size)
        
        # Draw connections (sample a few)
        for lp in left_positions[::max(1, left_size//5)]:
            for rp in right_positions[::max(1, right_size//3)]:
                axes[2].plot([i*0.3, (i+1)*0.3], [lp, rp], 'gray', alpha=0.2, linewidth=0.5)
        
        # Draw nodes
        if i == 0:
            axes[2].scatter([i*0.3]*left_size, left_positions, c='#1a5fb4', s=30, zorder=3)
        axes[2].scatter([(i+1)*0.3]*right_size, right_positions, c='#2e8b57', s=30, zorder=3)
    
    axes[2].set_title('3. Модель (CNN/RNN)')
    axes[2].set_xlim([-0.1, 1.0])
    axes[2].set_ylim([0, 1])
    axes[2].axis('off')
    
    # Step 4: Classification Results
    classes = ['Музыка', 'Речь', 'Шум', 'Тишина']
    probabilities = [0.05, 0.85, 0.08, 0.02]
    colors = ['#d32f2f' if p < 0.5 else '#2e8b57' for p in probabilities]
    
    axes[3].barh(classes, probabilities, color=colors, alpha=0.7)
    axes[3].set_title('4. Классификация')
    axes[3].set_xlabel('Вероятность')
    axes[3].set_xlim([0, 1])
    
    plt.tight_layout()
    
    return fig_to_base64(fig)

def generate_feature_comparison():
    """Generate comparison of different audio features for classification."""
    np.random.seed(42)
    
    # Simulate different feature types for two classes
    n_samples = 50
    
    # MFCC features
    mfcc_class1 = np.random.randn(n_samples, 2) + np.array([2, 2])
    mfcc_class2 = np.random.randn(n_samples, 2) + np.array([-2, -2])
    
    # Spectral features
    spec_class1 = np.random.randn(n_samples, 2) + np.array([1, 3])
    spec_class2 = np.random.randn(n_samples, 2) + np.array([3, -1])
    
    # Chroma features
    chroma_class1 = np.random.randn(n_samples, 2) + np.array([0, 3])
    chroma_class2 = np.random.randn(n_samples, 2) + np.array([3, 0])
    
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    
    # MFCC
    axes[0].scatter(mfcc_class1[:, 0], mfcc_class1[:, 1], c='#1a5fb4', alpha=0.6, s=50, label='Класс 1')
    axes[0].scatter(mfcc_class2[:, 0], mfcc_class2[:, 1], c='#d32f2f', alpha=0.6, s=50, label='Класс 2')
    axes[0].set_title('MFCC признаки')
    axes[0].set_xlabel('MFCC 1')
    axes[0].set_ylabel('MFCC 2')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Spectral
    axes[1].scatter(spec_class1[:, 0], spec_class1[:, 1], c='#1a5fb4', alpha=0.6, s=50, label='Класс 1')
    axes[1].scatter(spec_class2[:, 0], spec_class2[:, 1], c='#d32f2f', alpha=0.6, s=50, label='Класс 2')
    axes[1].set_title('Spectral признаки')
    axes[1].set_xlabel('Centroid')
    axes[1].set_ylabel('Rolloff')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Chroma
    axes[2].scatter(chroma_class1[:, 0], chroma_class1[:, 1], c='#1a5fb4', alpha=0.6, s=50, label='Класс 1')
    axes[2].scatter(chroma_class2[:, 0], chroma_class2[:, 1], c='#d32f2f', alpha=0.6, s=50, label='Класс 2')
    axes[2].set_title('Chroma признаки')
    axes[2].set_xlabel('Chroma 1')
    axes[2].set_ylabel('Chroma 2')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    return fig_to_base64(fig)

def generate_confusion_matrix():
    """Generate example confusion matrix for audio classification."""
    np.random.seed(42)
    
    classes = ['Музыка', 'Речь', 'Шум', 'Природа']
    n_classes = len(classes)
    
    # Create a confusion matrix with good diagonal
    confusion = np.array([
        [85, 5, 8, 2],
        [3, 92, 3, 2],
        [7, 4, 87, 2],
        [2, 3, 5, 90]
    ])
    
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(confusion, cmap='Blues', alpha=0.8)
    
    # Set ticks and labels
    ax.set_xticks(np.arange(n_classes))
    ax.set_yticks(np.arange(n_classes))
    ax.set_xticklabels(classes)
    ax.set_yticklabels(classes)
    
    # Rotate the tick labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Add text annotations
    for i in range(n_classes):
        for j in range(n_classes):
            text = ax.text(j, i, confusion[i, j],
                          ha="center", va="center", 
                          color="white" if confusion[i, j] > 50 else "black",
                          fontsize=12, fontweight='bold')
    
    ax.set_title('Confusion Matrix для классификации звуков')
    ax.set_ylabel('Истинный класс')
    ax.set_xlabel('Предсказанный класс')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Количество образцов', rotation=270, labelpad=20)
    
    plt.tight_layout()
    
    return fig_to_base64(fig)

# ============================================================================
# ASR MODELS ILLUSTRATIONS
# ============================================================================

def generate_asr_pipeline():
    """Generate ASR pipeline visualization."""
    fig, axes = plt.subplots(1, 5, figsize=(16, 3))
    
    # Step 1: Audio Input
    audio, sr, t = generate_synthetic_audio(duration=0.5)
    axes[0].plot(t, audio, linewidth=0.8, color='#1a5fb4')
    axes[0].set_title('1. Аудио\nвход')
    axes[0].set_xticks([])
    axes[0].set_yticks([])
    axes[0].set_xlabel('Время')
    
    # Step 2: Preprocessing
    nperseg = 128
    f, t_spec, Sxx = signal.spectrogram(audio, sr, nperseg=nperseg, noverlap=nperseg//2)
    Sxx_db = 10 * np.log10(Sxx + 1e-10)
    axes[1].pcolormesh(t_spec, f[:50], Sxx_db[:50], shading='gouraud', cmap='viridis')
    axes[1].set_title('2. Препроцессинг\n(Спектрограмма)')
    axes[1].set_xticks([])
    axes[1].set_yticks([])
    axes[1].set_xlabel('Время')
    
    # Step 3: Encoder (Neural network representation)
    layer_sizes = [10, 20, 15]
    x_offset = 0
    for i, (left_size, right_size) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
        left_positions = np.linspace(0.2, 0.8, left_size)
        right_positions = np.linspace(0.2, 0.8, right_size)
        
        # Draw some connections
        for lp in left_positions[::max(1, left_size//4)]:
            for rp in right_positions[::max(1, right_size//4)]:
                axes[2].plot([i*0.4, (i+1)*0.4], [lp, rp], 'gray', alpha=0.15, linewidth=0.5)
        
        if i == 0:
            axes[2].scatter([i*0.4]*left_size, left_positions, c='#1a5fb4', s=20, zorder=3)
        axes[2].scatter([(i+1)*0.4]*right_size, right_positions, c='#2e8b57', s=20, zorder=3)
    
    axes[2].set_title('3. Encoder\n(RNN/Transformer)')
    axes[2].set_xlim([-0.1, 1.0])
    axes[2].set_ylim([0, 1])
    axes[2].axis('off')
    
    # Step 4: Decoder
    layer_sizes = [15, 20, 10]
    for i, (left_size, right_size) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
        left_positions = np.linspace(0.2, 0.8, left_size)
        right_positions = np.linspace(0.2, 0.8, right_size)
        
        # Draw some connections
        for lp in left_positions[::max(1, left_size//4)]:
            for rp in right_positions[::max(1, right_size//4)]:
                axes[3].plot([i*0.4, (i+1)*0.4], [lp, rp], 'gray', alpha=0.15, linewidth=0.5)
        
        if i == 0:
            axes[3].scatter([i*0.4]*left_size, left_positions, c='#2e8b57', s=20, zorder=3)
        axes[3].scatter([(i+1)*0.4]*right_size, right_positions, c='#d32f2f', s=20, zorder=3)
    
    axes[3].set_title('4. Decoder\n(Attention)')
    axes[3].set_xlim([-0.1, 1.0])
    axes[3].set_ylim([0, 1])
    axes[3].axis('off')
    
    # Step 5: Text Output
    axes[4].text(0.5, 0.5, '"Привет\nмир"', 
                ha='center', va='center', 
                fontsize=16, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='#e6f0ff', edgecolor='#1a5fb4', linewidth=2))
    axes[4].set_title('5. Текст\nвыход')
    axes[4].set_xlim([0, 1])
    axes[4].set_ylim([0, 1])
    axes[4].axis('off')
    
    plt.tight_layout()
    
    return fig_to_base64(fig)

def generate_wer_comparison():
    """Generate WER comparison chart for different ASR models."""
    models = ['HMM-GMM', 'DeepSpeech', 'Wav2Vec 2.0', 'Whisper', 'Conformer']
    wer_values = [25.5, 12.3, 6.8, 5.2, 7.1]
    
    colors = ['#d32f2f' if wer > 15 else '#ff9800' if wer > 8 else '#2e8b57' for wer in wer_values]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(models, wer_values, color=colors, alpha=0.8)
    
    # Add value labels
    for i, (bar, wer) in enumerate(zip(bars, wer_values)):
        ax.text(wer + 0.5, i, f'{wer:.1f}%', va='center', fontweight='bold')
    
    ax.set_xlabel('WER (Word Error Rate), %')
    ax.set_title('Сравнение моделей ASR по WER (меньше = лучше)')
    ax.set_xlim([0, max(wer_values) + 5])
    ax.grid(True, axis='x', alpha=0.3)
    
    # Add reference line for "good" performance
    ax.axvline(x=10, color='green', linestyle='--', alpha=0.5, label='Хороший результат (<10%)')
    ax.legend()
    
    plt.tight_layout()
    
    return fig_to_base64(fig)

def generate_attention_heatmap():
    """Generate attention mechanism visualization for ASR."""
    np.random.seed(42)
    
    # Simulate attention weights
    input_len = 20  # Audio frames
    output_len = 8   # Text tokens
    
    # Create diagonal-ish attention pattern (typical for ASR)
    attention = np.zeros((output_len, input_len))
    for i in range(output_len):
        center = int((i / output_len) * input_len)
        for j in range(input_len):
            attention[i, j] = np.exp(-((j - center) ** 2) / 10)
    
    # Normalize
    attention = attention / attention.sum(axis=1, keepdims=True)
    
    # Add some noise
    attention = attention + np.random.rand(output_len, input_len) * 0.05
    attention = attention / attention.sum(axis=1, keepdims=True)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(attention, cmap='YlOrRd', aspect='auto', interpolation='bilinear')
    
    ax.set_xlabel('Входные фреймы (аудио)')
    ax.set_ylabel('Выходные токены (текст)')
    ax.set_title('Attention механизм в ASR модели')
    
    # Set labels
    output_tokens = ['<s>', 'При', 'вет', 'мир', '</s>', '<pad>', '<pad>', '<pad>']
    ax.set_yticks(range(output_len))
    ax.set_yticklabels(output_tokens)
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Вес внимания', rotation=270, labelpad=20)
    
    plt.tight_layout()
    
    return fig_to_base64(fig)

def generate_ctc_alignment():
    """Generate CTC alignment visualization."""
    np.random.seed(42)
    
    # Text and alignment
    text = "HELLO"
    text_with_blank = list("_H_E_L_L_O_")  # _ represents blank
    n_frames = 25
    
    # Create probability matrix
    n_chars = len(text_with_blank)
    probs = np.random.rand(n_frames, n_chars) * 0.2
    
    # Create a typical CTC alignment path
    path = []
    chars_per_frame = n_frames / len(text)
    
    for i in range(n_frames):
        # Which character should be active
        char_idx = min(int(i / chars_per_frame) * 2, n_chars - 1)
        
        # Boost probability for correct character
        probs[i, char_idx] += 0.6
        
        # Add some transition
        if char_idx > 0:
            probs[i, char_idx - 1] += 0.2
        if char_idx < n_chars - 1:
            probs[i, char_idx + 1] += 0.2
    
    # Normalize
    probs = probs / probs.sum(axis=1, keepdims=True)
    
    fig, ax = plt.subplots(figsize=(12, 5))
    im = ax.imshow(probs.T, cmap='Blues', aspect='auto', interpolation='bilinear')
    
    ax.set_xlabel('Временные фреймы')
    ax.set_ylabel('Символы (с blank)')
    ax.set_title('CTC Alignment для распознавания речи')
    ax.set_yticks(range(n_chars))
    ax.set_yticklabels(text_with_blank)
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Вероятность', rotation=270, labelpad=20)
    
    plt.tight_layout()
    
    return fig_to_base64(fig)

# ============================================================================
# MAIN GENERATION FUNCTION
# ============================================================================

def generate_all_illustrations():
    """Generate all illustrations and return as dictionary."""
    print("Generating audio processing illustrations...")
    
    illustrations = {}
    
    # Audio Processing Spectrograms & MFCC
    print("  - Waveform...")
    illustrations['waveform'] = generate_waveform()
    
    print("  - Spectrogram...")
    illustrations['spectrogram'] = generate_spectrogram()
    
    print("  - Mel-spectrogram...")
    illustrations['mel_spectrogram'] = generate_mel_spectrogram()
    
    print("  - MFCC...")
    illustrations['mfcc'] = generate_mfcc()
    
    print("  - Spectral features...")
    illustrations['spectral_features'] = generate_spectral_features()
    
    print("  - Mel scale comparison...")
    illustrations['mel_scale_comparison'] = generate_mel_scale_comparison()
    
    # Audio Classification
    print("  - Audio classification pipeline...")
    illustrations['audio_classification_pipeline'] = generate_audio_classification_pipeline()
    
    print("  - Feature comparison...")
    illustrations['feature_comparison'] = generate_feature_comparison()
    
    print("  - Confusion matrix...")
    illustrations['confusion_matrix'] = generate_confusion_matrix()
    
    # ASR Models
    print("  - ASR pipeline...")
    illustrations['asr_pipeline'] = generate_asr_pipeline()
    
    print("  - WER comparison...")
    illustrations['wer_comparison'] = generate_wer_comparison()
    
    print("  - Attention heatmap...")
    illustrations['attention_heatmap'] = generate_attention_heatmap()
    
    print("  - CTC alignment...")
    illustrations['ctc_alignment'] = generate_ctc_alignment()
    
    print(f"✓ Generated {len(illustrations)} illustrations")
    
    return illustrations

if __name__ == '__main__':
    # Test generation
    illustrations = generate_all_illustrations()
    print("\nAll illustrations generated successfully!")
    print(f"Total size: {sum(len(img) for img in illustrations.values()) / 1024:.1f} KB")
