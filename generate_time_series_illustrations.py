#!/usr/bin/env python3
"""
Generate matplotlib illustrations for time series cheatsheets.
This script creates high-quality visualizations and encodes them as base64 for inline HTML embedding.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
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

def generate_synthetic_time_series(n_points=200, trend=True, seasonal=True, noise=True):
    """Generate synthetic time series data with trend, seasonality, and noise."""
    np.random.seed(42)
    t = np.arange(n_points)
    
    # Base series
    y = np.zeros(n_points)
    
    # Add trend
    if trend:
        y += 0.05 * t + 10
    
    # Add seasonality
    if seasonal:
        y += 5 * np.sin(2 * np.pi * t / 12) + 3 * np.sin(2 * np.pi * t / 24)
    
    # Add noise
    if noise:
        y += np.random.normal(0, 1, n_points)
    
    return t, y

# ============================================================================
# TIME SERIES BASIC ILLUSTRATIONS
# ============================================================================

def generate_arima_forecast():
    """Generate ARIMA forecasting example."""
    from statsmodels.tsa.arima.model import ARIMA
    
    # Generate data
    t, y = generate_synthetic_time_series(n_points=150)
    train_size = 120
    y_train = y[:train_size]
    y_test = y[train_size:]
    
    # Fit ARIMA model
    model = ARIMA(y_train, order=(1, 1, 1))
    fitted = model.fit()
    
    # Forecast
    forecast = fitted.forecast(steps=len(y_test))
    forecast_index = np.arange(train_size, len(y))
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(t[:train_size], y_train, label='Обучающие данные', color='blue', linewidth=2)
    ax.plot(t[train_size:], y_test, label='Тестовые данные', color='green', linewidth=2)
    ax.plot(forecast_index, forecast, label='Прогноз ARIMA', 
            color='red', linestyle='--', linewidth=2)
    
    ax.axvline(x=train_size, color='gray', linestyle=':', linewidth=1.5, label='Граница train/test')
    ax.set_xlabel('Время')
    ax.set_ylabel('Значение')
    ax.set_title('ARIMA прогнозирование временного ряда')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return fig_to_base64(fig)

def generate_time_series_components():
    """Generate visualization of time series components."""
    t, y_full = generate_synthetic_time_series(n_points=200)
    _, y_trend = generate_synthetic_time_series(n_points=200, seasonal=False, noise=False)
    _, y_seasonal = generate_synthetic_time_series(n_points=200, trend=False, noise=False)
    _, y_noise = generate_synthetic_time_series(n_points=200, trend=False, seasonal=False)
    
    fig, axes = plt.subplots(4, 1, figsize=(10, 10))
    
    # Full series
    axes[0].plot(t, y_full, color='blue', linewidth=1.5)
    axes[0].set_ylabel('Значение')
    axes[0].set_title('Исходный ряд = Тренд + Сезонность + Шум')
    axes[0].grid(True, alpha=0.3)
    
    # Trend
    axes[1].plot(t, y_trend, color='orange', linewidth=2)
    axes[1].set_ylabel('Тренд')
    axes[1].set_title('Компонента: Тренд')
    axes[1].grid(True, alpha=0.3)
    
    # Seasonal
    axes[2].plot(t, y_seasonal, color='green', linewidth=1.5)
    axes[2].set_ylabel('Сезонность')
    axes[2].set_title('Компонента: Сезонность')
    axes[2].grid(True, alpha=0.3)
    
    # Noise
    axes[3].plot(t, y_noise, color='red', linewidth=1, alpha=0.7)
    axes[3].set_ylabel('Шум')
    axes[3].set_xlabel('Время')
    axes[3].set_title('Компонента: Остаток (Шум)')
    axes[3].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig_to_base64(fig)

# ============================================================================
# EXPONENTIAL SMOOTHING ILLUSTRATIONS
# ============================================================================

def generate_exponential_smoothing_comparison():
    """Generate comparison of different exponential smoothing methods."""
    from statsmodels.tsa.holtwinters import SimpleExpSmoothing, ExponentialSmoothing
    
    t, y = generate_synthetic_time_series(n_points=120)
    
    # Simple Exponential Smoothing
    ses = SimpleExpSmoothing(y).fit(smoothing_level=0.2, optimized=False)
    ses_forecast = ses.fittedvalues
    
    # Holt's Linear Trend
    holt = ExponentialSmoothing(y, trend='add', seasonal=None).fit()
    holt_forecast = holt.fittedvalues
    
    # Holt-Winters with seasonality
    hw = ExponentialSmoothing(y, trend='add', seasonal='add', seasonal_periods=12).fit()
    hw_forecast = hw.fittedvalues
    
    fig, axes = plt.subplots(3, 1, figsize=(10, 10))
    
    # Simple Exponential Smoothing
    axes[0].plot(t, y, label='Исходные данные', alpha=0.5, color='gray')
    axes[0].plot(t, ses_forecast, label='Simple ES', color='blue', linewidth=2)
    axes[0].set_ylabel('Значение')
    axes[0].set_title('Simple Exponential Smoothing (без тренда, без сезонности)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Holt's Linear Trend
    axes[1].plot(t, y, label='Исходные данные', alpha=0.5, color='gray')
    axes[1].plot(t, holt_forecast, label="Holt's Linear", color='orange', linewidth=2)
    axes[1].set_ylabel('Значение')
    axes[1].set_title('Holt\'s Linear Trend (с трендом)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Holt-Winters
    axes[2].plot(t, y, label='Исходные данные', alpha=0.5, color='gray')
    axes[2].plot(t, hw_forecast, label='Holt-Winters', color='green', linewidth=2)
    axes[2].set_ylabel('Значение')
    axes[2].set_xlabel('Время')
    axes[2].set_title('Holt-Winters (с трендом и сезонностью)')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig_to_base64(fig)

def generate_smoothing_alpha_comparison():
    """Generate visualization showing effect of alpha parameter."""
    from statsmodels.tsa.holtwinters import SimpleExpSmoothing
    
    t, y = generate_synthetic_time_series(n_points=100)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(t, y, label='Исходные данные', color='black', alpha=0.5, linewidth=1.5)
    
    alphas = [0.1, 0.3, 0.5, 0.9]
    colors = ['blue', 'green', 'orange', 'red']
    
    for alpha, color in zip(alphas, colors):
        ses = SimpleExpSmoothing(y).fit(smoothing_level=alpha, optimized=False)
        ax.plot(t, ses.fittedvalues, label=f'α={alpha}', color=color, linewidth=2)
    
    ax.set_xlabel('Время')
    ax.set_ylabel('Значение')
    ax.set_title('Влияние параметра сглаживания α (alpha)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return fig_to_base64(fig)

# ============================================================================
# SEASONAL DECOMPOSITION ILLUSTRATIONS
# ============================================================================

def generate_seasonal_decomposition():
    """Generate seasonal decomposition visualization."""
    from statsmodels.tsa.seasonal import seasonal_decompose
    
    # Generate time series with clear components
    t, y = generate_synthetic_time_series(n_points=200)
    
    # Create time series
    ts = pd.Series(y, index=pd.date_range(start='2020-01-01', periods=len(y), freq='M'))
    
    # Decompose
    result = seasonal_decompose(ts, model='additive', period=12)
    
    fig, axes = plt.subplots(4, 1, figsize=(10, 10))
    
    # Original
    axes[0].plot(result.observed, color='blue', linewidth=1.5)
    axes[0].set_ylabel('Наблюдаемый')
    axes[0].set_title('Сезонная декомпозиция временного ряда')
    axes[0].grid(True, alpha=0.3)
    
    # Trend
    axes[1].plot(result.trend, color='orange', linewidth=2)
    axes[1].set_ylabel('Тренд')
    axes[1].grid(True, alpha=0.3)
    
    # Seasonal
    axes[2].plot(result.seasonal, color='green', linewidth=1.5)
    axes[2].set_ylabel('Сезонность')
    axes[2].grid(True, alpha=0.3)
    
    # Residual
    axes[3].plot(result.resid, color='red', linewidth=1, alpha=0.7)
    axes[3].set_ylabel('Остаток')
    axes[3].set_xlabel('Время')
    axes[3].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig_to_base64(fig)

def generate_additive_vs_multiplicative():
    """Generate comparison of additive vs multiplicative decomposition."""
    from statsmodels.tsa.seasonal import seasonal_decompose
    
    # Generate data with increasing seasonal amplitude
    np.random.seed(42)
    t = np.arange(120)
    trend = 0.1 * t + 10
    seasonal = (1 + 0.01 * t) * 5 * np.sin(2 * np.pi * t / 12)  # Increasing amplitude
    y = trend + seasonal + np.random.normal(0, 0.5, 120)
    
    ts = pd.Series(y, index=pd.date_range(start='2020-01-01', periods=len(y), freq='M'))
    
    # Additive decomposition
    result_add = seasonal_decompose(ts, model='additive', period=12)
    
    # Multiplicative decomposition (need positive values)
    ts_pos = ts - ts.min() + 1
    result_mult = seasonal_decompose(ts_pos, model='multiplicative', period=12)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Additive
    axes[0, 0].plot(result_add.observed, color='blue', linewidth=1.5)
    axes[0, 0].set_title('Аддитивная: Исходный ряд')
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].plot(result_add.seasonal, color='green', linewidth=1.5)
    axes[0, 1].set_title('Аддитивная: Сезонная компонента')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Multiplicative
    axes[1, 0].plot(result_mult.observed, color='orange', linewidth=1.5)
    axes[1, 0].set_title('Мультипликативная: Исходный ряд')
    axes[1, 0].set_xlabel('Время')
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].plot(result_mult.seasonal, color='red', linewidth=1.5)
    axes[1, 1].set_title('Мультипликативная: Сезонная компонента')
    axes[1, 1].set_xlabel('Время')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig_to_base64(fig)

# ============================================================================
# TIME SERIES VALIDATION ILLUSTRATIONS
# ============================================================================

def generate_train_test_split():
    """Generate visualization of time series train/test split."""
    t, y = generate_synthetic_time_series(n_points=100)
    
    fig, axes = plt.subplots(3, 1, figsize=(10, 8))
    
    # Simple split
    split_idx = 80
    axes[0].plot(t[:split_idx], y[:split_idx], 'b-', label='Train', linewidth=2)
    axes[0].plot(t[split_idx:], y[split_idx:], 'r-', label='Test', linewidth=2)
    axes[0].axvline(x=split_idx, color='gray', linestyle='--', linewidth=1.5)
    axes[0].set_title('Простое разделение Train/Test (80/20)')
    axes[0].set_ylabel('Значение')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Rolling window cross-validation
    axes[1].set_title('Rolling Window Cross-Validation')
    axes[1].set_ylabel('Fold')
    colors = ['blue', 'green', 'orange']
    for i, (train_end, test_end) in enumerate([(40, 50), (50, 60), (60, 70)]):
        axes[1].barh(i, train_end, left=0, height=0.5, color=colors[i], alpha=0.5, label=f'Train Fold {i+1}')
        axes[1].barh(i, test_end - train_end, left=train_end, height=0.5, color=colors[i], alpha=1.0)
    axes[1].set_xlim(0, 100)
    axes[1].set_ylim(-0.5, 2.5)
    axes[1].set_yticks([0, 1, 2])
    axes[1].set_yticklabels(['Fold 1', 'Fold 2', 'Fold 3'])
    axes[1].grid(True, alpha=0.3, axis='x')
    
    # Expanding window
    axes[2].set_title('Expanding Window Cross-Validation')
    axes[2].set_ylabel('Fold')
    axes[2].set_xlabel('Время')
    for i, (train_end, test_end) in enumerate([(40, 50), (50, 60), (60, 70)]):
        axes[2].barh(i, train_end, left=0, height=0.5, color=colors[i], alpha=0.5)
        axes[2].barh(i, test_end - train_end, left=train_end, height=0.5, color=colors[i], alpha=1.0)
    axes[2].set_xlim(0, 100)
    axes[2].set_ylim(-0.5, 2.5)
    axes[2].set_yticks([0, 1, 2])
    axes[2].set_yticklabels(['Fold 1', 'Fold 2', 'Fold 3'])
    axes[2].grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    return fig_to_base64(fig)

# ============================================================================
# TIME SERIES FEATURE ENGINEERING ILLUSTRATIONS
# ============================================================================

def generate_lag_features():
    """Generate visualization of lag features."""
    t, y = generate_synthetic_time_series(n_points=50)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(t, y, 'o-', label='Original (t)', color='blue', linewidth=2, markersize=6)
    ax.plot(t[1:], y[:-1], 's--', label='Lag 1 (t-1)', color='orange', linewidth=2, markersize=5, alpha=0.7)
    ax.plot(t[2:], y[:-2], '^:', label='Lag 2 (t-2)', color='green', linewidth=2, markersize=5, alpha=0.7)
    ax.plot(t[3:], y[:-3], 'd-.', label='Lag 3 (t-3)', color='red', linewidth=2, markersize=5, alpha=0.7)
    
    ax.set_xlabel('Время')
    ax.set_ylabel('Значение')
    ax.set_title('Лаговые признаки (Lag Features)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return fig_to_base64(fig)

def generate_rolling_window_features():
    """Generate visualization of rolling window statistics."""
    t, y = generate_synthetic_time_series(n_points=100)
    
    # Calculate rolling statistics
    window = 10
    rolling_mean = pd.Series(y).rolling(window=window).mean()
    rolling_std = pd.Series(y).rolling(window=window).std()
    
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    
    # Rolling mean
    axes[0].plot(t, y, label='Исходные данные', alpha=0.5, color='gray')
    axes[0].plot(t, rolling_mean, label=f'Скользящее среднее (окно={window})', 
                 color='blue', linewidth=2)
    axes[0].set_ylabel('Значение')
    axes[0].set_title('Скользящее среднее (Rolling Mean)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Rolling std
    axes[1].plot(t, rolling_std, label=f'Скользящее std (окно={window})', 
                 color='orange', linewidth=2)
    axes[1].set_ylabel('Стандартное отклонение')
    axes[1].set_xlabel('Время')
    axes[1].set_title('Скользящее стандартное отклонение (Rolling Std)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig_to_base64(fig)

# ============================================================================
# RNN/LSTM ILLUSTRATIONS
# ============================================================================

def generate_lstm_architecture():
    """Generate simple LSTM architecture diagram."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis('off')
    
    # Input layer
    for i in range(3):
        circle = plt.Circle((1, 1 + i*1.5), 0.3, color='lightblue', ec='black', linewidth=2)
        ax.add_patch(circle)
        ax.text(0.3, 1 + i*1.5, f't-{2-i}', fontsize=10, ha='center', va='center')
    
    # LSTM cells
    for i in range(3):
        rect = plt.Rectangle((2.5, 0.7 + i*1.5), 1.5, 0.6, 
                             color='lightgreen', ec='black', linewidth=2)
        ax.add_patch(rect)
        ax.text(3.25, 1 + i*1.5, 'LSTM', fontsize=10, ha='center', va='center', weight='bold')
        
        # Arrows from input to LSTM
        ax.arrow(1.35, 1 + i*1.5, 1.0, 0, head_width=0.15, head_length=0.15, fc='black', ec='black')
    
    # Hidden state connections
    for i in range(2):
        ax.arrow(3.25, 1.3 + i*1.5, 0, 0.9, head_width=0.2, head_length=0.15, 
                fc='orange', ec='orange', linewidth=2, linestyle='--')
    
    # Output layer
    for i in range(3):
        circle = plt.Circle((5.5, 1 + i*1.5), 0.3, color='lightcoral', ec='black', linewidth=2)
        ax.add_patch(circle)
        
        # Arrows from LSTM to output
        ax.arrow(4.05, 1 + i*1.5, 1.0, 0, head_width=0.15, head_length=0.15, fc='black', ec='black')
    
    # Dense layer
    rect = plt.Rectangle((7, 2), 1.5, 1.5, color='lightyellow', ec='black', linewidth=2)
    ax.add_patch(rect)
    ax.text(7.75, 2.75, 'Dense\nLayer', fontsize=10, ha='center', va='center', weight='bold')
    
    # Arrows to dense
    for i in range(3):
        ax.arrow(5.85, 1 + i*1.5, 0.8, 1.75 - i*0.75, head_width=0.1, head_length=0.1, 
                fc='gray', ec='gray', alpha=0.5)
    
    # Output
    circle = plt.Circle((9, 2.75), 0.3, color='lightsteelblue', ec='black', linewidth=2)
    ax.add_patch(circle)
    ax.arrow(8.55, 2.75, 0.4, 0, head_width=0.15, head_length=0.1, fc='black', ec='black')
    ax.text(9.5, 2.75, 'Прогноз', fontsize=10, ha='left', va='center')
    
    # Labels
    ax.text(1, 5.5, 'Входные данные', fontsize=11, weight='bold')
    ax.text(3.25, 5.5, 'LSTM слой', fontsize=11, weight='bold')
    ax.text(5.5, 5.5, 'Скрытые состояния', fontsize=11, weight='bold')
    
    ax.set_title('Архитектура LSTM для временных рядов', fontsize=14, weight='bold', pad=20)
    
    return fig_to_base64(fig)

def generate_lstm_prediction():
    """Generate LSTM prediction example."""
    # Simulate LSTM predictions
    t, y_true = generate_synthetic_time_series(n_points=120)
    train_size = 100
    
    # Simulate predictions with some noise
    np.random.seed(42)
    y_pred = y_true.copy()
    y_pred[train_size:] += np.random.normal(0, 0.5, len(y_true) - train_size)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(t[:train_size], y_true[:train_size], 'b-', label='Обучающие данные', linewidth=2)
    ax.plot(t[train_size:], y_true[train_size:], 'g-', label='Истинные значения', linewidth=2)
    ax.plot(t[train_size:], y_pred[train_size:], 'r--', label='Прогноз LSTM', linewidth=2)
    
    ax.axvline(x=train_size, color='gray', linestyle=':', linewidth=2, label='Граница train/test')
    ax.fill_between(t[train_size:], 
                     y_pred[train_size:] - 1, 
                     y_pred[train_size:] + 1,
                     alpha=0.2, color='red', label='Доверительный интервал')
    
    ax.set_xlabel('Время')
    ax.set_ylabel('Значение')
    ax.set_title('LSTM прогнозирование временного ряда')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return fig_to_base64(fig)

# ============================================================================
# TRANSFORMER ILLUSTRATIONS
# ============================================================================

def generate_attention_mechanism():
    """Generate attention mechanism heatmap."""
    np.random.seed(42)
    seq_len = 10
    
    # Simulate attention weights
    attention = np.random.rand(seq_len, seq_len)
    # Make it more diagonal (attending to nearby positions)
    for i in range(seq_len):
        for j in range(seq_len):
            attention[i, j] *= np.exp(-abs(i - j) / 2)
    
    # Normalize
    attention = attention / attention.sum(axis=1, keepdims=True)
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    im = ax.imshow(attention, cmap='YlOrRd', aspect='auto')
    ax.set_xlabel('Позиция в последовательности (Key)')
    ax.set_ylabel('Позиция в последовательности (Query)')
    ax.set_title('Матрица внимания (Attention Weights)\nв Transformer для временных рядов')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Вес внимания', rotation=270, labelpad=20)
    
    # Set ticks
    ax.set_xticks(range(seq_len))
    ax.set_yticks(range(seq_len))
    ax.set_xticklabels([f't-{seq_len-i}' for i in range(seq_len)])
    ax.set_yticklabels([f't-{seq_len-i}' for i in range(seq_len)])
    
    return fig_to_base64(fig)

def generate_transformer_prediction():
    """Generate Transformer prediction example."""
    t, y_true = generate_synthetic_time_series(n_points=120)
    train_size = 100
    
    # Simulate transformer predictions (slightly better than LSTM)
    np.random.seed(43)
    y_pred = y_true.copy()
    y_pred[train_size:] += np.random.normal(0, 0.3, len(y_true) - train_size)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(t[:train_size], y_true[:train_size], 'b-', label='Обучающие данные', linewidth=2)
    ax.plot(t[train_size:], y_true[train_size:], 'g-', label='Истинные значения', linewidth=2)
    ax.plot(t[train_size:], y_pred[train_size:], 'r--', label='Прогноз Transformer', linewidth=2)
    
    ax.axvline(x=train_size, color='gray', linestyle=':', linewidth=2, label='Граница train/test')
    ax.fill_between(t[train_size:], 
                     y_pred[train_size:] - 0.8, 
                     y_pred[train_size:] + 0.8,
                     alpha=0.2, color='red', label='Доверительный интервал')
    
    ax.set_xlabel('Время')
    ax.set_ylabel('Значение')
    ax.set_title('Transformer прогнозирование временного ряда')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return fig_to_base64(fig)

# ============================================================================
# MAIN GENERATION FUNCTION
# ============================================================================

def generate_all_illustrations():
    """Generate all time series illustrations and return dictionary."""
    print("Generating time series illustrations...")
    
    illustrations = {}
    
    # Basic time series
    print("  Basic time series...")
    illustrations['arima_forecast'] = generate_arima_forecast()
    illustrations['time_series_components'] = generate_time_series_components()
    
    # Exponential smoothing
    print("  Exponential smoothing...")
    illustrations['exponential_smoothing_comparison'] = generate_exponential_smoothing_comparison()
    illustrations['smoothing_alpha_comparison'] = generate_smoothing_alpha_comparison()
    
    # Seasonal decomposition
    print("  Seasonal decomposition...")
    illustrations['seasonal_decomposition'] = generate_seasonal_decomposition()
    illustrations['additive_vs_multiplicative'] = generate_additive_vs_multiplicative()
    
    # Time series validation
    print("  Time series validation...")
    illustrations['train_test_split'] = generate_train_test_split()
    
    # Feature engineering
    print("  Feature engineering...")
    illustrations['lag_features'] = generate_lag_features()
    illustrations['rolling_window_features'] = generate_rolling_window_features()
    
    # RNN/LSTM
    print("  RNN/LSTM...")
    illustrations['lstm_architecture'] = generate_lstm_architecture()
    illustrations['lstm_prediction'] = generate_lstm_prediction()
    
    # Transformers
    print("  Transformers...")
    illustrations['attention_mechanism'] = generate_attention_mechanism()
    illustrations['transformer_prediction'] = generate_transformer_prediction()
    
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
