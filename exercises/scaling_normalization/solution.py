#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Решение упражнения: Масштабирование и нормализация

Этот скрипт демонстрирует различные методы масштабирования:
1. MinMaxScaler - масштабирование в диапазон [0, 1]
2. StandardScaler - стандартизация (mean=0, std=1)
3. RobustScaler - робастное масштабирование (медиана и IQR)
4. MaxAbsScaler - масштабирование по максимальному абсолютному значению
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import (
    MinMaxScaler,
    StandardScaler,
    RobustScaler,
    MaxAbsScaler
)

def main():
    # Шаг 1: Загрузить данные
    print("="*60)
    print("Загрузка данных")
    print("="*60)
    
    df = pd.read_csv('input.csv')
    print(f"Загружено {len(df)} строк")
    print("\nИсходные данные:")
    print(df.head())
    print("\nСтатистика исходных данных:")
    print(df.describe())
    
    # Подготовить результирующий DataFrame
    result = pd.DataFrame()
    
    # Шаг 2: MinMaxScaler для 'height'
    print("\n" + "="*60)
    print("Шаг 1: MinMaxScaler для 'height'")
    print("="*60)
    print("Формула: x_scaled = (x - min) / (max - min)")
    
    minmax_scaler = MinMaxScaler()
    height_minmax = minmax_scaler.fit_transform(df[['height']])
    result['height_minmax'] = height_minmax.flatten()
    
    print(f"\nПараметры:")
    print(f"Min: {df['height'].min():.4f}")
    print(f"Max: {df['height'].max():.4f}")
    print(f"Range: {df['height'].max() - df['height'].min():.4f}")
    
    print("\nПримеры преобразования:")
    comparison = pd.DataFrame({
        'original': df['height'].head(),
        'minmax': result['height_minmax'].head()
    })
    print(comparison)
    
    # Шаг 3: StandardScaler для 'weight'
    print("\n" + "="*60)
    print("Шаг 2: StandardScaler для 'weight'")
    print("="*60)
    print("Формула: x_scaled = (x - mean) / std")
    
    standard_scaler = StandardScaler()
    weight_standard = standard_scaler.fit_transform(df[['weight']])
    result['weight_standard'] = weight_standard.flatten()
    
    print(f"\nПараметры:")
    print(f"Mean: {standard_scaler.mean_[0]:.4f}")
    print(f"Std: {standard_scaler.scale_[0]:.4f}")
    
    print("\nПримеры преобразования:")
    comparison = pd.DataFrame({
        'original': df['weight'].head(),
        'standard': result['weight_standard'].head()
    })
    print(comparison)
    
    # Шаг 4: RobustScaler для 'age'
    print("\n" + "="*60)
    print("Шаг 3: RobustScaler для 'age'")
    print("="*60)
    print("Формула: x_scaled = (x - median) / IQR, где IQR = Q3 - Q1")
    
    robust_scaler = RobustScaler()
    age_robust = robust_scaler.fit_transform(df[['age']])
    result['age_robust'] = age_robust.flatten()
    
    q1 = df['age'].quantile(0.25)
    q3 = df['age'].quantile(0.75)
    median = df['age'].median()
    iqr = q3 - q1
    
    print(f"\nПараметры:")
    print(f"Median: {median:.4f}")
    print(f"Q1 (25%): {q1:.4f}")
    print(f"Q3 (75%): {q3:.4f}")
    print(f"IQR (Q3 - Q1): {iqr:.4f}")
    
    print("\nПримеры преобразования:")
    comparison = pd.DataFrame({
        'original': df['age'].head(),
        'robust': result['age_robust'].head()
    })
    print(comparison)
    
    # Шаг 5: MaxAbsScaler для 'income'
    print("\n" + "="*60)
    print("Шаг 4: MaxAbsScaler для 'income'")
    print("="*60)
    print("Формула: x_scaled = x / max(abs(x))")
    
    maxabs_scaler = MaxAbsScaler()
    income_maxabs = maxabs_scaler.fit_transform(df[['income']])
    result['income_maxabs'] = income_maxabs.flatten()
    
    max_abs_value = df['income'].abs().max()
    
    print(f"\nПараметры:")
    print(f"Max absolute value: {max_abs_value:.4f}")
    
    print("\nПримеры преобразования:")
    comparison = pd.DataFrame({
        'original': df['income'].head(),
        'maxabs': result['income_maxabs'].head()
    })
    print(comparison)
    
    # Шаг 6: Округление и сохранение
    print("\n" + "="*60)
    print("Шаг 5: Округление и сохранение результатов")
    print("="*60)
    
    result = result.round(4)
    
    print("\nФинальные данные (первые 10 строк):")
    print(result.head(10))
    
    output_file = 'answer.csv'
    result.to_csv(output_file, index=False)
    print(f"\n✅ Результаты сохранены в {output_file}")
    
    # Проверка
    print("\n" + "="*60)
    print("Проверка результатов")
    print("="*60)
    
    df_check = pd.read_csv(output_file)
    print(f"\nФайл {output_file}:")
    print(f"- Строк: {len(df_check)}")
    print(f"- Столбцов: {len(df_check.columns)}")
    print(f"- Названия столбцов: {list(df_check.columns)}")
    
    print("\nСтатистика результатов:")
    print(df_check.describe())
    
    print("\n✅ Упражнение выполнено успешно!")
    
    # Дополнительная информация о методах
    print("\n" + "="*60)
    print("Когда использовать каждый метод?")
    print("="*60)
    
    print("""
    📊 MinMaxScaler:
    - Когда нужен определенный диапазон (например, [0, 1])
    - Чувствителен к выбросам
    - Хорош для нейронных сетей и алгоритмов, требующих ограниченного диапазона
    
    📊 StandardScaler:
    - Для данных с нормальным распределением
    - Когда важно сохранить информацию о выбросах
    - Часто используется с линейными моделями, SVM, нейросетями
    
    📊 RobustScaler:
    - Когда в данных много выбросов
    - Использует медиану и IQR, устойчив к экстремальным значениям
    - Хорош для данных с аномалиями
    
    📊 MaxAbsScaler:
    - Когда нужно сохранить разреженность данных
    - Масштабирует в диапазон [-1, 1]
    - Полезен для разреженных матриц
    """)

if __name__ == "__main__":
    main()
