#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Решение упражнения: Предобработка данных (Preprocessing)

Этот скрипт демонстрирует как:
1. Загрузить данные из CSV
2. Применить StandardScaler к числовым признакам
3. Применить LabelEncoder к категориальным признакам
4. Сохранить результаты
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder

def main():
    # Шаг 1: Загрузить входные данные
    print("Загружаем данные из input.csv...")
    df = pd.read_csv('input.csv')
    print(f"Загружено {len(df)} строк")
    print("\nПервые 5 строк исходных данных:")
    print(df.head())
    
    # Шаг 2: Разделить на признаки и целевую переменную
    print("\n" + "="*50)
    print("Шаг 2: Разделение на признаки и целевую переменную")
    print("="*50)
    
    # Признаки
    X = df[['age', 'income', 'education']]
    # Целевая переменная (не используется в этом упражнении, но выделяем)
    y = df['purchased']
    
    print(f"Признаки: {list(X.columns)}")
    print(f"Целевая переменная: purchased")
    
    # Шаг 3: Применить StandardScaler к числовым признакам
    print("\n" + "="*50)
    print("Шаг 3: StandardScaler для числовых признаков")
    print("="*50)
    
    # Извлечь числовые признаки
    numerical_features = X[['age', 'income']]
    
    # Создать и обучить scaler
    scaler = StandardScaler()
    numerical_scaled = scaler.fit_transform(numerical_features)
    
    print("\nПараметры StandardScaler:")
    print(f"Mean (age): {scaler.mean_[0]:.4f}")
    print(f"Mean (income): {scaler.mean_[1]:.4f}")
    print(f"Std (age): {scaler.scale_[0]:.4f}")
    print(f"Std (income): {scaler.scale_[1]:.4f}")
    
    # Создать DataFrame со scaled значениями
    df_scaled = pd.DataFrame(
        numerical_scaled,
        columns=['age_scaled', 'income_scaled']
    )
    
    print("\nПримеры преобразованных значений:")
    print(df_scaled.head())
    
    # Шаг 4: Применить LabelEncoder к категориальному признаку
    print("\n" + "="*50)
    print("Шаг 4: LabelEncoder для категориальных признаков")
    print("="*50)
    
    # Извлечь категориальный признак
    categorical_feature = X['education']
    
    # Создать и обучить encoder
    encoder = LabelEncoder()
    education_encoded = encoder.fit_transform(categorical_feature)
    
    print("\nКодирование education:")
    for i, class_name in enumerate(encoder.classes_):
        print(f"{class_name} -> {i}")
    
    # Добавить encoded признак
    df_scaled['education_encoded'] = education_encoded
    
    print("\nПримеры закодированных значений:")
    print(pd.DataFrame({
        'education': categorical_feature.head(),
        'education_encoded': education_encoded[:5]
    }))
    
    # Шаг 5: Округлить до 4 знаков после запятой
    print("\n" + "="*50)
    print("Шаг 5: Округление результатов")
    print("="*50)
    
    df_result = df_scaled.round(4)
    
    print("\nФинальные данные:")
    print(df_result.head())
    
    # Шаг 6: Сохранить результаты
    print("\n" + "="*50)
    print("Шаг 6: Сохранение результатов")
    print("="*50)
    
    output_file = 'answer.csv'
    df_result.to_csv(output_file, index=False)
    print(f"Результаты сохранены в {output_file}")
    
    # Проверка
    print("\n" + "="*50)
    print("Проверка результатов")
    print("="*50)
    
    # Загрузить сохраненный файл
    df_check = pd.read_csv(output_file)
    print(f"\nФайл {output_file} содержит:")
    print(f"- Строк: {len(df_check)}")
    print(f"- Столбцов: {len(df_check.columns)}")
    print(f"- Названия столбцов: {list(df_check.columns)}")
    
    print("\n✅ Упражнение выполнено успешно!")

if __name__ == "__main__":
    main()
