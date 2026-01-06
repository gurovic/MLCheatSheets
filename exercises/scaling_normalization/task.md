# Упражнение: Масштабирование и нормализация

## Задание

Дан файл `input.csv` с данными о людях. Необходимо применить различные методы масштабирования:

1. **MinMaxScaler** для признака 'height':
   - Масштабировать значения в диапазон [0, 1]

2. **StandardScaler** для признака 'weight':
   - Стандартизировать с использованием среднего и стандартного отклонения

3. **RobustScaler** для признака 'age':
   - Использовать медиану и IQR для масштабирования

4. **MaxAbsScaler** для признака 'income':
   - Масштабировать делением на максимальное абсолютное значение

5. Сохранить результат с колонками: height_minmax, weight_standard, age_robust, income_maxabs

Все значения округлить до 4 знаков после запятой.

## Формулы

**MinMaxScaler:** x_scaled = (x - min) / (max - min)

**StandardScaler:** x_scaled = (x - mean) / std

**RobustScaler:** x_scaled = (x - median) / IQR, где IQR = Q3 - Q1

**MaxAbsScaler:** x_scaled = x / max(abs(x))

## Ожидаемый формат выходного файла

CSV файл с колонками: height_minmax, weight_standard, age_robust, income_maxabs (округлено до 4 знаков)

## Ответ

Результат должен находиться в файле `answer.csv`
