# Упражнение: Предобработка данных (Preprocessing)

## Задание

Дан файл `input.csv` с данными о клиентах. Необходимо выполнить следующие шаги предобработки:

1. Разделить данные на признаки (X) и целевую переменную (y)
   - Признаки: age, income, education
   - Целевая переменная: purchased

2. Применить StandardScaler к числовым признакам (age, income)

3. Применить LabelEncoder к категориальному признаку (education)

4. Сохранить преобразованные данные в формате: age_scaled, income_scaled, education_encoded

## Формула StandardScaler

z = (x - μ) / σ

где:
- μ = среднее значение
- σ = стандартное отклонение

## Формула LabelEncoder

Для education: Bachelor=0, Master=1, PhD=2 (в алфавитном порядке)

## Ожидаемый формат выходного файла

CSV файл с колонками: age_scaled, income_scaled, education_encoded (округление до 4 знаков после запятой)

## Ответ

Результат должен находиться в файле `answer.csv`
