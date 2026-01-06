# Упражнение: Кодирование категориальных признаков

## Задание

Дан файл `input.csv` с данными о продуктах. Необходимо применить различные методы кодирования:

1. **Label Encoding** для признака 'size':
   - Преобразовать размеры в числа: S=0, M=1, L=2 (алфавитный порядок)

2. **One-Hot Encoding** для признака 'category':
   - Создать бинарные колонки для каждой категории
   - Порядок: Clothing, Electronics, Furniture (алфавитный порядок)

3. **Target Encoding** для признака 'color':
   - Заменить каждый цвет средним значением price для этого цвета
   - Округлить до 2 знаков после запятой

4. Сохранить результат с колонками: size_encoded, category_Clothing, category_Electronics, category_Furniture, color_target_encoded, price

## Формулы

**Label Encoding:** Алфавитный порядок S=0, M=1, L=2

**One-Hot Encoding:** 1 если категория совпадает, иначе 0

**Target Encoding:** Для каждого цвета = mean(price) для всех строк с этим цветом

## Ожидаемый формат выходного файла

CSV файл с колонками: size_encoded, category_Clothing, category_Electronics, category_Furniture, color_target_encoded, price

## Ответ

Результат должен находиться в файле `answer.csv`
