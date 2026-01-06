# Упражнения по машинному обучению

Этот файл содержит набор упражнений для практического освоения различных методов предобработки и работы с данными в машинном обучении.

## Список упражнений

### 1. Предобработка данных (Preprocessing)

**Описание:** Применение StandardScaler к числовым признакам и LabelEncoder к категориальным признакам.

**Файлы:**
- [Входные данные](exercises/preprocessing/input.csv)
- [Условие задачи](exercises/preprocessing/task.md)
- [Ответ](exercises/preprocessing/answer.csv)

---

### 2. Обработка пропущенных значений и кодирование (Data Preprocessing: Missing Values & Encoding)

**Описание:** Заполнение пропущенных значений (mean для числовых, mode для категориальных) и применение One-Hot Encoding.

**Файлы:**
- [Входные данные](exercises/data_preprocessing_missing_encoding/input.csv)
- [Условие задачи](exercises/data_preprocessing_missing_encoding/task.md)
- [Ответ](exercises/data_preprocessing_missing_encoding/answer.csv)

---

### 3. Обработка выбросов (Outliers Handling)

**Описание:** Определение и удаление выбросов с использованием метода IQR (Interquartile Range).

**Файлы:**
- [Входные данные](exercises/outliers_handling/input.csv)
- [Условие задачи](exercises/outliers_handling/task.md)
- [Ответ](exercises/outliers_handling/answer.csv)

---

### 4. Кодирование категориальных признаков (Categorical Encoding)

**Описание:** Применение различных методов кодирования: Label Encoding, One-Hot Encoding и Target Encoding.

**Файлы:**
- [Входные данные](exercises/categorical_encoding/input.csv)
- [Условие задачи](exercises/categorical_encoding/task.md)
- [Ответ](exercises/categorical_encoding/answer.csv)

---

### 5. Масштабирование и нормализация (Scaling and Normalization)

**Описание:** Применение различных методов масштабирования: MinMaxScaler, StandardScaler, RobustScaler и MaxAbsScaler.

**Файлы:**
- [Входные данные](exercises/scaling_normalization/input.csv)
- [Условие задачи](exercises/scaling_normalization/task.md)
- [Ответ](exercises/scaling_normalization/answer.csv)

---

### 6. Выбор признаков (Feature Selection)

**Описание:** Выбор наиболее значимых признаков на основе корреляции Пирсона с целевой переменной.

**Файлы:**
- [Входные данные](exercises/feature_selection/input.csv)
- [Условие задачи](exercises/feature_selection/task.md)
- [Ответ](exercises/feature_selection/answer.csv)

---

### 7. Снижение размерности (Dimensionality Reduction)

**Описание:** Применение метода PCA (Principal Component Analysis) для снижения размерности данных.

**Файлы:**
- [Входные данные](exercises/dimensionality_reduction/input.csv)
- [Условие задачи](exercises/dimensionality_reduction/task.md)
- [Ответ](exercises/dimensionality_reduction/answer.csv)

---

### 8. Балансировка классов (Imbalanced Data)

**Описание:** Балансировка несбалансированных данных с использованием метода Random Oversampling.

**Файлы:**
- [Входные данные](exercises/imbalanced_data/input.csv)
- [Условие задачи](exercises/imbalanced_data/task.md)
- [Ответ](exercises/imbalanced_data/answer.csv)

---

## Как работать с упражнениями

1. Скачайте входной CSV файл для выбранного упражнения
2. Прочитайте условие задачи в файле task.md
3. Напишите код для решения задачи
4. Сравните ваш результат с файлом answer.csv

## Требования

Для выполнения упражнений потребуются следующие Python библиотеки:
- pandas
- numpy
- scikit-learn

Установка:
```bash
pip install pandas numpy scikit-learn
```

## Связанные чит-шиты

- [Предобработка данных (Preprocessing)](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/preprocessing_cheatsheet.html)
- [Data Preprocessing (missing values, encoding)](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/data_preprocessing_missing_encoding_cheatsheet.html)
- [Обработка выбросов](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/outliers_handling_cheatsheet.html)
- [Кодирование категориальных признаков](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/categorical_encoding_cheatsheet.html)
- [Масштабирование и нормализация](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/scaling_normalization_cheatsheet.html)
- [Выбор признаков (Feature Selection)](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/feature_selection_cheatsheet.html)
- [Снижение размерности (Dimensionality Reduction)](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/dimensionality_reduction_cheatsheet.html)
- [Балансировка классов (Imbalanced Data)](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/imbalanced_data_cheatsheet.html)

---

© 2024 MLCheatSheets. Автор: Владимир Гуровиц (школа "Летово")
