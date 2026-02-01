# 💪 Упражнения по машинному обучению

> **Практические задания для закрепления теории из [cheatsheets](README.md)**

Этот файл содержит набор упражнений для практического освоения различных методов предобработки и работы с данными в машинном обучении.

## 🎯 О практических упражнениях

Упражнения разработаны для **отработки навыков** после изучения соответствующих cheatsheets. Каждое упражнение включает:

- 📁 **Реальные данные** - CSV файлы для практики
- 📝 **Подробное условие** - что нужно сделать
- ✅ **Правильные ответы** - для самопроверки
- 💡 **Готовые решения** - с комментариями и объяснениями (для некоторых упражнений)

### ✨ Новое: Подробные решения!

Для некоторых упражнений доступны файлы `solution.py` с подробными комментариями. Используйте их для:
- 📖 Изучения best practices
- 🔍 Понимания пошагового процесса решения
- ✔️ Проверки своего подхода
- 🎓 Обучения на примерах

### Связь с курсом

Упражнения соответствуют разделам из [Дорожной карты](ROADMAP.md):
- **Этап 3 (Путь 1: Полный новичок)**: Упражнения 1-8
- **Подготовка данных**: Все упражнения полезны
- **Feature Engineering**: Упражнения 6.1-6.7

## Список упражнений

### 1. Предобработка данных (Preprocessing)

**Описание:** Применение StandardScaler к числовым признакам и LabelEncoder к категориальным признакам.

**Файлы:**
- [Входные данные](exercises/preprocessing/input.csv)
- [Условие задачи](exercises/preprocessing/task.md)
- [Ответ](exercises/preprocessing/answer.csv)
- [Решение с комментариями](exercises/preprocessing/solution.py) ✨ NEW!

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
- [Решение с комментариями](exercises/scaling_normalization/solution.py) ✨ NEW!

---

### 6. Выбор признаков (Feature Selection) - Цикл упражнений

**Описание:** Комплексный цикл упражнений по различным методам отбора признаков (Filter, Wrapper, Embedded).

**Упражнения:**

#### 6.1. Variance Threshold (Удаление низкодисперсных признаков)
- [Входные данные](exercises/feature_selection/exercise1_variance_threshold/input.csv)
- [Условие задачи](exercises/feature_selection/exercise1_variance_threshold/task.md)
- [Ответ](exercises/feature_selection/exercise1_variance_threshold/answer.csv)

#### 6.2. Mutual Information (Взаимная информация)
- [Входные данные](exercises/feature_selection/exercise2_mutual_information/input.csv)
- [Условие задачи](exercises/feature_selection/exercise2_mutual_information/task.md)
- [Ответ](exercises/feature_selection/exercise2_mutual_information/answer.csv)

#### 6.3. RFE - Recursive Feature Elimination
- [Входные данные](exercises/feature_selection/exercise3_rfe/input.csv)
- [Условие задачи](exercises/feature_selection/exercise3_rfe/task.md)
- [Ответ](exercises/feature_selection/exercise3_rfe/answer.csv)

#### 6.4. Lasso (L1 регуляризация)
- [Входные данные](exercises/feature_selection/exercise4_lasso/input.csv)
- [Условие задачи](exercises/feature_selection/exercise4_lasso/task.md)
- [Ответ](exercises/feature_selection/exercise4_lasso/answer.csv)

#### 6.5. Tree-based Feature Importance
- [Входные данные](exercises/feature_selection/exercise5_tree_importance/input.csv)
- [Условие задачи](exercises/feature_selection/exercise5_tree_importance/task.md)
- [Ответ](exercises/feature_selection/exercise5_tree_importance/answer.csv)

#### 6.6. Удаление коррелирующих признаков
- [Входные данные](exercises/feature_selection/exercise6_remove_correlated/input.csv)
- [Условие задачи](exercises/feature_selection/exercise6_remove_correlated/task.md)
- [Ответ](exercises/feature_selection/exercise6_remove_correlated/answer.csv)

#### 6.7. Корреляция с целевой переменной
- [Входные данные](exercises/feature_selection/exercise7_correlation_target/input.csv)
- [Условие задачи](exercises/feature_selection/exercise7_correlation_target/task.md)
- [Ответ](exercises/feature_selection/exercise7_correlation_target/answer.csv)

**См. также:** [README цикла упражнений](exercises/feature_selection/README.md)

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

## 📚 Как работать с упражнениями

### Рекомендуемый подход (для максимального обучения)

#### Шаг 1: Изучите теорию
Перед решением упражнения обязательно прочитайте соответствующий cheatsheet:
- Упражнение 1 → [Preprocessing cheatsheet](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/preprocessing_cheatsheet.html)
- Упражнение 2 → [Data Preprocessing cheatsheet](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/data_preprocessing_missing_encoding_cheatsheet.html)
- И так далее (ссылки в конце документа)

#### Шаг 2: Попробуйте решить самостоятельно
1. Перейдите в папку упражнения: `cd exercises/имя_упражнения/`
2. Прочитайте условие в `task.md`
3. Изучите данные в `input.csv`
4. Напишите свой код
5. Сравните результат с `answer.csv`

#### Шаг 3: Проверьте решение
- Если получилось - отлично! Переходите к следующему
- Если не получилось - изучите `solution.py` (где доступно)
- Экспериментируйте с параметрами

### Способ 1: Самостоятельное решение (рекомендуется)

```bash
cd exercises/preprocessing/
# Прочитайте task.md
# Изучите input.csv
# Напишите свой код в новом файле my_solution.py
python my_solution.py
# Сравните результат с answer.csv
```

**Когда использовать:** Вы изучили cheatsheet и хотите проверить понимание.

### Способ 2: Изучение через готовое решение

```bash
cd exercises/имя_упражнения/
# Откройте solution.py в редакторе
# Изучите код построчно с комментариями
python solution.py
# Экспериментируйте: меняйте параметры и смотрите результат
```

**Когда использовать:** Упражнение оказалось сложным или нужны подсказки.

### Способ 3: Интерактивная работа в Jupyter

```bash
# Установите jupyter если нужно
pip install jupyter

# Запустите Jupyter
jupyter notebook

# Создайте новый notebook в папке упражнения
# Копируйте код из solution.py по блокам
# Выполняйте и изучайте каждый блок отдельно
```

**Когда использовать:** Хотите экспериментировать и визуализировать промежуточные результаты.

### 🎓 Советы по решению упражнений

✅ **Решайте по порядку** - упражнения расположены по возрастанию сложности  
✅ **Не спешите** - лучше разобраться с одним упражнением, чем пробежаться по всем  
✅ **Экспериментируйте** - меняйте параметры, смотрите что произойдет  
✅ **Ведите заметки** - записывайте что узнали и что не получилось  
✅ **Используйте отладку** - добавляйте print() для понимания процесса

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

## 🔗 Навигация по курсу

### 📚 Основные разделы
- [🏠 Главная страница](README.md) - полный список cheatsheets
- [🗺️ Дорожная карта](ROADMAP.md) - структурированные пути обучения
- [🎥 Анимации](animations/README.md) - визуализации алгоритмов

### 🛠️ Для разработчиков
- [📋 Setup Guide](SETUP.md) - инструкции по установке
- [🤝 Contributing](CONTRIBUTING.md) - как внести вклад

### 💡 Нужна помощь?
- Возникли вопросы? [Создайте Issue](https://github.com/gurovic/MLCheatSheets/issues)
- Хотите обсудить? Загляните в [Discussions](https://github.com/gurovic/MLCheatSheets/discussions)

---

<div align="center">

**Практика делает совершенным! 💪**

© 2024 MLCheatSheets. Автор: Владимир Гуровиц (школа "Летово")

[⬆ Наверх](#-упражнения-по-машинному-обучению)

</div>
