# Цикл упражнений: Выбор признаков (Feature Selection)

Этот цикл упражнений охватывает различные методы отбора признаков, основанные на материале из [cheatsheet по Feature Selection](https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/feature_selection_cheatsheet.html).

## Упражнения

### Упражнение 1: Variance Threshold
**Папка:** `exercise1_variance_threshold/`

**Метод:** Filter метод - удаление низкодисперсных признаков

**Описание:** Удаление признаков с дисперсией меньше порога (0.5)

---

### Упражнение 2: Mutual Information
**Папка:** `exercise2_mutual_information/`

**Метод:** Filter метод - оценка взаимной информации

**Описание:** Выбор топ-3 признаков на основе Mutual Information с целевой переменной

---

### Упражнение 3: RFE (Recursive Feature Elimination)
**Папка:** `exercise3_rfe/`

**Метод:** Wrapper метод - рекурсивное удаление признаков

**Описание:** Использование Random Forest для последовательного отбора топ-3 признаков

---

### Упражнение 4: Lasso (L1 регуляризация)
**Папка:** `exercise4_lasso/`

**Метод:** Embedded метод - встроенная регуляризация

**Описание:** Выбор признаков с помощью Lasso регрессии (L1 регуляризация)

---

### Упражнение 5: Tree-based Feature Importance
**Папка:** `exercise5_tree_importance/`

**Метод:** Embedded метод - важность признаков в деревьях

**Описание:** Выбор топ-4 признаков на основе feature importance из Random Forest

---

### Упражнение 6: Удаление коррелирующих признаков
**Папка:** `exercise6_remove_correlated/`

**Метод:** Filter метод - анализ корреляций

**Описание:** Удаление признаков с высокой корреляцией (> 0.8)

---

### Упражнение 7: Корреляция с целевой переменной
**Папка:** `exercise7_correlation_target/`

**Метод:** Filter метод - корреляция Пирсона

**Описание:** Выбор топ-3 признаков с наибольшей абсолютной корреляцией с target

---

## Как работать с упражнениями

1. Перейдите в папку упражнения
2. Прочитайте `task.md` с подробным условием
3. Изучите входные данные в `input.csv`
4. Реализуйте решение
5. Сравните результат с `answer.csv`

## Требования

```bash
pip install pandas numpy scikit-learn
```

## Связанный материал

- [Feature Selection Cheatsheet](https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/feature_selection_cheatsheet.html)
