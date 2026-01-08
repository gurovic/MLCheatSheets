# Transfer Learning Illustrations Documentation

## Обзор

Этот документ описывает matplotlib-иллюстрации, добавленные в раздел "Трансферное обучение" проекта MLCheatSheets.

## Добавленные иллюстрации

### 1. transfer_learning_cheatsheet.html (5 иллюстраций)

1. **Концепция Transfer Learning: Source → Target**
   - Визуализация процесса переноса знаний между доменами
   - Показывает source domain и target domain с общими признаками

2. **Типы Transfer Learning**
   - Feature-based transfer
   - Instance-based transfer (TrAdaBoost)
   - Parameter-based transfer
   - Relational transfer

3. **Domain Shift: различие между source и target**
   - Визуализация распределений source и target данных
   - Показывает расхождение между доменами

4. **Процесс TrAdaBoost: эволюция весов**
   - Динамика изменения весов на каждой итерации
   - Иллюстрирует увеличение весов для target и уменьшение для source

5. **Self-training: постепенное добавление псевдометок**
   - Процесс итеративного добавления меток
   - Показывает метки labeled, pseudo-labeled и unlabeled данных

### 2. transfer_learning_cnn_cheatsheet.html (1 иллюстрация)

1. **Архитектура Transfer Learning для CNN**
   - Визуализация pre-trained модели с frozen и fine-tuned слоями
   - Показывает структуру: входной слой → Conv/Pool блоки → FC слои

### 3. transfer_learning_deep_cheatsheet.html (5 иллюстраций)

1. **Fine-tuning стратегии**
   - Linear probing: заморозка всех слоев, кроме последнего
   - Full fine-tuning: дообучение всех слоев
   - Layer-wise fine-tuning: постепенное размораживание

2. **Pre-trained Models: learning curves**
   - Сравнение кривых обучения для:
     - Training from scratch
     - Pre-trained model
   - Показывает преимущество transfer learning

3. **Feature extraction vs Fine-tuning**
   - Сравнительная иллюстрация двух подходов
   - Показывает, какие слои обучаются в каждом случае

4. **Процесс TrAdaBoost: эволюция весов**
   - Динамика изменения весов на каждой итерации (копия из основного раздела)

5. **Self-training: постепенное добавление псевдометок**
   - Процесс итеративного добавления меток (копия из основного раздела)

### 4. domain_adaptation_cheatsheet.html (3 иллюстрации)

1. **Domain Shift Problem**
   - Визуализация проблемы domain shift
   - Показывает распределения source и target данных

2. **Maximum Mean Discrepancy (MMD)**
   - Визуализация MMD как метрики расхождения
   - Показывает вычисление MMD между доменами

3. **Методы Domain Adaptation**
   - Feature-level: Alignment, DANN
   - Instance-level: Re-weighting
   - Model-level: Self-training

## Стиль иллюстраций

Все иллюстрации созданы с использованием matplotlib и имеют единообразный стиль:
- Цветовая схема: seaborn-v0_8 (или seaborn для старых версий matplotlib)
- Русский язык для всех подписей
- Высокое разрешение (300 DPI)
- Формат: PNG с base64 кодированием для встраивания в HTML

## Код генерации

Иллюстрации генерируются с помощью скрипта `generate_transfer_learning_illustrations.py`, который:
1. Создает графики с использованием matplotlib
2. Конвертирует их в base64-формат
3. Возвращает готовые для встраивания строки

Встраивание в HTML выполняется скриптом `add_transfer_learning_illustrations_to_html.py`, который:
1. Генерирует все иллюстрации
2. Находит подходящие места в HTML-файлах
3. Вставляет иллюстрации в соответствующие разделы

## Воспроизведение

Для пересоздания иллюстраций выполните:
```bash
python3 add_transfer_learning_illustrations_to_html.py
```

Требования:
- Python 3.6+
- matplotlib
- numpy

## Примечания

- Все иллюстрации встроены непосредственно в HTML как base64-данные
- Это упрощает распространение и использование cheatsheets
- Иллюстрации оптимизированы для отображения на экране и печати
