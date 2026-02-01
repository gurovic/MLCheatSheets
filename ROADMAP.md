# 🗺️ Дорожная карта обучения ML Cheatsheets

> **Структурированное руководство по использованию материалов проекта для эффективного изучения машинного обучения**

Это руководство поможет вам эффективно использовать материалы проекта в зависимости от вашего уровня подготовки и целей обучения.

## 🎯 Как пользоваться этой дорожной картой

1. **Выберите свой путь** из 4 предложенных траекторий ниже
2. **Следуйте этапам** последовательно - каждый этап включает список cheatsheets
3. **Практикуйтесь** - после теории решайте [упражнения](exercises.md)
4. **Отмечайте прогресс** - используйте чек-листы для самопроверки
5. **Возвращайтесь** - периодически перечитывайте сложные темы

💡 **Совет:** Не обязательно строго следовать одному пути - комбинируйте разные траектории под свои цели!

---

## 📊 Уровни сложности

- 🟢 **Начальный** - базовые концепции, не требуют глубоких знаний математики
- 🟡 **Средний** - требуют понимания линейной алгебры, статистики и основ ML
- 🔴 **Продвинутый** - сложные алгоритмы, требующие глубоких знаний

## 🎯 Рекомендуемые траектории обучения

### 🌱 Путь 1: Полный новичок в ML (3-6 месяцев)

#### Этап 1: Основы Python и библиотек (2-4 недели) 🟢
1. [NumPy для ML](cheatsheets/numpy_ml_cheatsheet.html)
2. [Pandas для ML](cheatsheets/pandas_ml_cheatsheet.html)
3. [Matplotlib/Seaborn для визуализации](cheatsheets/matplotlib_seaborn_ml_cheatsheet.html)
4. **Упражнения:** Начните с preprocessing

#### Этап 2: Основы теории ML (3-4 недели) 🟢
1. [Смещение-дисперсия](cheatsheets/bias_variance_cheatsheet.html)
2. [Переобучение и недообучение](cheatsheets/overfitting_underfitting_cheatsheet.html)
3. [Кросс-валидация](cheatsheets/cross_validation_cheatsheet.html)
4. [Градиентный спуск](cheatsheets/gradient_descent_cheatsheet.html)

#### Этап 3: Подготовка данных (2-3 недели) 🟢
1. [Предобработка данных](cheatsheets/preprocessing_cheatsheet.html)
2. [Обработка пропусков и кодирование](cheatsheets/data_preprocessing_missing_encoding_cheatsheet.html)
3. [Обработка выбросов](cheatsheets/outliers_handling_cheatsheet.html)
4. [Кодирование категориальных признаков](cheatsheets/categorical_encoding_cheatsheet.html)
5. [Масштабирование и нормализация](cheatsheets/scaling_normalization_cheatsheet.html)
6. **Упражнения:** Все упражнения из раздела подготовки данных

#### Этап 4: Классические алгоритмы - Регрессия (2-3 недели) 🟢-🟡
1. [Линейная регрессия](cheatsheets/linreg_cheatsheet.html)
2. [Метрики регрессии](cheatsheets/regression_metrics_cheatsheet.html)
3. [Полиномиальная регрессия](cheatsheets/polynomial_regression_cheatsheet.html)
4. [Ридж, Лассо, ElasticNet](cheatsheets/ridge_lasso_elasticnet_cheatsheet.html)

#### Этап 5: Классические алгоритмы - Классификация (2-3 недели) 🟢-🟡
1. [Логистическая регрессия](cheatsheets/logreg_cheatsheet.html)
2. [Метрики классификации](cheatsheets/classification_metrics_cheatsheet.html)
3. [Матрица ошибок](cheatsheets/confusion_matrix_cheatsheet.html)
4. [ROC и AUC](cheatsheets/roc_auc_cheatsheet.html)
5. [k-ближайших соседей (k-NN)](cheatsheets/knn_cheatsheet.html)
6. [Наивный Байес](cheatsheets/naive_bayes_cheatsheet.html)

#### Этап 6: Деревья и ансамбли (3-4 недели) 🟡
1. [Деревья решений](cheatsheets/decision_trees_cheatsheet.html)
2. [Случайный лес](cheatsheets/random_forest_cheatsheet.html)
3. [Градиентный бустинг](cheatsheets/boosting_cheatsheet.html)
4. [XGBoost](cheatsheets/xgboost_cheatsheet.html)
5. [LightGBM](cheatsheets/lightgbm_cheatsheet.html)
6. [CatBoost](cheatsheets/catboost_cheatsheet.html)

---

### 🚀 Путь 2: Специализация в глубоком обучении (4-8 месяцев)

**Предварительные требования:** Завершен Путь 1 или есть базовые знания ML

#### Этап 1: Основы нейросетей (3-4 недели) 🟡
1. [Искусственный нейрон (перцептрон)](cheatsheets/perceptron_cheatsheet.html)
2. [Архитектура MLP](cheatsheets/mlp_cheatsheet.html)
3. [Функции активации](cheatsheets/activation_functions_cheatsheet.html)
4. [Функции потерь](cheatsheets/loss_functions_cheatsheet.html)
5. [Обратное распространение ошибки](cheatsheets/backpropagation_cheatsheet.html)
6. [Оптимизаторы](cheatsheets/optimizers_cheatsheet.html)

#### Этап 2: Фреймворки (2-3 недели) 🟡
1. [PyTorch полный гайд](cheatsheets/pytorch_full_guide_cheatsheet.html)
2. [TensorFlow/Keras полный гайд](cheatsheets/tensorflow_keras_full_guide_cheatsheet.html)
3. [PyTorch Lightning](cheatsheets/pytorch_lightning_cheatsheet.html)

#### Этап 3: Сверточные сети (Computer Vision) (3-4 недели) 🟡-🔴
1. [Основы свертки](cheatsheets/cnn_basics_cheatsheet.html)
2. [Пулинг-слои](cheatsheets/pooling_layers_cheatsheet.html)
3. [Архитектуры CNN](cheatsheets/cnn_architectures_cheatsheet.html)
4. [ResNet](cheatsheets/resnet_cheatsheet.html)
5. [Transfer learning для CNN](cheatsheets/transfer_learning_cnn_cheatsheet.html)
6. [Data augmentation](cheatsheets/data_augmentation_cheatsheet.html)
7. Применение: [Детекция объектов (YOLO)](cheatsheets/yolo_cheatsheet.html)

#### Этап 4: Рекуррентные сети и NLP (3-4 недели) 🟡-🔴
1. [Основы RNN](cheatsheets/rnn_basics_cheatsheet.html)
2. [LSTM](cheatsheets/lstm_cheatsheet.html)
3. [GRU](cheatsheets/gru_cheatsheet.html)
4. [Векторные представления слов](cheatsheets/word_embeddings_cheatsheet.html)
5. [Механизм внимания](cheatsheets/attention_mechanism_cheatsheet.html)

#### Этап 5: Трансформеры и LLM (4-5 недель) 🔴
1. [Self-attention](cheatsheets/self_attention_cheatsheet.html)
2. [Multi-head Attention](cheatsheets/multi_head_attention_cheatsheet.html)
3. [Архитектура Transformer](cheatsheets/transformers_cheatsheet.html)
4. [BERT и языковые модели](cheatsheets/bert_language_models_cheatsheet.html)
5. [GPT-архитектуры](cheatsheets/gpt_architectures_cheatsheet.html)
6. [Prompt Engineering](cheatsheets/prompt_engineering_cheatsheet.html)
7. [Vision Transformers (ViT)](cheatsheets/vision_transformers_vit_cheatsheet.html)

#### Этап 6: Генеративные модели (3-4 недели) 🔴
1. [Автоэнкодеры](cheatsheets/autoencoders_cheatsheet.html)
2. [VAE](cheatsheets/vae_cheatsheet.html)
3. [GAN](cheatsheets/gan_cheatsheet.html)
4. [Diffusion модели](cheatsheets/diffusion_models_cheatsheet.html)

---

### 💼 Путь 3: ML для работы (Production ML) (2-4 месяца)

**Предварительные требования:** Базовые знания ML

#### Этап 1: Оптимизация моделей (2-3 недели) 🟡
1. [Grid Search / Random Search](cheatsheets/hyperparameter_tuning_cheatsheet.html)
2. [Bayesian Optimization](cheatsheets/bayesian_optimization_cheatsheet.html)
3. [Optuna / Hyperopt](cheatsheets/optuna_hyperopt_cheatsheet.html)
4. [Важность признаков](cheatsheets/feature_importance_cheatsheet.html)

#### Этап 2: Интерпретируемость (2 недели) 🟡
1. [SHAP значения](cheatsheets/shap_cheatsheet.html)
2. [LIME](cheatsheets/lime_cheatsheet.html)
3. [Частичные зависимости](cheatsheets/partial_dependence_cheatsheet.html)
4. [Explainable AI (XAI)](cheatsheets/explainable_ai_xai_cheatsheet.html)

#### Этап 3: Production и MLOps (3-4 недели) 🟡-🔴
1. [Production ML Best Practices](cheatsheets/production_ml_best_practices_cheatsheet.html)
2. [MLOps Best Practices](cheatsheets/mlops_best_practices_cheatsheet.html)
3. [Сериализация моделей](cheatsheets/model_serialization_cheatsheet.html)
4. [Пайплайны ML](cheatsheets/ml_pipelines_cheatsheet.html)
5. [Мониторинг моделей и A/B тестирование](cheatsheets/model_monitoring_ab_testing_cheatsheet.html)
6. [Feature stores](cheatsheets/feature_stores_cheatsheet.html)
7. [CI/CD для ML](cheatsheets/cicd_ml_cheatsheet.html)

#### Этап 4: Специальные темы (2-3 недели) 🟡
1. [Несбалансированные данные](cheatsheets/imbalanced_data_cheatsheet.html)
2. [Дрейф данных](cheatsheets/concept_drift_cheatsheet.html)
3. [Fairness в ML](cheatsheets/fairness_ml_cheatsheet.html)
4. [Adversarial robustness](cheatsheets/adversarial_robustness_cheatsheet.html)

---

### 📈 Путь 4: Специализация по доменам

#### Computer Vision специалист 🔴
1. Завершите Этапы 1-3 из Пути 2
2. [3D Computer Vision](cheatsheets/3d_computer_vision_cheatsheet.html)
3. [Сегментация изображений](cheatsheets/image_segmentation_cheatsheet.html)
4. [Детекция объектов - продвинутое](cheatsheets/object_detection_cheatsheet.html)
5. [Оценка позы человека](cheatsheets/pose_estimation_cheatsheet.html)
6. [Neural style transfer](cheatsheets/neural_style_transfer_cheatsheet.html)

#### NLP специалист 🔴
1. Завершите Этапы 1-2, 4-5 из Пути 2
2. [Tokenization](cheatsheets/tokenization_cheatsheet.html)
3. [BERT fine-tuning](cheatsheets/bert_finetuning_cheatsheet.html)
4. [Задачи NLP](cheatsheets/nlp_tasks_ner_classification_summarization_cheatsheet.html)
5. [Тематическое моделирование](cheatsheets/topic_modeling_lda_nmf_cheatsheet.html)

#### Time Series специалист 🟡-🔴
1. Классические методы:
   - [ARIMA / SARIMA](cheatsheets/time_series_cheatsheet.html)
   - [Экспоненциальное сглаживание](cheatsheets/exponential_smoothing_cheatsheet.html)
   - [Prophet](cheatsheets/prophet_time_series_cheatsheet.html)
2. DL методы:
   - [RNN/LSTM для временных рядов](cheatsheets/rnn_lstm_time_series_cheatsheet.html)
   - [CNN для временных рядов (TCN)](cheatsheets/cnn_time_series_tcn_cheatsheet.html)
   - [Transformers для временных рядов](cheatsheets/transformers_time_series_cheatsheet.html)

#### Reinforcement Learning специалист 🔴
1. Классическое RL:
   - [RL Basics](cheatsheets/reinforcement_learning_basics_cheatsheet.html)
   - [Q-learning и SARSA](cheatsheets/q_learning_sarsa_cheatsheet.html)
   - [Multi-armed bandits](cheatsheets/multi_armed_bandits_cheatsheet.html)
2. Deep RL:
   - [DQN](cheatsheets/dqn_cheatsheet.html)
   - [Policy Gradient методы](cheatsheets/policy_gradient_cheatsheet.html)
   - [PPO](cheatsheets/ppo_reinforcement_learning_cheatsheet.html)

---

## 📚 Дополнительные ресурсы

### По категориям сложности

#### 🟢 Начальный уровень (начните здесь!)
- Все cheatsheets из раздела "Подготовка данных"
- Линейная и логистическая регрессия
- k-NN и Наивный Байес
- Базовые метрики
- Все упражнения

#### 🟡 Средний уровень
- Деревья решений и ансамбли
- SVM
- Кластеризация
- Feature Engineering
- Базовые нейросети (MLP, CNN, RNN)
- Временные ряды (классические методы)

#### 🔴 Продвинутый уровень
- Трансформеры и LLM
- Генеративные модели (GAN, VAE, Diffusion)
- Графовые нейронные сети
- Meta-learning
- Квантовое ML
- Reinforcement Learning

---

## 💡 Советы по обучению

1. **Практика важнее теории** - после каждого cheatsheet пишите код
2. **Делайте упражнения** - в папке exercises есть практические задания
3. **Не спешите** - лучше глубоко изучить один алгоритм, чем поверхностно десять
4. **Ведите проекты** - применяйте знания на реальных данных
5. **Участвуйте в Kaggle** - соревнования помогают расти
6. **Изучайте код** - смотрите реализации в scikit-learn, PyTorch
7. **Возвращайтесь к основам** - периодически перечитывайте базовые темы

---

## 🎯 Чек-листы для самопроверки

### ✅ Я готов работать Junior ML Engineer, если:
- [ ] Понимаю основы ML (bias-variance, overfitting, validation)
- [ ] Умею работать с pandas, numpy, scikit-learn
- [ ] Знаю основные алгоритмы (регрессия, классификация, деревья)
- [ ] Могу подготовить данные (preprocessing, feature engineering)
- [ ] Понимаю метрики и могу выбрать правильную
- [ ] Умею настраивать гиперпараметры
- [ ] Знаю основы git и могу работать в команде

### ✅ Я готов работать Middle ML Engineer, если:
- [ ] Все из Junior +
- [ ] Знаю градиентный бустинг (XGBoost, LightGBM, CatBoost)
- [ ] Понимаю основы нейросетей и могу обучить простую модель
- [ ] Умею работать с несбалансированными данными
- [ ] Знаю методы интерпретации моделей (SHAP, LIME)
- [ ] Могу запустить модель в production
- [ ] Понимаю MLOps и знаю инструменты (Docker, CI/CD)

### ✅ Я готов работать Senior ML Engineer / ML Researcher, если:
- [ ] Все из Middle +
- [ ] Глубоко понимаю математику ML
- [ ] Могу реализовать алгоритмы с нуля
- [ ] Знаю современные архитектуры (Transformers, Diffusion models)
- [ ] Читаю и могу воспроизвести статьи
- [ ] Умею оптимизировать модели (pruning, quantization, distillation)
- [ ] Могу разработать ML систему от начала до конца

---

## 🗓️ Примерный график обучения

### Интенсивный (20+ часов в неделю)
- **Путь 1**: 3-4 месяца
- **Путь 2**: 4-6 месяцев
- **Путь 3**: 2-3 месяца

### Умеренный (10-15 часов в неделю)
- **Путь 1**: 5-6 месяцев
- **Путь 2**: 6-8 месяцев
- **Путь 3**: 3-4 месяца

### Спокойный (5-10 часов в неделю)
- **Путь 1**: 8-12 месяцев
- **Путь 2**: 10-14 месяцев
- **Путь 3**: 5-6 месяцев

---

## 🔗 Полезные ссылки

- [🏠 Главная страница](README.md) - полный список всех cheatsheets
- [💪 Упражнения](exercises.md) - практические задания для отработки навыков
- [🎥 Анимации](animations/README.md) - визуализации алгоритмов
- [🤝 Contributing](CONTRIBUTING.md) - как внести вклад в проект
- [📋 Setup Guide](SETUP.md) - установка и настройка окружения
- [🚀 Перспективы](PROSPECTS.md) - будущие возможности проекта

---

## 📝 Отслеживание прогресса

Рекомендуем вести свой трекер обучения. Вот простой шаблон:

```markdown
# Мой прогресс по ML Cheatsheets

## Выбранный путь: [Название пути]

### Текущий этап: [Название этапа]

#### Изучено:
- [x] Cheatsheet 1
- [x] Cheatsheet 2
- [ ] Cheatsheet 3

#### Упражнения:
- [x] Упражнение 1
- [ ] Упражнение 2

#### Заметки:
- Что понял:
- Что нужно повторить:
- Вопросы:
```

---

<div align="center">

**Удачи в обучении! 🚀**

© 2024 MLCheatSheets. Автор: Владимир Гуровиц (школа "Летово")
