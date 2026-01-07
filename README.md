# 🎓 ML Cheatsheets

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![GitHub Stars](https://img.shields.io/github/stars/gurovic/MLCheatSheets?style=social)](https://github.com/gurovic/MLCheatSheets/stargazers)
[![GitHub Forks](https://img.shields.io/github/forks/gurovic/MLCheatSheets?style=social)](https://github.com/gurovic/MLCheatSheets/network/members)
[![Contributions Welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg)](CONTRIBUTING.md)
[![Russian](https://img.shields.io/badge/lang-Russian-blue.svg)](README.md)

> **Комплексная коллекция из 500+ cheatsheets по машинному обучению и глубокому обучению на русском языке**

Автор: **Владимир Гуровиц** (школа "Летово")  
При поддержке: DeepSeek, Github Copilot, Perplexity Comet

---

## 📚 О проекте

ML Cheatsheets - это бесплатная образовательная коллекция справочных материалов, охватывающая все основные темы машинного обучения от базовых алгоритмов до современных нейросетевых архитектур. Каждый cheatsheet содержит:

- ✅ Теоретические основы
- ✅ Математические формулы
- ✅ Примеры кода на Python
- ✅ Визуализации и диаграммы
- ✅ Советы по применению

## 🚀 Быстрый старт

### Для начинающих

1. **Прочитайте** [дорожную карту обучения](ROADMAP.md) - выберите подходящий путь
2. **Начните с основ** - раздел "Подготовка данных" и базовые алгоритмы
3. **Практикуйтесь** - решайте [упражнения](exercises.md)
4. **Углубляйтесь** - переходите к продвинутым темам

### Установка зависимостей

```bash
# Клонируйте репозиторий
git clone https://github.com/gurovic/MLCheatSheets.git
cd MLCheatSheets

# Установите зависимости для упражнений
pip install -r requirements.txt

# (Опционально) Для работы с анимациями
pip install -r animations/requirements.txt
```

## 📖 Содержание

* 📘 [I. Классическое машинное обучение](#i-классическое-машинное-обучение) (32 раздела, 250+ cheatsheets)
* 🧠 [II. Глубокое обучение](#ii-глубокое-обучение) (27 разделов, 200+ cheatsheets)
* 🔬 [III. Разные темы машинного обучения](#iii-разные-темы-машинного-обучения) (6 разделов, 50+ cheatsheets)
* 💪 EXPERIMENTAL [Практические упражнения](exercises.md) (8 упражнений с решениями)
* 🎥 EXPERIMENTAL [Анимации и визуализации](animations/README.md)

## 🗺️ Рекомендуемые пути обучения

### 🌱 [Путь новичка](ROADMAP.md#-путь-1-полный-новичок-в-ml-3-6-месяцев) (3-6 месяцев)
Изучение основ ML с нуля: Python → Теория → Подготовка данных → Классические алгоритмы → Ансамбли

### 🚀 [Путь специалиста по Deep Learning](ROADMAP.md#-путь-2-специализация-в-глубоком-обучении-4-8-месяцев) (4-8 месяцев)
От основ нейросетей до трансформеров и генеративных моделей

### 💼 [Путь ML Engineer](ROADMAP.md#-путь-3-ml-для-работы-production-ml-2-4-месяца) (2-4 месяца)
Production ML, MLOps, интерпретируемость и deployment

### 🎯 [Специализация по доменам](ROADMAP.md#-путь-4-специализация-по-доменам)
Computer Vision, NLP, Time Series, Reinforcement Learning

---

## 📋 Навигация по разделам

* [Классическое машинное обучение](https://github.com/gurovic/MLCheatSheets/blob/main/README.md#%D0%BA%D0%BB%D0%B0%D1%81%D1%81%D0%B8%D1%87%D0%B5%D1%81%D0%BA%D0%BE%D0%B5-%D0%BC%D0%B0%D1%88%D0%B8%D0%BD%D0%BD%D0%BE%D0%B5-%D0%BE%D0%B1%D1%83%D1%87%D0%B5%D0%BD%D0%B8%D0%B5)
* [Глубокое обучение](https://github.com/gurovic/MLCheatSheets/blob/main/README.md#%D0%B3%D0%BB%D1%83%D0%B1%D0%BE%D0%BA%D0%BE%D0%B5-%D0%BE%D0%B1%D1%83%D1%87%D0%B5%D0%BD%D0%B8%D0%B5)
* [Разные темы машинного обучения](https://github.com/gurovic/MLCheatSheets/blob/main/README.md#%D0%B4%D1%80%D1%83%D0%B3%D0%B8%D0%B5-%D1%82%D0%B5%D0%BC%D1%8B-%D0%BC%D0%B0%D1%88%D0%B8%D0%BD%D0%BD%D0%BE%D0%B3%D0%BE-%D0%BE%D0%B1%D1%83%D1%87%D0%B5%D0%BD%D0%B8%D1%8F)

## I Классическое машинное обучение

### **1. Подготовка данных**
1. [Предобработка данных (Preprocessing)](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/preprocessing_cheatsheet.html)
1. [Data Preprocessing (missing values, encoding)](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/data_preprocessing_missing_encoding_cheatsheet.html)
1. [Обработка выбросов](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/outliers_handling_cheatsheet.html)
1. [Кодирование категориальных признаков](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/categorical_encoding_cheatsheet.html)
1. [Масштабирование и нормализация](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/scaling_normalization_cheatsheet.html)
1. [Выбор признаков (Feature Selection)](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/feature_selection_cheatsheet.html)
1. [Снижение размерности (Dimensionality Reduction)](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/dimensionality_reduction_cheatsheet.html)
1. [Балансировка классов (Imbalanced Data)](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/imbalanced_data_cheatsheet.html)

### **2. 🎨 Классификация**
1. [Логистическая регрессия](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/logreg_cheatsheet.html)
1. [Метод опорных векторов (SVM)](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/svm_cheatsheet.html)
1. [Наивный Байес (Naive Bayes)](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/naive_bayes_cheatsheet.html)
1. [k-ближайших соседей (k-NN)](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/knn_cheatsheet.html)
1. [Деревья решений для классификации](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/decision_trees_cheatsheet.html)
1. [Случайный лес для классификации](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/random_forest_cheatsheet.html)
1. [Ансамбли методов (Voting, Stacking)](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/ensemble_methods_cheatsheet.html)

### **3. 🎨 Регрессия**
1. [Линейная регрессия](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/linreg_cheatsheet.html)
1. [Полиномиальная регрессия](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/polynomial_regression_cheatsheet.html)
1. [Ридж, Лассо, ElasticNet](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/ridge_lasso_elasticnet_cheatsheet.html)
1. [Деревья решений для регрессии](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/decision_trees_regression_cheatsheet.html)
1. [Случайный лес для регрессии](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/random_forest_regression_cheatsheet.html)
1. [Методы регуляризации](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/regularization_methods_cheatsheet.html)
1. [Непараметрическая регрессия](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/nonparametric_regression_cheatsheet.html)

### **4. 🎨 Кластеризация**
1. [K-means и K-means++](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/kmeans_cheatsheet.html)
1. [Иерархическая кластеризация](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/hierarchical_clustering_cheatsheet.html)
1. [DBSCAN](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/dbscan_cheatsheet.html)
1. [OPTICS](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/optics_cheatsheet.html)
1. [Gaussian Mixture Models (GMM)](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/gmm_cheatsheet.html)
1. [Метрики оценки кластеризации](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/clustering_metrics_cheatsheet.html)
1. [Выбор числа кластеров](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/choosing_clusters_number_cheatsheet.html)
1. [Spectral clustering](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/spectral_clustering_cheatsheet.html)
1. [Mean shift](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/mean_shift_cheatsheet.html)
1. [Affinity propagation](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/affinity_propagation_cheatsheet.html)
1. [BIRCH](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/birch_cheatsheet.html)
1. [CURE](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/cure_cheatsheet.html)
1. [CLARANS](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/clarans_cheatsheet.html)
1. [HDBSCAN](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/hdbscan_cheatsheet.html)
1. [Subspace clustering](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/subspace_clustering_cheatsheet.html)

### **5. Понижение размерности**
1. [PCA (Principal Component Analysis)](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/pca_cheatsheet.html)
1. [t-SNE](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/tsne_cheatsheet.html)
1. [UMAP](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/umap_cheatsheet.html)
1. [LDA (Linear Discriminant Analysis)](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/lda_cheatsheet.html)
1. [SVD (Singular Value Decomposition)](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/svd_cheatsheet.html)
1. [Автоэнкодеры (линейные)](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/linear_autoencoders_cheatsheet.html)
1. [Manifold learning](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/manifold_learning_cheatsheet.html)
1. [Isomap](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/isomap_cheatsheet.html)
1. [LLE (Locally Linear Embedding)](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/lle_cheatsheet.html)
1. [MDS (Multidimensional Scaling)](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/mds_cheatsheet.html)
1. [Spectral embedding](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/spectral_embedding_cheatsheet.html)
1. [Kernel PCA](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/kernel_pca_cheatsheet.html)

### **6. Оценка моделей и валидация**
1. [Метрики классификации](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/classification_metrics_cheatsheet.html)
1. [Матрица ошибок (Confusion Matrix)](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/confusion_matrix_cheatsheet.html)
1. [ROC и AUC](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/roc_auc_cheatsheet.html)
1. [Метрики регрессии](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/regression_metrics_cheatsheet.html)
1. [Кросс-валидация](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/cross_validation_cheatsheet.html)
1. [Кривые обучения](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/learning_curves_cheatsheet.html)
1. [Калибровка моделей](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/model_calibration_cheatsheet.html)
1. [Статистические тесты для сравнения моделей](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/statistical_tests_cheatsheet.html)
1. [Статистические тесты (t-test, McNemar)](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/statistical_tests_t_test_mcnemar_cheatsheet.html)
1. [Кривые обучения и валидации](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/learning_validation_curves_cheatsheet.html)
1. [Bias-variance decomposition](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/bias_variance_decomposition_cheatsheet.html)
1. [Неметрические критерии (интерпретируемость, скорость)](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/non_metric_criteria_cheatsheet.html)

### **7. Поиск гиперпараметров**
1. [Grid Search / Random Search](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/hyperparameter_tuning_cheatsheet.html)
1. [Bayesian Optimization](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/bayesian_optimization_cheatsheet.html)
1. [Optuna / Hyperopt](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/optuna_hyperopt_cheatsheet.html)
1. [Стратегии поиска гиперпараметров](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/hyperparameter_strategies_cheatsheet.html)
1. [Байесовская оптимизация гиперпараметров](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/bayesian_hyperparameter_optimization_cheatsheet.html)
1. [Байесовские нейронные сети](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/bayesian_neural_networks_cheatsheet.html)

### **8. Работа с временными рядами (классические методы)**
1. [ARIMA / SARIMA](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/time_series_cheatsheet.html)
1. [Экспоненциальное сглаживание](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/exponential_smoothing_cheatsheet.html)
1. [Методы на основе скользящих средних](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/moving_average_methods_cheatsheet.html)
1. [Feature Engineering для временных рядов](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/time_series_feature_engineering_cheatsheet.html)
1. [Валидация временных рядов](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/time_series_validation_cheatsheet.html)
1. [Prophet и современные методы временных рядов](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/prophet_time_series_cheatsheet.html)
1. [State space models](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/state_space_models_cheatsheet.html)
1. [Kalman filters](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/kalman_filters_cheatsheet.html)
1. [Сезонная декомпозиция](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/seasonal_decomposition_cheatsheet.html)
1. [Теория временных рядов Box-Jenkins](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/box_jenkins_cheatsheet.html)

### **9. Обнаружение аномалий**
1. [Isolation Forest](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/isolation_forest_cheatsheet.html)
1. [One-Class SVM](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/one_class_svm_cheatsheet.html)
1. [Local Outlier Factor](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/local_outlier_factor_cheatsheet.html)
1. [Elliptic Envelope](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/elliptic_envelope_cheatsheet.html)
1. [Статистические методы](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/statistical_anomaly_detection_cheatsheet.html)

### **10. Рекомендательные системы (классические)**
1. [Коллаборативная фильтрация](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/collaborative_filtering_cheatsheet.html)
1. [Контентная фильтрация](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/content_based_filtering_cheatsheet.html)
1. [Матричная факторизация (SVD)](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/matrix_factorization_svd_cheatsheet.html)
1. [ALS (Alternating Least Squares)](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/als_recommender_cheatsheet.html)
1. [Коллаборативная фильтрация (user-based, item-based)](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/collaborative_filtering_user_item_based_cheatsheet.html)
1. [Матричная факторизация (SVD, NMF)](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/matrix_factorization_svd_nmf_cheatsheet.html)
1. [Гибридные системы](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/hybrid_recommender_systems_cheatsheet.html)
1. [Context-aware рекомендации](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/context_aware_recommendations_cheatsheet.html)
1. [Sequence-aware рекомендации](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/sequence_aware_recommendations_cheatsheet.html)

### **11. Градиентный бустинг**
1. [Градиентный бустинг для классификации](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/boosting_cheatsheet.html)
1. [Градиентный бустинг для регрессии](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/gradient_boosting_regression_cheatsheet.html)
1. [XGBoost](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/xgboost_cheatsheet.html)
1. [LightGBM](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/lightgbm_cheatsheet.html)
1. [CatBoost](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/catboost_cheatsheet.html)
1. [Гиперпараметры градиентного бустинга](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/gradient_boosting_hyperparameters_cheatsheet.html)
1. [Особенности и тонкости настройки ансамблей](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/ensemble_tuning_cheatsheet.html)
1. [Бэггинг (Bagging)](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/bagging_cheatsheet.html)
1. [AdaBoost](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/adaboost_cheatsheet.html)
1. [Стекинг и Voting (Stacking, Voting)](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/voting_stacking_cheatsheet.html)
1. [Advanced Model Ensembling](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/advanced_ensembling_cheatsheet.html)
1. [Блендинг (Blending)](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/blending_cheatsheet.html)
1. [Ансамбли деревьев](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/tree_ensembles_cheatsheet.html)
1. [Ансамбли линейных моделей](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/linear_ensembles_cheatsheet.html)
1. [Диверсификация ансамблей](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/ensemble_diversification_cheatsheet.html)

### **12. Статистические методы**
1. [Статистические методы для ML (Bootstrap, тесты)](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/statistical_methods_cheatsheet.html)
1. [Линейные модели (GLM)](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/glm_cheatsheet.html)
1. [Множественная регрессия](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/multiple_regression_cheatsheet.html)
1. [Дисперсионный анализ (ANOVA)](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/anova_cheatsheet.html)
1. [Bootstrap](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/bootstrap_methods_cheatsheet.html)
1. [Статистическая проверка гипотез](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/hypothesis_testing_cheatsheet.html)
1. [Байесовский вывод](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/bayesian_inference_cheatsheet.html)
1. [Скрытые марковские модели (HMM)](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/hmm_cheatsheet.html)
1. [Гауссовские процессы](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/gaussian_processes_cheatsheet.html)
1. [Вариационный вывод](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/variational_inference_cheatsheet.html)
1. [Марковские цепи Монте-Карло (MCMC) и Байесовский вывод](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/mcmc_bayesian_inference_cheatsheet.html)
1. [Expectation-Maximization (EM)](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/expectation_maximization_cheatsheet.html)
1. [Графические модели](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/graphical_models_cheatsheet.html)
1. [Наивный Байес (как байесовский метод)](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/naive_bayes_bayesian_method_cheatsheet.html)

### **13. Проектирование признаков (Feature Engineering)**
1. [Feature Engineering (основы)](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/feature_engineering_cheatsheet.html)
1. [Feature Engineering (подробно)](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/feature_engineering_detailed_cheatsheet.html)
1. [Создание полиномиальных признаков и взаимодействия](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/feature_interaction_cheatsheet.html)
1. [Биннинг (дискретизация)](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/binning_discretization_cheatsheet.html)
1. [Target Encoding](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/target_encoding_cheatsheet.html)
1. [Дата-специфичные признаки (временные, текстовые)](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/date_time_features_cheatsheet.html)

### **14. Работа с несбалансированными данными**
1. [Методы семплирования (SMOTE и др.)](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/smote_imbalanced_cheatsheet.html)
1. [Cost-Sensitive Learning и веса классов](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/cost_sensitive_learning_cheatsheet.html)
1. [Метрики для несбалансированных данных](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/imbalanced_metrics_cheatsheet.html)
1. [Алгоритмические подходы](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/imbalanced_algorithms_cheatsheet.html)

### **15. Интерпретация моделей**
1. [Важность признаков (Feature Importance)](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/feature_importance_cheatsheet.html)
1. [SHAP значения](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/shap_cheatsheet.html)
1. [LIME](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/lime_cheatsheet.html)
1. [Частичные зависимости (Partial Dependence Plots)](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/partial_dependence_cheatsheet.html)
1. [Explainable AI (XAI) - Полный обзор](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/explainable_ai_xai_cheatsheet.html)
1. [Surrogate models](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/surrogate_models_cheatsheet.html)

### **16. Производственные аспекты**
1. [Production ML Best Practices](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/production_ml_best_practices_cheatsheet.html)
1. [Сериализация моделей](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/model_serialization_cheatsheet.html)
1. [Дрейф данных и концептуальный дрейф](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/concept_drift_cheatsheet.html)
1. [Пайплайны ML](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/ml_pipelines_cheatsheet.html)
1. [MLOps Best Practices](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/mlops_best_practices_cheatsheet.html)
1. [Мониторинг моделей и A/B тестирование](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/model_monitoring_ab_testing_cheatsheet.html)
1. [Модельные реестры](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/model_registries_cheatsheet.html)
1. [Feature stores](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/feature_stores_cheatsheet.html)
1. [CI/CD для ML](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/cicd_ml_cheatsheet.html)

### **17. Специальные алгоритмы**
1. [Метод главных компонент (PCA)](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/pca_method_cheatsheet.html)
1. [Линейный дискриминантный анализ (LDA)](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/lda_method_cheatsheet.html)
1. [Канонический корреляционный анализ](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/canonical_correlation_analysis_method_cheatsheet.html)
1. [Скрытые марковские модели (HMM)](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/hmm_cheatsheet.html)
1. [EM-алгоритм](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/em_algorithm_cheatsheet.html)

### **18. Визуализация в ML**
1. [Визуализация данных](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/data_visualization_ml_cheatsheet.html)
1. [Визуализация кластеров](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/cluster_visualization_cheatsheet.html)
1. [Визуализация деревьев](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/tree_visualization_cheatsheet.html)
1. [Визуализация метрик](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/metrics_visualization_cheatsheet.html)

### **19. Основы теории**
1. [Смещение-дисперсия](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/bias_variance_cheatsheet.html)
1. [Переобучение и недообучение (Overfitting/Underfitting)](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/overfitting_underfitting_cheatsheet.html)
1. [Градиентный спуск (Gradient Descent)](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/gradient_descent_cheatsheet.html)
1. [Теория вероятностей для ML](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/probability_theory_cheatsheet.html)
1. [Регуляризация](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/regularization_cheatsheet.html)
1. [Байесовский подход](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/bayesian_approach_cheatsheet.html)
1. [Теория информации в ML](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/information_theory_ml_cheatsheet.html)
1. [Линейная алгебра и оптимизация](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/linear_algebra_optimization_cheatsheet.html)
1. [Теория графов для ML](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/graph_theory_ml_cheatsheet.html)
1. [Теория решений](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/decision_theory_cheatsheet.html)

### **20. Инструменты и библиотеки**
1. [Scikit-learn полный обзор](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/sklearn_overview_cheatsheet.html)
1. [Pandas для ML](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/pandas_ml_cheatsheet.html)
1. [NumPy для ML](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/numpy_ml_cheatsheet.html)
1. [Matplotlib/Seaborn для визуализации](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/matplotlib_seaborn_ml_cheatsheet.html)

### **21. Дополнительные темы**
1. [Active Learning](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/active_learning_cheatsheet.html)
1. [Transfer Learning (классические подходы)](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/transfer_learning_cheatsheet.html)
1. [Multi-task Learning](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/multi_task_learning_deep_dive_cheatsheet.html)
1. [Multi-output регрессионные модели](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/multi_output_regression_cheatsheet.html)
1. [Квантильная регрессия](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/quantile_regression_cheatsheet.html)

### **22. Этика, безопасность и приватность**
1. [Fairness в ML (Справедливость)](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/fairness_ml_cheatsheet.html)
1. [Adversarial Attacks (Состязательные атаки)](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/adversarial_attacks_cheatsheet.html)
1. [Adversarial robustness (классические атаки)](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/adversarial_robustness_cheatsheet.html)
1. [Differential privacy](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/differential_privacy_cheatsheet.html)
1. [Explainability и accountability](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/explainability_accountability_cheatsheet.html)

### **23. Безопасность и обнаружение аномалий в приложениях**
1. [Обнаружение вторжений](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/intrusion_detection_cheatsheet.html)
1. [Malware detection](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/malware_detection_cheatsheet.html)
1. [Fraud detection](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/fraud_detection_cheatsheet.html)

### **24. Финансовое и бизнес ML**
1. [Алгоритмический трейдинг](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/algorithmic_trading_cheatsheet.html)
1. [Кредитный скоринг](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/credit_scoring_cheatsheet.html)
1. [Risk management](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/risk_management_cheatsheet.html)

### **25. Графовые алгоритмы (классические)**
1. [Алгоритмы на графах (кластеризация, поиск путей)](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/graph_algorithms_cheatsheet.html)
1. [Социальный сетевой анализ](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/social_network_analysis_cheatsheet.html)
1. [Графовые эмбеддинги (Node2Vec, DeepWalk)](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/graph_embeddings_cheatsheet.html)
1. [Community detection](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/community_detection_cheatsheet.html)
1. [Графовые ядра (Graph kernels)](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/graph_kernels_cheatsheet.html)
1. [Influence maximization](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/influence_maximization_cheatsheet.html)

### **26. Традиционная обработка текста (NLP)**
1. [Тематическое моделирование (LDA, NMF)](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/topic_modeling_lda_nmf_cheatsheet.html)
1. [TF-IDF и векторное пространство](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/tfidf_cheatsheet.html)
1. [Латентно-семантический анализ (LSA)](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/latent_semantic_analysis_cheatsheet.html)
1. [Стемминг, лемматизация](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/stemming_lemmatization_cheatsheet.html)
1. [Правила и паттерны (regex, finite-state)](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/regex_finite_state_cheatsheet.html)
1. [CRF (Conditional Random Fields) для NER](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/crf_conditional_random_fields_cheatsheet.html)
1. [Синтаксический разбор (constituency, dependency)](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/syntactic_parsing_cheatsheet.html)

### **27. Полуавтоматическое обучение (Semi-supervised)**
1. [Semi-supervised Learning](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/semi_supervised_learning_cheatsheet.html)
1. [Self-training](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/self_training_cheatsheet.html)
1. [Co-training](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/cotraining_cheatsheet.html)
1. [Label propagation](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/label_propagation_cheatsheet.html)
1. [Graph-based SSL](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/graph_based_ssl_cheatsheet.html)
1. [Pseudo-labeling](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/pseudo_labeling_cheatsheet.html)
1. [Generative models (классические)](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/generative_models_classical_cheatsheet.html)

### **28. Онлайн обучение и инкрементное обучение**
1. [Online Learning (SGD, Perceptron, Concept Drift)](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/online_learning_cheatsheet.html)
1. [Incremental Learning (Инкрементное обучение)](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/incremental_learning_cheatsheet.html)
1. [Concept drift detection](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/concept_drift_cheatsheet.html)
1. [Adaptive windowing](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/adaptive_windowing_cheatsheet.html)
1. [Ensemble methods для streaming](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/ensemble_methods_streaming_cheatsheet.html)
1. [Reservoir sampling](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/reservoir_sampling_cheatsheet.html)
1. [Hoeffding trees](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/hoeffding_trees_cheatsheet.html)

### **29. Мультимодальное обучение (классическое)**
1. [Early fusion и late fusion](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/early_late_fusion_cheatsheet.html)
1. [Canonical Correlation Analysis (CCA)](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/canonical_correlation_analysis_cheatsheet.html)
1. [Kernel CCA](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/kernel_cca_cheatsheet.html)
1. [Multi-view learning](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/multi_view_learning_cheatsheet.html)
1. [Co-training](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/cotraining_semi_supervised_cheatsheet.html)

### **30. Обучение с ограничениями**
1. [Constrained optimization в ML](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/constrained_optimization_ml_cheatsheet.html)
1. [Fairness constraints](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/fairness_constraints_cheatsheet.html)
1. [Monotonic constraints](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/monotonic_constraints_cheatsheet.html)
1. [Shape constraints](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/shape_constraints_cheatsheet.html)

### **31. Комбинаторная оптимизация и ML**
1. [Генетические алгоритмы](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/genetic_algorithms_cheatsheet.html)
1. [Эволюционные стратегии](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/evolutionary_strategies_cheatsheet.html)
1. [Swarm intelligence (PSO, ant colony)](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/swarm_intelligence_cheatsheet.html)
1. [Simulated annealing](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/simulated_annealing_cheatsheet.html)
1. [Tabu search](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/tabu_search_cheatsheet.html)

### **32. Обучение с подкреплением (классическое RL)**
1. [Reinforcement Learning Basics (RL основы)](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/reinforcement_learning_basics_cheatsheet.html)
1. [Классическое RL (табличное, без нейросетей)](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/classical_rl_tabular_cheatsheet.html)
1. [Многорукие бандиты (Multi-armed bandits)](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/multi_armed_bandits_cheatsheet.html)
1. [Марковские процессы принятия решений](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/markov_decision_processes_cheatsheet.html)
1. [Динамическое программирование](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/dynamic_programming_rl_cheatsheet.html)
1. [Монте-Карло методы](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/monte_carlo_methods_cheatsheet.html)
1. [Временная разница (TD-learning)](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/td_learning_cheatsheet.html)
1. [Q-learning и SARSA](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/q_learning_sarsa_cheatsheet.html)
1. [Функции приближения (линейные, ядерные)](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/function_approximation_rl_cheatsheet.html)
1. [Иерархическое RL](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/hierarchical_rl_detailed_cheatsheet.html)
1. [Многозадачное RL](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/multitask_rl_cheatsheet.html)
1. [Инверсное RL](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/inverse_rl_cheatsheet.html)

## II Глубокое обучение

### **1. Основы нейронных сетей**
1. [Искусственный нейрон (перцептрон)](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/perceptron_cheatsheet.html)
1. [Архитектура MLP (многослойный перцептрон)](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/mlp_cheatsheet.html)
1. [Функции активации (ReLU, Sigmoid, Tanh, Softmax)](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/activation_functions_cheatsheet.html)
1. [Функции потерь для классификации и регрессии](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/loss_functions_cheatsheet.html)
1. [Обратное распространение ошибки (Backpropagation)](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/backpropagation_cheatsheet.html)
1. [Оптимизаторы (SGD, Momentum, Adam, RMSProp)](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/optimizers_cheatsheet.html)
1. [Продвинутые оптимизаторы (AdaGrad, RMSProp, AdamW, Nadam)](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/optimizers_advanced_cheatsheet.html)
1. [Learning Rate Scheduling](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/learning_rate_scheduling_cheatsheet.html)
1. [Инициализация весов (Weight Initialization)](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/weight_initialization_cheatsheet.html)
1. [Ранняя остановка и кривые обучения (Early Stopping)](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/early_stopping_cheatsheet.html)
1. [Регуляризация нейронных сетей (Dropout и др.)](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/dropout_regularization_cheatsheet.html)

### **2. Сверточные нейронные сети (CNN)**
1. [Основы свертки](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/cnn_basics_cheatsheet.html)
1. [Пулинг-слои (Max, Average, Global Pooling)](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/pooling_layers_cheatsheet.html)
1. [Архитектуры CNN (LeNet, AlexNet, VGG)](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/cnn_architectures_cheatsheet.html)
1. Современные архитектуры ([ResNet](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/resnet_cheatsheet.html), [Inception, EfficientNet](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/inception_efficientnet_cheatsheet.html))
1. [Attention для Computer Vision (CBAM, SENet)](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/vision_attention_mechanisms_cheatsheet.html)
1. [Neural Architecture Design Patterns](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/neural_architecture_patterns_cheatsheet.html)
1. [Transfer learning для CNN](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/transfer_learning_cnn_cheatsheet.html)
1. [Data augmentation для изображений](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/data_augmentation_cheatsheet.html)
1. [Batch Normalization](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/batch_normalization_cheatsheet.html)
1. [Визуализация CNN](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/cnn_visualization_techniques_cheatsheet.html)
1. [1D-CNN и 3D-CNN](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/1d_3d_cnn_cheatsheet.html)

### **3. Рекуррентные нейронные сети (RNN)**
1. [Основы RNN](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/rnn_basics_cheatsheet.html)
1. [Temporal Convolutional Networks (TCN)](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/tcn_cheatsheet.html)
1. [Проблема исчезающего градиента](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/vanishing_gradient_cheatsheet.html)
1. [LSTM (Long Short-Term Memory)](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/lstm_cheatsheet.html)
1. [GRU (Gated Recurrent Units)](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/gru_cheatsheet.html)
1. [Регуляризация RNN (Dropout, Recurrent Dropout, Zoneout)](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/rnn_regularization_cheatsheet.html)
1. [Двунаправленные RNN](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/bidirectional_rnn_cheatsheet.html)
1. [Архитектура Encoder-Decoder](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/encoder_decoder_cheatsheet.html)

### **4. Трансформеры и механизмы внимания**
1. [Механизм внимания (attention)](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/attention_mechanism_cheatsheet.html)
1. [Self-attention](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/self_attention_cheatsheet.html)
1. [Multi-head Attention](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/multi_head_attention_cheatsheet.html)
1. [Архитектура Transformer](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/transformers_cheatsheet.html)
1. [Positional Encoding](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/positional_encoding_cheatsheet.html)
1. [BERT и языковые модели](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/bert_language_models_cheatsheet.html)
1. [Vision Transformers (ViT)](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/vision_transformers_vit_cheatsheet.html)
1. [Efficient Transformers (Memory-efficient attention)](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/efficient_transformers_cheatsheet.html)

### **5. Генеративные модели**
1. [Автоэнкодеры](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/autoencoders_cheatsheet.html)
1. [Вариационные автоэнкодеры (VAE)](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/vae_cheatsheet.html)
1. [Генеративно-состязательные сети (GAN)](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/gan_cheatsheet.html)
1. [DCGAN и Conditional GAN](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/dcgan_conditional_gan_cheatsheet.html)
1. [StyleGAN](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/stylegan_architecture_cheatsheet.html)
1. [Diffusion модели](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/diffusion_models_cheatsheet.html)
1. [Нормализующие потоки (Normalizing Flows)](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/normalizing_flows_cheatsheet.html)

### **6. Обучение с подкреплением (Deep RL)**
1. [Deep Q-Networks (DQN)](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/dqn_cheatsheet.html)
1. [Policy Gradient методы](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/policy_gradient_cheatsheet.html)
1. [Actor-Critic методы](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/actor_critic_cheatsheet.html)
1. [PPO (Proximal Policy Optimization)](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/ppo_reinforcement_learning_cheatsheet.html)
1. [Многозадачное RL](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/multitask_rl_cheatsheet.html)
1. [Иерархическое RL](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/hierarchical_rl_cheatsheet.html)
1. [Имитационное обучение (Imitation Learning)](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/imitation_learning_cheatsheet.html)

### **7. Обработка естественного языка (NLP)**
1. [Векторные представления слов (Word2Vec, GloVe, FastText)](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/word_embeddings_cheatsheet.html)
1. [RNN/LSTM для NLP](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/rnn_lstm_nlp_cheatsheet.html)
1. [Transformer для NLP](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/transformer_nlp_cheatsheet.html)
1. [BERT и его вариации](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/bert_finetuning_cheatsheet.html)
1. [GPT-архитектуры](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/gpt_architectures_cheatsheet.html)
1. [Prompt Engineering для LLM (ChatGPT, GPT-4, Claude)](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/prompt_engineering_cheatsheet.html)
1. [Токенизация (BPE, WordPiece, SentencePiece)](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/tokenization_cheatsheet.html)
1. [Задачи NLP (NER, классификация, суммаризация)](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/nlp_tasks_ner_classification_summarization_cheatsheet.html)

### **8. Компьютерное зрение (Computer Vision)**
1. [Классификация изображений](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/image_classification_dl_cheatsheet.html)
1. [Детекция объектов (YOLO, SSD, Faster R-CNN)](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/yolo_cheatsheet.html)
1. [Детекция объектов - Faster R-CNN, SSD, RetinaNet, EfficientDet](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/object_detection_cheatsheet.html)
1. [Сегментация изображений (U-Net, DeepLab, Mask R-CNN)](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/image_segmentation_cheatsheet.html)
1. [Детекция ключевых точек и оценка позы](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/keypoint_detection_pose_estimation_cheatsheet.html)
1. [Слежение за объектами](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/object_tracking_video_cheatsheet.html)
1. [Оценка позы человека](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/pose_estimation_cheatsheet.html)
1. [3D Computer Vision](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/3d_computer_vision_cheatsheet.html)

### **9. Обработка аудио и речь**
1. [Спектрограммы и MFCC](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/audio_processing_spectrograms_mfcc_cheatsheet.html)
1. [Модели для распознавания речи (ASR)](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/asr_models_cheatsheet.html)
1. [Модели для синтеза речи (TTS)](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/tts_models_cheatsheet.html)
1. [Классификация звуков](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/audio_classification_cheatsheet.html)
1. [Source separation](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/source_separation_cheatsheet.html)
1. [Audio generation](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/audio_generation_cheatsheet.html)

### **10. Мультимодальное обучение**
1. [Модели для изображений и текста](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/image_text_models_cheatsheet.html)
1. [CLIP и похожие архитектуры](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/clip_multimodal_models_cheatsheet.html)
1. [Видео-аудио модели](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/video_audio_models_cheatsheet.html)
1. [Cross-modal retrieval](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/cross_modal_retrieval_cheatsheet.html)
1. [Мультимодальные трансформеры](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/multimodal_transformers_cheatsheet.html)

### **11. Малоресурсное обучение**
1. [Few-shot Learning (Siamese, Prototypical Networks, MAML)](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/few_shot_learning_cheatsheet.html)
1. [Metric Learning (Triplet Loss, ArcFace)](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/metric_learning_cheatsheet.html)
1. [Self-Supervised Learning](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/self_supervised_learning_cheatsheet.html)
1. [Contrastive Learning (SimCLR, MoCo, BYOL)](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/contrastive_learning_cheatsheet.html)
1. [Curriculum Learning](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/curriculum_learning_cheatsheet.html)
1. [Zero-shot learning](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/zero_shot_learning_cheatsheet.html)
1. [Transfer learning (Deep Learning)](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/transfer_learning_deep_cheatsheet.html)
1. [Meta-learning (MAML, Prototypical Networks)](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/meta_learning_maml_cheatsheet.html)
1. [Domain adaptation](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/domain_adaptation_cheatsheet.html)

### **12. Объяснимость и интерпретируемость**
1. [CAM и Grad-CAM](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/grad_cam_cheatsheet.html)
1. [Saliency maps](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/saliency_maps_cheatsheet.html)
1. [Integrated gradients](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/integrated_gradients_cheatsheet.html)
1. [LIME для нейросетей](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/lime_neural_networks_cheatsheet.html)
1. [SHAP для нейросетей](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/shap_neural_networks_cheatsheet.html)
1. [Concept activation vectors](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/concept_activation_vectors_cheatsheet.html)

### **13. Оптимизация и ускорение**
1. [Сжатие моделей (Квантизация, Прунинг, Knowledge Distillation)](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/model_compression_cheatsheet.html)
1. [Mixed precision training](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/mixed_precision_training_cheatsheet.html)
1. [Gradient checkpointing](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/gradient_checkpointing_cheatsheet.html)
1. [Distributed training](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/distributed_training_cheatsheet.html)
1. [Neural Architecture Search (NAS)](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/nas_cheatsheet.html)

### **14. Графовые нейронные сети (GNN)**
1. [Graph Neural Networks Basics (GCN, GAT, GraphSAGE)](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/gnn_basics_cheatsheet.html)
1. [Graph Convolutional Networks (GCN)](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/gcn_cheatsheet.html)
1. [Graph Attention Networks (GAT)](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/gat_cheatsheet.html)
1. [Message Passing Networks](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/message_passing_networks_cheatsheet.html)
1. [Применения GNN](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/gnn_applications_cheatsheet.html)

### **15. Временные ряды и прогнозирование**
1. [RNN/LSTM для временных рядов](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/rnn_lstm_time_series_cheatsheet.html)
1. [CNN для временных рядов (TCN)](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/cnn_time_series_tcn_cheatsheet.html)
1. [Transformers для временных рядов](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/transformers_time_series_cheatsheet.html)
1. [Многомерные временные ряды](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/multivariate_time_series_cheatsheet.html)
1. [Аномалии во временных рядах](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/time_series_anomalies_cheatsheet.html)
1. [Прогнозирование спроса и трафика](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/demand_traffic_forecasting_cheatsheet.html)

### **16. Нейросети для табличных данных**
1. [MLP для табличных данных](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/mlp_tabular_data_cheatsheet.html)
1. [TabNet](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/tabnet_cheatsheet.html)
1. [Neural Oblivious Decision Ensembles](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/node_ensemble_cheatsheet.html)
1. [AutoML для нейросетей](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/automl_neural_networks_cheatsheet.html)
1. [Сравнение с классическими методами](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/comparison_neural_classical_cheatsheet.html)

### **17. Этичные аспекты и безопасность**
1. [Adversarial attacks](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/adversarial_attacks_deep_learning_cheatsheet.html)
1. [Defensive distillation](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/defensive_distillation_cheatsheet.html)
1. [Adversarial training](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/adversarial_training_cheatsheet.html)
1. [Bias в нейросетях](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/bias_neural_networks_cheatsheet.html)
1. [Fairness в ML](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/fairness_ml_deep_learning_cheatsheet.html)
1. [Federated learning](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/federated_learning_cheatsheet.html)
1. [Differential privacy](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/differential_privacy_cheatsheet.html)

### **18. Производство и MLOps**
1. [Model Deployment](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/model_deployment_cheatsheet.html)
1. [Сериализация нейросетей](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/neural_network_serialization_cheatsheet.html)
1. [ONNX формат](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/onnx_format_cheatsheet.html)
1. [TensorFlow Serving](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/tensorflow_serving_cheatsheet.html)
1. [TorchServe](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/torchserve_cheatsheet.html)
1. [Мониторинг моделей](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/model_monitoring_cheatsheet.html)
1. [CI/CD для ML](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/cicd_ml_cheatsheet.html)
1. [Feature stores](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/feature_stores_cheatsheet.html)

### **19. Фреймворки и библиотеки**
1. [PyTorch полный гайд](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/pytorch_full_guide_cheatsheet.html)
1. [TensorFlow/Keras полный гайд](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/tensorflow_keras_full_guide_cheatsheet.html)
1. [JAX и Flax](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/jax_flax_cheatsheet.html)
1. [Hugging Face Transformers](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/hugging_face_transformers_cheatsheet.html)
1. [PyTorch Lightning](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/pytorch_lightning_cheatsheet.html)
1. [FastAI](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/fastai_cheatsheet.html)

### **20. Теория глубокого обучения**
1. [Теорема универсальной аппроксимации](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/universal_approximation_theorem_cheatsheet.html)
1. [Теория обобщения для нейросетей](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/generalization_theory_neural_networks_cheatsheet.html)
1. [Динамика обучения нейросетей](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/neural_network_training_dynamics_cheatsheet.html)
1. [Neural tangent kernel](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/neural_tangent_kernel_cheatsheet.html)
1. [Information bottleneck theory](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/information_bottleneck_theory_cheatsheet.html)

### **21. Специальные архитектуры**
1. [Capsule networks](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/capsule_networks_cheatsheet.html)
1. [Spiking neural networks](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/spiking_neural_networks_cheatsheet.html)
1. [Neural ODEs](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/neural_odes_cheatsheet.html)
1. [Differentiable programming](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/differentiable_programming_cheatsheet.html)
1. [Memory networks](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/memory_networks_cheatsheet.html)
1. [Hypernetworks](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/hypernetworks_cheatsheet.html)

### **22. Нейронаука и биологически правдоподобные модели**
1. [Биологически вдохновленные архитектуры](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/biologically_inspired_architectures_cheatsheet.html)
1. [Нейроморфные вычисления](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/neuromorphic_computing_cheatsheet.html)
1. [Связь нейросетей и мозга](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/brain_neural_networks_connection_cheatsheet.html)

### **23. Креативные приложения**
1. [Neural style transfer](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/neural_style_transfer_cheatsheet.html)
1. [Image-to-image translation](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/image_to_image_translation_cheatsheet.html)
1. [Text-to-image generation](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/text_to_image_generation_cheatsheet.html)
1. [Музыкальные нейросети](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/musical_neural_networks_cheatsheet.html)
1. [Генерация видео](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/video_generation_cheatsheet.html)

### **24. Научные применения**
1. [Нейросети в физике](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/neural_networks_physics_cheatsheet.html)
1. [Нейросети в химии](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/neural_networks_chemistry_cheatsheet.html)
1. [Нейросети в биологии](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/neural_networks_biology_cheatsheet.html)
1. [Нейросети в медицине](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/neural_networks_medicine_cheatsheet.html)
1. [Нейросети в астрономии](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/neural_networks_astronomy_cheatsheet.html)

### **25. История и эволюция**
1. [Хронология глубокого обучения](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/deep_learning_timeline_cheatsheet.html)
1. [Ключевые прорывы](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/deep_learning_key_breakthroughs_cheatsheet.html)
1. [Современные тенденции](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/modern_trends_deep_learning_cheatsheet.html)
1. [Будущее глубокого обучения](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/future_deep_learning_cheatsheet.html)

### **26. Квантовое машинное обучение**
1. [Квантовые алгоритмы для ML](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/quantum_algorithms_ml_cheatsheet.html)
1. [Квантовые нейронные сети](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/quantum_neural_networks_cheatsheet.html)
1. [Квантовое усиление классических алгоритмов](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/quantum_classical_enhancement_cheatsheet.html)

### **27. Нейро-символьное ИИ**
1. [Гибридные нейро-символьные системы](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/neuro_symbolic_ai_systems_cheatsheet.html)
1. [Rule-based системы с ML](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/rule_based_ml_systems_cheatsheet.html)
1. [Логический вывод с обучением](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/logical_inference_learning_cheatsheet.html)

## III Разные темы машинного обучения

### **1. Мета-обучение (Meta-learning) - продвинутые темы**
1. [Model-agnostic meta-learning (MAML)](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/maml_meta_learning_cheatsheet.html)
1. [Metric-based (Siamese networks, Matching networks)](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/metric_based_learning_siamese_cheatsheet.html)
1. [Optimization-based meta-learning](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/optimization_based_meta_learning_cheatsheet.html)
1. [Few-shot learning (традиционные подходы)](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/few_shot_learning_traditional_cheatsheet.html)
1. [Meta-features и meta-learning](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/meta_features_learning_cheatsheet.html)

### **2. Специализированные прикладные области**
1. [Биоинформатика и computational biology](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/bioinformatics_computational_biology_cheatsheet.html)
1. [Computational chemistry](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/computational_chemistry_ml_cheatsheet.html)
1. [Астрофизика и космология](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/astrophysics_cosmology_ml_cheatsheet.html)
1. [Геоинформационные системы](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/geoinformation_systems_ml_cheatsheet.html)
1. [Цифровая гуманитаристика](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/digital_humanities_cheatsheet.html)
1. [Computational social science](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/computational_social_science_cheatsheet.html)

### **3. Композиционное и структурное обучение**
1. [Grammatical inference](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/grammatical_inference_cheatsheet.html)
1. [Inductive logic programming](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/inductive_logic_programming_cheatsheet.html)
1. [Program synthesis](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/program_synthesis_ml_cheatsheet.html)

### **4. Материаловедение и ML**
1. [Материальные информатики](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/material_informatics_cheatsheet.html)
1. [Свойства материалов и ML](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/material_properties_ml_cheatsheet.html)
1. [Дизайн материалов](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/material_design_cheatsheet.html)

### **5. Квантовое машинное обучение**
1. [Квантовые алгоритмы для ML](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/quantum_algorithms_ml_cheatsheet.html)
1. [Квантовые нейронные сети](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/quantum_neural_networks_cheatsheet.html)
1. [Квантовое усиление классических алгоритмов](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/quantum_classical_enhancement_cheatsheet.html)

### **6. Нейро-символьное ИИ**
1. [Гибридные нейро-символьные системы](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/neuro_symbolic_ai_systems_cheatsheet.html)
1. [Rule-based системы с ML](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/rule_based_ml_systems_cheatsheet.html)
1. [Логический вывод с обучением](https://html-preview.github.io/?url=https://github.com/gurovic/MLCheatSheets/blob/main/cheatsheets/logical_inference_learning_cheatsheet.html)

---

## 🤝 Как внести вклад

Мы приветствуем любой вклад в проект! Вы можете:

- 🐛 Сообщить об ошибке через [Issues](https://github.com/gurovic/MLCheatSheets/issues)
- ✨ Предложить новый cheatsheet или улучшение
- 📝 Исправить опечатки или улучшить формулировки
- 💻 Добавить примеры кода или упражнения
- 🎨 Создать визуализации или анимации

**Пожалуйста, прочитайте [руководство по внесению вклада](CONTRIBUTING.md) перед началом работы.**

## 📜 Лицензия

Этот проект распространяется под лицензией MIT. Подробности см. в файле [LICENSE](LICENSE).

## 🌟 Поддержка проекта

Если этот проект был вам полезен:

- ⭐ Поставьте звезду на GitHub
- 🔄 Поделитесь с друзьями и коллегами
- 📢 Упомяните проект в своих работах
- 🤝 Внесите свой вклад

## 📞 Контакты и связь

- **GitHub**: [gurovic/MLCheatSheets](https://github.com/gurovic/MLCheatSheets)
- **Issues**: [Задать вопрос или сообщить о проблеме](https://github.com/gurovic/MLCheatSheets/issues)

## 🙏 Благодарности

Особая благодарность:

- Школе "Летово" за поддержку проекта
- DeepSeek, Github Copilot, Perplexity Comet за помощь в создании материалов
- Всем контрибьюторам проекта
- Сообществу ML энтузиастов

## 📚 Дополнительные ресурсы

### Документация проекта

- 📖 [Дорожная карта обучения](ROADMAP.md) - структурированные пути изучения ML
- 💪 [Упражнения](exercises.md) - практические задания
- 🎥 [Анимации](animations/README.md) - визуализации алгоритмов
- 🤝 [Руководство по внесению вклада](CONTRIBUTING.md)
- 📜 [Кодекс поведения](CODE_OF_CONDUCT.md)

### Рекомендуемые ресурсы для изучения ML

**Онлайн курсы:**
- [Andrew Ng's Machine Learning Course](https://www.coursera.org/learn/machine-learning) (Coursera)
- [Deep Learning Specialization](https://www.coursera.org/specializations/deep-learning) (Coursera)
- [Fast.ai Practical Deep Learning](https://course.fast.ai/)

**Книги:**
- "Hands-On Machine Learning" by Aurélien Géron
- "Deep Learning" by Ian Goodfellow
- "Pattern Recognition and Machine Learning" by Christopher Bishop

**Практика:**
- [Kaggle](https://www.kaggle.com/) - соревнования и датасеты
- [Papers with Code](https://paperswithcode.com/) - статьи с реализациями
- [HuggingFace](https://huggingface.co/) - модели и датасеты

---

<div align="center">

**Сделано с ❤️ для ML сообщества**

© 2024 ML Cheatsheets. Автор: Владимир Гуровиц (школа "Летово")

[⬆ Наверх](#-ml-cheatsheets)

</div>
