#!/usr/bin/env python3
import os

# HTML template (condensed)
def create_html(title, subtitle, sections):
    content_blocks = ""
    for section in sections:
        content_blocks += f'''
  <div class="block">
    <h2>{section['num']}. {section['title']}</h2>
{section['content']}
  </div>
'''
    
    html = f'''<!DOCTYPE html>
<html lang="ru">
<head>
  <meta charset="UTF-8">
  <title>{title} Cheatsheet — 3 колонки</title>
  <style>
    @media screen {{body {{font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif;color: #333;background: #fafcff;padding: 10px;}}}}
    @media print {{body {{background: white;padding: 0;}}@page {{size: A4 landscape;margin: 10mm;}}}}
    .container {{column-count: 3;column-gap: 20px;max-width: 100%;}}
    .block {{break-inside: avoid;margin-bottom: 1.2em;padding: 12px;background: white;border-radius: 6px;box-shadow: 0 1px 3px rgba(0,0,0,0.05);}}
    h1 {{font-size: 1.6em;font-weight: 700;color: #1a5fb4;text-align: center;margin: 0 0 8px;column-span: all;}}
    .subtitle {{text-align: center;color: #666;font-size: 0.9em;margin-bottom: 12px;column-span: all;}}
    h2 {{font-size: 1.15em;font-weight: 700;color: #1a5fb4;margin: 0 0 8px;padding-bottom: 4px;border-bottom: 1px solid #e0e7ff;}}
    p, ul, ol {{font-size: 0.92em;margin: 0.6em 0;}}
    ul, ol {{padding-left: 18px;}}
    li {{margin-bottom: 4px;}}
    code {{font-family: 'Consolas', 'Courier New', monospace;background-color: #f0f4ff;padding: 1px 4px;border-radius: 3px;font-size: 0.88em;}}
    pre {{background-color: #f0f4ff;padding: 8px;border-radius: 4px;overflow-x: auto;font-size: 0.84em;margin: 6px 0;}}
    pre code {{padding: 0;background: none;white-space: pre-wrap;}}
    strong {{color: #1a5fb4;font-weight: 600;}}
    .formula {{background: #fff9e6;padding: 6px;border-left: 3px solid #ffcc00;margin: 8px 0;font-style: italic;}}
    table {{width: 100%;border-collapse: collapse;font-size: 0.88em;margin: 8px 0;}}
    table th {{background-color: #e0e7ff;padding: 6px;text-align: left;font-weight: 600;}}
    table td {{padding: 5px 6px;border-bottom: 1px solid #e0e7ff;}}
  </style>
</head>
<body>
<h1>{title}</h1>
<div class="subtitle">{subtitle}</div>
<div class="container">
{content_blocks}
</div>
</body>
</html>'''
    return html

# Cheatsheets data (condensed comprehensive content)
cheatsheets = [
    {
        "filename": "matrix_factorization_svd_cheatsheet.html",
        "title": "Матричная факторизация (SVD) для рекомендательных систем",
        "subtitle": "Разложение матриц для collaborative filtering и рекомендаций",
        "sections": [
            {"num": 1, "title": "Основы SVD", "content": "<p><strong>Singular Value Decomposition</strong> — разложение матрицы на три матрицы.</p><div class='formula'>A = UΣVᵀ</div><ul><li>U: левые сингулярные векторы (пользователи)</li><li>Σ: сингулярные значения (диагональная)</li><li>V: правые сингулярные векторы (items)</li></ul><pre><code>import numpy as np\nfrom scipy.linalg import svd\n\n# SVD разложение\nU, sigma, Vt = svd(rating_matrix, full_matrices=False)</code></pre>"},
            {"num": 2, "title": "SVD для рекомендаций", "content": "<p><strong>Применение SVD:</strong></p><ul><li>Понижение размерности</li><li>Заполнение пропусков в матрице рейтингов</li><li>Предсказание рейтингов</li></ul><pre><code># Реконструкция с k компонентами\nk = 50\nU_k = U[:, :k]\nsigma_k = np.diag(sigma[:k])\nVt_k = Vt[:k, :]\n\n# Предсказанные рейтинги\nratings_pred = U_k @ sigma_k @ Vt_k</code></pre>"},
            {"num": 3, "title": "Truncated SVD", "content": "<p><strong>Усеченное SVD</strong> — только k наибольших сингулярных значений.</p><pre><code>from sklearn.decomposition import TruncatedSVD\n\nsvd = TruncatedSVD(n_components=50, random_state=42)\nuser_factors = svd.fit_transform(rating_matrix)\nitem_factors = svd.components_.T\n\n# Предсказание\ndef predict(user_id, item_id):\n    return user_factors[user_id] @ item_factors[item_id]</code></pre>"},
            {"num": 4, "title": "SVD++ расширение", "content": "<p><strong>SVD++</strong> — учитывает неявные отзывы.</p><div class='formula'>r̂_ui = μ + b_u + b_i + qᵢᵀ(p_u + |N(u)|^(-0.5)Σ_j∈N(u) y_j)</div><p>где N(u) — множество items с которыми взаимодействовал пользователь u</p>"},
            {"num": 5, "title": "Работа с пропусками", "content": "<p><strong>Проблема:</strong> В реальных данных много пропусков (sparse matrix).</p><pre><code># Заполнение средними\nuser_mean = np.nanmean(rating_matrix, axis=1, keepdims=True)\nrating_matrix_filled = np.where(\n    np.isnan(rating_matrix),\n    user_mean,\n    rating_matrix\n)\n\n# Или item mean\nitem_mean = np.nanmean(rating_matrix, axis=0, keepdims=True)</code></pre>"},
            {"num": 6, "title": "NMF (Non-negative Matrix Factorization)", "content": "<p><strong>Неотрицательная факторизация</strong> — W и H неотрицательны.</p><div class='formula'>A ≈ WH, W ≥ 0, H ≥ 0</div><pre><code>from sklearn.decomposition import NMF\n\nnmf = NMF(n_components=50, init='nndsvd', random_state=42)\nW = nmf.fit_transform(rating_matrix)\nH = nmf.components_\n\n# Предсказанная матрица\nrating_pred = W @ H</code></pre>"},
            {"num": 7, "title": "Регуляризованная факторизация", "content": "<p><strong>С L2 регуляризацией:</strong></p><div class='formula'>min Σ(r_ui - p_uᵀq_i)² + λ(||p_u||² + ||q_i||²)</div><p>Предотвращает переобучение</p><pre><code>from sklearn.decomposition import NMF\n\nnmf = NMF(\n    n_components=50,\n    alpha=0.01,  # L2 regularization\n    l1_ratio=0,\n    random_state=42\n)</code></pre>"},
            {"num": 8, "title": "Градиентный спуск для MF", "content": "<pre><code>def matrix_factorization(R, K, alpha=0.002, beta=0.02, iterations=5000):\n    m, n = R.shape\n    P = np.random.rand(m, K)\n    Q = np.random.rand(n, K)\n    \n    for iteration in range(iterations):\n        for i in range(m):\n            for j in range(n):\n                if R[i, j] > 0:\n                    eij = R[i, j] - np.dot(P[i, :], Q[j, :].T)\n                    P[i, :] += alpha * (2 * eij * Q[j, :] - beta * P[i, :])\n                    Q[j, :] += alpha * (2 * eij * P[i, :] - beta * Q[j, :])\n        \n        if iteration % 100 == 0:\n            print(f'Iteration {iteration}, MSE: {compute_mse(R, P, Q):.4f}')\n    \n    return P, Q</code></pre>"},
            {"num": 9, "title": "Оценка качества", "content": "<p><strong>Метрики:</strong></p><ul><li><strong>RMSE:</strong> корень из среднеквадратичной ошибки</li><li><strong>MAE:</strong> средняя абсолютная ошибка</li><li><strong>Precision@K, Recall@K</strong></li></ul><pre><code>def rmse(true, pred):\n    mask = ~np.isnan(true)\n    return np.sqrt(np.mean((true[mask] - pred[mask]) ** 2))\n\ndef mae(true, pred):\n    mask = ~np.isnan(true)\n    return np.mean(np.abs(true[mask] - pred[mask]))</code></pre>"},
            {"num": 10, "title": "Cold start проблема", "content": "<p><strong>Решения:</strong></p><ul><li>Гибридные методы (content + collaborative)</li><li>Использование сайд-информации</li><li>Популярные рекомендации для новых</li><li>Feature-based факторизация</li></ul>"},
        ]
    },
    {
        "filename": "als_alternating_least_squares_cheatsheet.html",
        "title": "ALS (Alternating Least Squares)",
        "subtitle": "Метод чередующихся наименьших квадратов для рекомендательных систем",
        "sections": [
            {"num": 1, "title": "Основная идея ALS", "content": "<p><strong>ALS</strong> — итеративный алгоритм для матричной факторизации.</p><p><strong>Принцип:</strong> Чередуем оптимизацию факторов пользователей и items.</p><div class='formula'>min Σ(r_ui - x_uᵀy_i)² + λ(Σ||x_u||² + Σ||y_i||²)</div>"},
            {"num": 2, "title": "Алгоритм ALS", "content": "<p><strong>Шаги:</strong></p><ol><li>Инициализировать X (user factors) случайно</li><li>Зафиксировать X, оптимизировать Y (item factors)</li><li>Зафиксировать Y, оптимизировать X</li><li>Повторять до сходимости</li></ol><p>Каждый шаг решается как задача наименьших квадратов.</p>"},
            {"num": 3, "title": "Реализация с NumPy", "content": "<pre><code>def als(R, K, lambda_reg=0.1, iterations=10):\n    m, n = R.shape\n    X = np.random.rand(m, K)\n    Y = np.random.rand(n, K)\n    \n    for iteration in range(iterations):\n        # Фиксируем X, обновляем Y\n        for i in range(n):\n            users = np.where(R[:, i] > 0)[0]\n            if len(users) > 0:\n                X_u = X[users, :]\n                r_i = R[users, i]\n                Y[i, :] = np.linalg.solve(\n                    X_u.T @ X_u + lambda_reg * np.eye(K),\n                    X_u.T @ r_i\n                )\n        \n        # Фиксируем Y, обновляем X\n        for u in range(m):\n            items = np.where(R[u, :] > 0)[0]\n            if len(items) > 0:\n                Y_i = Y[items, :]\n                r_u = R[u, items]\n                X[u, :] = np.linalg.solve(\n                    Y_i.T @ Y_i + lambda_reg * np.eye(K),\n                    Y_i.T @ r_u\n                )\n        \n        # Вычислить loss\n        loss = compute_loss(R, X, Y, lambda_reg)\n        print(f'Iteration {iteration}, Loss: {loss:.4f}')\n    \n    return X, Y</code></pre>"},
            {"num": 4, "title": "Implicit feedback ALS", "content": "<p><strong>Для неявной обратной связи</strong> (клики, просмотры, покупки).</p><div class='formula'>min Σ c_ui(p_ui - x_uᵀy_i)² + λ(Σ||x_u||² + Σ||y_i||²)</div><p>где:</p><ul><li>p_ui ∈ {0,1}: предпочтение</li><li>c_ui = 1 + α·r_ui: уверенность</li></ul><pre><code># Библиотека implicit\nfrom implicit.als import AlternatingLeastSquares\n\nmodel = AlternatingLeastSquares(\n    factors=50,\n    regularization=0.01,\n    iterations=15,\n    calculate_training_loss=True\n)\n\nmodel.fit(sparse_item_user_matrix)\n\n# Рекомендации\nrecommendations = model.recommend(\n    userid=0,\n    user_items=user_item_matrix[0],\n    N=10\n)</code></pre>"},
            {"num": 5, "title": "Weighted ALS", "content": "<p><strong>Веса для уверенности:</strong></p><pre><code>def weighted_als(R, W, K, lambda_reg=0.1, iterations=10):\n    \"\"\"\n    W: матрица весов (уверенности)\n    \"\"\"\n    m, n = R.shape\n    X = np.random.rand(m, K)\n    Y = np.random.rand(n, K)\n    \n    for iteration in range(iterations):\n        # Update Y\n        for i in range(n):\n            W_i = np.diag(W[:, i])\n            Y[i, :] = np.linalg.solve(\n                X.T @ W_i @ X + lambda_reg * np.eye(K),\n                X.T @ W_i @ R[:, i]\n            )\n        \n        # Update X\n        for u in range(m):\n            W_u = np.diag(W[u, :])\n            X[u, :] = np.linalg.solve(\n                Y.T @ W_u @ Y + lambda_reg * np.eye(K),\n                Y.T @ W_u @ R[u, :]\n            )\n    \n    return X, Y</code></pre>"},
            {"num": 6, "title": "Spark MLlib ALS", "content": "<p><strong>Для больших данных:</strong></p><pre><code>from pyspark.ml.recommendation import ALS\n\nals = ALS(\n    maxIter=10,\n    regParam=0.1,\n    userCol='userId',\n    itemCol='itemId',\n    ratingCol='rating',\n    coldStartStrategy='drop',\n    implicitPrefs=False  # True для implicit\n)\n\nmodel = als.fit(train_df)\n\n# Предсказания\npredictions = model.transform(test_df)\n\n# Топ рекомендации\nuser_recs = model.recommendForAllUsers(10)\nitem_recs = model.recommendForAllItems(10)</code></pre>"},
            {"num": 7, "title": "Гиперпараметры ALS", "content": "<p><strong>Ключевые параметры:</strong></p><ul><li><strong>factors (K):</strong> число латентных факторов (10-200)</li><li><strong>regularization (λ):</strong> регуляризация (0.01-0.1)</li><li><strong>iterations:</strong> число итераций (10-20)</li><li><strong>alpha:</strong> для implicit (1-40)</li></ul><pre><code># Grid search\nfrom sklearn.model_selection import ParameterGrid\n\nparam_grid = {\n    'factors': [50, 100, 150],\n    'regularization': [0.01, 0.05, 0.1],\n    'iterations': [10, 15, 20]\n}\n\nfor params in ParameterGrid(param_grid):\n    model = AlternatingLeastSquares(**params)\n    # обучение и валидация</code></pre>"},
            {"num": 8, "title": "Оценка качества", "content": "<pre><code>def precision_at_k(recommended, relevant, k=10):\n    recommended_k = recommended[:k]\n    return len(set(recommended_k) & set(relevant)) / k\n\ndef recall_at_k(recommended, relevant, k=10):\n    recommended_k = recommended[:k]\n    return len(set(recommended_k) & set(relevant)) / len(relevant)\n\ndef mean_average_precision(recommended, relevant):\n    precisions = []\n    for k in range(1, len(recommended) + 1):\n        if recommended[k-1] in relevant:\n            precisions.append(precision_at_k(recommended, relevant, k))\n    return np.mean(precisions) if precisions else 0</code></pre>"},
            {"num": 9, "title": "Преимущества ALS", "content": "<ul><li>✓ Эффективен для больших sparse матриц</li><li>✓ Параллелизуется легко</li><li>✓ Работает с implicit feedback</li><li>✓ Масштабируется (Spark)</li><li>✓ Гарантирует сходимость</li></ul>"},
            {"num": 10, "title": "Практические советы", "content": "<ul><li>Начните с K=50-100</li><li>Используйте cross-validation</li><li>Нормализуйте рейтинги</li><li>Добавьте side features при возможности</li><li>Мониторьте training loss</li><li>Early stopping по validation метрикам</li></ul>"},
        ]
    }
]

for cheatsheet in cheatsheets:
    html_content = create_html(
        cheatsheet["title"],
        cheatsheet["subtitle"],
        cheatsheet["sections"]
    )
    filepath = os.path.join("cheatsheets", cheatsheet["filename"])
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(html_content)
    print(f"Created: {filepath} ({len(html_content)} bytes)")

print("\nAll cheatsheets created successfully!")
