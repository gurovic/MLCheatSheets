#!/usr/bin/env python3
"""
Generate matplotlib illustrations for recommender systems cheatsheets.
This script creates high-quality visualizations and encodes them as base64 for inline HTML embedding.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from scipy.sparse.linalg import svds
import base64
from io import BytesIO
import warnings
warnings.filterwarnings('ignore')

# Set consistent style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9

def fig_to_base64(fig):
    """Convert matplotlib figure to base64 string."""
    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=300)
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()
    plt.close(fig)
    return img_base64

# ============================================================================
# COLLABORATIVE FILTERING ILLUSTRATIONS
# ============================================================================

def generate_user_item_matrix():
    """Generate user-item rating matrix heatmap."""
    np.random.seed(42)
    
    # Create sample rating matrix with some missing values
    ratings = np.array([
        [5, 3, 0, 1, 4],
        [4, 0, 0, 1, 2],
        [1, 1, 0, 5, 4],
        [0, 1, 5, 4, 0],
        [2, 4, 3, 0, 5],
        [5, 0, 4, 2, 3]
    ])
    
    # Mask zeros for visualization
    masked_ratings = np.ma.masked_where(ratings == 0, ratings)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Create heatmap
    im = ax.imshow(masked_ratings, cmap='YlOrRd', aspect='auto', vmin=1, vmax=5)
    
    # Set ticks
    ax.set_xticks(np.arange(5))
    ax.set_yticks(np.arange(6))
    ax.set_xticklabels([f'Товар {i+1}' for i in range(5)])
    ax.set_yticklabels([f'User {i+1}' for i in range(6)])
    
    # Add text annotations
    for i in range(6):
        for j in range(5):
            if ratings[i, j] > 0:
                text = ax.text(j, i, f'{ratings[i, j]:.0f}',
                             ha="center", va="center", color="black", fontsize=11, fontweight='bold')
            else:
                text = ax.text(j, i, '?',
                             ha="center", va="center", color="gray", fontsize=11)
    
    ax.set_title('Матрица рейтингов пользователь-товар\n(0 = отсутствующий рейтинг)')
    ax.set_xlabel('Товары')
    ax.set_ylabel('Пользователи')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Рейтинг', rotation=270, labelpad=20)
    
    plt.tight_layout()
    return fig_to_base64(fig)

def generate_similarity_matrices():
    """Generate user and item similarity matrices side by side."""
    np.random.seed(42)
    
    # Sample rating matrix
    ratings = np.array([
        [5, 3, 0, 1],
        [4, 0, 0, 1],
        [1, 1, 0, 5],
        [0, 1, 5, 4]
    ])
    
    # Replace zeros with column means for similarity calculation
    ratings_filled = ratings.copy().astype(float)
    for j in range(ratings.shape[1]):
        col_mean = ratings[:, j][ratings[:, j] > 0].mean()
        ratings_filled[ratings_filled[:, j] == 0, j] = col_mean
    
    # Calculate similarities
    user_similarity = cosine_similarity(ratings_filled)
    item_similarity = cosine_similarity(ratings_filled.T)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # User similarity
    im1 = axes[0].imshow(user_similarity, cmap='coolwarm', vmin=0, vmax=1)
    axes[0].set_xticks(np.arange(4))
    axes[0].set_yticks(np.arange(4))
    axes[0].set_xticklabels([f'U{i+1}' for i in range(4)])
    axes[0].set_yticklabels([f'U{i+1}' for i in range(4)])
    axes[0].set_title('User Similarity (User-based CF)')
    
    # Add values
    for i in range(4):
        for j in range(4):
            text = axes[0].text(j, i, f'{user_similarity[i, j]:.2f}',
                              ha="center", va="center", 
                              color="white" if user_similarity[i, j] > 0.5 else "black",
                              fontsize=9)
    
    plt.colorbar(im1, ax=axes[0])
    
    # Item similarity
    im2 = axes[1].imshow(item_similarity, cmap='coolwarm', vmin=0, vmax=1)
    axes[1].set_xticks(np.arange(4))
    axes[1].set_yticks(np.arange(4))
    axes[1].set_xticklabels([f'I{i+1}' for i in range(4)])
    axes[1].set_yticklabels([f'I{i+1}' for i in range(4)])
    axes[1].set_title('Item Similarity (Item-based CF)')
    
    # Add values
    for i in range(4):
        for j in range(4):
            text = axes[1].text(j, i, f'{item_similarity[i, j]:.2f}',
                              ha="center", va="center",
                              color="white" if item_similarity[i, j] > 0.5 else "black",
                              fontsize=9)
    
    plt.colorbar(im2, ax=axes[1])
    
    plt.tight_layout()
    return fig_to_base64(fig)

def generate_cf_prediction_example():
    """Generate collaborative filtering prediction visualization."""
    np.random.seed(42)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Sample data for demonstration
    users = ['User 1', 'User 2', 'User 3', 'User 4', 'Target User']
    similarities = [0.85, 0.72, 0.45, 0.15]
    ratings = [5, 4, 3, 2]
    
    # Create bar chart
    y_pos = np.arange(len(similarities))
    colors = plt.cm.viridis(np.array(similarities))
    
    bars = ax.barh(y_pos, similarities, color=colors, alpha=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([f'{users[i]}\n(рейтинг: {ratings[i]})' for i in range(len(similarities))])
    ax.set_xlabel('Cosine Similarity с Target User')
    ax.set_title('User-based CF: Поиск похожих пользователей для предсказания')
    ax.set_xlim(0, 1)
    
    # Add value labels
    for i, (bar, sim, rating) in enumerate(zip(bars, similarities, ratings)):
        width = bar.get_width()
        ax.text(width + 0.02, bar.get_y() + bar.get_height()/2,
                f'{sim:.2f}',
                ha='left', va='center', fontweight='bold')
    
    # Add prediction formula
    weighted_sum = sum(s * r for s, r in zip(similarities, ratings))
    similarity_sum = sum(similarities)
    predicted = weighted_sum / similarity_sum
    
    ax.text(0.5, -0.8, 
            f'Предсказанный рейтинг = Σ(similarity × rating) / Σ(similarity)\n' +
            f'= ({similarities[0]:.2f}×{ratings[0]} + {similarities[1]:.2f}×{ratings[1]} + ...) / {similarity_sum:.2f}\n' +
            f'= {predicted:.2f}',
            ha='center', va='top', fontsize=10, 
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3),
            transform=ax.transData)
    
    plt.tight_layout()
    return fig_to_base64(fig)

# ============================================================================
# MATRIX FACTORIZATION (SVD) ILLUSTRATIONS
# ============================================================================

def generate_svd_decomposition():
    """Generate SVD matrix decomposition visualization."""
    fig, ax = plt.subplots(figsize=(12, 5))
    
    # Define box properties
    def draw_matrix(ax, x, y, width, height, text, color='lightblue'):
        rect = plt.Rectangle((x, y), width, height, 
                            facecolor=color, edgecolor='black', linewidth=2)
        ax.add_patch(rect)
        ax.text(x + width/2, y + height/2, text,
               ha='center', va='center', fontsize=11, fontweight='bold')
    
    # Draw matrices
    # R matrix
    draw_matrix(ax, 0, 2, 2, 3, 'R\n(m×n)\nРейтинги', 'lightcoral')
    
    # = sign
    ax.text(2.5, 3.5, '=', fontsize=24, ha='center', va='center')
    
    # U matrix
    draw_matrix(ax, 3, 2, 1.5, 3, 'U\n(m×k)\nПользователи', 'lightgreen')
    
    # × sign
    ax.text(4.8, 3.5, '×', fontsize=20, ha='center', va='center')
    
    # Σ matrix
    draw_matrix(ax, 5.2, 2.5, 1.5, 2, 'Σ\n(k×k)\nВеса', 'lightyellow')
    
    # × sign
    ax.text(7, 3.5, '×', fontsize=20, ha='center', va='center')
    
    # V^T matrix
    draw_matrix(ax, 7.5, 3, 2, 1.5, 'V^T\n(k×n)\nТовары', 'lightblue')
    
    # Add dimension labels
    ax.text(1, 0.8, 'm = количество пользователей', fontsize=9, style='italic')
    ax.text(1, 0.3, 'n = количество товаров', fontsize=9, style='italic')
    ax.text(1, -0.2, 'k = количество латентных факторов (k << min(m,n))', 
           fontsize=9, style='italic', color='red')
    
    ax.set_xlim(-0.5, 10)
    ax.set_ylim(-0.5, 5.5)
    ax.axis('off')
    ax.set_title('SVD разложение матрицы рейтингов', fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    return fig_to_base64(fig)

def generate_svd_example():
    """Generate example of SVD reconstruction."""
    np.random.seed(42)
    
    # Create sample rating matrix
    R = np.array([
        [5, 3, 0, 1],
        [4, 0, 0, 1],
        [1, 1, 0, 5],
        [0, 1, 5, 4],
    ], dtype=float)
    
    # Fill zeros with mean for SVD
    R_mean = np.mean(R[R > 0])
    R_filled = np.where(R > 0, R, R_mean)
    
    # Perform SVD
    k = 2
    U, sigma, Vt = svds(R_filled, k=k)
    sigma = np.diag(sigma)
    
    # Reconstruct
    R_reconstructed = np.dot(np.dot(U, sigma), Vt)
    
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    
    # Original matrix
    masked_R = np.ma.masked_where(R == 0, R)
    im1 = axes[0].imshow(masked_R, cmap='YlOrRd', aspect='auto', vmin=1, vmax=5)
    axes[0].set_title('Исходная матрица\n(с пропусками)')
    axes[0].set_xticks(np.arange(4))
    axes[0].set_yticks(np.arange(4))
    axes[0].set_xticklabels([f'I{i+1}' for i in range(4)])
    axes[0].set_yticklabels([f'U{i+1}' for i in range(4)])
    
    for i in range(4):
        for j in range(4):
            if R[i, j] > 0:
                axes[0].text(j, i, f'{R[i, j]:.0f}',
                           ha="center", va="center", color="black", fontweight='bold')
            else:
                axes[0].text(j, i, '?',
                           ha="center", va="center", color="gray")
    
    plt.colorbar(im1, ax=axes[0])
    
    # Reconstructed matrix (all values)
    im2 = axes[1].imshow(R_reconstructed, cmap='YlOrRd', aspect='auto', vmin=1, vmax=5)
    axes[1].set_title(f'Восстановленная (k={k})\n(предсказания)')
    axes[1].set_xticks(np.arange(4))
    axes[1].set_yticks(np.arange(4))
    axes[1].set_xticklabels([f'I{i+1}' for i in range(4)])
    axes[1].set_yticklabels([f'U{i+1}' for i in range(4)])
    
    for i in range(4):
        for j in range(4):
            axes[1].text(j, i, f'{R_reconstructed[i, j]:.1f}',
                       ha="center", va="center", color="black", fontsize=9)
    
    plt.colorbar(im2, ax=axes[1])
    
    # Error matrix
    error = np.abs(R - R_reconstructed)
    error_masked = np.ma.masked_where(R == 0, error)
    im3 = axes[2].imshow(error_masked, cmap='Reds', aspect='auto', vmin=0, vmax=1)
    axes[2].set_title('Ошибка\n(только известные)')
    axes[2].set_xticks(np.arange(4))
    axes[2].set_yticks(np.arange(4))
    axes[2].set_xticklabels([f'I{i+1}' for i in range(4)])
    axes[2].set_yticklabels([f'U{i+1}' for i in range(4)])
    
    for i in range(4):
        for j in range(4):
            if R[i, j] > 0:
                axes[2].text(j, i, f'{error[i, j]:.2f}',
                           ha="center", va="center", color="black", fontsize=9)
    
    plt.colorbar(im3, ax=axes[2])
    
    plt.tight_layout()
    return fig_to_base64(fig)

def generate_svd_k_selection():
    """Generate plot showing impact of k (number of factors) on RMSE."""
    np.random.seed(42)
    
    # Create larger sample rating matrix
    n_users, n_items = 50, 30
    R = np.random.randint(1, 6, size=(n_users, n_items)).astype(float)
    
    # Randomly set 30% to zero (missing)
    mask = np.random.random((n_users, n_items)) < 0.3
    R[mask] = 0
    
    # Store original non-zero values
    R_test = R.copy()
    
    # Fill for SVD
    R_mean = np.mean(R[R > 0])
    R_filled = np.where(R > 0, R, R_mean)
    
    k_values = [1, 2, 5, 10, 15, 20, 25]
    rmse_values = []
    
    for k in k_values:
        if k >= min(n_users, n_items):
            break
        U, sigma, Vt = svds(R_filled, k=k)
        sigma = np.diag(sigma)
        R_pred = np.dot(np.dot(U, sigma), Vt)
        
        # Calculate RMSE only on known ratings
        mask_known = R_test > 0
        rmse = np.sqrt(np.mean((R_test[mask_known] - R_pred[mask_known])**2))
        rmse_values.append(rmse)
    
    fig, ax = plt.subplots(figsize=(9, 6))
    
    ax.plot(k_values[:len(rmse_values)], rmse_values, 'o-', linewidth=2, markersize=8, color='darkblue')
    ax.set_xlabel('Количество латентных факторов (k)')
    ax.set_ylabel('RMSE')
    ax.set_title('Выбор оптимального количества факторов k\n(Trade-off между точностью и сложностью)')
    ax.grid(True, alpha=0.3)
    
    # Mark optimal k
    optimal_k_idx = np.argmin(rmse_values)
    optimal_k = k_values[optimal_k_idx]
    ax.axvline(x=optimal_k, color='red', linestyle='--', alpha=0.5, label=f'Оптимальное k={optimal_k}')
    ax.legend()
    
    # Add annotation
    ax.annotate(f'Минимум RMSE\nk={optimal_k}, RMSE={rmse_values[optimal_k_idx]:.3f}',
               xy=(optimal_k, rmse_values[optimal_k_idx]),
               xytext=(optimal_k + 3, rmse_values[optimal_k_idx] + 0.1),
               arrowprops=dict(arrowstyle='->', color='red', lw=2),
               fontsize=10, bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))
    
    plt.tight_layout()
    return fig_to_base64(fig)

# ============================================================================
# CONTENT-BASED FILTERING ILLUSTRATIONS
# ============================================================================

def generate_tfidf_example():
    """Generate TF-IDF feature matrix visualization."""
    # Sample movie data
    movies = ['Terminator', 'Titanic', 'Avatar', 'True Lies', 'Aliens']
    features = [
        'action scifi cameron',
        'drama romance cameron',
        'action scifi adventure cameron',
        'action comedy cameron',
        'action scifi cameron'
    ]
    
    # TF-IDF vectorization
    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform(features).toarray()
    feature_names = tfidf.get_feature_names_out()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create heatmap
    im = ax.imshow(tfidf_matrix, cmap='YlGnBu', aspect='auto')
    
    # Set ticks
    ax.set_xticks(np.arange(len(feature_names)))
    ax.set_yticks(np.arange(len(movies)))
    ax.set_xticklabels(feature_names, rotation=45, ha='right')
    ax.set_yticklabels(movies)
    
    # Add values
    for i in range(len(movies)):
        for j in range(len(feature_names)):
            text = ax.text(j, i, f'{tfidf_matrix[i, j]:.2f}',
                         ha="center", va="center", 
                         color="white" if tfidf_matrix[i, j] > 0.3 else "black",
                         fontsize=9)
    
    ax.set_title('TF-IDF матрица признаков фильмов\n(Content-based Filtering)')
    ax.set_xlabel('Признаки (слова)')
    ax.set_ylabel('Фильмы')
    
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    return fig_to_base64(fig)

def generate_content_similarity():
    """Generate content-based similarity visualization."""
    movies = ['Terminator', 'Titanic', 'Avatar', 'True Lies', 'Aliens']
    features = [
        'action scifi cameron',
        'drama romance cameron',
        'action scifi adventure cameron',
        'action comedy cameron',
        'action scifi cameron'
    ]
    
    # TF-IDF and similarity
    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform(features)
    similarity = cosine_similarity(tfidf_matrix)
    
    fig, ax = plt.subplots(figsize=(8, 7))
    
    # Create heatmap
    im = ax.imshow(similarity, cmap='coolwarm', vmin=0, vmax=1)
    
    # Set ticks
    ax.set_xticks(np.arange(len(movies)))
    ax.set_yticks(np.arange(len(movies)))
    ax.set_xticklabels(movies, rotation=45, ha='right')
    ax.set_yticklabels(movies)
    
    # Add values
    for i in range(len(movies)):
        for j in range(len(movies)):
            text = ax.text(j, i, f'{similarity[i, j]:.2f}',
                         ha="center", va="center",
                         color="white" if similarity[i, j] > 0.5 else "black",
                         fontsize=10, fontweight='bold')
    
    ax.set_title('Матрица схожести фильмов по контенту\n(Cosine Similarity)')
    
    plt.colorbar(im, ax=ax, label='Similarity')
    plt.tight_layout()
    return fig_to_base64(fig)

def generate_user_profile_example():
    """Generate user profile building visualization."""
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    
    # Movie features
    movies = ['Avatar', 'Terminator', 'Aliens']
    genres = ['Action', 'Sci-Fi', 'Adventure', 'Drama']
    
    # User liked these movies with these feature weights
    movie_features = np.array([
        [0.8, 0.9, 0.7, 0.1],  # Avatar
        [0.9, 0.8, 0.2, 0.0],  # Terminator
        [0.9, 0.9, 0.3, 0.0],  # Aliens
    ])
    
    user_ratings = np.array([5, 4, 5])
    
    # Calculate weighted user profile
    weighted_features = movie_features * user_ratings[:, np.newaxis]
    user_profile = weighted_features.mean(axis=0)
    
    # Plot 1: Movie features
    x = np.arange(len(genres))
    width = 0.25
    
    for i, (movie, features) in enumerate(zip(movies, movie_features)):
        axes[0].bar(x + i*width, features, width, label=f'{movie} (рейтинг: {user_ratings[i]})', alpha=0.8)
    
    axes[0].set_xlabel('Жанры')
    axes[0].set_ylabel('Вес признака')
    axes[0].set_title('Признаки фильмов, которые понравились пользователю')
    axes[0].set_xticks(x + width)
    axes[0].set_xticklabels(genres)
    axes[0].legend()
    axes[0].grid(axis='y', alpha=0.3)
    
    # Plot 2: User profile
    colors = plt.cm.viridis(user_profile / user_profile.max())
    bars = axes[1].bar(genres, user_profile, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    axes[1].set_xlabel('Жанры')
    axes[1].set_ylabel('Вес в профиле пользователя')
    axes[1].set_title('Профиль пользователя (усредненный по рейтингам)')
    axes[1].grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar, val in zip(bars, user_profile):
        height = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.2f}',
                    ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    return fig_to_base64(fig)

def generate_collaborative_vs_content():
    """Generate comparison chart between collaborative and content-based filtering."""
    fig, ax = plt.subplots(figsize=(10, 7))
    
    aspects = ['Точность\n(известные\nпредпочт.)', 
               'Разнообразие\n(serendipity)',
               'Холодный старт\n(новые товары)',
               'Холодный старт\n(новые польз.)',
               'Масштаби-\nруемость',
               'Объясни-\nмость']
    
    collaborative_scores = [4.5, 4.8, 2.0, 2.0, 3.5, 2.5]
    content_scores = [3.5, 2.5, 4.5, 2.0, 4.5, 4.8]
    
    x = np.arange(len(aspects))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, collaborative_scores, width, 
                   label='Коллаборативная', color='steelblue', alpha=0.8)
    bars2 = ax.bar(x + width/2, content_scores, width,
                   label='Контентная', color='coral', alpha=0.8)
    
    ax.set_ylabel('Оценка (1-5)')
    ax.set_title('Сравнение методов рекомендаций\n(Коллаборативная vs Контентная фильтрация)')
    ax.set_xticks(x)
    ax.set_xticklabels(aspects, fontsize=9)
    ax.legend()
    ax.set_ylim(0, 5.5)
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}',
                   ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    plt.tight_layout()
    return fig_to_base64(fig)

# ============================================================================
# MAIN FUNCTION
# ============================================================================

def generate_all_illustrations():
    """Generate all illustrations and return as dictionary."""
    print("Generating recommender systems illustrations...")
    
    illustrations = {}
    
    print("  - User-item matrix...")
    illustrations['user_item_matrix'] = generate_user_item_matrix()
    
    print("  - Similarity matrices...")
    illustrations['similarity_matrices'] = generate_similarity_matrices()
    
    print("  - CF prediction example...")
    illustrations['cf_prediction'] = generate_cf_prediction_example()
    
    print("  - SVD decomposition...")
    illustrations['svd_decomposition'] = generate_svd_decomposition()
    
    print("  - SVD example...")
    illustrations['svd_example'] = generate_svd_example()
    
    print("  - SVD k selection...")
    illustrations['svd_k_selection'] = generate_svd_k_selection()
    
    print("  - TF-IDF example...")
    illustrations['tfidf_example'] = generate_tfidf_example()
    
    print("  - Content similarity...")
    illustrations['content_similarity'] = generate_content_similarity()
    
    print("  - User profile example...")
    illustrations['user_profile'] = generate_user_profile_example()
    
    print("  - Collaborative vs Content comparison...")
    illustrations['collab_vs_content'] = generate_collaborative_vs_content()
    
    print("✓ All illustrations generated successfully!")
    return illustrations

if __name__ == '__main__':
    illustrations = generate_all_illustrations()
    print(f"\nGenerated {len(illustrations)} illustrations")
    for name in illustrations.keys():
        print(f"  - {name}")
