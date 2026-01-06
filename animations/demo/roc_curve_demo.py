#!/usr/bin/env python3
"""
ROC Curve Illustration Demo using Matplotlib
Создание статичных иллюстраций ROC кривых для cheatsheet

Использование:
    python roc_curve_demo.py
    
Выход:
    output/roc_curve_comparison.png
    output/roc_curve_comparison.svg
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import os

# Настройка стиля
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.unicode_minus'] = False

def generate_roc_curves():
    """Генерация точек для разных ROC кривых"""
    fpr = np.linspace(0, 1, 100)
    
    # Идеальная модель (шаг)
    tpr_perfect = np.ones_like(fpr)
    tpr_perfect[fpr < 0.01] = fpr[fpr < 0.01] * 100
    
    # Хорошая модель (плавная кривая близко к идеальной)
    tpr_good = 1 - np.exp(-8 * fpr)
    
    # Средняя модель
    tpr_medium = np.sqrt(fpr)
    tpr_medium = tpr_medium * 0.9 + fpr * 0.1
    
    # Слабая модель (близко к диагонали)
    tpr_weak = fpr + 0.1 * np.sin(10 * np.pi * fpr)
    tpr_weak = np.clip(tpr_weak, 0, 1)
    
    # Случайная модель (диагональ)
    tpr_random = fpr
    
    return fpr, {
        'perfect': tpr_perfect,
        'good': tpr_good,
        'medium': tpr_medium,
        'weak': tpr_weak,
        'random': tpr_random
    }

def calculate_auc(fpr, tpr):
    """Расчет AUC по методу трапеций"""
    from scipy import integrate
    return integrate.trapezoid(tpr, fpr)

def create_roc_comparison():
    """Создание сравнительной диаграммы ROC кривых"""
    fig, ax = plt.subplots(figsize=(10, 8), dpi=150)
    
    fpr, tpr_dict = generate_roc_curves()
    
    # Цвета для разных моделей
    colors = {
        'perfect': '#2e7d32',  # зеленый
        'good': '#1565c0',     # синий
        'medium': '#ef6c00',   # оранжевый
        'weak': '#c62828',     # красный
        'random': '#999999'    # серый
    }
    
    # Подписи и стили
    labels = {
        'perfect': 'Идеальная (AUC=1.0)',
        'good': f'Хорошая (AUC={calculate_auc(fpr, tpr_dict["good"]):.2f})',
        'medium': f'Средняя (AUC={calculate_auc(fpr, tpr_dict["medium"]):.2f})',
        'weak': f'Слабая (AUC={calculate_auc(fpr, tpr_dict["weak"]):.2f})',
        'random': 'Случайная (AUC=0.5)'
    }
    
    styles = {
        'perfect': {'linestyle': '-', 'linewidth': 2.5, 'alpha': 0.9},
        'good': {'linestyle': '-', 'linewidth': 2.5, 'alpha': 0.9},
        'medium': {'linestyle': '-', 'linewidth': 2, 'alpha': 0.8},
        'weak': {'linestyle': '--', 'linewidth': 2, 'alpha': 0.7},
        'random': {'linestyle': '--', 'linewidth': 2, 'alpha': 0.5}
    }
    
    # Заливка под хорошей кривой (демонстрация AUC)
    ax.fill_between(fpr, 0, tpr_dict['good'], alpha=0.1, color=colors['good'])
    
    # Отрисовка кривых
    for model_name in ['perfect', 'good', 'medium', 'weak', 'random']:
        ax.plot(fpr, tpr_dict[model_name], 
                color=colors[model_name],
                label=labels[model_name],
                **styles[model_name])
    
    # Точки на хорошей кривой для демонстрации порогов
    threshold_points = [10, 30, 50, 70]  # индексы
    for i in threshold_points:
        ax.plot(fpr[i], tpr_dict['good'][i], 'o', 
                color=colors['good'], markersize=6, 
                markeredgecolor='white', markeredgewidth=1.5)
    
    # Настройка осей
    ax.set_xlabel('False Positive Rate (FPR)', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Positive Rate (TPR)', fontsize=12, fontweight='bold')
    ax.set_title('ROC Кривые Разных Моделей', fontsize=14, fontweight='bold', pad=20)
    
    # Сетка
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax.set_axisbelow(True)
    
    # Пределы
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    
    # Равные пропорции осей
    ax.set_aspect('equal')
    
    # Легенда
    legend = ax.legend(loc='lower right', frameon=True, fontsize=10, 
                       fancybox=True, shadow=True)
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_alpha(0.95)
    
    # Текстовое пояснение AUC
    textstr = 'AUC = Площадь под\nROC кривой'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.35, 0.15, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    return fig

def create_confusion_matrix_illustration():
    """Создание иллюстрации матрицы ошибок"""
    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Размеры и позиции
    box_size = 3
    start_x, start_y = 2, 3
    
    # Цвета для ячеек
    colors_matrix = {
        'TN': '#e8f5e9',  # светло-зеленый
        'FP': '#ffebee',  # светло-красный
        'FN': '#fff3e0',  # светло-оранжевый
        'TP': '#e3f2fd'   # светло-синий
    }
    
    edge_colors = {
        'TN': '#2e7d32',
        'FP': '#c62828',
        'FN': '#ef6c00',
        'TP': '#1565c0'
    }
    
    # Матрица 2x2
    cells = [
        {'name': 'TN', 'x': start_x, 'y': start_y + box_size, 
         'full': 'True Negative', 'desc': 'Правильно'},
        {'name': 'FP', 'x': start_x + box_size, 'y': start_y + box_size,
         'full': 'False Positive', 'desc': 'Ошибка I рода'},
        {'name': 'FN', 'x': start_x, 'y': start_y,
         'full': 'False Negative', 'desc': 'Ошибка II рода'},
        {'name': 'TP', 'x': start_x + box_size, 'y': start_y,
         'full': 'True Positive', 'desc': 'Правильно'}
    ]
    
    # Отрисовка ячеек
    for cell in cells:
        fancy_box = FancyBboxPatch(
            (cell['x'], cell['y']), box_size, box_size,
            boxstyle="round,pad=0.1", 
            facecolor=colors_matrix[cell['name']],
            edgecolor=edge_colors[cell['name']],
            linewidth=3
        )
        ax.add_patch(fancy_box)
        
        # Текст в ячейке
        center_x = cell['x'] + box_size / 2
        center_y = cell['y'] + box_size / 2
        
        ax.text(center_x, center_y + 0.5, cell['name'], 
                ha='center', va='center', fontsize=16, fontweight='bold',
                color=edge_colors[cell['name']])
        ax.text(center_x, center_y, cell['full'], 
                ha='center', va='center', fontsize=9)
        ax.text(center_x, center_y - 0.5, cell['desc'], 
                ha='center', va='center', fontsize=8, style='italic')
    
    # Заголовки
    ax.text(5, 9, 'Матрица Ошибок (Confusion Matrix)', 
            ha='center', va='center', fontsize=14, fontweight='bold')
    
    # Подписи осей
    ax.text(start_x + box_size, start_y + 2 * box_size + 0.5, 'Предсказано', 
            ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax.text(start_x - 0.5, start_y + box_size, 'Факт', 
            ha='right', va='center', fontsize=11, fontweight='bold',
            rotation=90)
    
    # Метки столбцов
    ax.text(start_x + box_size/2, start_y + 2*box_size + 0.3, 'Negative (0)', 
            ha='center', va='bottom', fontsize=9)
    ax.text(start_x + 1.5*box_size, start_y + 2*box_size + 0.3, 'Positive (1)', 
            ha='center', va='bottom', fontsize=9)
    
    # Метки строк
    ax.text(start_x - 0.3, start_y + 1.5*box_size, 'Neg (0)', 
            ha='right', va='center', fontsize=9)
    ax.text(start_x - 0.3, start_y + 0.5*box_size, 'Pos (1)', 
            ha='right', va='center', fontsize=9)
    
    plt.tight_layout()
    return fig

def main():
    """Основная функция"""
    # Создание выходной директории
    output_dir = '../output'
    os.makedirs(output_dir, exist_ok=True)
    
    print("Создание иллюстраций ROC и AUC...")
    
    # 1. Сравнение ROC кривых
    print("  1. Генерация сравнения ROC кривых...")
    fig1 = create_roc_comparison()
    fig1.savefig(f'{output_dir}/roc_curve_comparison.png', 
                 bbox_inches='tight', dpi=150)
    fig1.savefig(f'{output_dir}/roc_curve_comparison.svg', 
                 bbox_inches='tight')
    print(f"     ✓ Сохранено: {output_dir}/roc_curve_comparison.png")
    print(f"     ✓ Сохранено: {output_dir}/roc_curve_comparison.svg")
    
    # 2. Матрица ошибок
    print("  2. Генерация матрицы ошибок...")
    fig2 = create_confusion_matrix_illustration()
    fig2.savefig(f'{output_dir}/confusion_matrix.png', 
                 bbox_inches='tight', dpi=150)
    fig2.savefig(f'{output_dir}/confusion_matrix.svg', 
                 bbox_inches='tight')
    print(f"     ✓ Сохранено: {output_dir}/confusion_matrix.png")
    print(f"     ✓ Сохранено: {output_dir}/confusion_matrix.svg")
    
    print("\n✅ Все иллюстрации успешно созданы!")
    print(f"\nФайлы находятся в: {output_dir}/")
    
    plt.close('all')

if __name__ == '__main__':
    main()
