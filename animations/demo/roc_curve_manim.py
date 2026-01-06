#!/usr/bin/env python3
"""
ROC Curve Animation using Manim
Создание анимированных иллюстраций ROC кривых с помощью Manim

Требования:
    pip install manim
    
Использование:
    # Низкое качество (быстро)
    manim -pql roc_curve_manim.py ROCCurveScene
    
    # Высокое качество
    manim -pqh roc_curve_manim.py ROCCurveScene
    
    # Только последний кадр (для статичного изображения)
    manim -pql -s roc_curve_manim.py ROCCurveScene
"""

from manim import *
import numpy as np

class ROCCurveScene(Scene):
    """Анимация построения ROC кривой"""
    
    def construct(self):
        # Заголовок
        title = Text("ROC Кривая", font_size=48, color=BLUE)
        title.to_edge(UP)
        self.play(Write(title))
        self.wait(0.5)
        
        # Создание осей
        axes = Axes(
            x_range=[0, 1, 0.1],
            y_range=[0, 1, 0.1],
            x_length=6,
            y_length=6,
            axis_config={"color": GREY},
            tips=False
        )
        
        # Подписи осей
        x_label = Text("False Positive Rate", font_size=24).next_to(axes.x_axis, DOWN)
        y_label = Text("True Positive Rate", font_size=24).next_to(axes.y_axis, LEFT).rotate(90*DEGREES)
        
        labels = VGroup(axes, x_label, y_label)
        labels.move_to(ORIGIN)
        
        self.play(Create(axes), Write(x_label), Write(y_label))
        self.wait(0.5)
        
        # Диагональ (случайная модель)
        random_line = axes.plot(
            lambda x: x,
            color=GRAY,
            stroke_width=3
        )
        random_label = Text("Random (AUC=0.5)", font_size=20, color=GRAY)
        random_label.next_to(axes.c2p(0.6, 0.6), UR, buff=0.2)
        
        self.play(
            Create(random_line),
            run_time=1.5
        )
        self.play(FadeIn(random_label))
        self.wait(0.5)
        
        # Хорошая ROC кривая
        def good_roc_curve(x):
            """Функция хорошей ROC кривой"""
            return 1 - np.exp(-8 * x)
        
        good_curve = axes.plot(
            good_roc_curve,
            color=BLUE,
            stroke_width=4
        )
        
        # Анимация отрисовки кривой
        self.play(
            Create(good_curve),
            run_time=2,
            rate_func=smooth
        )
        
        # Заливка AUC
        area = axes.get_area(
            good_curve,
            x_range=[0, 1],
            color=BLUE,
            opacity=0.3
        )
        
        auc_label = Text("AUC = 0.92", font_size=24, color=BLUE, weight=BOLD)
        auc_label.next_to(axes.c2p(0.4, 0.3), RIGHT)
        
        self.play(FadeIn(area))
        self.play(Write(auc_label))
        self.wait(0.5)
        
        # Точки на кривой (пороги)
        threshold_points = [0.1, 0.3, 0.5, 0.7]
        dots = VGroup()
        threshold_labels = VGroup()
        
        for i, t in enumerate(threshold_points):
            y = good_roc_curve(t)
            dot = Dot(axes.c2p(t, y), color=YELLOW, radius=0.08)
            label = Text(f"t={1-t:.1f}", font_size=16, color=YELLOW)
            label.next_to(dot, UP+RIGHT, buff=0.1)
            
            dots.add(dot)
            threshold_labels.add(label)
        
        self.play(
            *[GrowFromCenter(dot) for dot in dots],
            run_time=1
        )
        self.play(
            *[FadeIn(label) for label in threshold_labels],
            run_time=0.8
        )
        self.wait(1)
        
        # Финальная пауза
        self.wait(2)


class ThresholdSelectionScene(Scene):
    """Анимация выбора оптимального порога"""
    
    def construct(self):
        title = Text("Выбор Оптимального Порога", font_size=40, color=BLUE)
        title.to_edge(UP)
        self.play(Write(title))
        self.wait(0.5)
        
        # Создание двух графиков: ROC кривая и распределения
        
        # ROC кривая (слева)
        axes_roc = Axes(
            x_range=[0, 1, 0.2],
            y_range=[0, 1, 0.2],
            x_length=4,
            y_length=4,
            axis_config={"color": GREY, "include_tip": False}
        ).shift(LEFT * 3)
        
        roc_curve = axes_roc.plot(
            lambda x: 1 - np.exp(-8 * x),
            color=BLUE,
            stroke_width=3
        )
        
        # Распределения (справа)
        axes_dist = Axes(
            x_range=[0, 1, 0.2],
            y_range=[0, 2, 0.5],
            x_length=4,
            y_length=4,
            axis_config={"color": GREY, "include_tip": False}
        ).shift(RIGHT * 3)
        
        # Нормальные распределения для двух классов
        def gaussian(x, mu, sigma):
            return (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
        
        negative_dist = axes_dist.plot(
            lambda x: gaussian(x, 0.3, 0.15),
            color=RED,
            stroke_width=3
        )
        
        positive_dist = axes_dist.plot(
            lambda x: gaussian(x, 0.7, 0.15),
            color=GREEN,
            stroke_width=3
        )
        
        # Отрисовка
        self.play(
            Create(axes_roc),
            Create(axes_dist),
            run_time=1
        )
        
        self.play(
            Create(roc_curve),
            Create(negative_dist),
            Create(positive_dist),
            run_time=2
        )
        
        # Порог (вертикальная линия)
        threshold = ValueTracker(0.3)
        
        threshold_line_dist = always_redraw(
            lambda: axes_dist.get_vertical_line(
                axes_dist.c2p(threshold.get_value(), 1.5),
                color=YELLOW,
                stroke_width=3
            )
        )
        
        # Точка на ROC кривой
        threshold_dot = always_redraw(
            lambda: Dot(
                axes_roc.c2p(
                    threshold.get_value(),
                    1 - np.exp(-8 * threshold.get_value())
                ),
                color=YELLOW,
                radius=0.1
            )
        )
        
        self.play(
            Create(threshold_line_dist),
            GrowFromCenter(threshold_dot)
        )
        
        # Анимация движения порога
        self.play(
            threshold.animate.set_value(0.7),
            run_time=3,
            rate_func=there_and_back
        )
        
        # Остановка на оптимальном значении
        self.play(
            threshold.animate.set_value(0.5),
            run_time=1
        )
        
        optimal_label = Text("Оптимальный\nпорог", font_size=20, color=YELLOW)
        optimal_label.next_to(threshold_line_dist, UP)
        self.play(Write(optimal_label))
        
        self.wait(2)


class ConfusionMatrixScene(Scene):
    """Анимация матрицы ошибок"""
    
    def construct(self):
        title = Text("Матрица Ошибок", font_size=48, color=BLUE)
        title.to_edge(UP)
        self.play(Write(title))
        self.wait(0.5)
        
        # Создание 2x2 матрицы
        cell_size = 2
        colors = {
            'TN': GREEN_E,
            'FP': RED_E,
            'FN': ORANGE,
            'TP': BLUE_E
        }
        
        cells_data = [
            {'name': 'TN', 'pos': LEFT * 1.5 + UP * 1.5, 'full': 'True\nNegative'},
            {'name': 'FP', 'pos': RIGHT * 1.5 + UP * 1.5, 'full': 'False\nPositive'},
            {'name': 'FN', 'pos': LEFT * 1.5 + DOWN * 1.5, 'full': 'False\nNegative'},
            {'name': 'TP', 'pos': RIGHT * 1.5 + DOWN * 1.5, 'full': 'True\nPositive'}
        ]
        
        cells = VGroup()
        for cell_data in cells_data:
            # Квадрат
            square = Square(
                side_length=cell_size,
                color=colors[cell_data['name']],
                fill_opacity=0.3,
                stroke_width=4
            )
            square.move_to(cell_data['pos'])
            
            # Текст
            label = Text(cell_data['name'], font_size=36, weight=BOLD)
            label.move_to(cell_data['pos'])
            
            full_label = Text(cell_data['full'], font_size=20)
            full_label.next_to(label, DOWN, buff=0.2)
            
            cell_group = VGroup(square, label, full_label)
            cells.add(cell_group)
        
        # Анимация появления ячеек
        for cell in cells:
            self.play(
                FadeIn(cell[0]),  # квадрат
                Write(cell[1]),    # название
                Write(cell[2]),    # полное название
                run_time=0.8
            )
        
        self.wait(0.5)
        
        # Подписи осей
        predicted_label = Text("Predicted", font_size=24).to_edge(UP).shift(DOWN * 2)
        actual_label = Text("Actual", font_size=24).to_edge(LEFT).shift(RIGHT * 0.5).rotate(90*DEGREES)
        
        self.play(
            Write(predicted_label),
            Write(actual_label)
        )
        
        self.wait(2)


# Для запуска всех сцен последовательно
class AllScenes(Scene):
    def construct(self):
        # Можно скомбинировать все сцены
        pass
