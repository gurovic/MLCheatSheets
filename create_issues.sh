#!/bin/bash
# Скрипт для автоматического создания GitHub issues
# Требует установленного GitHub CLI (gh)
#
# Установка gh CLI:
# - macOS: brew install gh
# - Linux: https://github.com/cli/cli/blob/trunk/docs/install_linux.md
# - Windows: https://github.com/cli/cli/releases
#
# Использование:
#   chmod +x create_issues.sh
#   ./create_issues.sh

echo "🚀 Начинаем создание GitHub issues для иллюстраций..."
echo ""

# Проверяем наличие gh CLI
if ! command -v gh &> /dev/null; then
    echo "❌ GitHub CLI (gh) не установлен. Установите его сначала."
    exit 1
fi

# Проверяем авторизацию
if ! gh auth status &> /dev/null; then
    echo "❌ Не авторизованы в GitHub CLI. Выполните: gh auth login"
    exit 1
fi

echo "✅ GitHub CLI готов к работе"
echo ""

# Issue 1: Архитектуры нейронных сетей
echo "📝 Создаем issue 1/29: Архитектуры нейронных сетей..."
gh issue create \
    --title "Добавить matplotlib-иллюстрации: Архитектуры нейронных сетей" \
    --label "enhancement,documentation,visualization,matplotlib" \
    --body-file "issues_to_create/01-архитектуры-нейронных-сетей.md" || echo "⚠️  Ошибка при создании issue 1"
echo ""

# Issue 2: Рекуррентные сети
echo "📝 Создаем issue 2/29: Рекуррентные сети..."
gh issue create \
    --title "Добавить matplotlib-иллюстрации: Рекуррентные сети" \
    --label "enhancement,documentation,visualization,matplotlib" \
    --body-file "issues_to_create/02-рекуррентные-сети.md" || echo "⚠️  Ошибка при создании issue 2"
echo ""

# Issue 3: Графовые нейронные сети
echo "📝 Создаем issue 3/29: Графовые нейронные сети..."
gh issue create \
    --title "Добавить matplotlib-иллюстрации: Графовые нейронные сети" \
    --label "enhancement,documentation,visualization,matplotlib" \
    --body-file "issues_to_create/03-графовые-нейронные-сети.md" || echo "⚠️  Ошибка при создании issue 3"
echo ""

# Issue 4: Оптимизация и обучение
echo "📝 Создаем issue 4/29: Оптимизация и обучение..."
gh issue create \
    --title "Добавить matplotlib-иллюстрации: Оптимизация и обучение" \
    --label "enhancement,documentation,visualization,matplotlib" \
    --body-file "issues_to_create/04-оптимизация-и-обучение.md" || echo "⚠️  Ошибка при создании issue 4"
echo ""

# Issue 5: Активационные функции
echo "📝 Создаем issue 5/29: Активационные функции..."
gh issue create \
    --title "Добавить matplotlib-иллюстрации: Активационные функции" \
    --label "enhancement,documentation,visualization,matplotlib" \
    --body-file "issues_to_create/05-активационные-функции.md" || echo "⚠️  Ошибка при создании issue 5"
echo ""

# Issue 6: Сверточные слои и пулинг
echo "📝 Создаем issue 6/29: Сверточные слои и пулинг..."
gh issue create \
    --title "Добавить matplotlib-иллюстрации: Сверточные слои и пулинг" \
    --label "enhancement,documentation,visualization,matplotlib" \
    --body-file "issues_to_create/06-сверточные-слои-и-пулинг.md" || echo "⚠️  Ошибка при создании issue 6"
echo ""

# Issue 7: Кластеризация
echo "📝 Создаем issue 7/29: Кластеризация..."
gh issue create \
    --title "Добавить matplotlib-иллюстрации: Кластеризация" \
    --label "enhancement,documentation,visualization,matplotlib" \
    --body-file "issues_to_create/07-кластеризация.md" || echo "⚠️  Ошибка при создании issue 7"
echo ""

# Issue 8: Снижение размерности
echo "📝 Создаем issue 8/29: Снижение размерности..."
gh issue create \
    --title "Добавить matplotlib-иллюстрации: Снижение размерности" \
    --label "enhancement,documentation,visualization,matplotlib" \
    --body-file "issues_to_create/08-снижение-размерности.md" || echo "⚠️  Ошибка при создании issue 8"
echo ""

# Issue 9: Классификация
echo "📝 Создаем issue 9/29: Классификация..."
gh issue create \
    --title "Добавить matplotlib-иллюстрации: Классификация" \
    --label "enhancement,documentation,visualization,matplotlib" \
    --body-file "issues_to_create/09-классификация.md" || echo "⚠️  Ошибка при создании issue 9"
echo ""

# Issue 10: Регрессия
echo "📝 Создаем issue 10/29: Регрессия..."
gh issue create \
    --title "Добавить matplotlib-иллюстрации: Регрессия" \
    --label "enhancement,documentation,visualization,matplotlib" \
    --body-file "issues_to_create/10-регрессия.md" || echo "⚠️  Ошибка при создании issue 10"
echo ""

# Issue 11: Ансамбли
echo "📝 Создаем issue 11/29: Ансамбли..."
gh issue create \
    --title "Добавить matplotlib-иллюстрации: Ансамбли" \
    --label "enhancement,documentation,visualization,matplotlib" \
    --body-file "issues_to_create/11-ансамбли.md" || echo "⚠️  Ошибка при создании issue 11"
echo ""

# Issue 12: Метрики и оценка
echo "📝 Создаем issue 12/29: Метрики и оценка..."
gh issue create \
    --title "Добавить matplotlib-иллюстрации: Метрики и оценка" \
    --label "enhancement,documentation,visualization,matplotlib" \
    --body-file "issues_to_create/12-метрики-и-оценка.md" || echo "⚠️  Ошибка при создании issue 12"
echo ""

# Issue 13: Обработка данных
echo "📝 Создаем issue 13/29: Обработка данных..."
gh issue create \
    --title "Добавить matplotlib-иллюстрации: Обработка данных" \
    --label "enhancement,documentation,visualization,matplotlib" \
    --body-file "issues_to_create/13-обработка-данных.md" || echo "⚠️  Ошибка при создании issue 13"
echo ""

# Issue 14: Временные ряды
echo "📝 Создаем issue 14/29: Временные ряды..."
gh issue create \
    --title "Добавить matplotlib-иллюстрации: Временные ряды" \
    --label "enhancement,documentation,visualization,matplotlib" \
    --body-file "issues_to_create/14-временные-ряды.md" || echo "⚠️  Ошибка при создании issue 14"
echo ""

# Issue 15: Обучение с подкреплением
echo "📝 Создаем issue 15/29: Обучение с подкреплением..."
gh issue create \
    --title "Добавить matplotlib-иллюстрации: Обучение с подкреплением" \
    --label "enhancement,documentation,visualization,matplotlib" \
    --body-file "issues_to_create/15-обучение-с-подкреплением.md" || echo "⚠️  Ошибка при создании issue 15"
echo ""

# Issue 16: Computer Vision
echo "📝 Создаем issue 16/29: Computer Vision..."
gh issue create \
    --title "Добавить matplotlib-иллюстрации: Computer Vision" \
    --label "enhancement,documentation,visualization,matplotlib" \
    --body-file "issues_to_create/16-computer-vision.md" || echo "⚠️  Ошибка при создании issue 16"
echo ""

# Issue 17: NLP
echo "📝 Создаем issue 17/29: NLP..."
gh issue create \
    --title "Добавить matplotlib-иллюстрации: NLP" \
    --label "enhancement,documentation,visualization,matplotlib" \
    --body-file "issues_to_create/17-nlp.md" || echo "⚠️  Ошибка при создании issue 17"
echo ""

# Issue 18: Интерпретируемость
echo "📝 Создаем issue 18/29: Интерпретируемость..."
gh issue create \
    --title "Добавить matplotlib-иллюстрации: Интерпретируемость" \
    --label "enhancement,documentation,visualization,matplotlib" \
    --body-file "issues_to_create/18-интерпретируемость.md" || echo "⚠️  Ошибка при создании issue 18"
echo ""

# Issue 19: Байесовские методы
echo "📝 Создаем issue 19/29: Байесовские методы..."
gh issue create \
    --title "Добавить matplotlib-иллюстрации: Байесовские методы" \
    --label "enhancement,documentation,visualization,matplotlib" \
    --body-file "issues_to_create/19-байесовские-методы.md" || echo "⚠️  Ошибка при создании issue 19"
echo ""

# Issue 20: Трансферное обучение
echo "📝 Создаем issue 20/29: Трансферное обучение..."
gh issue create \
    --title "Добавить matplotlib-иллюстрации: Трансферное обучение" \
    --label "enhancement,documentation,visualization,matplotlib" \
    --body-file "issues_to_create/20-трансферное-обучение.md" || echo "⚠️  Ошибка при создании issue 20"
echo ""

# Issue 21: Meta-learning и Few-shot
echo "📝 Создаем issue 21/29: Meta-learning и Few-shot..."
gh issue create \
    --title "Добавить matplotlib-иллюстрации: Meta-learning и Few-shot" \
    --label "enhancement,documentation,visualization,matplotlib" \
    --body-file "issues_to_create/21-meta-learning-и-few-shot.md" || echo "⚠️  Ошибка при создании issue 21"
echo ""

# Issue 22: Графические модели
echo "📝 Создаем issue 22/29: Графические модели..."
gh issue create \
    --title "Добавить matplotlib-иллюстрации: Графические модели" \
    --label "enhancement,documentation,visualization,matplotlib" \
    --body-file "issues_to_create/22-графические-модели.md" || echo "⚠️  Ошибка при создании issue 22"
echo ""

# Issue 23: Обнаружение аномалий
echo "📝 Создаем issue 23/29: Обнаружение аномалий..."
gh issue create \
    --title "Добавить matplotlib-иллюстрации: Обнаружение аномалий" \
    --label "enhancement,documentation,visualization,matplotlib" \
    --body-file "issues_to_create/23-обнаружение-аномалий.md" || echo "⚠️  Ошибка при создании issue 23"
echo ""

# Issue 24: Рекомендательные системы
echo "📝 Создаем issue 24/29: Рекомендательные системы..."
gh issue create \
    --title "Добавить matplotlib-иллюстрации: Рекомендательные системы" \
    --label "enhancement,documentation,visualization,matplotlib" \
    --body-file "issues_to_create/24-рекомендательные-системы.md" || echo "⚠️  Ошибка при создании issue 24"
echo ""

# Issue 25: Валидация и тюнинг
echo "📝 Создаем issue 25/29: Валидация и тюнинг..."
gh issue create \
    --title "Добавить matplotlib-иллюстрации: Валидация и тюнинг" \
    --label "enhancement,documentation,visualization,matplotlib" \
    --body-file "issues_to_create/25-валидация-и-тюнинг.md" || echo "⚠️  Ошибка при создании issue 25"
echo ""

# Issue 26: Специальные архитектуры
echo "📝 Создаем issue 26/29: Специальные архитектуры..."
gh issue create \
    --title "Добавить matplotlib-иллюстрации: Специальные архитектуры" \
    --label "enhancement,documentation,visualization,matplotlib" \
    --body-file "issues_to_create/26-специальные-архитектуры.md" || echo "⚠️  Ошибка при создании issue 26"
echo ""

# Issue 27: Аудио обработка
echo "📝 Создаем issue 27/29: Аудио обработка..."
gh issue create \
    --title "Добавить matplotlib-иллюстрации: Аудио обработка" \
    --label "enhancement,documentation,visualization,matplotlib" \
    --body-file "issues_to_create/27-аудио-обработка.md" || echo "⚠️  Ошибка при создании issue 27"
echo ""

# Issue 28: Самообучение и полуобучение
echo "📝 Создаем issue 28/29: Самообучение и полуобучение..."
gh issue create \
    --title "Добавить matplotlib-иллюстрации: Самообучение и полуобучение" \
    --label "enhancement,documentation,visualization,matplotlib" \
    --body-file "issues_to_create/28-самообучение-и-полуобучение.md" || echo "⚠️  Ошибка при создании issue 28"
echo ""

# Issue 29: Дополнительно
echo "📝 Создаем issue 29/29: Дополнительно..."
gh issue create \
    --title "Добавить matplotlib-иллюстрации: Дополнительно" \
    --label "enhancement,documentation,visualization,matplotlib" \
    --body-file "issues_to_create/29-дополнительно.md" || echo "⚠️  Ошибка при создании issue 29"
echo ""

echo "✨ Завершено! Создано 29 issues."
echo "🔗 Просмотреть все issues: gh issue list --label matplotlib"
