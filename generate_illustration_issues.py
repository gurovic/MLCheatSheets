#!/usr/bin/env python3
"""
Скрипт для генерации GitHub issues для добавления matplotlib-иллюстраций
в каждый раздел cheatsheets.

Использование:
    python generate_illustration_issues.py
"""

import re
from pathlib import Path
from typing import List, Dict, Tuple


def parse_illustrations_file(filepath: str) -> List[Dict[str, any]]:
    """
    Парсит файл pages_for_illustrations.md и извлекает разделы с файлами.
    
    Returns:
        List of dicts with 'section' name and 'pages' list
    
    Raises:
        FileNotFoundError: If the file doesn't exist
        IOError: If there's an error reading the file
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
    except FileNotFoundError:
        raise FileNotFoundError(f"Файл '{filepath}' не найден. Убедитесь, что вы запускаете скрипт из корневой директории проекта.")
    except IOError as e:
        raise IOError(f"Ошибка при чтении файла '{filepath}': {e}")
    
    sections = []
    current_section = None
    
    for line in content.split('\n'):
        # Проверяем заголовок раздела (## название)
        if line.startswith('## '):
            if current_section:
                sections.append(current_section)
            current_section = {
                'section': line[3:].strip(),
                'pages': []
            }
        # Проверяем элемент списка с файлом
        elif line.strip().startswith('- ') and current_section:
            page = line.strip()[2:].strip()
            if page.endswith('.html'):
                current_section['pages'].append(page)
    
    # Добавляем последний раздел
    if current_section:
        sections.append(current_section)
    
    return sections


def generate_issue_content(section_name: str, pages: List[str]) -> Tuple[str, str]:
    """
    Генерирует содержимое issue для раздела.
    
    Returns:
        Tuple of (title, body) for the issue
    """
    # Создаем красивый заголовок
    title = f"Добавить matplotlib-иллюстрации: {section_name}"
    
    # Создаем тело issue
    body = f"""## 📊 Добавление иллюстраций для раздела "{section_name}"

### Описание
Необходимо добавить matplotlib-иллюстрации (графики, схемы, диаграммы) в cheatsheets раздела "{section_name}" для улучшения визуального представления материала и облегчения понимания концепций.

### Страницы раздела ({len(pages)})
"""
    
    # Добавляем список страниц с чекбоксами
    for page in pages:
        # Извлекаем имя без расширения для более читабельного названия
        page_name = page.replace('_cheatsheet.html', '').replace('_', ' ').title()
        body += f"- [ ] `{page}` - {page_name}\n"
    
    body += """
### Рекомендации по созданию иллюстраций

1. **Стиль**: Используйте matplotlib с единообразным стилем (например, seaborn)
2. **Качество**: Иллюстрации должны быть четкими и информативными
3. **Формат**: Сохраняйте в формате PNG или SVG с высоким разрешением
4. **Размещение**: Располагайте иллюстрации в релевантных местах cheatsheet
5. **Код**: Включайте примеры кода matplotlib для воспроизводимости

### Примеры иллюстраций

- Графики функций (для функций активации, оптимизации)
- Схемы архитектур (для нейронных сетей)
- Диаграммы процессов (для алгоритмов)
- Визуализации данных (для методов обработки)
- Сравнительные графики (для метрик и результатов)

### Критерии выполнения

- [ ] Все страницы раздела содержат релевантные иллюстрации
- [ ] Иллюстрации имеют единообразный стиль
- [ ] Код для генерации иллюстраций задокументирован
- [ ] Качество иллюстраций соответствует стандартам проекта

### Метки
- `enhancement` - улучшение
- `documentation` - документация
- `visualization` - визуализация
- `matplotlib` - использование matplotlib

---
*Этот issue был создан автоматически на основе файла `pages_for_illustrations.md`*
"""
    
    return title, body


def generate_all_issues(output_dir: str = 'issues_to_create'):
    """
    Генерирует файлы с содержимым issues для всех разделов.
    
    Raises:
        FileNotFoundError: If pages_for_illustrations.md doesn't exist
        IOError: If there's an error reading or writing files
    """
    # Создаем директорию для issues
    output_path = Path(output_dir)
    try:
        output_path.mkdir(exist_ok=True)
    except OSError as e:
        raise IOError(f"Ошибка при создании директории '{output_dir}': {e}")
    
    # Парсим файл
    try:
        sections = parse_illustrations_file('pages_for_illustrations.md')
    except (FileNotFoundError, IOError) as e:
        print(f"❌ Ошибка: {e}")
        return
    
    print(f"📋 Найдено разделов: {len(sections)}")
    print(f"📁 Создание issues в директории: {output_dir}/\n")
    
    # Генерируем файлы для каждого раздела
    for i, section_data in enumerate(sections, 1):
        section_name = section_data['section']
        pages = section_data['pages']
        
        title, body = generate_issue_content(section_name, pages)
        
        # Создаем безопасное имя файла
        safe_filename = re.sub(r'[^\w\s-]', '', section_name.lower())
        safe_filename = re.sub(r'[-\s]+', '-', safe_filename)
        filename = output_path / f"{i:02d}-{safe_filename}.md"
        
        # Записываем содержимое
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(f"# {title}\n\n")
                f.write(f"**Заголовок issue:** `{title}`\n\n")
                f.write("---\n\n")
                f.write(body)
            print(f"✅ {i:2d}. {section_name} ({len(pages)} страниц) -> {filename.name}")
        except IOError as e:
            print(f"❌ Ошибка при записи файла {filename}: {e}")
    
    print(f"\n✨ Успешно создано {len(sections)} файлов с issues!")
    print(f"\n📝 Следующие шаги:")
    print(f"   1. Просмотрите созданные файлы в директории '{output_dir}/'")
    print(f"   2. Создайте issues в GitHub, используя содержимое этих файлов")
    print(f"   3. Для автоматического создания можно использовать GitHub CLI:")
    print(f"      gh issue create --title \"<заголовок>\" --body-file <файл.md>")


def generate_batch_script():
    """
    Генерирует скрипт для пакетного создания issues через GitHub CLI.
    
    Raises:
        FileNotFoundError: If pages_for_illustrations.md doesn't exist
        IOError: If there's an error reading or writing files
    """
    try:
        sections = parse_illustrations_file('pages_for_illustrations.md')
    except (FileNotFoundError, IOError) as e:
        print(f"❌ Ошибка при генерации bash скрипта: {e}")
        return
    
    script_content = """#!/bin/bash
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

"""
    
    for i, section_data in enumerate(sections, 1):
        section_name = section_data['section']
        safe_filename = re.sub(r'[^\w\s-]', '', section_name.lower())
        safe_filename = re.sub(r'[-\s]+', '-', safe_filename)
        filename = f"issues_to_create/{i:02d}-{safe_filename}.md"
        
        script_content += f"""# Issue {i}: {section_name}
echo "📝 Создаем issue {i}/{len(sections)}: {section_name}..."
gh issue create \\
    --title "Добавить matplotlib-иллюстрации: {section_name}" \\
    --label "enhancement,documentation,visualization,matplotlib" \\
    --body-file "{filename}" || echo "⚠️  Ошибка при создании issue {i}"
echo ""

"""
    
    script_content += f"""echo "✨ Завершено! Создано {len(sections)} issues."
echo "🔗 Просмотреть все issues: gh issue list --label matplotlib"
"""
    
    try:
        with open('create_issues.sh', 'w', encoding='utf-8') as f:
            f.write(script_content)
        print("✅ Создан скрипт create_issues.sh для автоматического создания issues")
        print("   Для использования: chmod +x create_issues.sh && ./create_issues.sh")
    except IOError as e:
        print(f"❌ Ошибка при создании скрипта create_issues.sh: {e}")


if __name__ == '__main__':
    print("=" * 70)
    print("  Генератор GitHub Issues для matplotlib-иллюстраций")
    print("=" * 70)
    print()
    
    # Генерируем markdown файлы с содержимым issues
    generate_all_issues()
    
    print()
    print("=" * 70)
    
    # Генерируем bash скрипт для автоматического создания
    generate_batch_script()
    
    print()
    print("=" * 70)
    print()
