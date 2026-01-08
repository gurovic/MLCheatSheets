#!/usr/bin/env python3
"""
Скрипт для удаления дублирующихся последовательных изображений в HTML файлах.
"""

import re
from pathlib import Path


def fix_duplicate_images(html_file):
    """
    Удаляет дублирующиеся последовательные строки с тегами <img> из HTML файла.
    
    Args:
        html_file: путь к HTML файлу
        
    Returns:
        tuple: (количество удаленных дубликатов, True если файл был изменен)
    """
    with open(html_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    new_lines = []
    duplicates_removed = 0
    i = 0
    
    while i < len(lines):
        current_line = lines[i]
        
        # Проверяем, содержит ли текущая строка тег <img>
        if '<img' in current_line:
            # Проверяем следующую строку
            if i + 1 < len(lines):
                next_line = lines[i + 1]
                
                # Если следующая строка идентична текущей, пропускаем ее
                if current_line.strip() == next_line.strip() and '<img' in next_line:
                    new_lines.append(current_line)
                    duplicates_removed += 1
                    i += 2  # Пропускаем дубликат
                    continue
        
        new_lines.append(current_line)
        i += 1
    
    # Если были найдены дубликаты, записываем исправленный файл
    if duplicates_removed > 0:
        with open(html_file, 'w', encoding='utf-8') as f:
            f.writelines(new_lines)
        return duplicates_removed, True
    
    return 0, False


def main():
    """Основная функция для обработки всех HTML файлов в директории cheatsheets."""
    cheatsheets_dir = Path('cheatsheets')
    
    if not cheatsheets_dir.exists():
        print(f"❌ Директория {cheatsheets_dir} не найдена!")
        return
    
    total_files = 0
    total_duplicates = 0
    modified_files = []
    
    print("=" * 70)
    print("Поиск и удаление дублирующихся изображений в cheatsheets")
    print("=" * 70)
    print()
    
    # Обрабатываем все HTML файлы
    for html_file in sorted(cheatsheets_dir.glob('*.html')):
        duplicates, modified = fix_duplicate_images(html_file)
        
        if modified:
            total_files += 1
            total_duplicates += duplicates
            modified_files.append((html_file.name, duplicates))
            print(f"✓ {html_file.name}: удалено {duplicates} дубликат(ов)")
    
    print()
    print("=" * 70)
    print(f"Итого обработано: {total_files} файл(ов)")
    print(f"Всего удалено дубликатов: {total_duplicates}")
    print("=" * 70)
    
    if modified_files:
        print("\nИзмененные файлы:")
        for filename, count in modified_files:
            print(f"  • {filename} ({count} дубликат(ов))")


if __name__ == '__main__':
    main()
